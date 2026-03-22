# -*- coding: utf-8 -*-
"""Chat management API（中文注释版）。"""
from __future__ import annotations
from typing import Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from agentscope.memory import InMemoryMemory

from .session import SafeJSONSession
from .manager import ChatManager
from .models import (
    ChatSpec,
    ChatHistory,
)
from .utils import agentscope_msg_to_message


router = APIRouter(prefix="/chats", tags=["chats"])

# 依赖注入函数：
# - `chat_manager`：由应用启动时初始化，并放入 `request.app.state`；
# - `runner.session`：用于从 JSON 文件取出 agent 的 memory 状态。

def get_chat_manager(request: Request) -> ChatManager:
    """Get the chat manager from app state.

    中文说明：
    该函数是 FastAPI 的 `Depends` 依赖注入入口。
    如果应用启动阶段尚未完成 `chat_manager` 注入，则返回 503，避免后续接口空指针错误。

    Args:
        request: FastAPI request object

    Returns:
        ChatManager instance

    Raises:
        HTTPException: If manager is not initialized
    """
    mgr = getattr(request.app.state, "chat_manager", None)
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Chat manager not initialized",
        )
    return mgr


def get_session(request: Request) -> SafeJSONSession:
    """Get the session from app state.

    中文说明：
    通过 `request.app.state.runner` 获取 runner 实例，
    再返回其中的 `SafeJSONSession`（负责会话 JSON 的读写）。

    Args:
        request: FastAPI request object

    Returns:
        SafeJSONSession instance

    Raises:
        HTTPException: If session is not initialized
    """
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="Session not initialized",
        )
    return runner.session


@router.get("", response_model=list[ChatSpec])
async def list_chats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
    mgr: ChatManager = Depends(get_chat_manager),
):
    """List all chats with optional filters.

    中文说明：
    - 不带参数时返回全部 ChatSpec；
    - 带 `user_id`/`channel` 时将过滤条件下推到 `ChatManager -> repo.filter_chats`。

    Args:
        user_id: Optional user ID to filter chats
        channel: Optional channel name to filter chats
        mgr: Chat manager dependency
    """
    return await mgr.list_chats(user_id=user_id, channel=channel)


@router.post("", response_model=ChatSpec)
async def create_chat(
    request: ChatSpec,
    mgr: ChatManager = Depends(get_chat_manager),
):
    """Create a new chat.

    中文说明：
    - `id`（UUID）由服务端生成，因此客户端不需要传入；
    - 其它字段直接来自请求体 `ChatSpec`。

    Server generates chat_id (UUID) automatically.

    Args:
        request: Chat creation request
        mgr: Chat manager dependency

    Returns:
        Created chat spec with UUID
    """
    chat_id = str(uuid4())
    spec = ChatSpec(
        id=chat_id,
        name=request.name,
        session_id=request.session_id,
        user_id=request.user_id,
        channel=request.channel,
        meta=request.meta,
    )
    return await mgr.create_chat(spec)


@router.post("/batch-delete", response_model=dict)
async def batch_delete_chats(
    chat_ids: list[str],
    mgr: ChatManager = Depends(get_chat_manager),
):
    """Delete chats by chat IDs.

    中文说明：
    - 只删除 ChatSpec（UUID 映射/元信息），不删除 runner 的 JSON session 中的 agent memory；
    - 返回 `{"deleted": bool}` 供调用方判断实际删除结果。

    Args:
        chat_ids: List of chat IDs
        mgr: Chat manager dependency
    Returns:
        True if deleted, False if failed

    """
    deleted = await mgr.delete_chats(chat_ids=chat_ids)
    return {"deleted": deleted}


@router.get("/{chat_id}", response_model=ChatHistory)
async def get_chat(
    chat_id: str,
    mgr: ChatManager = Depends(get_chat_manager),
    session: SafeJSONSession = Depends(get_session),
):
    """Get detailed information about a specific chat by UUID.

    中文说明：
    这个接口把“ChatSpec（元信息）”与“JSON session 里的 agent.memory（消息内容）”合并成返回的 `ChatHistory`。
    - 如果 session 不存在/为空：返回 `ChatHistory(messages=[])`；
    - 否则：把 memory 里的消息结构转换成可展示 messages。

    Args:
        chat_id: Chat UUID
        mgr: Chat manager dependency
        session: SafeJSONSession dependency

    Returns:
        ChatHistory with messages

    Raises:
        HTTPException: If chat not found (404)
    """
    chat_spec = await mgr.get_chat(chat_id)
    if not chat_spec:
        raise HTTPException(
            status_code=404,
            detail=f"Chat not found: {chat_id}",
        )

    state = await session.get_session_state_dict(
        chat_spec.session_id,
        chat_spec.user_id,
    )
    if not state:
        return ChatHistory(messages=[])
    memories = state.get("agent", {}).get("memory", [])
    memory = InMemoryMemory()
    memory.load_state_dict(memories)

    memories = await memory.get_memory()
    messages = agentscope_msg_to_message(memories)
    return ChatHistory(messages=messages)


@router.put("/{chat_id}", response_model=ChatSpec)
async def update_chat(
    chat_id: str,
    spec: ChatSpec,
    mgr: ChatManager = Depends(get_chat_manager),
):
    """Update an existing chat.

    中文说明：
    - 会校验 `spec.id` 是否与 path 参数 `chat_id` 一致；
    - 如果不一致：返回 400；
    - 如果对应 ChatSpec 不存在：返回 404。

    Args:
        chat_id: Chat UUID
        spec: Updated chat specification
        mgr: Chat manager dependency

    Returns:
        Updated chat spec

    Raises:
        HTTPException: If chat_id mismatch (400) or not found (404)
    """
    if spec.id != chat_id:
        raise HTTPException(
            status_code=400,
            detail="chat_id mismatch",
        )

    # Check if exists
    existing = await mgr.get_chat(chat_id)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Chat not found: {chat_id}",
        )

    updated = await mgr.update_chat(spec)
    return updated


@router.delete("/{chat_id}", response_model=dict)
async def delete_chat(
    chat_id: str,
    mgr: ChatManager = Depends(get_chat_manager),
):
    """Delete a chat by UUID.

    中文说明：
    删除的是 ChatSpec 记录（UUID 映射），不会删除 JSONSession 中的 agent.memory（聊天历史）。

    Note: This only deletes the chat spec (UUID mapping).
    JSONSession state is NOT deleted.

    Args:
        chat_id: Chat UUID
        mgr: Chat manager dependency

    Returns:
        True if deleted, False if failed

    Raises:
        HTTPException: If chat not found (404)
    """
    deleted = await mgr.delete_chats(chat_ids=[chat_id])
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Chat not found: {chat_id}",
        )
    return {"deleted": True}
