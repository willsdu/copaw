# -*- coding: utf-8 -*-
"""Chat models for runner with UUID management（中文注释版）。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field
from agentscope_runtime.engine.schemas.agent_schemas import Message

from ..channels.schema import DEFAULT_CHANNEL


class ChatSpec(BaseModel):
    """Chat specification with UUID identifier.

    中文说明：
    - `ChatSpec` 用于描述“一个会话（聊天线程）”的元信息；
    - 通过 `id`（UUID）作为公开/外部 API 的稳定标识；
    - `session_id` 用于与 runner 内部的 session（按 channel/user_id 匹配）建立对应关系。

    Stored in Redis and can be persisted in JSON file.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Chat UUID identifier",
    )
    # 展示用名称（例如页面标题/列表项显示）。
    name: str = Field(default="New Chat", description="Chat name")
    session_id: str = Field(
        ...,
        # 会话标识，约定为 `channel:user_id`（便于人眼/日志定位）。
        description="Session identifier (channel:user_id format)",
    )
    # 平台侧用户标识。
    user_id: str = Field(..., description="User identifier")
    # 平台侧通道/来源标识（如 discord/telegram/console 等）。
    channel: str = Field(default=DEFAULT_CHANNEL, description="Channel name")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chat creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Chat last update timestamp",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class ChatHistory(BaseModel):
    """Complete chat view with spec and state."""
    # 该模型返回“某个 chat_id 对应的历史消息列表”，用于 API 层的聚合响应。

    messages: list[Message] = Field(default_factory=list)


class ChatsFile(BaseModel):
    """Chat registry file for JSON repository.

    中文说明：
    - 这是 JSON 仓库落盘时的顶层结构；
    - 通过 `version` 标记格式版本；
    - `chats` 保存全部 `ChatSpec` 列表。

    Stores chat_id (UUID) -> session_id mappings for persistence.
    """

    version: int = 1
    chats: list[ChatSpec] = Field(default_factory=list)
