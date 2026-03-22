# -*- coding: utf-8 -*-
"""Command dispatch（中文注释版）：run command path without creating CoPawAgent.

Yields (Msg, last) compatible with query_handler stream.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from agentscope.message import Msg, TextBlock
from reme.memory.file_based.reme_in_memory_memory import ReMeInMemoryMemory

from .daemon_commands import (
    DaemonContext,
    DaemonCommandHandlerMixin,
    parse_daemon_query,
)
from ...agents.command_handler import CommandHandler
from ...agents.utils.token_counting import _get_token_counter
from ...config import load_config

logger = logging.getLogger(__name__)


def _get_last_user_text(msgs) -> str | None:
    """Extract last user message text from msgs (runtime message list)."""
    # 目标：从运行时 message list 中，取出最后一条用户消息的“可展示文本”。
    # 该函数同时兼容两种结构：
    # - Msg 对象：直接使用 get_text_content()
    # - dict 对象：尝试从 content/text 字段提取纯文本 block。
    if not msgs or len(msgs) == 0:
        return None
    last = msgs[-1]
    if hasattr(last, "get_text_content"):
        return last.get_text_content()
    if isinstance(last, dict):
        content = last.get("content") or last.get("text")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text")
    return None


def _is_conversation_command(query: str | None) -> bool:
    """True if query is a conversation command (/compact, /new, etc.)."""
    # conversation command：以 `/` 开头，且命令名必须存在于 CommandHandler.SYSTEM_COMMANDS
    if not query or not query.startswith("/"):
        return False
    cmd = query.strip().lstrip("/").split()[0] if query.strip() else ""
    return cmd in CommandHandler.SYSTEM_COMMANDS


def _is_command(query: str | None) -> bool:
    """True if query is any known command (daemon or conversation)."""
    # 先判断是否为 `/daemon ...`（parse_daemon_query 可解析则为 daemon 命令）
    # 否则再判断是否为 conversation command
    if not query or not query.startswith("/"):
        return False
    if parse_daemon_query(query) is not None:
        return True
    return _is_conversation_command(query)


async def run_command_path(
    request,
    msgs,
    runner,
) -> AsyncIterator[tuple]:
    """Run command path and yield (msg, last) for each response.

    中文说明：
    - 该函数是“命令走捷径”的入口：当用户输入以 `/` 开头且被判定为命令，
      则不进入 CoPawAgent 的完整推理/工具链路；
    - daemon 路径：执行管理命令并 yield 一个 assistant Msg；
    - conversation 路径：在轻量 memory（ReMeInMemoryMemory）上运行 CommandHandler，
      然后把更新后的 memory 回写到 session（用于保持命令对话语义的一致性）。

    Args:
        request: AgentRequest (session_id, user_id, etc.)
        msgs: List of messages from runtime (last is user input)
        runner: AgentRunner (session, memory_manager, etc.)

    Yields:
        (Msg, bool) compatible with query_handler stream
    """
    query = _get_last_user_text(msgs)
    if not query:
        return

    session_id = getattr(request, "session_id", "") or ""
    user_id = getattr(request, "user_id", "") or ""

    # Daemon path
    parsed = parse_daemon_query(query)
    if parsed is not None:
        # daemon 命令：这里不会创建 agent，只执行预设的 daemon handler。
        handler = DaemonCommandHandlerMixin()
        restart_cb = getattr(runner, "_restart_callback", None)
        if parsed[0] == "restart":
            logger.info(
                "run_command_path: daemon restart, callback=%s",
                "set" if restart_cb is not None else "None",
            )
            # Yield hint first so user sees it before restart runs.
            hint = Msg(
                name="Friday",
                role="assistant",
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "**Restart in progress**\n\n"
                            "- The service may be unresponsive for a while. "
                            "Please wait."
                        ),
                    ),
                ],
            )
            yield hint, True
        context = DaemonContext(
            load_config_fn=load_config,
            memory_manager=runner.memory_manager,
            restart_callback=restart_cb,
            session_id=session_id,
        )
        msg = await handler.handle_daemon_command(query, context)
        yield msg, True
        logger.info("handle_daemon_command %s completed", query)
        return

    # Conversation path: lightweight memory + CommandHandler
    memory = ReMeInMemoryMemory(token_counter=_get_token_counter())
    # 从 runner.session 加载历史 memory，然后在内存副本上处理命令
    session_state = await runner.session.get_session_state_dict(
        session_id=session_id,
        user_id=user_id,
    )
    memory_state = session_state.get("agent", {}).get("memory")
    memory.load_state_dict(memory_state)

    conv_handler = CommandHandler(
        agent_name="Friday",
        memory=memory,
        memory_manager=runner.memory_manager,
        enable_memory_manager=runner.memory_manager is not None,
    )
    try:
        response_msg = await conv_handler.handle_conversation_command(query)
    except RuntimeError as e:
        response_msg = Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text=str(e))],
        )
    yield response_msg, True

    # Update memory key with session_id & user_id to session,
    # but only if identifiers are present
    if session_id and user_id:
        await runner.session.update_session_state(
            session_id=session_id,
            key="agent.memory",
            value=memory.state_dict(),
            user_id=user_id,
        )
    else:
        logger.warning(
            "Skipping session_state update for conversation"
            " memory due to missing session_id or user_id (session_id=%r, "
            "user_id=%r)",
            session_id,
            user_id,
        )
