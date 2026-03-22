# -*- coding: utf-8 -*-
"""AgentRunner（中文注释版）。

该类是 CoPaw 的“核心运行器/路由器”，负责把上层请求转换为：
1) tool-guard 审批链路（approve/deny/timeout）；
2) 命令捷径链路（`/daemon ...`、`/compact` 等）；
3) 正常 agent 推理链路（创建 `CoPawAgent`、加载 session 状态、执行流式输出）。

同时，它还在异常时写出调试 dump，并在关闭时释放 memory manager 资源。
"""
# pylint: disable=unused-argument too-many-branches too-many-statements
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from agentscope.message import Msg, TextBlock
from agentscope.pipeline import stream_printing_messages
from agentscope_runtime.engine.runner import Runner
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from dotenv import load_dotenv

from .command_dispatch import (
    _get_last_user_text,
    _is_command,
    run_command_path,
)
from .query_error_dump import write_query_error_dump
from .session import SafeJSONSession
from .utils import build_env_context
from ..channels.schema import DEFAULT_CHANNEL
from ...agents.memory import MemoryManager
from ...agents.react_agent import CoPawAgent
from ...security.tool_guard.models import TOOL_GUARD_DENIED_MARK
from ...config import load_config
from ...constant import (
    TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS,
    WORKING_DIR,
)
from ...security.tool_guard.approval import ApprovalDecision

logger = logging.getLogger(__name__)


class AgentRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.framework_type = "agentscope"
        # ChatManager：用于把 chat 元信息（ChatSpec）与 session 关联、在会话结束时更新聊天列表。
        self._chat_manager = None
        # MCP client manager：用于热重载（restart/reload 时刷新可用的 MCP clients）。
        self._mcp_manager = None
        self.memory_manager: MemoryManager | None = None

    def set_chat_manager(self, chat_manager):
        """Set chat manager for auto-registration.

        中文说明：
        - 由外部注入 ChatManager 后，runner 才能在首次看到某个 session/channel 时自动创建 ChatSpec；
        - 同时在 query 结束时对 chat 元信息进行更新。

        Args:
            chat_manager: ChatManager instance
        """
        self._chat_manager = chat_manager

    def set_mcp_manager(self, mcp_manager):
        """Set MCP client manager for hot-reload support.

        中文说明：
        - 注入后，query_handler 会从该 manager 拉取 MCP clients（支持 restart 后“新客户端热生效”）。

        Args:
            mcp_manager: MCPClientManager instance
        """
        self._mcp_manager = mcp_manager

    _APPROVAL_TIMEOUT_SECONDS = TOOL_GUARD_APPROVAL_TIMEOUT_SECONDS

    async def _resolve_pending_approval(
        self,
        session_id: str,
        query: str | None,
    ) -> tuple[Msg | None, bool]:
        """Check for a pending tool-guard approval for *session_id*.

        中文说明：
        - 当某个 tool 调用需要用户审批时，会在审批服务中形成 pending 记录；
        - runner 会在收到新消息时先检查是否存在“当前 session 的 pending”；
        - 根据 pending 的状态与用户输入（是否发送 `/daemon approve` / `/approve`）决定：
          - 超时：直接 deny 并返回解释 Msg，且视为已消费审批；
          - 用户明确 approve：将请求标记为已批准，然后让消息继续流入 LLM，让其重新调用工具；
          - 默认：deny 并返回解释 Msg，且清理路径（在 query_handler 内）会把 denial 内容从 memory 中擦除/落库。

        Returns ``(response_msg, was_consumed)``:

        - ``(None, False)`` — no pending approval, continue normally.
        - ``(Msg, True)``   — denied; yield the Msg and stop.
        - ``(None, True)``  — approved; skip the command path and let
          the message reach the agent so the LLM can re-call the tool.
        """
        if not session_id:
            return None, False

        from ..approvals import get_approval_service

        svc = get_approval_service()
        pending = await svc.get_pending_by_session(session_id)
        if pending is None:
            return None, False

        elapsed = time.time() - pending.created_at
        if elapsed > self._APPROVAL_TIMEOUT_SECONDS:
            await svc.resolve_request(
                pending.request_id,
                ApprovalDecision.TIMEOUT,
            )
            return (
                Msg(
                    name="Friday",
                    role="assistant",
                    content=[
                        TextBlock(
                            type="text",
                            text=(
                                f"⏰ Tool `{pending.tool_name}` approval "
                                f"timed out ({int(elapsed)}s) — denied.\n"
                                f"工具 `{pending.tool_name}` 审批超时"
                                f"（{int(elapsed)}s），已拒绝执行。"
                            ),
                        ),
                    ],
                ),
                True,
            )

        normalized = (query or "").strip().lower()
        if normalized in ("/daemon approve", "/approve"):
            await svc.resolve_request(
                pending.request_id,
                ApprovalDecision.APPROVED,
            )
            return None, True

        await svc.resolve_request(
            pending.request_id,
            ApprovalDecision.DENIED,
        )
        return (
            Msg(
                name="Friday",
                role="assistant",
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"❌ Tool `{pending.tool_name}` denied.\n"
                            f"工具 `{pending.tool_name}` 已拒绝执行。"
                        ),
                    ),
                ],
            ),
            True,
        )

    async def query_handler(
        self,
        msgs,
        request: AgentRequest = None,
        **kwargs,
    ):
        """
        Handle agent query.

        中文说明（总体流程）：
        1. 从 runtime 的 `msgs` 中取出最后一条用户输入文本（作为 query）；
        2. 先处理 tool-guard pending 审批（approve/deny/timeout）：
           - 若审批被消费：直接 yield 对应的解释 Msg 并终止；
           - 若审批被批准：跳过命令捷径，让消息继续进入 LLM 流程；
        3. 若不是审批路径：并且 query 形如 `/daemon ...` 或 `/compact ...` 等命令，则走 `run_command_path`；
        4. 最终进入正常 agent 推理：创建 `CoPawAgent`、加载 session 状态、重建系统提示词，并流式输出消息；
        5. 在 `finally` 中保存 session state，并把 Chat 元信息（ChatSpec）更新回 ChatManager。
        """
        query = _get_last_user_text(msgs)
        # `msgs` 来自运行时消息队列，最后一个通常是用户输入；这里抽取为命令/查询字符串。
        session_id = getattr(request, "session_id", "") or ""

        (
            approval_response,
            approval_consumed,
        ) = await self._resolve_pending_approval(session_id, query)
        if approval_response is not None:
            # 审批服务返回了“要展示给用户的解释 Msg”（通常是 deny/timeout 解释）。
            yield approval_response, True
            # 如果是 deny 分支，这里会把该解释以及工具守卫标记从持久化 memory 中清理掉，
            # 避免对话历史中留下不该出现的“审批否决文本/标记”。
            user_id = getattr(request, "user_id", "") or ""
            await self._cleanup_denied_session_memory(
                session_id,
                user_id,
                denial_response=approval_response,
            )
            return

        if not approval_consumed and query and _is_command(query):
            # 命令捷径：在不进入完整 agent 推理链路的情况下，直接分派命令执行。
            logger.info("Command path: %s", query.strip()[:50])
            async for msg, last in run_command_path(request, msgs, self):
                yield msg, last
            return

        agent = None
        chat = None
        session_state_loaded = False
        try:
            session_id = request.session_id
            user_id = request.user_id
            # channel 用于区分不同来源（例如 discord/telegram/console），也是 session_id 组织的一部分约定。
            channel = getattr(request, "channel", DEFAULT_CHANNEL)

            logger.info(
                "Handle agent query:\n%s",
                json.dumps(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "channel": channel,
                        "msgs_len": len(msgs) if msgs else 0,
                        "msgs_str": str(msgs)[:300] + "...",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            env_context = build_env_context(
                session_id=session_id,
                user_id=user_id,
                channel=channel,
                working_dir=str(WORKING_DIR),
            )
            # env_context 会被注入到 CoPawAgent，作为系统/环境上下文前缀，帮助模型理解“当前运行环境”。

            # Get MCP clients from manager (hot-reloadable)
            mcp_clients = []
            if self._mcp_manager is not None:
                mcp_clients = await self._mcp_manager.get_clients()
                # 如果注入了 MCP manager，则 restart/reload 后可以在不重启进程的情况下获得新 clients。

            config = load_config()
            max_iters = config.agents.running.max_iters
            max_input_length = config.agents.running.max_input_length

            agent = CoPawAgent(
                env_context=env_context,
                mcp_clients=mcp_clients,
                memory_manager=self.memory_manager,
                request_context={
                    "session_id": session_id,
                    "user_id": user_id,
                    "channel": channel,
                },
                max_iters=max_iters,
                max_input_length=max_input_length,
            )
            # 这里开始创建真正的 agent 推理链路（会使用上面加载的 memory_manager）。
            await agent.register_mcp_clients()
            agent.set_console_output_enabled(enabled=False)

            logger.debug(
                f"Agent Query msgs {msgs}",
            )

            name = "New Chat"
            if len(msgs) > 0:
                content = msgs[0].get_text_content()
                if content:
                    name = msgs[0].get_text_content()[:10]
                else:
                    name = "Media Message"

            if self._chat_manager is not None:
                # ChatManager 负责“ChatSpec 元信息”的 auto-registration（建立 chat_id <-> session_id 映射）。
                chat = await self._chat_manager.get_or_create_chat(
                    session_id,
                    user_id,
                    channel,
                    name=name,
                )

            try:
                # 从持久化存储加载 session state（包含 agent 的历史 memory 等）。
                await self.session.load_session_state(
                    session_id=session_id,
                    user_id=user_id,
                    agent=agent,
                )
            except KeyError as e:
                logger.warning(
                    "load_session_state skipped (state schema mismatch): %s; "
                    "will save fresh state on completion to recover file",
                    e,
                )
            session_state_loaded = True

            # Rebuild system prompt so it always reflects the latest
            # AGENTS.md / SOUL.md / PROFILE.md, not the stale one saved
            # in the session state.
            agent.rebuild_sys_prompt()
            # 说明：即便 session state 里保存了系统提示词旧版本，这里也会重建一份“最新”版本供本轮使用。

            async for msg, last in stream_printing_messages(
                agents=[agent],
                coroutine_task=agent(msgs),
            ):
                yield msg, last

        except asyncio.CancelledError as exc:
            logger.info(f"query_handler: {session_id} cancelled!")
            if agent is not None:
                # 取消时主动中断 agent，避免后台继续跑工具/生成。
                await agent.interrupt()
            raise RuntimeError("Task has been cancelled!") from exc
        except Exception as e:
            # 任意异常都会写一个可追踪的 dump 文件（用于后续排障/复现）。
            debug_dump_path = write_query_error_dump(
                request=request,
                exc=e,
                locals_=locals(),
            )
            path_hint = (
                f"\n(Details:  {debug_dump_path})" if debug_dump_path else ""
            )
            logger.exception(f"Error in query handler: {e}{path_hint}")
            if debug_dump_path:
                setattr(e, "debug_dump_path", debug_dump_path)
                if hasattr(e, "add_note"):
                    e.add_note(
                        f"(Details:  {debug_dump_path})",
                    )
                suffix = f"\n(Details:  {debug_dump_path})"
                e.args = (
                    (f"{e.args[0]}{suffix}" if e.args else suffix.strip()),
                ) + e.args[1:]
            raise
        finally:
            if agent is not None and session_state_loaded:
                # 正常/异常都会尝试保存最新 session state，尽量保证历史 memory 不丢失。
                await self.session.save_session_state(
                    session_id=session_id,
                    user_id=user_id,
                    agent=agent,
                )

            if self._chat_manager is not None and chat is not None:
                # 如果本轮发生了对话（或被自动注册了 chat），则更新 ChatSpec 的 updated_at/元信息。
                await self._chat_manager.update_chat(chat)

    async def _cleanup_denied_session_memory(
        self,
        session_id: str,
        user_id: str,
        denial_response: "Msg | None" = None,
    ) -> None:
        """Clean up session memory after a tool-guard denial.

        中文说明：
        当工具守卫（tool-guard）决定“拒绝执行”某次敏感工具调用时，
        runner 会进入“deny 路径”（通常不会创建/跑 agent）。
        由于 LLM 可能已经在 memory 中写入了“拒绝解释文本”和对应的标记，
        为了让后续状态保持一致，本函数会：
        - 找到最后一次带 `TOOL_GUARD_DENIED_MARK` 的标记条目；
        - 删除其后紧跟的 assistant 拒绝解释消息（如果存在）；
        - 从剩余标记里移除 `TOOL_GUARD_DENIED_MARK`，把工具调用信息恢复为正常 memory 结构；
        - 如有 `denial_response`，把最终拒绝消息作为新条目追加到 memory 中并落盘。

        In the deny path (no agent is created), this method:

        1. Removes the LLM denial explanation (the assistant message
           immediately following the last marked entry).
        2. Strips ``TOOL_GUARD_DENIED_MARK`` from all marks lists so
           the kept tool-call info becomes normal memory entries.
        3. Appends *denial_response* (e.g. "❌ Tool denied") to the
           persisted session memory.
        """
        if not hasattr(self, "session") or self.session is None:
            return

        path = self.session._get_save_path(  # pylint: disable=protected-access
            session_id,
            user_id,
        )
        if not Path(path).exists():
            # session 文件不存在时，没有历史 memory 可以清理，直接返回。
            return

        try:
            # 读取已持久化的 states；其结构里通常包含 states["agent"]["memory"]["content"]。
            with open(
                path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as f:
                states = json.load(f)

            agent_state = states.get("agent", {})
            memory_state = agent_state.get("memory", {})
            content = memory_state.get("content", [])

            if not content:
                return

            def _is_marked(entry):
                # memory.content 里每条 entry 的结构通常为：
                #   [msg_dict, marks_list]
                # 其中 marks_list 可能包含 TOOL_GUARD_DENIED_MARK，用于标识该条是否来自“拒绝敏感工具”的等待状态。
                return (
                    isinstance(entry, list)
                    and len(entry) >= 2
                    and isinstance(entry[1], list)
                    and TOOL_GUARD_DENIED_MARK in entry[1]
                )

            last_marked_idx = -1
            # 找到最后一次带“拒绝标记”的条目位置；之后紧跟的 assistant 文本（如存在）会被删掉。
            for i, entry in enumerate(content):
                if _is_marked(entry):
                    last_marked_idx = i

            modified = False

            if last_marked_idx >= 0 and last_marked_idx + 1 < len(content):
                next_entry = content[last_marked_idx + 1]
                if (
                    isinstance(next_entry, list)
                    and len(next_entry) >= 1
                    and isinstance(next_entry[0], dict)
                    and next_entry[0].get("role") == "assistant"
                ):
                    # 删除被 LLM 写入的“拒绝解释消息”（通常紧跟在标记条目之后）。
                    del content[last_marked_idx + 1]
                    modified = True

            for entry in content:
                if _is_marked(entry):
                    # 把拒绝标记从 marks 列表中移除，让后续渲染/记忆结构回归正常。
                    entry[1].remove(TOOL_GUARD_DENIED_MARK)
                    modified = True

            if denial_response is not None:
                # 把最终需要展示给用户的 denial_response 追加为新的 memory entry。
                ts = getattr(denial_response, "timestamp", None)
                msg_dict = {
                    "id": getattr(denial_response, "id", ""),
                    "name": getattr(denial_response, "name", "Friday"),
                    "role": getattr(denial_response, "role", "assistant"),
                    "content": denial_response.content,
                    "metadata": getattr(
                        denial_response,
                        "metadata",
                        None,
                    ),
                    "timestamp": str(ts) if ts is not None else "",
                }
                content.append([msg_dict, []])
                modified = True

            if modified:
                with open(
                    path,
                    "w",
                    encoding="utf-8",
                    errors="surrogatepass",
                ) as f:
                    json.dump(states, f, ensure_ascii=False)
                logger.info(
                    "Tool guard: cleaned up denied session memory in %s",
                    path,
                )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "Failed to clean up denied messages from session %s",
                session_id,
                exc_info=True,
            )

    async def init_handler(self, *args, **kwargs):
        """
        Init handler.

        中文说明：
        - 从仓库根目录（相对 `__file__` 的 parents[4]）加载 `.env` 环境变量；
        - 初始化 `SafeJSONSession`，将会话状态落到 `WORKING_DIR/sessions` 目录；
        - 启动 `MemoryManager`（如果尚未初始化），让工具/记忆相关组件可用。
        """
        # Load environment variables from .env file
        env_path = Path(__file__).resolve().parents[4] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded environment variables from {env_path}")
        else:
            logger.debug(
                f".env file not found at {env_path}, "
                "using existing environment variables",
            )

        session_dir = str(WORKING_DIR / "sessions")
        self.session = SafeJSONSession(save_dir=session_dir)

        try:
            if self.memory_manager is None:
                self.memory_manager = MemoryManager(
                    working_dir=str(WORKING_DIR),
                )
            await self.memory_manager.start()
        except Exception as e:
            logger.exception(f"MemoryManager start failed: {e}")

    async def shutdown_handler(self, *args, **kwargs):
        """
        Shutdown handler.

        中文说明：
        - 关闭 `MemoryManager`，释放可能持有的资源（例如文件句柄/后台任务）。
        """
        try:
            if self.memory_manager is not None:
                await self.memory_manager.close()
        except Exception as e:
            logger.warning(f"MemoryManager stop failed: {e}")
