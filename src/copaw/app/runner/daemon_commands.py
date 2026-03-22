# -*- coding: utf-8 -*-
"""Daemon command execution layer and DaemonCommandHandlerMixin（中文注释版）。

Shared by in-chat /daemon <sub> and CLI `copaw daemon <sub>`.
Logs: tail WORKING_DIR / "copaw.log". Restart: in-process reload of channels,
cron and MCP (no process exit); works on Mac/Windows without a process manager.

中文说明：
- 用于在“聊天对话内”或“CLI”中执行管理类命令；
- 通过 in-process 重载（不退出进程）实现 restart 的轻量化；
- logs 子命令会读取 `WORKING_DIR / copaw.log` 的尾部内容返回给用户。
"""
# pylint: disable=too-many-return-statements
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from agentscope.message import Msg, TextBlock

from ...constant import WORKING_DIR
from ...config import load_config

RestartCallback = Callable[[], Awaitable[None]]
logger = logging.getLogger(__name__)


class RestartInProgressError(Exception):
    """Raised when /daemon restart is invoked while another restart runs."""


DAEMON_PREFIX = "/daemon"
DAEMON_SUBCOMMANDS = frozenset(
    {"status", "restart", "reload-config", "version", "logs", "approve"},
)
# Short names: /restart -> /daemon restart, etc.
DAEMON_SHORT_ALIASES = {
    "restart": "restart",
    "status": "status",
    "reload-config": "reload-config",
    "reload_config": "reload-config",
    "version": "version",
    "logs": "logs",
    "approve": "approve",
}


@dataclass
class DaemonContext:
    """Context for daemon commands (inject deps from runner or CLI)."""

    working_dir: Path = WORKING_DIR
    load_config_fn: Callable[[], Any] = load_config
    memory_manager: Optional[Any] = None
    # Optional: async restart (channels, cron, MCP) in-process.
    restart_callback: Optional[RestartCallback] = None
    # Session ID for approval commands.
    session_id: str = ""


def _get_last_lines(
    path: Path,
    lines: int = 100,
    max_bytes: int = 512 * 1024,
) -> str:
    """Read last N lines from a text file (tail) with bounded memory.

    中文说明：
    - 读取尾部日志时使用 `max_bytes` 限制，避免日志文件过大导致内存占用/延迟；
    - 若文件过大，只读取最后 `max_bytes` 字节，再取最后 `lines` 行返回。

    Reads at most max_bytes from the end of the file so large logs
    do not cause high memory usage or latency.
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return f"(Log file not found: {path})"
    try:
        size = path.stat().st_size
        if size == 0:
            return "(empty)"
        with open(path, "rb") as f:
            if size <= max_bytes:
                content = f.read().decode("utf-8", errors="replace")
            else:
                f.seek(size - max_bytes)
                content = f.read().decode("utf-8", errors="replace")
                first_nl = content.find("\n")
                if first_nl != -1:
                    content = content[first_nl + 1 :]
                else:
                    content = ""
        all_lines = content.splitlines()
        last = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return "\n".join(last) if last else "(empty)"
    except OSError as e:
        return f"(Error reading log: {e})"


def run_daemon_status(context: DaemonContext) -> str:
    """Return status text (health, config, memory_manager)."""
    # 该函数只做“信息拼装”：不执行重载、不触碰 agent/rerun，
    # 主要用于让用户快速确认配置是否加载、内存管理是否挂载成功。
    parts = ["**Daemon Status**", ""]
    try:
        cfg = context.load_config_fn()
        parts.append("- Config loaded: yes")
        if getattr(cfg, "agents", None) and getattr(
            cfg.agents,
            "running",
            None,
        ):
            max_in = getattr(cfg.agents.running, "max_input_length", "N/A")
            parts.append(f"- Max input length: {max_in}")
    except Exception as e:
        parts.append(f"- Config loaded: no ({e})")

    parts.append(f"- Working dir: {context.working_dir}")
    if context.memory_manager is not None:
        parts.append("- Memory manager: running")
    else:
        parts.append("- Memory manager: not attached")
    return "\n".join(parts)


async def run_daemon_restart(context: DaemonContext) -> str:
    """Trigger in-process restart (channels, cron, MCP) or instruct user."""
    # restart 的核心：如果注入了 restart_callback，则由回调在当前进程内完成重载；
    # 否则返回提示，让用户使用外部工具/进程管理器来重启服务。
    if context.restart_callback is not None:
        try:
            await context.restart_callback()
            return (
                "**Restart completed**\n\n"
                "- Channels, cron and MCP reloaded in-process (no exit)."
            )
        except RestartInProgressError:
            return (
                "**Restart skipped**\n\n"
                "- A restart is already in progress. Please wait for it to "
                "finish."
            )
        except Exception as e:
            return f"**Restart failed**\n\n- {e}"
    return (
        "**Restart**\n\n"
        "- No restart callback (e.g. not running inside app). "
        "Run the app (e.g. `copaw app`) and use /daemon restart in chat, "
        "or restart the process with systemd/supervisor/docker."
    )


def run_daemon_reload_config(context: DaemonContext) -> str:
    """Reload config (re-call load_config); no process restart."""
    # reload-config：只重新调用 `load_config()` 用于刷新配置对象，
    # 不会触发 channels/cron/MCP 的重载（与 restart 区别在这里）。
    try:
        context.load_config_fn()
        return (
            "**Config reloaded**\n\n- load_config() re-invoked successfully."
        )
    except Exception as e:
        return f"**Reload failed**\n\n- {e}"


def run_daemon_version(context: DaemonContext) -> str:
    """Return version and paths."""
    # version：当前版本号来源于 `...__version__`，若找不到则回退为 unknown。
    try:
        from ...__version__ import __version__ as ver
    except ImportError:
        ver = "unknown"
    return (
        f"**Daemon version**\n\n"
        f"- Version: {ver}\n"
        f"- Working dir: {context.working_dir}\n"
        f"- Log file: {context.working_dir / 'copaw.log'}"
    )


def run_daemon_logs(context: DaemonContext, lines: int = 100) -> str:
    """Tail last N lines from WORKING_DIR / copaw.log."""
    # 直接返回给用户的内容会被包裹成代码块，便于复制/阅读。
    log_path = context.working_dir / "copaw.log"
    content = _get_last_lines(log_path, lines=lines)
    return f"**Console log (last {lines} lines)**\n\n```\n{content}\n```"


async def run_daemon_approve(
    _context: DaemonContext,
    session_id: str = "",
) -> str:
    """Resolve the pending tool-guard approval for *session_id*.

    中文说明：
    - tool-guard 审批链路允许用户以 `/daemon approve` 结束“等待审批”的状态；
    - runner 在多数情况下会先拦截消息走主审批流程，但这里仍保留兜底实现：
      如果系统当前并没有 pending 请求，则返回“没有待审批”的提示。

    Called when the user sends ``/daemon approve`` in the chat while a
    tool-guard approval is pending.  The runner intercepts the message
    before it reaches this function in most cases, but this serves as
    a fallback and returns a helpful message when no approval is
    pending.
    """
    try:
        from ..approvals import get_approval_service
        from ...security.tool_guard.approval import ApprovalDecision

        svc = get_approval_service()
        pending = await svc.get_pending_by_session(session_id)
        if pending is None:
            return (
                "**No pending approval**\n\n"
                "- There is no tool-guard approval waiting for this "
                "session.\n"
                "- This command is only valid when a sensitive tool "
                "call is awaiting your review."
            )
        await svc.resolve_request(
            pending.request_id,
            ApprovalDecision.APPROVED,
        )
        return (
            f"**Tool execution approved** ✅\n\n"
            f"- Tool: `{pending.tool_name}`\n"
            f"- Request: `{pending.request_id[:8]}…`"
        )
    except Exception as exc:
        logger.warning("run_daemon_approve error: %s", exc, exc_info=True)
        return f"**Approve failed**\n\n- {exc}"


def parse_daemon_query(query: str) -> Optional[tuple[str, list[str]]]:
    """Parse /daemon <sub> or /<short>. Return (subcommand, args) or None."""
    # 支持两种写法：
    # - 全称：`/daemon status`、`/daemon restart`...
    # - 短命令：`/status`、`/restart`、`/logs 200`（可带参数）
    # 返回 (subcommand, args)，用于后续路由到具体执行函数。
    if not query or not isinstance(query, str):
        return None
    raw = query.strip()
    if not raw.startswith("/"):
        return None
    rest = raw.lstrip("/").strip()
    if not rest:
        return None
    parts = rest.split()
    first = parts[0].lower() if parts else ""

    if first == "daemon":
        if len(parts) < 2:
            return ("status", [])
        sub = parts[1].lower().replace("_", "-")
        if sub not in DAEMON_SUBCOMMANDS and "reload" in sub:
            sub = "reload-config"
        if sub not in DAEMON_SUBCOMMANDS:
            return None
        args = parts[2:] if len(parts) > 2 else []
        return (sub, args)
    if first in DAEMON_SHORT_ALIASES:
        sub = DAEMON_SHORT_ALIASES[first]
        return (sub, parts[1:] if len(parts) > 1 else [])
    return None


class DaemonCommandHandlerMixin:
    """Mixin for daemon commands: /daemon status, restart, logs, etc."""

    def is_daemon_command(self, query: str | None) -> bool:
        """True if query is /daemon <sub> or short name (/restart, etc.)."""
        return parse_daemon_query(query or "") is not None

    async def handle_daemon_command(
        self,
        query: str,
        context: DaemonContext,
    ) -> Msg:
        """Run daemon subcommand; return a single assistant Msg."""
        # 该 handler 总是返回一个单独的 `assistant Msg`，
        # 让上层对话流把输出作为“模型消息”渲染给用户。
        parsed = parse_daemon_query(query)
        if not parsed:
            return Msg(
                name="Friday",
                role="assistant",
                content=[
                    TextBlock(type="text", text="Unknown daemon command."),
                ],
            )
        sub, args = parsed
        if sub == "status":
            text = run_daemon_status(context)
        elif sub == "restart":
            text = await run_daemon_restart(context)
        elif sub == "reload-config":
            text = run_daemon_reload_config(context)
        elif sub == "version":
            text = run_daemon_version(context)
        elif sub == "logs":
            n = 100
            for a in args:
                if a.isdigit():
                    n = max(1, min(int(a), 2000))
                    break
            text = run_daemon_logs(context, lines=n)
        elif sub == "approve":
            session_id = getattr(context, "session_id", "") or ""
            text = await run_daemon_approve(context, session_id=session_id)
        else:
            text = "Unknown daemon subcommand."
        logger.info("handle_daemon_command %s completed", query)
        return Msg(
            name="Friday",
            role="assistant",
            content=[TextBlock(type="text", text=text)],
        )
