# -*- coding: utf-8 -*-
"""Write query-handler error log and agent/memory state to a temp JSON file（中文注释版）。"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from datetime import datetime
from typing import Any

from ..channels.schema import DEFAULT_CHANNEL

logger = logging.getLogger(__name__)

#
# 该模块的目标是：当 `query_handler` 在处理用户请求时发生异常，
# 把“人类可读的 trace + 可机器读取的上下文信息（请求、agent_state）”落盘，
# 以便后续离线分析/复现。
#

def _safe_json_serialize(obj: object) -> object:
    """Convert object to JSON-serializable form; use str() for unknowns."""
    # JSON 原生只能处理少量类型（dict/list/str/number/bool/None）。
    # 对于其它类型（例如 datetime、pydantic 模型等），这里用 `str(obj)` 保底。
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_json_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json_serialize(v) for k, v in obj.items()}
    return str(obj)


def _request_to_dict(request: Any) -> Any:
    """Serialize request to a JSON-serializable dict (Pydantic or vars)."""
    # request 可能是 Pydantic 对象，也可能是简单对象（vars 可用），
    # 因此这里按优先级尝试 model_dump -> dict -> vars(request)。
    if request is None:
        return None
    try:
        raw: dict[str, Any]
        if hasattr(request, "model_dump"):
            raw = request.model_dump()
        elif hasattr(request, "dict"):
            raw = request.dict()
        else:
            raw = dict(vars(request))
        if not isinstance(raw, dict):
            raw = dict(vars(request))
        return _safe_json_serialize(raw)
    except Exception:
        return {"_serialize_error": str(request)}


def write_query_error_dump(
    request: Any,
    exc: BaseException,
    locals_: dict,
) -> str | None:
    """Write error log, traceback and agent/memory state to a temp JSON file.

    中文说明：
    - 返回值是写入成功后的临时文件路径；
    - 若在写入过程中再次失败（例如序列化错误、磁盘问题），则返回 None，
      上层会继续抛出原异常（只是不再附带 dump 路径）。

    Returns the temp file path, or None if write failed.
    """
    try:
        # 1) 先把 request 中的关键信息抽出来（session_id/user_id/channel）
        # 2) 再把 request 完整对象序列化成可 JSON 的 dict
        request_info: dict[str, Any] = {}
        request_full: dict[str, Any] | None = None
        if request is not None:
            request_info = {
                "session_id": getattr(request, "session_id", None),
                "user_id": getattr(request, "user_id", None),
                "channel": getattr(request, "channel", DEFAULT_CHANNEL),
            }
            request_full = _request_to_dict(request)
        trace_str = traceback.format_exc()
        agent_state = None
        # 这里从 locals_ 里尝试拿到上层的 `agent`，以便保存 agent.state_dict()
        agent = locals_.get("agent")
        if agent is not None:
            try:
                if hasattr(agent, "state_dict"):
                    agent_state = _safe_json_serialize(agent.state_dict())
            except Exception as state_err:
                agent_state = {"_serialize_error": str(state_err)}
        payload = {
            "trace": trace_str,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "request_info": request_info,
            "request": request_full,
            "agent_state": agent_state,
            "ts_utc": datetime.utcnow().isoformat() + "Z",
        }
        fd, path = tempfile.mkstemp(
            prefix="copaw_query_error_",
            suffix=".json",
            dir=tempfile.gettempdir(),
            text=True,
        )
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return path
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
    except Exception as dump_err:
        logger.warning("Failed to write query error dump: %s", dump_err)
        return None
