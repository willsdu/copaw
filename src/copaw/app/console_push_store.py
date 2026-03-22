# -*- coding: utf-8 -*-
"""控制台消息的进程内内存缓存（console channel push store）。

用途：
- 给 Web 控制台推送“事件/文本消息”（例如 cron 任务输出）。

数据保留策略（用于限制内存增长）：
- 按消息数量上限：最多保留 `_MAX_MESSAGES` 条；当超过上限时丢弃最旧的消息。
- 按消息年龄上限：消息超过 `_MAX_AGE_SECONDS` 后，在“读取最近消息”时会被丢弃。

与前端的配合：
- 每条消息包含唯一 `id`，前端可以基于 `id` 做去重。
- 前端会对“已见消息集合”做数量上限控制，避免无限增长。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List

# 单一列表实现：所有消息都放在同一个 `_list` 里。
# 每条消息包含以下字段：
# - `id`：UUID，用于前端去重
# - `text`：要展示的文本内容
# - `ts`：写入该消息时的时间戳（用于年龄裁剪）
# - `session_id`：所属会话/连接标识（用于分 session 读取）
# - `sticky`：是否“置顶/粘滞”（由前端决定如何展示）
# 同时通过“数量上限”和“年龄上限”限制内存增长。
_list: List[Dict[str, Any]] = []
_lock = asyncio.Lock()
_MAX_AGE_SECONDS = 60
_MAX_MESSAGES = 500


async def append(session_id: str, text: str, *, sticky: bool = False) -> None:
    """追加一条消息到缓存中（超出数量上限时丢弃最旧消息）。"""
    if not session_id or not text:
        # 缺少关键字段时直接忽略，避免向前端传递空消息。
        return
    async with _lock:
        # 由于该 store 是进程内共享的，必须在锁内保证 _list 的一致性。
        _list.append(
            {
                # 前端去重使用的唯一标识。
                "id": str(uuid.uuid4()),
                "text": text,
                # 由调用方决定是否需要“粘滞展示”（例如在 UI 上更靠前）。
                "sticky": sticky,
                # 写入时间戳，用于 get_recent 的年龄裁剪。
                "ts": time.time(),
                # session_id 用于 take(session_id) 精准取出该会话的消息。
                "session_id": session_id,
            },
        )
        if len(_list) > _MAX_MESSAGES:
            # 数量上限保护：超过上限就删除最旧的一部分。
            _list.sort(key=lambda m: m["ts"])
            del _list[: len(_list) - _MAX_MESSAGES]


async def take(session_id: str) -> List[Dict[str, Any]]:
    """返回并移除指定 `session_id` 下的所有消息（一次性消费）。"""
    if not session_id:
        return []
    async with _lock:
        # 取出“属于当前会话”的消息，同时把它们从缓存中删掉。
        out = [m for m in _list if m.get("session_id") == session_id]
        _list[:] = [m for m in _list if m.get("session_id") != session_id]
        # 前端不需要 ts；剔除后减小 payload，同时避免前端错误依赖时间戳。
        return _strip_ts(out)


async def take_all() -> List[Dict[str, Any]]:
    """返回并移除缓存中的所有消息（全量消费）。"""
    async with _lock:
        # 由于是“消费式读取”，读出后必须清空，避免消息重复展示。
        out = list(_list)
        _list.clear()
        return _strip_ts(out)


def _strip_ts(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 控制返回字段：只保留前端实际需要的内容字段。
    # `ts` 用于服务端内部裁剪，但不参与前端展示/去重逻辑。
    return [
        {
            "id": m["id"],
            "text": m["text"],
            "sticky": bool(m.get("sticky", False)),
        }
        for m in msgs
    ]


async def get_recent(
    max_age_seconds: int = _MAX_AGE_SECONDS,
) -> List[Dict[str, Any]]:
    """
    返回“最近消息”（不消费）。

    行为分两步：
    1) 只返回 `ts >= now - max_age_seconds` 的消息；
    2) 同时把 store 里更老的消息从 `_list` 中裁剪掉，从而限制内存占用。
    """
    now = time.time()
    cutoff = now - max_age_seconds
    async with _lock:
        # 在锁内进行裁剪，确保返回与删除是原子一致的。
        out = [m for m in _list if m["ts"] >= cutoff]
        _list[:] = out
        return _strip_ts(out)
