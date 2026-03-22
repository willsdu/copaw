# -*- coding: utf-8 -*-
"""后台模型下载任务的进程内内存存储（download task store）。

特点：
- 多个下载任务可以并发运行。
- 任务完成/失败/取消后，其结果会保留在内存中，直到显式清理。
  这样前端可以通过轮询接口拿到最终状态（completed/failed/cancelled）。

并发安全：
- 所有对 `_tasks` 的读写都使用 `_lock` 进行保护，避免协程并发导致的数据竞争。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DownloadTaskStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadTask(BaseModel):
    # 任务唯一标识：由服务端生成，用于前端轮询/取消。
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # 要下载的仓库/模型标识（由上层业务提供）。
    repo_id: str
    filename: Optional[str] = None
    # 下载目标文件名（可选，部分后端可能不需要该字段）。
    backend: str
    # 后端/执行器标识（例如不同存储后端或下载实现）。
    source: str
    # 数据源（例如模型来源地址、下载方式等）。
    status: DownloadTaskStatus = DownloadTaskStatus.PENDING
    # 当前任务状态。
    error: Optional[str] = None
    # 失败时的错误信息（成功则通常为 None）。
    result: Optional[Dict[str, Any]] = None
    # 成功时的结构化结果（字段形状由业务决定）。
    # 创建/更新时间戳：用于前端展示或服务端调试。
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


_tasks: Dict[str, DownloadTask] = {}
_lock = asyncio.Lock()


async def create_task(
    repo_id: str,
    filename: Optional[str],
    backend: str,
    source: str,
) -> DownloadTask:
    """创建一个新的“待处理”下载任务，并将其加入内存索引。"""
    async with _lock:
        # 在锁内完成写操作，确保生成的任务不会与并发读写冲突。
        task = DownloadTask(
            repo_id=repo_id,
            filename=filename,
            backend=backend,
            source=source,
        )
        _tasks[task.task_id] = task
        return task


async def get_tasks(backend: Optional[str] = None) -> List[DownloadTask]:
    """获取所有任务（可选按 `backend` 过滤）。"""
    async with _lock:
        # 先拷贝出列表，释放锁后再进行过滤，减少锁占用时间。
        tasks = list(_tasks.values())
    if backend:
        tasks = [t for t in tasks if t.backend == backend]
    return tasks


async def get_task(task_id: str) -> Optional[DownloadTask]:
    """按 `task_id` 获取单个任务；任务不存在则返回 None。"""
    async with _lock:
        # `_tasks` 的读取在锁内完成，避免刚好被删除/更新造成不一致。
        return _tasks.get(task_id)


async def update_status(
    task_id: str,
    status: DownloadTaskStatus,
    *,
    error: Optional[str] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """更新指定任务的状态。

    - `task_id` 不存在时：直接返回（no-op）。
    - 同时更新时间戳 `updated_at`。
    - `error`/`result` 仅在对应参数非 None 时才写入。
    """
    async with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return
        task.status = status
        # 记录状态变更时刻，便于前端显示或后续排查。
        task.updated_at = time.time()
        if error is not None:
            task.error = error
        if result is not None:
            task.result = result


async def cancel_task(task_id: str) -> bool:
    """取消指定任务（仅允许取消 pending/downloading 状态的任务）。

    返回值：
    - True：已取消
    - False：任务不存在或当前状态不可取消（例如已完成/失败/已取消）
    """
    async with _lock:
        task = _tasks.get(task_id)
        if task is None:
            return False
        # 只允许取消尚未结束的任务，避免覆盖“最终结果”。
        if task.status not in (
            DownloadTaskStatus.PENDING,
            DownloadTaskStatus.DOWNLOADING,
        ):
            return False
        task.status = DownloadTaskStatus.CANCELLED
        task.updated_at = time.time()
        return True


async def clear_completed(backend: Optional[str] = None) -> None:
    """清理处于“终态”的任务（completed/failed/cancelled）。

    如提供 `backend` 则只清理该后端下的终态任务。
    """
    async with _lock:
        # 先计算需要删除的 task_id，避免在遍历时直接修改字典造成问题。
        to_remove = [
            tid
            for tid, t in _tasks.items()
            if t.status
            in (
                DownloadTaskStatus.COMPLETED,
                DownloadTaskStatus.FAILED,
                DownloadTaskStatus.CANCELLED,
            )
            and (backend is None or t.backend == backend)
        ]
        for tid in to_remove:
            del _tasks[tid]
