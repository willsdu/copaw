# -*- coding: utf-8 -*-
"""Chat manager for managing chat specifications（中文注释版）。"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from .models import ChatSpec
from .repo import BaseChatRepository
from ..channels.schema import DEFAULT_CHANNEL

logger = logging.getLogger(__name__)


class ChatManager:
    """Manages chat specifications in repository.

    中文说明：
    - 本类只负责“聊天规格（ChatSpec）的增删改查/统计”，并把持久化工作交给 `BaseChatRepository`；
    - Redis/文件系统里的 runner session state 由 `AgentRunner.session` 负责；
      因此这里不会删除/修改“聊天消息历史本身”，只改“聊天线程元信息（UUID 映射等）”。

    Only handles ChatSpec CRUD operations.
    Does NOT manage Redis session state - that's handled by runner's session.

    Similar to CronManager's role in crons module.
    """

    def __init__(
        self,
        *,
        repo: BaseChatRepository,
    ):
        """Initialize chat manager.

        中文说明：
        - `repo`：底层持久化实现（JSON/Redis/DB 等）；
        - `self._lock`：用于避免并发写导致的“读-改-写”竞态问题。

        Args:
            repo: Chat spec repository for persistence
        """
        self._repo = repo
        self._lock = asyncio.Lock()

    # ----- Read Operations -----

    async def list_chats(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> list[ChatSpec]:
        """List chat specs with optional filters.

        中文说明：
        - 支持按 `user_id`、`channel` 过滤；
        - 具体过滤逻辑在 `BaseChatRepository.filter_chats` 中完成。

        Args:
            user_id: Optional user ID filter
            channel: Optional channel filter

        Returns:
            List of chat specifications
        """
        async with self._lock:
            return await self._repo.filter_chats(
                user_id=user_id,
                channel=channel,
            )

    async def get_chat(self, chat_id: str) -> Optional[ChatSpec]:
        """Get chat spec by chat_id (UUID).

        Args:
            chat_id: Chat UUID

        Returns:
            Chat spec or None if not found
        """
        async with self._lock:
            return await self._repo.get_chat(chat_id)

    async def get_or_create_chat(
        self,
        session_id: str,
        user_id: str,
        channel: str = DEFAULT_CHANNEL,
        name: str = "New Chat",
    ) -> ChatSpec:
        """Get existing chat or create new one.

        中文说明：
        - 当上层收到“某个来源的消息”时，如果还没有对应的 ChatSpec，就自动创建一个；
        - 命中条件由仓库实现决定，这里默认通过 `get_chat_by_id(session_id,user_id,channel)` 查找。

        Useful for auto-registration when chats come from channels.

        Args:
            session_id: Session identifier (channel:user_id)
            user_id: User identifier
            channel: Channel name
            name: Chat name

        Returns:
            Chat specification (existing or newly created)
        """
        async with self._lock:
            # Try to find existing by session_id
            existing = await self._repo.get_chat_by_id(
                session_id,
                user_id,
                channel,
            )
            if existing:
                return existing

            # Create new
            spec = ChatSpec(
                session_id=session_id,
                user_id=user_id,
                channel=channel,
                name=name,
            )
            # Call internal create without lock (already locked)
            await self._repo.upsert_chat(spec)
            logger.debug(
                f"Auto-registered new chat: {spec.id} -> {session_id}",
            )
            return spec

    async def create_chat(self, spec: ChatSpec) -> ChatSpec:
        """Create a new chat.

        Args:
            spec: Chat specification (chat_id will be generated if not set)

        Returns:
            Chat spec
        """
        async with self._lock:
            await self._repo.upsert_chat(spec)
            return spec

    async def update_chat(self, spec: ChatSpec) -> ChatSpec:
        """Update an existing chat spec.

        中文说明：
        - 更新时会刷新 `updated_at` 时间戳；
        - 最终通过 `repo.upsert_chat` 做“覆盖式更新/插入”。

        Args:
            spec: Updated chat specification

        Returns:
            Updated chat spec
        """
        async with self._lock:
            spec.updated_at = datetime.now(timezone.utc)
            await self._repo.upsert_chat(spec)
            return spec

    async def delete_chats(self, chat_ids: list[str]) -> bool:
        """Delete a chat spec.

        Note: This only deletes the spec. Redis session state is NOT deleted.

        中文说明：
        - 删除的是聊天线程的“UUID 映射/元信息”，而不是用户实际的对话内容；
        - 返回值用于告诉 API 层是否真的存在对应记录。

        Args:
            chat_ids: List of chat IDs

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            deleted = await self._repo.delete_chats(chat_ids)

            if deleted:
                logger.debug(f"Deleted chats: {chat_ids}")

            return deleted

    async def count_chats(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> int:
        """Count chats matching filters.

        中文说明：
        - 本质是复用 `filter_chats` 后做长度统计；
        - 若未来需要性能优化，可在 repo 层增加更高效的统计接口。

        Args:
            user_id: Optional user ID filter
            channel: Optional channel filter

        Returns:
            Number of matching chats
        """
        async with self._lock:
            chats = await self._repo.filter_chats(
                user_id=user_id,
                channel=channel,
            )
            return len(chats)
