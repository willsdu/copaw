# -*- coding: utf-8 -*-
"""Chat repository for storing chat/session specs（中文注释版）。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import ChatSpec, ChatsFile
from ...channels.schema import DEFAULT_CHANNEL


class BaseChatRepository(ABC):
    """Abstract repository for chat specs persistence."""
    # 说明：
    # 该类定义了“聊天规格（ChatSpec）如何持久化”的抽象接口。
    # runner 的其他层（ChatManager、API）只依赖这些方法，而不关心底层存储介质。

    @abstractmethod
    async def load(self) -> ChatsFile:
        """Load all chat specs from storage."""
        raise NotImplementedError

    @abstractmethod
    async def save(self, chats_file: ChatsFile) -> None:
        """Persist all chat specs to storage (should be atomic if possible)."""
        raise NotImplementedError

    # ---- Convenience operations ----

    async def list_chats(self) -> list[ChatSpec]:
        """List all chat specifications."""
        # 通过 load() 把全部 ChatSpec 拉取出来后返回；
        # 由于不同仓库实现可能有不同性能特征，这里提供一个统一的便利方法。
        cf = await self.load()
        return cf.chats

    async def get_chat(self, chat_id: str) -> Optional[ChatSpec]:
        """Get chat spec by chat_id (UUID).

        Args:
            chat_id: Chat UUID

        Returns:
            ChatSpec or None if not found
        """
        # 这里采用遍历查找：对于单文件 JSON 仓库足够简单；
        # 若仓库实现更高效（例如 DB/Redis），可以在子类里重写以提升性能。
        cf = await self.load()
        for chat in cf.chats:
            if chat.id == chat_id:
                return chat
        return None

    async def get_chat_by_id(
        self,
        session_id: str,
        user_id: str,
        channel: str = DEFAULT_CHANNEL,
    ) -> Optional[ChatSpec]:
        """Get chat spec by session_id and user_id.

        Args:
            session_id: Session identifier (e.g., "discord:alice")
            user_id: User identifier
            channel: Channel identifier

        Returns:
            ChatSpec or None if not found
        """
        # 说明：这里的命中条件是 (session_id, user_id, channel) 三元组；
        # 其中 session_id 通常由上层按 channel:user_id 格式组织。
        cf = await self.load()
        for chat in cf.chats:
            if (
                chat.session_id == session_id
                and chat.user_id == user_id
                and chat.channel == channel
            ):
                return chat
        return None

    async def upsert_chat(self, spec: ChatSpec) -> None:
        """Insert or update a chat spec.

        Args:
            spec: Chat specification to upsert
        """
        # upsert 语义：
        # - 若已有相同 id 的 ChatSpec，则替换；
        # - 否则追加到 chats 列表中。
        cf = await self.load()
        for i, c in enumerate(cf.chats):
            if c.id == spec.id:
                cf.chats[i] = spec
                break
        else:
            cf.chats.append(spec)
        await self.save(cf)

    async def delete_chats(self, chat_ids: list[str]) -> bool:
        """Delete a chat spec by chat_id (UUID).

        Args:
            chat_ids: List of chat IDs

        Returns:
            True if deleted, False if not found
        """
        # 删除的粒度是 chat_id（即 ChatSpec.id）。
        # 返回值用于让 API 层区分“找到了并删除”与“没找到”。 
        if not chat_ids:
            return False

        cf = await self.load()
        before = len(cf.chats)
        cf.chats = [c for c in cf.chats if c.id not in chat_ids]
        if len(cf.chats) == before:
            return False
        await self.save(cf)
        return True

    async def filter_chats(
        self,
        user_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> list[ChatSpec]:
        """Filter chats by user_id and/or channel.

        Args:
            user_id: Optional user ID filter
            channel: Optional channel filter

        Returns:
            Filtered list of chat specs
        """
        # 过滤逻辑是“逐条件缩小结果集”：
        # - 若 user_id 不为 None，则只保留匹配的；
        # - 若 channel 不为 None，则只保留匹配的。
        cf = await self.load()
        results = cf.chats

        if user_id is not None:
            results = [c for c in results if c.user_id == user_id]

        if channel is not None:
            results = [c for c in results if c.channel == channel]

        return results
