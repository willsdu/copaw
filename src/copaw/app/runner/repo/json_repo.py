# -*- coding: utf-8 -*-
"""JSON-based chat repository（聊天仓库实现：中文注释版）。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .base import BaseChatRepository
from ..models import ChatsFile


class JsonChatRepository(BaseChatRepository):
    """chats.json repository (single-file storage).

    中文说明：
    - 该实现把所有聊天规格（`ChatSpec` 列表）统一存放在一个 JSON 文件里；
    - JSON 文件结构中会包含 `version` 和 `chats`（每个 chat 对应一个 `ChatSpec`）；
    - 写入时采用“原子写入”：先写入同目录下的临时文件，再通过 `shutil.move` 替换，
      以降低写入过程中程序异常导致文件损坏的概率。

    Stores chat_id (UUID) -> session_id mappings in a JSON file.
    Similar to JsonJobRepository pattern from crons.

    Notes:
    - Single-machine, no cross-process lock.
    - Atomic write: write tmp then replace.
    """

    def __init__(self, path: Path | str):
        """Initialize JSON chat repository.

        中文说明：
        - `path` 可以是字符串或 `Path`；
        - 会做 `expanduser()`，从而支持 `~` 用户目录写法。

        Args:
            path: Path to chats.json file
        """
        if isinstance(path, str):
            path = Path(path)
        self._path = path.expanduser()

    @property
    def path(self) -> Path:
        """Get the repository file path."""
        return self._path

    async def load(self) -> ChatsFile:
        """Load chat specs from JSON file.

        中文说明：
        - 如果目标文件不存在：返回一个空的 `ChatsFile(version=1, chats=[])`；
        - 如果存在：读取文件并通过 Pydantic `model_validate` 完成校验/反序列化。

        Returns:
            ChatsFile with all chat specs
        """
        if not self._path.exists():
            return ChatsFile(version=1, chats=[])

        data = json.loads(self._path.read_text(encoding="utf-8"))
        return ChatsFile.model_validate(data)

    async def save(self, chats_file: ChatsFile) -> None:
        """Save chat specs to JSON file atomically.

        中文说明：
        - 先确保父目录存在；
        - 将 `ChatsFile` 转为可 JSON 序列化的 dict（`model_dump(mode="json")`）；
        - 写入临时文件（避免直接覆盖原文件）；
        - 最后用 `shutil.move` 做原子替换（在跨平台语义上更稳妥）。

        Args:
            chats_file: ChatsFile to persist
        """
        # Create parent directory if needed
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first (atomic write)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = chats_file.model_dump(mode="json")

        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Atomic replace (shutil.move handles cross-disk on Windows)
        shutil.move(str(tmp_path), str(self._path))
