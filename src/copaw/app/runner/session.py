# -*- coding: utf-8 -*-
"""Safe JSON session with filename sanitization for cross-platform
compatibility.

中文说明：
- 运行时会把 `session_id` / `user_id` 作为“文件名的一部分”落盘；
- 但在不同操作系统（尤其 Windows）里，文件名存在禁用字符（例如 `\ / : * ? " < > |`），
  因此本模块会在落盘前对这些字符做替换，避免因为非法文件名导致存储失败；
- 另外，为了不阻塞事件循环，本实现对读取/写入都使用 `aiofiles` 做异步 IO。

Windows filenames cannot contain: \\ / : * ? " < > |
This module wraps agentscope's SessionBase so that session_id and user_id
are sanitized before being used as filenames.
"""
import os
import re
import json
import logging

from typing import Union, Sequence

import aiofiles
from agentscope.session import SessionBase

logger = logging.getLogger(__name__)


# Characters forbidden in Windows filenames
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(name: str) -> str:
    """Replace characters that are illegal in Windows filenames with ``--``.

    中文说明：
    - 只要入参里包含 Windows 禁用字符，就会用 `--` 进行替换；
    - 这样既能保证跨平台可用，也能最大限度保持原始字符串信息。

    >>> sanitize_filename('discord:dm:12345')
    'discord--dm--12345'
    >>> sanitize_filename('normal-name')
    'normal-name'
    """
    return _UNSAFE_FILENAME_RE.sub("--", name)


class SafeJSONSession(SessionBase):
    """SessionBase subclass with filename sanitization and async file I/O.

    中文说明：
    - 继承自 agentscope 的 `SessionBase`，但重写了保存/读取/更新会话状态的方法；
    - 通过 `_get_save_path` 统一生成“跨平台安全”的落盘路径；
    - JSON 读写使用 UTF-8，并保留 `surrogatepass`，以兼容潜在的特殊字符。

    Overrides all file-reading/writing methods to use :mod:`aiofiles` so
    that disk I/O does not block the event loop.
    """

    def __init__(
        self,
        save_dir: str = "./",
    ) -> None:
        """Initialize the JSON session class.

        中文说明：
        - `save_dir` 是会话 JSON 文件的目录（会话文件按 `session_id`/`user_id` 组合命名）；
        - 如果目录不存在，会在实际写入时创建。

        Args:
            save_dir (`str`, defaults to `"./"):
                The directory to save the session state.
        """
        self.save_dir = save_dir

    def _get_save_path(self, session_id: str, user_id: str) -> str:
        """Return a filesystem-safe save path.

        中文说明：
        - 会确保 `save_dir` 存在；
        - 会对 `session_id` / `user_id` 做文件名清理；
        - 若同时提供了 `user_id`，则生成形如 `"{safe_uid}_{safe_sid}.json"` 的文件名；
          否则仅使用 `session_id` 生成 `"{safe_sid}.json"`。

        Overrides the parent implementation to ensure the generated
        filename is valid on Windows, macOS and Linux.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        safe_sid = sanitize_filename(session_id)
        safe_uid = sanitize_filename(user_id) if user_id else ""
        if safe_uid:
            file_path = f"{safe_uid}_{safe_sid}.json"
        else:
            file_path = f"{safe_sid}.json"
        return os.path.join(self.save_dir, file_path)

    async def save_session_state(
        self,
        session_id: str,
        user_id: str = "",
        **state_modules_mapping,
    ) -> None:
        """Save state modules to a JSON file using async I/O."""
        # state_modules_mapping 的约定：键为“状态模块名”，值为带 state_dict() 的对象
        state_dicts = {
            name: state_module.state_dict()
            for name, state_module in state_modules_mapping.items()
        }
        session_save_path = self._get_save_path(session_id, user_id=user_id)
        with open(
            session_save_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(state_dicts, ensure_ascii=False))

        logger.info(
            "Saved session state to %s successfully.",
            session_save_path,
        )

    async def load_session_state(
        self,
        session_id: str,
        user_id: str = "",
        allow_not_exist: bool = True,
        **state_modules_mapping,
    ) -> None:
        """Load state modules from a JSON file using async I/O."""
        # 如果文件存在：读取并按模块名回填 state_dict
        # 如果文件不存在：可选择静默忽略（allow_not_exist=True）或直接报错
        session_save_path = self._get_save_path(session_id, user_id=user_id)
        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as f:
                content = await f.read()
                states = json.loads(content)

            for name, state_module in state_modules_mapping.items():
                if name in states:
                    state_module.load_state_dict(states[name])
            logger.info(
                "Load session state from %s successfully.",
                session_save_path,
            )

        elif allow_not_exist:
            logger.info(
                "Session file %s does not exist. Skip loading session state.",
                session_save_path,
            )

        else:
            raise ValueError(
                f"Failed to load session state for file {session_save_path} "
                "because it does not exist.",
            )

    async def update_session_state(
        self,
        session_id: str,
        key: Union[str, Sequence[str]],
        value,
        user_id: str = "",
        create_if_not_exist: bool = True,
    ) -> None:
        # `key` 支持两种形式：
        # - "a.b.c"：用点号表示嵌套路径；
        # - ["a", "b", "c"]：直接传递路径段列表。
        session_save_path = self._get_save_path(session_id, user_id=user_id)

        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as f:
                content = await f.read()
                states = json.loads(content)

        else:
            if not create_if_not_exist:
                raise ValueError(
                    f"Session file {session_save_path} does not exist.",
                )
            states = {}

        path = key.split(".") if isinstance(key, str) else list(key)
        if not path:
            raise ValueError("key path is empty")

        cur = states
        for k in path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]

        cur[path[-1]] = value

        async with aiofiles.open(
            session_save_path,
            "w",
            encoding="utf-8",
            errors="surrogatepass",
        ) as f:
            await f.write(json.dumps(states, ensure_ascii=False))

        logger.info(
            "Updated session state key '%s' in %s successfully.",
            key,
            session_save_path,
        )

    async def get_session_state_dict(
        self,
        session_id: str,
        user_id: str = "",
        allow_not_exist: bool = True,
    ) -> dict:
        """Return the session state dict from the JSON file.

        中文说明：
        - 该方法直接返回 JSON 文件反序列化后的“整个状态字典”；
        - 若文件不存在且 allow_not_exist=True，则返回空字典 `{}`；
        - 若 allow_not_exist=False，则抛出 ValueError，提醒上层调用者状态缺失。

        Args:
            session_id (`str`):
                The session id.
            user_id (`str`, default to `""`):
                The user ID for the storage.
            allow_not_exist (`bool`, defaults to `True`):
                Whether to allow the session to not exist. If `False`, raises
                an error if the session does not exist.

        Returns:
            `dict`:
                The session state dict loaded from the JSON file. Returns an
                empty dict if the file does not exist and
                `allow_not_exist=True`.
        """
        session_save_path = self._get_save_path(session_id, user_id=user_id)
        if os.path.exists(session_save_path):
            async with aiofiles.open(
                session_save_path,
                "r",
                encoding="utf-8",
                errors="surrogatepass",
            ) as file:
                content = await file.read()
                states = json.loads(content)

            logger.info(
                "Get session state dict from %s successfully.",
                session_save_path,
            )
            return states

        if allow_not_exist:
            logger.info(
                "Session file %s does not exist. Return empty state dict.",
                session_save_path,
            )
            return {}

        raise ValueError(
            f"Failed to get session state for file {session_save_path} "
            "because it does not exist.",
        )
