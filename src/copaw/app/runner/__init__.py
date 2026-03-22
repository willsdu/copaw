# -*- coding: utf-8 -*-
"""Runner module（中文注释版）：用于对外导出 runner 相关能力。

该文件主要做“聚合导出”，把 runner 的关键对象（`AgentRunner`、`ChatManager`、以及 API router/模型/仓库接口）
暴露给包外使用，减少 import 路径的心智负担。
"""
from .runner import AgentRunner
from .api import router
from .manager import ChatManager
from .models import (
    ChatSpec,
    ChatHistory,
    ChatsFile,
)
from .repo import (
    BaseChatRepository,
    JsonChatRepository,
)


__all__ = [
    # Core classes
    "AgentRunner",
    "ChatManager",
    # API
    "router",
    # Models
    "ChatSpec",
    "ChatHistory",
    "ChatsFile",
    # Chat Repository
    "BaseChatRepository",
    "JsonChatRepository",
]
