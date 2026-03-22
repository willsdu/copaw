# -*- coding: utf-8 -*-
"""Chat repository implementations（中文注释版）。"""
from .base import BaseChatRepository
from .json_repo import JsonChatRepository

# __all__ 用于控制 `from ...repo import *` 的导出内容，
# 并让 IDE/阅读者知道“公共 API”有哪些。
__all__ = ["BaseChatRepository", "JsonChatRepository"]
