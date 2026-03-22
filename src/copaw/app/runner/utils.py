# -*- coding: utf-8 -*-
"""
runner 模块工具函数（中文注释版）。

本文件主要做两件事：
1) `build_env_context`：把“当前会话/用户/通道/工作目录”等运行时信息，拼接成喂给 LLM 的环境上下文字符串；
2) `agentscope_msg_to_message`：把外部框架（agentscope）产生的消息结构（包含 text/thinking/tool_use/tool_result/image/audio 等 block）
   转换为运行时引擎（agentscope_runtime）的 `Message` 结构，并保持必要的元信息映射。
"""
import json
from datetime import datetime, timezone
from typing import Optional, Union, List
from urllib.parse import urlparse
from agentscope.message import Msg
from agentscope_runtime.engine.schemas.agent_schemas import (
    Message,
    FunctionCall,
    FunctionCallOutput,
    MessageType,
)
from agentscope_runtime.engine.helpers.agent_api_builder import ResponseBuilder


def build_env_context(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    channel: Optional[str] = None,
    working_dir: Optional[str] = None,
    add_hint: bool = True,
) -> str:
    """
    Build environment context with current request context prepended.

    中文说明：
    - 该函数会生成一段固定格式的文本，把“当前 UTC 时间”和“请求上下文”作为前缀提供给模型；
    - 若 `add_hint=True`，会附带一段重要提示（例如：优先使用 skills、写文件前先 read_file 等），
      用于降低模型直接操作文件/工具时的误用风险。

    Args:
        session_id: Current session ID
        user_id: Current user ID
        channel: Current channel name
        working_dir: Working directory path
        add_hint: Whether to add hint context
    Returns:
        Formatted environment context string
    """
    parts = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC (%A)")
    parts.append(f"- 当前 UTC 时间: {now_utc}")
    if session_id is not None:
        parts.append(f"- 当前的session_id: {session_id}")
    if user_id is not None:
        parts.append(f"- 当前的user_id: {user_id}")
    if channel is not None:
        parts.append(f"- 当前的channel: {channel}")

    if working_dir is not None:
        parts.append(f"- 工作目录: {working_dir}")

    if add_hint:
        parts.append(
            "- 重要提示:\n"
            "  1. 完成任务时，优先考虑使用 skills"
            "（例如定时任务，优先使用 cron skill）。"
            "对于不清楚的 skills，请先查阅相关对应文档。\n"
            "  2. 使用 write_file 写文件时，如果担心覆盖原有内容，"
            "可以先用 read_file 查看文件内容，"
            "再使用 edit_file 工具进行局部内容更新或追加内容。",
        )

    return (
        "====================\n" + "\n".join(parts) + "\n===================="
    )


# pylint: disable=too-many-branches,too-many-statements
def agentscope_msg_to_message(
    messages: Union[Msg, List[Msg]],
) -> List[Message]:
    """
    Convert AgentScope Msg(s) into one or more runtime Message objects

    中文说明：
    agentscope 的 `Msg` 可能包含多种 block（纯文本、思考、工具调用、工具结果、图片、音频等）。
    为了让运行时引擎能够正确渲染/路由这些内容，本函数会：
    - 对每个 block 按“高层消息类型”进行分组/拆分；
    - 为需要时创建新的 `MessageBuilder`（例如 tool_use/tool_result 需要分别映射到 PLUGIN_CALL / PLUGIN_CALL_OUTPUT）；
    - 将 messages 的 `id/name/metadata` 等元信息保存到 runtime message metadata 中，便于后续追踪来源。

    Args:
        messages: AgentScope message(s) from streaming.

    Returns:
        List[Message]: One or more constructed runtime Message objects.
    """
    if isinstance(messages, Msg):
        msgs = [messages]
    elif isinstance(messages, list):
        msgs = messages
    else:
        raise TypeError(f"Expected Msg or list[Msg], got {type(messages)}")

    results: List[Message] = []

    for msg in msgs:
        role = msg.role or "assistant"

        if isinstance(msg.content, str):
            # 只有纯文本内容：直接生成 MESSAGE 类型
            rb = ResponseBuilder()
            mb = rb.create_message_builder(
                role=role,
                message_type=MessageType.MESSAGE,
            )
            # 元信息：保留 agentscope 原始的 id/name/metadata
            # （后续在渲染/追踪时可以把 runtime message 映射回上游消息来源）。
            mb.message.metadata = {
                "original_id": msg.id,
                "original_name": msg.name,
                "metadata": msg.metadata,
            }
            cb = mb.create_content_builder(content_type="text")
            cb.set_text(msg.content)
            cb.complete()
            mb.complete()
            results.append(mb.get_message_data())
            continue

        # msg.content is a list of blocks
        # 按“高层消息类型”分组 blocks：例如文本/推理/工具调用需要分别落到不同的 MessageType
        current_mb = None
        current_type = None

        for block in msg.content:
            if isinstance(block, dict):
                btype = block.get("type", "text")
            else:
                continue

            if btype == "text":
                # text：映射为 MessageType.MESSAGE（普通回答文本）
                if current_type != MessageType.MESSAGE:
                    if current_mb:
                        current_mb.complete()
                        results.append(current_mb.get_message_data())
                    rb = ResponseBuilder()
                    current_mb = rb.create_message_builder(
                        role=role,
                        message_type=MessageType.MESSAGE,
                    )
                    # add meta field to store old id and name
                    current_mb.message.metadata = {
                        "original_id": msg.id,
                        "original_name": msg.name,
                        "metadata": msg.metadata,
                    }
                    current_type = MessageType.MESSAGE
                cb = current_mb.create_content_builder(content_type="text")
                cb.set_text(block.get("text", ""))
                cb.complete()

            elif btype == "thinking":
                # thinking：映射为 MessageType.REASONING（模型推理/思考段落）
                if current_type != MessageType.REASONING:
                    if current_mb:
                        current_mb.complete()
                        results.append(current_mb.get_message_data())
                    rb = ResponseBuilder()
                    current_mb = rb.create_message_builder(
                        role=role,
                        message_type=MessageType.REASONING,
                    )
                    # add meta field to store old id and name
                    current_mb.message.metadata = {
                        "original_id": msg.id,
                        "original_name": msg.name,
                        "metadata": msg.metadata,
                    }
                    current_type = MessageType.REASONING
                cb = current_mb.create_content_builder(content_type="text")
                cb.set_text(block.get("thinking", ""))
                cb.complete()

            elif btype == "tool_use":
                # tool_use：映射到 PLUGIN_CALL；并且“总是”开启一个新的插件调用消息
                if current_mb:
                    current_mb.complete()
                    results.append(current_mb.get_message_data())
                rb = ResponseBuilder()
                current_mb = rb.create_message_builder(
                    role=role,
                    message_type=MessageType.PLUGIN_CALL,
                )
                # add meta field to store old id and name
                current_mb.message.metadata = {
                    "original_id": msg.id,
                    "original_name": msg.name,
                    "metadata": msg.metadata,
                }
                current_type = MessageType.PLUGIN_CALL
                cb = current_mb.create_content_builder(content_type="data")

                if isinstance(block.get("input"), (dict, list)):
                    arguments = json.dumps(
                        block.get("input"),
                        ensure_ascii=False,
                    )
                else:
                    arguments = block.get("input")

                call_data = FunctionCall(
                    call_id=block.get("id"),
                    name=block.get("name"),
                    arguments=arguments,
                ).model_dump()
                cb.set_data(call_data)
                cb.complete()

            elif btype == "tool_result":
                # tool_result：映射到 PLUGIN_CALL_OUTPUT；并且“总是”开启一个新的插件调用输出消息
                if current_mb:
                    current_mb.complete()
                    results.append(current_mb.get_message_data())
                rb = ResponseBuilder()
                current_mb = rb.create_message_builder(
                    role=role,
                    message_type=MessageType.PLUGIN_CALL_OUTPUT,
                )
                # add meta field to store old id and name
                current_mb.message.metadata = {
                    "original_id": msg.id,
                    "original_name": msg.name,
                    "metadata": msg.metadata,
                }
                current_type = MessageType.PLUGIN_CALL_OUTPUT
                cb = current_mb.create_content_builder(content_type="data")

                if isinstance(block.get("output"), (dict, list)):
                    output = json.dumps(
                        block.get("output"),
                        ensure_ascii=False,
                    )
                else:
                    output = block.get("output")

                output_data = FunctionCallOutput(
                    call_id=block.get("id"),
                    name=block.get("name"),
                    output=output,
                ).model_dump(exclude_none=True)
                cb.set_data(output_data)
                cb.complete()

            elif btype == "image":
                # image：映射为 MESSAGE，并使用 content_type="image"
                if current_type != MessageType.MESSAGE:
                    if current_mb:
                        current_mb.complete()
                        results.append(current_mb.get_message_data())
                    rb = ResponseBuilder()
                    current_mb = rb.create_message_builder(
                        role=role,
                        message_type=MessageType.MESSAGE,
                    )
                    # add meta field to store old id and name
                    current_mb.message.metadata = {
                        "original_id": msg.id,
                        "original_name": msg.name,
                        "metadata": msg.metadata,
                    }
                    current_type = MessageType.MESSAGE
                cb = current_mb.create_content_builder(content_type="image")

                if (
                    isinstance(block.get("source"), dict)
                    and block.get("source", {}).get("type") == "url"
                ):
                    cb.set_image_url(block.get("source", {}).get("url"))

                elif (
                    isinstance(block.get("source"), dict)
                    and block.get("source").get(
                        "type",
                    )
                    == "base64"
                ):
                    media_type = block.get("source", {}).get(
                        "media_type",
                        "image/jpeg",
                    )
                    base64_data = block.get("source", {}).get("data", "")
                    url = f"data:{media_type};base64,{base64_data}"
                    cb.set_image_url(url)

                cb.complete()

            elif btype == "audio":
                # audio：映射为 MESSAGE，并使用 content_type="audio"
                if current_type != MessageType.MESSAGE:
                    if current_mb:
                        current_mb.complete()
                        results.append(current_mb.get_message_data())
                    rb = ResponseBuilder()
                    current_mb = rb.create_message_builder(
                        role=role,
                        message_type=MessageType.MESSAGE,
                    )
                    # add meta field to store old id and name
                    current_mb.message.metadata = {
                        "original_id": msg.id,
                        "original_name": msg.name,
                        "metadata": msg.metadata,
                    }
                    current_type = MessageType.MESSAGE
                cb = current_mb.create_content_builder(content_type="audio")
                # URLSource runtime check (dict with type == "url")
                if (
                    isinstance(block.get("source"), dict)
                    and block.get("source", {}).get(
                        "type",
                    )
                    == "url"
                ):
                    url = block.get("source", {}).get("url")
                    cb.content.data = url
                    try:
                        cb.content.format = urlparse(url).path.split(".")[-1]
                    except (AttributeError, IndexError, ValueError):
                        cb.content.format = None

                # Base64Source runtime check (dict with type == "base64")
                elif (
                    isinstance(block.get("source"), dict)
                    and block.get("source").get(
                        "type",
                    )
                    == "base64"
                ):
                    media_type = block.get("source", {}).get(
                        "media_type",
                    )
                    base64_data = block.get("source", {}).get("data", "")
                    url = f"data:{media_type};base64,{base64_data}"

                    cb.content.data = url
                    cb.content.format = media_type

                cb.complete()

            else:
                # 未知/兜底 block：退回为 MESSAGE，尽可能把 block 转成文本展示
                if current_type != MessageType.MESSAGE:
                    if current_mb:
                        current_mb.complete()
                        results.append(current_mb.get_message_data())
                    rb = ResponseBuilder()
                    current_mb = rb.create_message_builder(
                        role=role,
                        message_type=MessageType.MESSAGE,
                    )
                    # add meta field to store old id and name
                    current_mb.message.metadata = {
                        "original_id": msg.id,
                        "original_name": msg.name,
                        "metadata": msg.metadata,
                    }
                    current_type = MessageType.MESSAGE
                cb = current_mb.create_content_builder(content_type="text")
                cb.set_text(str(block))
                cb.complete()

        # finalize last open message builder
        if current_mb:
            current_mb.complete()
            results.append(current_mb.get_message_data())

    return results
