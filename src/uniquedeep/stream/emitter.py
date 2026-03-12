#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/stream/emitter.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: StreamEventEmitter 类，用于统一流事件（思考、文本、工具调用）。
'''

"""
StreamEventEmitter - 统一事件格式

所有事件都包含 type 和相关数据。
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StreamEvent:
    """统一的流式事件"""

    type: str
    data: Dict[str, Any]


class StreamEventEmitter:
    """流式事件发射器"""

    @staticmethod
    def thinking(content: str, thinking_id: int = 0) -> StreamEvent:
        """思考内容事件（统一为 Agent 输出）"""
        # 为了保持兼容性，我们仍然使用 thinking 类型，但 CLI 会将其显示为 Agent
        # 或者我们直接修改类型为 "agent_output"？
        # 根据用户要求，"都按照目前think的逻辑输出"，意味着我们希望保留 thinking 的蓝色面板样式，
        # 只是名字改为 Agent，并且不再区分 response。
        # 最简单的改法是在 CLI 层面合并，但为了语义清晰，我们这里可以保留 thinking 类型，
        # 或者统一改为 text 类型，但指定 style。
        # 鉴于用户说"按照目前think的逻辑输出"，我们继续使用 thinking 事件，
        # 并在 CLI 中处理显示名称。
        return StreamEvent(
            "thinking", {"type": "thinking", "content": content, "id": thinking_id}
        )

    @staticmethod
    def text(content: str) -> StreamEvent:
        """文本内容事件（转换为 thinking 事件）"""
        # 将所有文本都视为 thinking (即 Agent 输出)
        return StreamEvent("thinking", {"type": "thinking", "content": content})

    @staticmethod
    def response(content: str) -> StreamEvent:
         # 如果有显式的 response 事件（虽然 agent.py 里没有直接用这个方法，是通过 done 或者 text 累积的），
         # 我们也将其转为 thinking
         return StreamEvent("thinking", {"type": "thinking", "content": content})

    @staticmethod
    def tool_call(name: str, args: Dict[str, Any], tool_id: str = "") -> StreamEvent:
        """工具调用事件"""
        return StreamEvent(
            "tool_call",
            {"type": "tool_call", "name": name, "args": args, "id": tool_id},
        )

    @staticmethod
    def tool_result(name: str, content: str, success: bool = True) -> StreamEvent:
        """工具结果事件"""
        return StreamEvent(
            "tool_result",
            {
                "type": "tool_result",
                "name": name,
                "content": content,
                "success": success,
            },
        )

    @staticmethod
    def done(response: str = "") -> StreamEvent:
        """完成事件"""
        return StreamEvent("done", {"type": "done", "response": response})

    @staticmethod
    def error(message: str) -> StreamEvent:
        """错误事件"""
        return StreamEvent("error", {"type": "error", "message": message})
