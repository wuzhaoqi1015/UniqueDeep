# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/stream/state.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: StreamState manages the state of streaming events for the CLI.
'''

class StreamState:
    """流式处理状态容器"""

    def __init__(self):
        # 统一的事件列表，按顺序存储所有显示的事件
        # 每一项是一个字典：{'type': 'thinking'|'tool'|'response', 'data': ..., 'is_completed': False}
        self.events = []
        
        # 当前正在累积的 thinking 内容
        self.current_thinking = ""
        # 当前正在累积的 response 内容
        self.current_response = ""
        
        # 辅助状态
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False
        
        # 工具调用状态追踪 (用于去重和更新)
        self.tool_map = {} # tool_id -> index in self.events

    def mark_last_event_completed(self):
        """标记最后一个事件为完成（如果存在）"""
        if self.events:
            self.events[-1]["is_completed"] = True

    def handle_event(self, event: dict) -> str:
        event_type = event.get("type")

        # 预处理：如果是文本事件，且当前正在 thinking，且文本内容非常短或者是解释性的，
        # 我们可能需要将其合并到 thinking 中。
        # 但在 agent.py 层面我们已经尝试根据 block_has_tool 做了转换。
        # 这里我们做更激进的兼容：
        # 如果当前事件是 response，但我们发现它实际上是工具调用前的解释（通过查看后续事件或当前状态），
        # 我们可以将其转换。但流式处理很难预知未来。
        # 替代方案：允许 response 和 thinking 共存，但在显示时，如果发现 response 后紧跟 tool，
        # 视觉上将其弱化或合并。不过用户要求是视为 thinking。
        
        # 策略：如果 is_thinking 为 True，且收到了 text，我们不立即结束 thinking，
        # 而是检查这个 text 是否看起来像是"好的，我将调用工具..."之类的废话。
        # 但最简单的办法是：相信 agent.py 的转换。
        # 这里只负责状态流转。

        if event_type == "thinking":
            content = event.get("content", "")
            
            # 如果之前不在思考状态，说明开始了新的一轮思考
            if not self.is_thinking:
                # 如果上一个事件存在且未完成（例如上一个是 response 但还没收到 done），这里需要根据逻辑判断
                # 通常 thinking 是新的一步，意味着上一步（如果是 tool 或 response）应该已经结束了
                # 但为了安全，我们只在明确切换类型时标记完成
                if self.events and not self.events[-1]["is_completed"]:
                     self.events[-1]["is_completed"] = True

                self.is_thinking = True
                self.is_responding = False
                self.is_processing = False
                self.current_thinking = content
                
                # 添加新的 thinking 事件
                self.events.append({
                    "type": "thinking",
                    "content": self.current_thinking,
                    "is_completed": False
                })
            else:
                # 继续累积当前 thinking
                self.current_thinking += content
                # 更新最后一个 thinking 事件的内容
                if self.events and self.events[-1]["type"] == "thinking":
                    self.events[-1]["content"] = self.current_thinking

        elif event_type == "text":
            # 收到文本
            content = event.get("content", "")
            
            # 兼容性逻辑：如果当前正在 thinking，且收到的文本不是特别长（或者符合特定模式），
            # 我们将其追加到 thinking 中，而不是开启新的 response。
            # 这可以解决 Anthropic 将部分推理作为 text 输出的问题。
            # 阈值判断：如果文本以 "Thought:" 开头，或者当前处于 thinking 模式且文本较短
            
            # 但要注意：真正的 response 也可能很短。
            # 关键在于：DeepSeek 的 reasoning_content 是明确分离的。
            # Anthropic 的 thinking 也是分离的，但普通的 Chain of Thought 是 text。
            
            # 如果我们决定将此 text 视为 thinking：
            if self.is_thinking:
                self.current_thinking += content
                if self.events and self.events[-1]["type"] == "thinking":
                    self.events[-1]["content"] = self.current_thinking
                return "thinking" # 伪装成 thinking 事件
            
            # 否则，结束 thinking，开始 response
            if self.is_thinking:
                self.is_thinking = False
                self.mark_last_event_completed()

            self.is_responding = True
            self.is_processing = False
            
            # 响应通常是最后一部分，但也可能是分段的
            if not self.current_response:
                # 如果之前有未完成的事件（非 response），标记为完成
                if self.events and self.events[-1]["type"] != "response":
                     self.events[-1]["is_completed"] = True

                self.current_response = content
                self.events.append({
                    "type": "response",
                    "content": self.current_response,
                    "is_completed": False
                })
            else:
                self.current_response += content
                # 查找并更新响应事件
                for i in range(len(self.events) - 1, -1, -1):
                    if self.events[i]["type"] == "response":
                        self.events[i]["content"] = self.current_response
                        break
                else:
                    self.events.append({
                        "type": "response",
                        "content": self.current_response,
                        "is_completed": False
                    })

        elif event_type == "tool_call":
            # 收到工具调用，意味着思考结束（如果有）
            if self.is_thinking:
                self.is_thinking = False
                self.mark_last_event_completed()
            
            self.is_responding = False
            self.is_processing = False

            tool_id = event.get("id", "")
            tc_data = {
                "id": tool_id,
                "name": event.get("name", "unknown"),
                "args": event.get("args", {}),
                "result": None, # 尚未有结果
                "status": "running"
            }

            if tool_id:
                if tool_id in self.tool_map:
                    # 更新已存在的工具调用
                    idx = self.tool_map[tool_id]
                    self.events[idx]["data"]["args"] = tc_data["args"]
                else:
                    # 如果上一个事件不是工具调用（并行的），且未完成，标记为完成
                    if self.events and self.events[-1]["type"] not in ("tool", "response") and not self.events[-1]["is_completed"]:
                        self.events[-1]["is_completed"] = True

                    # 新工具调用
                    self.events.append({
                        "type": "tool",
                        "data": tc_data,
                        "is_completed": False
                    })
                    self.tool_map[tool_id] = len(self.events) - 1
            else:
                if self.events and self.events[-1]["type"] not in ("tool", "response") and not self.events[-1]["is_completed"]:
                    self.events[-1]["is_completed"] = True
                
                self.events.append({
                    "type": "tool",
                    "data": tc_data,
                    "is_completed": False
                })

        elif event_type == "tool_result":
            self.is_processing = True
            
            # 查找匹配的工具并更新
            target_idx = -1
            # 优先找同名且 running 的
            name = event.get("name", "unknown")
            
            for i in range(len(self.events) - 1, -1, -1):
                evt = self.events[i]
                if evt["type"] == "tool" and evt["data"]["status"] == "running":
                    if evt["data"]["name"] == name:
                        target_idx = i
                        break
            
            # 如果没找到同名的，找任意一个 running 的（fallback）
            if target_idx == -1:
                for i in range(len(self.events) - 1, -1, -1):
                    if self.events[i]["type"] == "tool" and self.events[i]["data"]["status"] == "running":
                        target_idx = i
                        break
            
            if target_idx != -1:
                tool_data = self.events[target_idx]["data"]
                tool_data["status"] = "done"
                tool_data["result"] = {
                    "name": name,
                    "content": event.get("content", "")
                }
                # 工具执行完成，标记该事件为 completed
                self.events[target_idx]["is_completed"] = True
            else:
                pass

        elif event_type == "done":
            self.is_processing = False
            # 标记所有未完成的事件为完成
            for evt in self.events:
                evt["is_completed"] = True
                
            if not self.current_response:
                 # 如果没有流式响应，使用 done 事件中的完整响应
                response = event.get("response", "")
                if response:
                    self.current_response = response
                    self.events.append({
                        "type": "response",
                        "content": self.current_response,
                        "is_completed": True
                    })

        elif event_type == "error":
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            error_msg = event.get("message", "Unknown error")
            
            if self.current_response:
                self.current_response += f"\n\n[Error] {error_msg}"
                for i in range(len(self.events) - 1, -1, -1):
                    if self.events[i]["type"] == "response":
                        self.events[i]["content"] = self.current_response
                        # 出错后，通常响应也结束了
                        self.events[i]["is_completed"] = True
                        break
            else:
                self.current_response = f"[Error] {error_msg}"
                self.events.append({
                    "type": "response",
                    "content": self.current_response,
                    "is_completed": True
                })

        return event_type

    def get_display_args(self) -> dict:
        """获取用于 create_streaming_display 的参数"""
        return {
            "events": self.events,
            "is_waiting": False, # 由外部控制
            "is_processing": self.is_processing
        }
