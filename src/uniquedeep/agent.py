#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/agent.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: LangChainSkillsAgent 的核心实现，处理模型初始化、技能加载和流式传输。
'''

"""
LangChain Skills Agent 主体

使用 LangChain 1.0 的 create_agent API 实现 Skills Agent，演示三层加载机制：
- Level 1: 启动时将 Skills 元数据注入 system_prompt
- Level 2: load_skill tool 加载详细指令
- Level 3: bash tool 执行脚本

与 claude-agent-sdk 实现的对比：
- claude-agent-sdk: setting_sources=["user", "project"] 自动处理
- LangChain 实现: 显式调用 SkillLoader，过程透明可见

流式输出支持：
- 支持 Extended Thinking 显示模型思考过程
- 事件级流式输出 (thinking / text / tool_call / tool_result)
"""

import os
from pathlib import Path
from typing import Optional, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Patch langchain_openai to support reasoning_content serialization for DeepSeek
try:
    from langchain_openai.chat_models import base as openai_chat_base

    if not getattr(openai_chat_base, "_is_patched_for_deepseek", False):
        _original_convert_message_to_dict = openai_chat_base._convert_message_to_dict

        def _convert_message_to_dict_patch(message: BaseMessage) -> dict:
            # Use original implementation
            message_dict = _original_convert_message_to_dict(message)

            # Add reasoning_content if present in additional_kwargs
            if isinstance(message, AIMessage):
                reasoning_content = message.additional_kwargs.get("reasoning_content")
                if reasoning_content:
                    message_dict["reasoning_content"] = reasoning_content
            return message_dict

        # Apply patch to module function
        openai_chat_base._convert_message_to_dict = _convert_message_to_dict_patch
        openai_chat_base._is_patched_for_deepseek = True
except ImportError:
    pass

from .skill_loader import SkillLoader
from .tools import ALL_TOOLS, SkillAgentContext
from .stream import StreamEventEmitter, ToolCallTracker, is_success, DisplayLimits


# 加载环境变量（override=True 确保 .env 文件覆盖系统环境变量）
load_dotenv(override=True)


# 默认配置
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 1.0  # claude Extended Thinking 要求温度为 1.0
DEFAULT_THINKING_BUDGET = 10000


def get_model_config() -> tuple[str, str | None, str | None, str | None]:
    """
    获取模型配置

    Returns:
        (provider, model_name, api_key, base_url)
    """
    # 1. 确定 Provider 和 Model Name
    provider = os.getenv("LLM_PROVIDER", "").lower()
    model_name = os.getenv("LLM_MODEL", "")

    # 兼容旧的环境变量：仅当 provider 未指定或明确为 anthropic 时，才回退到 CLAUDE_MODEL
    if not model_name and (not provider or provider == "anthropic"):
        model_name = os.getenv("CLAUDE_MODEL", "")

    # 如果未指定 provider，尝试根据模型名称推断
    if not provider and model_name:
        if "claude" in model_name.lower():
            provider = "anthropic"
        elif "deepseek" in model_name.lower():
            provider = "deepseek"
        elif "gpt" in model_name.lower() or "o1-" in model_name.lower():
            provider = "openai"

    # 默认值
    if not provider:
        # 检查是否有特定厂商的 key 来推断默认 provider
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            provider = "anthropic"
        elif os.getenv("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "anthropic"  # 最终默认

    if not model_name:
        if provider == "anthropic":
            model_name = DEFAULT_MODEL
        elif provider == "deepseek":
            model_name = "deepseek-reasoner"
        elif provider == "openai":
            model_name = "o1-preview"  # OpenAI 默认模型
        else:
            model_name = DEFAULT_MODEL

    # 2. 获取 API Key 和 Base URL
    api_key = None
    base_url = None

    prefix = provider.upper()

    # 特殊处理 Anthropic 的 Auth Token
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    else:
        api_key = os.getenv(f"{prefix}_API_KEY")

    base_url = os.getenv(f"{prefix}_BASE_URL")

    # DeepSeek 默认 Base URL
    if provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com"

    return provider, model_name, api_key, base_url


def check_api_credentials() -> bool:
    """检查当前配置的模型是否有对应的凭证"""
    _, _, api_key, _ = get_model_config()
    return api_key is not None


class LangChainSkillsAgent:
    """
    基于 LangChain 1.0 的 Skills Agent

    演示目的：展示 Skills 三层加载机制的底层原理

    使用示例：
        agent = LangChainSkillsAgent()

        # 查看 system prompt（展示 Level 1）
        print(agent.get_system_prompt())

        # 运行 agent
        for chunk in agent.stream("搜索关于GJB2基因的最新论文进展"):
            response = agent.get_last_response(chunk)
            if response:
                print(response)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        skill_paths: Optional[list[Path]] = None,
        working_directory: Optional[Path] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ):
        """
        初始化 Agent

        Args:
            model: 模型名称，默认 claude-sonnet-4-5-20250929
            skill_paths: Skills 搜索路径
            working_directory: 工作目录
            max_tokens: 最大 tokens
            temperature: 温度参数 (启用 thinking 时强制为 1.0)
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: thinking 的 token 预算
        """
        # thinking 配置
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        # 配置 (启用 thinking 时温度必须为 1.0)

        # 获取模型配置
        provider, model_name, self.api_key, self.base_url = get_model_config()

        self.model_name = model or model_name
        self.provider = provider

        # Print loaded configuration for debugging
        if (
            os.getenv("SKILLS_DEBUG", "").lower() in ("1", "true", "yes") or True
        ):  # Always print for now to help user debug
            print(f"[Config] Provider: {self.provider}")
            print(f"[Config] Model: {self.model_name}")
            if self.base_url:
                print(f"[Config] Base URL: {self.base_url}")

        self.max_tokens = max_tokens or int(
            os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )

        # Anthropic Extended Thinking 要求温度为 1.0
        is_anthropic = (
            self.provider == "anthropic" or "claude" in self.model_name.lower()
        )
        if enable_thinking and is_anthropic:
            self.temperature = 1.0
        else:
            self.temperature = temperature or float(
                os.getenv("DEFAULT_TEMPERATURE", str(DEFAULT_TEMPERATURE))
            )
        self.working_directory = working_directory or Path.cwd()

        # 初始化 SkillLoader
        self.skill_loader = SkillLoader(skill_paths)

        # Level 1: 构建 system prompt（将 Skills 元数据注入）
        self.system_prompt = self._build_system_prompt()

        # 创建上下文（供 tools 使用）
        self.context = SkillAgentContext(
            skill_loader=self.skill_loader,
            working_directory=self.working_directory,
        )

        # 创建 LangChain Agent
        self.agent = self._create_agent()

    def _build_system_prompt(self) -> str:
        """
        构建 system prompt

        这是 Level 1 的核心：将所有 Skills 的元数据注入到 system prompt。
        每个 skill 约 100 tokens，启动时一次性加载。
        """
        base_prompt = """You are a helpful coding assistant with access to specialized skills.

Your capabilities include:
- Loading and using specialized skills for specific tasks
- Executing bash commands and scripts
- Reading and writing files
- Following skill instructions to complete complex tasks

When a user request matches a skill's description, use the load_skill tool to get detailed instructions before proceeding.

Note: The user may switch models during the conversation. System markers like "[System Note] Context Switch..." indicate these transitions. Each marker defines the boundary of the conversation segment generated by the preceding model. Be aware that different segments may reflect different model capabilities or behaviors."""

        return self.skill_loader.build_system_prompt(base_prompt)

    def _create_agent(self):
        """
        创建 LangChain Agent

        使用 LangChain 1.0 的 create_agent API:
        - model: 可以是字符串 ID 或 model 实例
        - tools: 工具列表
        - system_prompt: 系统提示（Level 1 注入 Skills 元数据）
        - context_schema: 上下文类型（供 ToolRuntime 使用）
        - checkpointer: 会话记忆

        Extended Thinking 支持:
        - 启用后可获取模型的思考过程
        - 温度必须为 1.0

        认证支持:
        - 支持 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN
        - 支持 ANTHROPIC_BASE_URL 第三方代理
        - 支持 DEEPSEEK_API_KEY (通过 OpenAI 兼容接口)
        - 支持其他 OpenAI 兼容接口
        """
        # 构建初始化参数
        init_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 显式指定 provider，避免 init_chat_model 猜测错误
        if self.provider:
            init_kwargs["model_provider"] = self.provider

        # 添加认证参数
        if self.api_key:
            init_kwargs["api_key"] = self.api_key

        if self.base_url:
            # DeepSeek SDK 使用 api_base 而不是 base_url，OpenAI 也常用 api_base
            if self.provider == "deepseek" or self.provider == "openai":
                init_kwargs["api_base"] = self.base_url
            else:
                init_kwargs["base_url"] = self.base_url

        # Extended Thinking 配置（仅 Anthropic 支持）
        if self.enable_thinking and self.provider == "anthropic":
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        # 初始化模型
        # 对于 DeepSeek 模型，max_tokens 不能超过 8192
        if "deepseek" in self.model_name.lower():
            init_kwargs["max_tokens"] = min(self.max_tokens, 65535)

        model = init_chat_model(
            self.model_name,
            **init_kwargs,
        )

        # 组合工具
        tools = list(ALL_TOOLS)

        # 确保 checkpointer 持久化 (支持 set_temperature 重建 agent)
        if not hasattr(self, "checkpointer"):
            self.checkpointer = InMemorySaver()

        # 创建 Agent
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=self.system_prompt,
            context_schema=SkillAgentContext,
            checkpointer=self.checkpointer,
        )

        return agent

    def get_system_prompt(self) -> str:
        """
        获取当前 system prompt

        用于演示和调试，展示 Level 1 注入的内容。
        """
        return self.system_prompt

    def get_discovered_skills(self) -> list[dict]:
        """
        获取发现的 Skills 列表

        用于演示 Level 1 的 Skills 发现过程。
        """
        skills = self.skill_loader.scan_skills()
        return [
            {
                "name": s.name,
                "description": s.description,
                "path": str(s.skill_path),
            }
            for s in skills
        ]

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        同步调用 Agent

        Args:
            message: 用户消息
            thread_id: 会话 ID（用于多轮对话）

        Returns:
            Agent 响应
        """
        config = {"configurable": {"thread_id": thread_id}}

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
        )

        return result

    def stream(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        流式调用 Agent (state 级别)

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            流式响应块 (完整状态更新)
        """
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=self.context,
            stream_mode="values",
        ):
            yield chunk

    def set_temperature(self, temperature: float) -> bool:
        """
        动态设置温度并重建 Agent
        
        Args:
            temperature: 新的温度值 (0.0 - 1.0)
            
        Returns:
            是否成功设置（如果启用了 Extended Thinking，可能无法更改）
        """
        # 检查是否允许更改
        is_anthropic = (
            self.provider == "anthropic" or "claude" in self.model_name.lower()
        )
        if self.enable_thinking and is_anthropic:
            # Extended Thinking 强制要求 temperature=1.0
            return False
            
        self.temperature = temperature
        # 重建 Agent 以应用新配置
        self.agent = self._create_agent()
        return True

    def switch_model(self, model_name: str, provider: str | None = None, thread_id: str = "default") -> bool:
        """
        动态切换模型
        
        Args:
            model_name: 新的模型名称
            provider: 可选的提供商名称
            thread_id: 会话 ID，用于插入切换标记
            
        Returns:
            是否切换成功
        """
        old_model = self.model_name

        # 1. 更新配置
        self.model_name = model_name
        if provider:
            self.provider = provider
        else:
            # 自动推断 provider
            if "claude" in model_name.lower():
                self.provider = "anthropic"
            elif "deepseek" in model_name.lower():
                self.provider = "deepseek"
            elif "gpt" in model_name.lower() or "o1-" in model_name.lower():
                self.provider = "openai"
        
        # 2. 更新 API Key 和 Base URL
        # 注意：这里我们重新读取环境变量，确保如果有新的 key 可以被加载
        _, _, api_key, base_url = get_model_config()
        
        # 如果当前 provider 与环境变量不一致，我们需要尝试获取对应 provider 的 key
        prefix = self.provider.upper()
        if self.provider == "anthropic":
            new_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        else:
            new_key = os.getenv(f"{prefix}_API_KEY")
            
        new_base_url = os.getenv(f"{prefix}_BASE_URL")
        
        # DeepSeek 默认 Base URL
        if self.provider == "deepseek" and not new_base_url:
            new_base_url = "https://api.deepseek.com"
            
        if new_key:
            self.api_key = new_key
        if new_base_url:
            self.base_url = new_base_url
            
        # 3. 处理 Extended Thinking 兼容性及温度设置
        is_anthropic = (
            self.provider == "anthropic" or "claude" in self.model_name.lower()
        )
        
        # 如果切换到 Anthropic/Claude 模型
        if is_anthropic:
            # 如果原本启用了 thinking（或者用户希望启用），则强制温度为 1.0
            # 这里我们默认如果是 Claude，且全局配置允许 thinking，就尝试启用
            # 注意：self.enable_thinking 可能在之前切到其他模型时被禁用了，
            # 这里我们根据环境变量中的默认配置来决定是否重新启用
            default_enable_thinking = not os.getenv("NO_THINKING", "false").lower() == "true"
            
            if default_enable_thinking:
                self.enable_thinking = True
                self.temperature = 1.0
                print(f"[Info] Switched to Claude model {model_name}. Extended Thinking enabled (temperature=1.0).")
            else:
                # 如果没有启用 thinking，则使用默认温度
                self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
                print(f"[Info] Switched to Claude model {model_name}. Extended Thinking disabled by default config.")
        
        # 如果切换到非 Anthropic 模型
        else:
            if self.enable_thinking:
                print(f"[Warn] Disabling Extended Thinking for non-Claude model {model_name}")
                self.enable_thinking = False
                
            # 恢复默认温度 (避免之前的 1.0 影响其他模型，虽然 1.0 对大多数模型也是合理的，但还是恢复默认更安全)
            self.temperature = float(os.getenv("DEFAULT_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            print(f"[Info] Temperature reset to {self.temperature}")
            
        # 4. 重建 Agent
        try:
            self.agent = self._create_agent()
            
            # 插入切换标记
            config = {"configurable": {"thread_id": thread_id}}
            marker_content = f"[System Note] Context Switch: The acting model has changed from {old_model} to {self.model_name}. The conversation segment immediately preceding this note was generated by {old_model}."
            
            # 尝试插入 HumanMessage 到对话历史 (SystemMessage 不能在中间)
            if hasattr(self.agent, "update_state"):
                try:
                    self.agent.update_state(config, {"messages": [HumanMessage(content=marker_content)]})
                except Exception as e:
                    print(f"[Warn] Failed to insert switch marker: {e}")
                    
            return True
        except Exception as e:
            print(f"[Error] Failed to switch model: {e}")
            return False

    def stream_events(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        事件级流式输出，支持 thinking 和 token 级流式
        
        Args:
            message: 用户消息
            thread_id: 会话 ID
            
        Yields:
            事件字典，格式如下:
            - {"type": "thinking", "content": "..."} - 思考内容片段
            - {"type": "text", "content": "..."} - 响应文本片段
            - {"type": "tool_call", "name": "...", "args": {...}, "id": "..."} - 工具调用
            - {"type": "tool_result", "name": "...", "content": "...", "success": bool} - 工具结果
            - {"type": "error", "content": "..."} - 错误信息
            - {"type": "done", "response": "..."} - 完成标记，包含完整响应
        """
        config = {"configurable": {"thread_id": thread_id}}
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()
        
        full_response = ""
        debug = os.getenv("SKILLS_DEBUG", "").lower() in ("1", "true", "yes")
        
        # 使用 messages 模式获取 token 级流式
        try:
            for event in self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                config=config,
                context=self.context,
                stream_mode="messages",
            ):
                # event 可能是 tuple(message, metadata) 或直接 message
                if isinstance(event, tuple) and len(event) >= 2:
                    chunk = event[0]
                else:
                    chunk = event
                
                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")
                
                # 处理 AIMessageChunk / AIMessage
                if isinstance(chunk, (AIMessageChunk, AIMessage)):
                    # 处理 content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data
                    
                    # 处理 tool_calls (有些情况下在 chunk.tool_calls 中)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(
                            chunk.tool_calls, emitter, tracker
                        ):
                            if debug:
                                print(f"[DEBUG] Yielding from tool_calls: {ev.type}")
                            yield ev.data
                
                # 处理 ToolMessage (工具执行结果)
                elif hasattr(chunk, "type") and chunk.type == "tool":
                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data
            
            if debug:
                print("[DEBUG] Stream completed normally")
            
        except Exception as e:
            if debug:
                import traceback
                
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            # 发送错误事件让用户知道发生了什么
            yield emitter.error(str(e)).data
            raise
        
        # 发送完成事件
        yield emitter.done(full_response).data

    def _process_chunk_content(
        self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker
    ):
        """处理 chunk 的 content"""
        content = chunk.content

        if isinstance(content, str):
            if content:
                yield emitter.text(content)
                return

        blocks = None
        if hasattr(chunk, "content_blocks"):
            try:
                blocks = chunk.content_blocks
            except Exception:
                blocks = None

        if blocks is None:
            if isinstance(content, dict):
                blocks = [content]
            elif isinstance(content, list):
                blocks = content
            else:
                return

        for raw_block in blocks:
            block = raw_block
            if not isinstance(block, dict):
                if hasattr(block, "model_dump"):
                    block = block.model_dump()
                elif hasattr(block, "dict"):
                    block = block.dict()
                else:
                    continue

            block_type = block.get("type")

            if block_type in ("thinking", "reasoning"):
                thinking_text = block.get("thinking") or block.get("reasoning") or ""
                if thinking_text:
                    yield emitter.thinking(thinking_text)

            elif block_type == "text":
                text = block.get("text") or block.get("content") or ""
                if text:
                    yield emitter.text(text)

            elif block_type in ("tool_use", "tool_call"):
                tool_id = block.get("id", "")
                name = block.get("name", "")
                args = (
                    block.get("input")
                    if block_type == "tool_use"
                    else block.get("args")
                )
                args_payload = args if isinstance(args, dict) else {}

                if tool_id:
                    tracker.update(tool_id, name=name, args=args_payload)
                    # 立即发送（显示"执行中"状态），参数可能尚不完整
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                # 累积 JSON 片段（args 分批到达）
                partial_json = block.get("partial_json", "")
                if partial_json:
                    tracker.append_json_delta(partial_json, block.get("index", 0))

            elif block_type == "tool_call_chunk":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id:
                    tracker.update(tool_id, name=name)
                partial_args = block.get("args", "")
                if isinstance(partial_args, str) and partial_args:
                    tracker.append_json_delta(partial_args, block.get("index", 0))

    def _handle_tool_use_block(
        self, block: dict, emitter: StreamEventEmitter, tracker: ToolCallTracker
    ):
        """处理 tool_use 块 - 立即发送 tool_call 事件

        在收到 tool_use 时立即发送，让 CLI 可以显示"正在执行"状态。
        避免重复发送（同一 tool 可能通过多个路径到达）。
        """
        tool_id = block.get("id", "")
        if tool_id:
            name = block.get("name", "")
            args = block.get("input", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_calls(
        self, tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker
    ):
        """处理 chunk.tool_calls - 立即发送 tool_call 事件

        避免重复发送（同一 tool 可能通过 tool_use block 已发送）。
        """
        for tc in tool_calls:
            tool_id = tc.get("id", "")
            if tool_id:
                name = tc.get("name", "")
                args = tc.get("args", {})
                args_payload = args if isinstance(args, dict) else {}

                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_result(
        self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker
    ):
        """处理工具结果"""
        # 最终化：解析累积的 JSON 片段为 args
        tracker.finalize_all()

        # 发送所有工具调用的更新（参数现在是完整的）
        # CLI 会用 tool_id 去重和更新
        for info in tracker.get_all():
            yield emitter.tool_call(info.name, info.args, info.id)

        # 发送结果
        name = getattr(chunk, "name", "unknown")
        raw_content = str(getattr(chunk, "content", ""))
        content = raw_content[: DisplayLimits.TOOL_RESULT_MAX]
        if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
            content += "\n... (truncated)"

        # 基于内容判断是否成功（统一使用 is_success）
        success = is_success(content)

        yield emitter.tool_result(name, content, success)

    def get_last_response(self, result: dict) -> str:
        """
        从结果中提取最后的 AI 响应文本

        Args:
            result: invoke 或 stream 的结果

        Returns:
            AI 响应文本
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # 处理多部分内容
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts)
        return ""


def create_skills_agent(
    model: Optional[str] = None,
    skill_paths: Optional[list[Path]] = None,
    working_directory: Optional[Path] = None,
    enable_thinking: bool = True,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> LangChainSkillsAgent:
    """
    便捷函数：创建 Skills Agent

    Args:
        model: 模型名称
        skill_paths: Skills 搜索路径
        working_directory: 工作目录
        enable_thinking: 是否启用 Extended Thinking
        thinking_budget: thinking 的 token 预算

    Returns:
        配置好的 LangChainSkillsAgent 实例
    """
    return LangChainSkillsAgent(
        model=model,
        skill_paths=skill_paths,
        working_directory=working_directory,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
