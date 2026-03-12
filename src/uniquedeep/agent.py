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
import re
import json
import warnings
from pathlib import Path
from typing import Optional, Iterator

# 忽略 ZhipuAI API Key 长度不足的警告 (GLM-5 等模型使用 HS256 签名，Key 较短会导致 cryptography 库发出警告)
warnings.filterwarnings("ignore", message=".*key is shorter than the recommended length.*")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, HumanMessage, ToolMessage
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
    
    1. 优先尝试从 models.json 读取 active_model
    2. 如果没有 models.json，回退到环境变量
    """
    # 尝试加载 models.json
    config = {}
    try:
        current_dir = Path.cwd()
        root_dir = Path(__file__).parent.parent.parent
        paths = [current_dir / "models.json", root_dir / "models.json"]
        
        config_path = None
        for p in paths:
            if p.exists():
                config_path = p
                break
        
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
    except Exception:
        pass

    # 从 json 获取
    provider = config.get("active_provider", "").lower()
    model_name = config.get("active_model", "")
    
    # 如果 json 没配置，从环境变量获取
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "").lower()
    if not model_name:
        model_name = os.getenv("LLM_MODEL", "")

    # 兼容旧的环境变量
    if not model_name and (not provider or provider == "anthropic"):
        model_name = os.getenv("CLAUDE_MODEL", "")

    # 如果还是没有，尝试根据名称推断 provider
    if not provider and model_name:
        if "claude" in model_name.lower():
            provider = "anthropic"
        elif "deepseek" in model_name.lower():
            provider = "deepseek"
        elif "gpt" in model_name.lower() or "o1-" in model_name.lower():
            provider = "openai"
        elif "glm" in model_name.lower():
            provider = "zhipuai"
        elif "kimi" in model_name.lower() or "moonshot" in model_name.lower():
            provider = "moonshot"
        elif "doubao" in model_name.lower():
            provider = "doubao"

    # 默认值
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
            provider = "anthropic"
        elif os.getenv("DEEPSEEK_API_KEY"):
            provider = "deepseek"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY"):
            provider = "zhipuai"
        else:
            provider = "anthropic"

    if not model_name:
        if provider == "anthropic":
            model_name = DEFAULT_MODEL
        elif provider == "deepseek":
            model_name = "deepseek-reasoner"
        elif provider == "openai":
            model_name = "o1-preview"
        elif provider == "zhipuai":
            model_name = "glm-4"
        else:
            model_name = DEFAULT_MODEL

    # 获取 API Key 和 Base URL
    api_key = None
    base_url = None
    
    # 1. 尝试从 models.json 获取配置 (支持 ${VAR} 环境变量引用)
    if provider and config.get("providers"):
        provider_config = config["providers"].get(provider, {})
        api_key = provider_config.get("api_key", "")
        base_url = provider_config.get("base_url", "")
        
        # 处理环境变量引用
        if api_key and isinstance(api_key, str):
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.getenv(env_var)
            elif api_key.startswith("$"):
                api_key = os.path.expandvars(api_key)
                
        if base_url and isinstance(base_url, str):
            if base_url.startswith("${") and base_url.endswith("}"):
                env_var = base_url[2:-1]
                base_url = os.getenv(env_var)
            elif base_url.startswith("$"):
                base_url = os.path.expandvars(base_url)
        
    # 2. 如果 json 没提供，回退到通用或特定环境变量
    # (仅当 json 中的值为空字符串时回退，如果 json 里写了 "sk-xxx" 则直接用)
    if not api_key:
        api_key = os.getenv("LLM_API_KEY")
    if not base_url:
        base_url = os.getenv("LLM_BASE_URL")

    prefix = provider.upper()
    if not api_key:
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        elif provider == "zhipuai":
            api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
        else:
            api_key = os.getenv(f"{prefix}_API_KEY")

    if not base_url:
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
        provider: Optional[str] = None,
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
            provider: 模型提供商 (anthropic, deepseek, openai)
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
        env_provider, env_model_name, self.api_key, self.base_url = get_model_config()

        self.model_name = model or env_model_name
        self.provider = provider or env_provider
        
        # Re-fetch API key if provider was overridden or different from env
        # Or simply always try to fetch the best config for the current model/provider
        if self.provider != env_provider or self.model_name != env_model_name:
             # Try to get config from models.json for the specific provider/model
             specific_config = self._get_model_specific_config(self.provider, self.model_name)
             
             if specific_config.get("api_key"):
                 self.api_key = specific_config["api_key"]
             else:
                 # Fallback to env vars
                 prefix = self.provider.upper()
                 if self.provider == "anthropic":
                     self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
                 elif self.provider == "zhipuai":
                     self.api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
                 else:
                     self.api_key = os.getenv(f"{prefix}_API_KEY")
             
             if specific_config.get("base_url"):
                 self.base_url = specific_config["base_url"]
             else:
                 prefix = self.provider.upper()
                 self.base_url = os.getenv(f"{prefix}_BASE_URL")

             if self.provider == "deepseek" and not self.base_url:
                 self.base_url = "https://api.deepseek.com"

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

        # 思考标签状态追踪
        self._tag_buffer = ""
        self._in_thinking_tag = False
        self._current_end_tag = "</thinking>"

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

Note: The user may switch models during the conversation. System markers like "[System Note] Context Switch..." indicate these transitions. Each marker defines the boundary of the conversation segment generated by the preceding model. Be aware that different segments may reflect different model capabilities or behaviors.

IMPORTANT: When you need to think before acting or responding (e.g., analyzing a request, deciding which tool to use, or planning steps), you MUST wrap your thoughts in <thinking>...</thinking> tags. This helps separate your internal reasoning from your final response.
For example:
<thinking>
The user wants to find a file. I should use the `find` command.
</thinking>
I will search for the file now."""

        # 针对 ZhipuAI / Moonshot / Doubao 等模型加强提示
        is_anthropic = (self.provider == "anthropic" or "claude" in self.model_name.lower())
        is_deepseek = (self.provider == "deepseek" or "reasoner" in self.model_name.lower())
        
        if self.enable_thinking and not is_anthropic and not is_deepseek:
             base_prompt += """
             
Please output your thinking process enclosed in <thinking> tags before your final response.
"""

        return self.skill_loader.build_system_prompt(base_prompt)

    def _get_model_specific_config(self, provider: str, model_name: str) -> dict:
        """
        从 models.json 获取特定模型的配置
        
        Returns:
            dict: 包含 api_key, base_url, temperature, thinking 等配置
        """
        config = {}
        try:
            current_dir = Path.cwd()
            root_dir = Path(__file__).parent.parent.parent
            paths = [current_dir / "models.json", root_dir / "models.json"]
            
            config_path = None
            for p in paths:
                if p.exists():
                    config_path = p
                    break
            
            if config_path:
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                    
                # 查找 provider 配置
                provider_config = full_config.get("providers", {}).get(provider, {})
                
                # 基础配置 (API Key, Base URL)
                api_key = provider_config.get("api_key")
                base_url = provider_config.get("base_url")
                
                # 处理环境变量引用 ${VAR}
                if api_key and isinstance(api_key, str):
                    if api_key.startswith("${") and api_key.endswith("}"):
                        env_var = api_key[2:-1]
                        api_key = os.getenv(env_var)
                    elif api_key.startswith("$"):
                        api_key = os.path.expandvars(api_key)
                
                if base_url and isinstance(base_url, str):
                    if base_url.startswith("${") and base_url.endswith("}"):
                        env_var = base_url[2:-1]
                        base_url = os.getenv(env_var)
                    elif base_url.startswith("$"):
                        base_url = os.path.expandvars(base_url)
                
                config["api_key"] = api_key
                config["base_url"] = base_url
                
                # 查找特定模型配置
                models = provider_config.get("models", [])
                for m in models:
                    if m["name"] == model_name:
                        # 合并模型特定配置
                        if "temperature" in m:
                            config["temperature"] = m["temperature"]
                        if "thinking" in m:
                            config["thinking"] = m["thinking"]
                        if "max_tokens" in m:
                            config["max_tokens"] = m["max_tokens"]
                        break
                        
                # 获取默认配置作为兜底
                default_config = full_config.get("default_config", {})
                if "temperature" not in config and "temperature" in default_config:
                    config["default_temperature"] = default_config["temperature"]
                if "max_tokens" not in config and "max_tokens" in default_config:
                    config["default_max_tokens"] = default_config["max_tokens"]
                    
        except Exception:
            pass
            
        return config

    def _init_chat_model(self):
        """
        初始化 ChatModel
        """
        # 1. 确定 Temperature
        temperature = self.temperature
        
        # 尝试从 models.json 获取配置
        model_config = self._get_model_specific_config(self.provider, self.model_name)
        if "temperature" in model_config:
            temperature = model_config["temperature"]

        # 确定 max_tokens
        max_tokens = self.max_tokens
        if "max_tokens" in model_config:
            max_tokens = model_config["max_tokens"]
        elif "default_max_tokens" in model_config:
            max_tokens = model_config["default_max_tokens"]

        # 2. 构建初始化参数
        init_kwargs = {}
        
        # Anthropic Thinking 配置
        if self.enable_thinking and self.provider == "anthropic":
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Claude Extended Thinking 强制要求 temperature=1.0 (如果 json 里没配，这里兜底)
            if temperature != 1.0:
                 temperature = 1.0

        # 通用参数
        common_kwargs = {
            "temperature": temperature,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            common_kwargs["base_url"] = self.base_url
            
        if max_tokens:
             common_kwargs["max_tokens"] = max_tokens

        # 合并特定参数
        kwargs = {**common_kwargs, **init_kwargs}

        if self.provider == "zhipuai":
            # 使用 langchain-community 的 ChatZhipuAI (智谱官方 SDK 封装)
            from langchain_community.chat_models import ChatZhipuAI
            return ChatZhipuAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=temperature,
            )
            
        elif self.provider == "moonshot":
            # Moonshot (Kimi) 兼容 OpenAI 协议
            # 使用 ChatOpenAI，但 provider 设为 openai (因为 langchain 不认识 moonshot)
            # 关键是 base_url 指向 moonshot
            if "max_tokens" in kwargs:
                # Kimi 的 max_tokens 含义可能不同，或者不需要显式传递
                pass
            return init_chat_model(
                self.model_name,
                model_provider="openai",
                **kwargs
            )
            
        elif self.provider == "doubao":
            # Doubao 兼容 OpenAI 协议
            return init_chat_model(
                self.model_name,
                model_provider="openai",
                **kwargs
            )
            
        elif self.provider == "deepseek":
            # DeepSeek 兼容 OpenAI 协议，但也可能有专用 provider
            # langchain-deepseek 提供了 ChatDeepSeek
            # 但 init_chat_model 可能只认 "deepseek"
            return init_chat_model(
                self.model_name,
                model_provider="deepseek",
                **kwargs
            )

        # 默认使用 langchain 的工厂方法
        return init_chat_model(
            self.model_name,
            model_provider=self.provider,
            **kwargs
        )

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
        # 初始化模型
        model = self._init_chat_model()

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

        # 其他思考模型限制 (如 Kimi k2.5)
        if self.enable_thinking and "kimi-k2.5" in self.model_name and temperature != 1.0:
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
        # 优先从 models.json 获取配置
        # (get_model_config 只能获取当前 active_model 的配置，不适用于切换目标)
        model_config = self._get_model_specific_config(self.provider, self.model_name)
        
        if model_config.get("api_key"):
            self.api_key = model_config["api_key"]
            
        if model_config.get("base_url"):
            self.base_url = model_config["base_url"]

        # 如果 json 没配 API Key，尝试从环境变量兜底
        if not self.api_key:
             prefix = self.provider.upper()
             if self.provider == "anthropic":
                 self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
             elif self.provider == "zhipuai":
                 self.api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
             else:
                 self.api_key = os.getenv(f"{prefix}_API_KEY")
                 
        # DeepSeek 默认 Base URL
        if self.provider == "deepseek" and not self.base_url:
            self.base_url = "https://api.deepseek.com"

        # 3. 处理 Extended Thinking 兼容性及温度设置
        
        # 获取新模型的配置
        model_config = self._get_model_specific_config(self.provider, self.model_name)
        
        # 优先使用 json 配置的温度
        if "temperature" in model_config:
            self.temperature = model_config["temperature"]
            print(f"[Info] Temperature set to {self.temperature} (from models.json)")
        else:
            # 回退到默认
            default_temp = model_config.get("default_temperature", float(os.getenv("DEFAULT_TEMPERATURE", str(DEFAULT_TEMPERATURE))))
            
            # Anthropic 特殊处理
            is_anthropic = (self.provider == "anthropic" or "claude" in self.model_name.lower())
            
            # 如果是 Anthropic 且启用了 thinking，强制 1.0
            # 注意：这里我们还需要决定是否启用 thinking
            # 如果 json 里指定了 thinking: true，我们应该倾向于启用
            json_thinking = model_config.get("thinking", False)
            
            if is_anthropic and (self.enable_thinking or json_thinking):
                 self.temperature = 1.0
            else:
                 self.temperature = default_temp
            
            print(f"[Info] Temperature reset to {self.temperature}")

        # 更新 max_tokens
        if "max_tokens" in model_config:
            self.max_tokens = model_config["max_tokens"]
        elif "default_max_tokens" in model_config:
            self.max_tokens = model_config["default_max_tokens"]
        # else keep existing self.max_tokens or reset to global default? 
        # For safety, maybe we should not reset if not found, as self.max_tokens might be env set.
        # But consistency implies we should follow the config priority.
        
        # 处理 Thinking 状态
        # 如果 json 明确配置了 thinking，则更新状态
        if "thinking" in model_config:
            should_think = model_config["thinking"]
            if should_think and not self.enable_thinking:
                self.enable_thinking = True
                print(f"[Info] Extended Thinking enabled for {self.model_name} (from models.json)")
            elif not should_think and self.enable_thinking and not is_anthropic:
                # 只有非 Anthropic 模型才会被 json 配置强制关闭 thinking
                # 因为 Anthropic 的 thinking 是动态参数，而其他模型可能是原生能力
                # 如果 json 说不支持 (false)，那我们最好关闭，以免出错
                # 但目前我们只有 true 的情况需要处理
                pass
        
        # Anthropic 的特殊逻辑保留一部分
        is_anthropic = (self.provider == "anthropic" or "claude" in self.model_name.lower())
        if is_anthropic:
            if self.enable_thinking:
                self.temperature = 1.0
                print(f"[Info] Extended Thinking enabled for Claude (temperature=1.0)")
        else:
             # 非 Anthropic 模型，如果 enable_thinking 为 True，但 json 没说支持，
             # 或者是之前残留的状态，我们是否要关闭？
             # 如果用户在 CLI 启用了 --thinking，我们应该尝试支持（通过 prompting）
             # 除非模型明确不支持。
             pass

            
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
            事件字典
        """
        config = {"configurable": {"thread_id": thread_id}}
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()
        
        full_response = ""
        debug = os.getenv("SKILLS_DEBUG", "").lower() in ("1", "true", "yes")
        
        # 使用 messages 模式获取 token 级流式
        # 关键修正：stream_mode="messages" 会返回整个对话历史中的所有消息更新
        # 如果 thread_id 是旧的，LangChain 会先 yield 历史消息。
        # 我们必须过滤掉历史消息，只处理当前生成的 chunks。
        # 如何区分？
        # 1. 历史消息通常是完整的 BaseMessage 对象，而不是 Chunk。
        # 2. 或者通过消息 ID？
        # 3. LangChain 文档建议：使用 stream_events(version="v2") API 更好，
        #    或者在 stream 中只关注最后一条消息的更新。
        
        # 目前我们使用的是 agent.stream(..., stream_mode="messages")
        # 这会 yield (message, metadata) 元组或 message 对象。
        # 在 ReAct 循环中，它会 yield 每一步的输出。
        
        # 如果我们改用 stream_events API？
        # LangGraph 的 stream_events API 更适合。
        # 但我们用的是 CompiledGraph.stream。
        
        # 观察：用户说"第一轮输出aaaa，第二轮输出aaaa\nbbbb"。
        # 这说明第二轮 stream 时，不仅 yield 了 bbbb，还 yield 了 aaaa。
        # 这通常发生在 stream_mode="values" 时（返回整个 state）。
        # 但我们用的是 "messages"。
        # LangGraph 的 "messages" mode yields list of messages? Or message chunks?
        # 文档：stream_mode="messages" yields (message, metadata) for each message emitted.
        # 如果是聊天历史，它不应该重放历史，除非 graph 逻辑里有重放。
        
        # 另一个可能：我们自己在 CLI 里累积了？
        # CLI 的 StreamState 是新的。
        
        # 让我们看看 InMemorySaver。如果 checkpointer 保存了历史，
        # 而我们的 agent 是一个 LangGraph compiled graph。
        
        # 让我们加一个简单的过滤器：只处理 AIMessageChunk。
        # 历史消息通常是 AIMessage（非 Chunk）。
        # 新生成的内容是 AIMessageChunk。
        
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
                
                # 过滤掉非 Chunk 的消息（通常是历史记录或完整消息回显）
                # 只有 AIMessageChunk 代表流式增量
                # ToolMessage 也是完整的，但它是新产生的执行结果，需要处理。
                # 普通 AIMessage (非 Chunk) 可能是历史记录。
                if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk):
                    if debug:
                        print(f"[DEBUG] Skipping non-chunk AIMessage: {chunk.content[:20]}...")
                    continue
                
                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")
                
                # 处理 AIMessageChunk
                if isinstance(chunk, AIMessageChunk):
                    # 处理 content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        yield ev.data
                    
                    # 处理 tool_calls
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(
                            chunk.tool_calls, emitter, tracker
                        ):
                            yield ev.data
                
                # 处理 ToolMessage (工具执行结果)
                # ToolMessage 是完整的，不是 Chunk，但它是当前步骤产生的，不能过滤
                # 如何区分历史 ToolMessage 和当前 ToolMessage？
                # LangGraph stream 会 yield 所有新产生的消息。
                # 只要我们确信 LangGraph 不会重放历史消息即可。
                # 如果它重放了，那我们就得通过 ID 或时间戳过滤。
                # 但通常 stream_mode="messages" 只 yield 新消息。
                # 为何用户会看到历史？
                # 也许是因为 chunk.content 包含了累积的历史？
                # 不，chunk.content 是 delta。
                # 除非... 某个模型把历史作为 context 一起输出了？（不太可能）
                
                # 还有一个可能：HumanMessage 也被 yield 了？
                # 我们只处理 AIMessageChunk 和 ToolMessage。
                # 如果 yield 了 HumanMessage (user input)，我们忽略它。
                elif isinstance(chunk, ToolMessage): # hasattr(chunk, "type") and chunk.type == "tool":
                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        yield ev.data
            
            if debug:
                print("[DEBUG] Stream completed normally")
            
        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            yield emitter.error(str(e)).data
            raise
        
        # 发送完成事件
        yield emitter.done(full_response).data

    def _process_text_chunk_with_tags(self, text: str, emitter: StreamEventEmitter):
        """
        处理可能包含 <thinking> 或 <reasoning_content> 标签的文本流
        """
        if not text:
            return

        # 状态机处理流式标签
        # 扩展支持的标签列表
        START_TAGS = ["<thinking>", "<reasoning_content>", "<thought>"]
        END_TAGS = ["</thinking>", "</reasoning_content>", "</thought>"]
        MAX_TAG_LEN = max(len(t) for t in START_TAGS + END_TAGS)
        
        # 将新文本追加到缓冲区
        self._tag_buffer += text
        
        # 既然用户已经统一了 thinking 和 response，不需要复杂的缓冲区保留逻辑来防止"部分标签"输出
        # 因为所有输出都是 thinking。
        # 但为了保持标签本身的隐藏（不打印标签字符串），我们仍然需要解析。
        # 只是不需要那么激进地保留内容。
        
        # 之前的逻辑是：如果没找到完整标签，且缓冲区末尾可能是标签前缀，就保留不输出。
        # 这就是所谓的"延迟输出机制"。
        # 用户反馈"还没输出完就卡到调用工具的步骤了"，可能是因为缓冲区里的内容一直没机会输出（因为没有后续文本来冲刷它）。
        # 如果这是最后一段文本，缓冲区里的内容就会丢失。
        
        # 修复方案：
        # 1. 仍然进行标签解析，但如果长时间没有匹配到标签，应该设定超时或强制冲刷？
        # 2. 或者，在 stream 结束时（done事件前），显式冲刷缓冲区。
        # 3. 简化逻辑：只匹配完整的标签。对于不完整的，如果长度超过 MAX_TAG_LEN，就应该输出了。
        
        # 现在的逻辑已经包含 safe_len 判断：
        # safe_len = len(content_to_process) - (MAX_TAG_LEN - 1)
        # 只有最后几个字符会被保留。
        
        # 如果用户感觉"卡住"，可能是因为最后几个字符（例如 "Okay, I will"）长度不够，被保留了。
        # 而紧接着就是工具调用，没有更多文本了。
        # 此时缓冲区里还留着 "Okay, I will"。
        
        # 我们需要在 _process_chunk_content 结束时，或者在检测到工具调用时，强制冲刷缓冲区吗？
        # 但 _process_text_chunk_with_tags 是生成器，很难从外部干预。
        
        # 鉴于用户说"不需要这个人为延迟的机制了"，我们可以放宽保留策略，或者完全移除标签解析？
        # 不，标签还是要解析的（为了不显示标签本身）。
        # 但我们可以尝试更积极地输出。
        
        while True:
            # 始终使用缓冲区内容进行处理
            content_to_process = self._tag_buffer
            
            if not self._in_thinking_tag:
                # 寻找任一启动标签
                found_tag = None
                earliest_idx = -1
                
                for tag in START_TAGS:
                    idx = content_to_process.find(tag)
                    if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
                        earliest_idx = idx
                        found_tag = tag
                
                if found_tag:
                    # 找到了完整标签
                    # 输出标签前的内容为 text (现在统一为 thinking)
                    pre_text = content_to_process[:earliest_idx]
                    if pre_text:
                        yield emitter.text(pre_text)
                    
                    self._in_thinking_tag = True
                    # 记录当前匹配的结束标签（用于闭合）
                    tag_idx = START_TAGS.index(found_tag)
                    self._current_end_tag = END_TAGS[tag_idx]
                    
                    # 消耗缓冲区：移除前缀文本和标签，继续循环
                    self._tag_buffer = content_to_process[earliest_idx + len(found_tag):]
                    continue
                else:
                    # 没找到完整标签
                    # 保留末尾可能的部分标签（以防跨 chunk）
                    # 既然用户抱怨卡顿，我们尽量减少保留。
                    # 只有当末尾确实像标签前缀时才保留？太复杂。
                    # 维持原有的 safe_len 逻辑，但在外部（检测到工具调用或流结束时）进行冲刷。
                    
                    # 这里我们做一个妥协：如果 content_to_process 长度超过 MAX_TAG_LEN，
                    # 肯定可以输出一部分。
                    
                    safe_len = len(content_to_process) - (MAX_TAG_LEN - 1)
                    
                    if safe_len > 0:
                        to_emit = content_to_process[:safe_len]
                        yield emitter.text(to_emit)
                        self._tag_buffer = content_to_process[safe_len:]
                    
                    # 退出循环，等待更多数据
                    break
            else:
                # 在 thinking 模式，寻找对应的结束标签
                end_tag = getattr(self, "_current_end_tag", "</thinking>")
                end_tag_idx = content_to_process.find(end_tag)
                
                if end_tag_idx != -1:
                    # 找到了结束标签
                    # 输出标签前的内容为 thinking
                    thinking_content = content_to_process[:end_tag_idx]
                    if thinking_content:
                        yield emitter.thinking(thinking_content)
                    
                    self._in_thinking_tag = False
                    
                    # 消耗缓冲区：移除 thinking 内容和结束标签，继续循环
                    self._tag_buffer = content_to_process[end_tag_idx + len(end_tag):]
                    continue
                else:
                    # 没找到结束标签
                    # 同样保留末尾
                    safe_len = len(content_to_process) - (MAX_TAG_LEN - 1)
                    
                    if safe_len > 0:
                        to_emit = content_to_process[:safe_len]
                        yield emitter.thinking(to_emit)
                        self._tag_buffer = content_to_process[safe_len:]
                    
                    # 退出循环，等待更多数据
                    break

    def _process_chunk_content(
        self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker
    ):
        """处理 chunk 的 content"""
        # ... (reasoning_content handling) ...
        reasoning_content = chunk.additional_kwargs.get("reasoning_content")
        if reasoning_content:
            yield emitter.thinking(reasoning_content)

        content = chunk.content
        
        # ... (tag parsing check) ...
        is_anthropic = (self.provider == "anthropic" or "claude" in self.model_name.lower())
        should_parse_tags = (
            is_anthropic 
            or "glm" in self.model_name.lower() 
            or "doubao" in self.model_name.lower() 
            or "kimi" in self.model_name.lower()
        )

        if isinstance(content, str):
            if content:
                # 始终使用标签解析器处理文本，以确保统一的缓冲区管理
                # 即使不需要 thinking，我们也需要通过这个管道来避免缓冲区滞留（或者我们显式处理）
                # 但如果 disable thinking，我们就不解析标签？不，现在统一了。
                if self.enable_thinking and should_parse_tags:
                    yield from self._process_text_chunk_with_tags(content, emitter)
                else:
                    # 如果不解析标签，直接输出 thinking (Agent Output)
                    yield emitter.text(content)
            
            # 检查是否有工具调用
            # 如果有工具调用，这通常意味着文本流的结束。
            # 我们应该尝试冲刷缓冲区中的残留文本。
            # 但 _process_text_chunk_with_tags 是生成器，无法直接访问其内部状态。
            # 不过我们可以直接访问 self._tag_buffer。
            
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                # 强制冲刷缓冲区（如果有残留）
                if self._tag_buffer:
                    # 只有当缓冲区内容看起来不像是标签的一部分时才输出？
                    # 或者干脆全部输出，因为都要调用工具了，不可能再有标签闭合了。
                    # 除非标签跨越了 tool call（这在 OpenAI 格式中不可能，在 Anthropic 中也不常见）。
                    yield emitter.text(self._tag_buffer)
                    self._tag_buffer = ""
                
                for tc_chunk in chunk.tool_call_chunks:
                    # ... (tool call processing) ...
                    # tc_chunk 可能是 dict 或对象
                    if hasattr(tc_chunk, "dict"):
                        tc_data = tc_chunk.dict()
                    else:
                        tc_data = tc_chunk
                    
                    tool_id = tc_data.get("id")
                    name = tc_data.get("name")
                    args = tc_data.get("args")
                    index = tc_data.get("index")
                    
                    if tool_id:
                        tracker.update(tool_id, name=name)
                    
                    if args:
                        if tracker.append_json_delta(args, index):
                             # 如果成功解析了参数，立即发送更新
                             info = tracker.get(tracker._last_tool_id)
                             if info:
                                 yield emitter.tool_call(info.name, info.args, info.id)
            return

        # ... (blocks processing) ...
        # Anthropic Blocks 处理逻辑中也应该考虑冲刷
        blocks = None
        if hasattr(chunk, "content_blocks"):
            try:
                blocks = chunk.content_blocks
            except Exception:
                blocks = None

        if blocks is None:
            # ... (tool_call_chunks fallback logic - same as above) ...
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                # 同样冲刷
                if self._tag_buffer:
                    yield emitter.text(self._tag_buffer)
                    self._tag_buffer = ""
                    
                for tc_chunk in chunk.tool_call_chunks:
                     # ... (copy paste logic) ...
                     if hasattr(tc_chunk, "dict"):
                        tc_data = tc_chunk.dict()
                     else:
                        tc_data = tc_chunk
                    
                     tool_id = tc_data.get("id")
                     name = tc_data.get("name")
                     args = tc_data.get("args")
                     index = tc_data.get("index")
                    
                     if tool_id:
                        tracker.update(tool_id, name=name)
                    
                     if args:
                        if tracker.append_json_delta(args, index):
                             info = tracker.get(tracker._last_tool_id)
                             if info:
                                 yield emitter.tool_call(info.name, info.args, info.id)
                return

            if isinstance(content, dict):
                blocks = [content]
            elif isinstance(content, list):
                blocks = content
            else:
                return

        # ... (rest of function) ...

        # 检查这个 chunk 是否包含 thinking 块 (Native Extended Thinking)
        has_thinking_block = any(
            (b.get("type") in ("thinking", "reasoning")) 
            for b in blocks 
            if isinstance(b, dict)
        )
        
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
                    # 如果启用了 thinking 且是支持 tags 的模型，优先尝试解析 tags
                    if self.enable_thinking and should_parse_tags:
                         yield from self._process_text_chunk_with_tags(text, emitter)
                    else:
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
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                partial_json = block.get("partial_json", "")
                if partial_json:
                    if tracker.append_json_delta(partial_json, block.get("index", 0)):
                         # 如果成功解析了参数，立即发送更新
                         info = tracker.get(tracker._last_tool_id)
                         if info:
                             yield emitter.tool_call(info.name, info.args, info.id)

            elif block_type == "tool_call_chunk":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id:
                    tracker.update(tool_id, name=name)
                partial_args = block.get("args", "")
                if isinstance(partial_args, str) and partial_args:
                    if tracker.append_json_delta(partial_args, block.get("index", 0)):
                         # 如果成功解析了参数，立即发送更新
                         info = tracker.get(tracker._last_tool_id)
                         if info:
                             yield emitter.tool_call(info.name, info.args, info.id)

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
