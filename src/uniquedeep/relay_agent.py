#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/relay_agent.py
@Time: 2026/02/28
@Author: UniqueDeep
@Description: Relay Agent for multi-model collaboration (Planner -> Executor)
'''

import os
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from .agent import LangChainSkillsAgent, DEFAULT_THINKING_BUDGET
from .stream import StreamEventEmitter

class RelayAgent:
    """
    接力模式 Agent：由 Planner (DeepSeek) 和 Executor (Claude) 协作完成任务
    """
    
    def __init__(
        self,
        planner_model: str = "deepseek-reasoner",
        planner_provider: Optional[str] = "deepseek",
        executor_model: str = "claude-3-7-sonnet-20250219",
        executor_provider: Optional[str] = "anthropic",
        skill_paths: Optional[list[Path]] = None,
        working_directory: Optional[Path] = None,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
        enable_thinking: bool = True,
    ):
        self.planner_model = planner_model
        self.planner_provider = planner_provider
        self.executor_model = executor_model
        self.executor_provider = executor_provider
        self.skill_paths = skill_paths
        self.working_directory = working_directory or Path.cwd()
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking
        
        # 初始化 Planner Agent (DeepSeek)
        # 注意：Planner 不需要加载 Skills，只需生成计划
        self.planner = LangChainSkillsAgent(
            model=planner_model,
            provider=planner_provider, 
            skill_paths=skill_paths,  # 虽然不需要执行，但需要知道有哪些 Skills 可用
            working_directory=working_directory,
            enable_thinking=True,     # DeepSeek Reasoner always has reasoning
            thinking_budget=thinking_budget
        )
        
        # 初始化 Executor Agent (Claude)
        self.executor = LangChainSkillsAgent(
            model=executor_model,
            provider=executor_provider,
            skill_paths=skill_paths,
            working_directory=working_directory,
            enable_thinking=enable_thinking,     # Configurable
            thinking_budget=thinking_budget
        )
        
    def stream_events(self, message: str, thread_id: str = "relay-default") -> Iterator[dict]:
        """
        执行接力流程：
        1. Planner 分析并制定计划
        2. Executor 根据计划执行任务
        """
        emitter = StreamEventEmitter()
        
        # === Stage 1: Planning ===
        yield {"type": "stage_start", "stage": "planning", "model": self.planner_model}
        
        plan_content = ""
        
        # 构建 Planner 的 Prompt
        planner_prompt = f"""You are a master planner. Your goal is to analyze the user's request and create a detailed, step-by-step execution plan.
        
User Request: {message}

The executor has access to the following skills (tools):
{self._get_skills_description()}

Please output a clear plan. Do NOT execute any tools yourself. Just provide the text plan."""

        # 调用 Planner
        # 我们使用 invoke 而不是 stream_events，因为我们要完全捕获 plan 后再传给 executor
        # 但为了用户体验，我们还是流式输出给用户看
        try:
            for event in self.planner.stream_events(planner_prompt, thread_id=f"{thread_id}-planner"):
                # 过滤掉 tool_call 相关事件，因为 Planner 不应该调用工具
                if event["type"] in ("thinking", "text"):
                    yield event
                    if event["type"] == "text":
                        plan_content += event.get("content", "")
                elif event["type"] == "done":
                    # 获取完整响应（可能包含思考过程，我们主要关心 text 部分）
                    plan_content = event.get("response", "")
                    
            yield {"type": "stage_end", "stage": "planning"}
            
        except Exception as e:
            yield emitter.error(f"Planner failed: {str(e)}").data
            return

        if not plan_content:
            yield emitter.error("Planner produced no output.").data
            return

        # === Stage 2: Executing ===
        yield {"type": "stage_start", "stage": "executing", "model": self.executor_model}
        
        # 构建 Executor 的 Prompt
        executor_prompt = f"""You are an expert executor. You have received a plan from a planner agent.
        
Original User Request: {message}

Planner's Plan:
{plan_content}

Please execute this plan using your available tools. Report progress as you go."""

        # 调用 Executor
        try:
            for event in self.executor.stream_events(executor_prompt, thread_id=f"{thread_id}-executor"):
                yield event
                
            yield {"type": "stage_end", "stage": "executing"}
            
        except Exception as e:
            yield emitter.error(f"Executor failed: {str(e)}").data

    def _get_skills_description(self) -> str:
        """获取 Skills 描述供 Planner 参考"""
        skills = self.planner.get_discovered_skills()
        desc = []
        for s in skills:
            desc.append(f"- {s['name']}: {s['description']}")
        return "\n".join(desc)
