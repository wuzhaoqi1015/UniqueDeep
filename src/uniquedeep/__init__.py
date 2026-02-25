#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/__init__.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: LangChain Skills Agent 包初始化和导出。
'''

"""
LangChain Skills Agent

使用 LangChain 1.0 实现的 Skills Agent，演示三层加载机制的底层原理。

## 核心概念

Skills 是可重用的、基于文件系统的能力模块，通过三层加载机制实现：

- **Level 1 (启动时)**: 扫描 Skills 目录，将元数据注入 system_prompt
- **Level 2 (请求匹配时)**: load_skill 工具读取 SKILL.md 详细指令
- **Level 3 (执行时)**: bash 工具执行脚本，脚本代码不进入上下文

## 使用示例

```python
from uniquedeep import LangChainSkillsAgent

# 创建 agent
agent = LangChainSkillsAgent()

# 查看 system prompt（Level 1 演示）
print(agent.get_system_prompt())

# 查看发现的 Skills
for skill in agent.get_discovered_skills():
    print(f"- {skill['name']}: {skill['description']}")

# 运行 agent
result = agent.invoke("搜索关于GJB2基因的最新论文进展")
print(agent.get_last_response(result))
```

## CLI 使用

```bash
# 列出发现的 Skills
uv run uniquedeep --list-skills

# 显示 system prompt
uv run uniquedeep --show-prompt

# 执行请求
uv run uniquedeep "搜索关于GJB2基因的最新论文进展"

# 交互式模式
uv run uniquedeep --interactive
```

## LangChain 1.0 API 要点

### create_agent
```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-5-20250929",  # 模型
    tools=[load_skill, bash],             # 工具列表
    system_prompt="...",                  # 系统提示
    context_schema=MyContext,             # 上下文类型
    checkpointer=InMemorySaver(),         # 会话记忆
)
```

### @tool with ToolRuntime
```python
from langchain.tools import tool, ToolRuntime

@tool
def my_tool(arg: str, runtime: ToolRuntime[MyContext]) -> str:
    '''Tool description.'''
    # runtime.context 访问上下文
    # runtime.state 访问状态
    return result
```
"""

from .agent import LangChainSkillsAgent, create_skills_agent
from .skill_loader import (
    SkillLoader,
    SkillMetadata,
    SkillContent,
    discover_skills,
    get_skill_content,
)
from .tools import load_skill, bash, read_file, write_file, ALL_TOOLS, SkillAgentContext

__version__ = "0.1.0"

__all__ = [
    # Agent
    "LangChainSkillsAgent",
    "create_skills_agent",
    # Skill Loader
    "SkillLoader",
    "SkillMetadata",
    "SkillContent",
    "discover_skills",
    "get_skill_content",
    # Tools (注意：list_skills 已删除，skills 列表在 system prompt 中注入)
    "load_skill",
    "bash",
    "read_file",
    "write_file",
    "ALL_TOOLS",
    # Context
    "SkillAgentContext",
]
