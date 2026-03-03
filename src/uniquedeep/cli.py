#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/cli.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: LangChain Skills Agent 的命令行接口，支持交互模式和流式输出。
'''

"""
LangChain Skills Agent CLI

命令行入口，提供演示和交互功能：
- 列出发现的 Skills
- 显示 system prompt（演示 Level 1）
- 执行用户请求（支持流式输出和 thinking 显示）
- 交互式对话模式

流式输出特性（Claude Code 风格）：
- 🧠 Thinking 面板：实时显示模型思考过程（蓝色）
- ● Tool Calls：紧凑格式显示，如 Bash(git status)
  - 绿色圆点 ● 表示成功
  - 黄色圆点 ● 表示执行中
  - 红色圆点 ● 表示失败
- 树形输出：└ 连接子内容，折叠长输出显示 ... +X lines
- 💬 Response 面板：逐字显示最终响应（绿色）
"""

import argparse
import json
import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.layout import Layout
from rich.syntax import Syntax
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .agent import LangChainSkillsAgent, check_api_credentials
from .tools import bing_mcp_client
from .skill_loader import SkillLoader
from .stream import (
    ToolResultFormatter,
    has_args,
    DisplayLimits,
    ToolStatus,
    format_tool_compact,
    format_tree_output,
    count_lines,
    truncate_with_line_hint,
    is_success,
)


# 加载环境变量（override=True 确保 .env 文件覆盖系统环境变量）
load_dotenv(override=True)

# Rich Console 配置：支持 Windows 和 NO_COLOR 环境变量
console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

# 全局工具结果格式化器
formatter = ToolResultFormatter()


# === 流式处理状态 ===


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
                    # 注意：并行工具调用时，我们不希望把前一个正在 running 的工具标记为 completed
                    # 只有非工具事件才需要标记
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


def display_final_results(
    state: StreamState,
    thinking_max_length: int = DisplayLimits.THINKING_FINAL,
    tool_result_max_length: int = DisplayLimits.TOOL_RESULT_FINAL,
    args_max_length: int = DisplayLimits.ARGS_FORMATTED,
    show_thinking: bool = True,
    show_tools: bool = True,
    show_response_panel: bool = True,
):
    """
    显示最终结果（非流式）

    Args:
        state: 流式处理状态
        thinking_max_length: thinking 最大显示长度
        tool_result_max_length: 工具结果最大显示长度
        args_max_length: 参数最大显示长度
        show_thinking: 是否显示 thinking
        show_response_panel: 是否用 Panel 显示响应
    """
    thinking_count = 0
    
    # 遍历事件列表，按顺序生成显示元素
    for i, event in enumerate(state.events):
        event_type = event.get("type")
        
        if event_type == "thinking" and show_thinking:
            thinking_count += 1
            content = event.get("content", "")
            
            title = f"🧠 Thinking (Round {thinking_count})"
            display_content = content
            if len(display_content) > thinking_max_length:
                half = thinking_max_length // 2
                display_content = (
                    display_content[:half]
                    + "\n\n... (truncated) ...\n\n"
                    + display_content[-half:]
                )
            
            console.print(
                Panel(
                    Text(display_content, style="dim"),
                    title=title,
                    border_style="blue",
                )
            )

        elif event_type == "tool" and show_tools:
            data = event.get("data", {})
            status = data.get("status", "running")
            result = data.get("result")
            
            # 确定状态和颜色
            if status == "done" and result:
                content = result.get("content", "")
                if is_success(content):
                    status_enum = ToolStatus.SUCCESS
                    style = "bold green"
                else:
                    status_enum = ToolStatus.ERROR
                    style = "bold red"
            else:
                status_enum = ToolStatus.RUNNING
                style = "bold yellow"

            # 紧凑格式显示工具调用
            tool_compact = format_tool_compact(data['name'], data.get('args'))
            tool_text = Text()
            tool_text.append(f"{status_enum.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            console.print(tool_text)

            # 显示对应的结果
            if status == "done" and result:
                # 已有结果，显示树形输出
                result_elements = format_tool_result(
                    result['name'],
                    result.get('content', ''),
                    max_length=tool_result_max_length,
                    compact=True,
                )
                for elem in result_elements:
                    console.print(elem)
            console.print() # 空行分隔

        elif event_type == "response":
            content = event.get("content", "")
            if show_response_panel:
                console.print(
                    Panel(
                        Markdown(content),
                        title="💬 Response",
                        border_style="green",
                    )
                )
            else:
                console.print(f"\n[bold blue]Assistant:[/bold blue]")
                console.print(Markdown(content))
                console.print()


def format_tool_result(
    name: str, content: str, max_length: int = 800, compact: bool = False
) -> list:
    """
    智能格式化工具结果

    Args:
        name: 工具名称
        content: 工具输出内容
        max_length: 最大显示长度
        compact: 是否使用紧凑的树形格式（Claude Code 风格）

    Returns:
        Rich 可渲染元素列表
    """
    if compact:
        # Claude Code 风格：树形输出
        return format_tool_result_compact(name, content, max_lines=10)
    else:
        # 原有格式
        result = formatter.format(name, content, max_length)
        return result.elements


def format_tool_result_compact(name: str, content: str, max_lines: int = 5) -> list:
    """
    使用 Claude Code 风格格式化工具结果（树形输出）

    Args:
        name: 工具名称
        content: 工具输出内容
        max_lines: 最大显示行数

    Returns:
        Rich 可渲染元素列表
    """
    elements = []

    # load_skill 工具：只显示简短的成功消息
    if name.lower() == "load_skill":
        if is_success(content):
            elements.append(Text("  └ Successfully loaded skill", style="dim"))
        else:
            # 失败时显示错误内容
            elements.append(Text(f"  └ {content.strip()[:60]}", style="red dim"))
        return elements

    if not content.strip():
        elements.append(Text("  └ (empty)", style="dim"))
        return elements

    lines = content.strip().split("\n")
    total_lines = len(lines)

    # 显示前几行
    display_lines = lines[:max_lines]
    for i, line in enumerate(display_lines):
        prefix = "└" if i == 0 else " "
        # 截断过长的行
        if len(line) > 80:
            line = line[:77] + "..."
        style = "dim" if is_success(content) else "red dim"
        elements.append(Text(f"  {prefix} {line}", style=style))

    # 折叠提示
    remaining = total_lines - max_lines
    if remaining > 0:
        elements.append(Text(f"    ... +{remaining} lines", style="dim italic"))

    return elements


def format_tool_args(args: dict, max_length: int = 300) -> list:
    """
    格式化工具参数显示

    Args:
        args: 工具参数字典
        max_length: 最大显示长度

    Returns:
        Rich 可渲染元素列表
    """
    elements = []
    try:
        args_formatted = json.dumps(args, indent=2, ensure_ascii=False)
        if len(args_formatted) > max_length:
            args_formatted = args_formatted[:max_length] + "\n..."
        elements.append(
            Syntax(args_formatted, "json", theme="monokai", line_numbers=False)
        )
    except (TypeError, ValueError):
        args_str = str(args)
        if len(args_str) > max_length:
            args_str = args_str[:max_length] + "..."
        elements.append(Text(f"   {args_str}", style="dim"))
    return elements


def render_event_static(event: dict, thinking_count: int = 0) -> Group:
    """渲染静态（已完成）的事件"""
    elements = []
    event_type = event.get("type")

    if event_type == "thinking":
        content = event.get("content", "")
        title = f"🧠 Thinking (Round {thinking_count})"
        elements.append(
            Panel(
                Text(content, style="dim"),
                title=title,
                border_style="blue dim",
                padding=(0, 1),
            )
        )

    elif event_type == "tool":
        data = event.get("data", {})
        status = data.get("status", "done") # 默认 done
        result = data.get("result")
        
        # 紧凑格式显示工具调用
        status_enum = ToolStatus.SUCCESS if (result and is_success(result.get("content", ""))) else ToolStatus.ERROR
        style = "bold green" if status_enum == ToolStatus.SUCCESS else "bold red"

        tool_compact = format_tool_compact(data['name'], data.get('args'))
        tool_text = Text()
        tool_text.append(f"{status_enum.value} ", style=style)
        tool_text.append(tool_compact, style=style)
        elements.append(tool_text)

        if result:
            result_elements = format_tool_result(
                result['name'],
                result.get('content', ''),
                compact=True,
            )
            elements.extend(result_elements)
        elements.append(Text("")) # 空行

    elif event_type == "response":
        content = event.get("content", "")
        elements.append(
            Panel(
                Markdown(content),
                title="💬 Response",
                border_style="green",
                padding=(0, 1),
            )
        )

    return Group(*elements)

def create_streaming_display(
    events: list = None,
    is_waiting: bool = False,
    is_processing: bool = False,
    start_thinking_count: int = 1,
) -> Group:
    """
    创建流式显示的布局（仅显示活动事件）

    Args:
        events: 活动事件列表
        is_waiting: 是否处于初始等待状态
        is_processing: 工具执行后等待 AI 继续处理
        start_thinking_count: thinking 计数起始值

    Returns:
        Rich Group 对象
    """
    elements = []
    events = events or []

    # 初始等待状态 - 显示 spinner 提示
    if is_waiting and not events:
        spinner = Spinner("dots", text=" AI 正在思考中...", style="cyan")
        elements.append(spinner)
        return Group(*elements)
        
    thinking_count = start_thinking_count - 1

    for i, event in enumerate(events):
        event_type = event.get("type")
        
        if event_type == "thinking":
            thinking_count += 1
            content = event.get("content", "")
            title = f"🧠 Thinking (Round {thinking_count}) ..."
            
            # 流式显示时，为了性能和视觉，可以截断过长的内容（仅展示尾部）
            # 用户要求不要限制，但为了流式体验，尾部滚动是合理的
            # 只要静态打印时是完整的即可
            display_content = content
            if len(display_content) > DisplayLimits.THINKING_STREAM:
                display_content = "..." + display_content[-DisplayLimits.THINKING_STREAM:]
            
            elements.append(
                Panel(
                    Text(display_content, style="dim"),
                    title=title,
                    border_style="blue",
                    padding=(0, 1),
                )
            )

        elif event_type == "tool":
            data = event.get("data", {})
            status = data.get("status", "running")
            result = data.get("result")
            
            if status == "done" and result:
                content = result.get("content", "")
                if is_success(content):
                    status_enum = ToolStatus.SUCCESS
                    style = "bold green"
                else:
                    status_enum = ToolStatus.ERROR
                    style = "bold red"
            else:
                status_enum = ToolStatus.RUNNING
                style = "bold yellow"

            tool_compact = format_tool_compact(data['name'], data.get('args'))
            tool_text = Text()
            tool_text.append(f"{status_enum.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            elements.append(tool_text)

            if status == "done" and result:
                result_elements = format_tool_result(
                    result['name'],
                    result.get('content', ''),
                    compact=True,
                )
                elements.extend(result_elements)
            else:
                spinner = Spinner("dots", text=" 执行中...", style="yellow")
                elements.append(spinner)

        elif event_type == "response":
            content = event.get("content", "")
            title = "💬 Response ..."
            elements.append(
                Panel(
                    Markdown(content),
                    title=title,
                    border_style="green",
                    padding=(0, 1),
                )
            )

    if is_processing:
        if not events or events[-1]["type"] != "thinking":
            spinner = Spinner("dots", text=" AI 正在分析结果...", style="cyan")
            elements.append(spinner)
            
    if not is_processing and not elements:
         elements.append(Text("⏳ Processing...", style="dim"))

    return Group(*elements)


def print_banner():
    """打印欢迎横幅"""
    banner = """
[bold cyan]UniqueDepp[/bold cyan] 一个可实现Skills渐进式披露机制的Agent

原理：
[yellow]Level 1[/yellow]: 启动时 → Skills 元数据注入 system prompt
[yellow]Level 2[/yellow]: 请求匹配时 → load_skill 加载详细指令
[yellow]Level 3[/yellow]: 执行时 → bash 运行脚本，仅输出进入上下文
"""
    console.print(Panel(banner, title="UniqueDepp", border_style="cyan"))


def cmd_list_skills():
    """列出发现的 Skills"""
    console.print("\n[bold cyan]Discovering Skills...[/bold cyan]\n")

    loader = SkillLoader()
    skills = loader.scan_skills()

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        console.print("Skills are loaded from:")
        console.print("  - ~/.claude/skills/")
        console.print("  - .claude/skills/")
        console.print("  - ~/.agents/skills/")
        console.print("  - .agents/skills/")
        return

    table = Table(title=f"Found {len(skills)} Skills")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    table.add_column("Path", style="dim")

    for skill in skills:
        # 截断描述
        desc = skill.description
        if len(desc) > 60:
            desc = desc[:57] + "..."

        table.add_row(
            skill.name,
            desc,
            str(skill.skill_path.relative_to(skill.skill_path.parent.parent)),
        )

    console.print(table)


def cmd_show_prompt():
    """显示 system prompt（演示 Level 1）"""
    console.print("\n[bold cyan]Building System Prompt (Level 1)...[/bold cyan]\n")

    agent = LangChainSkillsAgent()
    prompt = agent.get_system_prompt()

    console.print(
        Panel(
            Markdown(prompt),
            title="System Prompt",
            subtitle="Skills metadata injected here",
            border_style="green",
        )
    )

    # 统计信息
    skills = agent.get_discovered_skills()
    token_estimate = len(prompt) // 4  # 粗略估算

    console.print(f"\n[dim]Skills discovered: {len(skills)}[/dim]")
    console.print(f"[dim]Estimated tokens: ~{token_estimate}[/dim]")


def cmd_run(prompt: str, enable_thinking: bool = True):
    """
    执行单次请求，支持流式输出和 thinking 显示

    Args:
        prompt: 用户请求
        enable_thinking: 是否启用 thinking 显示
    """
    console.print(Panel(f"[bold cyan]User Request:[/bold cyan]\n{prompt}"))
    console.print()

    # 检查 API 认证（支持 ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN 或 DEEPSEEK_API_KEY）
    if not check_api_credentials():
        console.print("[red]Error: API credentials not set[/red]")
        console.print("Please set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY in .env file")
        sys.exit(1)

    agent = LangChainSkillsAgent(enable_thinking=enable_thinking)

    console.print("[dim]Running agent with streaming output...[/dim]\n")

    try:
        state = StreamState()
        printed_count = 0
        thinking_round = 0

        with Live(console=console, refresh_per_second=10, transient=True) as live:
            # 立即显示等待状态
            live.update(create_streaming_display(is_waiting=True))

            for event in agent.stream_events(prompt):
                event_type = state.handle_event(event)
                
                # 检查是否有已完成的事件需要归档打印
                while printed_count < len(state.events):
                    evt = state.events[printed_count]
                    if evt.get("is_completed"):
                        if evt["type"] == "thinking":
                            thinking_round += 1
                        
                        # 在 Live 上方打印归档内容
                        # 为了避免 Live 清除不彻底，我们先清除 Live，打印，再恢复
                        # 但 rich.Live 默认就是在 update 时重绘，print 会在上方插入
                        console.print(render_event_static(evt, thinking_round))
                        printed_count += 1
                    else:
                        break

                # 更新 Live 显示（只显示未归档的活动事件）
                active_events = state.events[printed_count:]
                live.update(create_streaming_display(
                    events=active_events,
                    is_processing=state.is_processing,
                    start_thinking_count=thinking_round + 1
                ))

                if event_type in ("tool_call", "tool_result"):
                    live.refresh()
        
        # 打印剩余的事件
        while printed_count < len(state.events):
            evt = state.events[printed_count]
            if evt["type"] == "thinking":
                thinking_round += 1
            console.print(render_event_static(evt, thinking_round))
            printed_count += 1
            
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


def cmd_interactive(enable_thinking: bool = True):
    """
    交互式对话模式，支持流式输出和 thinking 显示

    Args:
        enable_thinking: 是否启用 thinking 显示
    """
    print_banner()

    # 检查 API 认证（支持 ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN 或 DEEPSEEK_API_KEY）
    if not check_api_credentials():
        console.print("[red]Error: API credentials not set[/red]")
        console.print("Please set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY in .env file")
        sys.exit(1)

    agent = LangChainSkillsAgent(enable_thinking=enable_thinking)

    # 显示发现的 Skills
    skills = agent.get_discovered_skills()
    console.print(f"\n[green]✓[/green] Discovered {len(skills)} skills")
    for skill in skills:
        console.print(f"  - {skill['name']}")
    
    # 检查 MCP 状态
    console.print("\n[dim]Checking MCP Status...[/dim]")
    try:
        async def check_mcp():
            try:
                return await bing_mcp_client.list_tools()
            finally:
                await bing_mcp_client.close()
            
        mcp_tools = asyncio.run(check_mcp())
        if mcp_tools:
            console.print(f"[green]✓[/green] MCP Connected: bing-cn-mcp ({len(mcp_tools)} tools)")
            for tool in mcp_tools:
                 # tool might be an object or dict
                 if hasattr(tool, 'name'):
                     name = tool.name
                 else:
                     name = tool.get('name', 'unknown')
                 console.print(f"  - {name} (MCP)")
        else:
            console.print("[yellow]! MCP Connected but no tools found[/yellow]")
    except Exception as e:
        console.print(f"[yellow]! MCP Connection Failed: {str(e)}[/yellow]")
        console.print("[dim]  Ensure 'bing-cn-mcp' is installed via npm and npx is in PATH.[/dim]")
        console.print("[dim]  Note: Some MCP servers require Node.js >= 20.[/dim]")
    
    console.print()

    thinking_status = (
        "[green]enabled[/green]" if enable_thinking else "[dim]disabled[/dim]"
    )
    console.print(f"[dim]Extended Thinking: {thinking_status}[/dim]")
    console.print(
        "[dim]Commands: /exit to quit, /skills to list skills, /prompt to show system prompt, /temp [val] to set temperature, /model <name> [provider] to switch model[/dim]\n"
    )

    thread_id = "interactive"

    # 初始化 prompt_toolkit session（跨平台兼容路径）
    # history_file = str(Path.home() / ".uniquedeep_history")
    session = PromptSession(
        # history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    while True:
        try:
            # 使用 prompt_toolkit 替代 console.input，支持中文删除和历史记录
            user_input = session.prompt(
                HTML('<ansigreen><b>You:</b></ansigreen> ')
            ).strip()

            if not user_input:
                continue

            # 特殊命令
            # 退出交互式窗口
            if user_input.lower() in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            # 查看skills列表
            if user_input.lower() == "/skills":
                cmd_list_skills()
                continue

            # 显示系统级提示词
            if user_input.lower() == "/prompt":
                cmd_show_prompt()
                continue

            # 显示当前温度，或设置温度
            if user_input.lower().startswith("/temp"):
                try:
                    parts = user_input.split()
                    if len(parts) == 2:
                        temp_val = float(parts[1])
                        if 0.0 <= temp_val <= 1.0:
                            if agent.set_temperature(temp_val):
                                console.print(
                                    f"[green]✓ Temperature set to {temp_val}[/green]"
                                )
                            else:
                                console.print(
                                    "[yellow]! Cannot change temperature: Extended Thinking is enabled (requires temperature=1.0)[/yellow]"
                                )
                        else:
                            console.print(
                                "[red]! Temperature must be between 0.0 and 1.0[/red]"
                            )
                    else:
                        console.print(
                            f"[dim]Current temperature: {agent.temperature}[/dim]"
                        )
                except ValueError:
                    console.print("[red]! Invalid temperature value[/red]")
                continue

            # 切换模型
            if user_input.lower().startswith("/model"):
                try:
                    parts = user_input.split()
                    if len(parts) >= 2:
                        new_model = parts[1]
                        provider = parts[2] if len(parts) > 2 else None
                        
                        console.print(f"[dim]Switching model to {new_model}...[/dim]")
                        if agent.switch_model(new_model, provider, thread_id=thread_id):
                            console.print(f"[green]✓ Switched to model: {agent.model_name} ({agent.provider})[/green]")
                            if agent.provider == "anthropic" and agent.enable_thinking:
                                console.print("[dim]Extended Thinking enabled[/dim]")
                            elif agent.provider != "anthropic" and not agent.enable_thinking:
                                console.print("[dim]Extended Thinking disabled (not supported)[/dim]")
                        else:
                            console.print("[red]! Failed to switch model[/red]")
                    else:
                        console.print(f"[dim]Current model: {agent.model_name} ({agent.provider})[/dim]")
                        console.print("[dim]Usage: /model <model_name> [provider][/dim]")
                except Exception as e:
                    console.print(f"[red]! Error switching model: {e}[/red]")
                continue

            # 运行 agent（流式输出）
            console.print()

            state = StreamState()
            printed_count = 0
            thinking_round = 0

            with Live(console=console, refresh_per_second=10, transient=True) as live:
                # 立即显示等待状态
                live.update(create_streaming_display(is_waiting=True))

                for event in agent.stream_events(user_input, thread_id=thread_id):
                    event_type = state.handle_event(event)

                    # 检查是否有已完成的事件需要归档打印
                    while printed_count < len(state.events):
                        evt = state.events[printed_count]
                        if evt.get("is_completed"):
                            if evt["type"] == "thinking":
                                thinking_round += 1
                            
                            console.print(render_event_static(evt, thinking_round))
                            printed_count += 1
                        else:
                            break

                    # 更新 Live 显示（只显示未归档的活动事件）
                    active_events = state.events[printed_count:]
                    live.update(create_streaming_display(
                        events=active_events,
                        is_processing=state.is_processing,
                        start_thinking_count=thinking_round + 1
                    ))

                    if event_type in ("tool_call", "tool_result"):
                        live.refresh()

            # 打印剩余的事件
            while printed_count < len(state.events):
                evt = state.events[printed_count]
                if evt["type"] == "thinking":
                    thinking_round += 1
                console.print(render_event_static(evt, thinking_round))
                printed_count += 1

            # 交互模式不需要最后的 Panel
            # display_final_results( ... ) 已经不需要了

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description="LangChain Skills Agent - 演示 Skills 三层加载机制（支持流式输出和 Extended Thinking）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 列出发现的 Skills
  %(prog)s --list-skills

  # 显示 system prompt（演示 Level 1）
  %(prog)s --show-prompt

  # 执行请求（默认启用 thinking）
  %(prog)s "搜索三篇关于GJB2基因研究的最新论文，并列出基本信息"

  # 执行请求（禁用 thinking）
  %(prog)s --no-thinking "列出当前目录的文件"

  # 交互式模式
  %(prog)s --interactive

Features:
  - 🧠 Extended Thinking: 显示模型的思考过程（蓝色面板）
  - 🔧 Tool Calls: 显示工具调用（黄色）
  - 💬 Streaming Response: 逐字显示响应（绿色面板）
""",
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="要执行的请求",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="进入交互式对话模式",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="列出发现的 Skills",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="显示 system prompt（演示 Level 1）",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="禁用 Extended Thinking（可降低延迟和成本）",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        help="设置工作目录",
    )

    args = parser.parse_args()

    # 设置工作目录
    if args.cwd:
        os.chdir(args.cwd)

    # thinking 开关
    enable_thinking = not args.no_thinking

    # 执行命令
    if args.list_skills:
        cmd_list_skills()
    elif args.show_prompt:
        cmd_show_prompt()
    elif args.interactive:
        cmd_interactive(enable_thinking=enable_thinking)
    elif args.prompt:
        cmd_run(args.prompt, enable_thinking=enable_thinking)
    else:
        # 默认进入交互模式
        cmd_interactive(enable_thinking=enable_thinking)


if __name__ == "__main__":
    main()
