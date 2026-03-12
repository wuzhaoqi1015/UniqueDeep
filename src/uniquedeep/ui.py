# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/ui.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: UI rendering components using Rich.
'''

import json
import os
import sys
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from rich.spinner import Spinner
from rich.syntax import Syntax

from .stream import (
    ToolResultFormatter,
    DisplayLimits,
    ToolStatus,
    format_tool_compact,
    is_success,
)

# Rich Console 配置：支持 Windows 和 NO_COLOR 环境变量
console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

# 全局工具结果格式化器
formatter = ToolResultFormatter()


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
        title = f"🧠 Agent"
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
        if content.strip():
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
            title = f"🧠 Agent"
            
            # 流式显示时，为了性能和视觉，可以截断过长的内容（仅展示尾部）
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
            title = "💬 Response"
            # 只有当内容不为空，且看起来不是纯粹的思考过程（已经被合并到 thinking 了）时才显示
            if content.strip():
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


def render_skills_list(skills: list):
    """渲染 Skills 列表"""
    console.print("\n[bold cyan]Discovering Skills...[/bold cyan]\n")

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


def render_system_prompt(prompt: str, skill_count: int, token_estimate: int):
    """渲染 system prompt"""
    console.print("\n[bold cyan]Building System Prompt (Level 1)...[/bold cyan]\n")

    console.print(
        Panel(
            Markdown(prompt),
            title="System Prompt",
            subtitle="Skills metadata injected here",
            border_style="green",
        )
    )

    console.print(f"\n[dim]Skills discovered: {skill_count}[/dim]")
    console.print(f"[dim]Estimated tokens: ~{token_estimate}[/dim]")


def render_models_list(models: list, active_model: str, default_temp: float, default_max_tokens: int):
    """渲染模型列表"""
    console.print("\n[bold cyan]Common Models:[/bold cyan]\n")
    
    if not models:
        console.print("[yellow]No models found in models.json[/yellow]")
        return

    table = Table(title="Available Models (from models.json)")
    table.add_column("Current", style="green", width=3)
    table.add_column("Provider", style="cyan")
    table.add_column("Model Name", style="green")
    table.add_column("Description", style="white")
    table.add_column("Thinking", style="dim")
    table.add_column("Temp", style="yellow")
    table.add_column("Max Tokens", style="magenta")

    for model in models:
        is_active = "*" if model["name"] == active_model else ""
        
        # Determine temperature display
        temp_val = model.get("temperature")
        if temp_val is not None:
            temp_str = str(temp_val)
        else:
            temp_str = f"{default_temp} (def)"
            
        # Determine max_tokens display
        tokens_val = model.get("max_tokens")
        if tokens_val is not None:
            tokens_str = str(tokens_val)
        else:
            tokens_str = f"{default_max_tokens} (def)"
            
        table.add_row(
            is_active,
            model.get("provider_display", "Unknown"),
            model.get("name", "Unknown"),
            model.get("description", ""),
            "Yes" if model.get("thinking") else "No",
            temp_str,
            tokens_str
        )

    console.print(table)
    console.print("\n[dim]Use /model <name> to switch. Edit models.json to add more.[/dim]")

def display_final_results(
    state, # StreamState
    thinking_max_length: int = DisplayLimits.THINKING_FINAL,
    tool_result_max_length: int = DisplayLimits.TOOL_RESULT_FINAL,
    args_max_length: int = DisplayLimits.ARGS_FORMATTED,
    show_thinking: bool = True,
    show_tools: bool = True,
    show_response_panel: bool = True,
):
    """
    显示最终结果（非流式）
    """
    # 遍历事件列表，按顺序生成显示元素
    for i, event in enumerate(state.events):
        event_type = event.get("type")
        
        if event_type == "thinking" and show_thinking:
            content = event.get("content", "")
            title = f"🧠 Agent"
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
            # Use logic similar to render_event_static
            render_event_static(event)

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
