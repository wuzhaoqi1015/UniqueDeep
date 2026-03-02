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
        self.thinking_text = ""
        self.response_text = ""
        self.tool_calls = []
        self.tool_results = []
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False  # 工具执行后等待 AI 继续处理

    def handle_event(self, event: dict) -> str:
        """
        处理单个流式事件，更新内部状态

        Args:
            event: 流式事件字典

        Returns:
            事件类型
        """
        event_type = event.get("type")

        if event_type == "thinking":
            self.is_thinking = True
            self.is_responding = False
            self.is_processing = False  # 收到新内容，不再是处理中
            self.thinking_text += event.get("content", "")

        elif event_type == "text":
            self.is_thinking = False
            self.is_responding = True
            self.is_processing = False  # 收到新内容，不再是处理中
            self.response_text += event.get("content", "")

        elif event_type == "tool_call":
            self.is_thinking = False
            self.is_responding = False
            self.is_processing = False

            tool_id = event.get("id", "")
            tc_data = {
                "id": tool_id,
                "name": event.get("name", "unknown"),
                "args": event.get("args", {}),
            }

            # 用 tool_id 去重和更新（finalize 后会发送带完整参数的更新）
            if tool_id:
                updated = False
                for i, tc in enumerate(self.tool_calls):
                    if tc.get("id") == tool_id:
                        self.tool_calls[i] = tc_data
                        updated = True
                        break
                if not updated:
                    self.tool_calls.append(tc_data)
            else:
                self.tool_calls.append(tc_data)

        elif event_type == "tool_result":
            self.is_processing = True  # 工具执行完成，等待 AI 继续处理
            self.tool_results.append(
                {
                    "name": event.get("name", "unknown"),
                    "content": event.get("content", ""),
                }
            )

        elif event_type == "done":
            self.is_processing = False
            if not self.response_text:
                self.response_text = event.get("response", "")

        elif event_type == "error":
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            # 将错误添加到响应中显示
            error_msg = event.get("message", "Unknown error")
            self.response_text += f"\n\n[Error] {error_msg}"

        return event_type

    def get_display_args(self) -> dict:
        """获取用于 create_streaming_display 的参数"""
        return {
            "thinking_text": self.thinking_text,
            "response_text": self.response_text,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_thinking": self.is_thinking,
            "is_responding": self.is_responding,
            "is_processing": self.is_processing,
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
    # 显示 thinking
    if show_thinking and state.thinking_text:
        display_thinking = state.thinking_text
        if len(display_thinking) > thinking_max_length:
            half = thinking_max_length // 2
            display_thinking = (
                display_thinking[:half]
                + "\n\n... (truncated) ...\n\n"
                + display_thinking[-half:]
            )
        console.print(
            Panel(
                Text(display_thinking, style="dim"),
                title="🧠 Thinking",
                border_style="blue",
            )
        )

    # 显示工具调用和结果（Claude Code 风格）
    if show_tools and state.tool_calls:
        for i, tc in enumerate(state.tool_calls):
            # 判断是否有结果及成功状态
            has_result = i < len(state.tool_results)
            tr = state.tool_results[i] if has_result else None
            content = tr.get('content', '') if tr else ''

            # 确定状态和颜色
            if has_result and is_success(content):
                status = ToolStatus.SUCCESS
                style = "bold green"
            elif has_result:
                status = ToolStatus.ERROR
                style = "bold red"
            else:
                status = ToolStatus.PENDING
                style = "dim"

            # 紧凑格式显示
            tool_compact = format_tool_compact(tc['name'], tc.get('args'))
            tool_text = Text()
            tool_text.append(f"{status.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            console.print(tool_text)

            # 显示工具结果（树形格式）
            if has_result:
                result_elements = format_tool_result(
                    tr['name'],
                    content,
                    max_length=tool_result_max_length,
                    compact=True,
                )
                for elem in result_elements:
                    console.print(elem)
        console.print()

    # 显示最终响应
    if state.response_text:
        if show_response_panel:
            console.print(
                Panel(
                    Markdown(state.response_text),
                    title="💬 Response",
                    border_style="green",
                )
            )
        else:
            console.print(f"\n[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(state.response_text))
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


def create_streaming_display(
    thinking_text: str = "",
    response_text: str = "",
    tool_calls: list = None,
    tool_results: list = None,
    is_thinking: bool = False,
    is_responding: bool = False,
    is_waiting: bool = False,
    is_processing: bool = False,
) -> Group:
    """
    创建流式显示的布局

    Args:
        thinking_text: 当前累积的 thinking 文本
        response_text: 当前累积的响应文本
        tool_calls: 工具调用列表
        tool_results: 工具结果列表
        is_thinking: 是否正在思考
        is_responding: 是否正在响应
        is_waiting: 是否处于初始等待状态
        is_processing: 工具执行后等待 AI 继续处理

    Returns:
        Rich Group 对象
    """
    elements = []
    tool_calls = tool_calls or []
    tool_results = tool_results or []

    # 判断是否有工具正在执行中
    is_tool_executing = len(tool_calls) > len(tool_results)

    # 初始等待状态 - 显示 spinner 提示
    if is_waiting and not thinking_text and not response_text and not tool_calls:
        spinner = Spinner("dots", text=" AI 正在思考中...", style="cyan")
        elements.append(spinner)
        return Group(*elements)

    # Thinking 面板
    if thinking_text:
        thinking_title = "🧠 Thinking"
        if is_thinking:
            thinking_title += " ..."
        # 限制显示长度，保留最新内容
        display_thinking = thinking_text
        if len(display_thinking) > DisplayLimits.THINKING_STREAM:
            display_thinking = (
                "..." + display_thinking[-DisplayLimits.THINKING_STREAM :]
            )
        elements.append(
            Panel(
                Text(display_thinking, style="dim"),
                title=thinking_title,
                border_style="blue",
                padding=(0, 1),
            )
        )

    # Tool Calls 和 Results 配对显示（Claude Code 风格）
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            # 判断工具状态
            has_result = i < len(tool_results)
            tr = tool_results[i] if has_result else None

            # 确定状态和颜色
            if has_result:
                # 已完成：根据结果判断成功/失败
                content = tr.get('content', '') if tr else ''
                if is_success(content):
                    status = ToolStatus.SUCCESS
                    style = "bold green"
                else:
                    status = ToolStatus.ERROR
                    style = "bold red"
            else:
                # 执行中
                status = ToolStatus.RUNNING
                style = "bold yellow"

            # 紧凑格式显示工具调用
            tool_compact = format_tool_compact(tc['name'], tc.get('args'))
            tool_text = Text()
            tool_text.append(f"{status.value} ", style=style)
            tool_text.append(tool_compact, style=style)
            elements.append(tool_text)

            # 显示对应的结果或"正在执行"状态
            if has_result:
                # 已有结果，显示树形输出
                result_elements = format_tool_result(
                    tr['name'],
                    tr.get('content', ''),
                    compact=True,  # 使用紧凑格式
                )
                elements.extend(result_elements)
            else:
                # 还没有结果，显示带 spinner 的"正在执行"状态
                spinner = Spinner("dots", text=" 执行中...", style="yellow")
                elements.append(spinner)

    # 工具执行后等待 AI 继续处理的状态
    if is_processing and not is_thinking and not is_responding and not response_text:
        spinner = Spinner("dots", text=" AI 正在分析结果...", style="cyan")
        elements.append(spinner)

    # Response 面板
    if response_text:
        response_title = "💬 Response"
        if is_responding:
            response_title += " ..."
        elements.append(
            Panel(
                Markdown(response_text),
                title=response_title,
                border_style="green",
                padding=(0, 1),
            )
        )
    elif is_responding and not thinking_text:
        # 显示等待指示器
        elements.append(Text("⏳ Generating response...", style="dim"))

    return Group(*elements) if elements else Text("⏳ Processing...", style="dim")


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

        with Live(console=console, refresh_per_second=10, transient=True) as live:
            # 立即显示等待状态
            live.update(create_streaming_display(is_waiting=True))

            for event in agent.stream_events(prompt):
                event_type = state.handle_event(event)

                # 更新 Live 显示
                live.update(create_streaming_display(**state.get_display_args()))

                # tool_call 和 tool_result 时强制刷新
                # tool_call: 确保"正在执行"状态立即可见
                # tool_result: 确保"正在分析结果"状态立即可见
                if event_type in ("tool_call", "tool_result"):
                    live.refresh()

        # 显示最终结果
        console.print()
        display_final_results(
            state,
            tool_result_max_length=1000,  # cmd_run 用较长的限制
            args_max_length=400,
            show_response_panel=True,
        )

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

            with Live(console=console, refresh_per_second=10, transient=True) as live:
                # 立即显示等待状态
                live.update(create_streaming_display(is_waiting=True))

                for event in agent.stream_events(user_input, thread_id=thread_id):
                    event_type = state.handle_event(event)

                    # 更新 Live 显示
                    live.update(create_streaming_display(**state.get_display_args()))

                    # tool_call 和 tool_result 时强制刷新
                    # tool_call: 确保"正在执行"状态立即可见
                    # tool_result: 确保"正在分析结果"状态立即可见
                    if event_type in ("tool_call", "tool_result"):
                        live.refresh()

            # 显示最终结果（交互模式：不用 Panel 包裹响应）
            display_final_results(
                state,
                thinking_max_length=500,  # 交互模式用较短的 thinking 显示
                tool_result_max_length=DisplayLimits.TOOL_RESULT_FINAL,
                args_max_length=DisplayLimits.ARGS_FORMATTED,
                show_thinking=False,
                show_tools=False,
                show_response_panel=False,  # 交互模式不用 Panel
            )

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
