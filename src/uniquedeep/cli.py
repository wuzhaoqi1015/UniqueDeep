#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/cli.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: Command Line Interface for LangChain Skills Agent.
'''

import argparse
import asyncio
import os
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from rich.live import Live
from rich.panel import Panel

from .agent import LangChainSkillsAgent
from .tools import bing_mcp_client
from .skill_loader import SkillLoader
from . import config
from . import ui
from .stream.state import StreamState


def run_agent(agent: LangChainSkillsAgent, prompt: str, thread_id: str = "default"):
    """
    运行 Agent 并处理流式输出
    """
    ui.console.print()
    state = StreamState()
    printed_count = 0
    thinking_round = 0

    try:
        with Live(console=ui.console, refresh_per_second=10, transient=True) as live:
            # 立即显示等待状态
            live.update(ui.create_streaming_display(is_waiting=True))

            for event in agent.stream_events(prompt, thread_id=thread_id):
                event_type = state.handle_event(event)
                
                # 检查是否有已完成的事件需要归档打印
                while printed_count < len(state.events):
                    evt = state.events[printed_count]
                    if evt.get("is_completed"):
                        if evt["type"] == "thinking":
                            thinking_round += 1
                        
                        ui.console.print(ui.render_event_static(evt, thinking_round))
                        printed_count += 1
                    else:
                        break

                # 更新 Live 显示（只显示未归档的活动事件）
                active_events = state.events[printed_count:]
                live.update(ui.create_streaming_display(
                    events=active_events,
                    is_processing=state.is_processing,
                    start_thinking_count=thinking_round + 1
                ))

                if event_type in ("tool_call", "tool_result", "thinking", "text"):
                    live.refresh()
        
        # 打印剩余的事件
        while printed_count < len(state.events):
            evt = state.events[printed_count]
            if evt["type"] == "thinking":
                thinking_round += 1
            ui.console.print(ui.render_event_static(evt, thinking_round))
            printed_count += 1
            
    except Exception as e:
        ui.console.print(f"[red]Error: {e}[/red]")
        # Re-raise to let caller handle if needed, or just log
        # raise


def cmd_list_skills():
    """列出发现的 Skills"""
    loader = SkillLoader()
    skills = loader.scan_skills()
    ui.render_skills_list(skills)


def cmd_show_prompt():
    """显示 system prompt（演示 Level 1）"""
    agent = LangChainSkillsAgent()
    prompt = agent.get_system_prompt()
    skills = agent.get_discovered_skills()
    token_estimate = len(prompt) // 4
    ui.render_system_prompt(prompt, len(skills), token_estimate)


def cmd_list_models():
    """列出常用模型列表"""
    cfg = config.load_models_config()
    models = config.get_flattened_models(cfg)
    
    active_model = cfg.get("active_model")
    default_temp = cfg.get("default_config", {}).get("temperature", 0.1)
    default_max_tokens = cfg.get("default_config", {}).get("max_tokens", 4096)
    
    ui.render_models_list(models, active_model, default_temp, default_max_tokens)


def cmd_run(prompt: str, enable_thinking: bool = True):
    """
    执行单次请求
    """
    ui.console.print(Panel(f"[bold cyan]User Request:[/bold cyan]\n{prompt}"))
    ui.console.print()

    if not config.check_api_credentials():
        ui.console.print("[red]Error: API credentials not set[/red]")
        ui.console.print("Please set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY in .env file")
        sys.exit(1)

    agent = LangChainSkillsAgent(enable_thinking=enable_thinking)
    ui.console.print("[dim]Running agent with streaming output...[/dim]\n")

    run_agent(agent, prompt)


# === Interactive Commands Handlers ===

def handle_skills(agent, args):
    cmd_list_skills()

def handle_prompt(agent, args):
    cmd_show_prompt()

def handle_models(agent, args):
    cmd_list_models()

def handle_temp(agent, args):
    parts = args.split()
    if len(parts) == 1:
        try:
            temp_val = float(parts[0])
            if 0.0 <= temp_val <= 1.0:
                if agent.set_temperature(temp_val):
                    ui.console.print(f"[green]✓ Temperature set to {temp_val}[/green]")
                else:
                    ui.console.print(
                        "[yellow]! Cannot change temperature: Extended Thinking is enabled (requires temperature=1.0)[/yellow]"
                    )
            else:
                ui.console.print("[red]! Temperature must be between 0.0 and 1.0[/red]")
        except ValueError:
            ui.console.print("[red]! Invalid temperature value[/red]")
    else:
        ui.console.print(f"[dim]Current temperature: {agent.temperature}[/dim]")

def handle_model_switch(agent, args, thread_id="interactive"):
    parts = args.split()
    if len(parts) >= 1:
        new_model_name = parts[0]
        
        cfg = config.load_models_config()
        models = config.get_flattened_models(cfg)
        
        target_model = None
        for m in models:
            if m["name"] == new_model_name:
                target_model = m
                break
        
        if not target_model:
            ui.console.print(f"[red]! Model '{new_model_name}' not found in models.json[/red]")
            ui.console.print("[dim]Use /models to list available models.[/dim]")
            return
            
        provider_key = target_model["provider"]
        ui.console.print(f"[dim]Switching model to {new_model_name} ({provider_key})...[/dim]")
        
        if agent.switch_model(new_model_name, provider_key, thread_id=thread_id):
            # Update models.json
            cfg["active_model"] = new_model_name
            cfg["active_provider"] = provider_key
            config.save_models_config(cfg)
            
            ui.console.print(f"[green]✓ Switched to model: {agent.model_name} ({agent.provider})[/green]")
            
            if agent.provider == "anthropic" and agent.enable_thinking:
                ui.console.print("[dim]Extended Thinking enabled[/dim]")
            elif agent.enable_thinking:
                ui.console.print(f"[dim]Extended Thinking enabled for {agent.model_name}[/dim]")
            else:
                ui.console.print("[dim]Extended Thinking disabled[/dim]")
        else:
            ui.console.print("[red]! Failed to switch model[/red]")
    else:
        ui.console.print(f"[dim]Current model: {agent.model_name} ({agent.provider})[/dim]")
        ui.console.print("[dim]Usage: /model <model_name>[/dim]")


COMMANDS = {
    "/skills": handle_skills,
    "/prompt": handle_prompt,
    "/models": handle_models,
    "/list-models": handle_models,
    "/temp": handle_temp,
    "/model": handle_model_switch,
}


def cmd_interactive(enable_thinking: bool = True):
    """
    交互式对话模式
    """
    ui.print_banner()

    # 从 models.json 加载初始配置
    models_config = config.load_models_config()
    initial_model = models_config.get("active_model")
    initial_provider = models_config.get("active_provider")
    
    agent = LangChainSkillsAgent(enable_thinking=enable_thinking)
    
    # 显式切换以确保状态一致
    if initial_model:
        thread_id = "interactive" # Use separate thread ID for init switch if needed? No, just switch.
        if agent.model_name != initial_model or agent.provider != initial_provider:
             agent.switch_model(initial_model, initial_provider)
        else:
             agent.switch_model(initial_model, initial_provider)

    # 显示发现的 Skills
    skills = agent.get_discovered_skills()
    ui.console.print(f"\n[green]✓[/green] Discovered {len(skills)} skills")
    for skill in skills:
        ui.console.print(f"  - {skill['name']}")
    
    # 检查 MCP 状态
    ui.console.print("\n[dim]Checking MCP Status...[/dim]")
    try:
        async def check_mcp():
            try:
                return await bing_mcp_client.list_tools()
            finally:
                await bing_mcp_client.close()
            
        mcp_tools = asyncio.run(check_mcp())
        if mcp_tools:
            ui.console.print(f"[green]✓[/green] MCP Connected: bing-cn-mcp ({len(mcp_tools)} tools)")
            for tool in mcp_tools:
                 if hasattr(tool, 'name'):
                     name = tool.name
                 else:
                     name = tool.get('name', 'unknown')
                 ui.console.print(f"  - {name} (MCP)")
        else:
            ui.console.print("[yellow]! MCP Connected but no tools found[/yellow]")
    except Exception as e:
        ui.console.print(f"[yellow]! MCP Connection Failed: {str(e)}[/yellow]")
        ui.console.print("[dim]  Ensure 'bing-cn-mcp' is installed via npm and npx is in PATH.[/dim]")
        ui.console.print("[dim]  Note: Some MCP servers require Node.js >= 20.[/dim]")
    
    ui.console.print()

    thinking_status = (
        "[green]enabled[/green]" if enable_thinking else "[dim]disabled[/dim]"
    )
    ui.console.print(f"[dim]Extended Thinking: {thinking_status}[/dim]")
    ui.console.print(
        "[dim]Commands: /exit to quit, /skills to list skills, /prompt to show system prompt, /models to list common models, /temp [val] to set temperature, /model <name> [provider] to switch model[/dim]\n"
    )

    thread_id = "interactive"
    session = PromptSession(
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    while True:
        try:
            user_input = session.prompt(
                HTML('<ansigreen><b>You:</b></ansigreen> ')
            ).strip()

            if not user_input:
                continue

            # 退出命令
            if user_input.lower() in ("/exit", "/quit", "/q"):
                ui.console.print("[dim]Goodbye![/dim]")
                break

            # 处理 slash commands
            command_processed = False
            for cmd, handler in COMMANDS.items():
                if user_input.lower().startswith(cmd):
                    args = user_input[len(cmd):].strip()
                    if cmd == "/model":
                         handler(agent, args, thread_id=thread_id)
                    else:
                         handler(agent, args)
                    command_processed = True
                    break
            
            if command_processed:
                continue

            # 运行 agent
            run_agent(agent, user_input, thread_id=thread_id)

        except KeyboardInterrupt:
            ui.console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            ui.console.print(f"[red]Error: {e}[/red]")


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
