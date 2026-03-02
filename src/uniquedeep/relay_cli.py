#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/relay_cli.py
@Time: 2026/02/28
@Author: UniqueDeep
@Description: CLI for Relay Mode
'''

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner

from .relay_agent import RelayAgent, DEFAULT_THINKING_BUDGET
from .stream import StreamEventEmitter, DisplayLimits, ToolStatus, format_tool_compact, is_success, ToolResultFormatter

# Load environment variables
load_dotenv(override=True)

console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

# Global formatter
formatter = ToolResultFormatter()

def format_tool_result_compact(name: str, content: str, max_lines: int = 5) -> list:
    """
    使用 Claude Code 风格格式化工具结果（树形输出）
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

def format_tool_result(name: str, content: str, max_length: int = 800, compact: bool = False) -> list:
    if compact:
        return format_tool_result_compact(name, content, max_lines=10)
    else:
        result = formatter.format(name, content, max_length)
        return result.elements

class RelayStreamState:
    """Relay Stream State Container"""
    def __init__(self):
        self.current_stage = "waiting" # waiting, planning, executing
        self.current_model = ""
        
        # Planning phase data
        self.planner_thinking = ""
        self.planner_response = ""
        
        # Execution phase data
        self.executor_thinking = ""
        self.executor_response = ""
        self.tool_calls = []
        self.tool_results = []
        
        # Flags
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False

    def handle_event(self, event: dict):
        event_type = event.get("type")
        
        if event_type == "stage_start":
            self.current_stage = event.get("stage")
            self.current_model = event.get("model", "")
            # Reset flags for new stage
            self.is_thinking = False
            self.is_responding = False
            
        elif event_type == "stage_end":
            self.current_stage = "transition" # Just finished a stage
            
        elif event_type == "thinking":
            self.is_thinking = True
            content = event.get("content", "")
            if self.current_stage == "planning":
                self.planner_thinking += content
            elif self.current_stage == "executing":
                self.executor_thinking += content
                
        elif event_type == "text":
            self.is_thinking = False
            self.is_responding = True
            content = event.get("content", "")
            if self.current_stage == "planning":
                self.planner_response += content
            elif self.current_stage == "executing":
                self.executor_response += content
                
        elif event_type == "tool_call":
            if self.current_stage == "executing":
                tool_id = event.get("id", "")
                tc_data = {
                    "id": tool_id,
                    "name": event.get("name", "unknown"),
                    "args": event.get("args", {}),
                }
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
            if self.current_stage == "executing":
                self.tool_results.append({
                    "name": event.get("name", "unknown"),
                    "content": event.get("content", ""),
                })
                
        elif event_type == "error":
             error_msg = event.get("message", "Unknown error")
             if self.current_stage == "planning":
                 self.planner_response += f"\n\n[Error] {error_msg}"
             else:
                 self.executor_response += f"\n\n[Error] {error_msg}"

    def get_display_args(self):
        return {
            "state": self
        }

def create_relay_display(state: RelayStreamState) -> Group:
    elements = []
    
    # 1. Planner Section
    if state.planner_thinking or state.planner_response or state.current_stage == "planning":
        elements.append(Text("Phase 1: Planning (DeepSeek-Reasoner)", style="bold purple"))
        
        if state.planner_thinking:
            display_thinking = state.planner_thinking
            if len(display_thinking) > DisplayLimits.THINKING_STREAM:
                display_thinking = "..." + display_thinking[-DisplayLimits.THINKING_STREAM:]
            
            elements.append(Panel(
                Text(display_thinking, style="dim"),
                title="Thinking",
                border_style="purple",
                padding=(0, 1)
            ))
            
        if state.planner_response:
            elements.append(Panel(
                Markdown(state.planner_response),
                title="Plan",
                border_style="green",
                padding=(0, 1)
            ))
        elif state.current_stage == "planning" and not state.planner_thinking:
             elements.append(Spinner("dots", text=" Planning...", style="purple"))
             
    # Separator if needed
    if state.current_stage in ("executing", "transition") and (state.executor_thinking or state.executor_response):
         elements.append(Text("\n" + "="*50 + "\n"))

    # 2. Executor Section
    if state.executor_thinking or state.executor_response or state.tool_calls or state.current_stage == "executing":
        elements.append(Text("Phase 2: Executing (Claude)", style="bold orange1"))
        
        if state.executor_thinking:
            display_thinking = state.executor_thinking
            if len(display_thinking) > DisplayLimits.THINKING_STREAM:
                 display_thinking = "..." + display_thinking[-DisplayLimits.THINKING_STREAM:]
            elements.append(Panel(
                Text(display_thinking, style="dim"),
                title="Thinking",
                border_style="orange1",
                padding=(0, 1)
            ))
            
        # Tool Calls
        if state.tool_calls:
            for i, tc in enumerate(state.tool_calls):
                has_result = i < len(state.tool_results)
                tr = state.tool_results[i] if has_result else None
                
                if has_result:
                    content = tr.get('content', '') if tr else ''
                    if is_success(content):
                        status = ToolStatus.SUCCESS
                        style = "bold green"
                    else:
                        status = ToolStatus.ERROR
                        style = "bold red"
                else:
                    status = ToolStatus.RUNNING
                    style = "bold yellow"
                    
                tool_compact = format_tool_compact(tc['name'], tc.get('args'))
                tool_text = Text()
                tool_text.append(f"{status.value} ", style=style)
                tool_text.append(tool_compact, style=style)
                elements.append(tool_text)
                
                if has_result:
                    result_elements = format_tool_result(
                        tr['name'],
                        tr.get('content', ''),
                        compact=True
                    )
                    elements.extend(result_elements)
                else:
                    elements.append(Spinner("dots", text=" Executing...", style="yellow"))

        if state.executor_response:
            elements.append(Panel(
                Markdown(state.executor_response),
                title="Result",
                border_style="green",
                padding=(0, 1)
            ))
        elif state.current_stage == "executing" and not state.executor_thinking and not state.tool_calls:
            elements.append(Spinner("dots", text=" Starting execution...", style="orange1"))

    return Group(*elements)

def main():
    parser = argparse.ArgumentParser(description="UniqueDeep Relay Mode CLI")
    parser.add_argument("prompt", nargs="?", help="User prompt")
    parser.add_argument("--planner", default="deepseek-reasoner", help="Planner model")
    parser.add_argument("--planner-provider", default="deepseek", help="Planner provider")
    parser.add_argument("--executor", default="claude-3-7-sonnet-20250219", help="Executor model")
    parser.add_argument("--executor-provider", default="anthropic", help="Executor provider")
    parser.add_argument("--no-thinking", action="store_true", help="Disable extended thinking for executor")
    
    args = parser.parse_args()
    
    if not args.prompt:
        console.print("[red]Please provide a prompt.[/red]")
        return

    # Initialize Relay Agent
    agent = RelayAgent(
        planner_model=args.planner,
        planner_provider=args.planner_provider,
        executor_model=args.executor,
        executor_provider=args.executor_provider,
        enable_thinking=not args.no_thinking
    )
    
    state = RelayStreamState()
    
    console.print(Panel(f"[bold cyan]Relay Mode Request:[/bold cyan]\n{args.prompt}"))
    console.print()
    
    with Live(console=console, refresh_per_second=10, transient=False) as live:
        live.update(create_relay_display(state))
        
        for event in agent.stream_events(args.prompt):
            state.handle_event(event)
            live.update(create_relay_display(state))
            
    console.print("\n[bold green]Relay Task Completed![/bold green]")

if __name__ == "__main__":
    main()
