#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: examples/multi_agent_demo.py
@Description: Demo of multi-agent collaboration using LangGraph and UniqueDeep skills.
'''

import sys
from pathlib import Path

# Add src to path to import uniquedeep
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel

from uniquedeep.agent import create_skills_agent
from uniquedeep.workflow import create_multi_agent_graph

# Load environment variables
load_dotenv(override=True)

console = Console()

def main():
    console.print(Panel.fit("[bold blue]Multi-Agent Collaboration Demo[/bold blue]"))

    # 1. Initialize specialized agents
    # In a real scenario, you would point skill_paths to different directories
    # for each agent to give them different capabilities.
    # For this demo, we'll use the default skills but give them different personas via prompt.
    
    # Coder Agent
    coder = create_skills_agent(
        enable_thinking=True
    )
    # Customize system prompt if needed (optional, as Level 1 injection handles skills)
    
    # Researcher Agent
    researcher = create_skills_agent(
        enable_thinking=True
    )
    
    agents = {
        "Coder": coder,
        "Researcher": researcher
    }
    
    # 2. Build the graph
    graph = create_multi_agent_graph(agents)
    
    # 3. Run the workflow
    user_input = "Find out what is the latest version of Python and write a script to print it."
    
    console.print(f"\n[bold green]User Request:[/bold green] {user_input}\n")
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # Stream the execution
    for output in graph.stream(initial_state):
        for key, value in output.items():
            if key == "supervisor":
                next_agent = value.get("next")
                console.print(f"[bold yellow]Supervisor[/bold yellow] -> [bold cyan]{next_agent}[/bold cyan]")
            else:
                # Worker agent output
                messages = value.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    console.print(Panel(
                        last_msg.content,
                        title=f"[bold cyan]{key}[/bold cyan]",
                        border_style="cyan"
                    ))

if __name__ == "__main__":
    main()
