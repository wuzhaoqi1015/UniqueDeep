#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/workflow.py
@Time: 2026/02/24
@Author: GeorgeWu
@Description: LangGraph multi-agent workflow implementation.
'''

import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END

from .agent import LangChainSkillsAgent, create_skills_agent, get_model_config


# --- State Definition ---

class AgentState(TypedDict):
    """The shared state of the agent workflow."""
    # The list of messages in the conversation
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The next agent to route to
    next: str


# --- Node Helper ---

def create_agent_node(agent: LangChainSkillsAgent, name: str):
    """
    Wraps a LangChainSkillsAgent into a LangGraph node function.
    
    Args:
        agent: The initialized LangChainSkillsAgent instance
        name: The name of the agent (e.g., "Coder", "Researcher")
    """
    def agent_node(state: AgentState) -> dict:
        # Get the last message content
        last_message = state["messages"][-1]
        content = last_message.content
        
        # Invoke the agent
        # Note: LangChainSkillsAgent.invoke returns a dict with 'messages' or 'output'
        # We need to extract the AI response
        result = agent.invoke(content)
        
        # Extract the response text
        response_text = agent.get_last_response(result)
        
        # Return as an AIMessage with the agent's name
        return {
            "messages": [AIMessage(content=response_text, name=name)]
        }
    
    return agent_node


# --- Supervisor / Router ---

def create_supervisor_chain(members: List[str], model_name: str | None = None):
    """
    Creates a supervisor chain that decides which agent should act next.
    """
    # Use provided model or fall back to environment config
    if not model_name:
        _, env_model, _, _ = get_model_config()
        model_name = env_model or "claude-3-7-sonnet-20250219"

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    
    options = ["FINISH"] + members
    
    # Using a simple JSON output for routing
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "human",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}."
                " Return the result as a JSON object with a single key 'next'."
            ),
        ]
    ).partial(options=str(options), members=", ".join(members))
    
    # Initialize a lightweight model for routing
    llm = init_chat_model(model_name)
    
    return prompt | llm | JsonOutputParser()


# --- Workflow Builder ---

def create_multi_agent_graph(
    agents: dict[str, LangChainSkillsAgent],
    supervisor_model: str | None = None
) -> StateGraph:
    """
    Builds a LangGraph StateGraph for multi-agent collaboration.
    
    Args:
        agents: A dictionary mapping agent names to LangChainSkillsAgent instances
        supervisor_model: The model to use for the supervisor
        
    Returns:
        A compiled LangGraph
    """
    members = list(agents.keys())
    supervisor_chain = create_supervisor_chain(members, supervisor_model)
    
    workflow = StateGraph(AgentState)
    
    # Add the supervisor node
    def supervisor_node(state: AgentState):
        result = supervisor_chain.invoke(state)
        return result
        
    workflow.add_node("supervisor", supervisor_node)
    
    # Add worker nodes
    for name, agent in agents.items():
        node = create_agent_node(agent, name)
        workflow.add_node(name, node)
        
    # Define edges
    # Supervisor decides who goes next
    workflow.set_entry_point("supervisor")
    
    # Conditional edges from supervisor
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        conditional_map
    )
    
    # Workers always report back to supervisor
    for name in members:
        workflow.add_edge(name, "supervisor")
        
    return workflow.compile()
