
import os
import asyncio
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv(override=True)

async def debug_anthropic_stream():
    """
    调试 Anthropic 流式响应的原始结构
    模拟 Skill Loading 场景
    """
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        print("Error: No ANTHROPIC_API_KEY found.")
        return

    # 初始化模型（启用 thinking）
    model = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        api_key=api_key,
        temperature=1.0,
        thinking={
            "type": "enabled",
            "budget_tokens": 1024
        }
    )

    # 模拟 system prompt (包含 load_skill 工具定义)
    tools = [
        {
            "name": "load_skill",
            "description": "Load detailed instructions for a specific skill.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Name of the skill to load"}
                },
                "required": ["skill_name"]
            }
        }
    ]
    
    # 绑定工具
    model_with_tools = model.bind_tools(tools)

    # 构造请求：模拟用户意图匹配某个 Skill
    messages = [
        HumanMessage(content="Use the news-extractor skill to find latest AI news.")
    ]

    print("\n--- Starting Stream Debug ---\n")
    
    # 捕获流式块
    async for chunk in model_with_tools.astream(messages):
        print(f"\n[Chunk Type]: {type(chunk).__name__}")
        
        # 打印原始 content 结构
        if hasattr(chunk, "content"):
            print(f"[Content]: {chunk.content!r}")
            
        # 打印 tool_calls
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            print(f"[Tool Calls]: {chunk.tool_calls}")
            
        # 打印 additional_kwargs (通常包含原始 API 响应)
        if hasattr(chunk, "additional_kwargs"):
            print(f"[Additional Kwargs]: {chunk.additional_kwargs}")
            
        # 打印 response_metadata (有时包含 finish_reason)
        if hasattr(chunk, "response_metadata"):
             print(f"[Metadata]: {chunk.response_metadata}")

if __name__ == "__main__":
    asyncio.run(debug_anthropic_stream())
