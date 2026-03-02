#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File: src/uniquedeep/mcp_client.py
@Time: 2026/02/28
@Author: UniqueDeep
@Description: MCP Client for integrating with MCP servers.
'''

import asyncio
import os
from typing import Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    """
    A client for connecting to and interacting with MCP servers.
    """
    def __init__(self, command: str, args: List[str]):
        self.command = command
        self.args = args
        self.session = None
        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None

    async def _ensure_connected(self):
        """Ensure connection to the MCP server is established."""
        if self.session:
            return

        # Prepare server parameters
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=os.environ.copy()
        )

        # Connect via stdio
        self._stdio_context = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._stdio_context.__aenter__()
        
        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()
        
        # Initialize
        await self.session.initialize()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server."""
        await self._ensure_connected()
        result = await self.session.list_tools()
        # Ensure result.tools is a list of dictionaries or objects with 'name' and 'description'
        return result.tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        await self._ensure_connected()
        result = await self.session.call_tool(name, arguments=arguments)
        return result

    async def close(self):
        """Close the connection."""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
        
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None
            self._read_stream = None
            self._write_stream = None

# Helper for synchronous execution (since UniqueDeep is largely sync)
def run_mcp_tool(client: MCPClient, tool_name: str, arguments: Dict[str, Any]) -> str:
    """Helper to run MCP tool synchronously."""
    try:
        async def _run():
            try:
                return await client.call_tool(tool_name, arguments)
            finally:
                await client.close()

        result = asyncio.run(_run())
        
        # Format the result content
        output = []
        if hasattr(result, 'content'):
            for content in result.content:
                if hasattr(content, 'text'):
                    output.append(content.text)
                else:
                    output.append(str(content))
        else:
            output.append(str(result))
            
        return "\n".join(output)
    except Exception as e:
        return f"[Error] MCP Tool Execution Failed: {str(e)}"
