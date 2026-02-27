# src/uniquedeep/mcp_manager.py

import asyncio
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

# Try to import MCP, if not available, provide dummy implementation or warn
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Dummy classes for type hinting
    ClientSession = Any
    StdioServerParameters = Any

logger = logging.getLogger(__name__)

class McpManager:
    """
    Manages connections to MCP servers and converts MCP tools to LangChain tools.
    Handles lifecycle of connections in a background thread/loop.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._exit_stack = AsyncExitStack()
        self._sessions: Dict[str, ClientSession] = {}
        self.tools: List[StructuredTool] = []
        self._running = False

    def start(self):
        """Starts the background event loop and connects to MCP servers."""
        if not MCP_AVAILABLE:
            logger.warning("MCP library not installed. Skipping MCP initialization.")
            return

        if self._running:
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True

        # Run connection logic in the loop
        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        try:
            future.result(timeout=10)  # Wait for connections with timeout
        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")

    def _run_loop(self):
        """Runs the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self):
        """Connects to all configured MCP servers."""
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"MCP config not found: {self.config_path}")
            return

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading MCP config: {e}")
            return

        servers = config.get("mcpServers", {})
        for name, server_config in servers.items():
            try:
                command = server_config.get("command")
                args = server_config.get("args", [])
                env = server_config.get("env", {})
                
                # Merge environment variables
                full_env = os.environ.copy()
                full_env.update(env)
                
                # Create params
                params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=full_env
                )

                # Connect
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                
                await session.initialize()
                self._sessions[name] = session
                
                # Load tools
                result = await session.list_tools()
                for tool in result.tools:
                    self.tools.append(self._convert_tool(name, tool, session))
                    
                logger.info(f"Connected to MCP server: {name}")
                print(f"[MCP] Connected to server: {name}, loaded {len(result.tools)} tools")
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")
                print(f"[MCP] Failed to connect to server {name}: {e}")

    def _convert_tool(self, server_name: str, mcp_tool: Any, session: ClientSession) -> StructuredTool:
        """Converts an MCP tool to a LangChain StructuredTool."""
        
        # Ensure unique name
        tool_name = f"{server_name}__{mcp_tool.name}"
        
        def _sync_func(**kwargs):
            """Sync wrapper for tool execution."""
            if not self._loop:
                raise RuntimeError("MCP loop not running")
            
            # Submit task to background loop
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(mcp_tool.name, arguments=kwargs),
                self._loop
            )
            result = future.result()
            
            # Format result
            output = []
            if hasattr(result, 'content'):
                for content in result.content:
                    if content.type == "text":
                        output.append(content.text)
                    elif content.type == "image":
                        output.append(f"[Image: {content.mimeType}]")
                    elif content.type == "resource":
                        output.append(f"[Resource: {content.resource.uri}]")
            else:
                output.append(str(result))
            
            return "\n".join(output)

        # Create Pydantic model for args
        args_schema = self._create_pydantic_model(f"{tool_name}Schema", mcp_tool.inputSchema)

        return StructuredTool.from_function(
            func=_sync_func,
            name=tool_name,
            description=mcp_tool.description or "",
            args_schema=args_schema
        )

    def _create_pydantic_model(self, name: str, schema: dict) -> type[BaseModel]:
        """Dynamically creates a Pydantic model from JSON Schema."""
        fields = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                # Determine type
                py_type = Any
                json_type = prop_schema.get("type")
                
                if json_type == "string":
                    py_type = str
                elif json_type == "integer":
                    py_type = int
                elif json_type == "boolean":
                    py_type = bool
                elif json_type == "number":
                    py_type = float
                elif json_type == "array":
                    py_type = List[Any] # Simplified
                elif json_type == "object":
                    py_type = Dict[str, Any] # Simplified
                
                # Determine required
                is_required = prop_name in schema.get("required", [])
                default = ... if is_required else None
                
                fields[prop_name] = (py_type, Field(default=default, description=prop_schema.get("description")))
                
        return create_model(name, **fields)

    def cleanup(self):
        """Stops the loop and closes connections."""
        if self._running and self._loop:
            # Cleanup sessions
            if self._exit_stack:
                future = asyncio.run_coroutine_threadsafe(self._exit_stack.aclose(), self._loop)
                try:
                    future.result(timeout=5)
                except Exception:
                    pass
            
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            self._running = False
