"""
mcp_bridge.py
=============
Connects the Pipecat LLM natively to the FastMCP tools using the official `mcp` SDK.
Provides async wrappers that Pipecat can register as standard tool functions.
"""

import json
from contextlib import AsyncExitStack

from loguru import logger
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


class MCPBridge:
    def __init__(self, sse_url: str = "http://localhost:8000/mcp/sse"):
        """Initialize the bridge pointing to a FastMCP streamable-http endpoint."""
        self.sse_url = sse_url
        self._exit_stack = None
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        """Open the persistent SSE link to the FastMCP server."""
        self._exit_stack = AsyncExitStack()
        logger.info(f"🔌 Connecting to MCP SSE at {self.sse_url}...")
        
        try:
            # mcp library's sse_client requires a URL
            # We obtain (read_stream, write_stream)
            sse = await self._exit_stack.enter_async_context(sse_client(self.sse_url))
            read, write = sse
            
            # Start the session
            self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            
            # Initialize with the server
            await self.session.initialize()
            logger.info("✅ MCP Session Initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP: {e}")
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Clean up the SSE connections."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self.session = None

    # --------------------------------------------------------------------------
    # Pipecat Tool Map 
    # --------------------------------------------------------------------------

    async def dnd_resolve_melee_attack(self, attacker_id: str, target_id: str, weapon: str) -> str:
        """
        Executes the melee attack through the FastMCP engine.
        Wrapped as a standard async-def so Pipecat can natively inject it to Claude.
        """
        if not self.session:
            return json.dumps({"error": "MCP session is offline."})

        logger.debug(f"[TOOL ->] dnd_resolve_melee_attack({attacker_id}, {target_id}, {weapon})")
        
        try:
            result = await self.session.call_tool("dnd_resolve_melee_attack", arguments={
                "attacker_id": attacker_id,
                "target_id": target_id,
                "weapon": weapon
            })
            
            if not result.content:
                return json.dumps({"error": "No output from tool"})
            
            # Usually tool returns a single TextContent block mapped to our _safe_json string
            output_str = "\n".join(c.text for c in result.content if c.type == "text")
            logger.debug(f"[<- TOOL] {output_str}")
            return output_str

        except Exception as e:
            logger.error(f"Error calling MCP tool: {e}")
            return json.dumps({"error": str(e)})

