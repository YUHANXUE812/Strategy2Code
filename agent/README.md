# Agent Directory

## Active Entrypoints

- `mcp_server.py`: MCP server exposing tools for Strategy2Code.
- `pipeline_service.py`: service layer used by MCP tools.

## Deprecated Entrypoint

- `chat_agent.py` is intentionally deprecated.
- The previous interactive chat planning/execution loop was removed.
- Use an MCP host (for example Cline) and call tools from `mcp_server.py`.
