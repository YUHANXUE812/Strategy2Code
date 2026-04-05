#!/usr/bin/env python3
"""Deprecated chat agent entrypoint.

This repository now exposes pipeline operations through the MCP server:
`agent/mcp_server.py`.
"""

from __future__ import annotations

import argparse
import json
import sys


DEPRECATION_MESSAGE = (
    "agent/chat_agent.py is deprecated and no longer provides an interactive "
    "orchestration loop. Use MCP tools from agent/mcp_server.py instead."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated entrypoint.")
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Emit deprecation payload as JSON.",
    )
    _ = parser.parse_args()

    payload = {
        "status": "deprecated",
        "message": DEPRECATION_MESSAGE,
        "replacement": "python agent/mcp_server.py",
    }
    if _.json_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(DEPRECATION_MESSAGE)
        print("Replacement: python agent/mcp_server.py")
    sys.exit(1)


if __name__ == "__main__":
    main()
