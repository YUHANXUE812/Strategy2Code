#!/usr/bin/env python3
"""MCP server entrypoint for Strategy2Code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

try:
    from .pipeline_service import Strategy2CodePipelineService
except ImportError:
    from pipeline_service import Strategy2CodePipelineService


SERVICE_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "timestamp",
        "call_id",
        "action",
        "status",
        "message",
        "artifacts",
        "metrics",
        "commands",
        "errors",
    ],
    "properties": {
        "timestamp": {"type": "string", "description": "ISO-8601 timestamp"},
        "call_id": {"type": "string"},
        "action": {"type": "string"},
        "status": {"type": "string", "enum": ["success", "error"]},
        "message": {"type": "string"},
        "artifacts": {"type": "object"},
        "metrics": {"type": "object"},
        "commands": {"type": "array", "items": {"type": "object"}},
        "errors": {"type": "array", "items": {"type": "string"}},
    },
}


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "run_pipeline": {
        "input": {
            "type": "object",
            "required": ["paper_name", "paper_json_path"],
            "properties": {
                "paper_name": {"type": "string"},
                "paper_json_path": {"type": "string"},
                "gpt_version": {"type": "string", "default": "o4-mini"},
                "output_dir": {"type": "string", "default": ""},
                "output_repo_dir": {"type": "string", "default": ""},
                "paper_json_cleaned_path": {"type": "string", "default": ""},
            },
        },
        "output": SERVICE_RESULT_SCHEMA,
    },
    "evaluate_strategy": {
        "input": {
            "type": "object",
            "properties": {
                "paper_name": {"type": "string", "default": ""},
                "paper_json_path": {"type": "string", "default": ""},
                "output_dir": {"type": "string", "default": ""},
                "output_repo_dir": {"type": "string", "default": ""},
                "eval_type": {"type": "string", "enum": ["ref_free", "ref_based"], "default": "ref_free"},
                "generated_n": {"type": "integer", "default": 4},
                "gpt_version": {"type": "string", "default": "o4-mini"},
                "gold_repo_dir": {"type": "string", "default": ""},
                "eval_result_dir": {"type": "string", "default": ""},
            },
        },
        "output": SERVICE_RESULT_SCHEMA,
    },
    "verify_repo": {
        "input": {
            "type": "object",
            "properties": {
                "target_repo_dir": {"type": "string", "default": ""},
                "timeout_sec": {"type": "integer", "default": 600},
            },
        },
        "output": SERVICE_RESULT_SCHEMA,
    },
    "explain_repo": {
        "input": {
            "type": "object",
            "properties": {
                "target_repo_dir": {"type": "string", "default": ""},
            },
        },
        "output": SERVICE_RESULT_SCHEMA,
    },
    "get_status": {
        "input": {
            "type": "object",
            "properties": {
                "target_repo_dir": {"type": "string", "default": ""},
            },
        },
        "output": SERVICE_RESULT_SCHEMA,
    },
}


def _resolve_repo_root() -> Path:
    env_repo_root = os.environ.get("STRATEGY2CODE_REPO_ROOT", "").strip()
    if env_repo_root:
        return Path(env_repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_artifact_root(repo_root: Path) -> Path:
    env_artifact_root = os.environ.get("STRATEGY2CODE_MCP_ARTIFACT_ROOT", "").strip()
    if env_artifact_root:
        return Path(env_artifact_root).expanduser().resolve()
    return (repo_root / "agent" / "mcp_artifacts").resolve()


REPO_ROOT = _resolve_repo_root()
ARTIFACT_ROOT = _resolve_artifact_root(REPO_ROOT)
DEFAULT_MODEL = os.environ.get("STRATEGY2CODE_DEFAULT_MODEL", "o4-mini")

service = Strategy2CodePipelineService(
    repo_root=REPO_ROOT,
    default_model=DEFAULT_MODEL,
    artifact_root=ARTIFACT_ROOT,
)

mcp = FastMCP("Strategy2Code MCP Server", json_response=True)


@mcp.tool()
def run_pipeline(
    paper_name: str,
    paper_json_path: str,
    gpt_version: str = "",
    output_dir: str = "",
    output_repo_dir: str = "",
    paper_json_cleaned_path: str = "",
) -> Dict[str, Any]:
    """Run the full generation pipeline.

    Input schema: TOOL_SCHEMAS["run_pipeline"]["input"].
    Output schema: TOOL_SCHEMAS["run_pipeline"]["output"].
    """
    return service.run_pipeline(
        paper_name=paper_name,
        paper_json_path=paper_json_path,
        gpt_version=gpt_version,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        paper_json_cleaned_path=paper_json_cleaned_path,
    ).to_dict()


@mcp.tool()
def evaluate_strategy(
    paper_name: str = "",
    paper_json_path: str = "",
    output_dir: str = "",
    output_repo_dir: str = "",
    eval_type: str = "ref_free",
    generated_n: int = 4,
    gpt_version: str = "",
    gold_repo_dir: str = "",
    eval_result_dir: str = "",
) -> Dict[str, Any]:
    """Evaluate generated repository quality.

    Input schema: TOOL_SCHEMAS["evaluate_strategy"]["input"].
    Output schema: TOOL_SCHEMAS["evaluate_strategy"]["output"].
    """
    return service.evaluate_strategy(
        paper_name=paper_name,
        paper_json_path=paper_json_path,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        eval_type=eval_type,
        generated_n=generated_n,
        gpt_version=gpt_version,
        gold_repo_dir=gold_repo_dir,
        eval_result_dir=eval_result_dir,
    ).to_dict()


@mcp.tool()
def verify_repo(target_repo_dir: str = "", timeout_sec: int = 600) -> Dict[str, Any]:
    """Run generated repository main.py for verification.

    Input schema: TOOL_SCHEMAS["verify_repo"]["input"].
    Output schema: TOOL_SCHEMAS["verify_repo"]["output"].
    """
    return service.verify_repo(
        target_repo_dir=target_repo_dir,
        timeout_sec=timeout_sec,
    ).to_dict()


@mcp.tool()
def explain_repo(target_repo_dir: str = "") -> Dict[str, Any]:
    """Explain generated repository structure and run command.

    Input schema: TOOL_SCHEMAS["explain_repo"]["input"].
    Output schema: TOOL_SCHEMAS["explain_repo"]["output"].
    """
    return service.explain_repo(target_repo_dir=target_repo_dir).to_dict()


@mcp.tool()
def get_status(target_repo_dir: str = "") -> Dict[str, Any]:
    """Return verification and evaluation status summary.

    Input schema: TOOL_SCHEMAS["get_status"]["input"].
    Output schema: TOOL_SCHEMAS["get_status"]["output"].
    """
    return service.get_status(target_repo_dir=target_repo_dir).to_dict()


def main() -> None:
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
