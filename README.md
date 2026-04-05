# Strategy2Code

Strategy2Code now runs as an MCP server.  
The external MCP host (for example Cline in VS Code) is responsible for orchestration.

## Architecture Overview

### MCP Entrypoint
- `agent/mcp_server.py`
- Registers MCP tools and starts transport (`stdio` by default).

### Tool Layer
- `run_pipeline`
- `evaluate_strategy`
- `verify_repo`
- `explain_repo`
- `get_status`

### Internal Pipeline Layer (reused)
- `agent/pipeline_service.py` (service wrapper)
- Existing scripts are still executed:
  - `Strategy2Code codes/codes/0_pdf_process.py`
  - `Strategy2Code codes/codes/1_planning.py`
  - `Strategy2Code codes/codes/1.1_extract_config.py`
  - `Strategy2Code codes/codes/2_analyzing.py`
  - `Strategy2Code codes/codes/3_coding.py`
  - `Strategy2Code codes/codes/eval.py`

### Logging and Artifacts
- MCP tool-call artifacts: `agent/mcp_artifacts/<timestamp>/`
- Per call:
  - `call_XXX_<tool>.json`
  - `call_XXX_<step>.stdout.log`
  - `call_XXX_<step>.stderr.log`

## Environment Requirements

- Python 3.10+
- `OPENAI_API_KEY` set in environment
- Install dependencies:

```bash
pip install -r "Strategy2Code codes/requirements.txt"
pip install "mcp>=1.0.0"
```

## Run MCP Server

```bash
export OPENAI_API_KEY="sk-..."
python agent/mcp_server.py
```

Optional environment variables:
- `STRATEGY2CODE_REPO_ROOT` (default: repository root)
- `STRATEGY2CODE_MCP_ARTIFACT_ROOT` (default: `agent/mcp_artifacts`)
- `STRATEGY2CODE_DEFAULT_MODEL` (default: `o4-mini`)
- `MCP_TRANSPORT` (default: `stdio`)

## MCP Tools

All tools return the same output object shape:
- `timestamp` `string`
- `call_id` `string`
- `action` `string`
- `status` `success|error`
- `message` `string`
- `artifacts` `object`
- `metrics` `object`
- `commands` `array`
- `errors` `array[string]`

### `run_pipeline`
- Input:
  - `paper_name` `string` (required)
  - `paper_json_path` `string` (required)
  - `gpt_version` `string` (optional)
  - `output_dir` `string` (optional)
  - `output_repo_dir` `string` (optional)
  - `paper_json_cleaned_path` `string` (optional)

### `evaluate_strategy`
- Input:
  - `paper_name` `string` (optional if pipeline state exists)
  - `paper_json_path` `string` (optional if pipeline state exists)
  - `output_dir` `string` (optional)
  - `output_repo_dir` `string` (optional)
  - `eval_type` `ref_free|ref_based` (optional, default `ref_free`)
  - `generated_n` `int` (optional, default `4`)
  - `gpt_version` `string` (optional)
  - `gold_repo_dir` `string` (optional)
  - `eval_result_dir` `string` (optional)

### `verify_repo`
- Input:
  - `target_repo_dir` `string` (optional if pipeline state exists)
  - `timeout_sec` `int` (optional, default `600`)

### `explain_repo`
- Input:
  - `target_repo_dir` `string` (optional if pipeline state exists)

### `get_status`
- Input:
  - `target_repo_dir` `string` (optional if pipeline state exists)

## Cline (VS Code) Integration

### 1. Open VS Code settings JSON
- Command Palette -> `Preferences: Open Settings (JSON)`

### 2. Add MCP server config

```json
{
  "cline.mcpServers": {
    "strategy2code": {
      "command": "python",
      "args": ["/home/xyh812/Strategy2Code/agent/mcp_server.py"],
      "cwd": "/home/xyh812/Strategy2Code",
      "env": {
        "OPENAI_API_KEY": "${env:OPENAI_API_KEY}",
        "STRATEGY2CODE_REPO_ROOT": "/home/xyh812/Strategy2Code",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

### 3. Reload VS Code window
- `Developer: Reload Window`

### 4. Verify in Cline
- Open Cline MCP servers/tools list.
- Confirm `strategy2code` is connected.
- Confirm tools:
  - `run_pipeline`
  - `evaluate_strategy`
  - `verify_repo`
  - `explain_repo`
  - `get_status`

## Removed Components

- Old orchestration loop removed from `agent/chat_agent.py`.
- Interactive REPL chat behavior removed.
- LLM decision-loop planning/replanning removed from local runtime.
- `agent/chat_agent.py` is now a deprecation shim that exits and points to MCP usage.
