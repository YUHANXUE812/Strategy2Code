# Strategy2Code Chat Agent

`agent/chat_agent.py` provides a chat-style entrypoint to orchestrate `pipeline / eval / verify / explain`.
It now uses a `Hybrid Plan-and-Execute + Tool Feedback Loop` flow:

1. plan from user intent (explicit command + natural language + context),
2. execute step by step,
3. if a step fails due missing dependencies/context, auto-replan and retry once.

## 1. Start interactive mode

```bash
python agent/chat_agent.py --interactive
```

## 2. Single-turn calls

```bash
python agent/chat_agent.py --message '/run {"paper_name":"ADDPG","paper_json_path":"Strategy2Code codes/examples/ADDPG.json"}'
python agent/chat_agent.py --message '/eval {"paper_name":"ADDPG"}'
python agent/chat_agent.py --message '/verify {"target_repo_dir":"Strategy2Code codes/outputs/ADDPG_repo"}'
python agent/chat_agent.py --message '/explain {"target_repo_dir":"Strategy2Code codes/outputs/ADDPG_repo"}'
python agent/chat_agent.py --message '/status'
```

## 3. Natural-language input

You can also use natural-language requests and include JSON or `key=value` pairs:

```text
Run the pipeline for this paper {"paper_name":"ADDPG","paper_json_path":"Strategy2Code codes/examples/ADDPG.json"}
Evaluate the generated code paper_name=ADDPG
What does this code do, how do I run it, and did it pass verification?
```

If a `run pipeline` request also contains verification/evaluation/explanation intent (for example: "did it pass", "evaluate", "explain"), the planner will auto-build a chain after successful code generation:

1. run `verify` immediately
2. if verification passed and evaluation intent is present, run `eval`
3. run `explain` and return one consolidated answer

You can also force this behavior with `auto_followup=true`, and force evaluation with `with_eval=true`.

## 4. Standardized outputs

Each turn is saved into the session directory:

- `agent/sessions/<timestamp>/turn_XXX.json`
- `agent/sessions/<timestamp>/turn_XXX_<step>.stdout.log`
- `agent/sessions/<timestamp>/turn_XXX_<step>.stderr.log`

`turn_XXX.json` includes standard fields:

- `request`
- `action`
- `status`
- `answer`
- `artifacts`
- `metrics`
- `commands`
- `errors`

For planned execution, `artifacts` also includes:

- `plan`: strategy + planned steps
- `step_results`: step-by-step execution trajectory
