# Strategy2Code Chat Agent

`agent/chat_agent.py` provides a chat-style entrypoint to orchestrate `pipeline / eval / verify / explain`.

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
