#!/usr/bin/env python3
"""Chat-style agent for Strategy2Code pipeline orchestration.

This agent supports:
1) running the Strategy2Code pipeline,
2) evaluating generated code, and
3) answering code explanation / execution / verification-status questions.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return {}
    return {}


def parse_kv_pairs(text: str) -> Dict[str, str]:
    pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)"
    parsed: Dict[str, str] = {}
    for key, value in re.findall(pattern, text):
        cleaned = value.strip().strip("'").strip('"')
        parsed[key] = cleaned
    return parsed


def guess_action(message: str) -> str:
    lower = message.lower()
    stripped = message.strip()

    if stripped.startswith("/"):
        cmd = stripped.split(maxsplit=1)[0]
        mapping = {
            "/run": "run_pipeline",
            "/eval": "evaluate",
            "/verify": "verify",
            "/explain": "explain",
            "/status": "status",
            "/help": "help",
        }
        return mapping.get(cmd, "help")

    run_keys = ["run pipeline", "generate code", "reproduce", "build repo", "run the pipeline"]
    eval_keys = ["evaluate", "score", "grade", "assess"]
    verify_keys = ["verify", "validation", "pass verification", "does it run", "did it pass"]
    explain_keys = ["explain", "what does this code do", "how to run", "describe this code"]

    if any(k in lower for k in run_keys):
        return "run_pipeline"
    if any(k in lower for k in eval_keys):
        return "evaluate"
    if any(k in lower for k in verify_keys):
        return "status"
    if any(k in lower for k in explain_keys):
        return "explain"
    return "help"


@dataclass
class AgentPaths:
    repo_root: Path
    pipeline_root: Path
    pipeline_codes_dir: Path
    outputs_root: Path
    eval_prompts_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "AgentPaths":
        pipeline_root = repo_root / "Strategy2Code codes"
        return cls(
            repo_root=repo_root,
            pipeline_root=pipeline_root,
            pipeline_codes_dir=pipeline_root / "codes",
            outputs_root=pipeline_root / "outputs",
            eval_prompts_dir=repo_root / "Data" / "data (paper2code)",
        )


@dataclass
class CommandRecord:
    name: str
    cmd: List[str]
    return_code: int
    stdout_log_path: str
    stderr_log_path: str
    elapsed_sec: float


@dataclass
class AgentResult:
    request: str
    action: str
    status: str
    answer: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "request": self.request,
            "action": self.action,
            "status": self.status,
            "answer": self.answer,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "commands": self.commands,
            "errors": self.errors,
        }


class Strategy2CodeChatAgent:
    """Chat-style orchestrator for Strategy2Code pipeline."""

    def __init__(
        self,
        repo_root: Path,
        default_model: str = "o4-mini",
        session_root: Optional[Path] = None,
    ) -> None:
        self.paths = AgentPaths.from_repo_root(repo_root.resolve())
        self.default_model = default_model
        self.session_root = session_root or (repo_root / "agent" / "sessions" / now_str())
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.turn_idx = 0
        self.state: Dict[str, Any] = {
            "paper_name": None,
            "paper_json_path": None,
            "paper_json_cleaned_path": None,
            "output_dir": None,
            "output_repo_dir": None,
            "last_eval_result_path": None,
            "last_verify_status": "unknown",
        }

    def _run_command(
        self,
        name: str,
        cmd: List[str],
        timeout_sec: int = 7200,
        cwd: Optional[Path] = None,
    ) -> CommandRecord:
        start = datetime.now()
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        elapsed = (datetime.now() - start).total_seconds()

        turn_prefix = f"turn_{self.turn_idx:03d}"
        stdout_log = self.session_root / f"{turn_prefix}_{name}.stdout.log"
        stderr_log = self.session_root / f"{turn_prefix}_{name}.stderr.log"
        stdout_log.write_text(completed.stdout or "", encoding="utf-8")
        stderr_log.write_text(completed.stderr or "", encoding="utf-8")

        return CommandRecord(
            name=name,
            cmd=cmd,
            return_code=completed.returncode,
            stdout_log_path=str(stdout_log),
            stderr_log_path=str(stderr_log),
            elapsed_sec=elapsed,
        )

    def _collect_payload(self, message: str) -> Dict[str, Any]:
        payload = extract_json_object(message)
        kvs = parse_kv_pairs(message)
        if kvs:
            payload = {**kvs, **payload}
        return payload

    def _resolve_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.state)
        merged.update(payload)

        paper_name = merged.get("paper_name")
        if paper_name:
            merged["output_dir"] = merged.get("output_dir") or str(self.paths.outputs_root / paper_name)
            merged["output_repo_dir"] = merged.get("output_repo_dir") or str(
                self.paths.outputs_root / f"{paper_name}_repo"
            )
        return merged

    def _save_turn_result(self, result: AgentResult) -> None:
        turn_path = self.session_root / f"turn_{self.turn_idx:03d}.json"
        turn_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _command_dict(self, record: CommandRecord) -> Dict[str, Any]:
        return {
            "name": record.name,
            "cmd": " ".join(shlex.quote(x) for x in record.cmd),
            "return_code": record.return_code,
            "elapsed_sec": round(record.elapsed_sec, 3),
            "stdout_log_path": record.stdout_log_path,
            "stderr_log_path": record.stderr_log_path,
        }

    def _run_pipeline(self, request: str, payload: Dict[str, Any]) -> AgentResult:
        ctx = self._resolve_context(payload)

        paper_name = ctx.get("paper_name")
        paper_json_path = ctx.get("paper_json_path")
        if not paper_name or not paper_json_path:
            return AgentResult(
                request=request,
                action="run_pipeline",
                status="error",
                answer="Missing `paper_name` or `paper_json_path`; cannot run the pipeline.",
                errors=["missing_required_fields"],
            )

        paper_json = Path(str(paper_json_path)).expanduser().resolve()
        if not paper_json.exists():
            return AgentResult(
                request=request,
                action="run_pipeline",
                status="error",
                answer=f"Paper JSON file not found: {paper_json}",
                errors=["paper_json_not_found"],
            )

        cleaned = ctx.get("paper_json_cleaned_path")
        if cleaned:
            cleaned_json = Path(str(cleaned)).expanduser().resolve()
        else:
            cleaned_json = paper_json.with_name(f"{paper_json.stem}_cleaned.json")

        output_dir = Path(str(ctx["output_dir"])).expanduser().resolve()
        output_repo_dir = Path(str(ctx["output_repo_dir"])).expanduser().resolve()
        gpt_version = str(ctx.get("gpt_version") or self.default_model)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_repo_dir.mkdir(parents=True, exist_ok=True)

        codes_dir = self.paths.pipeline_codes_dir
        py = sys.executable

        commands: List[CommandRecord] = []

        steps: List[Tuple[str, List[str]]] = [
            (
                "preprocess",
                [
                    py,
                    str(codes_dir / "0_pdf_process.py"),
                    "--input_json_path",
                    str(paper_json),
                    "--output_json_path",
                    str(cleaned_json),
                ],
            ),
            (
                "planning",
                [
                    py,
                    str(codes_dir / "1_planning.py"),
                    "--paper_name",
                    str(paper_name),
                    "--gpt_version",
                    gpt_version,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(output_dir),
                ],
            ),
            (
                "extract_config",
                [
                    py,
                    str(codes_dir / "1.1_extract_config.py"),
                    "--paper_name",
                    str(paper_name),
                    "--output_dir",
                    str(output_dir),
                ],
            ),
            (
                "analyzing",
                [
                    py,
                    str(codes_dir / "2_analyzing.py"),
                    "--paper_name",
                    str(paper_name),
                    "--gpt_version",
                    gpt_version,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(output_dir),
                ],
            ),
            (
                "coding",
                [
                    py,
                    str(codes_dir / "3_coding.py"),
                    "--paper_name",
                    str(paper_name),
                    "--gpt_version",
                    gpt_version,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(output_dir),
                    "--output_repo_dir",
                    str(output_repo_dir),
                ],
            ),
        ]

        for name, cmd in steps:
            rec = self._run_command(name=name, cmd=cmd, cwd=self.paths.repo_root)
            commands.append(rec)
            if rec.return_code != 0:
                return AgentResult(
                    request=request,
                    action="run_pipeline",
                    status="error",
                    answer=f"Pipeline failed at step `{name}`.",
                    artifacts={
                        "paper_name": paper_name,
                        "paper_json_path": str(paper_json),
                        "paper_json_cleaned_path": str(cleaned_json),
                        "output_dir": str(output_dir),
                        "output_repo_dir": str(output_repo_dir),
                    },
                    commands=[self._command_dict(c) for c in commands],
                    errors=[f"step_failed:{name}"],
                )

        config_src = output_dir / "planning_config.yaml"
        if config_src.exists():
            shutil.copy2(config_src, output_repo_dir / "config.yaml")

        self.state.update(
            {
                "paper_name": paper_name,
                "paper_json_path": str(paper_json),
                "paper_json_cleaned_path": str(cleaned_json),
                "output_dir": str(output_dir),
                "output_repo_dir": str(output_repo_dir),
            }
        )

        return AgentResult(
            request=request,
            action="run_pipeline",
            status="success",
            answer="Pipeline completed successfully. A standardized output repository was generated.",
            artifacts={
                "paper_name": paper_name,
                "paper_json_path": str(paper_json),
                "paper_json_cleaned_path": str(cleaned_json),
                "output_dir": str(output_dir),
                "output_repo_dir": str(output_repo_dir),
                "planning_config": str(config_src),
            },
            commands=[self._command_dict(c) for c in commands],
        )

    def _evaluate(self, request: str, payload: Dict[str, Any]) -> AgentResult:
        ctx = self._resolve_context(payload)
        paper_name = ctx.get("paper_name")
        paper_json_path = ctx.get("paper_json_path") or ctx.get("paper_json_cleaned_path")

        if not paper_name or not paper_json_path:
            return AgentResult(
                request=request,
                action="evaluate",
                status="error",
                answer="Missing `paper_name` or `paper_json_path`; cannot run evaluation.",
                errors=["missing_required_fields"],
            )

        output_dir = Path(str(ctx["output_dir"])).expanduser().resolve()
        target_repo_dir = Path(str(ctx["output_repo_dir"])).expanduser().resolve()
        eval_result_dir = Path(str(payload.get("eval_result_dir") or (output_dir / "eval_results"))).resolve()
        eval_result_dir.mkdir(parents=True, exist_ok=True)

        eval_type = str(payload.get("eval_type", "ref_free"))
        gpt_version = str(payload.get("gpt_version") or self.default_model)
        generated_n = int(payload.get("generated_n", 4))
        gold_repo_dir = str(payload.get("gold_repo_dir", "")).strip()

        py = sys.executable
        cmd = [
            py,
            str(self.paths.pipeline_codes_dir / "eval.py"),
            "--paper_name",
            str(paper_name),
            "--pdf_json_path",
            str(Path(str(paper_json_path)).expanduser().resolve()),
            "--data_dir",
            str(self.paths.eval_prompts_dir),
            "--output_dir",
            str(output_dir),
            "--target_repo_dir",
            str(target_repo_dir),
            "--eval_result_dir",
            str(eval_result_dir),
            "--eval_type",
            eval_type,
            "--generated_n",
            str(generated_n),
            "--gpt_version",
            gpt_version,
            "--papercoder",
        ]

        if gold_repo_dir:
            cmd.extend(["--gold_repo_dir", str(Path(gold_repo_dir).expanduser().resolve())])

        rec = self._run_command(name="evaluate", cmd=cmd, cwd=self.paths.repo_root)
        if rec.return_code != 0:
            return AgentResult(
                request=request,
                action="evaluate",
                status="error",
                answer="Code evaluation failed to execute.",
                artifacts={"eval_result_dir": str(eval_result_dir)},
                commands=[self._command_dict(rec)],
                errors=["eval_script_failed"],
            )

        pattern = f"{paper_name}_eval_{eval_type}_{gpt_version}_*.json"
        result_files = sorted(eval_result_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        latest = result_files[-1] if result_files else None

        metrics: Dict[str, Any] = {}
        if latest and latest.exists():
            parsed = safe_json_load(latest)
            eval_result = parsed.get("eval_result", {})
            metrics = {
                "score": eval_result.get("score"),
                "valid_n": eval_result.get("valid_n"),
                "score_list": eval_result.get("scroe_lst", []),
            }
            self.state["last_eval_result_path"] = str(latest)

        return AgentResult(
            request=request,
            action="evaluate",
            status="success",
            answer="Code evaluation completed.",
            artifacts={
                "eval_result_dir": str(eval_result_dir),
                "latest_eval_result": str(latest) if latest else "",
            },
            metrics=metrics,
            commands=[self._command_dict(rec)],
        )

    def _verify(self, request: str, payload: Dict[str, Any]) -> AgentResult:
        ctx = self._resolve_context(payload)
        repo_dir_raw = payload.get("target_repo_dir") or ctx.get("output_repo_dir")
        if not repo_dir_raw:
            return AgentResult(
                request=request,
                action="verify",
                status="error",
                answer="Missing `target_repo_dir` (or run pipeline first to populate context).",
                errors=["missing_target_repo_dir"],
            )

        repo_dir = Path(str(repo_dir_raw)).expanduser().resolve()
        main_file = repo_dir / "main.py"
        if not main_file.exists():
            return AgentResult(
                request=request,
                action="verify",
                status="error",
                answer=f"Executable entrypoint not found: {main_file}",
                errors=["main_py_not_found"],
            )

        timeout_sec = int(payload.get("timeout_sec", 600))
        cmd = [sys.executable, str(main_file)]
        rec = self._run_command(name="verify_run_main", cmd=cmd, cwd=repo_dir, timeout_sec=timeout_sec)

        passed = rec.return_code == 0
        verify_status = "passed" if passed else "failed"
        self.state["last_verify_status"] = verify_status

        return AgentResult(
            request=request,
            action="verify",
            status="success" if passed else "error",
            answer="Verification passed (main.py ran successfully)."
            if passed
            else "Verification failed (main.py execution failed).",
            artifacts={"target_repo_dir": str(repo_dir), "main_py": str(main_file)},
            metrics={"verify_status": verify_status},
            commands=[self._command_dict(rec)],
            errors=[] if passed else ["run_main_failed"],
        )

    def _python_api_index(self, file_path: Path) -> Dict[str, List[str]]:
        if not file_path.exists():
            return {"classes": [], "functions": []}
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except Exception:
            return {"classes": [], "functions": []}
        classes: List[str] = []
        functions: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return {"classes": classes, "functions": functions}

    def _explain(self, request: str, payload: Dict[str, Any]) -> AgentResult:
        ctx = self._resolve_context(payload)
        repo_dir_raw = payload.get("target_repo_dir") or ctx.get("output_repo_dir")
        if not repo_dir_raw:
            return AgentResult(
                request=request,
                action="explain",
                status="error",
                answer="No target code directory available to explain. Run pipeline first or provide `target_repo_dir`.",
                errors=["missing_target_repo_dir"],
            )

        repo_dir = Path(str(repo_dir_raw)).expanduser().resolve()
        files = [
            repo_dir / "main.py",
            repo_dir / "data" / "loader_yahoo.py",
            repo_dir / "strategy" / "strategy.py",
            repo_dir / "backtest" / "backtester.py",
            repo_dir / "backtest" / "metrics.py",
            repo_dir / "config.yaml",
        ]
        existing = [f for f in files if f.exists()]

        api_summary = {}
        for f in existing:
            if f.suffix == ".py":
                api_summary[str(f)] = self._python_api_index(f)

        verify_status = self.state.get("last_verify_status") or "unknown"
        eval_result_path = self.state.get("last_eval_result_path")
        eval_info = safe_json_load(Path(eval_result_path)) if eval_result_path else {}
        eval_score = eval_info.get("eval_result", {}).get("score")

        how_to_run = f"cd {shlex.quote(str(repo_dir))} && {shlex.quote(sys.executable)} main.py"
        answer = (
            "Purpose: load Yahoo Finance data, generate strategy signals, run a backtest, and report metrics.\n"
            f"How to run: `{how_to_run}`.\n"
            f"Validation: run_status={verify_status}; eval_score={eval_score if eval_score is not None else 'N/A'}."
        )

        return AgentResult(
            request=request,
            action="explain",
            status="success",
            answer=answer,
            artifacts={
                "target_repo_dir": str(repo_dir),
                "existing_files": [str(p) for p in existing],
                "latest_eval_result": eval_result_path or "",
            },
            metrics={
                "verify_status": verify_status,
                "eval_score": eval_score,
                "api_summary": api_summary,
            },
        )

    def _status(self, request: str, payload: Dict[str, Any]) -> AgentResult:
        explain_result = self._explain(request, payload)
        if explain_result.status != "success":
            return explain_result

        verify_status = explain_result.metrics.get("verify_status", "unknown")
        eval_score = explain_result.metrics.get("eval_score", None)
        answer = f"Current verification status: {verify_status}; current evaluation score: {eval_score if eval_score is not None else 'N/A'}."

        explain_result.action = "status"
        explain_result.answer = answer
        return explain_result

    def _help(self, request: str) -> AgentResult:
        help_text = (
            "Available commands:\n"
            "1) /run {\"paper_name\":\"ADDPG\",\"paper_json_path\":\".../ADDPG.json\"}\n"
            "2) /eval {\"paper_name\":\"ADDPG\"}\n"
            "3) /verify {\"target_repo_dir\":\".../ADDPG_repo\"}\n"
            "4) /explain {\"target_repo_dir\":\".../ADDPG_repo\"}\n"
            "5) /status\n"
            "Natural-language English requests are also supported, with optional JSON or key=value parameters."
        )
        return AgentResult(
            request=request,
            action="help",
            status="success",
            answer=help_text,
            artifacts={"session_dir": str(self.session_root)},
        )

    def handle_message(self, message: str) -> AgentResult:
        self.turn_idx += 1
        action = guess_action(message)
        payload = self._collect_payload(message)

        if message.strip().startswith("/"):
            parts = message.strip().split(maxsplit=1)
            if len(parts) > 1:
                payload = {**parse_kv_pairs(parts[1]), **extract_json_object(parts[1]), **payload}

        if action == "run_pipeline":
            result = self._run_pipeline(message, payload)
        elif action == "evaluate":
            result = self._evaluate(message, payload)
        elif action == "verify":
            result = self._verify(message, payload)
        elif action == "explain":
            result = self._explain(message, payload)
        elif action == "status":
            result = self._status(message, payload)
        else:
            result = self._help(message)

        self._save_turn_result(result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat-style agent for Strategy2Code.")
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root path.",
    )
    parser.add_argument("--model", type=str, default="o4-mini", help="Default model name.")
    parser.add_argument("--session_dir", type=str, default="", help="Optional session output directory.")
    parser.add_argument("--message", type=str, default="", help="Single-turn message.")
    parser.add_argument("--interactive", action="store_true", help="Start REPL chat mode.")
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Print only JSON result without extra chat prefix.",
    )
    return parser


def print_result(result: AgentResult, json_only: bool) -> None:
    data = result.to_dict()
    if json_only:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print(f"[{data['action']}] {data['status']}")
    print(data["answer"])
    print(json.dumps(data, ensure_ascii=False, indent=2))


def repl(agent: Strategy2CodeChatAgent, json_only: bool) -> None:
    print("Strategy2Code Chat Agent started. Use /help for usage and type exit to quit.")
    while True:
        try:
            msg = input("You> ").strip()
        except EOFError:
            print()
            break
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            break
        result = agent.handle_message(msg)
        print_result(result, json_only=json_only)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    session_dir = Path(args.session_dir).expanduser().resolve() if args.session_dir else None
    agent = Strategy2CodeChatAgent(repo_root=repo_root, default_model=args.model, session_root=session_dir)

    if args.interactive or not args.message:
        repl(agent, json_only=args.json_only)
        return

    result = agent.handle_message(args.message)
    print_result(result, json_only=args.json_only)


if __name__ == "__main__":
    main()
