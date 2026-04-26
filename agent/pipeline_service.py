#!/usr/bin/env python3
"""Service layer for Strategy2Code pipeline and repository operations."""

from __future__ import annotations

import ast
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class ServicePaths:
    """Resolved filesystem locations used by the service."""

    repo_root: Path
    pipeline_root: Path
    pipeline_codes_dir: Path
    outputs_root: Path
    eval_prompts_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "ServicePaths":
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
    """Single subprocess invocation with captured logs."""

    name: str
    cmd: List[str]
    return_code: int
    stdout_log_path: str
    stderr_log_path: str
    elapsed_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cmd": " ".join(shlex.quote(arg) for arg in self.cmd),
            "return_code": self.return_code,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "stdout_log_path": self.stdout_log_path,
            "stderr_log_path": self.stderr_log_path,
        }


@dataclass
class ToolResult:
    """Normalized payload returned by service methods."""

    action: str
    status: str
    message: str
    call_id: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "call_id": self.call_id,
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "commands": self.commands,
            "errors": self.errors,
        }


class Strategy2CodePipelineService:
    """Reusable service used by MCP tools."""

    def __init__(
        self,
        repo_root: Path,
        default_model: str = "o4-mini",
        artifact_root: Optional[Path] = None,
    ) -> None:
        self.paths = ServicePaths.from_repo_root(repo_root.resolve())
        self.default_model = default_model
        if artifact_root is not None:
            self.session_root = artifact_root.resolve() / now_str()
        else:
            self.session_root = repo_root / "agent" / "mcp_artifacts" / now_str()
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.call_idx = 0
        self.state: Dict[str, Any] = {
            "paper_name": None,
            "paper_json_path": None,
            "paper_json_cleaned_path": None,
            "output_dir": None,
            "output_repo_dir": None,
            "last_eval_result_path": None,
            "last_verify_status": "unknown",
        }

    def _next_call_id(self, action: str) -> str:
        self.call_idx += 1
        return f"call_{self.call_idx:03d}_{action}"

    def _save_result(self, result: ToolResult) -> None:
        result_path = self.session_root / f"{result.call_id}.json"
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_command(
        self,
        call_id: str,
        name: str,
        cmd: List[str],
        timeout_sec: int = 7200,
        cwd: Optional[Path] = None,
    ) -> CommandRecord:
        start = datetime.now()
        stdout_text = ""
        stderr_text = ""
        return_code = 0

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
            )
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            return_code = completed.returncode
        except subprocess.TimeoutExpired as timeout_error:
            stdout_text = timeout_error.stdout or ""
            stderr_text = (timeout_error.stderr or "") + "\n[ERROR] command timed out"
            return_code = 124
        except Exception as exec_error:  # pragma: no cover - defensive fallback
            stderr_text = f"[ERROR] unexpected command failure: {exec_error}"
            return_code = 1

        elapsed = (datetime.now() - start).total_seconds()
        stdout_log = self.session_root / f"{call_id}_{name}.stdout.log"
        stderr_log = self.session_root / f"{call_id}_{name}.stderr.log"
        stdout_log.write_text(stdout_text, encoding="utf-8")
        stderr_log.write_text(stderr_text, encoding="utf-8")

        return CommandRecord(
            name=name,
            cmd=cmd,
            return_code=return_code,
            stdout_log_path=str(stdout_log),
            stderr_log_path=str(stderr_log),
            elapsed_sec=elapsed,
        )

    def _resolve_output_dir(self, paper_name: str, output_dir: str) -> Path:
        if output_dir:
            return Path(output_dir).expanduser().resolve()
        return (self.paths.outputs_root / paper_name).resolve()

    def _resolve_output_repo_dir(self, paper_name: str, output_repo_dir: str) -> Path:
        if output_repo_dir:
            return Path(output_repo_dir).expanduser().resolve()
        return (self.paths.outputs_root / f"{paper_name}_repo").resolve()

    def _resolve_paper_json_path(self, paper_json_path: str) -> Optional[Path]:
        if not paper_json_path:
            state_path = self.state.get("paper_json_path") or self.state.get("paper_json_cleaned_path")
            if not state_path:
                return None
            return Path(str(state_path)).expanduser().resolve()
        return Path(paper_json_path).expanduser().resolve()

    def run_pipeline(
        self,
        paper_name: str,
        paper_json_path: str,
        gpt_version: str = "",
        output_dir: str = "",
        output_repo_dir: str = "",
        paper_json_cleaned_path: str = "",
    ) -> ToolResult:
        """Run preprocess -> planning -> extract_config -> analyzing -> coding."""
        call_id = self._next_call_id("run_pipeline")
        if not paper_name or not paper_json_path:
            result = ToolResult(
                action="run_pipeline",
                status="error",
                message="Missing required fields: `paper_name` and `paper_json_path` are required.",
                call_id=call_id,
                errors=["missing_required_fields"],
            )
            self._save_result(result)
            return result

        raw_json = Path(paper_json_path).expanduser().resolve()
        if not raw_json.exists():
            result = ToolResult(
                action="run_pipeline",
                status="error",
                message=f"Paper JSON file not found: {raw_json}",
                call_id=call_id,
                errors=["paper_json_not_found"],
            )
            self._save_result(result)
            return result

        cleaned_json = (
            Path(paper_json_cleaned_path).expanduser().resolve()
            if paper_json_cleaned_path
            else raw_json.with_name(f"{raw_json.stem}_cleaned.json")
        )

        resolved_output_dir = self._resolve_output_dir(paper_name=paper_name, output_dir=output_dir)
        resolved_output_repo_dir = self._resolve_output_repo_dir(
            paper_name=paper_name,
            output_repo_dir=output_repo_dir,
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        resolved_output_repo_dir.mkdir(parents=True, exist_ok=True)

        model_name = gpt_version or self.default_model
        py_exec = sys.executable
        codes_dir = self.paths.pipeline_codes_dir

        steps: List[tuple[str, List[str]]] = [
            (
                "preprocess",
                [
                    py_exec,
                    str(codes_dir / "0_pdf_process.py"),
                    "--input_json_path",
                    str(raw_json),
                    "--output_json_path",
                    str(cleaned_json),
                ],
            ),
            (
                "planning",
                [
                    py_exec,
                    str(codes_dir / "1_planning.py"),
                    "--paper_name",
                    paper_name,
                    "--gpt_version",
                    model_name,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(resolved_output_dir),
                ],
            ),
            (
                "extract_config",
                [
                    py_exec,
                    str(codes_dir / "1.1_extract_config.py"),
                    "--paper_name",
                    paper_name,
                    "--output_dir",
                    str(resolved_output_dir),
                ],
            ),
            (
                "analyzing",
                [
                    py_exec,
                    str(codes_dir / "2_analyzing.py"),
                    "--paper_name",
                    paper_name,
                    "--gpt_version",
                    model_name,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(resolved_output_dir),
                ],
            ),
            (
                "coding",
                [
                    py_exec,
                    str(codes_dir / "3_coding.py"),
                    "--paper_name",
                    paper_name,
                    "--gpt_version",
                    model_name,
                    "--pdf_json_path",
                    str(cleaned_json),
                    "--output_dir",
                    str(resolved_output_dir),
                    "--output_repo_dir",
                    str(resolved_output_repo_dir),
                ],
            ),
        ]

        command_records: List[CommandRecord] = []
        for step_name, command in steps:
            record = self._run_command(
                call_id=call_id,
                name=step_name,
                cmd=command,
                cwd=self.paths.repo_root,
            )
            command_records.append(record)
            if record.return_code != 0:
                result = ToolResult(
                    action="run_pipeline",
                    status="error",
                    message=f"Pipeline failed at step `{step_name}`.",
                    call_id=call_id,
                    artifacts={
                        "paper_name": paper_name,
                        "paper_json_path": str(raw_json),
                        "paper_json_cleaned_path": str(cleaned_json),
                        "output_dir": str(resolved_output_dir),
                        "output_repo_dir": str(resolved_output_repo_dir),
                    },
                    commands=[rec.to_dict() for rec in command_records],
                    errors=[f"step_failed:{step_name}"],
                )
                self._save_result(result)
                return result

        config_src = resolved_output_dir / "planning_config.yaml"
        if config_src.exists():
            shutil.copy2(config_src, resolved_output_repo_dir / "config.yaml")

        self.state.update(
            {
                "paper_name": paper_name,
                "paper_json_path": str(raw_json),
                "paper_json_cleaned_path": str(cleaned_json),
                "output_dir": str(resolved_output_dir),
                "output_repo_dir": str(resolved_output_repo_dir),
            }
        )

        result = ToolResult(
            action="run_pipeline",
            status="success",
            message="Pipeline completed successfully.",
            call_id=call_id,
            artifacts={
                "paper_name": paper_name,
                "paper_json_path": str(raw_json),
                "paper_json_cleaned_path": str(cleaned_json),
                "output_dir": str(resolved_output_dir),
                "output_repo_dir": str(resolved_output_repo_dir),
                "planning_config": str(config_src),
            },
            commands=[rec.to_dict() for rec in command_records],
        )
        self._save_result(result)
        return result

    def evaluate_strategy(
        self,
        paper_name: str = "",
        paper_json_path: str = "",
        output_dir: str = "",
        output_repo_dir: str = "",
        eval_type: str = "ref_free",
        generated_n: int = 4,
        gpt_version: str = "",
        gold_repo_dir: str = "",
        eval_result_dir: str = "",
    ) -> ToolResult:
        """Run existing eval.py over generated repository."""
        call_id = self._next_call_id("evaluate_strategy")
        resolved_paper_name = paper_name or str(self.state.get("paper_name") or "")
        resolved_paper_json = self._resolve_paper_json_path(paper_json_path)

        if not resolved_paper_name or not resolved_paper_json:
            result = ToolResult(
                action="evaluate_strategy",
                status="error",
                message="Missing required fields for evaluation. Provide `paper_name` and `paper_json_path`, or run pipeline first.",
                call_id=call_id,
                errors=["missing_required_fields"],
            )
            self._save_result(result)
            return result

        if not resolved_paper_json.exists():
            result = ToolResult(
                action="evaluate_strategy",
                status="error",
                message=f"Paper JSON file not found: {resolved_paper_json}",
                call_id=call_id,
                errors=["paper_json_not_found"],
            )
            self._save_result(result)
            return result

        resolved_output_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else Path(str(self.state.get("output_dir") or self._resolve_output_dir(resolved_paper_name, "")))
        )
        resolved_output_repo_dir = (
            Path(output_repo_dir).expanduser().resolve()
            if output_repo_dir
            else Path(
                str(
                    self.state.get("output_repo_dir")
                    or self._resolve_output_repo_dir(resolved_paper_name, "")
                )
            )
        )

        if not resolved_output_repo_dir.exists():
            result = ToolResult(
                action="evaluate_strategy",
                status="error",
                message=f"Target repository not found: {resolved_output_repo_dir}",
                call_id=call_id,
                errors=["target_repo_not_found"],
            )
            self._save_result(result)
            return result

        resolved_eval_result_dir = (
            Path(eval_result_dir).expanduser().resolve()
            if eval_result_dir
            else (resolved_output_dir / "eval_results").resolve()
        )
        resolved_eval_result_dir.mkdir(parents=True, exist_ok=True)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        model_name = gpt_version or self.default_model
        eval_type_value = eval_type if eval_type in {"ref_free", "ref_based"} else "ref_free"
        generated_n_value = max(1, int(generated_n))

        cmd = [
            sys.executable,
            str(self.paths.pipeline_codes_dir / "eval.py"),
            "--paper_name",
            resolved_paper_name,
            "--pdf_json_path",
            str(resolved_paper_json),
            "--data_dir",
            str(self.paths.eval_prompts_dir),
            "--output_dir",
            str(resolved_output_dir),
            "--target_repo_dir",
            str(resolved_output_repo_dir),
            "--eval_result_dir",
            str(resolved_eval_result_dir),
            "--eval_type",
            eval_type_value,
            "--generated_n",
            str(generated_n_value),
            "--gpt_version",
            model_name,
            "--papercoder",
        ]
        if gold_repo_dir:
            cmd.extend(["--gold_repo_dir", str(Path(gold_repo_dir).expanduser().resolve())])

        command_record = self._run_command(
            call_id=call_id,
            name="evaluate",
            cmd=cmd,
            cwd=self.paths.repo_root,
        )
        if command_record.return_code != 0:
            result = ToolResult(
                action="evaluate_strategy",
                status="error",
                message="Evaluation command failed.",
                call_id=call_id,
                artifacts={"eval_result_dir": str(resolved_eval_result_dir)},
                commands=[command_record.to_dict()],
                errors=["eval_script_failed"],
            )
            self._save_result(result)
            return result

        pattern = f"{resolved_paper_name}_eval_{eval_type_value}_{model_name}_*.json"
        result_files = sorted(
            resolved_eval_result_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
        )
        latest_result = result_files[-1] if result_files else None
        if latest_result is None:
            result = ToolResult(
                action="evaluate_strategy",
                status="error",
                message="Evaluation finished but no result file was produced.",
                call_id=call_id,
                artifacts={"eval_result_dir": str(resolved_eval_result_dir)},
                commands=[command_record.to_dict()],
                errors=["eval_result_not_found"],
            )
            self._save_result(result)
            return result

        parsed = safe_json_load(latest_result)
        eval_result = parsed.get("eval_result", {})
        metrics = {
            "score": eval_result.get("score"),
            "valid_n": eval_result.get("valid_n"),
            "score_list": eval_result.get("scroe_lst", []),
        }
        self.state["last_eval_result_path"] = str(latest_result)
        self.state["paper_name"] = resolved_paper_name
        self.state["paper_json_path"] = str(resolved_paper_json)
        self.state["output_dir"] = str(resolved_output_dir)
        self.state["output_repo_dir"] = str(resolved_output_repo_dir)

        result = ToolResult(
            action="evaluate_strategy",
            status="success",
            message="Evaluation completed.",
            call_id=call_id,
            artifacts={
                "eval_result_dir": str(resolved_eval_result_dir),
                "latest_eval_result": str(latest_result),
            },
            metrics=metrics,
            commands=[command_record.to_dict()],
        )
        self._save_result(result)
        return result

    def verify_repo(self, target_repo_dir: str = "", timeout_sec: int = 600) -> ToolResult:
        """Run generated repo main.py as a health check."""
        call_id = self._next_call_id("verify_repo")
        repo_dir_raw = target_repo_dir or str(self.state.get("output_repo_dir") or "")
        if not repo_dir_raw:
            result = ToolResult(
                action="verify_repo",
                status="error",
                message="Missing `target_repo_dir`. Provide it explicitly or run pipeline first.",
                call_id=call_id,
                errors=["missing_target_repo_dir"],
            )
            self._save_result(result)
            return result

        repo_dir = Path(repo_dir_raw).expanduser().resolve()
        main_file = repo_dir / "main.py"
        if not main_file.exists():
            result = ToolResult(
                action="verify_repo",
                status="error",
                message=f"Entrypoint not found: {main_file}",
                call_id=call_id,
                errors=["main_py_not_found"],
            )
            self._save_result(result)
            return result

        command_record = self._run_command(
            call_id=call_id,
            name="verify_run_main",
            cmd=[sys.executable, str(main_file)],
            cwd=repo_dir,
            timeout_sec=max(1, int(timeout_sec)),
        )
        passed = command_record.return_code == 0
        verify_status = "passed" if passed else "failed"
        self.state["last_verify_status"] = verify_status
        self.state["output_repo_dir"] = str(repo_dir)

        result = ToolResult(
            action="verify_repo",
            status="success" if passed else "error",
            message=(
                "Verification passed (main.py executed successfully)."
                if passed
                else "Verification failed (main.py execution failed)."
            ),
            call_id=call_id,
            artifacts={"target_repo_dir": str(repo_dir), "main_py": str(main_file)},
            metrics={"verify_status": verify_status},
            commands=[command_record.to_dict()],
            errors=[] if passed else ["run_main_failed"],
        )
        self._save_result(result)
        return result

    def _python_api_index(self, file_path: Path) -> Dict[str, List[str]]:
        if not file_path.exists():
            return {"classes": [], "functions": []}
        try:
            syntax_tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            return {"classes": [], "functions": []}
        classes: List[str] = []
        functions: List[str] = []
        for node in syntax_tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return {"classes": classes, "functions": functions}

    def _build_explanation(self, repo_dir: Path) -> Dict[str, Any]:
        files = [
            repo_dir / "main.py",
            repo_dir / "data" / "loader_yahoo.py",
            repo_dir / "strategy" / "strategy.py",
            repo_dir / "backtest" / "backtester.py",
            repo_dir / "backtest" / "metrics.py",
            repo_dir / "config.yaml",
        ]
        existing = [path for path in files if path.exists()]
        api_summary = {}
        for path in existing:
            if path.suffix == ".py":
                api_summary[str(path)] = self._python_api_index(path)

        verify_status = self.state.get("last_verify_status") or "unknown"
        eval_result_path = str(self.state.get("last_eval_result_path") or "")
        eval_payload = safe_json_load(Path(eval_result_path)) if eval_result_path else {}
        eval_score = eval_payload.get("eval_result", {}).get("score")
        run_cmd = f"cd {shlex.quote(str(repo_dir))} && {shlex.quote(sys.executable)} main.py"

        return {
            "answer": (
                "Purpose: load Yahoo Finance data, generate strategy signals, run a backtest, and report metrics.\n"
                f"How to run: `{run_cmd}`.\n"
                f"Validation: run_status={verify_status}; eval_score={eval_score if eval_score is not None else 'N/A'}."
            ),
            "artifacts": {
                "target_repo_dir": str(repo_dir),
                "existing_files": [str(path) for path in existing],
                "latest_eval_result": eval_result_path,
            },
            "metrics": {
                "verify_status": verify_status,
                "eval_score": eval_score,
                "api_summary": api_summary,
            },
        }

    def explain_repo(self, target_repo_dir: str = "") -> ToolResult:
        """Summarize generated repository structure and run status."""
        call_id = self._next_call_id("explain_repo")
        repo_dir_raw = target_repo_dir or str(self.state.get("output_repo_dir") or "")
        if not repo_dir_raw:
            result = ToolResult(
                action="explain_repo",
                status="error",
                message="Missing `target_repo_dir`. Provide it explicitly or run pipeline first.",
                call_id=call_id,
                errors=["missing_target_repo_dir"],
            )
            self._save_result(result)
            return result

        repo_dir = Path(repo_dir_raw).expanduser().resolve()
        if not repo_dir.exists():
            result = ToolResult(
                action="explain_repo",
                status="error",
                message=f"Repository does not exist: {repo_dir}",
                call_id=call_id,
                errors=["target_repo_not_found"],
            )
            self._save_result(result)
            return result

        payload = self._build_explanation(repo_dir)
        self.state["output_repo_dir"] = str(repo_dir)
        result = ToolResult(
            action="explain_repo",
            status="success",
            message=payload["answer"],
            call_id=call_id,
            artifacts=payload["artifacts"],
            metrics=payload["metrics"],
        )
        self._save_result(result)
        return result

    def get_status(self, target_repo_dir: str = "") -> ToolResult:
        """Return condensed verification + evaluation status."""
        call_id = self._next_call_id("get_status")
        repo_dir_raw = target_repo_dir or str(self.state.get("output_repo_dir") or "")
        if not repo_dir_raw:
            result = ToolResult(
                action="get_status",
                status="error",
                message="No active repository context. Run `run_pipeline` first or pass `target_repo_dir`.",
                call_id=call_id,
                errors=["missing_target_repo_dir"],
            )
            self._save_result(result)
            return result

        repo_dir = Path(repo_dir_raw).expanduser().resolve()
        if not repo_dir.exists():
            result = ToolResult(
                action="get_status",
                status="error",
                message=f"Repository does not exist: {repo_dir}",
                call_id=call_id,
                errors=["target_repo_not_found"],
            )
            self._save_result(result)
            return result

        explain_payload = self._build_explanation(repo_dir)
        verify_status = explain_payload["metrics"].get("verify_status", "unknown")
        eval_score = explain_payload["metrics"].get("eval_score")
        summary = (
            "Current verification status: "
            f"{verify_status}; current evaluation score: {eval_score if eval_score is not None else 'N/A'}."
        )
        self.state["output_repo_dir"] = str(repo_dir)
        result = ToolResult(
            action="get_status",
            status="success",
            message=summary,
            call_id=call_id,
            artifacts=explain_payload["artifacts"],
            metrics=explain_payload["metrics"],
        )
        self._save_result(result)
        return result

    def snapshot(self) -> Dict[str, Any]:
        """Read-only state snapshot for debugging and tests."""
        return {
            "session_root": str(self.session_root),
            "state": dict(self.state),
        }
