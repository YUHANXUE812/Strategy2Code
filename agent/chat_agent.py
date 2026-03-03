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
from typing import Any, Callable, Dict, List, Optional, Tuple


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


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


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

    run_keys = [
        "run pipeline",
        "generate code",
        "reproduce",
        "build repo",
        "run the pipeline",
        "运行流程",
        "运行pipeline",
        "生成代码",
    ]
    eval_keys = ["evaluate", "score", "grade", "assess", "评估", "打分"]
    verify_keys = [
        "verify",
        "validation",
        "pass verification",
        "does it run",
        "did it pass",
        "验证",
        "通过验证",
    ]
    explain_keys = [
        "explain",
        "what does this code do",
        "how to run",
        "describe this code",
        "解释",
        "说明",
    ]
    status_keys = ["status", "current status", "状态", "当前状态"]
    help_keys = ["help", "usage", "怎么用", "帮助"]

    if any(k in lower for k in run_keys):
        return "run_pipeline"
    if any(k in lower for k in eval_keys):
        return "evaluate"
    if any(k in lower for k in verify_keys):
        return "verify"
    if any(k in lower for k in explain_keys):
        return "explain"
    if any(k in lower for k in status_keys):
        return "status"
    if any(k in lower for k in help_keys):
        return "help"
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


@dataclass
class PlanStep:
    action: str
    reason: str
    required: bool = True
    inserted_by_feedback: bool = False
    attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "required": self.required,
            "inserted_by_feedback": self.inserted_by_feedback,
            "attempt": self.attempt,
        }


@dataclass
class ExecutionPlan:
    strategy: str
    steps: List[PlanStep] = field(default_factory=list)

    def add_step(
        self,
        action: str,
        reason: str,
        required: bool = True,
        inserted_by_feedback: bool = False,
        attempt: int = 0,
    ) -> None:
        if any(s.action == action for s in self.steps):
            return
        self.steps.append(
            PlanStep(
                action=action,
                reason=reason,
                required=required,
                inserted_by_feedback=inserted_by_feedback,
                attempt=attempt,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "steps": [s.to_dict() for s in self.steps],
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

    def _contains_any(self, text: str, keys: List[str]) -> bool:
        return any(k in text for k in keys)

    def _pipeline_followup_plan(self, message: str, payload: Dict[str, Any]) -> Dict[str, bool]:
        lower = message.lower()
        eval_keys = ["evaluate", "evaluation", "eval", "score", "grade", "assess"]
        verify_keys = [
            "verify",
            "verification",
            "validate",
            "validation",
            "did it pass",
            "pass verification",
        ]
        explain_keys = [
            "explain",
            "explanation",
            "describe",
            "what does this code do",
            "how to run",
        ]

        wants_eval = self._contains_any(lower, eval_keys)
        wants_verify = self._contains_any(lower, verify_keys)
        wants_explain = self._contains_any(lower, explain_keys)

        auto_followup = truthy(payload.get("auto_followup")) or truthy(payload.get("run_followup"))
        force_eval = truthy(payload.get("force_eval")) or truthy(payload.get("with_eval"))
        skip_eval = truthy(payload.get("skip_eval"))

        enabled = auto_followup or wants_eval or wants_verify or wants_explain
        run_eval = (wants_eval or force_eval) and not skip_eval
        run_explain = enabled

        return {
            "enabled": enabled,
            "run_eval": run_eval,
            "run_explain": run_explain,
        }

    def _intent_flags(
        self,
        message: str,
        payload: Dict[str, Any],
        base_action: str,
    ) -> Dict[str, bool]:
        lower = message.lower()
        run_keys = [
            "run pipeline",
            "run the pipeline",
            "generate code",
            "reproduce",
            "build repo",
            "运行流程",
            "运行pipeline",
            "生成代码",
        ]
        eval_keys = ["evaluate", "evaluation", "eval", "score", "grade", "assess", "评估", "打分"]
        verify_keys = [
            "verify",
            "verification",
            "validate",
            "validation",
            "did it pass",
            "pass verification",
            "验证",
            "通过验证",
            "能跑",
        ]
        explain_keys = ["explain", "description", "what does this code do", "how to run", "解释", "说明"]
        status_keys = ["status", "current status", "状态", "当前状态"]
        help_keys = ["help", "usage", "怎么用", "帮助"]

        flags = {
            "run_pipeline": base_action == "run_pipeline" or self._contains_any(lower, run_keys),
            "evaluate": base_action == "evaluate" or self._contains_any(lower, eval_keys),
            "verify": base_action == "verify" or self._contains_any(lower, verify_keys),
            "explain": base_action == "explain" or self._contains_any(lower, explain_keys),
            "status": base_action == "status" or self._contains_any(lower, status_keys),
            "help": base_action == "help" or self._contains_any(lower, help_keys),
        }

        # Keep compatibility with legacy payload toggles.
        if truthy(payload.get("with_eval")) or truthy(payload.get("force_eval")):
            flags["evaluate"] = True
        if truthy(payload.get("auto_followup")) or truthy(payload.get("run_followup")):
            flags["verify"] = True
            flags["explain"] = True

        return flags

    def _build_execution_plan(
        self,
        message: str,
        payload: Dict[str, Any],
        base_action: str,
    ) -> ExecutionPlan:
        plan = ExecutionPlan(strategy="hybrid_plan_execute_feedback")
        explicit_command = message.strip().startswith("/")
        flags = self._intent_flags(message, payload, base_action)

        if explicit_command and base_action in {"evaluate", "verify", "explain", "status", "help"}:
            if base_action == "help":
                plan.add_step("help", reason="Explicit /help command.")
            else:
                plan.add_step(base_action, reason=f"Explicit {base_action} command.")
            return plan

        if flags["run_pipeline"]:
            plan.add_step("run_pipeline", reason="User asks to generate/reproduce code pipeline output.")
            followup = self._pipeline_followup_plan(message, payload)
            if followup.get("enabled", False):
                plan.add_step("verify", reason="Follow-up health check after pipeline execution.")
                if followup.get("run_eval", False):
                    plan.add_step("evaluate", reason="User asks for quality scoring after verification.")
                if followup.get("run_explain", False):
                    plan.add_step("explain", reason="User asks for explanation/how-to-run summary.")
            if flags["status"]:
                plan.add_step("status", reason="User explicitly asks for current status summary.")
            return plan

        if flags["verify"]:
            plan.add_step("verify", reason="User asks verification/execution-pass status.")
        if flags["evaluate"]:
            if not any(step.action == "verify" for step in plan.steps):
                plan.add_step(
                    "verify",
                    reason="Run verification first to provide stronger context before evaluation.",
                    required=False,
                )
            plan.add_step("evaluate", reason="User asks for evaluation/score.")
        if flags["explain"]:
            plan.add_step("explain", reason="User asks for code explanation/how-to-run.")
        if flags["status"]:
            plan.add_step("status", reason="User asks for current summarized status.")

        if not plan.steps:
            plan.add_step("help", reason="No clear executable intent detected.")
        return plan

    def _prepare_step_payload(self, payload: Dict[str, Any], action: str) -> Dict[str, Any]:
        merged = dict(payload)
        ctx = self._resolve_context(merged)

        if action in {"verify", "explain", "status"}:
            if not merged.get("target_repo_dir") and ctx.get("output_repo_dir"):
                merged["target_repo_dir"] = ctx["output_repo_dir"]

        if action == "evaluate":
            if not merged.get("paper_name") and ctx.get("paper_name"):
                merged["paper_name"] = ctx["paper_name"]
            if not merged.get("paper_json_path"):
                paper_json = ctx.get("paper_json_path") or ctx.get("paper_json_cleaned_path")
                if paper_json:
                    merged["paper_json_path"] = paper_json
            if not merged.get("output_dir") and ctx.get("output_dir"):
                merged["output_dir"] = ctx["output_dir"]
            if not merged.get("output_repo_dir") and ctx.get("output_repo_dir"):
                merged["output_repo_dir"] = ctx["output_repo_dir"]

        return merged

    def _dispatch_action(self, action: str, request: str, payload: Dict[str, Any]) -> AgentResult:
        handlers: Dict[str, Callable[[str, Dict[str, Any]], AgentResult]] = {
            "run_pipeline": self._run_pipeline,
            "evaluate": self._evaluate,
            "verify": self._verify,
            "explain": self._explain,
            "status": self._status,
        }
        if action == "help":
            return self._help(request)
        handler = handlers.get(action)
        if handler is None:
            return self._help(request)
        return handler(request, payload)

    def _feedback_replan(
        self,
        plan: ExecutionPlan,
        step_idx: int,
        step: PlanStep,
        result: AgentResult,
        payload: Dict[str, Any],
    ) -> bool:
        if result.status == "success":
            return False
        if step.attempt >= 1:
            return False

        err_set = set(result.errors)
        ctx = self._resolve_context(payload)
        has_run_context = bool(
            (ctx.get("paper_name")) and (ctx.get("paper_json_path") or ctx.get("paper_json_cleaned_path"))
        )

        if "missing_target_repo_dir" in err_set and step.action in {"verify", "explain", "status"} and has_run_context:
            recovery = PlanStep(
                action="run_pipeline",
                reason=f"Feedback recovery: `{step.action}` needs repo context, run pipeline first.",
                required=True,
                inserted_by_feedback=True,
                attempt=step.attempt + 1,
            )
            retry = PlanStep(
                action=step.action,
                reason=f"Retry `{step.action}` after recovery pipeline run.",
                required=step.required,
                inserted_by_feedback=True,
                attempt=step.attempt + 1,
            )
            plan.steps[step_idx : step_idx + 1] = [recovery, retry]
            return True

        if "missing_required_fields" in err_set and step.action == "evaluate" and has_run_context:
            recovery = PlanStep(
                action="run_pipeline",
                reason="Feedback recovery: evaluation requires missing paper context.",
                required=True,
                inserted_by_feedback=True,
                attempt=step.attempt + 1,
            )
            retry = PlanStep(
                action="evaluate",
                reason="Retry evaluation after context recovery.",
                required=step.required,
                inserted_by_feedback=True,
                attempt=step.attempt + 1,
            )
            plan.steps[step_idx : step_idx + 1] = [recovery, retry]
            return True

        return False

    def _execute_plan(self, request: str, payload: Dict[str, Any], plan: ExecutionPlan) -> AgentResult:
        answer_lines = [f"Plan strategy: {plan.strategy}"]
        commands: List[Dict[str, Any]] = []
        errors: List[str] = []
        metrics: Dict[str, Any] = {}
        artifacts: Dict[str, Any] = {"plan": plan.to_dict(), "step_results": []}

        overall_status = "success"
        max_step_budget = max(10, len(plan.steps) * 3)
        step_idx = 0

        while step_idx < len(plan.steps):
            if step_idx >= max_step_budget:
                overall_status = "error"
                errors.append("planner_step_budget_exceeded")
                answer_lines.append("Planner stopped: step budget exceeded.")
                break

            step = plan.steps[step_idx]
            step_payload = self._prepare_step_payload(payload, step.action)
            step_result = self._dispatch_action(step.action, request, step_payload)

            commands.extend(step_result.commands)
            errors.extend(step_result.errors)
            if step_result.metrics:
                metrics.update(step_result.metrics)
            if step_result.artifacts:
                artifacts.update(step_result.artifacts)

            artifacts["step_results"].append(
                {
                    "step_index": step_idx,
                    "step": step.to_dict(),
                    "status": step_result.status,
                    "errors": step_result.errors,
                    "answer": step_result.answer,
                }
            )
            answer_lines.append(f"{step.action}: {step_result.answer}")

            if step_result.status != "success":
                replanned = self._feedback_replan(plan, step_idx, step, step_result, step_payload)
                if replanned:
                    artifacts["plan"] = plan.to_dict()
                    answer_lines.append(f"Feedback: replanned after `{step.action}` failure.")
                    continue
                if step.required:
                    overall_status = "error"
                    break

            step_idx += 1

        if not artifacts.get("step_results"):
            overall_status = "error"
            errors.append("no_step_executed")

        final_action = "plan_execute" if len(plan.steps) > 1 else plan.steps[0].action
        dedup_errors = sorted(set(errors))
        return AgentResult(
            request=request,
            action=final_action,
            status=overall_status,
            answer="\n".join(answer_lines),
            artifacts=artifacts,
            metrics=metrics,
            commands=commands,
            errors=dedup_errors,
        )

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
        base_action = guess_action(message)
        payload = self._collect_payload(message)

        if message.strip().startswith("/"):
            parts = message.strip().split(maxsplit=1)
            if len(parts) > 1:
                payload = {**parse_kv_pairs(parts[1]), **extract_json_object(parts[1]), **payload}

        exec_plan = self._build_execution_plan(message, payload, base_action)
        result = self._execute_plan(message, payload, exec_plan)

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
