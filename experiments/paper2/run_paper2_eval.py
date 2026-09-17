"""Resumable server runner for the bounded paper-2 experiment matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROTOCOL_SCHEMA = "paper2_eval_protocol_v1"
MANIFEST_SCHEMA = "paper2_case_manifest_v1"
ALLOWED_NATIVE_RETURN_CODES = {0, 2}


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path, schema):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise RuntimeError(f"unexpected schema in {path}: {value.get('schema')}")
    return path, value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*args):
    return subprocess.check_output(
        ["git", *args], text=True, encoding="utf-8"
    ).strip()


def _numeric_failure_diagnostics(case):
    history = []
    for step in case.get("history") or ():
        trial_failures = []
        for trial in step.get("angular_line_search_trials") or ():
            failed_constraints = tuple(trial.get("failed_constraints") or ())
            geodesic_update = trial.get("geodesic_update") or {}
            reason = trial.get("reason")
            if "numeric" not in failed_constraints and not str(reason).startswith(
                "nonfinite"
            ):
                continue
            trial_failures.append({
                "backtrack": trial.get("backtrack"),
                "theta_radians": trial.get("theta_radians"),
                "reason": reason,
                "failed_constraints": list(failed_constraints),
                "geodesic_update_status": geodesic_update.get(
                    "geodesic_update_status"
                ),
            })
        history.append({
            "iteration": step.get("iteration"),
            "step_rejection_reason": step.get("step_rejection_reason"),
            "joint_solver_status": (step.get("joint_solver") or {}).get(
                "solver_status"
            ),
            "numeric_trial_failures": trial_failures,
        })
    return {
        "second_order_state": case.get("second_order_state"),
        "step_rejection_reason": case.get("step_rejection_reason"),
        "rejection_reason": case.get("rejection_reason"),
        "history": history,
    }


def _validate_case_result(report_path, case_uid, method, budget):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    variant = f"geodesic_joint_sqp_k{int(budget)}"
    case = (
        report.get("variants", {}).get(variant, {})
        .get("correction_by_case", {}).get(str(case_uid))
    )
    if not isinstance(case, dict) or case.get("execution_skipped"):
        raise RuntimeError(
            f"{method} k{budget} did not execute preregistered case {case_uid}"
        )
    if case.get("numeric_failure"):
        diagnostics = _numeric_failure_diagnostics(case)
        raise RuntimeError(
            f"{method} k{budget} numeric failure for {case_uid}: "
            f"{json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)}"
        )
    return {
        "report": str(Path(report_path).resolve()),
        "report_sha256": _file_sha(report_path),
        "numeric_failure": False,
        "activation_aware_supported": report.get(
            "activation_aware_supported"
        ),
        "numeric_audit_complete": report.get("numeric_audit_complete"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument(
        "--phase",
        choices=("mechanism", "development", "formal", "sealed"),
        required=True,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--evaluation-bank", required=True)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frozen-severity-envelope")
    parser.add_argument("--frozen-repair-contract")
    parser.add_argument(
        "--method", choices=("g1f2", "g1f3", "both"), default="both"
    )
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--no-geodesic-acceleration-ablation", action="store_true")
    args = parser.parse_args()

    protocol_path, protocol = _load_json(args.protocol, PROTOCOL_SCHEMA)
    manifest_path, manifest = _load_json(args.case_manifest, MANIFEST_SCHEMA)
    role_by_phase = {
        "mechanism": "mechanism_train",
        "development": "development",
        "formal": "development",
        "sealed": "sealed_held_out",
    }
    expected_role = role_by_phase[args.phase]
    if manifest.get("role") != expected_role:
        raise RuntimeError(
            f"{args.phase} requires a {expected_role} case manifest"
        )
    if not manifest.get("created_before_results"):
        raise RuntimeError("case manifest was not frozen before results")
    case_uids = tuple(str(value) for value in manifest.get("case_uids", ()))
    if not case_uids or len(set(case_uids)) != len(case_uids):
        raise RuntimeError("case manifest must contain unique case UIDs")
    if args.phase == "mechanism" and not (
        int(protocol["mechanism_audit"]["case_count_min"])
        <= len(case_uids)
        <= int(protocol["mechanism_audit"]["case_count_max"])
    ):
        raise RuntimeError("mechanism case count is outside the frozen range")

    expected_commit = os.environ.get("EXPECTED_COMMIT")
    if not expected_commit:
        raise RuntimeError("EXPECTED_COMMIT must be exported")
    if _git_output("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("working tree HEAD differs from EXPECTED_COMMIT")
    if _git_output("rev-parse", "origin/main") != expected_commit:
        raise RuntimeError("origin/main differs from EXPECTED_COMMIT")
    if _git_output("status", "--porcelain"):
        raise RuntimeError("paper2 runner requires a clean worktree")

    train_bank = Path(args.train_bank).resolve()
    eval_bank = Path(args.evaluation_bank).resolve()
    adapter_state = Path(args.adapter_state).resolve()
    for path in (train_bank, eval_bank, adapter_state, protocol_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.phase == "mechanism" and train_bank != eval_bank:
        raise RuntimeError("mechanism phase must evaluate the train bank")
    if args.phase != "mechanism" and not (
        args.frozen_severity_envelope and args.frozen_repair_contract
    ):
        raise RuntimeError(
            "development/formal/sealed jobs require frozen train contracts"
        )

    default_budgets = (
        (5,)
        if args.phase in {"mechanism", "development"}
        else (2, 3)
        if args.phase == "formal"
        else (2, 3, 5)
    )
    budgets = tuple(args.budgets or default_budgets)
    if len(set(budgets)) != len(budgets) or not set(budgets) <= {2, 3, 5}:
        raise RuntimeError("budgets must be unique members of 2, 3, 5")
    methods = ("g1f2", "g1f3") if args.method == "both" else (args.method,)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    binding = {
        "schema": "paper2_run_binding_v1",
        "implementation_commit": expected_commit,
        "phase": args.phase,
        "protocol": str(protocol_path),
        "protocol_sha256": _file_sha(protocol_path),
        "case_manifest": str(manifest_path),
        "case_manifest_sha256": _file_sha(manifest_path),
        "train_bank": str(train_bank),
        "train_bank_sha256": _file_sha(train_bank),
        "evaluation_bank": str(eval_bank),
        "evaluation_bank_sha256": _file_sha(eval_bank),
        "adapter_state": str(adapter_state),
        "adapter_state_sha256": _file_sha(adapter_state),
        "methods": list(methods),
        "budgets": list(budgets),
    }
    if protocol.get("compute_budget"):
        binding["compute_budget"] = dict(protocol["compute_budget"])
    if args.frozen_severity_envelope:
        binding["frozen_severity_envelope"] = str(
            Path(args.frozen_severity_envelope).resolve()
        )
        binding["frozen_severity_envelope_sha256"] = _file_sha(
            args.frozen_severity_envelope
        )
    if args.frozen_repair_contract:
        binding["frozen_repair_contract"] = str(
            Path(args.frozen_repair_contract).resolve()
        )
        binding["frozen_repair_contract_sha256"] = _file_sha(
            args.frozen_repair_contract
        )
    binding["binding_sha256"] = _canonical_sha(binding)
    binding_path = output_root / "run_binding.json"
    if binding_path.exists():
        previous = json.loads(binding_path.read_text(encoding="utf-8"))
        if previous != binding:
            raise RuntimeError("output root is bound to a different experiment")
    else:
        _write_json(binding_path, binding)

    if args.phase == "sealed":
        launch_marker = output_root / "SEALED_LAUNCHED.json"
        if launch_marker.exists():
            previous = json.loads(launch_marker.read_text(encoding="utf-8"))
            if previous.get("binding_sha256") != binding["binding_sha256"]:
                raise RuntimeError("sealed output root was already consumed")
        else:
            _write_json(launch_marker, {
                "schema": "paper2_sealed_launch_v1",
                "binding_sha256": binding["binding_sha256"],
                "launched_unix_time": time.time(),
                "one_shot": True,
            })

    mechanism_outputs = {
        method: output_root / f"matched_candidate_same_ray_{method}.jsonl"
        for method in methods
    }
    completed_jobs = []
    for method in methods:
        for budget in budgets:
            for case_uid in case_uids:
                safe_uid = case_uid.replace(":", "__").replace("/", "_")
                job_name = f"{method}_k{budget}_{safe_uid}"
                job_container = output_root / "jobs" / job_name
                complete_path = job_container / "job.complete.json"
                job_spec = {
                    "binding_sha256": binding["binding_sha256"],
                    "method": method,
                    "budget": int(budget),
                    "case_uid": case_uid,
                }
                job_sha = _canonical_sha(job_spec)
                if complete_path.exists():
                    complete = json.loads(complete_path.read_text(encoding="utf-8"))
                    if complete.get("job_sha256") != job_sha:
                        raise RuntimeError(f"resume binding mismatch: {job_name}")
                    if not Path(complete["report"]).is_file():
                        raise RuntimeError(f"completed report is missing: {job_name}")
                    completed_jobs.append(complete)
                    print(json.dumps({
                        "stage": "paper2_job_resume_skip",
                        "job": job_name,
                    }), flush=True)
                    continue
                job_container.mkdir(parents=True, exist_ok=True)
                attempts = sorted(job_container.glob("attempt_*"))
                if args.phase == "sealed" and attempts:
                    raise RuntimeError(
                        "a sealed case was already launched and may not be retried; "
                        "treat it as development evidence and freeze a new manifest"
                    )
                attempt_root = job_container / f"attempt_{len(attempts) + 1:03d}"
                report_path = (
                    attempt_root / "fixed_budget_correction.report.json"
                )
                command = [
                    str(Path(args.python).resolve()),
                    "-u", "-m",
                    "training.refiner_v15_15g_fixed_budget_correction",
                    "--config", args.config,
                    "--train-teacher-bank", str(train_bank),
                    "--validation-teacher-bank", str(eval_bank),
                    "--adapter-state", str(adapter_state),
                    f"--activation-aware-{method}",
                    "--evaluation-role",
                    (
                        "train_calibration" if args.phase == "mechanism"
                        else "development_validation"
                        if args.phase in {"development", "formal"}
                        else "final_held_out"
                    ),
                    "--steps", str(budget),
                    "--paper2-protocol", str(protocol_path),
                    "--paper2-case-uid", case_uid,
                    "--output-dir", str(attempt_root),
                    *[str(value) for value in protocol["solver_common_args"]],
                ]
                if args.phase != "mechanism":
                    command.extend([
                        "--frozen-severity-envelope",
                        str(Path(args.frozen_severity_envelope).resolve()),
                        "--frozen-full-shadow-repair-contract",
                        str(Path(args.frozen_repair_contract).resolve()),
                    ])
                if args.phase == "mechanism":
                    mechanism_output = mechanism_outputs[method]
                    command.extend([
                        "--paper2-mechanism-output", str(mechanism_output),
                        "--paper2-mechanism-radii",
                        *[
                            str(value) for value in
                            protocol["mechanism_audit"]["radii_rms"]
                        ],
                        "--paper2-mechanism-max-candidates-per-case-budget",
                        str(protocol["mechanism_audit"][
                            "max_candidates_per_case_budget_source"
                        ]),
                    ])
                    for mechanism_uid in case_uids:
                        command.extend([
                            "--paper2-mechanism-case-uid", mechanism_uid
                        ])
                    if not args.no_geodesic_acceleration_ablation:
                        command.append(
                            "--paper2-geodesic-acceleration-ablation"
                        )
                print(json.dumps({
                    "stage": "paper2_job_started",
                    "job": job_name,
                }), flush=True)
                native_rc = subprocess.run(command, check=False).returncode
                if native_rc not in ALLOWED_NATIVE_RETURN_CODES:
                    raise RuntimeError(
                        f"{job_name} failed with native rc={native_rc}"
                    )
                if not report_path.is_file():
                    raise RuntimeError(f"{job_name} did not produce a report")
                try:
                    result = _validate_case_result(
                        report_path, case_uid, method, budget
                    )
                except Exception as exc:
                    failure = {
                        "schema": "paper2_job_failure_v1",
                        "job_sha256": job_sha,
                        "job": job_name,
                        "native_return_code": native_rc,
                        "report": str(report_path.resolve()),
                        "report_sha256": _file_sha(report_path),
                        "error": str(exc),
                        **job_spec,
                    }
                    failure_path = attempt_root / "paper2_job_failure.json"
                    _write_json(failure_path, failure)
                    print(json.dumps({
                        "stage": "paper2_job_failed",
                        "job": job_name,
                        "native_return_code": native_rc,
                        "failure": str(failure_path),
                        "error": str(exc),
                    }, ensure_ascii=False), flush=True)
                    raise
                complete = {
                    "schema": "paper2_job_complete_v1",
                    "job_sha256": job_sha,
                    "job": job_name,
                    "native_return_code": native_rc,
                    **job_spec,
                    **result,
                }
                _write_json(complete_path, complete)
                completed_jobs.append(complete)
                print(json.dumps({
                    "stage": "paper2_job_complete",
                    "job": job_name,
                    "native_return_code": native_rc,
                }), flush=True)

    index = {
        "schema": "paper2_phase_index_v1",
        "binding": binding,
        "completed_job_count": len(completed_jobs),
        "jobs": completed_jobs,
        "mechanism_artifacts": [
            {
                "method": method,
                "path": str(path),
                "sha256": _file_sha(path),
            }
            for method, path in mechanism_outputs.items()
            if path.exists()
        ],
    }
    _write_json(output_root / "phase.index.json", index)
    print(json.dumps({
        "stage": "paper2_phase_complete",
        "phase": args.phase,
        "completed_job_count": len(completed_jobs),
        "index": str(output_root / "phase.index.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
