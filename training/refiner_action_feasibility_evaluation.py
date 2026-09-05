"""Development-set runner for the observable action-feasibility protocol."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from motion_geometry.product_manifold import product_log_np
from training.motion_models import MotionGenerationConfig
from training.refiner_action_feasibility import (
    ACTION_DIM,
    PROTOCOL_VERSION,
    STATUS_BUDGET_EXHAUSTED,
    STATUS_INVALID_INPUT,
    STATUS_NUMERICAL_FAILURE,
    STATUS_VERIFIED_FEASIBLE,
    ActionFeasibilityCase,
    FeasibilitySolverConfig,
    evaluate_action_candidate,
    normalized_raw_action_norm,
    solve_action_feasibility,
)

SCHEMA = "refiner_action_feasibility_dev_report_v1"
MANIFEST_SCHEMA = "refiner_action_feasibility_case_manifest_v1"
ALLOWED_SPLITS = frozenset({"train", "dev", "development", "validation", "val"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _load_array(path: Path, *, key: str | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        if key:
            if key not in data:
                raise KeyError(f"{key!r} missing from {path}")
            return data[key]
        names = list(data.files)
        if len(names) != 1:
            raise ValueError(f"{path} requires an explicit array key")
        return data[names[0]]
    raise ValueError(f"unsupported array file: {path}")


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _array_from_row(
    row: Mapping[str, Any], base: Path, name: str, *, required: bool = True
) -> np.ndarray | None:
    value = row.get(f"{name}_path")
    if value is None:
        if required:
            raise ValueError(f"case is missing {name}_path")
        return None
    return _load_array(_resolve_path(base, str(value)), key=row.get(f"{name}_key"))


def _validate_case_split(row: Mapping[str, Any]) -> None:
    split = str(row.get("split", "")).strip().lower()
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"case {row.get('case_id')!r} is not a development split: {split!r}")
    if str(row.get("role", "")) == "cross_event":
        left = str(row.get("left_split", split)).strip().lower()
        right = str(row.get("right_split", split)).strip().lower()
        if left != right or left not in ALLOWED_SPLITS:
            raise ValueError(f"cross-event case {row.get('case_id')!r} crosses split boundary")


def load_case_manifest(path: str | Path, cfg: MotionGenerationConfig) -> tuple[list[ActionFeasibilityCase], dict[str, Any]]:
    """Load an explicit development manifest and enforce recording isolation."""
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("case manifest schema mismatch")
    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("case manifest must contain non-empty cases")
    cases: list[ActionFeasibilityCase] = []
    case_ids: set[str] = set()
    recording_splits: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("case manifest contains a non-object case")
        forbidden_clean_fields = {
            "clean_path", "clean_motion_path", "hidden_clean_path",
            "ground_truth_path", "target_motion_path",
        }
        metadata = row.get("metadata", {})
        if forbidden_clean_fields.intersection(row) or (
            isinstance(metadata, Mapping) and forbidden_clean_fields.intersection(metadata)
        ):
            raise ValueError(f"case {row.get('case_id')!r} exposes hidden clean input")
        _validate_case_split(row)
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in case_ids:
            raise ValueError(f"duplicate or missing case_id: {case_id!r}")
        case_ids.add(case_id)
        split = str(row["split"]).strip().lower()
        recording_ids = [str(row.get("recording_uid", ""))]
        if str(row.get("role", "")) == "cross_event":
            recording_ids.extend([str(row.get("left_recording_uid", "")), str(row.get("right_recording_uid", ""))])
        for recording_uid in recording_ids:
            if not recording_uid:
                raise ValueError(f"case {case_id!r} is missing recording identity")
            previous = recording_splits.setdefault(recording_uid, split)
            if previous != split:
                raise ValueError(f"recording {recording_uid!r} leaks across splits")
        cases.append(
            ActionFeasibilityCase(
                case_id=case_id,
                role=str(row["role"]),
                width=int(row["width"]),
                position_stratum=str(row["position_stratum"]),
                split=split,
                reference=_array_from_row(row, manifest_path.parent, "reference"),
                seam=_array_from_row(row, manifest_path.parent, "seam"),
                joint_mask=_array_from_row(row, manifest_path.parent, "joint_mask"),
                root_mask=_array_from_row(row, manifest_path.parent, "root_mask"),
                contact_mask=_array_from_row(row, manifest_path.parent, "contact_mask", required=False),
                condition=_array_from_row(row, manifest_path.parent, "condition", required=False),
                cfg=cfg,
                boundary_role=str(row.get("boundary_role", row["role"])),
                source_uid=str(row.get("source_uid", "")),
                recording_uid=str(row.get("recording_uid", "")),
                left_source_uid=str(row.get("left_source_uid", "")),
                right_source_uid=str(row.get("right_source_uid", "")),
                left_recording_uid=str(row.get("left_recording_uid", "")),
                right_recording_uid=str(row.get("right_recording_uid", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return cases, {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "schema": raw.get("schema"),
        "cases": len(cases),
        "recordings": len(recording_splits),
        "sources": len({case.source_uid for case in cases if case.source_uid}),
        "splits": dict(sorted(recording_splits.items())),
    }


def _proposal_action(
    case: ActionFeasibilityCase, row: Mapping[str, Any], manifest_path: Path
) -> np.ndarray | None:
    action_path = row.get("proposal_action_path")
    motion_path = row.get("proposal_motion_path")
    if action_path and motion_path:
        raise ValueError(f"case {case.case_id!r} specifies both proposal action and motion")
    if action_path:
        action = _load_array(_resolve_path(manifest_path.parent, str(action_path)), key=row.get("proposal_action_key"))
        action = np.asarray(action, dtype=np.float32)
    elif motion_path:
        proposal = _load_array(_resolve_path(manifest_path.parent, str(motion_path)), key=row.get("proposal_motion_key"))
        proposal = np.asarray(proposal, dtype=np.float32)
        if proposal.shape != case.reference.shape:
            raise ValueError(f"proposal motion shape mismatch for {case.case_id!r}")
        action = product_log_np(case.reference, proposal)
    else:
        return None
    if action.shape != (case.frames, ACTION_DIM):
        raise ValueError(f"proposal action shape mismatch for {case.case_id!r}")
    if not np.isfinite(action).all():
        raise ValueError(f"proposal action is non-finite for {case.case_id!r}")
    return action


def _status_from_result(result: Any) -> str:
    return str(getattr(result, "status", STATUS_INVALID_INPUT))


def _baseline_row(case: ActionFeasibilityCase, baseline: str, evaluation: Mapping[str, Any], *, solver: Any = None) -> dict[str, Any]:
    baseline_status = (
        STATUS_VERIFIED_FEASIBLE
        if evaluation.get("joint_pass")
        else (
            STATUS_INVALID_INPUT
            if evaluation.get("invalid_input")
            else (STATUS_NUMERICAL_FAILURE if evaluation.get("numerical_failure") else STATUS_BUDGET_EXHAUSTED)
        )
    )
    row = {
        **case.manifest_identity(),
        "baseline": baseline,
        "status": _status_from_result(solver) if solver is not None else baseline_status,
        "joint_pass": bool(evaluation.get("joint_pass", False)),
        "rollback": bool(getattr(solver, "rollback", False)) if solver is not None else False,
        "failure_reasons": list(evaluation.get("failure_reasons", [])),
        "raw_action_norm_normalized": evaluation.get("action", {}).get("raw_action_norm_normalized"),
        "gate_flags": {
            key: evaluation.get(key)
            for key in ("endpoint_pass", "temporal_pass", "jerk_pass", "physical_pass", "fidelity_pass", "finite_pass", "joint_pass")
        },
        "key_metrics": {
            "observable_boundary": evaluation.get("observable_boundary"),
            "physical_stage": evaluation.get("physical_stage"),
            "fixed_reference_support": evaluation.get("fixed_reference_support"),
            "reference_fidelity": evaluation.get("reference_fidelity"),
            "action": evaluation.get("action"),
        },
        "elapsed_seconds": evaluation.get("elapsed_seconds"),
    }
    if solver is not None:
        row.update({
            "status": solver.status,
            "rollback": bool(solver.rollback),
            "solver_detail": solver.detail,
            "solver_iterations": len(solver.iterations),
            "final_failure_reasons": list(solver.final_evaluation.get("failure_reasons", [])),
            "final_joint_pass": bool(solver.final_evaluation.get("joint_pass", False)),
            "final_action_norm_normalized": normalized_raw_action_norm(solver.returned_action, case.cfg),
        })
    return row


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    valid = [row for row in values if row.get("status") != STATUS_INVALID_INPUT]
    return {
        "cases": len(values),
        "valid_cases": len(valid),
        "verified_feasible": sum(row.get("status") == STATUS_VERIFIED_FEASIBLE for row in values),
        "budget_exhausted": sum(row.get("status") == STATUS_BUDGET_EXHAUSTED for row in values),
        "numerical_failure": sum(row.get("status") == STATUS_NUMERICAL_FAILURE for row in values),
        "invalid_input": sum(row.get("status") == STATUS_INVALID_INPUT for row in values),
        "joint_pass_rate": (sum(bool(row.get("joint_pass", False)) for row in values) / len(values)) if values else None,
        "rollback_rate": (sum(bool(row.get("rollback", False)) for row in values) / len(values)) if values else None,
        "failure_reasons": dict(
            sorted(
                {
                    reason: sum(
                        reason in row.get("failure_reasons", []) for row in values
                    )
                    for reason in {
                        item
                        for row in values
                        for item in row.get("failure_reasons", [])
                    }
                }.items()
            )
        ),
    }


def run_evaluation(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be fresh and empty: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    solver_cfg = FeasibilitySolverConfig(
        max_iterations=int(args.max_iterations),
        initial_trust_radius=float(args.initial_trust_radius),
        minimum_trust_radius=float(args.minimum_trust_radius),
    )
    solver_cfg.validate()
    cases, case_info = load_case_manifest(args.case_manifest, cfg)
    case_manifest_path = Path(args.case_manifest).resolve()
    checkpoint_info = None
    if args.v1_checkpoint:
        checkpoint_path = Path(args.v1_checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_info = {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)}
    rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    proposal_rows = json.loads(case_manifest_path.read_text(encoding="utf-8")).get("cases", [])
    proposal_by_id = {str(row["case_id"]): row for row in proposal_rows}
    for case in cases:
        zero = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
        b0_eval = evaluate_action_candidate(case, zero, label="B0_bridge_only_zero_action")
        rows.append(_baseline_row(case, "B0_bridge_only_zero_action", b0_eval))
        b1 = solve_action_feasibility(case, initial_action=zero, solver_config=solver_cfg)
        rows.append(_baseline_row(case, "B1_zero_plus_action_solver", b1.final_evaluation, solver=b1))
        solver_rows.extend({**case.manifest_identity(), "baseline": "B1_zero_plus_action_solver", **iteration} for iteration in b1.iterations)
        try:
            proposal = _proposal_action(case, proposal_by_id[case.case_id], case_manifest_path)
            proposal_error = None
        except (KeyError, OSError, ValueError, TypeError) as exc:
            proposal = None
            proposal_error = f"invalid_v1_proposal:{type(exc).__name__}"
        if proposal is None or checkpoint_info is None:
            missing = proposal_error or "missing_v1_proposal_or_verified_checkpoint"
            b2_eval = {"joint_pass": False, "failure_reasons": [missing], "invalid_input": True, "_cfg": case.cfg}
            rows.append(_baseline_row(case, "B2_frozen_v1_proposal", b2_eval))
            rows.append(_baseline_row(case, "B3_frozen_v1_proposal_plus_action_solver", b2_eval))
        else:
            b2_eval = evaluate_action_candidate(case, proposal, label="B2_frozen_v1_proposal")
            rows.append(_baseline_row(case, "B2_frozen_v1_proposal", b2_eval))
            b3 = solve_action_feasibility(case, initial_action=proposal, solver_config=solver_cfg)
            rows.append(_baseline_row(case, "B3_frozen_v1_proposal_plus_action_solver", b3.final_evaluation, solver=b3))
            solver_rows.extend({**case.manifest_identity(), "baseline": "B3_frozen_v1_proposal_plus_action_solver", **iteration} for iteration in b3.iterations)
    case_path = output / "case_level.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    iterations_path = output / "solver_iterations.jsonl"
    with iterations_path.open("w", encoding="utf-8") as handle:
        for row in solver_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    by_baseline = defaultdict(list)
    by_group = defaultdict(list)
    for row in rows:
        by_baseline[row["baseline"]].append(row)
        by_group[(row["role"], int(row["width"]), row["position_stratum"])].append(row)
    source_commit = args.source_commit or _git_value("rev-parse", "HEAD")
    dirty_state = args.dirty_state or _git_dirty_state()
    diff_hash = _git_diff_hash() if dirty_state != "clean" else None
    resolved_config = dataclasses.asdict(cfg)
    config_sha256 = _sha256_json(resolved_config)
    report = {
        "schema": SCHEMA,
        "completed": True,
        "protocol": PROTOCOL_VERSION,
        "development_only": True,
        "formal_preregistration": False,
        "training_started": False,
        "production_model_modified": False,
        "scientific_acceptance": False,
        "pilot_allowed": False,
        "provenance": {
            "source_commit": source_commit,
            "dirty_state": dirty_state,
            "diff_sha256": diff_hash,
            "resolved_config_sha256": config_sha256,
            "case_manifest": case_info,
            "case_manifest_sha256": case_info["sha256"],
            "checkpoint": checkpoint_info,
            "random_seed": int(args.seed),
            "decoder_protocol": "product_refiner_true_decoder_confidence_smoothing_taper_cap_v1",
            "metric_protocol": "observable_boundary_stage_physical_fixed_support_fidelity_v1",
        },
        "solver": solver_cfg.as_dict(),
        "baselines": {name: _summary(values) for name, values in sorted(by_baseline.items())},
        "groups": {f"{role}/{width}/{position}": _summary(values) for (role, width, position), values in sorted(by_group.items())},
        "cases": {"primary_cases": len(cases), "baseline_rows": len(rows), "recordings": case_info["recordings"], "sources": case_info["sources"]},
        "decision": {
            "result": "ACTION_FEASIBILITY_DEVELOPMENT_COMPLETE_NO_NETWORK_TRAINING",
            "next_action": "inspect_development_failure_modes_before_any_network_training",
        },
    }
    _write_json(output / "manifest.json", {
        "schema": "refiner_action_feasibility_dev_manifest_v1",
        "source_commit": source_commit,
        "dirty_state": dirty_state,
        "diff_sha256": diff_hash,
        "development_protocol_version": PROTOCOL_VERSION,
        "resolved_config": resolved_config,
        "config_sha256": config_sha256,
        "solver_budget": solver_cfg.as_dict(),
        "case_manifest_sha256": case_info["sha256"],
        "checkpoint_sha256": checkpoint_info["sha256"] if checkpoint_info else None,
        "random_seed": int(args.seed),
        "decoder_protocol": report["provenance"]["decoder_protocol"],
        "metric_protocol": report["provenance"]["metric_protocol"],
        "started_at": args.started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    _write_json(output / "report.json", report)
    (output / "evidence_summary.md").write_text(_evidence_summary(report), encoding="utf-8")
    print(json.dumps({"stage": "refiner_action_feasibility_dev_complete", "report": str(output / "report.json"), "cases": len(cases), "training_started": False}, ensure_ascii=False))
    return 0


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty_state() -> str:
    value = _git_value("status", "--porcelain")
    return "clean" if not value else "dirty"


def _git_diff_hash() -> str | None:
    try:
        payload = subprocess.check_output(["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _evidence_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Refiner action feasibility development diagnostic",
        "",
        "This is DEVELOPMENT / NOT A FORMAL PREREGISTRATION. No network training,",
        "production model update, pilot, or blind-test design adjustment is implied.",
        "",
        f"- source commit: `{report['provenance']['source_commit']}`",
        f"- cases: {report['cases']['primary_cases']}; recordings: {report['cases']['recordings']}; sources: {report['cases']['sources']}",
        f"- decision: `{report['decision']['result']}`",
        "",
        "Baselines are reported separately. Rollback is not counted as a rescue,",
        "and B2/B3 remain unavailable unless a verified V1 checkpoint and explicit",
        "proposal actions are present in the case manifest.",
        "",
    ]
    for name, summary in report["baselines"].items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- joint pass rate: {summary['joint_pass_rate']}")
        lines.append(f"- verified feasible: {summary['verified_feasible']}; budget exhausted: {summary['budget_exhausted']}; numerical failure: {summary['numerical_failure']}; invalid: {summary['invalid_input']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--v1-checkpoint")
    parser.add_argument("--source-commit")
    parser.add_argument("--dirty-state")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=24)
    parser.add_argument("--initial-trust-radius", type=float, default=0.25)
    parser.add_argument("--minimum-trust-radius", type=float, default=0.015625)
    parser.set_defaults(started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return parser


def main() -> int:
    return run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
