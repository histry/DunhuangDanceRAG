"""Small development-only probe for local contact transactions.

This command runs the lower-body solver on at most a few ownership windows.  It
does not run diffusion, IK replay, training, or a production quality gate.  Its
purpose is to verify that a local candidate has numeric before/after metrics,
stays inside its ownership window, and can produce a locally accepted solver
direction before an expensive full-sequence replay is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from training.motion_models import (
    MotionGenerationConfig,
    full_sequence_physical_diagnostics_np,
    true_lower_body_ik,
)


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _load_array(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = np.load(path, allow_pickle=False)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite {label}: {path}")
    return np.asarray(value, dtype=np.float32)


def _candidate_attempts(report: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    local = report.get("local_transactions", {})
    for transaction in local.get("transactions", []) or []:
        selection = transaction.get("candidate_selection", {})
        for attempt in selection.get("attempts", []) or []:
            yield attempt


def _summarize(report: Dict[str, Any], windows: List[List[int]]) -> Dict[str, Any]:
    attempts = list(_candidate_attempts(report))
    numeric = []
    accepted = []
    scope_violations = []
    finite_difference = []
    for attempt in attempts:
        exact = attempt.get("exact_audit", {}) or {}
        has_numeric = bool(
            exact.get("before_metrics")
            and exact.get("candidate_metrics")
            and exact.get("metric_delta")
        )
        if has_numeric:
            numeric.append(attempt)
        if bool(attempt.get("accepted", False)):
            accepted.append(attempt)
        if attempt.get("direction_source") == "finite_difference":
            finite_difference.append(attempt)
        scope = attempt.get("scope_audit", {}) or {}
        if scope.get("changed_frames_outside"):
            scope_violations.append(scope)
    infeasibility = report.get("local_infeasibility", {}) or {}
    return {
        "schema": "local_contact_transaction_probe_v1",
        "development_only": True,
        "training_started": False,
        "production_model_modified": False,
        "windows": windows,
        "attempt_count": len(attempts),
        "numeric_audit_count": len(numeric),
        "accepted_solver_candidate_count": len(accepted),
        "scope_violation_count": len(scope_violations),
        "numeric_audit_complete": len(numeric) == len(attempts),
        "local_solver_direction_exists": bool(accepted),
        "finite_difference_direction_count": len(finite_difference),
        "local_feasibility_status": (
            "feasible_direction_found"
            if accepted
            else "local_infeasible_under_current_action_basis"
        ),
        "local_infeasibility": infeasibility,
        "scope_safe": not scope_violations,
        "transactions": report.get("local_transactions", {}).get(
            "transactions", []
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--eligible", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-windows", type=int, default=1)
    args = parser.parse_args()

    motion = _load_array(args.motion, "motion")
    eligible = _load_array(args.eligible, "sliding_support_eligible").reshape(-1)
    eligible = eligible.astype(bool)
    if len(eligible) != len(motion):
        raise ValueError("sliding_support_eligible length mismatch")

    cfg = MotionGenerationConfig.from_json(args.config).apply_env()
    cfg.full_sequence_contact_repair_enable = True
    localization = full_sequence_physical_diagnostics_np(
        motion,
        cfg,
        sliding_support_eligible=eligible,
    )
    if args.start is not None or args.end is not None:
        start = 0 if args.start is None else int(args.start)
        end = len(motion) if args.end is None else int(args.end)
        windows = [[start, end]]
    else:
        windows = [
            [int(start), int(end)]
            for start, end in localization.get("repair_windows", [])
        ][: max(1, int(args.max_windows))]
    if not windows:
        raise RuntimeError("no repair window available for local probe")
    for start, end in windows:
        if start < 0 or end <= start or end > len(motion):
            raise ValueError(f"invalid probe window: {(start, end)}")

    candidate, solver_report = true_lower_body_ik(
        motion,
        cfg,
        sliding_support_eligible=eligible,
        repair_windows=windows,
        protected_frame_mask=None,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "probe_input.npy", motion, allow_pickle=False)
    np.save(output / "probe_candidate.npy", candidate, allow_pickle=False)
    report = _summarize(solver_report, windows)
    report.update({
        "motion_sha256": _hash_array(motion),
        "candidate_sha256": _hash_array(candidate),
        "input_localization": localization,
        "solver_report": solver_report,
    })
    (output / "probe.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "local_contact_transaction_probe_complete",
        "report": str(output / "probe.report.json"),
        "windows": windows,
        "attempts": report["attempt_count"],
        "numeric_audits": report["numeric_audit_count"],
        "accepted_solver_candidates": report[
            "accepted_solver_candidate_count"
        ],
        "scope_safe": report["scope_safe"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
