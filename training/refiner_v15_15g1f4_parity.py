"""Canonical v9 parity gate for the g1f4 shared execution path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "refiner_v15_15g1f4_v9_canonical_parity_gate_v1"
CANONICAL_V9_COMMIT = "be71ae12a637073714d602525166c620f41bd243"
IGNORED_ADDITIVE_KEYS = {
    "elapsed_seconds",
    "g1f4_mode",
    "g1f4_shared_policy_metric_path",
    "metric_operator",
    "new_guard_term_transition",
    "progress_mode",
    "progress_policy_decision",
    "trial_science_step_strict_improvement",
    "observable_severity_envelope",
    "train_frozen_full_shadow_repair_contract",
    "train_frozen_full_shadow_repair_contract_sha256",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_additive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_additive(item)
            for key, item in sorted(value.items())
            if key not in IGNORED_ADDITIVE_KEYS
        }
    if isinstance(value, list):
        return [_strip_additive(item) for item in value]
    return value


def _canonical_payload(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("activation_aware_summary") or {}
    return _strip_additive({
        "evaluation_split": report.get("evaluation_split"),
        "evaluation_role": report.get("evaluation_role"),
        "exact_radius_normalization_by_case": report.get(
            "exact_radius_normalization_by_case"
        ),
        "activation_aware_summary": summary,
        "variants": report.get("variants") or {},
        "riemannian_correction_supported": report.get(
            "riemannian_correction_supported"
        ),
        "geodesic_joint_sqp_supported": report.get(
            "geodesic_joint_sqp_supported"
        ),
        "geodesic_joint_sqp_supported_budgets": report.get(
            "geodesic_joint_sqp_supported_budgets"
        ),
        "activation_aware_supported": report.get(
            "activation_aware_supported"
        ),
        "numeric_audit_complete": report.get("numeric_audit_complete"),
    })


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_differences(
    reference: Any,
    candidate: Any,
    *,
    path: str = "$",
    limit: int = 100,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, cursor: str) -> None:
        if len(differences) >= limit or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if len(differences) >= limit:
                    return
                if key not in left or key not in right:
                    differences.append({
                        "path": f"{cursor}.{key}",
                        "reference": left.get(key, "<missing>"),
                        "candidate": right.get(key, "<missing>"),
                    })
                else:
                    visit(left[key], right[key], f"{cursor}.{key}")
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append({
                    "path": f"{cursor}.length",
                    "reference": len(left),
                    "candidate": len(right),
                })
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{cursor}[{index}]")
            return
        differences.append({
            "path": cursor,
            "reference": left,
            "candidate": right,
        })

    visit(reference, candidate, path)
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-reference-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_path = Path(args.v9_reference_report).resolve()
    candidate_path = Path(args.candidate_report).resolve()
    output_path = Path(args.output).resolve()
    reference_report = _read(reference_path)
    candidate_report = _read(candidate_path)
    if reference_report.get("implementation_commit") != CANONICAL_V9_COMMIT:
        raise RuntimeError(
            "reference report was not produced by canonical v9 commit "
            f"{CANONICAL_V9_COMMIT}"
        )
    if candidate_report.get("progress_mode") != "current_equal_share":
        raise RuntimeError("parity candidate is not current_equal_share")
    metric = candidate_report.get("metric_operator") or {}
    if metric.get("metric_mode") != "identity":
        raise RuntimeError("parity candidate is not identity metric")

    reference = _canonical_payload(reference_report)
    candidate = _canonical_payload(candidate_report)
    differences = _first_differences(reference, candidate)
    accepted = not differences
    result = {
        "schema": SCHEMA,
        "accepted": accepted,
        "comparison": "exact_canonical_v9_operational_parity",
        "canonical_v9_commit": CANONICAL_V9_COMMIT,
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
            "canonical_sha256": _canonical_sha256(reference),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256_file(candidate_path),
            "canonical_sha256": _canonical_sha256(candidate),
        },
        "required_equal_fields": [
            "case selection and candidate metrics",
            "second-order state and accepted-step count",
            "accepted theta and exact radius",
            "endpoint and temporal diagnostics",
            "hard Guard margins",
            "active-set and internal-witness generation sequence",
        ],
        "ignored_fields": sorted(IGNORED_ADDITIVE_KEYS),
        "differences": differences,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "g1f4_v9_canonical_parity_gate",
        "accepted": accepted,
        "report": str(output_path),
        "difference_count": len(differences),
    }), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
