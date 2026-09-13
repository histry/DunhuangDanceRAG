"""Package the accepted Adapter + second-order repair composite.

Packaging is allowed only after immutable train, development, and one-shot
held-out receipts exist.  The result is deliberately not described as a newly
trained Adapter checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training import motion_models as m


MODEL_NAME = "v15_15h_adapter_second_order_composite.pt"
CONTRACT_NAME = "v15_15h_adapter_second_order_composite.contract.json"
MODEL_SCHEMA = "v15_15h_adapter_second_order_repair_composite_v1"
CONTRACT_SCHEMA = "v15_15h_adapter_second_order_repair_composite_contract_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _evidence(path):
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _model_state(payload, label):
    state = payload.get("model_state_dict")
    _require(isinstance(state, dict) and state, f"{label} lacks model_state_dict")
    return state


def run(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_NAME
    contract_path = output_dir / CONTRACT_NAME
    _require(not model_path.exists(), f"composite already exists: {model_path}")
    _require(not contract_path.exists(), f"contract already exists: {contract_path}")

    frozen = _read_json(args.g1f3_frozen_contract)
    held_out = _read_json(args.held_out_acceptance)
    one_shot = _read_json(args.held_out_one_shot_receipt)
    _require(frozen.get("schema") == "refiner_v15_15g1f3_frozen_contract_v1",
             "g1f3 frozen contract schema mismatch")
    _require(frozen.get("immutable") is True, "g1f3 contract is not immutable")
    _require(frozen.get("implementation_commit") == args.implementation_commit,
             "g1f3/code commit mismatch")
    _require(held_out.get("accepted") is True, "held-out acceptance failed")
    _require(one_shot.get("rerun_allowed") is False,
             "held-out receipt does not enforce one-shot use")
    _require(held_out.get("one_shot_receipt_sha256") ==
             _sha256(args.held_out_one_shot_receipt),
             "held-out acceptance/receipt hash mismatch")
    frozen_evidence = frozen.get("evidence") or {}
    _require(frozen_evidence.get("train_report", {}).get("sha256") ==
             _sha256(args.train_report), "frozen/train report mismatch")
    _require(frozen_evidence.get("development_report", {}).get("sha256") ==
             _sha256(args.dev_report), "frozen/development report mismatch")
    _require(frozen_evidence.get("train_manifest", {}).get("sha256") ==
             _sha256(args.train_manifest), "frozen/train manifest mismatch")
    _require(frozen_evidence.get("development_manifest", {}).get("sha256") ==
             _sha256(args.dev_manifest),
             "frozen/development manifest mismatch")
    _require(held_out.get("report_sha256") == _sha256(args.held_out_report),
             "held-out acceptance/report mismatch")
    _require(held_out.get("manifest_sha256") == _sha256(args.held_out_manifest),
             "held-out acceptance/manifest mismatch")
    fixed = frozen.get("fixed_parameters") or {}
    _require(fixed.get("correction_budgets") == [2, 3, 5],
             "frozen correction budgets changed")
    _require(len(fixed.get("angular_line_search_radians") or []) == 12,
             "frozen angular ladder changed")
    _require(float(fixed.get("target_rms", 0.0)) == 1.0e-4,
             "frozen radius changed")
    _require(fixed.get("curvature_dtype") == "float64",
             "frozen curvature dtype changed")

    base_payload = m.torch.load(
        args.base_refiner_checkpoint, map_location="cpu", weights_only=False
    )
    adapter_payload = m.torch.load(
        args.adapter_state, map_location="cpu", weights_only=False
    )
    base_state = dict(_model_state(base_payload, "base Refiner checkpoint"))
    adapter_state = adapter_payload.get("adapter_state_dict")
    _require(isinstance(adapter_state, dict) and adapter_state,
             "Adapter payload lacks adapter_state_dict")
    for key, value in adapter_state.items():
        _require(key.startswith("observable_adapter_"),
                 f"non-Adapter parameter in adapter_state_dict: {key}")
        if key in base_state:
            _require(tuple(base_state[key].shape) == tuple(value.shape),
                     f"Adapter parameter shape mismatch: {key}")

    conformal = _read_json(args.conformal_envelope)
    runtime = {
        "base_model_state_dict": base_state,
        "adapter_state_dict": adapter_state,
        "config": dict(base_payload.get("config") or {}),
        "observable_adapter_gate_mode": adapter_payload.get(
            "observable_adapter_gate_mode"
        ),
        "observable_adapter_gate_floor": adapter_payload.get("gate_floor"),
    }
    artifact = {
        "schema": MODEL_SCHEMA,
        "artifact_name": "Adapter + second-order repair composite",
        "artifact_is_single_retrained_adapter_checkpoint": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": args.implementation_commit,
        "runtime_refiner": runtime,
        "conformal_envelope": conformal,
        "second_order_contract": frozen,
        "source_hashes": {
            "base_refiner_checkpoint_sha256": _sha256(
                args.base_refiner_checkpoint
            ),
            "adapter_state_sha256": _sha256(args.adapter_state),
            "conformal_envelope_sha256": _sha256(args.conformal_envelope),
            "g1f3_frozen_contract_sha256": _sha256(
                args.g1f3_frozen_contract
            ),
        },
    }
    m.torch.save(artifact, model_path)

    contract = {
        "schema": CONTRACT_SCHEMA,
        "artifact_name": "Adapter + second-order repair composite",
        "artifact_is_single_retrained_adapter_checkpoint": False,
        "implementation_commit": args.implementation_commit,
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "base_refiner_checkpoint": _evidence(args.base_refiner_checkpoint),
        "adapter_state": _evidence(args.adapter_state),
        "conformal_envelope": _evidence(args.conformal_envelope),
        "g1f3_frozen_contract": _evidence(args.g1f3_frozen_contract),
        "manifests": {
            "train": _evidence(args.train_manifest),
            "development": _evidence(args.dev_manifest),
            "held_out": _evidence(args.held_out_manifest),
        },
        "acceptance_reports": {
            "train": _evidence(args.train_report),
            "development": _evidence(args.dev_report),
            "held_out": _evidence(args.held_out_report),
            "held_out_acceptance": _evidence(args.held_out_acceptance),
            "held_out_one_shot_receipt": _evidence(
                args.held_out_one_shot_receipt
            ),
        },
        "fixed_runtime_contract": {
            "correction_budgets": [2, 3, 5],
            "angular_line_search_radians": fixed[
                "angular_line_search_radians"
            ],
            "curvature_dtype": "float64",
            "geodesic_acceleration_included": True,
            "second_order_model_builds_per_iteration": 1,
            "second_order_model_reused_across_frozen_angles": True,
            "second_order_grid_execution_device":
                "same_cuda_device_as_motion",
            "second_order_joint_subproblem_solver": fixed[
                "second_order_joint_subproblem_solver"
            ],
            "second_order_sqp_refinement_starts": int(
                fixed["second_order_sqp_refinement_starts"]
            ),
            "second_order_sqp_refinement_iterations": int(
                fixed["second_order_sqp_refinement_iterations"]
            ),
            "second_order_sqp_smoothing": list(
                fixed["second_order_sqp_smoothing"]
            ),
            "second_order_sqp_line_search_radians": list(
                fixed["second_order_sqp_line_search_radians"]
            ),
            "second_order_host_candidate_sorting": False,
            "second_order_nonfinite_basis_policy":
                "deterministic_verified_subspace_reduction",
            "second_order_unverified_directions_used": False,
            "second_order_hvp_recovery": dict(
                fixed["second_order_hvp_recovery"]
            ),
            "second_order_prediction_active_guard_terms_frozen_across_"
            "curvature_evaluations": True,
            "second_order_basis_dimension": int(
                fixed["second_order_basis_dimension"]
            ),
            "second_order_grid_levels": int(
                fixed["second_order_grid_levels"]
            ),
            "second_order_feasibility_tolerance": float(
                fixed["second_order_feasibility_tolerance"]
            ),
            "finite_gap_required_reduction_formula": fixed[
                "finite_gap_required_reduction_formula"
            ],
            "finite_gap_already_safe_term_requires_fresh_descent": False,
            "ownership": fixed.get("ownership"),
            "scope_null_space_projection": fixed.get(
                "scope_null_space_projection"
            ),
            "target_rms": 1.0e-4,
            "projector": fixed.get("projector"),
            "authoritative_guard": fixed.get("authoritative_guard"),
            "minimum_repair_gain": 0.03,
            "atomic_commit_or_identity": True,
            "runtime_case_labels_consumed": False,
            "runtime_case_whitelist_allowed": False,
        },
    }
    with contract_path.open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "model": str(model_path),
        "model_sha256": contract["model"]["sha256"],
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-refiner-checkpoint", required=True)
    parser.add_argument("--adapter-state", required=True)
    parser.add_argument("--conformal-envelope", required=True)
    parser.add_argument("--g1f3-frozen-contract", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--held-out-manifest", required=True)
    parser.add_argument("--train-report", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--held-out-report", required=True)
    parser.add_argument("--held-out-acceptance", required=True)
    parser.add_argument("--held-out-one-shot-receipt", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
