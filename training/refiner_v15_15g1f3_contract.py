"""Seal and verify the V15.15g1f3 train/dev/held-out evidence chain.

This command is intentionally evidence-only.  It never trains, refits, or
changes the Adapter, conformal envelope, second-order parameters, Projector,
or Guard.  Files described as frozen are created with exclusive-create
semantics so a prior scientific decision cannot be silently overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from training import refiner_v15_15g1f3_second_order as second_order


REPORT_SCHEMA = "refiner_v15_15g1f3_second_order_composite_closure_sqp_v1"
FROZEN_SCHEMA = "refiner_v15_15g1f3_frozen_contract_v1"
TARGET_CASE_UIDS = (
    "txn_0001_97ecf5fd6e62:169",
    "txn_0001_97ecf5fd6e62:43",
    "txn_0004_5f4af0aa4f40:171",
    "txn_0005_a6fbd294b71c:169",
    "txn_0007_0d8eea4df4f1:137",
)
FAILURE_STATES = {
    "insufficient_second_order_predicted_progress",
    "second_order_finite_radius_model_mismatch",
    "active_set_transition_model_mismatch",
    "nonfinite_or_unverified_curvature",
    "second_order_solver_failure",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _summary(report):
    return report.get("activation_aware_summary") or {}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _unchanged_contract(report):
    for key in (
        "fixed_guard_thresholds_changed",
        "physical_gate_changed",
        "fixed_support_gate_changed",
        "fidelity_gate_changed",
        "boundary_gate_changed",
        "observable_0p03_gate_changed",
    ):
        _require(report.get(key) is False, f"{key} must remain false")
    _require(float(report.get("target_rms", math.nan)) == 1.0e-4,
             "radius contract changed")
    _require(report.get("second_order_hessian_used") is True,
             "second-order Hessian evidence is absent")
    _require(report.get("second_order_joint_sqp") is True,
             "second-order joint SQP is absent")
    _require(report.get("finite_gap_required_reduction_formula") ==
             "current_delta+strict_limit+safety_margin",
             "finite-gap strict-boundary formula changed")
    _require(report.get(
        "finite_gap_already_safe_term_may_use_safe_slack"
    ) is True, "already-safe science slack is unavailable")
    _require(report.get("curvature_dtype") == "float64",
             "curvature dtype is not float64")
    _require(report.get("geodesic_acceleration_included") is True,
             "geodesic acceleration is absent")
    _require(report.get("ambient_hessian_materialized") is False,
             "ambient Hessian was materialized")
    _require(report.get("second_order_model_builds_per_iteration") == 1,
             "second-order curvature model is rebuilt per angle")
    _require(report.get(
        "second_order_model_reused_across_frozen_angles"
    ) is True, "second-order curvature model is not reused")
    _require(report.get("second_order_grid_execution_device") ==
             "same_cuda_device_as_motion",
             "second-order candidate grid is not device-resident")
    _require(report.get("second_order_joint_subproblem_solver") ==
             "deterministic_device_resident_riemannian_continuous_sqp",
             "second-order continuous joint SQP is absent")
    _require(report.get("second_order_sqp_refinement_starts") ==
             second_order.SECOND_ORDER_SQP_REFINEMENT_STARTS,
             "second-order SQP start count changed")
    _require(report.get("second_order_sqp_refinement_iterations") ==
             second_order.SECOND_ORDER_SQP_REFINEMENT_ITERATIONS,
             "second-order SQP iteration count changed")
    _require(report.get("second_order_sqp_smoothing") == [
        float(value) for value in second_order.SECOND_ORDER_SQP_SMOOTHING
    ], "second-order SQP smoothing schedule changed")
    _require(report.get("second_order_sqp_constraint_scaling") ==
             "absolute_signed_boundary_gap_floor_1e-12",
             "second-order SQP constraint scaling changed")
    _require(report.get("second_order_sqp_line_search_radians") == [
        float(value)
        for value in second_order.SECOND_ORDER_SQP_LINE_SEARCH_RADIANS
    ], "second-order SQP line search changed")
    _require(report.get("second_order_host_candidate_sorting") is False,
             "second-order candidates are sorted on the host")
    _require(report.get("second_order_nonfinite_basis_policy") ==
             "deterministic_verified_subspace_reduction",
             "second-order nonfinite-basis policy changed")
    _require(report.get("second_order_unverified_directions_used") is False,
             "unverified curvature directions were used")
    recovery = report.get("second_order_hvp_recovery") or {}
    _require(recovery.get("trigger") ==
             "nonfinite_autograd_second_derivative_only",
             "second-order HvP recovery trigger changed")
    _require(recovery.get("method") ==
             "symmetric_first_derivative_hvp_epsilon_ladder",
             "second-order HvP recovery method changed")
    _require(recovery.get("requires_consistent_estimates") is True,
             "second-order HvP recovery verification is absent")
    _require(report.get(
        "second_order_prediction_active_guard_terms_frozen_across_"
        "curvature_evaluations"
    ) is True, "prediction active Guard terms are not curvature-frozen")


def _require_zero_scope(summary, label):
    closures = summary.get("composite_closure_by_case") or {}
    _require(bool(closures), f"{label} has no composite closure evidence")
    _require(
        all(float(row.get("scope_leakage_abs_max", math.inf)) == 0.0
            for row in closures.values()),
        f"{label} scope leakage is nonzero",
    )


def _require_conformal_score_source(summary, expected, label):
    decisions = summary.get("decisions") or {}
    _require(bool(decisions), f"{label} has no conformal decisions")
    for uid, decision in decisions.items():
        severity = (
            (decision.get("selection") or {}).get("anchor_severity") or {}
        )
        _require(
            severity.get("score_source") == expected,
            f"{label} conformal score source changed for {uid}",
        )
        _require(severity.get("offline_label_consumed") is False,
                 f"{label} conformal gate consumed an offline label for {uid}")


def _validate_train(report):
    _require(report.get("schema") == REPORT_SCHEMA, "train report schema mismatch")
    _require(report.get("evaluation_role") == "train_calibration",
             "train report role mismatch")
    _require(report.get("evaluation_split") == "train",
             "train report split mismatch")
    _require(report.get("stable_pass_definition") ==
             "composite_selector_final_projected_full_guard_closure",
             "stable-pass definition mismatch")
    _require(report.get("numeric_audit_complete") is True,
             "train numeric audit is incomplete")
    _unchanged_contract(report)
    summary = _summary(report)
    _require(summary.get("g1f3_train_target_case_uids") == list(TARGET_CASE_UIDS),
             "train target cases changed")
    _require(summary.get("g1f3_train_calibration_probe_case_uids") ==
             list(TARGET_CASE_UIDS),
             "train-only g1f3 calibration probes changed")
    _require(summary.get(
        "g1f3_train_calibration_probes_excluded_from_single_controls"
    ) is True, "declared mismatch probes were audited as single controls")
    decisions = summary.get("decisions") or {}
    _require(all(
        (decisions.get(uid) or {}).get(
            "train_calibration_probe_forced_evaluation"
        ) is True
        for uid in TARGET_CASE_UIDS
    ), "a declared train calibration probe was not evaluated")
    _require(summary.get("g1f3_train_target_closure_complete") is True,
             "five train target cases did not all close")
    _require(summary.get("adapter_incumbent_preserved") is True,
             "an Adapter incumbent was lost")
    _require(summary.get("group_coverage_complete") is True,
             "required projected group coverage is incomplete")
    _require(summary.get("scope_safe") is True, "train scope audit failed")
    _require_zero_scope(summary, "train")
    _require_conformal_score_source(
        summary,
        "train_leave_one_transaction_out_observable_models",
        "train",
    )
    _require(summary.get("runtime_case_whitelist_used") is False,
             "runtime case whitelist is forbidden")
    _require(summary.get("activation_aware_supported") is True,
             "train composite gate failed")


def _validate_dev(report):
    _require(report.get("schema") == REPORT_SCHEMA, "dev report schema mismatch")
    _require(report.get("evaluation_role") == "development_validation",
             "dev report role mismatch")
    _require(report.get("numeric_audit_complete") is True,
             "dev numeric audit is incomplete")
    _unchanged_contract(report)
    summary = _summary(report)
    _require(summary.get("development_validation_reused") is True,
             "case 53 is not marked development_validation_reused")
    _require(summary.get("cross_exact_closure_complete") is True,
             "development cross closure is incomplete")
    _require(summary.get("adapter_incumbent_preserved") is True,
             "development replaced a closed Adapter incumbent")
    _require(summary.get("scope_safe") is True, "development scope audit failed")
    _require_zero_scope(summary, "development")
    _require_conformal_score_source(
        summary, "final_train_models", "development"
    )
    _require(summary.get("single_identity_safe") is True,
             "development single identity control failed")
    counts = summary.get("selected_projected_count_by_group") or {}
    _require(int(counts.get("cross_short", 0)) >= 8,
             "the eight development cross_short cases did not all pass")
    _require(int(counts.get("cross_long", 0)) >= 1,
             "development case 53 cross_long did not pass")
    _require(summary.get("activation_aware_supported") is True,
             "development composite gate failed")


def _decision_signature(report):
    decisions = _summary(report).get("decisions") or {}
    signature = {}
    for uid in sorted(decisions):
        row = decisions[uid]
        selection = row.get("selection") or {}
        candidates = selection.get("candidates") or {}
        signature[uid] = {
            "selected_method": row.get("selected_method"),
            "selected_nonzero": row.get("selected_nonzero"),
            "adapter_incumbent_locked": row.get("adapter_incumbent_locked"),
            "identity_fallback_reason": row.get("identity_fallback_reason"),
            "composite_final_closure": row.get("composite_final_closure"),
            "candidate_metrics": {
                method: {
                    key: value
                    for key, value in candidate.items()
                    if key in {
                        "eligible",
                        "exact_raw_closure_passed",
                        "effective_projected_candidate",
                        "outside_scope_abs_max",
                        "maximum_full_transaction_fixed_guard_shadow_margin",
                        "scientific_descent_score",
                        "full_shadow_reduction",
                        "step_rejection_reason",
                    }
                }
                for method, candidate in sorted(candidates.items())
            },
        }
    return signature


def compare_replicates(args):
    paths = [Path(value) for value in args.report]
    _require(len(paths) == 3, "exactly three cold-start reports are required")
    reports = [_read_json(path) for path in paths]
    for report in reports:
        _validate_train(report)
    signatures = [_decision_signature(report) for report in reports]
    hashes = [_canonical_sha256(value) for value in signatures]
    _require(len(set(hashes)) == 1,
             "cold starts disagree on selection, state, or selected metrics")
    value = {
        "schema": "refiner_v15_15g1f3_three_cold_start_acceptance_v1",
        "accepted": True,
        "comparison": "exact_canonical_selection_state_and_metric_equality",
        "reports": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in paths
        ],
        "decision_signature_sha256": hashes[0],
    }
    _write_exclusive(args.output, value)


def freeze_contract(args):
    train = _read_json(args.train_report)
    dev = _read_json(args.dev_report)
    replicas = _read_json(args.cold_start_acceptance)
    repair = _read_json(args.repair_contract)
    _validate_train(train)
    _validate_dev(dev)
    _require(replicas.get("accepted") is True, "cold-start acceptance failed")
    _require(repair.get("second_order_joint_sqp") is True,
             "repair contract is not g1f3")
    _require(repair.get("correction_budgets") == [2, 3, 5],
             "repair budgets changed")
    angles = repair.get("angular_line_search_radians") or []
    _require(len(angles) == 12, "the frozen angular ladder must have 12 levels")
    _require(repair.get("curvature_dtype") == "float64",
             "repair curvature dtype changed")
    value = {
        "schema": FROZEN_SCHEMA,
        "artifact_name": "V15.15g1f3 frozen second-order repair contract",
        "immutable": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": args.implementation_commit,
        "stable_pass_definition":
            "composite_selector_final_projected_full_guard_closure",
        "calibration_policy": {
            "train_only": True,
            "development_parameter_selection": False,
            "held_out_parameter_selection": False,
            "runtime_case_whitelist_allowed": False,
            "declared_train_probe_uids_used_for_calibration_audit_only": list(
                TARGET_CASE_UIDS
            ),
            "declared_train_probe_uids_packaged_for_runtime_activation": False,
        },
        "fixed_parameters": {
            "correction_budgets": [2, 3, 5],
            "angular_line_search_radians": angles,
            "target_rms": 1.0e-4,
            "curvature_dtype": "float64",
            "geodesic_acceleration_included": True,
            "ambient_hessian_materialized": False,
            "second_order_model_builds_per_iteration": 1,
            "second_order_model_reused_across_frozen_angles": True,
            "second_order_grid_execution_device":
                "same_cuda_device_as_motion",
            "second_order_joint_subproblem_solver":
                "deterministic_device_resident_riemannian_continuous_sqp",
            "second_order_sqp_refinement_starts": int(
                repair["second_order_sqp_refinement_starts"]
            ),
            "second_order_sqp_refinement_iterations": int(
                repair["second_order_sqp_refinement_iterations"]
            ),
            "second_order_sqp_smoothing": list(
                repair["second_order_sqp_smoothing"]
            ),
            "second_order_sqp_constraint_scaling": repair[
                "second_order_sqp_constraint_scaling"
            ],
            "second_order_sqp_line_search_radians": list(
                repair["second_order_sqp_line_search_radians"]
            ),
            "second_order_host_candidate_sorting": False,
            "second_order_nonfinite_basis_policy":
                "deterministic_verified_subspace_reduction",
            "second_order_unverified_directions_used": False,
            "second_order_hvp_recovery": dict(
                repair["second_order_hvp_recovery"]
            ),
            "second_order_prediction_active_guard_terms_frozen_across_"
            "curvature_evaluations": True,
            "second_order_basis_dimension": int(
                repair["second_order_basis_dimension"]
            ),
            "second_order_grid_levels": int(
                repair["second_order_grid_levels"]
            ),
            "second_order_feasibility_tolerance": float(
                repair["second_order_feasibility_tolerance"]
            ),
            "finite_gap_required_reduction_formula": repair[
                "finite_gap_required_reduction_formula"
            ],
            "finite_gap_already_safe_term_may_use_safe_slack": True,
            "ownership": "exact_boolean_transaction_ownership_mask",
            "scope_null_space_projection": repair.get(
                "scope_null_space_projection"
            ),
            "projector": "unchanged_EDGE151_contract_projector",
            "authoritative_guard":
                "complete_transaction_fixed_hard_guard_after_projector",
            "minimum_repair_gain": 0.03,
        },
        "evidence": {
            "train_report": {"path": str(Path(args.train_report).resolve()),
                             "sha256": _sha256(args.train_report)},
            "development_report": {"path": str(Path(args.dev_report).resolve()),
                                   "sha256": _sha256(args.dev_report)},
            "cold_start_acceptance": {
                "path": str(Path(args.cold_start_acceptance).resolve()),
                "sha256": _sha256(args.cold_start_acceptance),
            },
            "train_manifest": {"path": str(Path(args.train_manifest).resolve()),
                               "sha256": _sha256(args.train_manifest)},
            "development_manifest": {
                "path": str(Path(args.dev_manifest).resolve()),
                "sha256": _sha256(args.dev_manifest),
            },
            "repair_contract": {"path": str(Path(args.repair_contract).resolve()),
                                "sha256": _sha256(args.repair_contract)},
            "conformal_envelope": {
                "path": str(Path(args.conformal_envelope).resolve()),
                "sha256": _sha256(args.conformal_envelope),
            },
        },
    }
    _write_exclusive(args.output, value)


def seal_held_out(args):
    from training import motion_models as m

    train = m.torch.load(args.train_bank, map_location="cpu", weights_only=False)
    dev = m.torch.load(args.dev_bank, map_location="cpu", weights_only=False)
    candidate = m.torch.load(
        args.candidate_bank, map_location="cpu", weights_only=False
    )
    used = {
        str(sample["transaction_id"])
        for bank in (train, dev)
        for sample in bank.get("samples", [])
    }
    used_sources = {
        str(sample["source_case_uid"])
        for bank in (train, dev)
        for sample in bank.get("samples", [])
        if sample.get("source_case_uid") is not None
    }
    for prior_path in args.exclude_manifest or []:
        prior = _read_json(prior_path)
        if prior.get("transaction_id") is not None:
            used.add(str(prior["transaction_id"]))
        used_sources.update(
            str(value) for value in prior.get("source_case_uids", [])
        )
    candidate_transactions = {}
    for sample in candidate.get("samples", []):
        transaction_id = str(sample["transaction_id"])
        candidate_transactions.setdefault(transaction_id, []).append(sample)
    by_transaction = {
        transaction_id: rows
        for transaction_id, rows in candidate_transactions.items()
        if transaction_id not in used
        and all(
            row.get("source_case_uid") is None
            or str(row["source_case_uid"]) not in used_sources
            for row in rows
        )
    }
    _require(by_transaction, "candidate bank has no untouched transaction")
    selected_id = min(
        by_transaction,
        key=lambda value: (
            hashlib.sha256(
                f"{args.selection_seed}:{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )
    selected_samples = sorted(
        by_transaction[selected_id],
        key=lambda row: (int(row.get("local_case_index", row["case_index"])),
                         int(row["case_index"])),
    )
    source_indices = [int(row["case_index"]) for row in selected_samples]
    remapped_samples = []
    for case_index, sample in enumerate(selected_samples):
        remapped = dict(sample)
        remapped["source_bank_case_index"] = int(sample["case_index"])
        remapped["case_index"] = case_index
        remapped_samples.append(remapped)
    selected_samples = remapped_samples
    selected_uids = sorted(str(row["case_uid"]) for row in selected_samples)
    selected_source_uids = sorted({
        str(row["source_case_uid"])
        for row in selected_samples
        if row.get("source_case_uid") is not None
    })
    _require(not any(uid.endswith(":53") for uid in selected_uids),
             "reused development case 53 cannot enter final held-out")
    sealed_bank = dict(candidate)
    sealed_bank["samples"] = selected_samples
    index = m.torch.as_tensor(source_indices, dtype=m.torch.long)

    def select_rows(value):
        if m.torch.is_tensor(value):
            return value.index_select(0, index)
        if isinstance(value, dict):
            return {key: select_rows(item) for key, item in value.items()}
        return value

    sealed_bank["batch"] = select_rows(candidate["batch"])
    sealed_bank["baseline_prediction"] = candidate[
        "baseline_prediction"
    ].index_select(0, index)
    sealed_bank["baseline_identity"] = candidate[
        "baseline_identity"
    ].index_select(0, index)
    if candidate.get("transaction_schedules"):
        sealed_bank["transaction_schedules"] = {
            selected_id: candidate["transaction_schedules"][selected_id]
        }
        sealed_bank["transaction_context_indices"] = candidate[
            "transaction_schedules"
        ][selected_id]["context_indices"]
    if candidate.get("transaction_guard_contracts"):
        sealed_bank["transaction_guard_contracts"] = {
            selected_id: candidate["transaction_guard_contracts"][selected_id]
        }
    projected = Counter(
        str(row.get("audit_group"))
        for row in selected_samples
        if row.get("teacher_kind") == "exact_projected_direction"
    )
    sealed_bank["projected_teacher_count_by_group"] = dict(projected)
    sealed_bank["final_held_out_selection"] = {
        "transaction_id": selected_id,
        "selection_rule": "minimum_sha256(seed:transaction_id)_then_id",
        "selection_seed": args.selection_seed,
        "oracle_outcome_inspected_before_selection": False,
    }
    bank_path = Path(args.output_bank)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    _require(not bank_path.exists(), "sealed held-out bank already exists")
    m.torch.save(sealed_bank, bank_path)
    value = {
        "schema": "refiner_v15_15g1f3_final_held_out_manifest_v1",
        "immutable": True,
        "transaction_id": selected_id,
        "case_uids": selected_uids,
        "source_case_uids": selected_source_uids,
        "case_indices": sorted(source_indices),
        "sealed_bank_case_indices": list(range(len(selected_samples))),
        "selection_rule": "minimum_sha256(seed:transaction_id)_then_id",
        "selection_seed": args.selection_seed,
        "oracle_outcome_inspected_before_selection": False,
        "source_candidate_bank": str(Path(args.candidate_bank).resolve()),
        "source_candidate_bank_sha256": _sha256(args.candidate_bank),
        "sealed_bank": str(bank_path.resolve()),
        "sealed_bank_sha256": _sha256(bank_path),
        "train_bank_sha256": _sha256(args.train_bank),
        "development_bank_sha256": _sha256(args.dev_bank),
    }
    value["content_sha256"] = _canonical_sha256(value)
    _write_exclusive(args.output_manifest, value)


def record_oracle(args):
    manifest = _read_json(args.manifest)
    oracle = _read_json(args.oracle_report)
    transaction_id = manifest.get("transaction_id")
    _require(oracle.get("transaction_id") == transaction_id,
             "Oracle transaction does not match sealed manifest")
    _require(oracle.get("cross_long_feasible") is True,
             "Oracle found no real feasible cross_long")
    for key in (
        "raw_evidence", "projector_evidence", "scope_evidence",
        "full_guard_evidence",
    ):
        _require(bool(oracle.get(key)), f"Oracle lacks {key}")
    value = {
        "schema": "refiner_v15_15g1f3_final_held_out_oracle_receipt_v1",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "oracle_report": str(Path(args.oracle_report).resolve()),
        "oracle_report_sha256": _sha256(args.oracle_report),
        "transaction_id": transaction_id,
        "cross_long_feasible": True,
        "g1f3_executed_before_oracle_receipt": False,
    }
    _write_exclusive(args.output, value)


def consume_held_out(args):
    manifest = _read_json(args.manifest)
    oracle = _read_json(args.oracle_receipt)
    frozen = _read_json(args.frozen_contract)
    _require(frozen.get("schema") == FROZEN_SCHEMA, "frozen contract mismatch")
    _require(oracle.get("manifest_sha256") == _sha256(args.manifest),
             "Oracle receipt is not bound to this manifest")
    value = {
        "schema": "refiner_v15_15g1f3_final_held_out_one_shot_receipt_v1",
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_id": manifest.get("transaction_id"),
        "manifest_sha256": _sha256(args.manifest),
        "oracle_receipt_sha256": _sha256(args.oracle_receipt),
        "frozen_contract_sha256": _sha256(args.frozen_contract),
        "rerun_allowed": False,
        "failure_policy": (
            "transaction_becomes_development_evidence_and_a_new_unseen_"
            "transaction_must_be_sealed"
        ),
    }
    _write_exclusive(args.output, value)


def verify_held_out(args):
    report = _read_json(args.report)
    receipt = _read_json(args.one_shot_receipt)
    manifest = _read_json(args.manifest)
    _require(report.get("schema") == REPORT_SCHEMA, "held-out schema mismatch")
    _require(report.get("evaluation_role") == "final_held_out",
             "held-out role mismatch")
    _require(report.get("numeric_audit_complete") is True,
             "held-out numeric audit failed")
    _unchanged_contract(report)
    summary = _summary(report)
    _require(summary.get("cross_exact_closure_complete") is True,
             "held-out cross_long closure failed")
    _require(summary.get("single_identity_safe") is True,
             "held-out single case was activated")
    _require(summary.get("adapter_incumbent_preserved") is True,
             "held-out replaced a closed cross-short incumbent")
    _require(summary.get("scope_safe") is True, "held-out scope leaked")
    _require_zero_scope(summary, "held-out")
    _require_conformal_score_source(
        summary, "final_train_models", "held-out"
    )
    _require(summary.get("runtime_case_whitelist_used") is False,
             "held-out used a case whitelist")
    states = {
        str(row.get("state"))
        for row in (summary.get("composite_closure_by_case") or {}).values()
    }
    _require(not (states & FAILURE_STATES),
             "held-out contains abstention, curvature, or solver failure")
    _require(receipt.get("rerun_allowed") is False,
             "held-out receipt does not enforce one-shot use")
    _require(receipt.get("manifest_sha256") == _sha256(args.manifest),
             "held-out receipt/manifest mismatch")
    value = {
        "schema": "refiner_v15_15g1f3_final_held_out_acceptance_v1",
        "accepted": True,
        "transaction_id": manifest.get("transaction_id"),
        "report": str(Path(args.report).resolve()),
        "report_sha256": _sha256(args.report),
        "manifest_sha256": _sha256(args.manifest),
        "one_shot_receipt_sha256": _sha256(args.one_shot_receipt),
    }
    _write_exclusive(args.output, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare-replicates")
    compare.add_argument("--report", action="append", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(function=compare_replicates)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--train-report", required=True)
    freeze.add_argument("--dev-report", required=True)
    freeze.add_argument("--cold-start-acceptance", required=True)
    freeze.add_argument("--repair-contract", required=True)
    freeze.add_argument("--conformal-envelope", required=True)
    freeze.add_argument("--train-manifest", required=True)
    freeze.add_argument("--dev-manifest", required=True)
    freeze.add_argument("--implementation-commit", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(function=freeze_contract)

    seal = sub.add_parser("seal-held-out")
    seal.add_argument("--candidate-bank", required=True)
    seal.add_argument("--train-bank", required=True)
    seal.add_argument("--dev-bank", required=True)
    seal.add_argument("--selection-seed", required=True)
    seal.add_argument("--exclude-manifest", action="append")
    seal.add_argument("--output-bank", required=True)
    seal.add_argument("--output-manifest", required=True)
    seal.set_defaults(function=seal_held_out)

    oracle = sub.add_parser("record-oracle")
    oracle.add_argument("--manifest", required=True)
    oracle.add_argument("--oracle-report", required=True)
    oracle.add_argument("--output", required=True)
    oracle.set_defaults(function=record_oracle)

    consume = sub.add_parser("consume-held-out")
    consume.add_argument("--manifest", required=True)
    consume.add_argument("--oracle-receipt", required=True)
    consume.add_argument("--frozen-contract", required=True)
    consume.add_argument("--output", required=True)
    consume.set_defaults(function=consume_held_out)

    verify = sub.add_parser("verify-held-out")
    verify.add_argument("--report", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--one-shot-receipt", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(function=verify_held_out)

    args = parser.parse_args()
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
