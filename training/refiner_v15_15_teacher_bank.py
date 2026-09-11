"""Build a frozen V15.15 Adapter teacher bank from V15.14h reports.

This development-only command aggregates exact-closure projected directions.
It performs no optimization and cannot publish or train a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_log_torch
from training import motion_models as m
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15_observable_adapter_teacher_bank_v1"
MULTI_TRANSACTION_SCHEMA = (
    "refiner_v15_15e_multi_transaction_observable_adapter_teacher_bank_v1"
)


def _cpu_tree(value):
    if m.torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _concat_batch(left, right):
    if left is None:
        return _cpu_tree(right)
    if set(left) != set(right):
        raise RuntimeError("transaction batch fields differ")
    result = {}
    for key in left:
        if not m.torch.is_tensor(left[key]) or not m.torch.is_tensor(right[key]):
            raise TypeError(f"non-tensor transaction batch field: {key}")
        result[key] = m.torch.cat([left[key], right[key].detach().cpu()], dim=0)
    return result


def _manifest_contract(path, split):
    if path is None:
        return None, None
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != (
        "refiner_v15_15e_multi_transaction_teacher_manifest_v1"
    ):
        raise RuntimeError("unsupported V15.15e transaction manifest")
    if split not in {"train", "validation"}:
        raise RuntimeError("manifest-backed teacher bank requires a split")
    if manifest.get("train_validation_case_overlap"):
        raise RuntimeError("manifest case split overlaps")
    if manifest.get("train_validation_source_case_overlap"):
        raise RuntimeError("manifest source-case split overlaps")
    supplied_digest = manifest.get("manifest_content_sha256")
    canonical_payload = {
        key: value for key, value in manifest.items()
        if key != "manifest_content_sha256"
    }
    actual_digest = hashlib.sha256(json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if supplied_digest != actual_digest:
        raise RuntimeError("manifest content SHA256 mismatch")
    return manifest, manifest_path


def _manifest_case_maps(manifest, split):
    if manifest is None:
        return None, None
    oracle = {}
    controls = {}
    for transaction in manifest["transactions"]:
        if transaction["split"] != split:
            continue
        for row in transaction["oracle_cases"]:
            if row["case_uid"] in oracle:
                raise RuntimeError("duplicate manifest Oracle composite key")
            oracle[row["case_uid"]] = row
        for row in transaction["identity_control_cases"]:
            if row["case_uid"] in controls:
                raise RuntimeError("duplicate manifest control composite key")
            controls[row["case_uid"]] = row
    return oracle, controls


def _assign_stratified_sampling_weights(samples):
    density = Counter(
        (sample["audit_group"], sample["transaction_id"])
        for sample in samples
        if sample["teacher_kind"] == "exact_projected_direction"
    )
    transactions_per_group = Counter(
        group for group, _ in density
    )
    for sample in samples:
        if sample["teacher_kind"] != "exact_projected_direction":
            sample["stratified_sampling_weight"] = 0.0
            continue
        key = (sample["audit_group"], sample["transaction_id"])
        sample["stratified_sampling_weight"] = 1.0 / (
            2.0 * float(transactions_per_group[key[0]]) * float(density[key])
        )
    return {
        f"{group}/{transaction_id}": count
        for (group, transaction_id), count in sorted(density.items())
    }


def _passing_projection(case_report):
    projection = case_report.get("projector_result") or {}
    for trial in projection.get("projection_trials", []):
        if trial.get("audit", {}).get("passed"):
            return trial
    return None


def _load_context(report_path, cfg, device):
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if report.get("schema") != (
        "refiner_v15_14h_case_local_full_tangent_nonlinear_oracle_v1"
    ):
        raise RuntimeError(f"unsupported oracle report: {report_path}")
    if not report.get("numeric_audit_complete") or not report.get("scope_safe"):
        raise RuntimeError(f"incomplete oracle evidence: {report_path}")
    changed = [
        key
        for key in (
            "fixed_guard_thresholds_changed",
            "physical_gate_changed",
            "fixed_support_gate_changed",
            "fidelity_gate_changed",
            "boundary_gate_changed",
            "observable_0p03_gate_changed",
        )
        if report.get(key) is not False
    ]
    if changed:
        raise RuntimeError(f"oracle changed fixed gates: {changed}")
    source = Path(report["source_diagnostic"])
    state = m.torch.load(
        source / "diagnostic_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    artifact = m.torch.load(
        source / "fit_bank.pt",
        map_location="cpu",
        weights_only=False,
    )
    transaction_index = int(report.get("transaction_index", 0))
    anchor_batch, transaction_batch, schedule = (
        projected_probe._materialize_transaction(
            artifact, device, transaction_index
        )
    )
    batch = (
        transaction_batch
        if report.get("transaction_batch_materialized") else anchor_batch
    )
    transaction_id = projected_probe._transaction_identity(
        transaction_index, schedule
    )
    reported_transaction_id = report.get("transaction_id")
    if (
        reported_transaction_id is not None
        and reported_transaction_id != transaction_id
    ):
        raise RuntimeError("Oracle report transaction identity mismatch")
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
        residual_taper_frames=int(cfg.product_refiner_residual_taper_frames),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    baseline, identity = projected_probe._model_prediction(model, batch, cfg)
    if report.get("transaction_batch_materialized"):
        observed_anchor = projected_probe._float_guard(
            projected_probe._guard_values_for_prediction(
                model, batch, cfg, baseline, identity
            )
        )
        expected_anchor = report.get("fixed_guard_anchor") or {}
        if set(observed_anchor) != set(expected_anchor):
            raise RuntimeError("transaction Guard anchor layout mismatch")
        drift = {
            key: abs(float(observed_anchor[key]) - float(expected_anchor[key]))
            for key in observed_anchor
        }
        if max(drift.values(), default=0.0) > 1.0e-12:
            raise RuntimeError({
                "transaction_guard_anchor_replay_drift": max(drift.values())
            })
    return (
        report,
        source,
        batch,
        baseline,
        identity,
        schedule,
        transaction_index,
        transaction_id,
    )


def run(args):
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_paths = [Path(value).resolve() for value in args.oracle_report]
    if len({str(path) for path in report_paths}) != len(report_paths):
        raise RuntimeError("duplicate oracle reports are forbidden")
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    device = m.torch.device(cfg.device)
    if device.type != "cuda":
        raise RuntimeError("V15.15 teacher-bank construction requires CUDA")

    manifest, manifest_path = _manifest_contract(
        args.split_manifest, args.split
    )
    manifest_oracle, manifest_controls = _manifest_case_maps(
        manifest, args.split
    )

    source_key = None
    retained_batch = None
    retained_baseline = None
    retained_identity = None
    retained_schedules = {}
    retained_guard_contracts = {}
    transaction_offsets = {}
    samples = []
    evidence = []
    seen = set()
    audited_manifest_oracle = set()
    consumed_manifest_oracle = set()
    consumed_manifest_controls = set()
    control_difficulties = []
    for report_path in report_paths:
        (
            report,
            source,
            batch,
            baseline,
            identity,
            schedule,
            transaction_index,
            transaction_id,
        ) = _load_context(report_path, cfg, device)
        current_key = str(source.resolve())
        if source_key is None:
            source_key = current_key
            if manifest is not None:
                if str(Path(manifest["source_diagnostic"]).resolve()) != source_key:
                    raise RuntimeError("manifest source diagnostic mismatch")
                if _file_sha256(source / "fit_bank.pt") != manifest[
                    "fit_bank_sha256"
                ]:
                    raise RuntimeError("manifest fit-bank SHA256 mismatch")
        elif current_key != source_key:
            raise RuntimeError(
                "one teacher bank must use one frozen diagnostic context"
            )

        if manifest is not None:
            manifest_transaction = next(
                (
                    row for row in manifest["transactions"]
                    if row["transaction_id"] == transaction_id
                ),
                None,
            )
            if manifest_transaction is None:
                raise RuntimeError("Oracle transaction absent from manifest")
            if manifest_transaction["split"] != args.split:
                raise RuntimeError("Oracle report belongs to the other split")
            if report.get("teacher_split") != args.split:
                raise RuntimeError("Oracle report split identity mismatch")
            if report.get("pre_oracle_manifest_sha256") != (
                _file_sha256(manifest_path)
            ):
                raise RuntimeError("Oracle report manifest SHA256 mismatch")
            if not report.get("transaction_batch_materialized"):
                raise RuntimeError("manifest Oracle did not audit a transaction")
            if report.get("fixed_guard_anchor_source") != (
                "immutable_transaction_baseline_pre_oracle"
            ):
                raise RuntimeError("manifest Oracle Guard anchor domain mismatch")

        if transaction_id not in transaction_offsets:
            offset = (
                0 if retained_batch is None
                else int(retained_batch["bad"].shape[0])
            )
            transaction_offsets[transaction_id] = offset
            transaction_batch = _cpu_tree({
                key: batch[key]
                for key in (
                    # Repair-path inputs.
                    "bad",
                    "cond",
                    "seam",
                    "joint",
                    "root",
                    "contact",
                    "group",
                    # Frozen audit-only identity inputs. These never enter the
                    # Adapter forward call, but the unchanged fixed Guard
                    # requires the same fidelity/identity domain as V15.14h.
                    "clean",
                    "clean_joint",
                    "clean_root",
                    "clean_contact",
                )
            })
            retained_batch = _concat_batch(retained_batch, transaction_batch)
            retained_baseline = (
                baseline.detach().cpu()
                if retained_baseline is None
                else m.torch.cat(
                    [retained_baseline, baseline.detach().cpu()], dim=0
                )
            )
            retained_identity = (
                identity.detach().cpu()
                if retained_identity is None
                else m.torch.cat(
                    [retained_identity, identity.detach().cpu()], dim=0
                )
            )
            retained_schedules[transaction_id] = {
                "transaction_index": int(transaction_index),
                "context_indices": list(schedule),
                "global_case_offset": offset,
                "case_count": int(batch["bad"].shape[0]),
            }
            source_contract = json.loads(
                (source / "diagnostic_report.json").read_text(
                    encoding="utf-8-sig"
                )
            )["group_guard_contract"]
            retained_guard_contracts[transaction_id] = {
                "initial_anchor": dict(report.get(
                    "fixed_guard_anchor", source_contract["initial_anchor"]
                )),
                "relative_tolerance": dict(report.get(
                    "fixed_guard_relative_tolerance",
                    source_contract["relative_tolerance"],
                )),
                "absolute_tolerance": dict(report.get(
                    "fixed_guard_absolute_tolerance",
                    source_contract["absolute_tolerance"],
                )),
                "anchor_source": report.get(
                    "fixed_guard_anchor_source",
                    "source_diagnostic_seen_initial_anchor",
                ),
            }
        offset = transaction_offsets[transaction_id]
        features, difficulty, taper, ownership = (
            m._refiner_observable_adapter_features(
                batch["bad"],
                batch["seam"],
                cfg.fps,
                cfg.product_refiner_residual_taper_frames,
            )
        )
        # Calibrate the conservative dead-zone against every single case in
        # the frozen transaction, rather than only the two V15.14h controls.
        # The role is offline metadata; it is never serialized into the model
        # input or consumed by Adapter inference.
        for case_index in range(int(batch["bad"].shape[0])):
            group = m.REFINER_GROUP_LABELS[
                int(batch["group"][case_index].detach())
            ]
            if not group.startswith("single_"):
                continue
            case_uid = f"{transaction_id}:{case_index}"
            if manifest is not None and case_uid not in manifest_controls:
                continue
            key = (current_key, case_uid)
            if key in seen:
                continue
            global_case_index = offset + case_index
            active = batch["seam"][case_index, :, 0] >= 0.5
            difficulty_value = float(
                difficulty[case_index, active].amax().detach()
            ) if bool(active.any()) else 0.0
            teacher = m.torch.zeros(
                (int(baseline.shape[1]), 75),
                dtype=baseline.dtype,
                device=baseline.device,
            )
            seen.add(key)
            sample_index = len(samples)
            samples.append({
                "sample_index": sample_index,
                "case_index": global_case_index,
                "local_case_index": case_index,
                "transaction_id": transaction_id,
                "transaction_index": int(transaction_index),
                "case_uid": case_uid,
                "source_case_uid": (
                    manifest_controls[case_uid]["source_case_uid"]
                    if manifest is not None else case_uid
                ),
                "audit_group": group,
                "teacher_kind": "identity_control",
                "teacher_tangent": teacher.detach().cpu(),
                "observable_condition": features[case_index].detach().cpu(),
                "difficulty": difficulty[case_index].detach().cpu(),
                "c2_taper": taper[case_index].detach().cpu(),
                "ownership": ownership[case_index].detach().cpu(),
                "difficulty_max": difficulty_value,
                "oracle_report": None,
                "projection_backtracking_factor": None,
            })
            control_difficulties.append(difficulty_value)
            consumed_manifest_controls.add(case_uid)
            evidence.append({
                "sample_index": sample_index,
                "case_index": global_case_index,
                "local_case_index": case_index,
                "transaction_id": transaction_id,
                "case_uid": case_uid,
                "source_case_uid": (
                    manifest_controls[case_uid]["source_case_uid"]
                    if manifest is not None else case_uid
                ),
                "audit_group": group,
                "teacher_kind": "identity_control",
                "oracle_report": None,
                "oracle_implementation_commit": None,
                "control_source": "frozen_transaction_single_case",
                "projection_audit": None,
            })
        for case_report in report.get("case_reports", []):
            case_index = int(
                case_report.get("local_case_index", case_report["case_index"])
            )
            case_uid = case_report.get(
                "case_uid", f"{transaction_id}:{case_index}"
            )
            if not case_uid.startswith(f"{transaction_id}:"):
                raise RuntimeError("case report composite identity mismatch")
            if manifest is not None and case_uid not in manifest_oracle:
                continue
            key = (current_key, case_uid)
            if key in seen:
                continue
            role = str(case_report.get("role"))
            group = str(case_report.get("group"))
            if manifest is not None:
                if role != "primary":
                    raise RuntimeError("manifest Oracle case is not primary")
                manifest_row = manifest_oracle[case_uid]
                if (
                    int(manifest_row["case_index"]) != case_index
                    or manifest_row["group"] != group
                ):
                    raise RuntimeError("manifest Oracle case mismatch")
                audited_manifest_oracle.add(case_uid)
            active = batch["seam"][case_index, :, 0] >= 0.5
            difficulty_value = float(
                difficulty[case_index, active].amax().detach()
            ) if bool(active.any()) else 0.0
            if role == "control":
                teacher = m.torch.zeros(
                    (int(baseline.shape[1]), 75),
                    dtype=baseline.dtype,
                    device=baseline.device,
                )
                control_difficulties.append(difficulty_value)
                teacher_kind = "identity_control"
                projection = None
            else:
                projection = _passing_projection(case_report)
                if projection is None:
                    continue
                if (
                    not case_report.get("raw_exact_candidate_found")
                    or not case_report.get("effective_projected_candidate")
                    or not projection.get("audit", {}).get("passed")
                ):
                    raise RuntimeError(
                        "projected teacher lacks complete exact-closure evidence"
                    )
                candidate_file = case_report.get(
                    "projected_candidate_file",
                    f"projected_case_{case_index}.npy",
                )
                candidate_path = report_path.parent / candidate_file
                if not candidate_path.is_file():
                    raise FileNotFoundError(candidate_path)
                candidate = m.torch.as_tensor(
                    np.load(candidate_path),
                    dtype=baseline.dtype,
                    device=device,
                )
                teacher = product_log_torch(
                    baseline[case_index:case_index + 1],
                    candidate[case_index:case_index + 1],
                )[0]
                teacher_kind = "exact_projected_direction"
            seen.add(key)
            consumed_manifest_oracle.add(case_uid)
            sample_index = len(samples)
            global_case_index = offset + case_index
            samples.append({
                "sample_index": sample_index,
                "case_index": global_case_index,
                "local_case_index": case_index,
                "transaction_id": transaction_id,
                "transaction_index": int(transaction_index),
                "case_uid": case_uid,
                "source_case_uid": (
                    manifest_oracle[case_uid]["source_case_uid"]
                    if manifest is not None else case_uid
                ),
                "audit_group": group,
                "teacher_kind": teacher_kind,
                "teacher_tangent": teacher.detach().cpu(),
                "observable_condition": features[case_index].detach().cpu(),
                "difficulty": difficulty[case_index].detach().cpu(),
                "c2_taper": taper[case_index].detach().cpu(),
                "ownership": ownership[case_index].detach().cpu(),
                "difficulty_max": difficulty_value,
                "oracle_report": str(report_path),
                "projection_backtracking_factor": (
                    projection.get("factor") if projection else None
                ),
            })
            evidence.append({
                "sample_index": sample_index,
                "case_index": global_case_index,
                "local_case_index": case_index,
                "transaction_id": transaction_id,
                "case_uid": case_uid,
                "source_case_uid": (
                    manifest_oracle[case_uid]["source_case_uid"]
                    if manifest is not None else case_uid
                ),
                "audit_group": group,
                "teacher_kind": teacher_kind,
                "oracle_report": str(report_path),
                "oracle_implementation_commit": report.get(
                    "implementation_commit"
                ),
                "projection_audit": (
                    projection.get("audit") if projection else None
                ),
            })

    if manifest is not None:
        missing_oracle_audits = (
            set(manifest_oracle) - audited_manifest_oracle
        )
        if missing_oracle_audits:
            raise RuntimeError({
                "manifest_oracle_cases_not_audited": sorted(
                    missing_oracle_audits
                )
            })
        missing_controls = set(manifest_controls) - consumed_manifest_controls
        if missing_controls:
            raise RuntimeError({
                "manifest_identity_controls_missing": sorted(missing_controls)
            })

    if retained_batch is None or not samples:
        raise RuntimeError("no eligible V15.14h teacher samples were found")
    projected_count = sum(
        sample["teacher_kind"] == "exact_projected_direction"
        for sample in samples
    )
    projected_by_group = {
        group: sum(
            sample["teacher_kind"] == "exact_projected_direction"
            and sample["audit_group"] == group
            for sample in samples
        )
        for group in ("cross_short", "cross_long")
    }
    teacher_bank_ready = bool(all(projected_by_group.values()))
    if manifest is None and not teacher_bank_ready:
        raise RuntimeError(
            f"both cross groups require a projected teacher: {projected_by_group}"
        )
    stratum_density = _assign_stratified_sampling_weights(samples)
    calibration_floor = max(control_difficulties, default=0.0)
    schema = MULTI_TRANSACTION_SCHEMA if manifest is not None else SCHEMA
    transaction_context_indices = (
        retained_schedules
        if manifest is not None
        else next(iter(retained_schedules.values()))["context_indices"]
    )
    manifest_file_sha256 = (
        _file_sha256(manifest_path) if manifest_path is not None else None
    )
    oracle_without_teacher = (
        sorted(set(manifest_oracle) - consumed_manifest_oracle)
        if manifest is not None else []
    )
    payload = {
        "schema": schema,
        "development_only": True,
        "formal_training_allowed": False,
        "source_diagnostic": source_key,
        "transaction_context_indices": transaction_context_indices,
        "transaction_schedules": retained_schedules,
        "transaction_guard_contracts": retained_guard_contracts,
        "split": args.split if manifest is not None else None,
        "split_manifest": (
            str(manifest_path) if manifest_path is not None else None
        ),
        "split_manifest_file_sha256": manifest_file_sha256,
        "split_manifest_content_sha256": (
            manifest.get("manifest_content_sha256")
            if manifest is not None else None
        ),
        "train_validation_case_overlap": [],
        "train_validation_source_case_overlap": [],
        "batch": retained_batch,
        "baseline_prediction": retained_baseline,
        "baseline_identity": retained_identity,
        "samples": samples,
        "projected_teacher_count_by_group": projected_by_group,
        "teacher_bank_ready": teacher_bank_ready,
        "oracle_case_uids_without_teacher": oracle_without_teacher,
        "teacher_deduplication_key": "transaction_id:local_case_index",
        "sampling_weight_protocol": (
            "inverse_group_transaction_density_equal_group_mass_v1"
        ),
        "projected_teacher_count_by_group_transaction": stratum_density,
        "observable_adapter_gate_floor": calibration_floor,
        "role_label_consumed_at_inference": False,
        "role_label_used_for_offline_audit_and_calibration": True,
        "hidden_clean_consumed_by_adapter": False,
        "hidden_clean_used_for_fixed_guard_audit_only": True,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
    }
    bank_path = destination / "observable_adapter_teacher_bank.pt"
    m.torch.save(payload, bank_path)
    report = {
        "schema": schema,
        "development_only": True,
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "formal_training_allowed": False,
        "source_diagnostic": source_key,
        "split": args.split if manifest is not None else None,
        "split_manifest": (
            str(manifest_path) if manifest_path is not None else None
        ),
        "split_manifest_file_sha256": manifest_file_sha256,
        "split_manifest_content_sha256": (
            manifest.get("manifest_content_sha256")
            if manifest is not None else None
        ),
        "transaction_schedules": retained_schedules,
        "transaction_guard_contracts": retained_guard_contracts,
        "train_validation_case_overlap": [],
        "train_validation_source_case_overlap": [],
        "oracle_reports": [str(path) for path in report_paths],
        "teacher_bank": str(bank_path.resolve()),
        "sample_count": len(samples),
        "projected_teacher_count": projected_count,
        "projected_teacher_count_by_group": projected_by_group,
        "projected_teacher_count_by_group_transaction": stratum_density,
        "teacher_bank_ready": teacher_bank_ready,
        "teacher_deduplication_key": "transaction_id:local_case_index",
        "sampling_weight_protocol": (
            "inverse_group_transaction_density_equal_group_mass_v1"
        ),
        "oracle_case_uids_without_teacher": oracle_without_teacher,
        "identity_control_count": len(samples) - projected_count,
        "observable_adapter_gate_floor": calibration_floor,
        "role_label_consumed_at_inference": False,
        "hidden_clean_consumed_by_adapter": False,
        "hidden_clean_used_for_fixed_guard_audit_only": True,
        "fixed_guard_thresholds_changed": False,
        "physical_gate_changed": False,
        "fixed_support_gate_changed": False,
        "fidelity_gate_changed": False,
        "boundary_gate_changed": False,
        "observable_0p03_gate_changed": False,
        "evidence": evidence,
    }
    report_path = destination / "observable_adapter_teacher_bank.report.json"
    m.save_json(report, report_path)
    print(json.dumps({
        "stage": "v15_15_teacher_bank_complete",
        "report": str(report_path.resolve()),
        "teacher_bank": str(bank_path.resolve()),
        "sample_count": len(samples),
        "projected_teacher_count": projected_count,
        "projected_teacher_count_by_group": projected_by_group,
        "projected_teacher_count_by_group_transaction": stratum_density,
        "teacher_bank_ready": teacher_bank_ready,
        "split": args.split if manifest is not None else None,
        "identity_control_count": len(samples) - projected_count,
    }), flush=True)
    return 0 if teacher_bank_ready else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=("train", "validation"))
    args = parser.parse_args()
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be supplied together")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
