"""Build a frozen V15.15 Adapter teacher bank from V15.14h reports.

This development-only command aggregates exact-closure projected directions.
It performs no optimization and cannot publish or train a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from motion_geometry.product_manifold import product_log_torch
from training import motion_models as m
from training import refiner_projected_candidate_probe as projected_probe


SCHEMA = "refiner_v15_15_observable_adapter_teacher_bank_v1"


def _cpu_tree(value):
    if m.torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    return value


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
    batch, _, schedule = projected_probe._materialize_first_transaction(
        artifact, device
    )
    model = m.ProductManifoldTemporalRefiner(
        fps=cfg.fps,
        film_conditioning=bool(cfg.product_refiner_film_conditioning),
        residual_taper_frames=int(cfg.product_refiner_residual_taper_frames),
    ).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    baseline, identity = projected_probe._model_prediction(model, batch, cfg)
    return report, source, batch, baseline, identity, schedule


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

    source_key = None
    retained_batch = None
    retained_baseline = None
    retained_identity = None
    retained_schedule = None
    samples = []
    evidence = []
    seen = set()
    control_difficulties = []
    for report_path in report_paths:
        report, source, batch, baseline, identity, schedule = _load_context(
            report_path, cfg, device
        )
        current_key = str(source.resolve())
        if source_key is None:
            source_key = current_key
            retained_batch = _cpu_tree({
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
            retained_baseline = baseline.detach().cpu()
            retained_identity = identity.detach().cpu()
            retained_schedule = list(schedule)
        elif current_key != source_key:
            raise RuntimeError(
                "one teacher bank must use one frozen diagnostic context"
            )
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
            key = (current_key, case_index)
            if key in seen:
                continue
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
                "case_index": case_index,
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
            evidence.append({
                "sample_index": sample_index,
                "case_index": case_index,
                "audit_group": group,
                "teacher_kind": "identity_control",
                "oracle_report": None,
                "oracle_implementation_commit": None,
                "control_source": "frozen_transaction_single_case",
                "projection_audit": None,
            })
        for case_report in report.get("case_reports", []):
            case_index = int(case_report["case_index"])
            key = (current_key, case_index)
            if key in seen:
                continue
            role = str(case_report.get("role"))
            group = str(case_report.get("group"))
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
                candidate_path = report_path.parent / (
                    f"projected_case_{case_index}.npy"
                )
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
            sample_index = len(samples)
            samples.append({
                "sample_index": sample_index,
                "case_index": case_index,
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
                "case_index": case_index,
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

    if retained_batch is None or not samples:
        raise RuntimeError("no eligible V15.14h teacher samples were found")
    projected_count = sum(
        sample["teacher_kind"] == "exact_projected_direction"
        for sample in samples
    )
    if projected_count < 1:
        raise RuntimeError("teacher bank contains no projected oracle direction")
    projected_by_group = {
        group: sum(
            sample["teacher_kind"] == "exact_projected_direction"
            and sample["audit_group"] == group
            for sample in samples
        )
        for group in ("cross_short", "cross_long")
    }
    if not all(projected_by_group.values()):
        raise RuntimeError(
            f"both cross groups require a projected teacher: {projected_by_group}"
        )
    calibration_floor = max(control_difficulties, default=0.0)
    payload = {
        "schema": SCHEMA,
        "development_only": True,
        "formal_training_allowed": False,
        "source_diagnostic": source_key,
        "transaction_context_indices": retained_schedule,
        "batch": retained_batch,
        "baseline_prediction": retained_baseline,
        "baseline_identity": retained_identity,
        "samples": samples,
        "projected_teacher_count_by_group": projected_by_group,
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
        "schema": SCHEMA,
        "development_only": True,
        "implementation_commit": os.environ.get("EXPECTED_COMMIT"),
        "formal_training_allowed": False,
        "source_diagnostic": source_key,
        "oracle_reports": [str(path) for path in report_paths],
        "teacher_bank": str(bank_path.resolve()),
        "sample_count": len(samples),
        "projected_teacher_count": projected_count,
        "projected_teacher_count_by_group": projected_by_group,
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
        "identity_control_count": len(samples) - projected_count,
    }), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/motion_model.json")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
