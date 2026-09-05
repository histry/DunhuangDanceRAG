"""Opt-in generation traces and an adapter to the development feasibility runner.

No training or solver is executed by this module. Existing reports can be
summarized without loading torch, checkpoints or motion arrays.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

METRICS = (
    "joint_jerk_mps3_max", "foot_skate_mps_p95", "foot_skate_mps_max",
    "foot_support_drift_m_p95", "foot_support_drift_m_max",
    "foot_penetration_min_m", "contact_ratio", "foot_support_ratio",
)


def selected_diagnostic_slots():
    return {int(value) for value in os.environ.get(
        "BOUNDARY_DIAGNOSTIC_SLOTS", "25,26"
    ).split(",") if value.strip()}


def _write(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(directory, name, value):
    path = Path(directory) / (name + ".npy")
    np.save(path, np.asarray(value), allow_pickle=False)
    return {"path": str(path.resolve()), "sha256": _hash(path), "shape": list(value.shape)}


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), *args],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_guarded_candidate(directory, name, value):
    result = _array(directory, name + "_guarded_candidate", value)
    result["scope"] = "after_runtime_internal_guards_before_outer_transaction; not raw_network_output"
    return result


def save_upstream_stages(directory, arrays, event_path, runtime, cfg):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, value in arrays.items():
        entry = _array(directory, name, value)
        if len(value) >= 4:
            # These are descriptive local audits, NOT production acceptance.
            entry["audit"] = runtime.audit_motion_np(value, cfg, support_policy="source_observation")
        else:
            entry["audit_unavailable"] = "fewer_than_four_frames"
        entries[name] = entry
    report = {
        "schema": "generation_upstream_trace_v1", "stages": entries,
        "event_path": event_path, "event_sha256": _hash(event_path),
        "scope": "selected_candidate_only",
        "audit_policy": "source_observation; local floor and duration vary between stages",
        "production_acceptance": False,
    }
    _write(directory / "stages.json", report)
    return {"path": str((directory / "stages.json").resolve()), "sha256": _hash(directory / "stages.json")}


def summarize_report(report):
    stages = report.get("stage_reports", {})
    ik = stages.get("lower_body_ik_true_ik", {})
    local = ik.get("local_transactions", {})
    refiner = stages.get("boundary_refiner_transaction", {})
    diffusion = stages.get("motion_diffusion_transaction", {})
    env = report.get("closed_loop", {}).get("env", {})
    audits = {name: stages.get(name, {}) for name in ("pre_refine_audit", "boundary_refiner_audit", "motion_diffusion_audit", "final_audit")}
    audits.update({"ik_before": ik.get("audit_before", {}), "ik_candidate": ik.get("audit_after_candidate", {})})
    return {
        "schema": "generation_stage_diagnosis_v1",
        "scope": "report_evidence_only; equal metrics do not prove equal arrays",
        "switches": {key: env.get(key) for key in ("BOUNDARY_USE_REFINER", "BOUNDARY_USE_DIFFUSION", "BOUNDARY_USE_IK")},
        "metrics": {name: {key: value.get(key) for key in METRICS} for name, value in audits.items()},
        "transactions": {
            name: {key: transaction.get(key) for key in ("accepted", "rolled_back", "reasons", "exception")}
            for name, transaction in (("refiner", refiner), ("diffusion", diffusion))
        },
        "ik_local_transactions": {key: local.get(key) for key in ("attempted", "accepted", "rejected")},
        "ik_rejections": {field: dict(Counter(reason for row in local.get("transactions", []) for reason in row.get(field, [])))
                          for field in ("relative_reasons", "absolute_reasons", "kbo_reasons")},
        "ik_rollback": {key: ik.get(key) for key in ("rollback_triggered", "stage_guard_ik_rollback_to_fk", "stage_guard_rollback_reasons")},
        "assembly_decisions": dict(Counter(row.get("decision", "missing") for row in stages.get("closed_loop_concat", []))),
        "rounds": [{"round": row.get("round"), "unsafe_boundaries": row.get("unsafe_boundaries")} for row in report.get("closed_loop", {}).get("rounds", [])],
        "final_quality_gate": report.get("final_quality_gate"),
    }


def save_round_bundle(directory, round_id, reference, final, condition, seam,
                      slide, assembly, stages, cfg, db, args):
    arrays = {name: _array(directory, name, value) for name, value in (
        ("reference", reference), ("final", final), ("condition", condition),
        ("seam", seam), ("sliding_support_eligible", slide),
    )}
    checkpoints = {}
    for name in ("refiner", "diffusion"):
        path = getattr(args, name, None)
        active = bool(getattr(cfg, name + "_enable", False)) and os.environ.get("BOUNDARY_USE_" + name.upper(), "1") != "0"
        checkpoints[name] = {"active": active, "path": path,
                             "sha256": _hash(path) if active and path and Path(path).is_file() else None}
    config_path = Path(directory) / "config.json"
    _write(config_path, dataclasses.asdict(cfg))
    report = {
        "schema": "generation_round_bundle_v1", "round": int(round_id),
        "arrays": arrays, "assembly": assembly, "stage_reports": stages,
        "config_path": str(config_path.resolve()), "config_sha256": _hash(config_path),
        "source_commit": _git("rev-parse", "HEAD"), "dirty_state": _git("status", "--porcelain"),
        "tracked_diff_sha256": hashlib.sha256((_git("diff", "HEAD") or "").encode()).hexdigest(),
        "checkpoints": checkpoints,
        "runtime_environment": {k: v for k, v in os.environ.items() if k.startswith(("BOUNDARY_", "MOTION_", "CONTACT_", "STAGE_GUARD_"))},
        "development_only": True, "solver_executed": False,
    }
    path = Path(directory) / "bundle.json"
    _write(path, report)
    return {"path": str(path.resolve()), "sha256": _hash(path), "round": int(round_id)}


def export_cases(bundle_path, provenance_path, output, slots, context_frames):
    # Explicit provenance prevents a generated/test sequence being silently
    # relabelled as development data. Never infer identity from file names.
    from training import motion_models as runtime
    from training.refiner_action_feasibility_evaluation import (
        MANIFEST_SCHEMA,
        load_case_manifest,
    )

    bundle_path = Path(bundle_path).resolve()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    if bundle.get("schema") != "generation_round_bundle_v1":
        raise ValueError("export requires a captured generation_round_bundle_v1")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8-sig"))
    if provenance.get("development_only") is not True:
        raise ValueError("explicit development_only provenance is required")
    if context_frames < 4:
        raise ValueError("at least four context frames are required")
    if _hash(bundle["config_path"]) != bundle["config_sha256"]:
        raise ValueError("config hash mismatch")
    cfg = runtime.MotionGenerationConfig.from_json(bundle["config_path"])
    arrays = {}
    for name in ("reference", "condition", "seam"):
        entry = bundle["arrays"][name]
        if _hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"array hash mismatch: {name}")
        arrays[name] = np.load(entry["path"], allow_pickle=False)
    rows = bundle["assembly"]
    by_slot = {int(row["slot"]): index for index, row in enumerate(rows)}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    case_rows = []
    events = provenance["events"]
    for slot in slots:
        index = by_slot[slot]
        if index == 0:
            raise ValueError("first slot has no preceding boundary")
        row, previous = rows[index], rows[index - 1]
        left, right = events[previous["event_path"]], events[row["event_path"]]
        if left["split"] != right["split"] or left["split"] not in {"train", "dev", "development", "val", "validation"}:
            raise ValueError("boundary sources must share a development split")
        for event, identity in ((previous, left), (row, right)):
            if not identity.get("source_uid") or not identity.get("recording_uid"):
                raise ValueError("source_uid and recording_uid are mandatory")
            if _hash(event["event_path"]) != identity["event_sha256"]:
                raise ValueError("event provenance hash mismatch")
        if row.get("length_policy", {}).get("slot_exact_repair_applied"):
            raise ValueError("exact-length repair changed the span; export requires corrected provenance")
        start, end = map(int, row["transition_span"])
        lo, hi = max(0, start - context_frames), min(len(arrays["reference"]), end + context_frames)
        if start - lo < 4 or hi - end < 4:
            raise ValueError("insufficient observed boundary context")
        ref = arrays["reference"][lo:hi].copy()
        seam = np.zeros((len(ref), 1), dtype=np.float32)
        # Preserve this boundary's halo only; neighbouring seam cores are not
        # silently included in the single-boundary optimization objective.
        halo = round(cfg.transition_mask_halo_seconds * cfg.fps)
        a, b = max(0, start - lo - halo), min(len(ref), end - lo + halo)
        seam[a:b] = arrays["seam"][lo + a:lo + b]
        seam[start - lo:end - lo] = 1.0
        joint, root, contact = runtime._risk_masks_for_batch_np(ref[None], seam[None], cfg)
        t = runtime.torch
        # This helper consults two environment overrides. Reject conflicting
        # values rather than silently changing the captured decoder strengths.
        for kind in ("core", "transition"):
            override = os.environ.get("MOTION_REFINER_" + kind.upper() + "_STRENGTH")
            if override is not None and float(override) != float(getattr(cfg, "refiner_" + kind + "_strength")):
                raise ValueError("decoder strength environment differs from captured config")
        joint, root, _ = runtime._refiner_decode_masks(
            t.as_tensor(joint), t.as_tensor(root), t.as_tensor(contact), t.as_tensor(seam[None]), cfg,
        )
        directory = output / f"slot_{slot:03d}"
        directory.mkdir()
        paths = {}
        for name, value in (("reference", ref), ("seam", seam), ("condition", arrays["condition"][lo:hi]),
                            ("joint_mask", joint[0].numpy()), ("root_mask", root[0].numpy()),
                            ("contact_mask", np.zeros((len(ref), 4), dtype=np.float32))):
            paths[name + "_path"] = _array(directory, name, value)["path"]
        case_rows.append({
            "case_id": f"round{bundle['round']}_slot{slot}", "role": "cross_event",
            "width": end - start, "position_stratum": right["position_stratum"],
            "split": right["split"], "left_split": left["split"], "right_split": right["split"],
            "source_uid": right["source_uid"], "recording_uid": right["recording_uid"],
            "left_source_uid": left["source_uid"], "right_source_uid": right["source_uid"],
            "left_recording_uid": left["recording_uid"], "right_recording_uid": right["recording_uid"],
            "metadata": {"bundle_sha256": _hash(bundle_path), "frame_span": [lo, hi],
                         "mask_policy": "recomputed_local_development_masks; not original production masks",
                         "geometry_only": True, "production_replay_equivalent": False}, **paths,
        })
    manifest = output / "cases.json"
    _write(manifest, {"schema": MANIFEST_SCHEMA, "cases": case_rows,
                      "provenance_sha256": _hash(provenance_path)})
    load_case_manifest(manifest, cfg)
    _write(output / "config.json", dataclasses.asdict(cfg))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--report", required=True)
    summary.add_argument("--output", required=True)
    export = commands.add_parser("export")
    export.add_argument("--bundle", required=True)
    export.add_argument("--provenance", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--slots", type=int, nargs="+", default=[25, 26])
    export.add_argument("--context-frames", type=int, default=16)
    args = parser.parse_args()
    if args.command == "export":
        print(export_cases(args.bundle, args.provenance, args.output_dir, args.slots, args.context_frames))
    else:
        path = Path(args.output)
        if path.exists():
            raise FileExistsError(path)
        report = json.loads(Path(args.report).read_text(encoding="utf-8-sig"))
        result = summarize_report(report)
        result["report_sha256"] = _hash(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
