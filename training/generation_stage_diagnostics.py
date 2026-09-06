"""Opt-in generation traces and an adapter to the development feasibility runner.

No training or solver is executed by this module. Existing reports can be
summarized without loading torch, checkpoints or motion arrays.
"""
from __future__ import annotations

import argparse
import contextlib
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


def _load_hashed_array(entry, label):
    if not isinstance(entry, dict):
        raise ValueError(f"missing array entry: {label}")
    path = Path(entry.get("path", "")).resolve()
    expected = str(entry.get("sha256", "")).strip().lower()
    if not path.is_file() or not expected:
        raise ValueError(f"incomplete array entry: {label}")
    actual = _hash(path)
    if actual != expected:
        raise ValueError(f"array hash mismatch: {label}")
    value = np.load(path, allow_pickle=False)
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite array: {label}")
    return value, {"path": str(path), "sha256": actual, "shape": list(value.shape)}


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


def export_cases(
    bundle_path,
    provenance_path,
    output,
    slots,
    context_frames,
    proposal_stage=None,
):
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
        arrays[name], _ = _load_hashed_array(bundle["arrays"][name], name)
    proposal = None
    proposal_info = None
    proposal_checkpoint = None
    if proposal_stage is not None:
        if proposal_stage != "refiner":
            raise ValueError("only the captured refiner proposal is supported")
        proposal_key = "refiner_guarded_candidate"
        proposal, proposal_info = _load_hashed_array(
            bundle.get("stage_reports", {}).get(proposal_key), proposal_key
        )
        if proposal.shape != arrays["reference"].shape:
            raise ValueError("captured refiner proposal shape mismatch")
        proposal_checkpoint = bundle.get("checkpoints", {}).get("refiner", {})
        if (
            not isinstance(proposal_checkpoint, dict)
            or proposal_checkpoint.get("active") is not True
            or not proposal_checkpoint.get("sha256")
        ):
            raise ValueError("captured refiner proposal has no active checkpoint SHA256")
        checkpoint_path = Path(proposal_checkpoint.get("path", "")).resolve()
        if (
            not checkpoint_path.is_file()
            or _hash(checkpoint_path) != proposal_checkpoint["sha256"]
        ):
            raise ValueError("captured refiner checkpoint SHA256 mismatch")
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
        proposal_metadata = {}
        if proposal is not None:
            proposal_artifact = _array(
                directory, "proposal_motion", proposal[lo:hi].copy()
            )
            paths["proposal_motion_path"] = proposal_artifact["path"]
            proposal_metadata = {
                "proposal_checkpoint_sha256": proposal_checkpoint["sha256"],
                "proposal_motion_sha256": proposal_artifact["sha256"],
                "proposal_full_motion_sha256": proposal_info["sha256"],
                "proposal_capture_scope": (
                    "guarded_refiner_candidate_after_internal_guards_"
                    "before_outer_stage_transaction"
                ),
            }
        case_rows.append({
            "case_id": f"round{bundle['round']}_slot{slot}", "role": "cross_event",
            "width": end - start, "position_stratum": right["position_stratum"],
            "split": right["split"], "left_split": left["split"], "right_split": right["split"],
            "source_uid": right["source_uid"], "recording_uid": right["recording_uid"],
            "left_source_uid": left["source_uid"], "right_source_uid": right["source_uid"],
            "left_recording_uid": left["recording_uid"], "right_recording_uid": right["recording_uid"],
            "metadata": {"bundle_sha256": _hash(bundle_path), "frame_span": [lo, hi],
                         "mask_policy": "recomputed_local_development_masks; not original production masks",
                         "geometry_only": True, "production_replay_equivalent": False,
                         **proposal_metadata}, **paths,
        })
    manifest = output / "cases.json"
    _write(manifest, {"schema": MANIFEST_SCHEMA, "cases": case_rows,
                      "provenance_sha256": _hash(provenance_path)})
    load_case_manifest(manifest, cfg)
    _write(output / "config.json", dataclasses.asdict(cfg))
    return manifest


@contextlib.contextmanager
def _captured_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _linked_source_report(source_report_path, bundle_path, bundle_sha256):
    source_report_path = Path(source_report_path).resolve()
    source_report = json.loads(source_report_path.read_text(encoding="utf-8-sig"))
    link = source_report.get("stage_reports", {}).get(
        "generation_stage_diagnostics", {}
    )
    if Path(link.get("path", "")).resolve() != bundle_path:
        raise ValueError("source report does not select this bundle")
    if str(link.get("sha256", "")).lower() != bundle_sha256:
        raise ValueError("source report bundle SHA256 mismatch")
    if not isinstance(source_report.get("slots"), list):
        raise ValueError("source report has no slot activity contract")
    return source_report, source_report_path


def _solution_rows(feasibility_run, baseline):
    run = Path(feasibility_run).resolve()
    report_path = run / "report.json"
    case_path = run / "case_level.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    artifact = report.get("artifacts", {}).get("case_level", {})
    if artifact:
        if Path(artifact.get("path", "")).resolve() != case_path:
            raise ValueError("feasibility case-level path mismatch")
        if _hash(case_path) != artifact.get("sha256"):
            raise ValueError("feasibility case-level SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in case_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("baseline") == baseline]
    expected = int(report.get("cases", {}).get("primary_cases", 0))
    if len(selected) != expected or expected < 1:
        raise ValueError("replay requires one selected baseline row per primary case")
    if any(
        row.get("status") != "VERIFIED_FEASIBLE"
        or row.get("final_joint_pass") is not True
        or not row.get("solution_artifacts")
        for row in selected
    ):
        raise ValueError("replay requires VERIFIED_FEASIBLE solution artifacts")
    return report, report_path, selected


def _quality_gate(closed_loop, physical, boundary, activity):
    failures = []
    if closed_loop.env_bool("ROUTING_SAFETY_REQUIRE_FINAL_PHYSICAL_GATE", True) and not physical["ok"]:
        failures.append("physical:" + ",".join(map(str, physical["reasons"])))
    if closed_loop.env_bool("BOUNDARY_REQUIRE_FINAL_BOUNDARY_GATE", True) and not boundary["ok"]:
        failures.append("boundary:" + ",".join(map(str, boundary["reasons"])))
    if closed_loop.env_bool("MOTION_ACTIVITY_FINAL_GATE", True) and not activity["ok"]:
        failures.append("activity:" + ",".join(map(str, activity["reasons"])))
    return {
        "schema": "final_motion_quality_layers_v1",
        "ok": not failures,
        "reasons": failures,
        "layers": {
            "anti_freeze_anti_collapse": {
                "ok": bool(activity["ok"]),
                "reasons": list(activity["reasons"]),
            },
            **dict(physical.get("layers", {})),
            "boundary_continuity": {
                "ok": bool(boundary["ok"]),
                "reasons": list(boundary["reasons"]),
            },
        },
        "rejected_output_is_renderable": False,
    }


def _apply_verified_solution_windows(reference, rows, bundle_sha256, edit_tolerance):
    if not np.isfinite(edit_tolerance) or float(edit_tolerance) < 0.0:
        raise ValueError("edit_tolerance must be finite and non-negative")
    repaired = np.asarray(reference).copy()
    occupied = np.zeros((len(repaired),), dtype=bool)
    applied = []
    for row in rows:
        solution = row["solution_artifacts"]
        if solution.get("bundle_sha256") != bundle_sha256:
            raise ValueError(f"solution bundle mismatch: {row.get('case_id')}")
        span = solution.get("frame_span")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("solution is missing the original frame_span")
        lo, hi = map(int, span)
        if not 0 <= lo < hi <= len(reference):
            raise ValueError("solution frame_span is outside the captured reference")
        artifacts = solution.get("artifacts", {})
        saved_reference, saved_reference_info = _load_hashed_array(
            artifacts.get("reference"), f"{row['case_id']}:reference"
        )
        returned_motion, returned_motion_info = _load_hashed_array(
            artifacts.get("returned_motion"), f"{row['case_id']}:returned_motion"
        )
        returned_action, returned_action_info = _load_hashed_array(
            artifacts.get("returned_action"), f"{row['case_id']}:returned_action"
        )
        proposal_action, proposal_action_info = _load_hashed_array(
            artifacts.get("proposal_action"), f"{row['case_id']}:proposal_action"
        )
        if (
            saved_reference.shape != reference[lo:hi].shape
            or returned_motion.shape != saved_reference.shape
            or returned_action.shape != (saved_reference.shape[0], 75)
            or proposal_action.shape != returned_action.shape
            or not np.array_equal(saved_reference, reference[lo:hi])
        ):
            raise ValueError(f"solution reference mismatch: {row.get('case_id')}")
        changed = np.max(
            np.abs(
                returned_motion.astype(np.float64)
                - saved_reference.astype(np.float64)
            ),
            axis=1,
        ) > float(edit_tolerance)
        global_changed = np.zeros_like(occupied)
        global_changed[lo:hi] = changed
        if np.any(occupied & global_changed):
            raise ValueError("successful solution edit regions overlap")
        repaired_window = repaired[lo:hi]
        repaired_window[changed] = returned_motion[changed]
        occupied |= global_changed
        applied.append({
            "case_id": row["case_id"],
            "frame_span": [lo, hi],
            "changed_frames": int(changed.sum()),
            "reference": saved_reference_info,
            "proposal_action": proposal_action_info,
            "returned_action": returned_action_info,
            "returned_motion": returned_motion_info,
        })
    return repaired, occupied, applied


def replay_solutions(
    bundle_path,
    source_report_path,
    feasibility_run,
    output,
    baseline="B3_frozen_v1_proposal_plus_action_solver",
    edit_tolerance=1.0e-7,
):
    from routing import boundary_closed_loop as closed_loop
    from training import motion_models as runtime

    bundle_path = Path(bundle_path).resolve()
    bundle_sha256 = _hash(bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    if bundle.get("schema") != "generation_round_bundle_v1":
        raise ValueError("replay requires generation_round_bundle_v1")
    source_report, source_report_path = _linked_source_report(
        source_report_path, bundle_path, bundle_sha256
    )
    feasibility, feasibility_report_path, rows = _solution_rows(
        feasibility_run, baseline
    )
    refiner_checkpoint = bundle.get("checkpoints", {}).get("refiner", {})
    evaluated_checkpoint = feasibility.get("provenance", {}).get("checkpoint", {})
    if (
        not evaluated_checkpoint
        or evaluated_checkpoint.get("sha256") != refiner_checkpoint.get("sha256")
    ):
        raise ValueError("feasibility/refiner checkpoint SHA256 mismatch")

    reference, reference_info = _load_hashed_array(
        bundle.get("arrays", {}).get("reference"), "reference"
    )
    condition, condition_info = _load_hashed_array(
        bundle.get("arrays", {}).get("condition"), "condition"
    )
    seam, seam_info = _load_hashed_array(
        bundle.get("arrays", {}).get("seam"), "seam"
    )
    slide, slide_info = _load_hashed_array(
        bundle.get("arrays", {}).get("sliding_support_eligible"),
        "sliding_support_eligible",
    )
    repaired, occupied, applied = _apply_verified_solution_windows(
        reference, rows, bundle_sha256, edit_tolerance
    )

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repaired_refiner = _array(output, "repaired_refiner", repaired)
    config_path = Path(bundle["config_path"]).resolve()
    if _hash(config_path) != bundle.get("config_sha256"):
        raise ValueError("captured config SHA256 mismatch")
    captured_environment = bundle.get("runtime_environment", {})
    with _captured_environment(captured_environment):
        cfg = runtime.MotionGenerationConfig.from_json(config_path).apply_env()
        cfg.refiner_enable = False
        diffusion = bundle.get("checkpoints", {}).get("diffusion", {})
        diffusion_path = diffusion.get("path") if diffusion.get("active") else None
        if diffusion_path and _hash(diffusion_path) != diffusion.get("sha256"):
            raise ValueError("captured diffusion checkpoint SHA256 mismatch")
        replay_args = argparse.Namespace(
            out=str(output / "replayed.npy"),
            refiner=None,
            diffusion=diffusion_path,
        )
        final_motion, stage_reports = closed_loop.apply_generators(
            runtime,
            repaired,
            condition,
            seam,
            replay_args,
            cfg,
            sliding_support_eligible=slide,
        )
        boundary_rows = closed_loop.audit_boundaries(
            runtime, final_motion, bundle["assembly"], cfg
        )
        physical = closed_loop.physical_quality_gate(stage_reports["final_audit"])
        boundary = closed_loop.evaluate_boundary_continuity(
            boundary_rows,
            expected_boundaries=max(0, len(bundle["assembly"]) - 1),
        )
        activity = closed_loop.evaluate_final_motion_activity(
            final_motion,
            slots=source_report["slots"],
            assembly_report=bundle["assembly"],
            fps=float(cfg.fps),
        )
        final_gate = _quality_gate(closed_loop, physical, boundary, activity)

    final_motion_info = _array(output, "replayed_final", final_motion)
    report = {
        "schema": "refiner_solution_development_replay_v1",
        "completed": True,
        "development_only": True,
        "formal_preregistration": False,
        "training_started": False,
        "production_model_modified": False,
        "scientific_acceptance": False,
        "pilot_allowed": False,
        "baseline": baseline,
        "source": {
            "bundle": {"path": str(bundle_path), "sha256": bundle_sha256},
            "source_report": {
                "path": str(source_report_path),
                "sha256": _hash(source_report_path),
            },
            "feasibility_report": {
                "path": str(feasibility_report_path),
                "sha256": _hash(feasibility_report_path),
            },
            "reference": reference_info,
            "condition": condition_info,
            "seam": seam_info,
            "sliding_support_eligible": slide_info,
            "refiner_checkpoint_sha256": refiner_checkpoint.get("sha256"),
            "diffusion_checkpoint_sha256": bundle.get("checkpoints", {}).get("diffusion", {}).get("sha256"),
        },
        "solutions": applied,
        "edit_union_frames": int(occupied.sum()),
        "repaired_refiner": repaired_refiner,
        "replayed_final": final_motion_info,
        "stage_reports": stage_reports,
        "boundary_rows": boundary_rows,
        "final_physical_gate": physical,
        "final_boundary_continuity_gate": boundary,
        "final_motion_activity": activity,
        "final_quality_gate": final_gate,
    }
    report_path = output / "replay.report.json"
    _write(report_path, report)
    return report_path, bool(final_gate["ok"])


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
    export.add_argument(
        "--proposal-stage",
        choices=("refiner",),
        default=None,
        help="Export a hash-bound captured frozen proposal for B2/B3.",
    )
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--source-report", required=True)
    replay.add_argument("--feasibility-run", required=True)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument(
        "--baseline",
        default="B3_frozen_v1_proposal_plus_action_solver",
        choices=("B3_frozen_v1_proposal_plus_action_solver",),
    )
    replay.add_argument("--edit-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()
    if args.command == "export":
        print(export_cases(
            args.bundle,
            args.provenance,
            args.output_dir,
            args.slots,
            args.context_frames,
            args.proposal_stage,
        ))
    elif args.command == "replay":
        report_path, ok = replay_solutions(
            args.bundle,
            args.source_report,
            args.feasibility_run,
            args.output_dir,
            args.baseline,
            args.edit_tolerance,
        )
        print(json.dumps({
            "stage": "refiner_solution_development_replay_complete",
            "report": str(report_path),
            "ok": ok,
            "training_started": False,
            "production_model_modified": False,
        }, ensure_ascii=False))
        return 0 if ok else 2
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
