"""Opt-in generation traces and development-only solution replay.

The summarize/export paths are read-only.  Replay can execute the explicitly
requested frozen generators and V9 constrained contact repair, but never
starts training or modifies a checkpoint.
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


def _array_content_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _array(directory, name, value):
    path = Path(directory) / (name + ".npy")
    np.save(path, np.asarray(value), allow_pickle=False)
    return {"path": str(path.resolve()), "sha256": _hash(path), "shape": list(value.shape)}


def _load_hashed_array(entry, label):
    if not isinstance(entry, dict):
        raise TypeError(f"array entry must be an object: {label}")
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
        "ik_rollback": {
            key: ik.get(key)
            for key in (
                "rollback_triggered",
                "rollback_to_pre_ik_snapshot",
                "stage_guard_ik_rollback_to_pre_ik_snapshot",
                "stage_guard_rollback_reasons",
            )
        },
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
        raise TypeError("source report slot activity contract must be a list")
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


def _merge_frame_windows(windows, frames, gap=0):
    normalized = sorted(
        (
            max(0, int(window[0])),
            min(int(frames), int(window[1])),
        )
        for window in windows
        if isinstance(window, (list, tuple)) and len(window) == 2
    )
    merged = []
    for start, end in normalized:
        if end - start < 4:
            continue
        if merged and start - merged[-1][1] <= int(gap):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _unsafe_boundary_windows(rows, frames, halo):
    windows = []
    for row in rows:
        if bool(row.get("safe", False)):
            continue
        start = int(row.get("transition_start", row.get("content_start", 0)))
        end = int(row.get("transition_end", start))
        if end <= start:
            end = int(row.get("content_start", start)) + 1
        windows.append([
            max(0, start - int(halo)),
            min(int(frames), end + int(halo)),
        ])
    return windows


def _boundary_nonregression(before_rows, candidate_rows):
    before = {int(row["slot"]): row for row in before_rows}
    candidate = {int(row["slot"]): row for row in candidate_rows}
    metrics = (
        "actual_boundary_jerk_mps3",
        "actual_entry_fk_jump_m",
        "actual_exit_fk_jump_m",
        "actual_entry_fk_jump_max_m",
        "actual_exit_fk_jump_max_m",
        "actual_entry_rotation_step_rad",
        "actual_exit_rotation_step_rad",
        "actual_foot_slip",
        "actual_foot_slip_p95_mps",
        "actual_foot_slip_peak_mps",
        "actual_foot_penetration_depth_max_m",
    )
    regressions = []
    deltas = {}
    for slot, reference in before.items():
        trial = candidate.get(slot)
        if trial is None:
            regressions.append(f"slot_{slot}:missing_candidate_boundary")
            continue
        for metric in metrics:
            old = float(reference.get(metric, 0.0))
            new = float(trial.get(metric, 0.0))
            delta = new - old
            deltas[f"slot_{slot}:{metric}"] = float(delta)
            tolerance = max(1.0e-7, abs(old) * 1.0e-6)
            if delta > tolerance:
                regressions.append(f"slot_{slot}:{metric}_regressed")
    return {
        "accepted": not regressions,
        "reasons": regressions,
        "metric_deltas": deltas,
    }


def _load_stage_snapshot(entry):
    if not isinstance(entry, dict) or not entry.get("snapshot_saved"):
        return None
    path = Path(entry.get("snapshot_path", "")).resolve()
    if not path.is_file():
        return None
    value = np.load(path, allow_pickle=False)
    return value if np.isfinite(value).all() else None


def _v9_stage_diagnostics(runtime, cfg, eligible, stages, arrays):
    result = {
        "schema": "generation_stage_physical_diagnostics_v9",
        "support_contract": "final_fail_closed_with_sliding_eligibility",
        "sliding_support_eligible_sha256": _array_content_hash(eligible),
        "sliding_support_eligible_frames": int(np.asarray(eligible).sum()),
        "top_k": int(cfg.full_sequence_contact_repair_top_k),
        "stages": {},
    }
    for name, value in arrays.items():
        if value is None:
            continue
        diagnostic = runtime.full_sequence_physical_diagnostics_np(
            value,
            cfg,
            sliding_support_eligible=eligible,
        )
        diagnostic["motion_sha256"] = _array_content_hash(value)
        result["stages"][name] = diagnostic
    ik = stages.get("lower_body_ik_true_ik", {})
    if isinstance(ik, dict):
        for name, key in (
            ("ik_candidate", "candidate_localization"),
            ("ik_selected", "selected_localization"),
        ):
            value = ik.get(key)
            if isinstance(value, dict):
                diagnostic = dict(value)
                hash_key = "candidate" if name == "ik_candidate" else "selected"
                if name == "ik_selected" and ik.get(
                    "stage_guard_ik_rollback_to_pre_ik_snapshot"
                ):
                    motion_sha256 = ik.get("stage_guard_hashes", {}).get(
                        "selected"
                    )
                else:
                    motion_sha256 = ik.get("hashes", {}).get(hash_key)
                if motion_sha256:
                    diagnostic["motion_sha256"] = str(motion_sha256)
                result["stages"][name] = diagnostic
    return result


def replay_solutions(
    bundle_path,
    source_report_path,
    feasibility_run,
    output,
    baseline="B3_frozen_v1_proposal_plus_action_solver",
    edit_tolerance=1.0e-7,
):
    from contracts.physical_quality import evaluate_stage_reference_fidelity
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
    diagnostic_environment = {
        "BOUNDARY_STAGE_DIAGNOSTICS": "1",
        "MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS": "1",
    }
    with _captured_environment(captured_environment), _captured_environment(
        diagnostic_environment
    ):
        cfg = runtime.MotionGenerationConfig.from_json(config_path).apply_env()
        cfg.refiner_enable = False
        cfg.full_sequence_contact_repair_enable = True

        repair_input_localization = (
            runtime.full_sequence_physical_diagnostics_np(
                repaired,
                cfg,
                sliding_support_eligible=slide,
            )
        )
        repair_input_boundaries = closed_loop.audit_boundaries(
            runtime,
            repaired,
            bundle["assembly"],
            cfg,
        )
        halo = max(2, int(round(
            float(cfg.full_sequence_contact_repair_halo_seconds)
            * float(cfg.fps)
        )))
        merge_gap = max(0, int(round(
            float(cfg.full_sequence_contact_repair_merge_gap_seconds)
            * float(cfg.fps)
        )))
        repair_windows = _merge_frame_windows(
            list(repair_input_localization["repair_windows"])
            + _unsafe_boundary_windows(
                repair_input_boundaries,
                len(repaired),
                halo,
            ),
            len(repaired),
            gap=merge_gap,
        )

        def contact_candidate_guard(current, candidate, ownership, audit_span):
            start, end = map(int, audit_span)
            local_eligible = slide[start:end]
            reference_audit = runtime.audit_motion_np(
                repaired[start:end],
                cfg,
                sliding_support_eligible=local_eligible,
            )
            candidate_audit = runtime.audit_motion_np(
                candidate[start:end],
                cfg,
                sliding_support_eligible=local_eligible,
            )
            fidelity = evaluate_stage_reference_fidelity(
                reference_audit,
                candidate_audit,
            )
            fixed_support = (
                runtime.evaluate_fixed_support_contact_candidate_np(
                    current[start:end],
                    candidate[start:end],
                    cfg,
                    sliding_support_eligible=local_eligible,
                )
            )
            boundary = _boundary_nonregression(
                closed_loop.audit_boundaries(
                    runtime, current, bundle["assembly"], cfg
                ),
                closed_loop.audit_boundaries(
                    runtime, candidate, bundle["assembly"], cfg
                ),
            )
            observable_geometry_preserved = bool(
                np.array_equal(candidate[occupied, 4:], current[occupied, 4:])
            )
            reasons = []
            if not fidelity["accepted"]:
                reasons.extend(
                    f"fidelity:{reason}" for reason in fidelity["reasons"]
                )
            if not fixed_support["accepted"]:
                reasons.extend(
                    f"fixed_support:{reason}"
                    for reason in fixed_support["reasons"]
                )
            if not boundary["accepted"]:
                reasons.extend(
                    f"boundary:{reason}" for reason in boundary["reasons"]
                )
            if not observable_geometry_preserved:
                reasons.append("observable_solution_geometry_modified")
            return {
                "accepted": not reasons,
                "reasons": reasons,
                "ownership_span": list(map(int, ownership)),
                "audit_span": [start, end],
                "fidelity": fidelity,
                "fixed_support": fixed_support,
                "boundary": boundary,
                "observable_gate": 0.03,
                "observable_solution_geometry_preserved": (
                    observable_geometry_preserved
                ),
            }

        repair_candidate, contact_repair_report = runtime.true_lower_body_ik(
            repaired,
            cfg,
            sliding_support_eligible=slide,
            repair_windows=repair_windows,
            protected_frame_mask=occupied,
            candidate_guard=contact_candidate_guard,
        )
        repair_input_audit = runtime.audit_motion_np(
            repaired,
            cfg,
            sliding_support_eligible=slide,
        )
        repair_candidate_audit = runtime.audit_motion_np(
            repair_candidate,
            cfg,
            sliding_support_eligible=slide,
        )
        contact_decision = runtime._contact_restoration_decision(
            repair_input_audit,
            repair_candidate_audit,
            cfg,
        )
        repair_candidate_boundaries = closed_loop.audit_boundaries(
            runtime,
            repair_candidate,
            bundle["assembly"],
            cfg,
        )
        boundary_decision = _boundary_nonregression(
            repair_input_boundaries,
            repair_candidate_boundaries,
        )
        fidelity_decision = evaluate_stage_reference_fidelity(
            repair_input_audit,
            repair_candidate_audit,
        )
        fixed_support_decision = (
            runtime.evaluate_fixed_support_contact_candidate_np(
                repaired,
                repair_candidate,
                cfg,
                sliding_support_eligible=slide,
            )
        )
        observable_geometry_preserved = bool(
            np.array_equal(
                repair_candidate[occupied, 4:],
                repaired[occupied, 4:],
            )
        )
        contact_stage_accepted = bool(
            contact_decision["accepted"]
            and boundary_decision["accepted"]
            and fidelity_decision["accepted"]
            and fixed_support_decision["accepted"]
            and observable_geometry_preserved
        )
        repaired_contact = (
            repair_candidate if contact_stage_accepted else repaired.copy()
        )
        repaired_contact_info = _array(
            output,
            "repaired_contact_v9",
            repaired_contact,
        )
        contact_repair_transaction = {
            "schema": "full_sequence_constrained_contact_repair_v9",
            "development_only": True,
            "training_started": False,
            "production_model_modified": False,
            "observable_gate": 0.03,
            "repair_windows": repair_windows,
            "input_localization": repair_input_localization,
            "input_boundary_rows": repair_input_boundaries,
            "solver_report": contact_repair_report,
            "contact_decision": contact_decision,
            "boundary_decision": boundary_decision,
            "fidelity_decision": fidelity_decision,
            "fixed_support_decision": fixed_support_decision,
            "observable_solution_geometry_preserved": (
                observable_geometry_preserved
            ),
            "accepted": contact_stage_accepted,
            "rollback_to_pre_contact_repair_snapshot": (
                not contact_stage_accepted
            ),
            "hashes": {
                "input": _array_content_hash(repaired),
                "candidate": _array_content_hash(repair_candidate),
                "selected": _array_content_hash(repaired_contact),
            },
            "selected_artifact": repaired_contact_info,
        }
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
            repaired_contact,
            condition,
            seam,
            replay_args,
            cfg,
            sliding_support_eligible=slide,
            protected_geometry_mask=occupied,
            ik_protected_frame_mask=occupied,
            ik_candidate_guard=contact_candidate_guard,
        )
        stage_reports["full_sequence_constrained_contact_repair"] = (
            contact_repair_transaction
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

        diffusion_candidate = None
        diffusion_entry = stage_reports.get("diffusion_guarded_candidate")
        if isinstance(diffusion_entry, dict):
            try:
                diffusion_candidate, _ = _load_hashed_array(
                    diffusion_entry,
                    "diffusion_guarded_candidate",
                )
            except (OSError, TypeError, ValueError):
                diffusion_candidate = None
        diffusion_selected = _load_stage_snapshot(
            stage_reports.get("motion_activity_diffusion")
        )
        stage_reports["v9_stage_physical_diagnostics"] = (
            _v9_stage_diagnostics(
                runtime,
                cfg,
                slide,
                stage_reports,
                {
                    "repaired_refiner": repaired,
                    "contact_repair_candidate": repair_candidate,
                    "contact_repair_selected": repaired_contact,
                    "diffusion_candidate": diffusion_candidate,
                    "diffusion_selected": diffusion_selected,
                    "pre_ik": diffusion_selected,
                    "final": final_motion,
                },
            )
        )

    final_motion_info = _array(output, "replayed_final", final_motion)
    report = {
        "schema": "refiner_solution_development_replay_v1",
        "protocol": "full_sequence_constrained_contact_repair_replay_v9",
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
        "repaired_contact_v9": repaired_contact_info,
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
