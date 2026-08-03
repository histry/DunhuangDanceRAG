#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime anti-collapse patch for the research whole-song pipeline.

The patch is deliberately orthogonal to anatomy, physics and severe-heading
contracts. It adds activity-aware candidate ordering, rejects strongly static
candidates for slots whose music semantics require motion, records stage-wise
motion activity, and applies a final hard acceptance gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.motion_activity import (
    ActivityThresholds,
    candidate_activity_assessment,
    compare_motion_activity,
    diagnose_motion,
    motion_activity_metrics,
    reportable_metrics,
    slot_activity_target,
)

_INSTALLED = False
_EVENT_ACTIVITY_CACHE: Dict[str, Dict[str, Any]] = {}
_LAST_PREORDER_DIAGNOSTICS: Dict[str, Any] = {}
_CANDIDATE_ACTIVITY_STATS: Dict[str, Any] = {}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _fps(cfg: Any = None) -> float:
    if cfg is not None:
        try:
            value = float(getattr(cfg, "fps"))
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
    return max(1.0e-6, _env_float("V46_51_FPS", 30.0))


def _event_path(db: Mapping[str, Any], event_id: int) -> Optional[str]:
    try:
        paths = np.asarray(db["paths"], dtype=object)
        if 0 <= int(event_id) < len(paths):
            return str(paths[int(event_id)])
    except Exception:
        pass
    return None


def _event_metrics(
    db: Mapping[str, Any], event_id: int, fps: float
) -> Optional[Dict[str, Any]]:
    path_text = _event_path(db, event_id)
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    key = "%s|%.6f" % (str(path), float(fps))
    cached = _EVENT_ACTIVITY_CACHE.get(key)
    if cached is not None:
        return cached
    if not path.is_file():
        return None
    try:
        motion = np.load(str(path), allow_pickle=True, mmap_mode="r")
        metrics = motion_activity_metrics(np.asarray(motion), fps=fps)
    except Exception:
        return None
    compact = reportable_metrics(metrics)
    _EVENT_ACTIVITY_CACHE[key] = compact
    return compact


def _activity_order_cost(
    metrics: Optional[Mapping[str, Any]],
    target: Optional[float],
    rank: int,
) -> float:
    # Missing metrics retain their retrieval rank and are never silently banned.
    if metrics is None or target is None:
        return float(rank)
    density = float(metrics.get("motion_density_mean", 0.0))
    static_ratio = float(metrics.get("static_frame_ratio", 1.0))
    mismatch = abs(density - float(target))
    active_deficit = max(0.0, float(target) - density)
    return float(
        8.0 * mismatch
        + 4.0 * active_deficit
        + 2.0 * static_ratio * float(target)
        + 0.015 * rank
    )


def _capture_result_motion(result: Any) -> Optional[np.ndarray]:
    value = result[0] if isinstance(result, tuple) and result else result
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] < 151:
        return None
    return array[:, :151].copy()


def _save_stage_outputs(
    args: Any, stages: Mapping[str, np.ndarray]
) -> Dict[str, str]:
    if not _env_bool("MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS", True):
        return {}
    raw_out = getattr(args, "out", None)
    if not raw_out:
        return {}
    out = Path(str(raw_out))
    out.parent.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    suffixes = {
        "retrieval_only": "activity_retrieval",
        "retrieval_refiner": "activity_refiner",
        "retrieval_refiner_diffusion": "activity_diffusion",
        "full_pipeline_ik": "activity_full_ik",
    }
    for name, motion in stages.items():
        suffix = suffixes.get(name, "activity_" + name)
        path = out.with_name(out.stem + "." + suffix + ".npy")
        np.save(str(path), np.asarray(motion, dtype=np.float32))
        paths[name] = str(path)
    return paths


def _report_path(args: Any) -> Path:
    explicit = getattr(args, "json", None)
    if explicit:
        return Path(str(explicit))
    output = Path(str(getattr(args, "out")))
    return output.with_name(output.stem + ".v46_46_closed_loop_report.json")


def _json_write(path: Path, value: Mapping[str, Any], base: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = base.jsonable(value) if hasattr(base, "jsonable") else value
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def install(v53: Any) -> Dict[str, Any]:
    """Install activity-aware wrappers after V53 and feasibility patches."""

    global _INSTALLED
    if _INSTALLED:
        return {"installed": True, "already_installed": True}

    v50 = v53.v52.v4650
    base = v50.base
    original_proposal = v50._build_heading_proposal
    original_preorder = v53._global_route_preorder
    original_apply = v50.apply_generators_with_heading_guard
    original_generate = base.generate_closed_loop

    def activity_preorder(
        slots: Sequence[Mapping[str, Any]],
        candidate_lists: Sequence[Sequence[int]],
        db: Mapping[str, Any],
        banned: Optional[Dict[int, set]] = None,
    ) -> List[List[int]]:
        global _LAST_PREORDER_DIAGNOSTICS
        if not _env_bool("MOTION_ACTIVITY_PREORDER_ENABLE", True):
            return original_preorder(slots, candidate_lists, db, banned=banned)
        scan_topk = max(1, _env_int("MOTION_ACTIVITY_PREORDER_SCAN_TOPK", 96))
        fps = _fps()
        reordered: List[List[int]] = []
        rows: List[Dict[str, Any]] = []
        for slot_index, original in enumerate(candidate_lists):
            values = list(map(int, original))
            target = slot_activity_target(slots[slot_index])
            front = values[:scan_topk]
            tail = values[scan_topk:]
            scored: List[
                Tuple[float, int, int, Optional[Dict[str, Any]]]
            ] = []
            for rank, event_id in enumerate(front):
                metrics = _event_metrics(db, event_id, fps=fps)
                scored.append(
                    (
                        _activity_order_cost(metrics, target, rank),
                        rank,
                        event_id,
                        metrics,
                    )
                )
            scored.sort(key=lambda row: (row[0], row[1], row[2]))
            layer = [row[2] for row in scored] + tail
            reordered.append(layer)
            rows.append(
                {
                    "slot": int(slot_index),
                    "target_activity": target,
                    "scanned_candidates": int(len(scored)),
                    "original_top_event_id": int(values[0]) if values else None,
                    "activity_top_event_id": int(layer[0]) if layer else None,
                    "activity_top_metrics": scored[0][3] if scored else None,
                }
            )
        _LAST_PREORDER_DIAGNOSTICS = {
            "schema": "activity_aware_candidate_preorder_v1",
            "enabled": True,
            "scan_topk": int(scan_topk),
            "slots": rows,
            "hard_filtered": False,
        }
        return original_preorder(slots, reordered, db, banned=banned)

    def activity_proposal(*args: Any, **kwargs: Any):
        proposal, extra = original_proposal(*args, **kwargs)
        slot = kwargs.get("slot", {})
        cfg = kwargs.get("cfg")
        metrics = motion_activity_metrics(proposal.core, fps=_fps(cfg))
        target = slot_activity_target(
            slot if isinstance(slot, Mapping) else {}
        )
        assessment = candidate_activity_assessment(
            metrics,
            target,
            thresholds=ActivityThresholds.from_env(),
        )
        proposal.risk_score = float(
            proposal.risk_score
            + float(assessment["penalty"])
            + (1.0e6 if bool(assessment["hard_reject"]) else 0.0)
        )
        proposal.safe = bool(
            proposal.safe and not bool(assessment["hard_reject"])
        )
        proposal.risk["motion_activity"] = {
            **assessment,
            "metrics": reportable_metrics(metrics),
        }
        extra = dict(extra)
        extra["motion_activity"] = proposal.risk["motion_activity"]
        heading = extra.setdefault("heading_detail", {})
        heading["hard_reject"] = bool(
            heading.get("hard_reject", False)
            or bool(assessment["hard_reject"])
        )
        _CANDIDATE_ACTIVITY_STATS["evaluated"] = int(
            _CANDIDATE_ACTIVITY_STATS.get("evaluated", 0) + 1
        )
        _CANDIDATE_ACTIVITY_STATS["hard_rejected"] = int(
            _CANDIDATE_ACTIVITY_STATS.get("hard_rejected", 0)
            + int(bool(assessment["hard_reject"]))
        )
        _CANDIDATE_ACTIVITY_STATS["penalty_sum"] = float(
            _CANDIDATE_ACTIVITY_STATS.get("penalty_sum", 0.0)
            + float(assessment["penalty"])
        )
        return proposal, extra

    def activity_apply(
        v46: Any,
        motion_ref: np.ndarray,
        cond: np.ndarray,
        seam_mask: np.ndarray,
        args: Any,
        cfg: Any,
    ):
        captured: Dict[str, np.ndarray] = {
            "retrieval_only": np.asarray(motion_ref, dtype=np.float32).copy()
        }
        originals: Dict[str, Any] = {}

        def patch_capture(name: str, stage_name: str) -> None:
            fn = getattr(v46, name, None)
            if fn is None:
                return
            originals[name] = fn

            def wrapper(*call_args: Any, **call_kwargs: Any):
                result = fn(*call_args, **call_kwargs)
                motion = _capture_result_motion(result)
                if motion is not None:
                    captured[stage_name] = motion
                return result

            setattr(v46, name, wrapper)

        patch_capture("apply_refiner_model", "retrieval_refiner")
        patch_capture("apply_diffusion_model", "retrieval_refiner_diffusion")
        patch_capture("true_lower_body_ik", "full_pipeline_ik_raw")
        try:
            motion, stage = original_apply(
                v46,
                motion_ref,
                cond,
                seam_mask,
                args,
                cfg,
            )
        finally:
            for name, fn in originals.items():
                setattr(v46, name, fn)

        captured["full_pipeline_ik"] = np.asarray(
            motion, dtype=np.float32
        ).copy()
        previous = captured["retrieval_only"]
        for name in (
            "retrieval_refiner",
            "retrieval_refiner_diffusion",
            "full_pipeline_ik",
        ):
            if name not in captured:
                captured[name] = previous.copy()
            previous = captured[name]

        thresholds = ActivityThresholds.from_env()
        stage_metrics: Dict[str, Any] = {}
        stage_deltas: Dict[str, Any] = {}
        previous_name: Optional[str] = None
        for name in (
            "retrieval_only",
            "retrieval_refiner",
            "retrieval_refiner_diffusion",
            "full_pipeline_ik",
        ):
            stage_metrics[name] = reportable_metrics(
                motion_activity_metrics(
                    captured[name],
                    fps=_fps(cfg),
                    thresholds=thresholds,
                )
            )
            if previous_name is not None:
                stage_deltas[
                    previous_name + "_to_" + name
                ] = compare_motion_activity(
                    captured[previous_name],
                    captured[name],
                    fps=_fps(cfg),
                    thresholds=thresholds,
                )
            previous_name = name

        output_paths = _save_stage_outputs(
            args,
            {
                key: captured[key]
                for key in (
                    "retrieval_only",
                    "retrieval_refiner",
                    "retrieval_refiner_diffusion",
                    "full_pipeline_ik",
                )
            },
        )
        stage = dict(stage)
        stage["motion_activity"] = {
            "schema": "stage_wise_motion_activity_v1",
            "stages": stage_metrics,
            "stage_deltas": stage_deltas,
            "stage_output_paths": output_paths,
            "raw_ik_capture_available": "full_pipeline_ik_raw" in captured,
        }
        return np.asarray(motion, dtype=np.float32), stage

    def activity_generate(args: Any) -> int:
        global _CANDIDATE_ACTIVITY_STATS
        _CANDIDATE_ACTIVITY_STATS = {
            "evaluated": 0,
            "hard_rejected": 0,
            "penalty_sum": 0.0,
        }
        rc = int(original_generate(args))
        if rc != 0:
            return rc
        output = Path(str(getattr(args, "out")))
        if not output.is_file():
            raise RuntimeError(
                "Activity guard cannot find generated motion: %s" % output
            )
        report_path = _report_path(args)
        report: Dict[str, Any] = {}
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                report = {}
        slots = report.get("slots", []) if isinstance(report, Mapping) else []
        stage_reports = (
            report.get("stage_reports", {})
            if isinstance(report, Mapping)
            else {}
        )
        assembly = (
            stage_reports.get("closed_loop_concat", [])
            if isinstance(stage_reports, Mapping)
            else []
        )
        fps = (
            float(report.get("fps", _fps()))
            if isinstance(report, Mapping)
            else _fps()
        )
        motion = np.load(str(output), allow_pickle=True)
        diagnostics = diagnose_motion(
            motion,
            fps=fps,
            slots=slots if isinstance(slots, Sequence) else [],
            assembly_report=assembly if isinstance(assembly, Sequence) else [],
        )
        payload = {
            "schema": "static_motion_collapse_guard_v1",
            "final": diagnostics,
            "candidate_activity": dict(_CANDIDATE_ACTIVITY_STATS),
            "activity_preorder": dict(_LAST_PREORDER_DIAGNOSTICS),
            "hard_physical_anatomy_heading_gates_relaxed": False,
            "rejected_after_write_for_diagnostics": bool(
                not diagnostics["acceptance_gate"]["ok"]
            ),
        }
        report["motion_activity_guard"] = payload
        _json_write(report_path, report, base)
        sidecar = output.with_name(output.stem + ".motion_activity.json")
        _json_write(sidecar, payload, base)
        gate = diagnostics["acceptance_gate"]
        if bool(gate.get("hard_gate_enabled", True)) and not bool(
            gate.get("ok", False)
        ):
            raise RuntimeError(
                "Final motion activity gate rejected a physically safe but nearly "
                "static solution. reasons=%s; diagnostics=%s"
                % (gate.get("reasons", []), sidecar)
            )
        return rc

    v53._global_route_preorder = activity_preorder
    v50._build_heading_proposal = activity_proposal
    v50.apply_generators_with_heading_guard = activity_apply
    base.generate_closed_loop = activity_generate
    _INSTALLED = True
    return {
        "installed": True,
        "candidate_hard_gate": _env_bool(
            "MOTION_ACTIVITY_CANDIDATE_HARD_GATE", True
        ),
        "final_hard_gate": _env_bool("MOTION_ACTIVITY_FINAL_HARD_GATE", True),
        "save_stage_outputs": _env_bool(
            "MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS", True
        ),
    }
