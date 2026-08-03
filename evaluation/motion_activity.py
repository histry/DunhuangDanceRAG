#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion-activity diagnostics and anti-collapse gates for EDGE151 sequences.

The canonical EDGE representation is ``[contact(4), root_xyz(3), 24 x Rot6D]``.
Rot6D values are decoded with the repository column-concatenated convention.
This module measures activity independently from boundary/anatomy safety so a
physically valid but nearly static whole-song result cannot be accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from motion_geometry.rotations import angular_velocity_np, rot6d_to_matrix_np

EDGE_DIM = 151
ROOT = slice(4, 7)
ROT6D = slice(7, 151)
NUM_JOINTS = 24


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _as_motion(motion: np.ndarray) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] < EDGE_DIM:
        raise ValueError("Expected EDGE151 motion [T,151], got %r" % (x.shape,))
    return x[:, :EDGE_DIM]


def _longest_true_streak(mask: np.ndarray) -> Tuple[int, int, int]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = 0
    best_end = 0
    start: Optional[int] = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if (not value or index == len(values) - 1) and start is not None:
            end = index + 1 if value and index == len(values) - 1 else index
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = None
    return int(best_end - best_start), int(best_start), int(best_end)


@dataclass(frozen=True)
class ActivityThresholds:
    static_joint_speed_rad_s: float = 0.08
    static_root_speed_m_s: float = 0.015
    joint_speed_reference_rad_s: float = 0.55
    root_speed_reference_m_s: float = 0.35
    candidate_required_target: float = 0.55
    candidate_max_static_ratio: float = 0.84
    candidate_max_static_seconds: float = 1.75
    candidate_min_joint_speed_rad_s: float = 0.065
    candidate_penalty_weight: float = 1.25
    final_default_max_static_ratio: float = 0.72
    final_default_max_static_seconds: float = 4.0
    final_default_min_joint_speed_rad_s: float = 0.045
    final_high_target_slot_max_static_ratio: float = 0.75
    final_high_target_failed_slot_fraction: float = 0.34

    @classmethod
    def from_env(cls) -> "ActivityThresholds":
        return cls(
            static_joint_speed_rad_s=max(
                0.0, _env_float("MOTION_ACTIVITY_STATIC_JOINT_RADPS", 0.08)
            ),
            static_root_speed_m_s=max(
                0.0, _env_float("MOTION_ACTIVITY_STATIC_ROOT_MPS", 0.015)
            ),
            joint_speed_reference_rad_s=max(
                1.0e-6, _env_float("MOTION_ACTIVITY_JOINT_REFERENCE_RADPS", 0.55)
            ),
            root_speed_reference_m_s=max(
                1.0e-6, _env_float("MOTION_ACTIVITY_ROOT_REFERENCE_MPS", 0.35)
            ),
            candidate_required_target=float(
                np.clip(
                    _env_float("MOTION_ACTIVITY_CANDIDATE_REQUIRED_TARGET", 0.55),
                    0.0,
                    1.0,
                )
            ),
            candidate_max_static_ratio=float(
                np.clip(
                    _env_float("MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_RATIO", 0.84),
                    0.0,
                    1.0,
                )
            ),
            candidate_max_static_seconds=max(
                0.0,
                _env_float("MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_SECONDS", 1.75),
            ),
            candidate_min_joint_speed_rad_s=max(
                0.0,
                _env_float("MOTION_ACTIVITY_CANDIDATE_MIN_JOINT_RADPS", 0.065),
            ),
            candidate_penalty_weight=max(
                0.0, _env_float("MOTION_ACTIVITY_CANDIDATE_PENALTY_WEIGHT", 1.25)
            ),
            final_default_max_static_ratio=float(
                np.clip(
                    _env_float("MOTION_ACTIVITY_FINAL_MAX_STATIC_RATIO", 0.72),
                    0.0,
                    1.0,
                )
            ),
            final_default_max_static_seconds=max(
                0.0,
                _env_float("MOTION_ACTIVITY_FINAL_MAX_STATIC_SECONDS", 4.0),
            ),
            final_default_min_joint_speed_rad_s=max(
                0.0,
                _env_float("MOTION_ACTIVITY_FINAL_MIN_JOINT_RADPS", 0.045),
            ),
            final_high_target_slot_max_static_ratio=float(
                np.clip(
                    _env_float(
                        "MOTION_ACTIVITY_HIGH_TARGET_SLOT_MAX_STATIC_RATIO", 0.75
                    ),
                    0.0,
                    1.0,
                )
            ),
            final_high_target_failed_slot_fraction=float(
                np.clip(
                    _env_float(
                        "MOTION_ACTIVITY_HIGH_TARGET_FAILED_SLOT_FRACTION", 0.34
                    ),
                    0.0,
                    1.0,
                )
            ),
        )


def motion_activity_metrics(
    motion: np.ndarray,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    """Return whole-sequence angular/root activity and static-streak metrics."""

    x = _as_motion(motion)
    cfg = thresholds or ActivityThresholds.from_env()
    fps = max(float(fps), 1.0e-6)
    frames = int(len(x))
    duration = float(frames / fps)

    rotations = rot6d_to_matrix_np(
        x[:, ROT6D].reshape(frames, NUM_JOINTS, 6)
    )
    angular_velocity = angular_velocity_np(rotations, fps=fps)
    if len(angular_velocity):
        joint_speed_step = np.linalg.norm(angular_velocity, axis=-1).mean(axis=1)
        joint_speed_frame = np.concatenate(
            [joint_speed_step[:1], joint_speed_step], axis=0
        )[:frames]
    else:
        joint_speed_step = np.zeros((0,), dtype=np.float32)
        joint_speed_frame = np.zeros((frames,), dtype=np.float32)

    root = x[:, ROOT]
    root_delta = np.diff(root, axis=0)
    root_step = np.linalg.norm(root_delta, axis=-1)
    root_speed_step = root_step * fps
    if len(root_speed_step):
        root_speed_frame = np.concatenate(
            [root_speed_step[:1], root_speed_step], axis=0
        )[:frames]
    else:
        root_speed_frame = np.zeros((frames,), dtype=np.float32)

    static_mask = (
        joint_speed_frame < cfg.static_joint_speed_rad_s
    ) & (root_speed_frame < cfg.static_root_speed_m_s)
    streak, streak_start, streak_end = _longest_true_streak(static_mask)

    normalized_joint = np.clip(
        joint_speed_frame / cfg.joint_speed_reference_rad_s, 0.0, 2.0
    )
    normalized_root = np.clip(
        root_speed_frame / cfg.root_speed_reference_m_s, 0.0, 2.0
    )
    density_signal = np.clip(0.78 * normalized_joint + 0.22 * normalized_root, 0.0, 1.0)

    return {
        "shape": [frames, int(x.shape[1])],
        "fps": float(fps),
        "duration_seconds": duration,
        "root_total_travel_m": float(root_step.sum()) if root_step.size else 0.0,
        "root_travel_per_second_m_s": float(root_step.sum() / max(duration, 1.0e-6))
        if root_step.size
        else 0.0,
        "root_step_mean_m": float(root_step.mean()) if root_step.size else 0.0,
        "root_step_p95_m": float(np.percentile(root_step, 95)) if root_step.size else 0.0,
        "root_speed_mean_m_s": float(root_speed_step.mean())
        if root_speed_step.size
        else 0.0,
        "root_speed_p95_m_s": float(np.percentile(root_speed_step, 95))
        if root_speed_step.size
        else 0.0,
        "joint_speed_mean_rad_s": float(joint_speed_step.mean())
        if joint_speed_step.size
        else 0.0,
        "joint_speed_p95_rad_s": float(np.percentile(joint_speed_step, 95))
        if joint_speed_step.size
        else 0.0,
        "joint_speed_max_rad_s": float(joint_speed_step.max())
        if joint_speed_step.size
        else 0.0,
        "static_frame_ratio": float(static_mask.mean()) if frames else 1.0,
        "static_frames": int(static_mask.sum()),
        "longest_static_streak_frames": int(streak),
        "longest_static_streak_seconds": float(streak / fps),
        "longest_static_streak_span": [int(streak_start), int(streak_end)],
        "motion_density_mean": float(density_signal.mean()) if frames else 0.0,
        "motion_density_p95": float(np.percentile(density_signal, 95))
        if frames
        else 0.0,
        "motion_density_signal": density_signal.astype(np.float32),
        "static_mask": static_mask.astype(bool),
        "thresholds": {
            "static_joint_speed_rad_s": float(cfg.static_joint_speed_rad_s),
            "static_root_speed_m_s": float(cfg.static_root_speed_m_s),
        },
    }


def reportable_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop frame-wise arrays before writing JSON reports."""

    return {
        key: value
        for key, value in metrics.items()
        if key not in {"motion_density_signal", "static_mask"}
    }


def _numeric_slot_value(slot: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = slot.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            return float(np.clip(number, 0.0, 1.0))
    return None


def slot_activity_target(slot: Mapping[str, Any]) -> Optional[float]:
    """Infer desired motion density from scheduler fields without inventing labels."""

    direct = _numeric_slot_value(
        slot,
        (
            "target_motion_density",
            "motion_density",
            "desired_motion_density",
            "music_motion_density",
            "motion_energy",
            "music_energy",
            "energy",
            "arousal",
        ),
    )
    if direct is not None:
        return direct

    probabilities: Optional[Mapping[str, Any]] = None
    for key in ("music_semantic_probs", "semantic_probs", "event_semantic_probs"):
        value = slot.get(key)
        if isinstance(value, Mapping):
            probabilities = value
            break
    if probabilities is None:
        return None

    weights = {
        "pose_hold": 0.04,
        "hold": 0.06,
        "calm": 0.22,
        "calm_flow": 0.28,
        "resolution": 0.32,
        "transition": 0.55,
        "motif_recall": 0.58,
        "turning_flow": 0.76,
        "turn": 0.78,
        "build_up": 0.80,
        "dynamic": 0.86,
        "climax": 0.96,
    }
    weighted = 0.0
    mass = 0.0
    pose_hold_probability: Optional[float] = None
    for raw_key, raw_value in probabilities.items():
        try:
            probability = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(probability) or probability < 0.0:
            continue
        key = str(raw_key).strip().lower()
        if key == "pose_hold":
            pose_hold_probability = float(np.clip(probability, 0.0, 1.0))
        if key in weights:
            weighted += probability * weights[key]
            mass += probability

    target = weighted / mass if mass > 1.0e-8 else None
    if pose_hold_probability is not None:
        inverse_hold = 1.0 - pose_hold_probability
        target = inverse_hold if target is None else max(target, inverse_hold)
    if target is None:
        return None
    return float(np.clip(target, 0.0, 1.0))


def candidate_activity_assessment(
    metrics: Mapping[str, Any],
    target_activity: Optional[float],
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    cfg = thresholds or ActivityThresholds.from_env()
    target = None if target_activity is None else float(np.clip(target_activity, 0.0, 1.0))
    measured = float(metrics.get("motion_density_mean", 0.0))
    mismatch = abs(measured - target) if target is not None else 0.0
    active_required = bool(
        target is not None and target >= cfg.candidate_required_target
    )
    reasons: List[str] = []
    if active_required:
        if float(metrics.get("static_frame_ratio", 1.0)) > cfg.candidate_max_static_ratio:
            reasons.append("candidate_static_frame_ratio")
        if (
            float(metrics.get("longest_static_streak_seconds", 0.0))
            > cfg.candidate_max_static_seconds
        ):
            reasons.append("candidate_static_streak")
        if (
            float(metrics.get("joint_speed_mean_rad_s", 0.0))
            < cfg.candidate_min_joint_speed_rad_s
        ):
            reasons.append("candidate_joint_speed")
    hard_reject = bool(
        _env_bool("MOTION_ACTIVITY_CANDIDATE_HARD_GATE", True)
        and active_required
        and len(reasons) >= 2
    )
    penalty = cfg.candidate_penalty_weight * (
        mismatch
        + 0.35 * float(metrics.get("static_frame_ratio", 0.0))
        + (0.25 if active_required and reasons else 0.0)
    )
    return {
        "target_activity": target,
        "measured_activity": measured,
        "activity_mismatch": float(mismatch),
        "active_motion_required": active_required,
        "penalty": float(penalty),
        "hard_reject": hard_reject,
        "reasons": reasons,
        "immutable_physical_gates_relaxed": False,
    }


def _slot_span(
    slot_index: int,
    slots: Sequence[Mapping[str, Any]],
    assembly_report: Optional[Sequence[Mapping[str, Any]]],
    cursor: int,
    total_frames: int,
    fps: float,
) -> Tuple[int, int]:
    if assembly_report is not None and slot_index < len(assembly_report):
        row = assembly_report[slot_index]
        span = row.get("core_span")
        transition = row.get("transition_span")
        if isinstance(span, Sequence) and len(span) == 2:
            start = int(transition[0]) if isinstance(transition, Sequence) and len(transition) == 2 else int(span[0])
            end = int(span[1])
            return max(0, start), min(total_frames, max(start + 1, end))
    slot = slots[slot_index]
    target = None
    for key in ("target_frames", "music_length"):
        if slot.get(key) is not None:
            try:
                target = int(round(float(slot[key])))
                break
            except Exception:
                pass
    if target is None:
        duration = slot.get("duration", slot.get("duration_sec", 1.0))
        try:
            target = int(round(float(duration) * fps))
        except Exception:
            target = 1
    return int(cursor), min(total_frames, int(cursor + max(1, target)))


def per_slot_activity(
    motion: np.ndarray,
    slots: Sequence[Mapping[str, Any]],
    assembly_report: Optional[Sequence[Mapping[str, Any]]] = None,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
) -> List[Dict[str, Any]]:
    x = _as_motion(motion)
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for index, slot in enumerate(slots):
        start, end = _slot_span(
            index,
            slots,
            assembly_report,
            cursor,
            len(x),
            fps,
        )
        if end <= start:
            continue
        metrics = motion_activity_metrics(x[start:end], fps=fps, thresholds=thresholds)
        target = slot_activity_target(slot)
        row: Dict[str, Any] = {
            "slot": int(index),
            "span": [int(start), int(end)],
            "target_activity": target,
            **reportable_metrics(metrics),
        }
        if assembly_report is not None and index < len(assembly_report):
            source = assembly_report[index]
            for key in ("event_id", "event_path", "candidate_origin", "decision"):
                if key in source:
                    row[key] = source[key]
        rows.append(row)
        cursor = end
    return rows


def motion_density_alignment(per_slot: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    paired = [
        (
            float(row["target_activity"]),
            float(row.get("motion_density_mean", 0.0)),
        )
        for row in per_slot
        if row.get("target_activity") is not None
    ]
    if not paired:
        return {
            "available": False,
            "paired_slots": 0,
            "mae": None,
            "correlation": None,
            "alignment_score": None,
        }
    target = np.asarray([row[0] for row in paired], dtype=np.float64)
    measured = np.asarray([row[1] for row in paired], dtype=np.float64)
    mae = float(np.mean(np.abs(target - measured)))
    correlation: Optional[float] = None
    if len(target) >= 2 and target.std() > 1.0e-8 and measured.std() > 1.0e-8:
        correlation = float(np.corrcoef(target, measured)[0, 1])
    return {
        "available": True,
        "paired_slots": int(len(paired)),
        "target_mean": float(target.mean()),
        "measured_mean": float(measured.mean()),
        "mae": mae,
        "correlation": correlation,
        "alignment_score": float(np.clip(1.0 - mae, 0.0, 1.0)),
    }


def final_activity_gate(
    metrics: Mapping[str, Any],
    per_slot: Optional[Sequence[Mapping[str, Any]]] = None,
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    cfg = thresholds or ActivityThresholds.from_env()
    targets = [
        float(row["target_activity"])
        for row in (per_slot or [])
        if row.get("target_activity") is not None
    ]
    target_mean = float(np.mean(targets)) if targets else None
    if target_mean is None:
        max_static_ratio = cfg.final_default_max_static_ratio
        max_static_seconds = cfg.final_default_max_static_seconds
        min_joint_speed = cfg.final_default_min_joint_speed_rad_s
    else:
        max_static_ratio = float(np.clip(0.88 - 0.34 * target_mean, 0.50, 0.88))
        max_static_seconds = float(np.clip(5.0 - 3.0 * target_mean, 1.5, 5.0))
        min_joint_speed = float(0.03 + 0.05 * target_mean)

    reasons: List[str] = []
    if float(metrics.get("static_frame_ratio", 1.0)) > max_static_ratio:
        reasons.append("final_static_frame_ratio")
    if float(metrics.get("longest_static_streak_seconds", 0.0)) > max_static_seconds:
        reasons.append("final_static_streak")
    if float(metrics.get("joint_speed_mean_rad_s", 0.0)) < min_joint_speed:
        reasons.append("final_joint_speed")

    high_target = [
        row
        for row in (per_slot or [])
        if row.get("target_activity") is not None
        and float(row["target_activity"]) >= cfg.candidate_required_target
    ]
    failed_high_target = [
        row
        for row in high_target
        if float(row.get("static_frame_ratio", 1.0))
        > cfg.final_high_target_slot_max_static_ratio
    ]
    failed_fraction = float(len(failed_high_target) / max(1, len(high_target)))
    if high_target and failed_fraction > cfg.final_high_target_failed_slot_fraction:
        reasons.append("high_activity_slots_collapsed")

    hard_gate_enabled = _env_bool("MOTION_ACTIVITY_FINAL_HARD_GATE", True)
    return {
        "ok": bool(not reasons),
        "hard_gate_enabled": hard_gate_enabled,
        "reasons": reasons,
        "target_activity_mean": target_mean,
        "limits": {
            "max_static_frame_ratio": max_static_ratio,
            "max_static_streak_seconds": max_static_seconds,
            "min_joint_speed_mean_rad_s": min_joint_speed,
            "high_target_slot_max_static_ratio": cfg.final_high_target_slot_max_static_ratio,
            "high_target_failed_slot_fraction": cfg.final_high_target_failed_slot_fraction,
        },
        "high_target_slots": int(len(high_target)),
        "failed_high_target_slots": int(len(failed_high_target)),
        "failed_high_target_slot_fraction": failed_fraction,
        "failed_high_target_slot_indices": [
            int(row.get("slot", -1)) for row in failed_high_target
        ],
        "immutable_physical_gates_relaxed": False,
    }


def compare_motion_activity(
    reference: np.ndarray,
    candidate: np.ndarray,
    fps: float = 30.0,
    thresholds: Optional[ActivityThresholds] = None,
) -> Dict[str, Any]:
    ref = _as_motion(reference)
    out = _as_motion(candidate)
    length = min(len(ref), len(out))
    ref = ref[:length]
    out = out[:length]
    ref_metrics = motion_activity_metrics(ref, fps=fps, thresholds=thresholds)
    out_metrics = motion_activity_metrics(out, fps=fps, thresholds=thresholds)
    if length:
        ref_rot = rot6d_to_matrix_np(ref[:, ROT6D].reshape(length, NUM_JOINTS, 6))
        out_rot = rot6d_to_matrix_np(out[:, ROT6D].reshape(length, NUM_JOINTS, 6))
        relative = np.swapaxes(ref_rot, -1, -2) @ out_rot
        cosine = np.clip(
            (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
        angle = np.arccos(cosine)
        frame_rotation_delta = angle.mean(axis=1)
        frame_root_delta = np.linalg.norm(out[:, ROOT] - ref[:, ROOT], axis=-1)
        changed = (frame_rotation_delta > 0.01) | (frame_root_delta > 0.001)
    else:
        angle = np.zeros((0, NUM_JOINTS), dtype=np.float32)
        frame_root_delta = np.zeros((0,), dtype=np.float32)
        changed = np.zeros((0,), dtype=bool)
    return {
        "frames_compared": int(length),
        "rotation_delta_mean_rad": float(angle.mean()) if angle.size else 0.0,
        "rotation_delta_p95_rad": float(np.percentile(angle, 95)) if angle.size else 0.0,
        "root_delta_mean_m": float(frame_root_delta.mean())
        if frame_root_delta.size
        else 0.0,
        "root_delta_p95_m": float(np.percentile(frame_root_delta, 95))
        if frame_root_delta.size
        else 0.0,
        "changed_frame_ratio": float(changed.mean()) if changed.size else 0.0,
        "motion_density_delta": float(
            out_metrics["motion_density_mean"] - ref_metrics["motion_density_mean"]
        ),
        "static_frame_ratio_delta": float(
            out_metrics["static_frame_ratio"] - ref_metrics["static_frame_ratio"]
        ),
    }


def diagnose_motion(
    motion: np.ndarray,
    fps: float,
    slots: Optional[Sequence[Mapping[str, Any]]] = None,
    assembly_report: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    thresholds = ActivityThresholds.from_env()
    metrics = motion_activity_metrics(motion, fps=fps, thresholds=thresholds)
    slot_rows = per_slot_activity(
        motion,
        slots or [],
        assembly_report=assembly_report,
        fps=fps,
        thresholds=thresholds,
    )
    alignment = motion_density_alignment(slot_rows)
    gate = final_activity_gate(metrics, per_slot=slot_rows, thresholds=thresholds)
    return {
        "schema": "edge151_motion_activity_diagnostics_v1",
        "metrics": reportable_metrics(metrics),
        "per_slot": slot_rows,
        "motion_density_alignment": alignment,
        "acceptance_gate": gate,
    }


def _report_context(report: Mapping[str, Any]) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    slots = report.get("slots", [])
    stage_reports = report.get("stage_reports", {})
    assembly = stage_reports.get("closed_loop_concat", []) if isinstance(stage_reports, Mapping) else []
    return list(slots) if isinstance(slots, Sequence) else [], list(assembly) if isinstance(assembly, Sequence) else []


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-collapse", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    motion = np.load(args.input, allow_pickle=True)
    slots: List[Mapping[str, Any]] = []
    assembly: List[Mapping[str, Any]] = []
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        slots, assembly = _report_context(report)
    result = diagnose_motion(
        motion,
        fps=float(args.fps),
        slots=slots,
        assembly_report=assembly,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    if args.fail_on_collapse and not result["acceptance_gate"]["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
