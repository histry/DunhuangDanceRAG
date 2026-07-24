#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.52 fresh-WAV, anatomy-gated, posture-aware closed-loop generation.

This entrypoint preserves the V46.51 audio transaction and V46.50 heading state,
then installs four scientifically motivated policies:
1. event anatomy hard gate and core-warp gate;
2. support-floor alignment without forcing pelvis height;
3. C2 geodesic SO(3) transition bridge;
4. stage-wise anatomy audit with masked rollback.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scheduling.validate_schedule import audit_contract, save_json
import routing.heading_closed_loop as v4650
from contracts.anatomy import (
    AnatomyThresholds,
    anatomy_metrics_np,
    env_bool,
    env_float,
    env_int,
    evaluate_anatomy_contract,
    event_anatomy_features,
    frame_anomaly_score_np,
    transition_anatomy_risk,
)
from support.motion_geometry import (
    canonicalize_event_root_np,
    make_so3_transition,
    project_transition_floor_np,
    recompute_transition_contacts_np,
)

base = v4650.base
_ORIG_ALIGN = v4650._align_core_to_stage_heading
_ORIG_BUILD_PROPOSAL = v4650._build_heading_proposal
_ORIG_APPLY = v4650.apply_generators_with_heading_guard
_ORIG_TRANSITION_RISK = base.transition_risk
_ORIG_RISK_SCORE = base.risk_score
_ORIG_RISK_SAFE = base.risk_safe


def _arg_value(argv: Sequence[str], flag: str) -> Optional[str]:
    args = list(argv)
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    return args[idx + 1] if idx + 1 < len(args) else None


def _runtime_fps(argv: Sequence[str]) -> float:
    """Resolve one generation FPS and reject conflicting contracts."""
    values: Dict[str, float] = {}
    config_path = _arg_value(argv, "--config")
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Generation config does not exist: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise RuntimeError(f"Generation config is not a mapping: {path}")
        if config.get("fps") is not None:
            values["config"] = float(config["fps"])
    for name in ("V46_FPS", "V46_51_FPS"):
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            values[name] = float(raw)
    if not values:
        values["legacy_default"] = 30.0
    for source, value in values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"Invalid generation FPS from {source}: {value!r}"
            )
    reference = next(iter(values.values()))
    mismatched = {
        source: value
        for source, value in values.items()
        if abs(value - reference) > 1.0e-6
    }
    if mismatched:
        raise RuntimeError(f"Conflicting generation FPS contracts: {values}")
    return float(reference)


def _db_value(db: Dict[str, Any], key: str, event_id: int, default: Any) -> Any:
    try:
        arr = np.asarray(db[key])
        value = arr[int(event_id)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


def _align_core_v52(
    v46: Any,
    prev: Optional[np.ndarray],
    core: np.ndarray,
    stage_heading_rad: float,
    cfg: Any,
    event_id: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Normalize every event independently before applying planner-owned stage
    # heading and cumulative XZ placement.  Only a constant Root-Y/XZ
    # translation is changed, so kneels, crouches and jumps remain intact.
    localized, floor_report = canonicalize_event_root_np(
        core,
        target_floor_y=env_float("V46_54_STAGE_FLOOR_Y", 0.0),
        floor_quantile=env_float("V46_54_EVENT_FLOOR_QUANTILE", 5.0),
        max_floor_penetration_m=env_float(
            "V46_54_EVENT_MAX_FLOOR_PENETRATION_M",
            0.005,
        ),
    )
    out, report = _ORIG_ALIGN(
        v46,
        prev,
        localized,
        stage_heading_rad,
        cfg,
        event_id,
    )
    out = base.enforce_contract(v46, out, cfg, source_hint=f"v46_52_floor_align:{event_id}")
    report = dict(report or {})
    report["v46_52_support_floor_alignment"] = floor_report
    report["root_y_ramp_applied"] = False
    report["root_trajectory_policy"] = (
        "event-local first-frame XZ + cumulative stage endpoint"
    )
    report["root_y_policy"] = (
        "one event-wide FK floor offset; internal posture height is preserved"
    )
    return out.astype(np.float32), report


def _build_bridge_v52(v46: Any, prev: np.ndarray, core: np.ndarray, trans_len: int, cfg: Any) -> np.ndarray:
    trans_len = int(trans_len)
    if trans_len <= 0:
        return np.zeros((0, 151), dtype=np.float32)
    if not env_bool("V46_52_RIEMANNIAN_BRIDGE_ENABLE", True):
        return base._V46_52_ORIG_BUILD_BRIDGE(v46, prev, core, trans_len, cfg)
    fps = float(getattr(cfg, "fps", 30.0))
    bridge = make_so3_transition(
        prev,
        core,
        trans_len,
        fps=fps,
        angular_speed_cap_radps=env_float(
            "V46_54_TRANSITION_ANGULAR_SPEED_CAP_RADPS",
            8.0,
        ),
        root_horizontal_speed_cap_mps=env_float(
            "V46_54_TRANSITION_ROOT_XZ_SPEED_CAP_MPS",
            1.5,
        ),
        root_vertical_speed_cap_mps=env_float(
            "V46_54_TRANSITION_ROOT_Y_SPEED_CAP_MPS",
            0.9,
        ),
    )
    # Project rotations/contact once before the transition-specific physical
    # post-processing.  Calling the generic contract afterwards would derive
    # binary contacts again and silently destroy the endpoint contact ramps.
    bridge = base.enforce_contract(
        v46,
        bridge,
        cfg,
        source_hint="v46_52_c2_so3_bridge_pre_physics",
    )
    bridge, floor_report = project_transition_floor_np(
        bridge,
        target_floor_y=env_float("V46_54_STAGE_FLOOR_Y", 0.0),
        clearance_m=env_float(
            "V46_54_TRANSITION_FLOOR_CLEARANCE_M",
            0.002,
        ),
        smoothing_frames=max(
            1,
            int(
                round(
                    env_float(
                        "V46_54_TRANSITION_FLOOR_SMOOTH_SECONDS",
                        5.0 / 30.0,
                    )
                    * fps
                )
            ),
        ),
    )
    bridge, contact_report = recompute_transition_contacts_np(
        bridge,
        fps=fps,
        floor_y=env_float("V46_54_STAGE_FLOOR_Y", 0.0),
        left_contact=prev[-1, :4],
        right_contact=core[0, :4],
        ramp_seconds=env_float(
            "V46_54_TRANSITION_CONTACT_RAMP_SECONDS",
            4.0 / 30.0,
        ),
    )
    # Reports are attached to the candidate risk by the subsequent physical
    # audit; keep the bridge itself free of side-channel mutable state.
    _ = floor_report, contact_report
    return np.asarray(bridge, dtype=np.float32)


def _transition_risk_v52(v46: Any, previous: np.ndarray, transition: np.ndarray, following: np.ndarray, fps: float) -> Dict[str, Any]:
    risk = dict(_ORIG_TRANSITION_RISK(v46, previous, transition, following, fps))
    anatomy = transition_anatomy_risk(previous, transition, following, fps=fps)
    risk.update({
        "anatomy_quality": float(anatomy["anatomy_quality"]),
        "anatomy_valid": bool(anatomy["anatomy_valid"]),
        "anatomy_risk_score": float(anatomy["anatomy_risk_score"]),
        "anatomy_hard_reject": bool(anatomy["anatomy_hard_reject"]),
        "pelvis_height_gap_norm": float(anatomy["pelvis_height_gap_norm"]),
        "body_height_gap_norm": float(anatomy["body_height_gap_norm"]),
        "floor_offset_gap_m": float(anatomy["floor_offset_gap_m"]),
        "root_velocity_gap_mps": float(anatomy["root_velocity_gap_mps"]),
        "posture_gap": int(anatomy["posture_gap"]),
        "posture_exit": anatomy["posture_exit"],
        "posture_entry": anatomy["posture_entry"],
        "required_transition_seconds": float(anatomy["required_transition_seconds"]),
        "available_transition_seconds": float(anatomy["available_transition_seconds"]),
        "v46_52_anatomy_detail": anatomy,
    })
    return risk


def _risk_score_v52(risk: Dict[str, Any]) -> float:
    return float(
        _ORIG_RISK_SCORE(risk)
        + env_float("V46_52_COMBINED_ANATOMY_RISK_W", 0.85)
        * float(risk.get("anatomy_risk_score", 0.0))
    )


def _risk_safe_v52(risk: Dict[str, Any]) -> bool:
    return bool(
        _ORIG_RISK_SAFE(risk)
        and bool(risk.get("anatomy_valid", True))
        and not bool(risk.get("anatomy_hard_reject", False))
        and float(risk.get("pelvis_height_gap_norm", 0.0))
        <= env_float("V46_52_PELVIS_GAP_SAFE", 0.22)
        and float(risk.get("floor_offset_gap_m", 0.0))
        <= env_float("V46_52_FLOOR_GAP_SAFE_M", 0.15)
        and float(risk.get("root_velocity_gap_mps", 0.0))
        <= env_float("V46_52_ROOT_VELOCITY_GAP_SAFE_MPS", 1.25)
    )


def _build_proposal_v52(*args, **kwargs):
    proposal, extra = _ORIG_BUILD_PROPOSAL(*args, **kwargs)
    db = kwargs.get("db")
    event_id = int(kwargs.get("event_id", proposal.event_id))
    cfg = kwargs.get("cfg")
    event_valid = bool(_db_value(db, "anatomy_valid", event_id, False)) if db is not None else False
    db_quality = float(_db_value(db, "anatomy_quality", event_id, 0.0)) if db is not None else 0.0
    core_feat = event_anatomy_features(proposal.core, fps=float(getattr(cfg, "fps", 30.0)))

    raw = base.load_event_motion(
        kwargs.get("v46"),
        proposal.event_path,
        cfg,
        source_hint=f"v46_52_warp_probe:{event_id}",
    )
    warp = float(len(proposal.core) / max(1, len(raw)))
    warp_min = env_float("V46_52_CORE_WARP_MIN", 0.72)
    warp_max = env_float("V46_52_CORE_WARP_MAX", 1.32)
    warp_hard = not (warp_min <= warp <= warp_max)

    anatomy_detail = proposal.risk.get("v46_52_anatomy_detail", {}) if isinstance(proposal.risk, dict) else {}
    hard = (
        not event_valid
        or db_quality < env_float("V46_52_EVENT_ANATOMY_QUALITY_MIN", 0.48)
        or not bool(core_feat["anatomy_valid"])
        or bool(anatomy_detail.get("anatomy_hard_reject", False))
        or warp_hard
    )
    added = (
        env_float("V46_52_DB_QUALITY_PENALTY_W", 0.8) * (1.0 - db_quality)
        + env_float("V46_52_WARP_PENALTY_W", 1.2) * abs(math_log_ratio(warp))
    )
    proposal.risk_score = float(proposal.risk_score + added + (1e6 if hard else 0.0))
    proposal.safe = bool(proposal.safe and not hard)
    proposal.risk["v46_52_event_gate"] = {
        "db_anatomy_valid": event_valid,
        "db_anatomy_quality": db_quality,
        "runtime_core_anatomy": core_feat,
        "core_warp": warp,
        "core_warp_range": [warp_min, warp_max],
        "core_warp_hard_reject": warp_hard,
        "hard_reject": hard,
    }
    extra = dict(extra)
    extra.setdefault("heading_detail", {})["hard_reject"] = bool(
        extra.get("heading_detail", {}).get("hard_reject", False) or hard
    )
    extra["v46_52_event_gate"] = proposal.risk["v46_52_event_gate"]

    if not proposal.safe and not env_bool("V46_52_ALLOW_UNSAFE_RESCUE", False):
        extra["heading_detail"]["hard_reject"] = True
    return proposal, extra


def math_log_ratio(value: float) -> float:
    value = max(float(value), 1e-6)
    return float(np.log(value))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if radius <= 0:
        return m
    kernel = np.ones(2 * radius + 1, dtype=np.int32)
    return np.convolve(m.astype(np.int32), kernel, mode="same") > 0


def _apply_generators_v52(v46: Any, motion_ref: np.ndarray, cond: np.ndarray, seam_mask: np.ndarray, args: Any, cfg: Any):
    motion, stage = _ORIG_APPLY(v46, motion_ref, cond, seam_mask, args, cfg)
    ref_metrics = anatomy_metrics_np(motion_ref, fps=float(getattr(cfg, "fps", 30.0)))
    out_metrics = anatomy_metrics_np(motion, fps=float(getattr(cfg, "fps", 30.0)))
    out_ok, out_reasons = evaluate_anatomy_contract(out_metrics, AnatomyThresholds.from_env())
    rollback = {
        "enabled": True,
        "before": ref_metrics,
        "after_before_rollback": out_metrics,
        "after_ok_before_rollback": bool(out_ok),
        "reasons_before_rollback": out_reasons,
        "rolled_back_frames": 0,
        "full_rollback": False,
    }
    if not out_ok:
        ref_score = frame_anomaly_score_np(motion_ref)
        out_score = frame_anomaly_score_np(motion)
        seam = np.asarray(seam_mask).reshape(len(motion), -1).max(axis=1) > 0.01
        bad = seam & (out_score > np.maximum(ref_score + env_float("V46_52_ROLLBACK_SCORE_MARGIN", 0.06), env_float("V46_52_ROLLBACK_SCORE_ABS", 0.12)))
        bad = _dilate(bad, env_int("V46_52_ROLLBACK_DILATE", 3))
        motion = np.asarray(motion, dtype=np.float32).copy()
        motion[bad] = np.asarray(motion_ref, dtype=np.float32)[bad]
        rollback["rolled_back_frames"] = int(bad.sum())
        repaired = anatomy_metrics_np(motion, fps=float(getattr(cfg, "fps", 30.0)))
        repaired_ok, repaired_reasons = evaluate_anatomy_contract(repaired, AnatomyThresholds.from_env())
        if not repaired_ok and env_bool("V46_52_FULL_ROLLBACK_ON_ANATOMY_FAIL", True):
            motion = np.asarray(motion_ref, dtype=np.float32).copy()
            rollback["full_rollback"] = True
            repaired = anatomy_metrics_np(motion, fps=float(getattr(cfg, "fps", 30.0)))
            repaired_ok, repaired_reasons = evaluate_anatomy_contract(repaired, AnatomyThresholds.from_env())
        rollback["after"] = repaired
        rollback["after_ok"] = bool(repaired_ok)
        rollback["after_reasons"] = repaired_reasons
        if not repaired_ok:
            raise RuntimeError("V46.52 generator anatomy rollback failed: " + " | ".join(repaired_reasons))
    else:
        rollback["after"] = out_metrics
        rollback["after_ok"] = True
        rollback["after_reasons"] = []
    stage["v46_52_anatomy_guard"] = rollback
    return motion.astype(np.float32), stage


def _install_patches() -> None:
    if not hasattr(base, "_V46_52_ORIG_BUILD_BRIDGE"):
        base._V46_52_ORIG_BUILD_BRIDGE = base.build_bridge
    base.build_bridge = _build_bridge_v52
    base.transition_risk = _transition_risk_v52
    base.risk_score = _risk_score_v52
    base.risk_safe = _risk_safe_v52
    v4650._align_core_to_stage_heading = _align_core_v52
    v4650._build_heading_proposal = _build_proposal_v52
    v4650.apply_generators_with_heading_guard = _apply_generators_v52


def _resolve_motion_path(
    report_path: Path,
    report: Dict[str, Any],
    explicit_motion_path: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the generated NPY without assuming a *.report.json name."""

    raw_candidates = []

    if explicit_motion_path is not None:
        raw_candidates.append(Path(explicit_motion_path))

    for key in ("motion", "output", "motion_path", "npy"):
        value = report.get(key)
        if value:
            raw_candidates.append(Path(str(value)))

    if report_path.name.endswith(".report.json"):
        prefix = report_path.name[:-len(".report.json")]
        raw_candidates.append(report_path.with_name(prefix + ".npy"))

    # Current research launcher uses results/report.json + results/motion.npy.
    raw_candidates.append(report_path.with_name("motion.npy"))

    checked = set()

    for raw in raw_candidates:
        raw = Path(raw).expanduser()

        if raw.is_absolute():
            variants = [raw]
        else:
            variants = [
                report_path.parent / raw,
                ROOT / raw,
                raw,
            ]

        for candidate in variants:
            try:
                candidate = candidate.resolve()
            except Exception:
                continue

            key = str(candidate)

            if key in checked:
                continue

            checked.add(key)

            if candidate.is_file() and candidate.suffix.lower() == ".npy":
                return candidate

    return None


def _patch_report(
    report_path: Path,
    contract: Dict[str, Any],
    motion_path: Optional[Path] = None,
    fps: float = 30.0,
) -> None:
    if not report_path.is_file():
        return
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    report["version"] = "v46_52_fresh_wav_anatomy_posture_riemannian_closed_loop"
    report["audio_schedule_transaction"] = {
        "schema": contract.get("schema"),
        "ok": contract.get("ok"),
        "audio": contract.get("audio"),
        "schedule_path": contract.get("schedule_path"),
        "schedule_sha256": contract.get("schedule_sha256"),
        "required_run_id": contract.get("required_run_id"),
        "transaction": contract.get("transaction"),
        "num_slots": contract.get("num_slots"),
        "total_target_frames": contract.get("total_target_frames"),
        "expected_audio_target_frames": contract.get("expected_audio_target_frames"),
        "frame_error": contract.get("frame_error"),
        "overlap_count": contract.get("overlap_count"),
        "gap_count": contract.get("gap_count"),
    }
    report["v46_52_env"] = {k: v for k, v in os.environ.items() if k.startswith("V46_52_")}
    resolved_motion = _resolve_motion_path(
        report_path,
        report,
        explicit_motion_path=motion_path,
    )

    if resolved_motion is not None:
        x = np.load(resolved_motion, allow_pickle=True)

        if np.asarray(x).ndim == 3:
            x = np.asarray(x)[0]

        m = anatomy_metrics_np(x, fps=float(fps))
        ok, reasons = evaluate_anatomy_contract(m)

        report["v46_52_final_anatomy"] = {
            "ok": ok,
            "reasons": reasons,
            "motion_path": str(resolved_motion),
            **m,
        }
    save_json(report, report_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    audio = _arg_value(args, "--audio")
    schedule = _arg_value(args, "--slots_json")
    report_json = _arg_value(args, "--json")
    output = _arg_value(args, "--out")
    fps = _runtime_fps(args)
    if not audio or not schedule:
        raise RuntimeError("V46.52 requires --audio and a fresh --slots_json")
    required_run_id = os.environ.get("V46_51_SCHEDULE_RUN_ID")
    if not required_run_id:
        raise RuntimeError("V46_51_SCHEDULE_RUN_ID is required")
    contract = audit_contract(
        audio=audio,
        schedule=schedule,
        fps=fps,
        required_run_id=required_run_id,
        require_fresh=True,
        max_frame_error=int(float(os.environ.get("V46_51_MAX_FRAME_ERROR", "2"))),
        max_seconds_error=float(os.environ.get("V46_51_MAX_SECONDS_ERROR", "0.10")),
        require_raw_report=True,
    )
    save_json(contract, Path(schedule).with_suffix(Path(schedule).suffix + ".pre_generate_contract.json"))
    if not contract["ok"]:
        raise RuntimeError("Fresh-WAV contract failed: " + "; ".join(contract["reasons"]))
    _install_patches()
    rc = int(v4650.main(args))
    if report_json:
        _patch_report(
            Path(report_json),
            contract,
            motion_path=Path(output) if output else None,
            fps=fps,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
