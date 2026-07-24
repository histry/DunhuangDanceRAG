#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manifold-derived node and edge contracts for the time-expanded Event graph."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import numpy as np

from grounding.manifold_ops import lorentz_distance_sq_np
from motion_geometry.rotations import so3_geodesic_np


POSTURE_ORDER = {
    "floor_pose": 0,
    "kneeling": 1,
    "deep_squat": 2,
    "half_squat": 3,
    "standing": 4,
    "aerial": 5,
}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _db_value(
    db: Mapping[str, Any], key: str, event_id: int, default: Any
) -> Any:
    try:
        value = np.asarray(db[key])[int(event_id)]
        return value.item() if isinstance(value, np.generic) else value
    except Exception:
        return default


def _vector_gap(
    db: Mapping[str, Any],
    left_key: str,
    right_key: str,
    left_event: int,
    right_event: int,
) -> float:
    try:
        left = np.asarray(db[left_key], dtype=np.float64)[int(left_event)]
        right = np.asarray(db[right_key], dtype=np.float64)[int(right_event)]
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            return 0.0
        difference = left - right
        if difference.ndim == 0:
            return float(abs(difference))
        return float(np.mean(np.linalg.norm(difference, axis=-1)))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class EventGraphGeometryConfig:
    omega_weight: float = 0.10
    alpha_weight: float = 0.002
    posture_weight: float = 0.35
    pelvis_weight: float = 1.8
    floor_weight: float = 2.0
    contact_weight: float = 0.45
    root_velocity_weight: float = 0.30
    so3_weight: float = 0.55
    lorentz_weight: float = 0.15
    posture_hard: float = 2.0
    floor_hard_m: float = 0.20
    contact_hard: float = 0.75
    root_velocity_hard_mps: float = 2.0
    so3_hard_rad: float = 0.0

    @classmethod
    def from_environment(cls) -> "EventGraphGeometryConfig":
        return cls(
            omega_weight=_env_float("V46_53_GLOBAL_OMEGA_W", 0.10),
            alpha_weight=_env_float("V46_53_GLOBAL_ALPHA_W", 0.002),
            posture_weight=_env_float("V46_53_GLOBAL_POSTURE_W", 0.35),
            pelvis_weight=_env_float("V46_53_GLOBAL_PELVIS_W", 1.8),
            floor_weight=_env_float("V46_53_GLOBAL_FLOOR_W", 2.0),
            contact_weight=_env_float("V46_53_GLOBAL_CONTACT_W", 0.45),
            root_velocity_weight=_env_float(
                "V46_53_GLOBAL_ROOT_VEL_W", 0.30
            ),
            so3_weight=_env_float("V46_55_GRAPH_SO3_W", 0.55),
            lorentz_weight=_env_float("V46_55_GRAPH_LORENTZ_W", 0.15),
            posture_hard=_env_float("V46_55_GRAPH_POSTURE_HARD", 2.0),
            floor_hard_m=_env_float("V46_55_GRAPH_FLOOR_HARD_M", 0.20),
            contact_hard=_env_float("V46_55_GRAPH_CONTACT_HARD", 0.75),
            root_velocity_hard_mps=_env_float(
                "V46_55_GRAPH_ROOT_VEL_HARD_MPS", 2.0
            ),
            # Disabled by default: the downstream physical simulator remains
            # authoritative and existing code has no historical SO(3) hard cap.
            so3_hard_rad=_env_float("V46_55_GRAPH_SO3_HARD_RAD", 0.0),
        )


def event_node_feasibility(
    db: Mapping[str, Any],
    event_id: int,
) -> tuple[bool, tuple[str, ...]]:
    """Apply only immutable Event-level gates before graph construction."""

    index = int(event_id)
    count = len(np.asarray(db.get("paths", [])))
    reasons: list[str] = []
    if index < 0 or index >= count:
        reasons.append("event_index")
        return False, tuple(reasons)
    for key in ("anatomy_hard_valid", "anatomy_valid", "event_heading_valid"):
        if key in db:
            values = np.asarray(db[key], dtype=bool)
            if index >= len(values) or not bool(values[index]):
                reasons.append(key)
    quality = float(
        _db_value(
            db,
            "v46_53_combined_quality",
            index,
            _db_value(db, "event_quality_scores", index, 0.5),
        )
    )
    if not np.isfinite(quality):
        reasons.append("nonfinite_quality")
    return not reasons, tuple(reasons)


def so3_product_endpoint_distance(
    db: Mapping[str, Any],
    previous_event: int,
    current_event: int,
) -> tuple[float, bool]:
    """RMS product-manifold distance over the 24 endpoint rotations."""

    left_key = "v46_55_exit_rotation_matrix"
    right_key = "v46_55_entry_rotation_matrix"
    if left_key not in db or right_key not in db:
        return 0.0, False
    try:
        left = np.asarray(db[left_key], dtype=np.float64)[int(previous_event)]
        right = np.asarray(db[right_key], dtype=np.float64)[int(current_event)]
        if left.shape != (24, 3, 3) or right.shape != (24, 3, 3):
            return 0.0, False
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return 0.0, False
        angles = np.asarray(so3_geodesic_np(left, right), dtype=np.float64)
        return float(np.sqrt(np.mean(np.square(angles)))), True
    except Exception:
        return 0.0, False


def lorentz_hierarchy_distance(
    db: Mapping[str, Any],
    previous_event: int,
    current_event: int,
) -> tuple[float, bool]:
    """Distance between paper-one Lorentz factors when they are embedded."""

    key = "v46_53_mixed_lorentz"
    if key not in db:
        return 0.0, False
    try:
        points = np.asarray(db[key], dtype=np.float64)
        left = points[int(previous_event)]
        right = points[int(current_event)]
        if left.ndim != 1 or left.shape != right.shape or left.size < 2:
            return 0.0, False
        curvature = float(
            np.asarray(db.get("v46_53_mixed_curvature", 1.0)).reshape(-1)[0]
        )
        distance_sq = float(
            np.asarray(
                lorentz_distance_sq_np(left, right, curvature=curvature)
            ).reshape(-1)[0]
        )
        if not np.isfinite(distance_sq):
            return 0.0, False
        return float(np.sqrt(max(0.0, distance_sq))), True
    except Exception:
        return 0.0, False


def manifold_edge_cost(
    db: Mapping[str, Any],
    previous_event: int,
    current_event: int,
    *,
    config: EventGraphGeometryConfig | None = None,
    boundary_reset: bool = False,
) -> dict[str, Any]:
    """Return the composite Event-edge cost and hard support decision."""

    cfg = config or EventGraphGeometryConfig.from_environment()
    previous = int(previous_event)
    current = int(current_event)
    omega = _vector_gap(
        db, "v46_53_exit_omega", "v46_53_entry_omega", previous, current
    )
    alpha = _vector_gap(
        db, "v46_53_exit_alpha", "v46_53_entry_alpha", previous, current
    )
    root_velocity = _vector_gap(
        db,
        "v46_53_exit_root_velocity_mps",
        "v46_53_entry_root_velocity_mps",
        previous,
        current,
    )
    posture_exit = str(_db_value(db, "posture_exit", previous, "standing"))
    posture_entry = str(_db_value(db, "posture_entry", current, "standing"))
    posture = float(
        abs(
            POSTURE_ORDER.get(posture_exit, 4)
            - POSTURE_ORDER.get(posture_entry, 4)
        )
    )
    pelvis = abs(
        float(_db_value(db, "pelvis_height_exit_norm", previous, 0.8))
        - float(_db_value(db, "pelvis_height_entry_norm", current, 0.8))
    )
    floor = abs(
        float(_db_value(db, "exit_floor_offset_m", previous, 0.0))
        - float(_db_value(db, "entry_floor_offset_m", current, 0.0))
    )
    contact = 0.0
    try:
        left_contact = np.asarray(db["contact_exit"], dtype=np.float64)[previous]
        right_contact = np.asarray(db["contact_entry"], dtype=np.float64)[current]
        if left_contact.shape == right_contact.shape:
            contact = float(np.mean(np.abs(left_contact - right_contact)))
    except Exception:
        pass
    so3, has_so3 = so3_product_endpoint_distance(db, previous, current)
    lorentz, has_lorentz = lorentz_hierarchy_distance(
        db, previous, current
    )

    values = np.asarray(
        [omega, alpha, root_velocity, posture, pelvis, floor, contact, so3, lorentz],
        dtype=np.float64,
    )
    hard_reasons: list[str] = []
    if not np.isfinite(values).all():
        hard_reasons.append("nonfinite_edge_geometry")
    reset_multiplier = 1.35 if bool(boundary_reset) else 1.0
    if cfg.posture_hard > 0.0 and posture > cfg.posture_hard:
        hard_reasons.append("posture")
    if cfg.floor_hard_m > 0.0 and floor > cfg.floor_hard_m * reset_multiplier:
        hard_reasons.append("floor")
    if cfg.contact_hard > 0.0 and contact > cfg.contact_hard * reset_multiplier:
        hard_reasons.append("contact")
    if (
        cfg.root_velocity_hard_mps > 0.0
        and root_velocity > cfg.root_velocity_hard_mps * reset_multiplier
    ):
        hard_reasons.append("root_velocity")
    if cfg.so3_hard_rad > 0.0 and has_so3 and so3 > cfg.so3_hard_rad:
        hard_reasons.append("so3")

    physical = (
        cfg.omega_weight * omega
        + cfg.alpha_weight * alpha
        + cfg.posture_weight * posture
        + cfg.pelvis_weight * pelvis
        + cfg.floor_weight * floor
        + cfg.contact_weight * contact
        + cfg.root_velocity_weight * root_velocity
    )
    total = (
        physical
        + (cfg.so3_weight * so3 if has_so3 else 0.0)
        + (cfg.lorentz_weight * lorentz if has_lorentz else 0.0)
    )
    if not np.isfinite(total):
        hard_reasons.append("nonfinite_total")
        total = 1.0e6
    return {
        "total": float(total),
        "physical": float(physical),
        "omega_gap_radps": float(omega),
        "alpha_gap_radps2": float(alpha),
        "root_velocity_gap_mps": float(root_velocity),
        "posture_gap": float(posture),
        "pelvis_gap_norm": float(pelvis),
        "floor_gap_m": float(floor),
        "contact_gap": float(contact),
        "so3_product_distance_rad": float(so3),
        "so3_available": bool(has_so3),
        "lorentz_hierarchy_distance": float(lorentz),
        "lorentz_available": bool(has_lorentz),
        "hard_feasible": not hard_reasons,
        "hard_reasons": tuple(dict.fromkeys(hard_reasons)),
        "boundary_reset": bool(boundary_reset),
    }


__all__ = [
    "EventGraphGeometryConfig",
    "event_node_feasibility",
    "lorentz_hierarchy_distance",
    "manifold_edge_cost",
    "so3_product_endpoint_distance",
]
