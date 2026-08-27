"""Observable-only boundary contract shared by training, audit and inference.

No function in this module accepts the hidden clean interior or a noise seed.
Velocities in conditioning are per-frame product tangents; audit quantities
are world-space metres/seconds. Multiple disjoint seams are supported.
"""
from __future__ import annotations

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Geometry-only data inspection remains usable without Torch.
    torch = None
    F = None

from motion_geometry.product_manifold import product_log_torch


BOUNDARY_PROTOCOL = "observable_full_bridge_v1"
BOUNDARY_FEATURE_DIM = 302  # phase/core + two relative poses + two endpoint velocities


def _core(seam):
    if seam.ndim == 3:
        seam = seam[..., 0]
    return seam >= 0.5


def boundary_features_torch(motion, seam):
    """[B,T,302], using only observed motion and the supplied edit support.

    Anchor poses and velocities always come from OUTSIDE the corrupted core.
    A truncated seam without both external anchors has zero conditioning.
    There is no CPU transfer or batch loop on the CUDA path.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for boundary conditioning")
    core = _core(seam)
    batch, frames = core.shape
    index = torch.arange(frames, device=motion.device)[None].expand(batch, -1)
    starts = core & ~F.pad(core[:, :-1], (1, 0), value=False)
    ends = core & ~F.pad(core[:, 1:], (0, 1), value=False)
    left = torch.where(starts, index - 1, -torch.ones_like(index)).cummax(1).values
    right = torch.where(ends, index + 1, torch.full_like(index, frames))
    right = right.flip(1).cummin(1).values.flip(1)
    previous = torch.where(core, index, -torch.full_like(index, frames)).cummax(1).values
    following = torch.where(core, index, torch.full_like(index, 2 * frames))
    following = following.flip(1).cummin(1).values.flip(1)
    nearest = torch.where(index - previous <= following - index, previous, following).clamp(0, frames - 1)
    left, right = left.gather(1, nearest), right.gather(1, nearest)
    valid = (left >= 1) & (right < frames - 1) & core.any(1)[:, None]
    active = (seam[..., 0] if seam.ndim == 3 else seam) > 0

    def gather(at):
        return motion.gather(1, at.clamp(0, frames - 1)[..., None].expand(-1, -1, motion.shape[-1]))

    a, b = gather(left), gather(right)
    phase = ((index - left).to(motion.dtype) / (right - left).clamp_min(1)).clamp(0, 1)
    features = torch.cat([
        phase[..., None], core[..., None].to(motion.dtype),
        product_log_torch(motion, a), product_log_torch(motion, b),
        -product_log_torch(a, gather(left - 1)), product_log_torch(b, gather(right + 1)),
    ], dim=-1)
    return torch.where((valid & active)[..., None], features, torch.zeros_like(features))


def boundary_metrics_torch(joints, seam, fps):
    """Actual FK boundary velocity jump, acceleration and jerk, not GT error.

    Boundary velocity jumps compare the crossing step to its adjacent external
    step. Interior derivatives include all stencils touching a seam core.
    A missing endpoint is reported as invalid instead of silently passing.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for batched boundary metrics")
    if joints.shape[1] < 4:
        raise ValueError("boundary metrics require at least four frames")
    core = _core(seam)
    starts = core & ~F.pad(core[:, :-1], (1, 0), value=False)
    ends = core & ~F.pad(core[:, 1:], (0, 1), value=False)
    valid = core.any(1) & ~core[:, :2].any(1) & ~core[:, -2:].any(1)
    # Higher differences amplify float32 cancellation; FK itself stays on GPU.
    coords = joints.to(torch.float64)
    velocity = torch.diff(coords, dim=1) * float(fps)
    jump = torch.linalg.vector_norm(torch.diff(velocity, dim=1), dim=-1).mean(-1)
    boundary = starts[:, 2:] | ends[:, :-2]
    endpoint = (jump * boundary).sum(1) / boundary.sum(1).clamp_min(1)
    result = {"endpoint_velocity_jump_mps": endpoint, "valid": valid}
    for order, key in ((2, "seam_acceleration_mps2"), (3, "seam_jerk_mps3")):
        length = coords.shape[1] - order
        support = torch.stack([core[:, i:i + length] for i in range(order + 1)]).any(0)
        value = torch.linalg.vector_norm(torch.diff(coords, n=order, dim=1) * float(fps)**order, dim=-1).mean(-1)
        result[key] = (value * support).sum(1) / support.sum(1).clamp_min(1)
    result["temporal_energy"] = result["seam_acceleration_mps2"] / 10.0 + result["seam_jerk_mps3"] / 1000.0
    return result


def observable_gate(before, after, cfg):
    """Explicit endpoint and temporal decisions; neither depends on hidden GT."""
    floor = 1e-6
    finite = all(np.isfinite(float(row[k])) for row in (before, after)
                 for k in ("endpoint_velocity_jump_mps", "temporal_energy", "seam_jerk_mps3"))
    valid = bool(before["valid"] and after["valid"] and finite)

    def gain(key):
        a, b = float(before[key]), float(after[key])
        return (a - b) / a if a > floor else (1.0 if b <= floor else -1.0)

    endpoint_gain = gain("endpoint_velocity_jump_mps") if finite else -1.0
    temporal_gain = gain("temporal_energy") if finite else -1.0
    endpoint_ok = valid and endpoint_gain >= cfg.checkpoint_validation_min_endpoint_repair_gain
    jerk_ok = valid and after["seam_jerk_mps3"] <= before["seam_jerk_mps3"] * 1.02 + floor
    temporal_ok = valid and temporal_gain >= cfg.checkpoint_validation_min_temporal_repair_gain and jerk_ok
    reasons = []
    if not valid:
        reasons.append("invalid_or_unanchored_boundary")
    if not endpoint_ok:
        reasons.append("observable_endpoint_not_improved")
    if not temporal_ok:
        reasons.append("observable_temporal_not_improved")
    if not jerk_ok:
        reasons.append("actual_seam_jerk_regressed")
    return {"schema": BOUNDARY_PROTOCOL, "accepted": endpoint_ok and temporal_ok,
            "endpoint_accepted": endpoint_ok, "temporal_accepted": temporal_ok,
            "endpoint_informative": before["endpoint_velocity_jump_mps"] > floor,
            "temporal_informative": before["temporal_energy"] > floor,
            "endpoint_gain": endpoint_gain, "temporal_gain": temporal_gain,
            "jerk_non_regression": jerk_ok, "before": before, "after": after,
            "hidden_clean_used": False, "reasons": reasons}
