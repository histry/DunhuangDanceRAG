"""Product-manifold operations for the EDGE-151 motion representation.

The geometric part of one frame is modelled as

    R^3 (root translation) x SO(3)^24 (local joint rotations).

Contact channels are deliberately not part of this manifold; callers handle
their four logits separately.  The tangent layout is therefore 75D:
``root_xyz[3] + joint_rotvec[24, 3]``.

All NumPy operations are dependency-light.  Torch variants are differentiable
and are used by the V45/V46 training paths when PyTorch is available.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    so3_log_np,
)
from motion_geometry.smpl24 import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROT6D_END,
    ROT6D_START,
)

try:  # Training is optional for geometry-only and audit environments.
    import torch
except Exception:  # pragma: no cover
    torch = None

TANGENT_DIM = 3 + NUM_JOINTS * 3
PRODUCT_STATE_DIM = 4 + TANGENT_DIM
_EPS = 1.0e-8


def _validate_edge_np(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != MOTION_DIM:
        raise ValueError(f"{name} must end in EDGE-{MOTION_DIM}, got {array.shape}")
    return array


def _validate_tangent_np(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-1] != TANGENT_DIM:
        raise ValueError(
            f"tangent must have {TANGENT_DIM} channels, got {array.shape}"
        )
    return array


def _edge_rotations_np(value: np.ndarray) -> np.ndarray:
    shape = value.shape[:-1]
    rot6d = value[..., ROT6D_START:ROT6D_END].reshape(
        shape + (NUM_JOINTS, 6)
    )
    return rot6d_to_matrix_np(rot6d)


def _cap_vectors_np(value: np.ndarray, maximum: Optional[float]) -> np.ndarray:
    if maximum is None or float(maximum) <= 0.0:
        return value
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(maximum) / np.maximum(norm, _EPS))
    return value * scale


def product_log_np(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``Log_reference(target)`` in the canonical 75D tangent layout."""
    ref = _validate_edge_np(reference, "reference")
    dst = _validate_edge_np(target, "target")
    if ref.shape != dst.shape:
        raise ValueError(f"reference/target shapes differ: {ref.shape} vs {dst.shape}")
    root = dst[..., ROOT] - ref[..., ROOT]
    ref_r = _edge_rotations_np(ref)
    dst_r = _edge_rotations_np(dst)
    relative = np.swapaxes(ref_r, -1, -2) @ dst_r
    rotvec = so3_log_np(relative)
    return np.concatenate(
        [root, rotvec.reshape(ref.shape[:-1] + (NUM_JOINTS * 3,))],
        axis=-1,
    ).astype(np.float32)


def product_exp_np(reference: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Retract a 75D tangent at ``reference`` back to valid EDGE-151 motion."""
    ref = _validate_edge_np(reference, "reference")
    delta = _validate_tangent_np(tangent)
    if ref.shape[:-1] != delta.shape[:-1]:
        raise ValueError(
            f"reference/tangent prefixes differ: {ref.shape} vs {delta.shape}"
        )
    out = ref.copy()
    out[..., ROOT] = ref[..., ROOT] + delta[..., :3]
    ref_r = _edge_rotations_np(ref)
    rotvec = delta[..., 3:].reshape(ref.shape[:-1] + (NUM_JOINTS, 3))
    out_r = ref_r @ so3_exp_np(rotvec)
    out[..., ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(out_r).reshape(
        ref.shape[:-1] + (NUM_JOINTS * 6,)
    )
    return out.astype(np.float32)


def product_distance_np(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    root_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> np.ndarray:
    """Per-frame product distance with mean joint squared geodesic angle."""
    delta = product_log_np(reference, target)
    root_sq = np.sum(delta[..., :3] ** 2, axis=-1)
    joint = delta[..., 3:].reshape(delta.shape[:-1] + (NUM_JOINTS, 3))
    rot_sq = np.mean(np.sum(joint**2, axis=-1), axis=-1)
    return np.sqrt(
        np.maximum(
            float(root_weight) * root_sq + float(rotation_weight) * rot_sq,
            0.0,
        )
    ).astype(np.float32)


def parallel_transport_np(
    reference: np.ndarray,
    target: np.ndarray,
    tangent: np.ndarray,
) -> np.ndarray:
    """Transport a left-trivialised tangent along the shortest product geodesic.

    For each SO(3) factor this uses the midpoint adjoint transport
    ``Ad_Exp(-Log(R0^T R1)/2)``.  It is norm preserving and exact for the
    bi-invariant SO(3) metric used by this module.
    """
    ref = _validate_edge_np(reference, "reference")
    dst = _validate_edge_np(target, "target")
    vec = _validate_tangent_np(tangent)
    if ref.shape != dst.shape or ref.shape[:-1] != vec.shape[:-1]:
        raise ValueError("parallel_transport inputs must share their batch prefix")
    displacement = product_log_np(ref, dst)[..., 3:].reshape(
        ref.shape[:-1] + (NUM_JOINTS, 3)
    )
    joint_vec = vec[..., 3:].reshape(ref.shape[:-1] + (NUM_JOINTS, 3))
    transport_r = so3_exp_np(-0.5 * displacement)
    transported = np.einsum("...ij,...j->...i", transport_r, joint_vec)
    return np.concatenate(
        [vec[..., :3], transported.reshape(vec.shape[:-1] + (NUM_JOINTS * 3,))],
        axis=-1,
    ).astype(np.float32)


def _broadcast_mask_np(
    mask: Optional[np.ndarray],
    prefix: tuple[int, ...],
    trailing: int,
    name: str,
) -> np.ndarray:
    if mask is None:
        return np.ones(prefix + (trailing,), dtype=np.float32)
    value = np.asarray(mask, dtype=np.float32)
    if trailing == 1 and value.shape == prefix:
        value = value[..., None]
    if value.shape[-1:] != (trailing,):
        raise ValueError(f"{name} must end in {trailing}, got {value.shape}")
    try:
        return np.broadcast_to(value, prefix + (trailing,)).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to {prefix + (trailing,)}") from exc


def masked_retract_np(
    reference: np.ndarray,
    tangent: np.ndarray,
    *,
    joint_mask: Optional[np.ndarray] = None,
    root_mask: Optional[np.ndarray] = None,
    max_rotation_rad: Optional[float] = None,
    max_root_m: Optional[float] = None,
) -> np.ndarray:
    """Mask, norm-cap and retract a product tangent.

    ``joint_mask`` has shape ``[..., 24]`` and ``root_mask`` shape ``[...]`` or
    ``[..., 1]``.  Zero-mask factors remain bitwise equal to the reference.
    """
    ref = _validate_edge_np(reference, "reference")
    delta = _validate_tangent_np(tangent).copy()
    if ref.shape[:-1] != delta.shape[:-1]:
        raise ValueError("reference and tangent must share their batch prefix")
    prefix = ref.shape[:-1]
    rm = np.clip(_broadcast_mask_np(root_mask, prefix, 1, "root_mask"), 0.0, 1.0)
    jm = np.clip(
        _broadcast_mask_np(joint_mask, prefix, NUM_JOINTS, "joint_mask"),
        0.0,
        1.0,
    )
    root_delta = _cap_vectors_np(delta[..., :3], max_root_m) * rm
    joint_delta = delta[..., 3:].reshape(prefix + (NUM_JOINTS, 3))
    joint_delta = _cap_vectors_np(joint_delta, max_rotation_rad) * jm[..., None]
    masked = np.concatenate(
        [
            root_delta,
            joint_delta.reshape(prefix + (NUM_JOINTS * 3,)),
        ],
        axis=-1,
    )
    out = product_exp_np(ref, masked)
    out[..., ROOT] = np.where(rm > 0.0, out[..., ROOT], ref[..., ROOT])
    out_rot = out[..., ROT6D_START:ROT6D_END].reshape(
        prefix + (NUM_JOINTS, 6)
    )
    ref_rot = ref[..., ROT6D_START:ROT6D_END].reshape(
        prefix + (NUM_JOINTS, 6)
    )
    out[..., ROT6D_START:ROT6D_END] = np.where(
        jm[..., None] > 0.0, out_rot, ref_rot
    ).reshape(prefix + (NUM_JOINTS * 6,))
    return out.astype(np.float32)


def _smooth_time_np(value: np.ndarray) -> np.ndarray:
    if value.shape[0] < 3:
        return value.copy()
    out = value.copy()
    out[1:-1] = (
        0.25 * value[:-2] + 0.50 * value[1:-1] + 0.25 * value[2:]
    )
    return out


def _covariant_smoothness_np(motion: np.ndarray) -> float:
    if motion.shape[0] < 3:
        return 0.0
    velocity = product_log_np(motion[:-1], motion[1:])
    transported = parallel_transport_np(
        motion[:-2], motion[1:-1], velocity[:-1]
    )
    acceleration = velocity[1:] - transported
    root_energy = np.mean(np.sum(acceleration[..., :3] ** 2, axis=-1))
    joint = acceleration[..., 3:].reshape(
        acceleration.shape[:-1] + (NUM_JOINTS, 3)
    )
    joint_energy = np.mean(np.sum(joint**2, axis=-1))
    return float(root_energy + joint_energy)


def riemannian_trust_region_refine_np(
    reference: np.ndarray,
    proposal: np.ndarray,
    *,
    joint_mask: np.ndarray,
    root_mask: np.ndarray,
    contact_mask: Optional[np.ndarray] = None,
    steps: int = 5,
    initial_radius: float = 1.0,
    min_radius: float = 0.0625,
    max_rotation_rad: Optional[float] = 0.35,
    max_root_m: Optional[float] = 0.08,
    fidelity_weight: float = 0.35,
    smoothness_weight: float = 0.65,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair a proposal with accepted retractions on the masked product manifold.

    The feasible V45/V46 proposal is the initial point.  A three-tap tangent
    smoother supplies a local descent proposal; the trust radius expands after
    an accepted objective decrease and shrinks after rejection.  The returned
    objective is therefore never worse than the feasible initial proposal.
    """
    ref = _validate_edge_np(reference, "reference")
    raw_proposal = _validate_edge_np(proposal, "proposal")
    if ref.shape != raw_proposal.shape or ref.ndim != 2:
        raise ValueError("trust-region refinement expects equal [T,151] arrays")
    if len(ref) == 0:
        raise ValueError("trust-region refinement requires at least one frame")
    prefix = ref.shape[:-1]
    jm = np.clip(
        _broadcast_mask_np(joint_mask, prefix, NUM_JOINTS, "joint_mask"),
        0.0,
        1.0,
    )
    rm = np.clip(_broadcast_mask_np(root_mask, prefix, 1, "root_mask"), 0.0, 1.0)
    cm = np.clip(_broadcast_mask_np(contact_mask, prefix, 1, "contact_mask"), 0.0, 1.0)

    target = masked_retract_np(
        ref,
        product_log_np(ref, raw_proposal),
        joint_mask=jm,
        root_mask=rm,
        max_rotation_rad=max_rotation_rad,
        max_root_m=max_root_m,
    )
    target[..., CONTACT] = np.clip(
        ref[..., CONTACT] * (1.0 - cm)
        + raw_proposal[..., CONTACT] * cm,
        0.0,
        1.0,
    )

    def objective(value: np.ndarray) -> float:
        fidelity = float(np.mean(product_distance_np(value, target)))
        contact = float(np.mean((value[..., CONTACT] - target[..., CONTACT]) ** 2))
        smoothness = _covariant_smoothness_np(value)
        return (
            max(0.0, float(fidelity_weight)) * (fidelity + 0.25 * contact)
            + max(0.0, float(smoothness_weight)) * smoothness
        )

    current = target.copy()
    current_objective = objective(current)
    initial_objective = current_objective
    minimum_radius = float(np.clip(min_radius, 1.0e-6, 1.0))
    radius = float(
        np.clip(initial_radius, minimum_radius, 1.0)
    )
    accepted = 0
    rejected = 0
    history: list[dict[str, Any]] = []

    for iteration in range(max(0, int(steps))):
        state = product_log_np(ref, current)
        smoothed_state = _smooth_time_np(state)
        smoothed = masked_retract_np(
            ref,
            smoothed_state,
            joint_mask=(jm > 0.0).astype(np.float32),
            root_mask=(rm > 0.0).astype(np.float32),
            max_rotation_rad=max_rotation_rad,
            max_root_m=max_root_m,
        )
        smoothed_contacts = _smooth_time_np(current[..., CONTACT])
        smoothed[..., CONTACT] = np.where(
            cm > 0.0,
            smoothed_contacts,
            ref[..., CONTACT],
        )
        direction = product_log_np(current, smoothed)
        candidate = masked_retract_np(
            current,
            radius * direction,
            joint_mask=jm,
            root_mask=rm,
            max_rotation_rad=max_rotation_rad,
            max_root_m=max_root_m,
        )
        # Re-project relative to the immutable reference so repeated accepted
        # steps cannot accumulate beyond the geometric safety caps.
        candidate = masked_retract_np(
            ref,
            product_log_np(ref, candidate),
            joint_mask=(jm > 0.0).astype(np.float32),
            root_mask=(rm > 0.0).astype(np.float32),
            max_rotation_rad=max_rotation_rad,
            max_root_m=max_root_m,
        )
        candidate[..., CONTACT] = (
            current[..., CONTACT] * (1.0 - radius * cm)
            + smoothed[..., CONTACT] * (radius * cm)
        )
        candidate[..., CONTACT] = np.clip(candidate[..., CONTACT], 0.0, 1.0)
        candidate_objective = objective(candidate)
        improved = bool(
            np.isfinite(candidate_objective)
            and candidate_objective < current_objective - 1.0e-10
        )
        history.append(
            {
                "iteration": int(iteration),
                "radius": float(radius),
                "objective_before": float(current_objective),
                "objective_candidate": float(candidate_objective),
                "accepted": improved,
            }
        )
        if improved:
            current = candidate
            current_objective = candidate_objective
            accepted += 1
            radius = min(1.0, radius * 1.5)
        else:
            rejected += 1
            radius *= 0.5
            if radius < minimum_radius:
                break

    report = {
        "algorithm": "masked_product_manifold_adaptive_trust_region",
        "initial_point": "feasible_v45_v46_proposal",
        "initial_objective": float(initial_objective),
        "final_objective": float(current_objective),
        "objective_nonincreasing": bool(
            current_objective <= initial_objective + 1.0e-10
        ),
        "accepted_steps": int(accepted),
        "rejected_steps": int(rejected),
        "final_radius": float(radius),
        "iterations": history,
    }
    return current.astype(np.float32), report


def _torch_required() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for product-manifold torch operations")


def _validate_edge_torch(value: "torch.Tensor", name: str) -> "torch.Tensor":
    _torch_required()
    if value.shape[-1] != MOTION_DIM:
        raise ValueError(f"{name} must end in EDGE-{MOTION_DIM}, got {tuple(value.shape)}")
    return value


def _rot6d_to_matrix_torch(value: "torch.Tensor") -> "torch.Tensor":
    a1, a2 = value[..., :3], value[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=_EPS)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_orthogonal, dim=-1, eps=_EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _matrix_to_rot6d_torch(value: "torch.Tensor") -> "torch.Tensor":
    return torch.cat([value[..., :, 0], value[..., :, 1]], dim=-1)


def _so3_log_torch(value: "torch.Tensor") -> "torch.Tensor":
    # atan2 formulation is stable near the identity; the pi branch uses the
    # diagonal of (R + I) / 2 and keeps gradients finite for training inputs.
    vee = torch.stack(
        [
            value[..., 2, 1] - value[..., 1, 2],
            value[..., 0, 2] - value[..., 2, 0],
            value[..., 1, 0] - value[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    cos_theta = ((torch.diagonal(value, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(sin_theta, cos_theta)
    scale = theta / (2.0 * torch.sin(theta)).clamp_min(1.0e-7)
    regular = vee * scale[..., None]
    small = 0.5 * vee
    symmetric = 0.5 * (
        value + torch.eye(3, dtype=value.dtype, device=value.device)
    )
    diagonal = torch.diagonal(symmetric, dim1=-2, dim2=-1).clamp_min(0.0)
    axis = torch.sqrt(diagonal + _EPS)
    sign_hint = torch.sign(vee)
    sign_hint = torch.where(sign_hint == 0, torch.ones_like(sign_hint), sign_hint)
    axis = axis * sign_hint
    axis = torch.nn.functional.normalize(axis, dim=-1, eps=_EPS)
    near_pi = axis * theta[..., None]
    result = torch.where((theta < 1.0e-5)[..., None], small, regular)
    return torch.where((torch.pi - theta < 1.0e-4)[..., None], near_pi, result)


def _skew_torch(value: "torch.Tensor") -> "torch.Tensor":
    zero = torch.zeros_like(value[..., 0])
    x, y, z = value.unbind(dim=-1)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(value.shape[:-1] + (3, 3))


def _so3_exp_torch(value: "torch.Tensor") -> "torch.Tensor":
    theta_sq = (value * value).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_sq.clamp_min(_EPS))
    a = torch.where(
        theta_sq > 1.0e-8,
        torch.sin(theta) / theta,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
    )
    b = torch.where(
        theta_sq > 1.0e-8,
        (1.0 - torch.cos(theta)) / theta_sq.clamp_min(_EPS),
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
    )
    skew = _skew_torch(value)
    identity = torch.eye(3, dtype=value.dtype, device=value.device)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def product_log_torch(
    reference: "torch.Tensor", target: "torch.Tensor"
) -> "torch.Tensor":
    ref = _validate_edge_torch(reference, "reference")
    dst = _validate_edge_torch(target, "target")
    if ref.shape != dst.shape:
        raise ValueError("reference and target must have equal shapes")
    prefix = ref.shape[:-1]
    ref_r = _rot6d_to_matrix_torch(
        ref[..., ROT6D_START:ROT6D_END].reshape(prefix + (NUM_JOINTS, 6))
    )
    dst_r = _rot6d_to_matrix_torch(
        dst[..., ROT6D_START:ROT6D_END].reshape(prefix + (NUM_JOINTS, 6))
    )
    rotvec = _so3_log_torch(ref_r.transpose(-1, -2) @ dst_r)
    return torch.cat(
        [
            dst[..., ROOT] - ref[..., ROOT],
            rotvec.reshape(prefix + (NUM_JOINTS * 3,)),
        ],
        dim=-1,
    )


def product_exp_torch(
    reference: "torch.Tensor", tangent: "torch.Tensor"
) -> "torch.Tensor":
    ref = _validate_edge_torch(reference, "reference")
    if tangent.shape[-1] != TANGENT_DIM or ref.shape[:-1] != tangent.shape[:-1]:
        raise ValueError("tangent must be prefix-compatible and end in 75")
    prefix = ref.shape[:-1]
    ref_r = _rot6d_to_matrix_torch(
        ref[..., ROT6D_START:ROT6D_END].reshape(prefix + (NUM_JOINTS, 6))
    )
    rotvec = tangent[..., 3:].reshape(prefix + (NUM_JOINTS, 3))
    out_r = ref_r @ _so3_exp_torch(rotvec)
    return torch.cat(
        [
            ref[..., CONTACT],
            ref[..., ROOT] + tangent[..., :3],
            _matrix_to_rot6d_torch(out_r).reshape(
                prefix + (NUM_JOINTS * 6,)
            ),
        ],
        dim=-1,
    )


def product_distance_torch(
    reference: "torch.Tensor",
    target: "torch.Tensor",
    *,
    root_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> "torch.Tensor":
    delta = product_log_torch(reference, target)
    root_sq = (delta[..., :3] ** 2).sum(dim=-1)
    joint = delta[..., 3:].reshape(delta.shape[:-1] + (NUM_JOINTS, 3))
    rot_sq = (joint**2).sum(dim=-1).mean(dim=-1)
    return torch.sqrt(
        (
            float(root_weight) * root_sq + float(rotation_weight) * rot_sq
        ).clamp_min(0.0)
    )


def parallel_transport_torch(
    reference: "torch.Tensor",
    target: "torch.Tensor",
    tangent: "torch.Tensor",
) -> "torch.Tensor":
    displacement = product_log_torch(reference, target)
    prefix = tangent.shape[:-1]
    if tangent.shape[-1] != TANGENT_DIM or reference.shape[:-1] != prefix:
        raise ValueError("parallel_transport inputs must share their batch prefix")
    rotation = _so3_exp_torch(
        -0.5
        * displacement[..., 3:].reshape(prefix + (NUM_JOINTS, 3))
    )
    joint = tangent[..., 3:].reshape(prefix + (NUM_JOINTS, 3))
    transported = (rotation @ joint[..., None]).squeeze(-1)
    return torch.cat(
        [tangent[..., :3], transported.reshape(prefix + (NUM_JOINTS * 3,))],
        dim=-1,
    )


def _cap_vectors_torch(
    value: "torch.Tensor", maximum: Optional[float]
) -> "torch.Tensor":
    if maximum is None or float(maximum) <= 0.0:
        return value
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * (float(maximum) / norm.clamp_min(_EPS)).clamp(max=1.0)


def masked_retract_torch(
    reference: "torch.Tensor",
    tangent: "torch.Tensor",
    *,
    joint_mask: Optional["torch.Tensor"] = None,
    root_mask: Optional["torch.Tensor"] = None,
    max_rotation_rad: Optional[float] = None,
    max_root_m: Optional[float] = None,
) -> "torch.Tensor":
    ref = _validate_edge_torch(reference, "reference")
    prefix = ref.shape[:-1]
    if tangent.shape != prefix + (TANGENT_DIM,):
        raise ValueError("tangent must be prefix-compatible and end in 75")
    if root_mask is None:
        root = torch.ones(prefix + (1,), dtype=ref.dtype, device=ref.device)
    else:
        root = root_mask.to(dtype=ref.dtype, device=ref.device)
        if root.shape == prefix:
            root = root[..., None]
        root = torch.broadcast_to(root, prefix + (1,)).clamp(0.0, 1.0)
    if joint_mask is None:
        joint = torch.ones(
            prefix + (NUM_JOINTS,), dtype=ref.dtype, device=ref.device
        )
    else:
        joint = torch.broadcast_to(
            joint_mask.to(dtype=ref.dtype, device=ref.device),
            prefix + (NUM_JOINTS,),
        ).clamp(0.0, 1.0)
    root_delta = _cap_vectors_torch(tangent[..., :3], max_root_m) * root
    joint_delta = tangent[..., 3:].reshape(prefix + (NUM_JOINTS, 3))
    joint_delta = (
        _cap_vectors_torch(joint_delta, max_rotation_rad) * joint[..., None]
    )
    out = product_exp_torch(
        ref,
        torch.cat(
            [
                root_delta,
                joint_delta.reshape(prefix + (NUM_JOINTS * 3,)),
            ],
            dim=-1,
        ),
    )
    root_value = torch.where(root > 0.0, out[..., ROOT], ref[..., ROOT])
    out_rot = out[..., ROT6D_START:ROT6D_END].reshape(
        prefix + (NUM_JOINTS, 6)
    )
    ref_rot = ref[..., ROT6D_START:ROT6D_END].reshape(
        prefix + (NUM_JOINTS, 6)
    )
    rotation_value = torch.where(
        joint[..., None] > 0.0, out_rot, ref_rot
    ).reshape(prefix + (NUM_JOINTS * 6,))
    return torch.cat(
        [out[..., CONTACT], root_value, rotation_value], dim=-1
    )


def product_log(reference: Any, target: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return product_log_torch(reference, target)
    return product_log_np(reference, target)


def product_exp(reference: Any, tangent: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return product_exp_torch(reference, tangent)
    return product_exp_np(reference, tangent)


def product_distance(reference: Any, target: Any, **kwargs: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return product_distance_torch(reference, target, **kwargs)
    return product_distance_np(reference, target, **kwargs)


def parallel_transport(reference: Any, target: Any, tangent: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return parallel_transport_torch(reference, target, tangent)
    return parallel_transport_np(reference, target, tangent)


def masked_retract(reference: Any, tangent: Any, **kwargs: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return masked_retract_torch(reference, tangent, **kwargs)
    return masked_retract_np(reference, tangent, **kwargs)


__all__ = [
    "PRODUCT_STATE_DIM",
    "TANGENT_DIM",
    "masked_retract",
    "masked_retract_np",
    "masked_retract_torch",
    "parallel_transport",
    "parallel_transport_np",
    "parallel_transport_torch",
    "product_distance",
    "product_distance_np",
    "product_distance_torch",
    "product_exp",
    "product_exp_np",
    "product_exp_torch",
    "product_log",
    "product_log_np",
    "product_log_torch",
    "riemannian_trust_region_refine_np",
]
