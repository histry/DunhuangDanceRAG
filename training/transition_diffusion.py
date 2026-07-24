#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatible V32 continuous contact-aware transition sampler.

Schedulers keep importing this historical module. V32 accepts only the new
continuous contact-INR checkpoint for the main model. It samples multiple
latent candidates, evaluates them with real neighbouring context and falls
back to the deterministic C2 SO(3) path when learned generation is unsafe.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from training.boundary_dynamics import (
    boundary_state_from_context_np,
    boundary_state_to_torch,
)

from support.motion_geometry import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROOT_X,
    ROOT_Z,
    ROT,
)
from motion_geometry.rotations import (
    ROT6D_LAYOUT_COLUMN,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
)
from support.contact_inr import (
    V32ContactINRSystem,
    config_from_dict,
    linear_beta_schedule,
    make_c2_transition_np,
    selected_timesteps,
)
from support.transition_quality import (
    accept_candidate,
    transition_risk,
)
from support.checkpoint_contracts import assert_checkpoint_fps


def load_transition_diffusion(
    path: str | Path,
    device: torch.device | str = "cpu",
    fps: float | None = None,
) -> Dict[str, Any] | None:
    if not path:
        return None
    checkpoint_path = Path(str(path))
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Transition checkpoint not found: {checkpoint_path}")
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if fps is not None:
        assert_checkpoint_fps(
            checkpoint,
            role="Transition diffusion",
            runtime_fps=float(fps),
            path=str(checkpoint_path),
        )
    config_values = dict(checkpoint.get("config", {}))
    architecture = str(config_values.get("architecture", ""))
    supported = {
        "v32_continuous_c2_contact_inr_latent_diffusion",
        "v34_continuous_c3_contact_inr_latent_diffusion",
    }
    if architecture not in supported:
        raise RuntimeError(
            f"Checkpoint architecture={architecture!r} is not V32/V34. "
            "Retrain with the supplied train_v27_transition_diffusion.py."
        )
    model_config = config_from_dict(
        dict(config_values.get("model", config_values))
    )
    system = V32ContactINRSystem(model_config).to(device)
    state = checkpoint.get("system", checkpoint.get("model"))
    if state is None:
        raise RuntimeError("V32 checkpoint has no system/model state")
    system.load_state_dict(state)
    ema = checkpoint.get("ema_diffusion")
    if ema is not None:
        system.diffusion.load_state_dict(ema)
    system.eval()
    latent_mean = torch.as_tensor(
        checkpoint.get(
            "latent_mean",
            np.zeros((model_config.latent_dim,), np.float32),
        ),
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    latent_std = torch.as_tensor(
        checkpoint.get(
            "latent_std",
            np.ones((model_config.latent_dim,), np.float32),
        ),
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1).clamp_min(1e-4)
    return {
        "architecture": architecture,
        "system": system,
        "config": config_values,
        "path": str(checkpoint_path),
        "device": device,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "best_val_loss": checkpoint.get("best_val_loss"),
        "epoch": checkpoint.get("epoch"),
    }


def _geodesic_blend(
    base: np.ndarray,
    generated: np.ndarray,
    amount: float,
) -> np.ndarray:
    a = np.asarray(base, np.float32)
    b = np.asarray(generated, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    value = float(np.clip(amount, 0.0, 1.0))
    if value <= 0.0:
        return a.copy()
    if value >= 1.0:
        return _project_native_motion_np(b)
    with torch.no_grad():
        ra = rotation_6d_to_matrix(
            torch.from_numpy(a[:, ROT]).reshape(
                len(a), NUM_JOINTS, 6
            )
        )
        rb = rotation_6d_to_matrix(
            torch.from_numpy(b[:, ROT]).reshape(
                len(b), NUM_JOINTS, 6
            )
        )
        tangent = matrix_to_axis_angle(
            torch.matmul(ra.transpose(-1, -2), rb)
        )
        rotation = torch.matmul(
            ra, axis_angle_to_matrix(value * tangent)
        )
        rot6d = matrix_to_rotation_6d(rotation).reshape(
            len(a), -1
        ).cpu().numpy()
    out = a.copy()
    out[:, CONTACT] = (
        (1.0 - value) * a[:, CONTACT]
        + value * b[:, CONTACT]
    )
    out[:, ROOT] = (
        (1.0 - value) * a[:, ROOT]
        + value * b[:, ROOT]
    )
    out[:, ROT] = rot6d
    # Root XZ is a physical trajectory, not a local pose channel.  The
    # scheduler supplies a velocity-aware canonical bridge and restores it
    # after this optional learned proposal.  Keeping the blended trajectory
    # here is still important because candidate risk is evaluated before that
    # restoration; zeroing XZ would make that audit measure a different path.
    return _project_native_motion_np(out)


def _project_native_motion_np(motion: np.ndarray) -> np.ndarray:
    """Project the checkpoint-native PyTorch3D-row Rot6D representation."""

    x = np.asarray(motion, dtype=np.float32)
    with torch.no_grad():
        rotations = torch.from_numpy(x[:, ROT]).reshape(
            len(x),
            NUM_JOINTS,
            6,
        )
        matrices = rotation_6d_to_matrix(rotations)
        rot6d = matrix_to_rotation_6d(matrices).reshape(len(x), -1)
    out = x.copy()
    out[:, ROT] = rot6d.cpu().numpy().astype(np.float32)
    return out


def _native_transition_risk(
    previous: np.ndarray,
    transition: np.ndarray,
    following: np.ndarray,
    *,
    fps: float,
) -> Dict[str, float]:
    """Audit row-native model values under the canonical geometry contract."""

    def canonical(value: np.ndarray) -> np.ndarray:
        return convert_motion_rot6d_layout_np(
            value,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
            ROT6D_LAYOUT_COLUMN,
        )

    return transition_risk(
        canonical(previous),
        canonical(transition),
        canonical(following),
        fps=fps,
    )


def _seed(
    start: np.ndarray,
    end: np.ndarray,
    length: int,
    candidate: int,
) -> int:
    base = int(os.getenv("V32_TRANSITION_SEED", "20260610"))
    signature = int(np.round(
        np.sum(np.abs(start[:48])) * 1009.0
        + np.sum(np.abs(end[:48])) * 1709.0
    ))
    return int(
        (base + length * 65537 + candidate * 104729 + signature)
        % (2**31 - 1)
    )


def _sample_latent(
    bundle: Dict[str, Any],
    condition: torch.Tensor,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    system: V32ContactINRSystem = bundle["system"]
    diffusion_steps = int(
        bundle["config"].get(
            "diffusion_steps", system.config.diffusion_steps
        )
    )
    device = condition.device
    _, _, alpha_bar = linear_beta_schedule(
        diffusion_steps, device
    )
    indices = selected_timesteps(
        diffusion_steps, int(steps), device
    )
    latent = torch.randn(
        (1, system.config.latent_dim),
        device=device,
        generator=generator,
    )
    guidance = float(os.getenv("V32_GUIDANCE", "1.0"))
    with torch.no_grad():
        for position, index in enumerate(indices):
            time = torch.full(
                (1,),
                float(index.item()) / max(diffusion_steps - 1, 1),
                device=device,
            )
            conditional = system.diffusion(
                latent, time, condition
            )
            if abs(guidance - 1.0) > 1e-6:
                unconditional = system.diffusion(
                    latent, time, torch.zeros_like(condition)
                )
                epsilon = unconditional + guidance * (
                    conditional - unconditional
                )
            else:
                epsilon = conditional
            ab = alpha_bar[index]
            x0 = (
                latent - torch.sqrt(1.0 - ab) * epsilon
            ) / torch.sqrt(ab).clamp_min(1e-6)
            x0 = x0.clamp(-5.0, 5.0)
            if position + 1 < len(indices):
                previous = indices[position + 1]
                ab_previous = alpha_bar[previous]
                latent = (
                    torch.sqrt(ab_previous) * x0
                    + torch.sqrt(1.0 - ab_previous) * epsilon
                )
            else:
                latent = x0
    return latent


def sample_transition_diffusion(
    bundle: Dict[str, Any] | None,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    blend: float = 0.35,
    steps: int = 36,
    previous_context: np.ndarray | None = None,
    next_context: np.ndarray | None = None,
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    k = int(length)
    if k <= 0:
        return np.zeros((0, MOTION_DIM), np.float32), {
            "enabled": False, "reason": "zero_length"
        }

    start_np = np.asarray(start_frame, np.float32).reshape(-1)
    end_np = np.asarray(end_frame, np.float32).reshape(-1)
    previous = (
        np.asarray(previous_context, np.float32)
        if previous_context is not None
        else start_np[None]
    )
    following = (
        np.asarray(next_context, np.float32)
        if next_context is not None
        else end_np[None]
    )
    baseline = make_c2_transition_np(previous, following, k)
    if bundle is None:
        return baseline, {
            "enabled": False,
            "reason": "no_checkpoint",
            "architecture": "v32_c2_baseline",
        }

    system: V32ContactINRSystem = bundle["system"]
    device = torch.device(device)
    start_velocity_np = (
        previous[-1] - previous[-2]
        if len(previous) >= 2
        else np.zeros_like(start_np)
    )
    end_velocity_np = (
        following[1] - following[0]
        if len(following) >= 2
        else np.zeros_like(end_np)
    )
    start = torch.from_numpy(start_np).to(device).reshape(1, -1)
    end = torch.from_numpy(end_np).to(device).reshape(1, -1)
    start_velocity = torch.from_numpy(
        start_velocity_np
    ).to(device).reshape(1, -1)
    end_velocity = torch.from_numpy(
        end_velocity_np
    ).to(device).reshape(1, -1)
    music = torch.from_numpy(
        np.asarray(music_query, np.float32).reshape(1, -1)
    ).to(device)
    length_frames = torch.tensor(
        [[float(k)]], device=device
    )
    condition = system.condition(
        start, end, start_velocity, end_velocity,
        music, length_frames,
    )
    boundary_state = boundary_state_to_torch(
        boundary_state_from_context_np(previous, following),
        device=device,
        dtype=start.dtype,
    )
    coordinate = torch.linspace(
        1.0 / (k + 1), k / (k + 1), k,
        device=device,
    ).reshape(1, k, 1)

    baseline_risk = _native_transition_risk(
        previous, baseline, following, fps=fps
    )
    baseline_absolute_safe, baseline_gate = accept_candidate(
        baseline_risk,
        baseline_risk,
        max_total_ratio=1.0,
        max_entry_ratio=1.0,
        max_exit_ratio=1.0,
        max_jerk_ratio=1.0,
        max_foot_ratio=1.0,
        max_penetration_ratio=1.0,
        max_rotation_step_rad=float(
            os.getenv("V32_MAX_ROTATION_STEP_RAD", "0.20")
        ),
        max_boundary_jerk_abs=float(
            os.getenv("V34_MAX_BOUNDARY_JERK", "5000")
        ),
        max_boundary_angular_jerk_abs=float(
            os.getenv("V34_MAX_BOUNDARY_ANGULAR_JERK", "5000")
        ),
        max_entry_rotation_step_rad=float(
            os.getenv("V34_MAX_ENTRY_ROTATION_STEP_RAD", "0.16")
        ),
        max_exit_rotation_step_rad=float(
            os.getenv("V34_MAX_EXIT_ROTATION_STEP_RAD", "0.12")
        ),
        max_entry_fk_jump=float(
            os.getenv("V34_MAX_ENTRY_FK_JUMP", "0.060")
        ),
        max_exit_fk_jump=float(
            os.getenv("V34_MAX_EXIT_FK_JUMP", "0.040")
        ),
        max_exit_acceleration=float(
            os.getenv("V34_MAX_EXIT_ACCELERATION", "12.0")
        ),
    )
    candidate_count = max(
        1, int(os.getenv("V32_CANDIDATES", "8"))
    )
    trust = float(np.clip(
        float(os.getenv("V32_INR_TRUST", str(blend))),
        0.0, 0.65,
    ))
    candidates: List[Dict[str, Any]] = []
    accepted: List[Tuple[float, np.ndarray, Dict[str, Any], torch.Tensor]] = []

    for candidate_index in range(candidate_count):
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _seed(start_np, end_np, k, candidate_index)
        )
        normalised = _sample_latent(
            bundle, condition, steps, generator
        )
        latent = (
            normalised * bundle["latent_std"]
            + bundle["latent_mean"]
        )
        with torch.no_grad():
            generated = system.decode(
                latent,
                start,
                end,
                start_velocity,
                end_velocity,
                condition,
                coordinate,
                length_frames,
                boundary_state=boundary_state,
            )[0].cpu().numpy().astype(np.float32)
        candidate = _geodesic_blend(
            baseline, generated, trust
        )
        risk = _native_transition_risk(
            previous, candidate, following, fps=fps
        )
        safe, gate = accept_candidate(
            baseline_risk,
            risk,
            max_total_ratio=float(
                os.getenv("V32_MAX_TOTAL_RISK_RATIO", "1.02")
            ),
            max_entry_ratio=float(
                os.getenv("V32_MAX_ENTRY_RATIO", "1.05")
            ),
            max_exit_ratio=float(
                os.getenv("V32_MAX_EXIT_RATIO", "1.03")
            ),
            max_jerk_ratio=float(
                os.getenv("V32_MAX_JERK_RATIO", "1.03")
            ),
            max_foot_ratio=float(
                os.getenv("V32_MAX_FOOT_RATIO", "1.02")
            ),
            max_penetration_ratio=float(
                os.getenv("V32_MAX_PENETRATION_RATIO", "1.02")
            ),
            max_rotation_step_rad=float(
                os.getenv("V32_MAX_ROTATION_STEP_RAD", "0.20")
            ),
            max_boundary_jerk_abs=float(
                os.getenv("V34_MAX_BOUNDARY_JERK", "5000")
            ),
            max_boundary_angular_jerk_abs=float(
                os.getenv("V34_MAX_BOUNDARY_ANGULAR_JERK", "5000")
            ),
            max_entry_rotation_step_rad=float(
                os.getenv("V34_MAX_ENTRY_ROTATION_STEP_RAD", "0.16")
            ),
            max_exit_rotation_step_rad=float(
                os.getenv("V34_MAX_EXIT_ROTATION_STEP_RAD", "0.12")
            ),
            max_entry_fk_jump=float(
                os.getenv("V34_MAX_ENTRY_FK_JUMP", "0.060")
            ),
            max_exit_fk_jump=float(
                os.getenv("V34_MAX_EXIT_FK_JUMP", "0.040")
            ),
            max_exit_acceleration=float(
                os.getenv("V34_MAX_EXIT_ACCELERATION", "12.0")
            ),
        )
        row = {
            "index": candidate_index,
            "risk": risk,
            "gate": gate,
        }
        candidates.append(row)
        if safe:
            accepted.append((
                risk["total"],
                candidate,
                row,
                latent.detach().clone(),
            ))

    force_model = os.getenv(
        "V32_FORCE_MODEL", "0"
    ).lower() in {"1", "true", "yes", "on"}
    latent_blend_meta: Dict[str, Any] = {
        "enabled": os.getenv(
            "V34_LATENT_SNIPPET_BLEND", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "applied": False,
    }
    if accepted:
        accepted.sort(key=lambda x: x[0])
        result = accepted[0][1]
        selected_index = int(accepted[0][2]["index"])
        fallback = False
        if latent_blend_meta["enabled"] and len(accepted) >= 2:
            top_count = min(
                len(accepted),
                max(2, int(os.getenv("V34_LATENT_BLEND_TOP_K", "3"))),
            )
            temperature = max(
                1e-4, float(os.getenv("V34_LATENT_BLEND_TEMPERATURE", "0.08"))
            )
            top = accepted[:top_count]
            risk_values = torch.as_tensor(
                [float(row[0]) for row in top],
                device=device,
                dtype=torch.float32,
            )
            weights = torch.softmax(
                -(risk_values - risk_values.min()) / temperature,
                dim=0,
            )
            blended_latent = torch.zeros_like(top[0][3])
            for weight, row in zip(weights, top):
                blended_latent = blended_latent + weight.reshape(1, 1) * row[3]
            with torch.no_grad():
                generated = system.decode(
                    blended_latent,
                    start,
                    end,
                    start_velocity,
                    end_velocity,
                    condition,
                    coordinate,
                    length_frames,
                    boundary_state=boundary_state,
                )[0].cpu().numpy().astype(np.float32)
            blended_candidate = _geodesic_blend(
                baseline, generated, trust
            )
            blended_risk = _native_transition_risk(
                previous, blended_candidate, following, fps=fps
            )
            blended_safe, blended_gate = accept_candidate(
                baseline_risk,
                blended_risk,
                max_total_ratio=float(
                    os.getenv("V32_MAX_TOTAL_RISK_RATIO", "1.02")
                ),
                max_entry_ratio=float(
                    os.getenv("V32_MAX_ENTRY_RATIO", "1.05")
                ),
                max_exit_ratio=float(
                    os.getenv("V32_MAX_EXIT_RATIO", "1.03")
                ),
                max_jerk_ratio=float(
                    os.getenv("V32_MAX_JERK_RATIO", "1.03")
                ),
                max_foot_ratio=float(
                    os.getenv("V32_MAX_FOOT_RATIO", "1.02")
                ),
                max_penetration_ratio=float(
                    os.getenv("V32_MAX_PENETRATION_RATIO", "1.02")
                ),
                max_rotation_step_rad=float(
                    os.getenv("V32_MAX_ROTATION_STEP_RAD", "0.20")
                ),
                max_boundary_jerk_abs=float(
                    os.getenv("V34_MAX_BOUNDARY_JERK", "5000")
                ),
                max_boundary_angular_jerk_abs=float(
                    os.getenv("V34_MAX_BOUNDARY_ANGULAR_JERK", "5000")
                ),
                max_entry_rotation_step_rad=float(
                    os.getenv("V34_MAX_ENTRY_ROTATION_STEP_RAD", "0.16")
                ),
                max_exit_rotation_step_rad=float(
                    os.getenv("V34_MAX_EXIT_ROTATION_STEP_RAD", "0.12")
                ),
                max_entry_fk_jump=float(
                    os.getenv("V34_MAX_ENTRY_FK_JUMP", "0.060")
                ),
                max_exit_fk_jump=float(
                    os.getenv("V34_MAX_EXIT_FK_JUMP", "0.040")
                ),
                max_exit_acceleration=float(
                    os.getenv("V34_MAX_EXIT_ACCELERATION", "12.0")
                ),
            )
            keep_ratio = float(os.getenv("V34_LATENT_BLEND_KEEP_RATIO", "1.01"))
            use_blend = bool(
                blended_safe
                and float(blended_risk["total"]) <= float(accepted[0][0]) * keep_ratio
            )
            latent_blend_meta.update({
                "top_k": int(top_count),
                "temperature": float(temperature),
                "weights": [float(x) for x in weights.detach().cpu()],
                "candidate_risks": [float(row[0]) for row in top],
                "blended_risk": blended_risk,
                "blended_gate": blended_gate,
                "blended_safe": bool(blended_safe),
                "keep_ratio": float(keep_ratio),
                "selected": bool(use_blend),
            })
            if use_blend:
                result = blended_candidate
                selected_index = -2
                latent_blend_meta["applied"] = True
    elif force_model and candidates:
        best_index = int(np.argmin([
            row["risk"]["total"] for row in candidates
        ]))
        # Re-sample selected deterministic seed to avoid storing all arrays.
        generator = torch.Generator(device=device)
        generator.manual_seed(
            _seed(start_np, end_np, k, best_index)
        )
        normalised = _sample_latent(
            bundle, condition, steps, generator
        )
        latent = (
            normalised * bundle["latent_std"]
            + bundle["latent_mean"]
        )
        with torch.no_grad():
            generated = system.decode(
                latent, start, end,
                start_velocity, end_velocity,
                condition, coordinate, length_frames,
                boundary_state=boundary_state,
            )[0].cpu().numpy().astype(np.float32)
        result = _geodesic_blend(
            baseline, generated, trust
        )
        selected_index = best_index
        fallback = False
    else:
        result = baseline
        selected_index = -1
        fallback = True

    unsafe_fallback = bool(fallback and not baseline_absolute_safe)
    if unsafe_fallback and os.getenv(
        "V34_FAIL_ON_UNSAFE_BOUNDARY", "0"
    ).lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "V34 absolute boundary gate rejected every learned candidate and "
            f"the septic baseline is unsafe: {baseline_gate}"
        )

    if os.getenv(
        "V32_HARD_CONTACT_OUTPUT", "0"
    ).lower() in {"1", "true", "yes", "on"}:
        result[:, CONTACT] = (
            result[:, CONTACT] >= 0.5
        ).astype(np.float32)

    return result.astype(np.float32), {
        "enabled": True,
        "architecture":
            "v34_continuous_c3_contact_inr_latent_diffusion",
        "checkpoint": str(bundle.get("path", "")),
        "continuous_time": True,
        "fps": fps,
        "contact_aware": True,
        "candidate_count": candidate_count,
        "accepted_count": len(accepted),
        "selected_index": selected_index,
        "fallback_to_c2_baseline": fallback,
        "inr_trust": trust,
        "guidance": float(os.getenv("V32_GUIDANCE", "1.0")),
        "baseline_risk": baseline_risk,
        "baseline_gate": baseline_gate,
        "baseline_absolute_safe": bool(baseline_absolute_safe),
        "unsafe_fallback": unsafe_fallback,
        "latent_snippet_blend": latent_blend_meta,
        "candidate_audit": candidates,
    }
