#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDGE V32: continuous C3-safe SO(3) INR with differentiable contacts.

The historical V30 idea (latent diffusion over a continuous motion INR) is kept,
but the high-frequency SIREN decoder is removed. V32 uses:
  * a regularised septic SO(3) base matching pose/velocity/acceleration/jerk;
  * low-band Fourier coordinates and SiLU residual blocks;
  * a C3-zero envelope 256*t^4*(1-t)^4 for every learned residual;
  * deterministic transition latents (no VAE posterior mismatch);
  * a separate contact-logit head for differentiable contact supervision;
  * arbitrary-length decoding at continuous t in [0,1].

Native EDGE motion layout:
  [0:4] contacts, [4:7] root xyz, [7:151] 24 x 6D local rotations.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from training.boundary_dynamics import (
    boundary_state_from_training_batch,
    make_v34_transition_np,
    septic_so3_root_base,
)

from support.motion_geometry import (
    CONTACT,
    MOTION_DIM,
    NUM_JOINTS,
    ROOT,
    ROOT_X,
    ROOT_Y,
    ROOT_Z,
    ROT,
    project_motion_rotations_torch,
)


@dataclass
class V32INRConfig:
    motion_dim: int = MOTION_DIM
    music_dim: int = 12
    latent_dim: int = 128
    condition_dim: int = 256
    encoder_hidden: int = 320
    inr_hidden: int = 320
    inr_layers: int = 6
    fourier_bands: int = 5
    diffusion_hidden: int = 512
    diffusion_blocks: int = 6
    diffusion_steps: int = 100
    residual_rotation_scale: float = 0.16
    residual_root_y_scale: float = 0.045
    contact_logit_scale: float = 3.0
    max_len: int = 120
    dropout: float = 0.08


def config_from_dict(values: Dict[str, object]) -> V32INRConfig:
    allowed = set(V32INRConfig.__dataclass_fields__)
    return V32INRConfig(**{k: values[k] for k in values if k in allowed})


def config_to_dict(config: V32INRConfig) -> Dict[str, object]:
    return asdict(config)


def _norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    return torch.linalg.norm(x, dim=dim, keepdim=keepdim)


def _limit_norm(vector: torch.Tensor, maximum: torch.Tensor | float) -> torch.Tensor:
    norm = _norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)
    maximum = torch.as_tensor(maximum, dtype=vector.dtype, device=vector.device)
    return vector * torch.minimum(torch.ones_like(norm), maximum / norm)


def _rotation(frame: torch.Tensor) -> torch.Tensor:
    return rotation_6d_to_matrix(
        frame[..., ROT].reshape(*frame.shape[:-1], NUM_JOINTS, 6)
    )


def quintic_smootherstep(t: torch.Tensor) -> torch.Tensor:
    x = t.clamp(0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def c3_zero_envelope(t: torch.Tensor) -> torch.Tensor:
    """Residual envelope with zero value and first three derivatives at ends."""
    x = t.clamp(0.0, 1.0)
    return 256.0 * x**4 * (1.0 - x) ** 4


def c2_zero_envelope(t: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias; V34 now preserves the septic C3 boundary."""
    return c3_zero_envelope(t)


def _safe_contact_logit(contact: torch.Tensor) -> torch.Tensor:
    return torch.logit(contact.clamp(1e-4, 1.0 - 1e-4))


def c2_quintic_so3_base(
    start: torch.Tensor,
    end: torch.Tensor,
    start_velocity: torch.Tensor,
    end_velocity: torch.Tensor,
    t: torch.Tensor,
    length_frames: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base contact logits, root and local rotations.

    The tangent polynomial satisfies endpoint pose and first derivative while
    imposing zero second derivative at both endpoints. Tangents are capped to
    prevent overshoot on weak/synthetic pairs.
    """
    if t.shape[-1] != 1:
        t = t[..., :1]
    batch, count, _ = t.shape
    dtype = start.dtype

    r0 = _rotation(start)
    r1 = _rotation(end)
    start_next = project_motion_rotations_torch(start + start_velocity)
    end_prev = project_motion_rotations_torch(end - end_velocity)
    rs = _rotation(start_next)
    re = _rotation(end_prev)

    delta = matrix_to_axis_angle(torch.matmul(r0.transpose(-1, -2), r1))
    omega0 = matrix_to_axis_angle(torch.matmul(r0.transpose(-1, -2), rs))
    omega1 = matrix_to_axis_angle(torch.matmul(re.transpose(-1, -2), r1))

    scale = (length_frames.reshape(batch, 1, 1) + 1.0).to(dtype)
    m0 = omega0 * scale
    m1 = omega1 * scale
    path_angle = _norm(delta, dim=-1, keepdim=True)
    tangent_cap = torch.minimum(
        path_angle + 0.20,
        torch.full_like(path_angle, 0.75),
    )
    m0 = _limit_norm(m0, tangent_cap)
    m1 = _limit_norm(m1, tangent_cap)

    d = delta - m0
    e = m1 - m0
    a3 = 10.0 * d - 4.0 * e
    a4 = -15.0 * d + 7.0 * e
    a5 = 6.0 * d - 3.0 * e

    u = t.reshape(batch, count, 1, 1)
    tangent = (
        m0[:, None] * u
        + a3[:, None] * u**3
        + a4[:, None] * u**4
        + a5[:, None] * u**5
    )
    tangent = _limit_norm(
        tangent,
        torch.minimum(
            path_angle[:, None] + 0.20,
            torch.full_like(path_angle[:, None], 1.25),
        ),
    )
    rotations = torch.matmul(r0[:, None], axis_angle_to_matrix(tangent))

    root0 = start[..., ROOT]
    root1 = end[..., ROOT]
    root_scale = (
        length_frames.reshape(batch, 1) + 1.0
    ).to(dtype)
    root_v0 = start_velocity[..., ROOT] * root_scale
    root_v1 = end_velocity[..., ROOT] * root_scale
    root_d = root1 - root0 - root_v0
    root_e = root_v1 - root_v0
    root_a3 = 10.0 * root_d - 4.0 * root_e
    root_a4 = -15.0 * root_d + 7.0 * root_e
    root_a5 = 6.0 * root_d - 3.0 * root_e
    ur = t
    root = (
        root0[:, None]
        + root_v0[:, None] * ur
        + root_a3[:, None] * ur**3
        + root_a4[:, None] * ur**4
        + root_a5[:, None] * ur**5
    )
    root[..., ROOT_X - ROOT.start] = 0.0
    root[..., ROOT_Z - ROOT.start] = 0.0

    smooth = quintic_smootherstep(t)
    start_logits = _safe_contact_logit(start[..., CONTACT])[:, None]
    end_logits = _safe_contact_logit(end[..., CONTACT])[:, None]
    contact_logits = (1.0 - smooth) * start_logits + smooth * end_logits
    return contact_logits, root, rotations


def continuous_time_features(t: torch.Tensor, bands: int) -> torch.Tensor:
    """Band-limited continuous coordinates.

    Default bands=5 gives frequencies 1,2,4,8,16 instead of the unstable
    1..512 spectrum used by the first V30 implementation.
    """
    if t.shape[-1] != 1:
        t = t[..., :1]
    frequencies = (2.0 ** torch.arange(
        int(bands), device=t.device, dtype=t.dtype
    )).reshape(*([1] * (t.ndim - 1)), -1)
    phase = math.pi * t * frequencies
    return torch.cat(
        [t, t**2, t**3, torch.sin(phase), torch.cos(phase)], dim=-1
    )


class ConditionEncoder(torch.nn.Module):
    def __init__(self, config: V32INRConfig) -> None:
        super().__init__()
        self.music_dim = int(config.music_dim)
        input_dim = 4 * config.motion_dim + self.music_dim + 1
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, config.condition_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.condition_dim * 2, config.condition_dim),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        start_velocity: torch.Tensor,
        end_velocity: torch.Tensor,
        music: torch.Tensor,
        length_norm: torch.Tensor,
    ) -> torch.Tensor:
        if music.shape[-1] < self.music_dim:
            music = F.pad(music, (0, self.music_dim - music.shape[-1]))
        elif music.shape[-1] > self.music_dim:
            music = music[..., : self.music_dim]
        return self.net(torch.cat([
            start, end, start_velocity, end_velocity, music, length_norm
        ], dim=-1))


class MaskedTemporalBlock(torch.nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = torch.nn.GroupNorm(groups, channels)
        self.conv1 = torch.nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.norm2 = torch.nn.GroupNorm(groups, channels)
        self.conv2 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.conv1(F.silu(self.norm1(x))))
        h = self.conv2(F.silu(self.norm2(h)))
        return (x + h) * mask[:, None]


class TransitionINREncoder(torch.nn.Module):
    """Deterministic transition encoder.

    V32 intentionally removes posterior sampling and KL to avoid the V30
    train/inference latent mismatch. Latent variance/covariance regularisation
    is applied by the training script.
    """
    def __init__(self, config: V32INRConfig) -> None:
        super().__init__()
        self.input = torch.nn.Conv1d(
            config.motion_dim + 2, config.encoder_hidden, 1
        )
        self.blocks = torch.nn.ModuleList([
            MaskedTemporalBlock(config.encoder_hidden, dilation, config.dropout)
            for dilation in (1, 2, 4, 8, 16, 8, 4, 2)
        ])
        pooled_dim = config.encoder_hidden * 3 + config.condition_dim
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(pooled_dim),
            torch.nn.Linear(pooled_dim, config.encoder_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.encoder_hidden, config.latent_dim),
        )

    def forward(
        self,
        motion: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = motion.shape
        coordinate = torch.linspace(
            1.0 / (length + 1), length / (length + 1), length,
            device=motion.device, dtype=motion.dtype,
        ).reshape(1, length, 1).expand(batch, -1, -1)
        features = torch.cat([motion, coordinate, mask[..., None]], dim=-1)
        x = self.input(features.transpose(1, 2))
        for block in self.blocks:
            x = block(x, mask)
        sequence = x.transpose(1, 2)
        weight = mask[..., None]
        denominator = weight.sum(dim=1).clamp_min(1.0)
        mean = (sequence * weight).sum(dim=1) / denominator
        centered = sequence - mean[:, None]
        std = torch.sqrt(
            (centered.square() * weight).sum(dim=1) / denominator + 1e-6
        )
        maximum = sequence.masked_fill(
            mask[..., None] < 0.5, -1e4
        ).amax(dim=1)
        return self.output(torch.cat([mean, std, maximum, condition], dim=-1))


class INRResidualBlock(torch.nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden)
        self.linear1 = torch.nn.Linear(hidden, hidden * 2)
        self.linear2 = torch.nn.Linear(hidden * 2, hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear2(self.dropout(F.silu(self.linear1(self.norm(x)))))
        return x + h


class ContinuousContactINR(torch.nn.Module):
    def __init__(self, config: V32INRConfig) -> None:
        super().__init__()
        self.config = config
        time_dim = 3 + 2 * config.fourier_bands
        input_dim = time_dim + config.latent_dim + config.condition_dim
        self.input = torch.nn.Linear(input_dim, config.inr_hidden)
        self.blocks = torch.nn.ModuleList([
            INRResidualBlock(config.inr_hidden, config.dropout)
            for _ in range(max(2, config.inr_layers))
        ])
        self.norm = torch.nn.LayerNorm(config.inr_hidden)
        self.rotation_head = torch.nn.Linear(
            config.inr_hidden, NUM_JOINTS * 3
        )
        self.root_y_head = torch.nn.Linear(config.inr_hidden, 1)
        self.contact_head = torch.nn.Linear(config.inr_hidden, 4)
        for head, std in (
            (self.rotation_head, 1e-4),
            (self.root_y_head, 1e-4),
            (self.contact_head, 1e-3),
        ):
            torch.nn.init.normal_(head.weight, 0.0, std)
            torch.nn.init.zeros_(head.bias)

    def forward(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, count, _ = t.shape
        time_feature = continuous_time_features(
            t, self.config.fourier_bands
        )
        x = torch.cat([
            time_feature,
            latent[:, None].expand(-1, count, -1),
            condition[:, None].expand(-1, count, -1),
        ], dim=-1)
        x = F.silu(self.input(x))
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.norm(x))
        envelope = c3_zero_envelope(t)
        rotation = self.rotation_head(x).reshape(
            batch, count, NUM_JOINTS, 3
        )
        rotation = (
            torch.tanh(rotation)
            * envelope[..., None]
            * float(self.config.residual_rotation_scale)
        )
        root_y = (
            torch.tanh(self.root_y_head(x))
            * envelope
            * float(self.config.residual_root_y_scale)
        )
        contact_logits = (
            torch.tanh(self.contact_head(x))
            * envelope
            * float(self.config.contact_logit_scale)
        )
        return rotation, root_y, contact_logits


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.reshape(-1, 1)
        half = max(1, self.dim // 2)
        frequency = torch.exp(torch.linspace(
            math.log(1.0), math.log(10000.0), half,
            device=x.device, dtype=x.dtype,
        ))
        phase = x / frequency.reshape(1, -1)
        embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return F.pad(
            embedding,
            (0, max(0, self.dim - embedding.shape[-1])),
        )[..., : self.dim]


class LatentResidualBlock(torch.nn.Module):
    def __init__(self, hidden: int, cond_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden)
        self.linear1 = torch.nn.Linear(hidden, hidden * 2)
        self.linear2 = torch.nn.Linear(hidden * 2, hidden)
        self.condition = torch.nn.Linear(cond_dim, hidden * 2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.condition(condition).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + gamma) + beta
        h = self.linear2(self.dropout(F.silu(self.linear1(h))))
        return x + h


class LatentDiffusionDenoiser(torch.nn.Module):
    def __init__(self, config: V32INRConfig) -> None:
        super().__init__()
        self.time = torch.nn.Sequential(
            SinusoidalEmbedding(96),
            torch.nn.Linear(96, config.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.input = torch.nn.Linear(
            config.latent_dim, config.diffusion_hidden
        )
        self.blocks = torch.nn.ModuleList([
            LatentResidualBlock(
                config.diffusion_hidden,
                config.condition_dim,
                config.dropout,
            )
            for _ in range(config.diffusion_blocks)
        ])
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(config.diffusion_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(config.diffusion_hidden, config.latent_dim),
        )
        torch.nn.init.normal_(self.output[-1].weight, 0.0, 1e-3)
        torch.nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        diffusion_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        cond = condition + self.time(diffusion_time)
        x = self.input(noisy_latent)
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x)


class V32ContactINRSystem(torch.nn.Module):
    def __init__(self, config: V32INRConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = ConditionEncoder(config)
        self.encoder = TransitionINREncoder(config)
        self.inr = ContinuousContactINR(config)
        self.diffusion = LatentDiffusionDenoiser(config)

    def condition(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        start_velocity: torch.Tensor,
        end_velocity: torch.Tensor,
        music: torch.Tensor,
        length_frames: torch.Tensor,
    ) -> torch.Tensor:
        length_norm = (
            length_frames.reshape(-1, 1)
            / float(max(self.config.max_len, 1))
        ).clamp(0.0, 1.5)
        return self.condition_encoder(
            start, end, start_velocity, end_velocity,
            music, length_norm,
        )

    def encode(
        self,
        motion: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(motion, mask, condition)

    def decode(
        self,
        latent: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        start_velocity: torch.Tensor,
        end_velocity: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        length_frames: torch.Tensor,
        return_aux: bool = False,
        boundary_state: Dict[str, torch.Tensor] | None = None,
    ):
        if boundary_state is None:
            # Compatibility path for historical callers/checkpoints.  Velocity
            # is retained, while acceleration/jerk default to zero.
            batch = start.shape[0]
            r0 = _rotation(start)
            r1 = _rotation(end)
            rs = _rotation(project_motion_rotations_torch(start + start_velocity))
            re = _rotation(project_motion_rotations_torch(end - end_velocity))
            zeros_w = start.new_zeros((batch, NUM_JOINTS, 3))
            zeros_r = start.new_zeros((batch, 3))
            boundary_state = {
                "start_omega": matrix_to_axis_angle(
                    torch.matmul(r0.transpose(-1, -2), rs)
                ),
                "end_omega": matrix_to_axis_angle(
                    torch.matmul(re.transpose(-1, -2), r1)
                ),
                "start_alpha": zeros_w,
                "end_alpha": zeros_w,
                "start_angular_jerk": zeros_w,
                "end_angular_jerk": zeros_w,
                "start_root_velocity": start_velocity[..., ROOT],
                "end_root_velocity": end_velocity[..., ROOT],
                "start_root_acceleration": zeros_r,
                "end_root_acceleration": zeros_r,
                "start_root_jerk": zeros_r,
                "end_root_jerk": zeros_r,
            }
        base_logits, base_root, base_rotation = septic_so3_root_base(
            start, end, t, length_frames, boundary_state
        )
        residual_rotation, residual_root_y, residual_contact_logits = (
            self.inr(t, latent, condition)
        )
        rotation = torch.matmul(
            base_rotation,
            axis_angle_to_matrix(residual_rotation),
        )
        rot6d = matrix_to_rotation_6d(rotation).reshape(
            t.shape[0], t.shape[1], NUM_JOINTS * 6
        )
        contact_logits = base_logits + residual_contact_logits
        contacts = torch.sigmoid(contact_logits)
        root = base_root.clone()
        root[..., 1:2] += residual_root_y
        root[..., 0] = 0.0
        root[..., 2] = 0.0
        motion = project_motion_rotations_torch(
            torch.cat([contacts, root, rot6d], dim=-1)
        )
        if return_aux:
            return motion, {
                "contact_logits": contact_logits,
                "contact_prob": contacts,
                "residual_rotation": residual_rotation,
                "residual_root_y": residual_root_y,
            }
        return motion


def make_c2_transition_np(
    previous: np.ndarray,
    following: np.ndarray,
    length: int,
) -> np.ndarray:
    """Compatibility name for the V34 regularised septic boundary path."""
    return make_v34_transition_np(previous, following, length)


def linear_beta_schedule(
    steps: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1e-4, 0.02, int(steps), device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def selected_timesteps(
    train_steps: int,
    inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    count = max(2, min(int(inference_steps), int(train_steps)))
    values = torch.linspace(
        train_steps - 1, 0, count, device=device
    ).round().long()
    return torch.unique_consecutive(values)
