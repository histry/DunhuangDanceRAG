import os
import copy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pytorch3d.transforms import rotation_6d_to_matrix

from dataset.quaternion import ax_from_6v
from vis import audio_output_stem, skeleton_render

from .utils import extract, make_beta_schedule


def identity(t, *args, **kwargs):
    return t


def safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x ** 2, dim=dim) + eps)


def move_condition_to_device(cond, device):
    if isinstance(cond, dict):
        moved = {}
        for key, value in cond.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return moved
    return cond.to(device)


def maybe_unnormalize(normalizer, x):
    if normalizer is None:
        return x
    out = normalizer.unnormalize(x)
    if isinstance(out, np.ndarray):
        out = torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
    return out.to(device=x.device, dtype=x.dtype)


def rotation_angle_between(rot_mats):
    """
    rot_mats: [B, T, J, 3, 3]
    return: [B, T-1, J]
    """
    rel = torch.matmul(rot_mats[:, :-1].transpose(-1, -2), rot_mats[:, 1:])
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_angle)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(),
            ma_model.parameters(),
        ):
            old_weight = ma_params.data
            up_weight = current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        mmr_model=None,
        mmr_loss_weight=0.0,
        keyframe_condition_prob=0.7,
        keyframe_condition_width=3,
        keyframe_loss_weight=2.0,
        mid_keyframe_condition_prob=0.0,
        mid_keyframe_count=2,
        mid_keyframe_condition_width=1,
        mid_keyframe_selection="motion_peak",
        data_fps=30,
        contact_loss_weight=0.8,
        foot_loss_weight=2.5,
        sync_loss_weight=1.2,
        hard_keyframe_project=False,
        beat_guidance_weight=0.0,
        trajectory_loss_weight=1.0,
        trajectory_velocity_loss_weight=0.25,
        energy_condition_prob=0.7,
        energy_condition_drop_prob=0.15,
        energy_loss_weight=0.25,
        root_lower_coupling_loss_weight=0.5,
        root_lower_speed_threshold=0.012,
        root_lower_min_motion=0.010,
        force_audio_only_drop=False,
        disable_unpaired_audio_condition=True,
        tto_trajectory_loss_weight=4.0,
        tto_trajectory_velocity_loss_weight=0.5,
        tto_root_acc_loss_weight=0.05,
        tto_foot_loss_weight=0.25,
    ):
        super().__init__()
        self.disable_unpaired_audio_condition = bool(disable_unpaired_audio_condition)

        self.tto_trajectory_loss_weight = float(tto_trajectory_loss_weight)
        self.tto_trajectory_velocity_loss_weight = float(tto_trajectory_velocity_loss_weight)
        self.tto_root_acc_loss_weight = float(tto_root_acc_loss_weight)
        self.tto_foot_loss_weight = float(tto_foot_loss_weight)

        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.smpl = smpl
        self.dt = 1.0 / float(data_fps)

        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised
        self.guidance_weight = guidance_weight
        self.cond_drop_prob = cond_drop_prob
        self.force_audio_only_drop = force_audio_only_drop

        self.mmr_model = mmr_model
        self.mmr_loss_weight = float(mmr_loss_weight)
        if self.mmr_model is not None:
            self.mmr_model.eval()
            for param in self.mmr_model.parameters():
                param.requires_grad = False

        self.keyframe_condition_prob = float(keyframe_condition_prob)
        self.keyframe_condition_width = int(keyframe_condition_width)
        self.keyframe_loss_weight = float(keyframe_loss_weight)

        self.mid_keyframe_condition_prob = float(mid_keyframe_condition_prob)
        self.mid_keyframe_count = int(mid_keyframe_count)
        self.mid_keyframe_condition_width = int(mid_keyframe_condition_width)
        self.mid_keyframe_selection = str(mid_keyframe_selection)

        self.contact_loss_weight = float(contact_loss_weight)
        self.foot_loss_weight = float(foot_loss_weight)
        self.sync_loss_weight = float(sync_loss_weight)

        self.hard_keyframe_project = bool(hard_keyframe_project)
        self.beat_guidance_weight = float(beat_guidance_weight)

        self.trajectory_loss_weight = float(trajectory_loss_weight)
        self.trajectory_velocity_loss_weight = float(trajectory_velocity_loss_weight)

        self.energy_condition_prob = float(energy_condition_prob)
        self.energy_condition_drop_prob = float(energy_condition_drop_prob)
        self.energy_loss_weight = float(energy_loss_weight)
        self.root_lower_coupling_loss_weight = float(root_lower_coupling_loss_weight)
        self.root_lower_speed_threshold = float(root_lower_speed_threshold)
        self.root_lower_min_motion = float(root_lower_min_motion)

        # TTO 参数可以在 generate_controlled.py 中覆盖。
        self.tto_interval = 50
        self.tto_steps = 1
        self.tto_lr = 0.03
        self.tto_contact_threshold = 0.65

        # 151-D representation:
        # [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotation
        self.root_x_idx = 4
        self.root_y_idx = 5
        self.root_z_idx = 6
        self.contact_slice = slice(0, 4)
        self.root_slice = slice(4, 7)
        self.rot_slice = slice(7, 151)

        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.normalizer = None

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod",
            torch.log(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (
                self.p2_loss_weight_k
                + alphas_cumprod / torch.clamp(1.0 - alphas_cumprod, min=1e-8)
            )
            ** -self.p2_loss_weight_gamma,
        )

        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ---------------------------------------------------------------------
    # Diffusion math
    # ---------------------------------------------------------------------

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        return noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / torch.clamp(
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape),
            min=1e-8,
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped,
            t,
            x_t.shape,
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        cond,
        t,
        weight=None,
        clip_x_start=False,
        constraint=None,
    ):
        weight = self.guidance_weight if weight is None else weight

        force_mask = None
        force_x_clean = None
        if constraint is not None:
            force_mask = constraint.get("mask", None)
            force_x_clean = constraint.get("value", None)

        if hasattr(self.model, "guided_forward"):
            model_output = self.model.guided_forward(
                x,
                cond,
                t,
                weight,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )
        else:
            model_output = self.model(
                x,
                cond,
                t,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )

        if self.predict_epsilon:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        else:
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if clip_x_start:
            if x_start.shape[-1] > 7:
                x_start = torch.cat(
                    [
                        x_start[..., :7],
                        x_start[..., 7:].clamp(-1.0, 1.0),
                    ],
                    dim=-1,
                )
            else:
                x_start = x_start.clamp(-1.0, 1.0)

            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, cond, t, clip_denoised=True, constraint=None):
        _, x_recon = self.model_predictions(
            x,
            cond,
            t,
            clip_x_start=clip_denoised,
            constraint=constraint,
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    # ---------------------------------------------------------------------
    # Keyframe conditioning
    # ---------------------------------------------------------------------

    def _normalize_keyframe_scores(self, scores):
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores - scores.amin(dim=1, keepdim=True)
        denom = scores.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return scores / denom

    def _middle_keyframe_scores(self, x_start, cond):
        b, s, c = x_start.shape
        device = x_start.device
        mode = self.mid_keyframe_selection

        score_parts = []

        if mode in ["motion_peak", "mixed"]:
            motion_feat = x_start[..., 4:] if c == 151 else x_start
            motion_delta = torch.zeros((b, s), device=device, dtype=torch.float32)

            if s > 1:
                motion_diff = motion_feat[:, 1:].float() - motion_feat[:, :-1].float()
                motion_delta[:, 1:] = safe_norm(motion_diff, dim=-1)
                motion_delta[:, 0] = motion_delta[:, 1]

            kernel = min(5, s)
            if kernel % 2 == 0:
                kernel -= 1

            if kernel > 1:
                motion_delta = F.avg_pool1d(
                    motion_delta.unsqueeze(1),
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                ).squeeze(1)

            score_parts.append(self._normalize_keyframe_scores(motion_delta))

        if mode in ["audio_onset", "mixed"]:
            audio_feat = cond.get("audio", None) if isinstance(cond, dict) else None
            if audio_feat is not None:
                audio_feat = audio_feat.to(device=device).float()

                if audio_feat.shape[1] != s:
                    audio_feat = F.interpolate(
                        audio_feat.transpose(1, 2),
                        size=s,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)

                if audio_feat.shape[-1] > 768:
                    audio_score = audio_feat[..., 768]
                else:
                    audio_score = torch.zeros((b, s), device=device, dtype=torch.float32)
                    if s > 1:
                        audio_diff = audio_feat[:, 1:] - audio_feat[:, :-1]
                        audio_score[:, 1:] = safe_norm(audio_diff, dim=-1)
                        audio_score[:, 0] = audio_score[:, 1]

                score_parts.append(self._normalize_keyframe_scores(audio_score))

        if mode == "random" or not score_parts:
            return torch.rand((b, s), device=device, dtype=torch.float32)

        return torch.stack(score_parts, dim=0).mean(dim=0)

    def _build_keyframe_condition(self, x_start, cond=None):
        """
        Build known clean-frame mask/value for inpainting-style training.

        - start/end keyframes: controlled by keyframe_condition_prob
        - optional middle keyframes: controlled by mid_keyframe_condition_prob
        """
        b, s, c = x_start.shape
        device = x_start.device
        dtype = x_start.dtype

        force_mask = torch.zeros((b, s, 1), device=device, dtype=dtype)
        force_value = torch.zeros_like(x_start)

        if self.keyframe_condition_prob > 0:
            use_keyframe = torch.rand((b,), device=device) < self.keyframe_condition_prob
            width = max(1, min(int(self.keyframe_condition_width), s))

            for batch_idx in range(b):
                if bool(use_keyframe[batch_idx].item()):
                    force_mask[batch_idx, :width, 0] = 1.0
                    force_mask[batch_idx, s - width :, 0] = 1.0

        if self.mid_keyframe_condition_prob > 0 and self.mid_keyframe_count > 0 and s > 4:
            use_middle = torch.rand((b,), device=device) < self.mid_keyframe_condition_prob
            scores = self._middle_keyframe_scores(x_start, cond or {})

            # Do not select first/last frames as middle keyframes.
            scores[:, 0] = -1.0
            scores[:, -1] = -1.0

            max_mid = min(int(self.mid_keyframe_count), max(1, s - 2))
            width = max(0, int(self.mid_keyframe_condition_width))

            for batch_idx in range(b):
                if not bool(use_middle[batch_idx].item()):
                    continue

                k = max_mid
                indices = torch.topk(scores[batch_idx], k=k, largest=True).indices

                for frame_idx in indices:
                    frame = int(frame_idx.item())
                    start = max(0, frame - width)
                    end = min(s, frame + width + 1)
                    force_mask[batch_idx, start:end, 0] = 1.0

        # If trajectory is provided, root X/Z should be controlled by the
        # trajectory branch rather than by image-derived keyframes. This avoids
        # conflicts between pose constraints and path constraints.
        if (
            x_start.shape[-1] == 151
            and isinstance(cond, dict)
            and cond.get("trajectory", None) is not None
        ):
            feature_mask = force_mask.expand(-1, -1, x_start.shape[-1]).clone()
            feature_mask[..., self.root_x_idx] = 0.0
            feature_mask[..., self.root_z_idx] = 0.0

            force_value = x_start * feature_mask
            return {"mask": feature_mask, "value": force_value}

        force_value = x_start * force_mask
        return {"mask": force_mask, "value": force_value}

    def _merge_constraints(self, generated_constraint, external_constraint):
        """
        Merge generated training keyframe constraints and external inference constraints.

        Important:
        - Preserve feature-wise masks.
        - Do not collapse [B,T,151] mask to [B,T,1], otherwise root X/Z may be
        accidentally constrained by keyframes.
        """
        if generated_constraint is None:
            return external_constraint
        if external_constraint is None:
            return generated_constraint

        value_a = generated_constraint["value"]
        mask_a = generated_constraint["mask"].to(
            device=value_a.device,
            dtype=value_a.dtype,
        )

        value_b = external_constraint["value"].to(
            device=value_a.device,
            dtype=value_a.dtype,
        )
        mask_b = external_constraint["mask"].to(
            device=value_a.device,
            dtype=value_a.dtype,
        )

        def expand_mask(mask, value):
            if mask.shape[-1] == 1:
                return mask.expand_as(value)
            if mask.shape[-1] == value.shape[-1]:
                return mask
            raise ValueError(
                f"constraint mask last dim must be 1 or {value.shape[-1]}, "
                f"got {mask.shape[-1]}"
            )

        mask_a_full = expand_mask(mask_a, value_a)
        mask_b_full = expand_mask(mask_b, value_a)

        merged_mask_full = torch.maximum(mask_a_full, mask_b_full)

        # external constraint has higher priority
        merged_value = value_a * (1.0 - mask_b_full) + value_b * mask_b_full

        return {
            "mask": merged_mask_full,
            "value": merged_value,
        }

    # ---------------------------------------------------------------------
    # Training losses
    # ---------------------------------------------------------------------

    def _loss_per_sample(self, pred, target):
        return self.loss_fn(pred, target, reduction="none")

    def _p2_apply(self, loss_per_sample, t):
        """
        loss_per_sample: [B] or broadcastable to [B]
        """
        return loss_per_sample * extract(self.p2_loss_weight, t, loss_per_sample.shape)

    def _reconstruction_loss(self, model_motion_x0, target_motion_x0, t):
        loss = self._loss_per_sample(model_motion_x0, target_motion_x0).mean(dim=(1, 2))
        loss = self._p2_apply(loss, t)
        return loss.mean()

    def _velocity_loss(self, model_motion_x0, target_motion_x0, t):
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        pred_vel = model_motion_x0[:, 1:] - model_motion_x0[:, :-1]
        target_vel = target_motion_x0[:, 1:] - target_motion_x0[:, :-1]

        loss = self._loss_per_sample(pred_vel, target_vel).mean(dim=(1, 2))
        loss = self._p2_apply(loss, t)
        return loss.mean()

    def _keyframe_loss(self, model_motion_x0, constraint):
        if constraint is None:
            return model_motion_x0.new_tensor(0.0)

        mask = constraint.get("mask", None)
        value = constraint.get("value", None)
        if mask is None or value is None:
            return model_motion_x0.new_tensor(0.0)

        mask = mask.to(device=model_motion_x0.device, dtype=model_motion_x0.dtype)
        value = value.to(device=model_motion_x0.device, dtype=model_motion_x0.dtype)

        if mask.shape[-1] == 1:
            feature_mask = mask.expand_as(model_motion_x0)
            denom = mask.sum() * model_motion_x0.shape[-1]
        elif mask.shape[-1] == model_motion_x0.shape[-1]:
            feature_mask = mask
            denom = mask.sum()
        else:
            raise ValueError(
                f"constraint mask last dim must be 1 or {model_motion_x0.shape[-1]}, "
                f"got {mask.shape[-1]}"
            )

        if float(denom.item()) <= 1e-8:
            return model_motion_x0.new_tensor(0.0)

        sq = (model_motion_x0 - value) ** 2 * feature_mask
        return sq.sum() / denom.clamp_min(1e-6)

    def _trajectory_training_loss(self, model_motion_x0, cond, t):
        """
        Supervise generated root X/Z against trajectory condition in normalized space.

        This fixes the previous weak point:
        trajectory was used as condition, but the loss did not explicitly log/supervise
        root X/Z trajectory error during training.
        """
        zero = model_motion_x0.new_tensor(0.0)

        if (
            not isinstance(cond, dict)
            or cond.get("trajectory", None) is None
            or float(self.trajectory_loss_weight) <= 0.0
        ):
            return zero, zero

        target_traj = cond["trajectory"].to(
            device=model_motion_x0.device,
            dtype=model_motion_x0.dtype,
        )

        if target_traj.shape[1] != model_motion_x0.shape[1]:
            target_traj = F.interpolate(
                target_traj.transpose(1, 2),
                size=model_motion_x0.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        target_traj = target_traj[..., :2]

        if model_motion_x0.shape[-1] == 151:
            pred_traj = model_motion_x0[:, :, [self.root_x_idx, self.root_z_idx]]
        else:
            pred_traj = model_motion_x0[:, :, [0, 2]]

        traj_pos = self._loss_per_sample(pred_traj, target_traj).mean(dim=(1, 2))
        traj_pos = self._p2_apply(traj_pos, t).mean()

        if pred_traj.shape[1] > 1:
            pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]
            target_vel = target_traj[:, 1:] - target_traj[:, :-1]
            traj_vel = self._loss_per_sample(pred_vel, target_vel).mean(dim=(1, 2))
            traj_vel = self._p2_apply(traj_vel, t).mean()
        else:
            traj_vel = zero

        return traj_pos, traj_vel

    def _contact_loss(self, model_motion_x0, target_motion_x0):
        """
        Balanced contact regression in physical contact space.

        Why:
        - Raw normalized contact channels are not directly interpretable as 0/1.
        - Contacts are sparse/imbalanced; plain MSE can be dominated by non-contact frames.
        """
        if model_motion_x0.shape[-1] != 151:
            return model_motion_x0.new_tensor(0.0)

        if self.normalizer is not None:
            pred_physical = maybe_unnormalize(self.normalizer, model_motion_x0)
            target_physical = maybe_unnormalize(self.normalizer, target_motion_x0)
        else:
            pred_physical = model_motion_x0
            target_physical = target_motion_x0

        pred_contacts = pred_physical[:, :, self.contact_slice].clamp(0.0, 1.0)
        target_contacts = target_physical[:, :, self.contact_slice].clamp(0.0, 1.0)

        # 平衡 contact / non-contact，避免模型全部预测 non-contact 也能拿低 loss。
        pos = target_contacts
        neg = 1.0 - target_contacts
        pos_weight = neg.sum() / pos.sum().clamp_min(1.0)
        weight = neg + pos * pos_weight.clamp(max=10.0)

        return ((pred_contacts - target_contacts) ** 2 * weight).mean()

    def _fk_positions(self, motion_x0):
        """
        motion_x0 should be physical-space 151-D motion.
        return joints: [B, T, J, 3]
        """
        pos = motion_x0[:, :, self.root_slice]
        q = ax_from_6v(motion_x0[:, :, self.rot_slice].reshape(
            motion_x0.shape[0],
            motion_x0.shape[1],
            24,
            6,
        ))
        return self.smpl.forward(q, pos)

    def _fk_loss(self, model_motion_x0, target_motion_x0):
        if model_motion_x0.shape[-1] != 151:
            return model_motion_x0.new_tensor(0.0)

        if self.normalizer is not None:
            pred_physical = maybe_unnormalize(self.normalizer, model_motion_x0)
            target_physical = maybe_unnormalize(self.normalizer, target_motion_x0)
        else:
            pred_physical = model_motion_x0
            target_physical = target_motion_x0

        try:
            pred_joints = self._fk_positions(pred_physical)
            target_joints = self._fk_positions(target_physical)
            return F.mse_loss(pred_joints, target_joints)
        except Exception:
            return model_motion_x0.new_tensor(0.0)

    def _foot_sliding_loss(self, model_motion_x0, target_motion_x0=None):
        """
        Foot sliding loss gated by target contacts during training.

        Main fix:
        - Do NOT rely only on predicted contacts.
        - If predicted contacts are wrong, the old loss can disappear.
        - During training, target contacts are available and should gate foot-lock frames.
        """
        if model_motion_x0.shape[-1] != 151:
            return model_motion_x0.new_tensor(0.0)

        if self.normalizer is not None:
            pred_physical = maybe_unnormalize(self.normalizer, model_motion_x0)
            target_physical = (
                maybe_unnormalize(self.normalizer, target_motion_x0)
                if target_motion_x0 is not None
                else None
            )
        else:
            pred_physical = model_motion_x0
            target_physical = target_motion_x0

        try:
            # 优先使用 GT contact 判断落地帧。
            if target_physical is not None:
                contacts = target_physical[:, :, self.contact_slice] > 0.5
            else:
                contacts = pred_physical[:, :, self.contact_slice] > self.tto_contact_threshold

            contact_pairs = contacts[:, 1:] & contacts[:, :-1]

            if not bool(contact_pairs.any().item()):
                return model_motion_x0.new_tensor(0.0)

            joints = self._fk_positions(pred_physical)
            feet = joints[:, :, [7, 8, 10, 11], :]
            feet_delta = feet[:, 1:] - feet[:, :-1]

            # 只惩罚接触期间的水平滑动，不惩罚正常抬脚。
            horizontal_speed_sq = feet_delta[..., [0, 2]].pow(2).sum(dim=-1)

            return horizontal_speed_sq[contact_pairs].mean()

        except Exception:
            return model_motion_x0.new_tensor(0.0)

    def _anti_freeze_loss(self, model_motion_x0):
        """
        Penalize almost completely static outputs very weakly.
        This is a safety term, not a main objective.
        """
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        delta = model_motion_x0[:, 1:] - model_motion_x0[:, :-1]
        energy = safe_norm(delta, dim=-1).mean()
        return F.relu(0.015 - energy)

    def _motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        pred_energy = safe_norm(model_motion_x0[:, 1:] - model_motion_x0[:, :-1], dim=-1)
        target_energy = safe_norm(target_motion_x0[:, 1:] - target_motion_x0[:, :-1], dim=-1)
        return F.mse_loss(pred_energy, target_energy)

    def _body_stability_loss(self, model_motion_x0):
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 3:
            return model_motion_x0.new_tensor(0.0)

        if self.normalizer is not None:
            physical = maybe_unnormalize(self.normalizer, model_motion_x0)
        else:
            physical = model_motion_x0

        root = physical[:, :, self.root_slice]
        root_acc = root[:, 2:] - 2.0 * root[:, 1:-1] + root[:, :-2]
        return root_acc.pow(2).mean()

    def _root_turn_loss(self, model_motion_x0):
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        rot6d = model_motion_x0[:, :, self.rot_slice].reshape(
            model_motion_x0.shape[0],
            model_motion_x0.shape[1],
            24,
            6,
        )

        try:
            root_rot = rotation_6d_to_matrix(rot6d[:, :, 0])
            angle = rotation_angle_between(root_rot.unsqueeze(2)).squeeze(2)
            return angle.pow(2).mean()
        except Exception:
            return model_motion_x0.new_tensor(0.0)



    def _kinematic_sync_loss(self, model_motion_x0, target_motion_x0=None, cond=None):
        """Contrastive condition-driven root-lower coupling loss.

        v4 design:
        Absolute trajectory-speed thresholds are unreliable because trajectory
        conditions may be normalized and heavily scaled.  Instead, this loss
        uses within-sequence contrast:

        - frames with top trajectory speed are "fast phase"
        - frames with bottom trajectory speed are "slow phase"
        - lower-body motion in fast phase should exceed slow phase by a margin

        This directly supervises root-speed -> lower-body response without
        depending on a global unit scale.
        """
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 4:
            return model_motion_x0.new_tensor(0.0)

        if float(getattr(self, "root_lower_coupling_loss_weight", 0.0)) <= 0.0:
            return model_motion_x0.new_tensor(0.0)

        # Use physical space when possible.
        if self.normalizer is not None:
            pred = maybe_unnormalize(self.normalizer, model_motion_x0)
            target = (
                maybe_unnormalize(self.normalizer, target_motion_x0)
                if target_motion_x0 is not None
                else None
            )
        else:
            pred = model_motion_x0
            target = target_motion_x0

        b, t, _ = pred.shape
        device = pred.device
        dtype = pred.dtype

        # ------------------------------------------------------------------
        # 1. Drive signal: commanded trajectory speed if available.
        # ------------------------------------------------------------------
        target_traj = None
        if isinstance(cond, dict):
            target_traj = cond.get("trajectory", None)

        if target_traj is not None:
            target_traj = target_traj.to(device=device, dtype=dtype)
            if target_traj.shape[1] != t:
                target_traj = F.interpolate(
                    target_traj.transpose(1, 2),
                    size=t,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            drive_root = target_traj[..., :2]
            drive_speed = safe_norm(drive_root[:, 1:] - drive_root[:, :-1], dim=-1)
        else:
            root_xz = pred[:, :, [self.root_x_idx, self.root_z_idx]]
            drive_speed = safe_norm(root_xz[:, 1:] - root_xz[:, :-1], dim=-1)

        # Normalize drive speed within each sample.
        drive_min = drive_speed.amin(dim=1, keepdim=True)
        drive_max = drive_speed.amax(dim=1, keepdim=True)
        drive_span = (drive_max - drive_min).clamp_min(1e-8)
        drive_norm = (drive_speed - drive_min) / drive_span

        # If a sequence is almost perfectly constant in trajectory speed,
        # contrastive supervision is not meaningful.
        valid_seq = (drive_max.squeeze(1) - drive_min.squeeze(1)) > 1e-7

        # ------------------------------------------------------------------
        # 2. Lower-body motion.
        # ------------------------------------------------------------------
        lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
        lower_indices = []
        for joint in lower_joints:
            start = 7 + 6 * joint
            lower_indices.extend(range(start, start + 6))
        lower_indices = torch.as_tensor(lower_indices, device=device, dtype=torch.long)

        pred_lower_delta = pred[:, 1:, lower_indices] - pred[:, :-1, lower_indices]
        pred_lower_motion = torch.sqrt(pred_lower_delta.pow(2).mean(dim=-1) + 1e-8)

        # Contact phase change is an auxiliary signal, not the main driver.
        contacts = pred[:, :, self.contact_slice].clamp(0.0, 1.0)
        contact_change = torch.abs(contacts[:, 1:] - contacts[:, :-1]).mean(dim=-1)

        # ------------------------------------------------------------------
        # 3. Top/bottom phase masks.
        # ------------------------------------------------------------------
        high_th = torch.quantile(drive_norm.detach(), 0.70, dim=1, keepdim=True)
        low_th = torch.quantile(drive_norm.detach(), 0.30, dim=1, keepdim=True)

        high_mask = (drive_norm >= high_th).to(dtype)
        low_mask = (drive_norm <= low_th).to(dtype)

        # Avoid empty masks.
        high_den = high_mask.sum(dim=1).clamp_min(1.0)
        low_den = low_mask.sum(dim=1).clamp_min(1.0)

        high_lower = (pred_lower_motion * high_mask).sum(dim=1) / high_den
        low_lower = (pred_lower_motion * low_mask).sum(dim=1) / low_den

        high_contact = (contact_change * high_mask).sum(dim=1) / high_den
        low_contact = (contact_change * low_mask).sum(dim=1) / low_den

        # root_lower_min_motion now acts as contrast margin.
        margin = pred.new_tensor(float(getattr(self, "root_lower_min_motion", 0.010)))

        # Encourage lower-body response to be stronger during fast commanded
        # trajectory phases than slow commanded phases.
        contrast_loss = torch.relu(low_lower + margin - high_lower)

        # Also encourage some contact phase change in fast phases.
        contact_margin = pred.new_tensor(0.005)
        contact_contrast_loss = torch.relu(low_contact + contact_margin - high_contact)

        # Optional absolute activity floor on fast phases.
        abs_floor = 0.5 * margin
        activity_floor_loss = torch.relu(abs_floor - high_lower)

        total = contrast_loss + 0.10 * contact_contrast_loss + 0.25 * activity_floor_loss
        total = total * valid_seq.to(dtype)

        if bool(int(__import__("os").environ.get("EDGE_DEBUG_ROOT_LOWER", "0"))):
            if not hasattr(self, "_root_lower_debug_printed"):
                self._root_lower_debug_printed = 0
            if self._root_lower_debug_printed < 20:
                with torch.no_grad():
                    print(
                        "🧪 root-lower v4 | "
                        f"drive raw mean/max={drive_speed.mean().item():.6f}/{drive_speed.max().item():.6f} | "
                        f"drive_norm mean/max={drive_norm.mean().item():.6f}/{drive_norm.max().item():.6f} | "
                        f"high_lower={high_lower.mean().item():.6f} | "
                        f"low_lower={low_lower.mean().item():.6f} | "
                        f"margin={margin.item():.6f} | "
                        f"contrast={contrast_loss.mean().item():.8f} | "
                        f"high_contact={high_contact.mean().item():.6f} | "
                        f"low_contact={low_contact.mean().item():.6f} | "
                        f"contact_contrast={contact_contrast_loss.mean().item():.8f} | "
                        f"activity_floor={activity_floor_loss.mean().item():.8f} | "
                        f"valid={valid_seq.float().mean().item():.3f}",
                        flush=True,
                    )
                self._root_lower_debug_printed += 1

        return total.mean() * float(getattr(self, "root_lower_coupling_loss_weight", 1.0))

    def _biomech_loss(self, model_motion_x0):
        """
        Conservative biomechanical smoothness term on rotations.
        """
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 3:
            return model_motion_x0.new_tensor(0.0)

        rot = model_motion_x0[:, :, self.rot_slice]
        rot_acc = rot[:, 2:] - 2.0 * rot[:, 1:-1] + rot[:, :-2]
        return rot_acc.pow(2).mean()

    def _contact_turn_loss(self, model_motion_x0):
        if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        contacts = model_motion_x0[:, :, self.contact_slice]
        contact_delta = contacts[:, 1:] - contacts[:, :-1]
        return contact_delta.pow(2).mean()

    def _mmr_loss(self, model_motion_x0, cond):
        if self.mmr_model is None or self.mmr_loss_weight <= 0:
            return model_motion_x0.new_tensor(0.0)

        if not isinstance(cond, dict) or cond.get("audio", None) is None:
            return model_motion_x0.new_tensor(0.0)

        audio = cond["audio"].to(device=model_motion_x0.device, dtype=model_motion_x0.dtype)

        try:
            with torch.no_grad():
                self.mmr_model.eval()

            # Try common CrossModalMMR signatures.
            if hasattr(self.mmr_model, "compute_loss"):
                return self.mmr_model.compute_loss(model_motion_x0, audio)

            out = self.mmr_model(model_motion_x0, audio)
            if isinstance(out, dict) and "loss" in out:
                return out["loss"]

            if torch.is_tensor(out):
                return out.mean()

            return model_motion_x0.new_tensor(0.0)
        except Exception:
            return model_motion_x0.new_tensor(0.0)

    @staticmethod
    def _linear_warmup(current_epoch, start=1, end=50):
        """
        Linear warmup coefficient for auxiliary losses.

        Returns:
            0.0 before or at `start`
            1.0 after or at `end`
            linearly increases between start and end

        Why:
            Diffusion training should first learn denoising / motion distribution.
            Keyframe, trajectory and physical constraints are auxiliary losses.
            If they are too strong at the beginning, they may destabilize the
            denoising objective and cause jitter, frozen motion, or bad contacts.
        """
        if current_epoch is None:
            return 1.0

        current_epoch = float(current_epoch)
        start = float(start)
        end = float(end)

        if current_epoch <= start:
            return 0.0
        if current_epoch >= end:
            return 1.0

        return float(current_epoch - start) / float(max(1.0, end - start))

    def _make_loss_tuple(
        self,
        recon_loss,
        velocity_loss,
        contact_loss,
        fk_loss,
        foot_loss,
        anti_freeze_loss,
        mmr_loss,
        trajectory_loss,
        keyframe_loss,
        sync_loss,
        biomech_loss,
        root_turn_loss,
        contact_turn_loss,
        body_stability_loss,
        motion_energy_loss,
    ):
        return (
            recon_loss,
            velocity_loss,
            contact_loss,
            fk_loss,
            foot_loss,
            anti_freeze_loss,
            mmr_loss,
            trajectory_loss,
            keyframe_loss,
            sync_loss,
            biomech_loss,
            root_turn_loss,
            contact_turn_loss,
            body_stability_loss,
            motion_energy_loss,
        )

    def p_losses(self, x_start, cond, t, noise=None, current_epoch=None, constraint=None):
        """
        Compute training loss with staged warmup.

        Design:
        - Main diffusion objective is always active.
        - Control losses (keyframe / trajectory) warm up early.
        - Physical losses (contact / FK / foot / sync / stability) warm up later.
        - Biomechanical regularization starts last because it is the easiest to over-constrain.

        This makes the training story easier to explain:
        first learn denoising and the motion distribution, then gradually strengthen
        controllability and physical plausibility.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        cond = move_condition_to_device(cond, x_start.device)

        train_constraint = self._build_keyframe_condition(x_start, cond)
        train_constraint = self._merge_constraints(train_constraint, constraint)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self._project_known_keyframes(x_noisy, train_constraint, t)

        force_mask = (
            train_constraint.get("mask", None)
            if train_constraint is not None
            else None
        )
        force_value = (
            train_constraint.get("value", None)
            if train_constraint is not None
            else None
        )

        cond_drop_prob = float(self.cond_drop_prob)
        batch_size = x_start.shape[0]
        device = x_start.device

        keep_audio_mask = None
        keep_traj_mask = None

        # If the dataset explicitly marks whether audio is genuinely paired,
        # never let unpaired/proxy audio become a strong training condition.
        # Paired audio still receives normal classifier-free dropout.
        if (
            self.disable_unpaired_audio_condition
            and isinstance(cond, dict)
            and cond.get("audio_paired", None) is not None
        ):
            paired_mask = cond["audio_paired"].to(
                device=device,
                dtype=torch.float32,
            ).view(-1) > 0.5

            if cond_drop_prob > 0:
                random_keep = (
                    torch.rand((batch_size,), device=device)
                    >= cond_drop_prob
                )
                keep_audio_mask = paired_mask & random_keep
            else:
                keep_audio_mask = paired_mask

            # We have already handled audio dropout explicitly above.
            # Keep trajectory dropout controlled by cond_drop_prob unless a stage
            # below overrides it.
            cond_drop_prob_for_model = 0.0
        else:
            cond_drop_prob_for_model = cond_drop_prob

        if self.force_audio_only_drop and isinstance(cond, dict):
            # Stage-wise control adaptation:
            # drop audio completely but keep trajectory/keyframe conditions.
            keep_audio_mask = torch.zeros(
                (batch_size,),
                dtype=torch.bool,
                device=device,
            )
            keep_traj_mask = torch.ones(
                (batch_size,),
                dtype=torch.bool,
                device=device,
            )
            cond_drop_prob_for_model = 0.0

        model_out = self.model(
            x_noisy,
            cond,
            t,
            cond_drop_prob=cond_drop_prob_for_model,
            force_mask=force_mask,
            force_x_clean=force_value,
            keep_audio_mask=keep_audio_mask,
            keep_traj_mask=keep_traj_mask,
        )

        if self.predict_epsilon:
            target = noise
            pred = model_out
            model_motion_x0 = self.predict_start_from_noise(x_noisy, t, model_out)
            target_motion_x0 = x_start
            recon_loss = self._loss_per_sample(pred, target).mean(dim=(1, 2))
            recon_loss = self._p2_apply(recon_loss, t).mean()
        else:
            model_motion_x0 = model_out
            target_motion_x0 = x_start
            recon_loss = self._reconstruction_loss(
                model_motion_x0,
                target_motion_x0,
                t,
            )

        # Optional direct x0 reconstruction loss for strict reconstruction / single-unit overfit.
        # Normal EDGE training optimizes epsilon prediction. For sanity reconstruction we also
        # need the predicted clean motion x0 to match the training motion directly.
        x0_recon_loss = model_motion_x0.new_tensor(0.0)
        if os.environ.get("EDGE_X0_RECON_LOSS", "0").lower() in {"1", "true", "yes", "on"}:
            x0_recon_raw = F.mse_loss(model_motion_x0, target_motion_x0)
            x0_recon_loss = float(os.environ.get("EDGE_X0_RECON_LOSS_WEIGHT", "1.0")) * x0_recon_raw

        # Main temporal smoothness loss.
        velocity_loss = self._velocity_loss(
            model_motion_x0,
            target_motion_x0,
            t,
        )

        # Raw auxiliary losses before warmup / weighting.
        keyframe_loss = self._keyframe_loss(model_motion_x0, train_constraint)

        traj_pos_loss, traj_vel_loss = self._trajectory_training_loss(
            model_motion_x0,
            cond,
            t,
        )

        if x_start.shape[-1] == 151:
            contact_loss = self._contact_loss(
                model_motion_x0,
                target_motion_x0,
            )
            fk_loss = self._fk_loss(model_motion_x0, target_motion_x0)

            # Important fix:
            # pass target_motion_x0 so training foot-lock uses target contacts
            # instead of relying only on predicted contacts.
            foot_loss = self._foot_sliding_loss(
                model_motion_x0,
                target_motion_x0,
            )

            anti_freeze_loss = self._anti_freeze_loss(model_motion_x0)
            sync_loss = self._kinematic_sync_loss(model_motion_x0, x_start, cond)
            biomech_loss = self._biomech_loss(model_motion_x0)
            root_turn_loss = self._root_turn_loss(model_motion_x0)
            contact_turn_loss = self._contact_turn_loss(model_motion_x0)
            body_stability_loss = self._body_stability_loss(model_motion_x0)
            motion_energy_loss = self._motion_energy_loss(
                model_motion_x0,
                target_motion_x0,
            )
        else:
            zero = x_start.new_tensor(0.0)
            contact_loss = zero
            fk_loss = zero
            foot_loss = zero
            anti_freeze_loss = zero
            sync_loss = zero
            biomech_loss = zero
            root_turn_loss = zero
            contact_turn_loss = zero
            body_stability_loss = zero
            motion_energy_loss = zero

        mmr_loss = self._mmr_loss(model_motion_x0, cond)

        # ------------------------------------------------------------
        # Warmup schedules
        # ------------------------------------------------------------
        control_w = self._linear_warmup(current_epoch, start=1, end=30)
        physical_w = self._linear_warmup(current_epoch, start=5, end=80)
        biomech_w = physical_w * self._linear_warmup(
            current_epoch,
            start=100,
            end=200,
        )

        # ------------------------------------------------------------
        # Weighted loss terms.
        # Keep tuple order compatible with EDGE._loss_keys().
        # ------------------------------------------------------------
        recon_term = recon_loss
        velocity_term = 3.0 * velocity_loss

        trajectory_term = control_w * (
            float(self.trajectory_loss_weight) * traj_pos_loss
            + float(self.trajectory_velocity_loss_weight) * traj_vel_loss
        )
        keyframe_term = (
            control_w
            * float(self.keyframe_loss_weight)
            * keyframe_loss
        )

        contact_term = (
            physical_w
            * float(self.contact_loss_weight)
            * contact_loss
        )
        fk_term = physical_w * 0.15 * fk_loss
        foot_term = (
            physical_w
            * float(self.foot_loss_weight)
            * foot_loss
        )
        anti_freeze_term = physical_w * 0.05 * anti_freeze_loss
        # Root-lower coupling is a control-adapter objective, not a late physical regularizer.
        # Use a short warmup so Stage-A receives coupling gradients early.
        root_lower_w = self._linear_warmup(current_epoch, start=1, end=5)
        sync_term = root_lower_w * float(self.sync_loss_weight) * sync_loss

        biomech_term = biomech_w * 0.02 * biomech_loss
        root_turn_term = physical_w * 0.01 * root_turn_loss
        contact_turn_term = physical_w * 0.02 * contact_turn_loss
        body_stability_term = physical_w * 0.05 * body_stability_loss
        motion_energy_term = physical_w * 0.05 * motion_energy_loss

        # Audio-motion contrastive/MMR loss should only be non-zero when caller
        # has enabled it; EDGE.__init__ already disables it for unpaired modes.
        mmr_term = control_w * float(self.mmr_loss_weight) * mmr_loss

        losses = self._make_loss_tuple(
            recon_loss=recon_term,
            velocity_loss=velocity_term,
            contact_loss=contact_term,
            fk_loss=fk_term,
            foot_loss=foot_term,
            anti_freeze_loss=anti_freeze_term,
            mmr_loss=mmr_term,
            trajectory_loss=trajectory_term,
            keyframe_loss=keyframe_term,
            sync_loss=sync_term,
            biomech_loss=biomech_term,
            root_turn_loss=root_turn_term,
            contact_turn_loss=contact_turn_term,
            body_stability_loss=body_stability_term,
            motion_energy_loss=motion_energy_term,
        )

        total_loss = sum(losses) + x0_recon_loss

        if (
            os.environ.get("EDGE_X0_RECON_LOSS", "0").lower() in {"1", "true", "yes", "on"}
            and os.environ.get("EDGE_X0_RECON_LOSS_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
        ):
            try:
                if not hasattr(self, "_edge_x0_debug_counter"):
                    self._edge_x0_debug_counter = 0
                self._edge_x0_debug_counter += 1
                if self._edge_x0_debug_counter <= 20 or self._edge_x0_debug_counter % 100 == 0:
                    print(
                        f"🧪 EDGE_X0_RECON_LOSS_DEBUG "
                        f"step={self._edge_x0_debug_counter} "
                        f"x0_weighted={float(x0_recon_loss.detach().mean().cpu()):.6f} "
                        f"total={float(total_loss.detach().mean().cpu()):.6f}"
                    )
            except Exception:
                pass

        total_loss = torch.nan_to_num(
            total_loss,
            nan=0.0,
            posinf=1e4,
            neginf=0.0,
        )
        losses = tuple(
            torch.nan_to_num(item, nan=0.0, posinf=1e4, neginf=0.0)
            if torch.is_tensor(item)
            else item
            for item in losses
        )

        return total_loss, losses

    def forward(self, x, cond, current_epoch=None, constraint=None):
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.n_timestep, (b,), device=device).long()
        return self.p_losses(x, cond, t, current_epoch=current_epoch, constraint=constraint)

    # ---------------------------------------------------------------------
    # TTO / inference guidance
    # ---------------------------------------------------------------------

    def _should_run_tto(self, use_tto, cond, constraint, t):
        if not use_tto:
            return False

        has_traj = isinstance(cond, dict) and cond.get("trajectory", None) is not None
        has_keyframes = (
            constraint is not None
            and constraint.get("mask", None) is not None
            and constraint.get("value", None) is not None
        )

        if not has_traj and not has_keyframes:
            return False

        time_value = int(t[0].item())
        if time_value > int(self.n_timestep * 0.75):
            return False
        if time_value < int(self.n_timestep * 0.05):
            return False

        return time_value % max(1, int(self.tto_interval)) == 0

    def _trajectory_target_to_physical(self, target_traj_norm, dtype, device):
        target_traj_norm = target_traj_norm.to(device=device, dtype=dtype)[..., :2]

        if self.normalizer is None or not hasattr(self.normalizer, "mean"):
            return target_traj_norm

        mean_x = target_traj_norm.new_tensor(self.normalizer.mean[self.root_x_idx])
        mean_z = target_traj_norm.new_tensor(self.normalizer.mean[self.root_z_idx])
        std_x = target_traj_norm.new_tensor(self.normalizer.std[self.root_x_idx])
        std_z = target_traj_norm.new_tensor(self.normalizer.std[self.root_z_idx])

        target_traj = target_traj_norm.clone()
        target_traj[..., 0] = target_traj_norm[..., 0] * std_x + mean_x
        target_traj[..., 1] = target_traj_norm[..., 1] * std_z + mean_z
        return target_traj

    def _tto_loss(self, pred_xstart, cond, constraint=None):
        loss = pred_xstart.new_tensor(0.0)

        physical_xstart = maybe_unnormalize(self.normalizer, pred_xstart)

        if physical_xstart.shape[-1] == 151:
            root_xz = physical_xstart[:, :, [self.root_x_idx, self.root_z_idx]]
        else:
            root_xz = physical_xstart[:, :, [0, 2]]

        if isinstance(cond, dict) and cond.get("trajectory", None) is not None:
            target_traj_norm = cond["trajectory"].to(
                device=pred_xstart.device,
                dtype=pred_xstart.dtype,
            )

            if target_traj_norm.shape[1] != root_xz.shape[1]:
                target_traj_norm = F.interpolate(
                    target_traj_norm.transpose(1, 2),
                    size=root_xz.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            target_traj = self._trajectory_target_to_physical(
                target_traj_norm,
                dtype=pred_xstart.dtype,
                device=pred_xstart.device,
            )

            traj_loss = F.mse_loss(root_xz, target_traj)
            traj_velocity_loss = F.mse_loss(
                root_xz[:, 1:] - root_xz[:, :-1],
                target_traj[:, 1:] - target_traj[:, :-1],
            )

            loss = (
                loss
                + float(self.tto_trajectory_loss_weight) * traj_loss
                + float(self.tto_trajectory_velocity_loss_weight) * traj_velocity_loss
            )

        if self.beat_guidance_weight > 0 and isinstance(cond, dict):
            onset_cond = cond.get("onset", None)

            if onset_cond is not None:
                onset_curve = onset_cond.to(device=pred_xstart.device, dtype=pred_xstart.dtype)
                if onset_curve.shape[1] != pred_xstart.shape[1]:
                    onset_curve = F.interpolate(
                        onset_curve.transpose(1, 2),
                        size=pred_xstart.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                onset = onset_curve[:, 1:, 0].clamp_min(0.0)
            else:
                audio_feat = cond.get("audio", None)
                onset = None
                if audio_feat is not None and audio_feat.shape[-1] > 768 and pred_xstart.shape[1] > 2:
                    audio_feat = audio_feat.to(device=pred_xstart.device, dtype=pred_xstart.dtype)
                    if audio_feat.shape[1] != pred_xstart.shape[1]:
                        audio_feat = F.interpolate(
                            audio_feat.transpose(1, 2),
                            size=pred_xstart.shape[1],
                            mode="linear",
                            align_corners=False,
                        ).transpose(1, 2)
                    onset = audio_feat[:, 1:, 768].clamp_min(0.0)

            if onset is not None and pred_xstart.shape[1] > 2:
                onset = onset / onset.amax(dim=1, keepdim=True).clamp_min(1e-6)

                root_delta = physical_xstart[:, 1:, self.root_slice] - physical_xstart[:, :-1, self.root_slice]

                if physical_xstart.shape[-1] == 151:
                    pose_delta = physical_xstart[:, 1:, self.rot_slice] - physical_xstart[:, :-1, self.rot_slice]
                else:
                    pose_delta = physical_xstart[:, 1:] - physical_xstart[:, :-1]

                root_energy = safe_norm(root_delta, dim=-1)
                pose_energy = safe_norm(pose_delta, dim=-1)

                root_energy = root_energy / root_energy.amax(dim=1, keepdim=True).clamp_min(1e-6)
                pose_energy = pose_energy / pose_energy.amax(dim=1, keepdim=True).clamp_min(1e-6)

                motion_energy = 0.35 * root_energy + 0.65 * pose_energy
                motion_energy = motion_energy / motion_energy.amax(dim=1, keepdim=True).clamp_min(1e-6)

                loss = loss + float(self.beat_guidance_weight) * F.mse_loss(motion_energy, onset)

        if constraint is not None and constraint.get("mask", None) is not None and constraint.get("value", None) is not None:
            key_loss = self._keyframe_loss(pred_xstart, constraint)
            loss = loss + float(self.keyframe_loss_weight) * key_loss

        if root_xz.shape[1] > 2:
            root_acc = root_xz[:, 2:] - 2.0 * root_xz[:, 1:-1] + root_xz[:, :-2]
            loss = loss + float(self.tto_root_acc_loss_weight) * root_acc.pow(2).mean()

        if pred_xstart.shape[-1] == 151:
            # During TTO there is no ground-truth target motion, so use predicted
            # contacts / fallback contact threshold only.
            foot_loss = self._foot_sliding_loss(pred_xstart)
            loss = loss + float(self.tto_foot_loss_weight) * foot_loss

        return loss

    def _apply_tto(self, x, cond, t, constraint=None):
        x_opt = x.detach()

        for _ in range(max(1, int(self.tto_steps))):
            with torch.enable_grad():
                x_opt = x_opt.detach().requires_grad_(True)

                _, pred_xstart = self.model_predictions(
                    x_opt,
                    cond,
                    t,
                    clip_x_start=False,
                    constraint=constraint,
                )

                tto_loss = self._tto_loss(pred_xstart, cond, constraint)
                grad = torch.autograd.grad(tto_loss, x_opt, allow_unused=True)[0]

                if grad is None:
                    break

                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = safe_norm(grad.flatten(1), dim=1).clamp_min(1e-6)
                grad = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1)))

                x_opt = x_opt - float(self.tto_lr) * grad

        return x_opt.detach()

    def _project_known_keyframes(self, x, constraint, t):
        if (not self.hard_keyframe_project) or constraint is None:
            return x

        if constraint.get("mask", None) is None or constraint.get("value", None) is None:
            return x

        mask = constraint["mask"].to(device=x.device, dtype=x.dtype)
        value = constraint["value"].to(device=x.device, dtype=x.dtype)

        if mask.shape[-1] == 1:
            feature_mask = mask.expand_as(x)
        elif mask.shape[-1] == x.shape[-1]:
            feature_mask = mask
        else:
            raise ValueError(
                f"constraint mask last dim must be 1 or {x.shape[-1]}, got {mask.shape[-1]}"
            )

        known_t = self.q_sample(value, t)
        clean_t = (t == 0).to(dtype=x.dtype).reshape(
            x.shape[0],
            *((1,) * (x.ndim - 1)),
        )
        known_t = known_t * (1.0 - clean_t) + value * clean_t

        return x * (1.0 - feature_mask) + known_t * feature_mask

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def p_sample(self, x, cond, t, constraint=None, use_tto=True):
        b = x.shape[0]

        if self._should_run_tto(use_tto, cond, constraint, t):
            x = self._apply_tto(x, cond, t, constraint=constraint)

        with torch.no_grad():
            model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
                x,
                cond,
                t,
                constraint=constraint,
            )

            noise = torch.randn_like(model_mean)
            nonzero_mask = (1.0 - (t == 0).float()).reshape(
                b,
                *((1,) * (len(noise.shape) - 1)),
            )

            x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
            x_out = self._project_known_keyframes(x_out, constraint, t)

        return x_out, pred_xstart

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
        use_tto=True,
    ):
        device = self.betas.device
        batch_size = shape[0]
        start_point = self.n_timestep if start_point is None else int(start_point)

        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)

        diffusion = [x] if return_diffusion else None

        for i in tqdm(reversed(range(0, start_point)), total=start_point, desc="ddpm sampling"):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(
                x,
                cond,
                timesteps,
                constraint=constraint,
                use_tto=use_tto,
            )

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion

        return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, constraint=None, sampling_timesteps=50, eta=0.0, **kwargs):
        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.n_timestep

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        for time, time_next in tqdm(time_pairs, desc="ddim sampling"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(
                x,
                cond,
                time_cond,
                clip_x_start=self.clip_denoised,
                constraint=constraint,
            )

            if time_next < 0:
                x = x_start
                final_time_cond = torch.zeros((batch,), device=device, dtype=torch.long)
                x = self._project_known_keyframes(x, constraint, final_time_cond)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)).sqrt()
            c = (1.0 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            next_time_cond = torch.full(
                (batch,),
                max(time_next, 0),
                device=device,
                dtype=torch.long,
            )
            x = self._project_known_keyframes(x, constraint, next_time_cond)

        return x

    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, constraint=None, **kwargs):
        batch = shape[0]

        if batch == 1:
            return self.ddim_sample(shape, cond, constraint=constraint, **kwargs)

        device = self.betas.device
        total_timesteps = self.n_timestep
        sampling_timesteps = int(kwargs.get("sampling_timesteps", 50))
        eta = float(kwargs.get("eta", 0.0))

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        weights = np.clip(
            np.linspace(0.0, self.guidance_weight * 2.0, sampling_timesteps),
            None,
            self.guidance_weight,
        )
        time_pairs = list(zip(times[:-1], times[1:], weights))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        for time, time_next, weight in tqdm(time_pairs, desc="long ddim sampling"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(
                x,
                cond,
                time_cond,
                weight=weight,
                clip_x_start=self.clip_denoised,
                constraint=constraint,
            )

            if time_next < 0:
                x = x_start
                final_time_cond = torch.zeros((batch,), device=device, dtype=torch.long)
                x = self._project_known_keyframes(x, constraint, final_time_cond)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)).sqrt()
            c = (1.0 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            next_time_cond = torch.full(
                (batch,),
                max(time_next, 0),
                device=device,
                dtype=torch.long,
            )
            x = self._project_known_keyframes(x, constraint, next_time_cond)

            if time > 0:
                x[1:, :half] = x[:-1, half:].clone()

        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
        use_tto=True,
    ):
        return self.p_sample_loop(
            shape,
            cond,
            noise=noise,
            constraint=constraint,
            return_diffusion=return_diffusion,
            start_point=start_point,
            use_tto=use_tto,
        )

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
        use_tto=True,
    ):
        batch_size = shape[0]

        if batch_size == 1:
            return self.inpaint_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
                use_tto=use_tto,
            )

        device = self.betas.device
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = move_condition_to_device(cond, device)

        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else int(start_point)
        diffusion = [x] if return_diffusion else None

        for i in tqdm(reversed(range(0, start_point)), total=start_point, desc="long inpaint sampling"):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            x, _ = self.p_sample(
                x,
                cond,
                timesteps,
                constraint=constraint,
                use_tto=use_tto,
            )

            if i > 0:
                x[1:, :half] = x[:-1, half:].clone()

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion

        return x

    @torch.no_grad()
    def conditional_sample(self, shape, cond, constraint=None, *args, horizon=None, **kwargs):
        return self.p_sample_loop(
            shape,
            cond,
            constraint=constraint,
            *args,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Rendering compatibility helper
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        constraint=None,
        render=True,
        **kwargs,
    ):
        """
        Compatibility helper for older EDGE scripts.

        It samples motion, unnormalizes it if a normalizer is provided, saves .npy,
        and optionally renders skeleton video.
        """
        if mode in ["inpaint", "inpainting"]:
            samples = self.inpaint_loop(shape, cond, constraint=constraint, **kwargs)
        elif mode in ["long", "long_ddim"]:
            samples = self.long_ddim_sample(shape, cond, constraint=constraint, **kwargs)
        elif mode == "ddim":
            samples = self.ddim_sample(shape, cond, constraint=constraint, **kwargs)
        else:
            samples = self.p_sample_loop(shape, cond, constraint=constraint, **kwargs)

        motion = samples

        if normalizer is not None:
            motion = normalizer.unnormalize(motion)
            if isinstance(motion, np.ndarray):
                motion = torch.from_numpy(motion).to(samples.device)

        motion_np = motion.detach().cpu().numpy()

        render_out = Path(render_out)
        render_out.mkdir(parents=True, exist_ok=True)

        stem = name if name is not None else f"sample_{epoch}"
        npy_path = render_out / f"{stem}.npy"
        np.save(npy_path, motion_np)

        if render:
            try:
                for batch_idx in range(motion_np.shape[0]):
                    video_path = render_out / f"{stem}_{batch_idx}.mp4"
                    skeleton_render(
                        motion_np[batch_idx],
                        str(video_path),
                        sound=sound,
                    )
            except Exception as exc:
                print(f"⚠️ skeleton_render failed: {exc}")

        return motion_np