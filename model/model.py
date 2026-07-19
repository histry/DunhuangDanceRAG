from typing import Any, Callable, Optional, Union
import os

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like


ROOT_X_IDX = 4
ROOT_Z_IDX = 6


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_channels, embed_channels * 2),
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        return pos_encoding.chunk(2, dim=-1)


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1.0) * x + shift


class ZeroInitTrajectoryAdapter(nn.Module):
    """ControlNet-like per-layer trajectory adapter.

    The last linear layer is zero-initialized, so the adapter starts as an
    identity residual branch. This is stable for full retraining and also avoids
    trajectory control overwhelming local pose synthesis at the start.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, traj_tokens: Optional[Tensor]) -> Tensor:
        if traj_tokens is None:
            return x
        return x + torch.tanh(self.gate) * self.net(traj_tokens)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x_out)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class FiLMTransformerDecoderLayer(nn.Module):
    """EDGE decoder layer with ControlNet-like trajectory adapters."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        # Stage 4: per-layer trajectory adapters.
        self.traj_adapter_self = ZeroInitTrajectoryAdapter(d_model)
        self.traj_adapter_cross = ZeroInitTrajectoryAdapter(d_model)
        self.traj_adapter_ff = ZeroInitTrajectoryAdapter(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        traj_tokens=None,
    ):
        x = tgt
        if self.norm_first:
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x = self.traj_adapter_self(x, traj_tokens)

            x_2 = self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            x = self.traj_adapter_cross(x, traj_tokens)

            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
            x = self.traj_adapter_ff(x, traj_tokens)
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask),
                    self.film1(t),
                )
            )
            x = self.traj_adapter_self(x, traj_tokens)

            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.traj_adapter_cross(x, traj_tokens)

            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
            x = self.traj_adapter_ff(x, traj_tokens)

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x_out = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x_out)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x_out = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x_out)

    def _ff_block(self, x):
        return self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )


class DecoderLayerStack(nn.Module):
    def __init__(self, stack, use_gradient_checkpointing=False):
        super().__init__()
        self.stack = stack
        self.use_gradient_checkpointing = use_gradient_checkpointing

    @staticmethod
    def _checkpoint_layer(layer, x, cond, t, tgt_mask, traj_tokens):
        def custom_forward(x_, cond_, t_, tgt_mask_, traj_tokens_):
            return layer(
                x_,
                cond_,
                t_,
                tgt_mask=tgt_mask_,
                traj_tokens=traj_tokens_,
            )

        return torch_checkpoint(
            custom_forward,
            x,
            cond,
            t,
            tgt_mask,
            traj_tokens,
            use_reentrant=False,
        )

    def forward(self, x, cond, t, tgt_mask=None, traj_tokens=None):
        for layer in self.stack:
            if self.use_gradient_checkpointing and self.training:
                x = self._checkpoint_layer(layer, x, cond, t, tgt_mask, traj_tokens)
            else:
                x = layer(x, cond, t, tgt_mask=tgt_mask, traj_tokens=traj_tokens)
        return x


class RootTrajectoryGenerator(nn.Module):
    """Stage 5 root generator.

    The generator predicts a small residual over the requested normalized X/Z
    trajectory. The local diffusion decoder then only needs to synthesize pose,
    contact, root height, and a small root-X/Z residual instead of discovering
    the global route from scratch.
    """

    def __init__(self, d_model: int, residual_scale: float = 0.15):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, trajectory_abs: Tensor, traj_tokens: Tensor) -> Tensor:
        residual = self.net(traj_tokens)
        return trajectory_abs + self.residual_scale * residual


class DanceDecoder(nn.Module):
    """EDGE decoder with stage-4 and stage-5 trajectory control.

    Representation:
      [0:4] contacts
      [4:7] root xyz
      [7:151] 24 joints * 6D rotation

    Trajectory condition:
      cond["trajectory"] is expected to be normalized absolute X/Z, [B,T,2].
      The model internally augments it with ΔX/ΔZ velocity, so training data and
      generation code do not need to store a second trajectory feature.
    """

    def __init__(
        self,
        nfeats: int,
        seq_len: int = 150,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        use_gradient_checkpointing=False,
        use_sparse_attn=False,
        sparse_attn_window=24,
        local_root_residual_scale=0.05,
        **kwargs,
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.seq_len = seq_len
        self.cond_feature_dim = cond_feature_dim
        self.num_heads = num_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_sparse_attn = use_sparse_attn
        self.sparse_attn_window = sparse_attn_window
        self.local_root_residual_scale = float(local_root_residual_scale)

        self.rotary = RotaryEmbedding(dim=latent_dim) if use_rotary else None
        self.abs_pos_encoding = PositionalEncoding(latent_dim, dropout, batch_first=True)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim))
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.null_trajectory_embed = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.traj_type_embed = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # TEA-MotionAdapter:
        # Scalar motion-energy condition. It is added to timestep/global
        # conditioning rather than cross-attention because it is a global
        # continuous control axis.
        self.energy_embed = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.null_energy_embed = nn.Parameter(torch.zeros(1, latent_dim))

        # V9 RAG Summary Token branch.
        # EDGE passes enable_rag_summary_token / rag_summary_dim through **kwargs
        # to preserve old constructor compatibility.
        self.enable_rag_summary_token = bool(kwargs.get("enable_rag_summary_token", False))
        self.rag_summary_dim = int(kwargs.get("rag_summary_dim", 7))
        self.rag_summary_drop_prob = float(kwargs.get("rag_summary_drop_prob", 0.15))
        if self.enable_rag_summary_token:
            self.rag_summary_projection = nn.Sequential(
                nn.Linear(self.rag_summary_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.null_rag_summary_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))
            self.rag_type_embed = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        expanded_input_dim = nfeats * 2 + 1
        self.input_projection = nn.Linear(expanded_input_dim, latent_dim)

        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.cond_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Stage 2/3:
        # trajectory_abs + trajectory_velocity -> trajectory encoder/memory tokens.
        self.trajectory_projection = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.trajectory_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(2)
            ]
        )

        self.traj_modulate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 2),
        )

        # Stage 5 root generator.
        self.root_generator = RootTrajectoryGenerator(latent_dim, residual_scale=0.15)

        decoderstack = nn.ModuleList(
            [
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
                for _ in range(num_layers)
            ]
        )
        self.seqTransDecoder = DecoderLayerStack(
            decoderstack,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.norm_cond = nn.LayerNorm(latent_dim)
        self.final_layer = nn.Linear(latent_dim, nfeats)

    @staticmethod
    def _trajectory_velocity(trajectory_abs: Tensor) -> Tensor:
        vel = torch.zeros_like(trajectory_abs)
        if trajectory_abs.shape[1] > 1:
            vel[:, 1:] = trajectory_abs[:, 1:] - trajectory_abs[:, :-1]
            vel[:, 0] = vel[:, 1]
        return vel

    def _encode_condition_tokens(self, cond_tokens):
        for layer in self.cond_encoder:
            if self.use_gradient_checkpointing and self.training:
                cond_tokens = torch_checkpoint(
                    lambda x_, m=layer: m(x_),
                    cond_tokens,
                    use_reentrant=False,
                )
            else:
                cond_tokens = layer(cond_tokens)
        return cond_tokens

    def _encode_trajectory_tokens(self, trajectory_tokens):
        for layer in self.trajectory_encoder:
            if self.use_gradient_checkpointing and self.training:
                trajectory_tokens = torch_checkpoint(
                    lambda x_, m=layer: m(x_),
                    trajectory_tokens,
                    use_reentrant=False,
                )
            else:
                trajectory_tokens = layer(trajectory_tokens)
        return trajectory_tokens

    def _project_rag_summary_tokens(self, rag_summary_cond, batch_size, seq_len, device, dtype):
        if not getattr(self, "enable_rag_summary_token", False):
            return None

        if rag_summary_cond is None:
            rag_tokens = self.null_rag_summary_embed.to(device=device, dtype=dtype).expand(
                batch_size, seq_len, -1
            )
            return rag_tokens + self.rag_type_embed.to(device=device, dtype=dtype)

        rag_summary_cond = rag_summary_cond.to(device=device, dtype=dtype)

        # Support [B,D], [B,1,D], or [B,T,D].
        if rag_summary_cond.ndim == 2:
            rag_summary_cond = rag_summary_cond[:, None, :].expand(-1, seq_len, -1)
        elif rag_summary_cond.ndim == 3:
            if rag_summary_cond.shape[1] == 1:
                rag_summary_cond = rag_summary_cond.expand(-1, seq_len, -1)
            elif rag_summary_cond.shape[1] != seq_len:
                rag_summary_cond = F.interpolate(
                    rag_summary_cond.transpose(1, 2),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
        else:
            raise ValueError(
                f"rag_summary must be [B,D], [B,1,D], or [B,T,D], got {tuple(rag_summary_cond.shape)}"
            )

        if rag_summary_cond.shape[-1] != self.rag_summary_dim:
            raise ValueError(
                f"Expected rag_summary dim={self.rag_summary_dim}, got {rag_summary_cond.shape[-1]}"
            )

        rag_tokens = self.rag_summary_projection(rag_summary_cond)
        rag_tokens = self.abs_pos_encoding(rag_tokens)
        rag_tokens = rag_tokens + self.rag_type_embed.to(
            device=rag_tokens.device,
            dtype=rag_tokens.dtype,
        )
        return rag_tokens

    def _prepare_cond_inputs(self, cond_embed, batch_size, seq_len, device, dtype):
        if isinstance(cond_embed, dict):
            audio_cond = cond_embed.get("audio", None)
            trajectory_cond = cond_embed.get("trajectory", None)
            energy_cond = cond_embed.get("energy", None)
            rag_summary_cond = cond_embed.get("rag_summary", None)
        else:
            audio_cond = cond_embed
            trajectory_cond = None
            energy_cond = None
            rag_summary_cond = None

        if audio_cond is None:
            audio_cond = torch.zeros(
                (batch_size, seq_len, self.cond_feature_dim),
                device=device,
                dtype=dtype,
            )
        else:
            audio_cond = audio_cond.to(device=device, dtype=dtype)
            if audio_cond.shape[-1] != self.cond_feature_dim:
                raise ValueError(
                    f"Expected audio feature dim {self.cond_feature_dim}, got {audio_cond.shape[-1]}"
                )
            if audio_cond.shape[1] != seq_len:
                audio_cond = F.interpolate(
                    audio_cond.transpose(1, 2),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        if trajectory_cond is not None:
            trajectory_cond = trajectory_cond.to(device=device, dtype=dtype)
            if trajectory_cond.shape[-1] < 2:
                raise ValueError(
                    f"Trajectory condition must have at least 2 channels, got {trajectory_cond.shape[-1]}"
                )
            trajectory_cond = trajectory_cond[..., :2]
            if trajectory_cond.shape[1] != seq_len:
                trajectory_cond = F.interpolate(
                    trajectory_cond.transpose(1, 2),
                    size=seq_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        if energy_cond is not None:
            energy_cond = energy_cond.to(device=device, dtype=dtype)
            if energy_cond.ndim == 1:
                energy_cond = energy_cond[:, None]
            elif energy_cond.ndim == 2:
                if energy_cond.shape[-1] != 1:
                    energy_cond = energy_cond[:, :1]
            elif energy_cond.ndim == 3:
                energy_cond = energy_cond[..., :1]
                if energy_cond.shape[1] != seq_len:
                    energy_cond = F.interpolate(
                        energy_cond.transpose(1, 2),
                        size=seq_len,
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
            else:
                raise ValueError(f"energy condition must be [B,1] or [B,T,1], got {energy_cond.shape}")
            energy_cond = energy_cond.clamp(0.0, 1.0)

        return audio_cond, trajectory_cond, energy_cond, rag_summary_cond

    def _build_sparse_attn_mask(self, batch_size, seq_len, device, force_mask=None):
        if (not self.use_sparse_attn) or self.sparse_attn_window <= 0:
            return None

        idx = torch.arange(seq_len, device=device)
        base_mask = (idx[None, :] - idx[:, None]).abs() > self.sparse_attn_window
        per_batch_mask = base_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()

        if force_mask is not None:
            # Soft keyframes should still become globally visible.
            keyframe_mask = force_mask.amax(dim=-1) > 0.05
            for batch_index in range(batch_size):
                keyframes = keyframe_mask[batch_index]
                if torch.any(keyframes):
                    per_batch_mask[batch_index, :, keyframes] = False
                    per_batch_mask[batch_index, keyframes, :] = False

        return per_batch_mask.repeat_interleave(self.num_heads, dim=0)

    def _resize_null_cond_embed(self, target_len):
        null_cond_embed = self.null_cond_embed
        if null_cond_embed.shape[1] == target_len:
            return null_cond_embed
        return F.interpolate(
            null_cond_embed.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def guided_forward(
        self,
        x,
        cond_embed,
        times,
        guidance_weight,
        force_mask=None,
        force_x_clean=None,
    ):
        """Classifier-free guidance with an optional separate energy axis."""
        b = x.shape[0]
        device = x.device
        drop_all = torch.zeros((b,), dtype=torch.bool, device=device)
        keep_all = torch.ones((b,), dtype=torch.bool, device=device)

        unc = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=1.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=drop_all,
            keep_traj_mask=drop_all,
            keep_energy_mask=drop_all,
        )

        try:
            energy_scale = float(os.environ.get("EDGE_ENERGY_CFG_SCALE", "0"))
        except Exception:
            energy_scale = 0.0

        has_energy = isinstance(cond_embed, dict) and cond_embed.get("energy", None) is not None

        if energy_scale > 0.0 and has_energy:
            base = self.forward(
                x,
                cond_embed,
                times,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
                keep_audio_mask=keep_all,
                keep_traj_mask=keep_all,
                keep_energy_mask=drop_all,
            )
            energy_cond = self.forward(
                x,
                cond_embed,
                times,
                cond_drop_prob=0.0,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
                keep_audio_mask=keep_all,
                keep_traj_mask=keep_all,
                keep_energy_mask=keep_all,
            )
            return unc + (base - unc) * guidance_weight + (energy_cond - base) * energy_scale

        conditioned = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=0.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=keep_all,
            keep_traj_mask=keep_all,
            keep_energy_mask=keep_all,
        )

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self,
        x: Tensor,
        cond_embed: Any,
        times: Tensor,
        cond_drop_prob: float = 0.0,
        force_mask: Optional[Tensor] = None,
        force_x_clean: Optional[Tensor] = None,
        keep_audio_mask: Optional[Tensor] = None,
        keep_traj_mask: Optional[Tensor] = None,
        keep_energy_mask: Optional[Tensor] = None,
        keep_rag_mask: Optional[Tensor] = None,
    ):
        batch_size, seq_len, _, device = *x.shape, x.device

        if force_mask is None:
            force_mask = torch.zeros((batch_size, seq_len, 1), device=device, dtype=x.dtype)
        if force_x_clean is None:
            force_x_clean = torch.zeros_like(x)

        force_mask = force_mask.to(device=device, dtype=x.dtype)
        force_x_clean = force_x_clean.to(device=device, dtype=x.dtype)

        if force_mask.shape[-1] == 1:
            feature_force_mask = force_mask.expand_as(x)
            force_indicator = force_mask
        elif force_mask.shape[-1] == x.shape[-1]:
            feature_force_mask = force_mask
            force_indicator = force_mask.amax(dim=-1, keepdim=True)
        else:
            raise ValueError(
                f"force_mask last dim must be 1 or {x.shape[-1]}, got {force_mask.shape[-1]}"
            )

        # Soft keyframe fix:
        # The mask strength should indicate confidence, not shrink the clean pose.
        presence_mask = (feature_force_mask > 1e-6).to(dtype=x.dtype)
        masked_x_clean = force_x_clean * presence_mask

        x_concat = torch.cat([x, masked_x_clean, force_indicator], dim=-1)
        x = self.input_projection(x_concat)
        x = self.abs_pos_encoding(x)

        audio_cond, trajectory_abs, energy_cond, rag_summary_cond = self._prepare_cond_inputs(
            cond_embed,
            batch_size,
            seq_len,
            device,
            x.dtype,
        )

        cond_drop_prob = float(max(0.0, min(1.0, cond_drop_prob)))
        keep_prob = 1.0 - cond_drop_prob

        if keep_audio_mask is None:
            keep_audio_mask = prob_mask_like((batch_size,), keep_prob, device=device)
        if keep_traj_mask is None:
            keep_traj_mask = prob_mask_like((batch_size,), keep_prob, device=device)

        if keep_energy_mask is None:
            energy_drop_prob = cond_drop_prob
            try:
                energy_drop_prob = float(os.environ.get("EDGE_ENERGY_DROP_PROB", energy_drop_prob))
            except Exception:
                pass
            energy_keep_prob = 1.0 - max(0.0, min(1.0, energy_drop_prob))
            keep_energy_mask = prob_mask_like((batch_size,), energy_keep_prob, device=device)

        keep_audio_mask_embed = rearrange(keep_audio_mask, "b -> b 1 1")
        keep_audio_mask_hidden = rearrange(keep_audio_mask, "b -> b 1")
        keep_traj_mask_embed = rearrange(keep_traj_mask, "b -> b 1 1")
        keep_traj_mask_root = rearrange(keep_traj_mask, "b -> b 1")
        if keep_rag_mask is None:
            keep_rag_mask = prob_mask_like((batch_size,), keep_prob, device=device)
        keep_rag_mask_embed = rearrange(keep_rag_mask, "b -> b 1 1")

        keep_energy_mask_hidden = rearrange(keep_energy_mask, "b -> b 1")

        rag_tokens = self._project_rag_summary_tokens(
            rag_summary_cond,
            batch_size,
            seq_len,
            device,
            x.dtype,
        )
        if rag_tokens is not None:
            null_rag_tokens = self.null_rag_summary_embed.to(
                device=rag_tokens.device,
                dtype=rag_tokens.dtype,
            ).expand_as(rag_tokens)
            rag_tokens = torch.where(keep_rag_mask_embed, rag_tokens, null_rag_tokens)

        cond_tokens = self.cond_projection(audio_cond)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self._encode_condition_tokens(cond_tokens)

        null_cond_embed = self._resize_null_cond_embed(cond_tokens.shape[1]).to(
            device=cond_tokens.device,
            dtype=cond_tokens.dtype,
        )
        cond_tokens = torch.where(keep_audio_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        null_cond_hidden = self.null_cond_hidden.to(device=t.device, dtype=t.dtype)
        cond_hidden = torch.where(keep_audio_mask_hidden, cond_hidden, null_cond_hidden)

        energy_tokens = None
        if energy_cond is None:
            energy_hidden = self.null_energy_embed.to(device=t.device, dtype=t.dtype).expand(batch_size, -1)
        else:
            energy_in = energy_cond.to(device=t.device, dtype=t.dtype)
            if energy_in.ndim == 3:
                energy_tokens = self.energy_embed(energy_in)
                energy_hidden = energy_tokens.mean(dim=1)
            else:
                energy_hidden = self.energy_embed(energy_in)
            null_energy_hidden = self.null_energy_embed.to(device=t.device, dtype=t.dtype).expand_as(energy_hidden)
            energy_hidden = torch.where(keep_energy_mask_hidden, energy_hidden, null_energy_hidden)
            if energy_tokens is not None:
                energy_tokens = torch.where(
                    rearrange(keep_energy_mask, "b -> b 1 1"),
                    energy_tokens,
                    self.null_energy_embed.to(device=t.device, dtype=t.dtype).view(1, 1, -1).expand_as(energy_tokens),
                )

        trajectory_tokens = None
        root_path = None

        if trajectory_abs is not None:
            trajectory_vel = self._trajectory_velocity(trajectory_abs)
            trajectory_feat = torch.cat([trajectory_abs, trajectory_vel], dim=-1)

            trajectory_tokens = self.trajectory_projection(trajectory_feat)
            trajectory_tokens = self.abs_pos_encoding(trajectory_tokens)
            trajectory_tokens = self._encode_trajectory_tokens(trajectory_tokens)

            null_traj_embed = self.null_trajectory_embed.to(
                device=trajectory_tokens.device,
                dtype=trajectory_tokens.dtype,
            ).expand_as(trajectory_tokens)

            trajectory_tokens = torch.where(
                keep_traj_mask_embed,
                trajectory_tokens,
                null_traj_embed,
            )

            scale_shift = self.traj_modulate(trajectory_tokens)
            scale, shift = scale_shift.chunk(2, dim=-1)
            fused_audio_tokens = cond_tokens * (1.0 + scale) + shift

            traj_memory_tokens = trajectory_tokens + self.traj_type_embed.to(
                device=trajectory_tokens.device,
                dtype=trajectory_tokens.dtype,
            )

            trajectory_for_root = torch.where(
                keep_traj_mask_embed,
                trajectory_abs,
                torch.zeros_like(trajectory_abs),
            )
            root_path = self.root_generator(trajectory_for_root, trajectory_tokens)

            t = t + cond_hidden + energy_hidden
            memory_parts = [fused_audio_tokens, traj_memory_tokens]
            if energy_tokens is not None:
                memory_parts.append(energy_tokens)
            if rag_tokens is not None:
                memory_parts.append(rag_tokens)
            memory_parts.append(t_tokens)
            memory = torch.cat(tuple(memory_parts), dim=-2)
        else:
            t = t + cond_hidden + energy_hidden
            memory_parts = [cond_tokens]
            if energy_tokens is not None:
                memory_parts.append(energy_tokens)
            if rag_tokens is not None:
                memory_parts.append(rag_tokens)
            memory_parts.append(t_tokens)
            memory = torch.cat(tuple(memory_parts), dim=-2)

        memory = self.norm_cond(memory)

        self_attn_mask = self._build_sparse_attn_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            force_mask=force_mask,
        )

        decoded = self.seqTransDecoder(
            x,
            memory,
            t,
            tgt_mask=self_attn_mask,
            traj_tokens=trajectory_tokens,
        )

        output = self.final_layer(decoded)

        # Stage 5:
        # When trajectory is present, root X/Z comes mainly from the root
        # trajectory generator and the diffusion head supplies only a small local
        # residual. When trajectory is dropped for CFG training/inference, do not
        # leak the target path: keep the raw diffusion root prediction.
        if root_path is not None and output.shape[-1] >= 151:
            output = output.clone()
            keep_root = keep_traj_mask_root.to(device=output.device, dtype=output.dtype)
            cond_root_x = (
                root_path[:, :, 0]
                + self.local_root_residual_scale * output[:, :, ROOT_X_IDX]
            )
            cond_root_z = (
                root_path[:, :, 1]
                + self.local_root_residual_scale * output[:, :, ROOT_Z_IDX]
            )

            output[:, :, ROOT_X_IDX] = (
                keep_root * cond_root_x
                + (1.0 - keep_root) * output[:, :, ROOT_X_IDX]
            )
            output[:, :, ROOT_Z_IDX] = (
                keep_root * cond_root_z
                + (1.0 - keep_root) * output[:, :, ROOT_Z_IDX]
            )

        return output
