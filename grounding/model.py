#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-branch structure-guided grounding for low-resource MSSD--AESD routing.

The design follows a coherent two-branch principle:
- semantic branch: MSSD/AESD, hierarchy, stage and posture;
- geometry branch: intrinsic SO(3) flow and anatomy-gated event dynamics.
A structure-guided gate fuses both branches.  Training uses multi-positive
contrastive, hierarchy, source-invariance and shuffled-geometry objectives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from events.semantic_descriptor import (
    MUSIC_SEMANTIC_LABELS,
    event_probs_from_fields,
)

SCHEMA = "v46_53_dual_branch_structure_guided_grounding_v1"
POSTURES = ("floor_pose", "kneeling", "deep_squat", "half_squat", "standing", "aerial")
ROLE_HASH_DIM = 8
FAMILY_HASH_DIM = 16
SEMANTIC_DIM = len(MUSIC_SEMANTIC_LABELS) + 2 + len(POSTURES) + ROLE_HASH_DIM + FAMILY_HASH_DIM


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _hash_onehot(value: Any, width: int) -> np.ndarray:
    out = np.zeros(width, dtype=np.float32)
    token = str(value or "unknown").strip().lower()
    h = int(hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    out[h % width] = 1.0
    return out


def _posture_onehot(value: Any) -> np.ndarray:
    out = np.zeros(len(POSTURES), dtype=np.float32)
    token = str(value or "standing")
    out[POSTURES.index(token) if token in POSTURES else 4] = 1.0
    return out


def _arr(db: Mapping[str, Any], key: str, n: int, default: Any) -> np.ndarray:
    if key in db:
        a = np.asarray(db[key])
        if a.ndim >= 1 and a.shape[0] == n:
            return a
    return np.asarray([default] * n, dtype=object)


def _farr(db: Mapping[str, Any], key: str, n: int, default: float) -> np.ndarray:
    if key in db:
        a = np.asarray(db[key], dtype=np.float32)
        if a.ndim >= 1 and a.shape[0] == n:
            return a
    return np.ones(n, dtype=np.float32) * float(default)


def _scalar(db: Mapping[str, Any], key: str) -> Any:
    if key not in db:
        return None
    value = np.asarray(db[key])
    if value.size != 1:
        raise RuntimeError(f"Grounding contract field {key!r} must be scalar")
    scalar = value.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _load_grounder_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Grounder checkpoint is not a mapping: {path}")
    return checkpoint


def _training_geometry_contract(
    db: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if "v46_53_geometry_desc" not in db:
        raise RuntimeError("Run v46_53 event geometry augmentation before grounding")
    schema = _scalar(db, "v46_53_geometry_schema_version")
    fps = _scalar(db, "v46_53_geometry_fps")
    skeleton = _scalar(db, "skeleton_contract_json")
    if schema is None or fps is None or skeleton is None:
        raise RuntimeError(
            "Grounder training database lacks geometry FPS/schema or skeleton contract"
        )
    raw = np.asarray(db["v46_53_geometry_desc"], dtype=np.float32)
    mean = np.asarray(db.get("v46_53_geometry_mean"), dtype=np.float32)
    std = np.asarray(db.get("v46_53_geometry_std"), dtype=np.float32)
    if raw.ndim != 2 or mean.shape != (1, raw.shape[1]) or std.shape != (1, raw.shape[1]):
        raise RuntimeError(
            "Grounder geometry statistics have incompatible shapes: "
            f"raw={raw.shape}, mean={mean.shape}, std={std.shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Grounder geometry/statistics contain NaN or Inf")
    if np.any(std <= 0.0):
        raise RuntimeError("Grounder geometry standard deviations must be positive")
    normalized = ((raw - mean) / std).astype(np.float32)
    contract = {
        "geometry_schema": str(schema),
        "fps": float(fps),
        "skeleton_contract_json": str(skeleton),
        "geometry_dim": int(raw.shape[1]),
    }
    return normalized, mean, std, contract


def _geometry_for_checkpoint(
    db: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> np.ndarray:
    """Transform any split with train-only geometry statistics."""

    if "v46_53_geometry_desc" not in db:
        raise RuntimeError("V46.53 raw geometry descriptor is missing")
    schema = _scalar(db, "v46_53_geometry_schema_version")
    fps = _scalar(db, "v46_53_geometry_fps")
    skeleton = _scalar(db, "skeleton_contract_json")
    expected = checkpoint.get("geometry_contract")
    if not isinstance(expected, Mapping):
        raise RuntimeError(
            "Grounder checkpoint has no train-split geometry contract; rebuild it"
        )
    mismatches = []
    if str(schema) != str(expected.get("geometry_schema")):
        mismatches.append(
            f"geometry_schema: db={schema!r}, checkpoint={expected.get('geometry_schema')!r}"
        )
    try:
        if abs(float(fps) - float(expected.get("fps"))) > 1.0e-6:
            mismatches.append(f"fps: db={fps!r}, checkpoint={expected.get('fps')!r}")
    except (TypeError, ValueError):
        mismatches.append(f"fps: db={fps!r}, checkpoint={expected.get('fps')!r}")
    if str(skeleton) != str(expected.get("skeleton_contract_json")):
        mismatches.append("skeleton_contract_json")
    raw = np.asarray(db["v46_53_geometry_desc"], dtype=np.float32)
    dimension = int(expected.get("geometry_dim", -1))
    if raw.ndim != 2 or raw.shape[1] != dimension:
        mismatches.append(f"geometry_dim: db={raw.shape}, checkpoint={dimension}")
    if mismatches:
        raise RuntimeError("Grounder geometry contract mismatch: " + "; ".join(mismatches))

    mean = np.asarray(checkpoint.get("geometry_train_mean"), dtype=np.float32)
    std = np.asarray(checkpoint.get("geometry_train_std"), dtype=np.float32)
    if mean.shape != (1, dimension) or std.shape != (1, dimension):
        raise RuntimeError(
            "Grounder checkpoint has invalid train statistics: "
            f"mean={mean.shape}, std={std.shape}, dim={dimension}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("Grounder geometry/statistics contain NaN or Inf")
    if np.any(std <= 0.0):
        raise RuntimeError("Grounder train standard deviations must be positive")
    return ((raw - mean) / std).astype(np.float32)


def _normalize_probs(value: Any, top: Optional[str] = None) -> np.ndarray:
    labels = list(MUSIC_SEMANTIC_LABELS)
    out = np.zeros(len(labels), dtype=np.float32)
    if isinstance(value, Mapping):
        for k, v in value.items():
            if str(k) in labels:
                out[labels.index(str(k))] = max(0.0, float(v))
    elif value is not None:
        try:
            a = np.asarray(value, dtype=np.float32).reshape(-1)
            out[: min(len(out), len(a))] = np.maximum(a[: len(out)], 0.0)
        except Exception:
            pass
    if out.sum() <= 1.0e-8 and top in labels:
        out[labels.index(str(top))] = 1.0
    if out.sum() <= 1.0e-8:
        out[:] = 1.0 / len(out)
    else:
        out /= out.sum()
    return out


def event_semantic_matrix(db: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(np.asarray(db["paths"]))
    if "aesd_music_alignment_probs" in db:
        probs = np.asarray(db["aesd_music_alignment_probs"], dtype=np.float32)
    else:
        dance = _arr(db, "dance_keys", n, "unknown")
        family = _arr(db, "event_families", n, "unknown")
        align = _arr(db, "music_alignment_labels", n, "unknown")
        energy = _arr(db, "energy_labels", n, "unknown")
        rhythm = _arr(db, "rhythm_labels", n, "unknown")
        loco = _arr(db, "locomotion_labels", n, "unknown")
        support = _arr(db, "support_labels", n, "unknown")
        quality = _farr(db, "v46_53_combined_quality", n, 0.5)
        sem_conf = _farr(db, "semantic_confidence", n, 0.5)
        desc = np.asarray(db.get("desc", np.zeros((n, 32), np.float32)), dtype=np.float32)
        probs = np.stack([
            event_probs_from_fields(
                dance_key=dance[i],
                event_family=family[i],
                music_alignment_label=align[i],
                energy_label=energy[i],
                rhythm_label=rhythm[i],
                locomotion_label=loco[i],
                support_label=support[i],
                quality=float(quality[i]),
                semantic_confidence=float(sem_conf[i]),
                desc=desc[i] if desc.ndim == 2 else None,
            )
            for i in range(n)
        ]).astype(np.float32)

    durations = _farr(db, "durations", n, 2.0)
    quality = _farr(db, "v46_53_combined_quality", n, 0.5)
    posture = _arr(db, "posture_mode", n, "standing")
    roles = _arr(db, "motion_stage_roles", n, "unknown")
    family = _arr(db, "event_families", n, "unknown")
    features = []
    for i in range(n):
        features.append(np.concatenate([
            _normalize_probs(probs[i]),
            np.asarray([np.clip(durations[i] / 6.0, 0.0, 2.0), quality[i]], np.float32),
            _posture_onehot(posture[i]),
            _hash_onehot(roles[i], ROLE_HASH_DIM),
            _hash_onehot(family[i], FAMILY_HASH_DIM),
        ]))
    top = np.argmax(probs, axis=1).astype(np.int64)
    posture_id = np.asarray([POSTURES.index(str(p)) if str(p) in POSTURES else 4 for p in posture], dtype=np.int64)
    family_id = np.asarray([
        int(np.argmax(_hash_onehot(f, FAMILY_HASH_DIM))) for f in family
    ], dtype=np.int64)
    return np.stack(features).astype(np.float32), top, posture_id, family_id


def slot_semantic_vector(slot: Mapping[str, Any]) -> np.ndarray:
    raw = None
    for key in ("music_semantic_probs", "music_alignment_probs", "probs", "probabilities", "slot_probs"):
        if key in slot:
            raw = slot[key]
            break
    top = str(slot.get("music_semantic_top_label", slot.get("music_alignment_label", slot.get("top_label", ""))))
    probs = _normalize_probs(raw, top=top)
    duration = float(slot.get("duration", float(slot.get("end", 0.0)) - float(slot.get("start", 0.0))))
    role = str(slot.get("role", "normal"))
    preferred = slot.get("preferred_event_families", slot.get("preferred_families", slot.get("preferred_dance_keys", [])))
    family_token = ";".join(map(str, preferred)) if isinstance(preferred, (list, tuple)) else str(preferred)
    posture_hint = str(slot.get("target_posture", "standing"))
    quality = float(slot.get("semantic_confidence", slot.get("confidence", 1.0)))
    return np.concatenate([
        probs,
        np.asarray([np.clip(duration / 6.0, 0.0, 2.0), np.clip(quality, 0.0, 1.0)], np.float32),
        _posture_onehot(posture_hint),
        _hash_onehot(role, ROLE_HASH_DIM),
        _hash_onehot(family_token, FAMILY_HASH_DIM),
    ]).astype(np.float32)




def _poincare_project_torch(x: "torch.Tensor", eps: float = 1.0e-5) -> "torch.Tensor":
    """Project unconstrained vectors to the unit Poincare ball."""
    y = torch.tanh(x)
    norm = torch.linalg.norm(y, dim=-1, keepdim=True).clamp_min(eps)
    max_norm = 1.0 - eps
    return torch.where(norm > max_norm, y / norm * max_norm, y)


def _poincare_distance_torch(x: "torch.Tensor", y: "torch.Tensor", eps: float = 1.0e-5) -> "torch.Tensor":
    x2 = (x * x).sum(dim=-1).clamp_max(1.0 - eps)
    y2 = (y * y).sum(dim=-1).clamp_max(1.0 - eps)
    diff2 = ((x - y) ** 2).sum(dim=-1)
    z = 1.0 + 2.0 * diff2 / ((1.0 - x2) * (1.0 - y2)).clamp_min(eps)
    return torch.acosh(z.clamp_min(1.0 + eps))

def probabilistic_association(slot: Mapping[str, Any], event_probs: np.ndarray) -> float:
    p = slot_semantic_vector(slot)[: len(MUSIC_SEMANTIC_LABELS)]
    q = _normalize_probs(event_probs)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(np.maximum(p, 1e-8) / np.maximum(m, 1e-8))))
    kl_qm = float(np.sum(q * np.log(np.maximum(q, 1e-8) / np.maximum(m, 1e-8))))
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(1.0 - js / math.log(2.0), 0.0, 1.0))


if nn is not None:
    class DualBranchGrounder(nn.Module):
        def __init__(
            self,
            geometry_dim: int,
            semantic_dim: int = SEMANTIC_DIM,
            hidden: int = 192,
            embed: int = 96,
            hyp_dim: int = 32,
        ):
            super().__init__()
            self.geometry_dim = int(geometry_dim)
            self.semantic_dim = int(semantic_dim)
            self.hyp_dim = int(hyp_dim)
            self.geom = nn.Sequential(nn.Linear(geometry_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, embed))
            self.sem = nn.Sequential(nn.Linear(semantic_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, embed))
            self.slot = nn.Sequential(nn.Linear(semantic_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, embed))
            # The hierarchy branch is genuinely non-Euclidean: AESD family, stage
            # and posture features are embedded in a Poincare ball.
            self.hierarchy = nn.Sequential(nn.Linear(semantic_dim, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hyp_dim))
            self.hyp_to_embed = nn.Linear(hyp_dim, embed)
            self.gate = nn.Sequential(nn.Linear(embed * 3 + 1, embed), nn.GELU(), nn.Linear(embed, embed), nn.Sigmoid())
            self.jigsaw = nn.Sequential(nn.Linear(embed, embed // 2), nn.GELU(), nn.Linear(embed // 2, 1))

        def encode_hierarchy(self, semantic: torch.Tensor) -> torch.Tensor:
            return _poincare_project_torch(self.hierarchy(semantic))

        def encode_event(self, geometry: torch.Tensor, semantic: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
            g = self.geom(geometry)
            s = self.sem(semantic)
            h_ball = self.encode_hierarchy(semantic)
            h = self.hyp_to_embed(h_ball)
            q = quality.reshape(-1, 1)
            gate = self.gate(torch.cat([g, s, h, q], dim=-1))
            # Structure-guided synergistic fusion: high-quality intrinsic motion
            # receives authority, while semantic/hierarchical branches anchor
            # culture and posture relations.
            semantic_anchor = s + 0.35 * h
            fused = gate * g + (1.0 - gate) * semantic_anchor + 0.15 * g * semantic_anchor
            return F.normalize(fused, dim=-1)

        def encode_slot(self, semantic: torch.Tensor) -> torch.Tensor:
            h = self.hyp_to_embed(self.encode_hierarchy(semantic))
            return F.normalize(self.slot(semantic) + 0.35 * h, dim=-1)
else:  # pragma: no cover
    DualBranchGrounder = object


def _multi_positive_loss(logits: "torch.Tensor", positive: "torch.Tensor", quality: "torch.Tensor") -> "torch.Tensor":
    all_lse = torch.logsumexp(logits, dim=1)
    masked = logits.masked_fill(~positive, -1.0e9)
    pos_lse = torch.logsumexp(masked, dim=1)
    valid = positive.any(dim=1)
    loss = -(pos_lse - all_lse)
    w = quality.clamp(0.1, 1.0)
    return (loss[valid] * w[valid]).sum() / w[valid].sum().clamp_min(1e-6)


def train_grounder(
    db_path: Path,
    out_path: Path,
    steps: int = 1200,
    batch_size: int = 128,
    seed: int = 20260717,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to train V46.53 grounding")
    raw = np.load(db_path, allow_pickle=True)
    db = {k: raw[k] for k in raw.files}
    geometry, geometry_mean, geometry_std, geometry_contract = (
        _training_geometry_contract(db)
    )
    semantic, top, posture, family = event_semantic_matrix(db)
    quality = np.asarray(db.get("v46_53_combined_quality", np.ones(len(geometry), np.float32)), dtype=np.float32)
    sources = np.asarray(db.get("source_uids", np.asarray(["unknown"] * len(geometry), object)), dtype=object)
    source_ids = np.asarray([int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16) % 1024 for s in sources], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() and _env_bool("V46_53_GROUNDER_CUDA", True) else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DualBranchGrounder(
        geometry_dim=geometry.shape[1],
        semantic_dim=semantic.shape[1],
        hidden=_env_int("V46_53_GROUNDER_HIDDEN", 192),
        embed=_env_int("V46_53_GROUNDER_EMBED", 96),
        hyp_dim=_env_int("V46_53_HYPERBOLIC_DIM", 32),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=_env_float("V46_53_GROUNDER_LR", 2e-4), weight_decay=1e-4)

    g_all = torch.from_numpy(geometry).to(device)
    s_all = torch.from_numpy(semantic).to(device)
    q_all = torch.from_numpy(quality).to(device)
    top_t = torch.from_numpy(top).to(device)
    posture_t = torch.from_numpy(posture).to(device)
    family_t = torch.from_numpy(family).to(device)
    source_t = torch.from_numpy(source_ids).to(device)
    n = len(geometry)
    temp = _env_float("V46_53_GROUNDER_TEMP", 0.08)
    history = []

    model.train()
    for step in range(1, int(steps) + 1):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        geom = g_all[idx]
        sem = s_all[idx]
        qual = q_all[idx]
        # Pseudo music slots are noisy semantic views of valid events.  This is
        # appropriate for unpaired learning and avoids pretending filename proxy
        # audio is paired supervision.
        slot_sem = sem + 0.035 * torch.randn_like(sem)
        drop = torch.rand_like(slot_sem) < 0.04
        slot_sem = slot_sem.masked_fill(drop, 0.0)

        event_z = model.encode_event(geom, sem, qual)
        slot_z = model.encode_slot(slot_sem)
        logits = slot_z @ event_z.T / temp
        pos = (top_t[idx, None] == top_t[idx][None, :]) & (
            (posture_t[idx, None] - posture_t[idx][None, :]).abs() <= 1
        )
        pos.fill_diagonal_(True)
        loss_con = _multi_positive_loss(logits, pos, qual)

        sim = event_z @ event_z.T
        same_family = family_t[idx, None] == family_t[idx][None, :]
        same_source = source_t[idx, None] == source_t[idx][None, :]
        offdiag = ~torch.eye(len(idx), device=device, dtype=torch.bool)
        # Explicit Poincare hierarchy: same-family/posture neighbours are pulled
        # together, while unrelated families keep a margin.
        h_ball = model.encode_hierarchy(sem)
        hdist = _poincare_distance_torch(h_ball[:, None, :], h_ball[None, :, :])
        pos_h = same_family & offdiag
        neg_h = (~same_family) & offdiag
        hierarchy_pos = hdist[pos_h].mean() if pos_h.any() else hdist.sum() * 0.0
        hierarchy_neg = F.relu(_env_float("V46_53_HYPERBOLIC_MARGIN", 1.25) - hdist[neg_h]).mean() if neg_h.any() else hdist.sum() * 0.0
        hierarchy_loss = hierarchy_pos + 0.35 * hierarchy_neg
        # Source invariance prevents the shared branch from becoming a dancer ID.
        source_loss = sim[same_source & ~same_family & offdiag].square().mean() if (same_source & ~same_family & offdiag).any() else sim.sum() * 0.0

        perm = torch.randperm(len(idx), device=device)
        shuffled = model.encode_event(geom[perm], sem, qual)
        real_logit = model.jigsaw(event_z).reshape(-1)
        fake_logit = model.jigsaw(shuffled).reshape(-1)
        jigsaw_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
            + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
        )

        # Non-linear decay mirrors a dynamic regularization schedule: strong
        # geometry/semantic consistency early, more discriminative freedom late.
        decay = math.exp(-3.0 * step / max(1, steps))
        consistency = (event_z - model.encode_slot(sem)).square().mean()
        loss = (
            loss_con
            + _env_float("V46_53_HIERARCHY_LOSS_W", 0.20) * hierarchy_loss
            + _env_float("V46_53_SOURCE_INVARIANCE_W", 0.05) * source_loss
            + _env_float("V46_53_JIGSAW_LOSS_W", 0.12) * jigsaw_loss
            + _env_float("V46_53_DYNAMIC_CONSISTENCY_W", 0.18) * decay * consistency
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        if step == 1 or step % max(1, _env_int("V46_53_GROUNDER_LOG_EVERY", 100)) == 0 or step == steps:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "contrastive": float(loss_con.detach().cpu()),
                "hierarchy": float(hierarchy_loss.detach().cpu()),
                "jigsaw": float(jigsaw_loss.detach().cpu()),
                "dynamic_consistency_weight": float(decay),
            }
            history.append(row)
            print("[V46.53 GROUND] " + json.dumps(row, ensure_ascii=False), flush=True)

    model.eval()
    with torch.no_grad():
        event_embed = model.encode_event(g_all, s_all, q_all).cpu().numpy().astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": SCHEMA,
        "state_dict": model.state_dict(),
        "geometry_dim": int(geometry.shape[1]),
        "semantic_dim": int(semantic.shape[1]),
        "hidden": _env_int("V46_53_GROUNDER_HIDDEN", 192),
        "embed": _env_int("V46_53_GROUNDER_EMBED", 96),
        "hyp_dim": _env_int("V46_53_HYPERBOLIC_DIM", 32),
        "music_semantic_labels": list(MUSIC_SEMANTIC_LABELS),
        "history": history,
        "seed": int(seed),
        "geometry_contract": geometry_contract,
        "geometry_train_mean": geometry_mean,
        "geometry_train_std": geometry_std,
    }, out_path)

    payload = dict(db)
    payload["v46_53_grounding_schema_version"] = np.asarray(SCHEMA, dtype=object)
    payload["v46_53_grounding_embedding"] = event_embed
    payload["v46_53_grounder_geometry_z"] = geometry
    payload["v46_53_grounder_normalization"] = np.asarray(
        "train_split_statistics", dtype=object
    )
    backup = db_path.with_name(db_path.stem + ".pre_v46_53_grounding.npz")
    if not backup.exists():
        shutil.copy2(db_path, backup)
    np.savez_compressed(db_path, **payload)

    report = {
        "schema": SCHEMA,
        "db": str(db_path),
        "checkpoint": str(out_path),
        "device": str(device),
        "events": int(n),
        "geometry_dim": int(geometry.shape[1]),
        "semantic_dim": int(semantic.shape[1]),
        "embedding_dim": int(event_embed.shape[1]),
        "steps": int(steps),
        "history": history,
        "geometry_contract": geometry_contract,
        "normalization": "train_split_statistics",
        "ok": True,
    }
    out_path.with_suffix(out_path.suffix + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def embed_database(db_path: Path, checkpoint: Path) -> Dict[str, Any]:
    """Embed a non-train split with a grounder learned only on train sources."""
    if torch is None:
        raise RuntimeError("PyTorch is required for V46.53 grounding embedding")
    raw = np.load(db_path, allow_pickle=True)
    db = {k: raw[k] for k in raw.files}
    ckpt = _load_grounder_checkpoint(checkpoint)
    geom = _geometry_for_checkpoint(db, ckpt)
    model = DualBranchGrounder(
        geometry_dim=int(ckpt["geometry_dim"]),
        semantic_dim=int(ckpt["semantic_dim"]),
        hidden=int(ckpt.get("hidden", 192)),
        embed=int(ckpt.get("embed", 96)),
        hyp_dim=int(ckpt.get("hyp_dim", 32)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() and _env_bool("V46_53_GROUNDER_INFER_CUDA", False) else "cpu")
    model.to(device).eval()
    sem, _, _, _ = event_semantic_matrix(db)
    quality = np.asarray(db.get("v46_53_combined_quality", np.ones(len(geom), np.float32)), dtype=np.float32)
    with torch.no_grad():
        event_embed = model.encode_event(
            torch.from_numpy(geom).to(device),
            torch.from_numpy(sem).to(device),
            torch.from_numpy(quality).to(device),
        ).cpu().numpy().astype(np.float32)
    payload = dict(db)
    payload["v46_53_grounding_schema_version"] = np.asarray(SCHEMA, dtype=object)
    payload["v46_53_grounding_embedding"] = event_embed
    payload["v46_53_grounder_geometry_z"] = geom
    payload["v46_53_grounder_normalization"] = np.asarray(
        "train_split_statistics", dtype=object
    )
    np.savez_compressed(db_path, **payload)
    report = {
        "schema": SCHEMA,
        "db": str(db_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "events": int(len(geom)),
        "embedding_dim": int(event_embed.shape[1]),
        "mode": "train-checkpoint embedding only",
        "normalization": "train_split_statistics",
        "ok": True,
    }
    db_path.with_name(db_path.stem + ".v46_53_grounding.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class GroundingRuntime:
    def __init__(self, db: Mapping[str, Any], checkpoint: Optional[str]):
        self.db = db
        self.checkpoint = str(checkpoint or "")
        self.event_probs = np.asarray(
            db.get("aesd_music_alignment_probs", np.zeros((len(db.get("paths", [])), len(MUSIC_SEMANTIC_LABELS)), np.float32)),
            dtype=np.float32,
        )
        self.model = None
        self.event_embedding = np.asarray(db.get("v46_53_grounding_embedding", np.zeros((len(self.event_probs), 1), np.float32)), dtype=np.float32)
        self.device = None
        if torch is not None and self.checkpoint and Path(self.checkpoint).is_file() and "v46_53_geometry_desc" in db:
            ckpt = _load_grounder_checkpoint(Path(self.checkpoint))
            aligned_geometry = _geometry_for_checkpoint(db, ckpt)
            self.model = DualBranchGrounder(
                geometry_dim=int(ckpt["geometry_dim"]),
                semantic_dim=int(ckpt["semantic_dim"]),
                hidden=int(ckpt.get("hidden", 192)),
                embed=int(ckpt.get("embed", 96)),
                hyp_dim=int(ckpt.get("hyp_dim", 32)),
            )
            self.model.load_state_dict(ckpt["state_dict"], strict=True)
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() and _env_bool("V46_53_GROUNDER_INFER_CUDA", False) else "cpu")
            self.model.to(self.device)
            if self.event_embedding.ndim != 2 or self.event_embedding.shape[1] != int(ckpt.get("embed", 96)):
                sem, _, _, _ = event_semantic_matrix(db)
                quality = np.asarray(
                    db.get(
                        "v46_53_combined_quality",
                        np.ones(len(aligned_geometry), np.float32),
                    ),
                    dtype=np.float32,
                )
                with torch.no_grad():
                    self.event_embedding = self.model.encode_event(
                        torch.from_numpy(aligned_geometry).to(self.device),
                        torch.from_numpy(sem).to(self.device),
                        torch.from_numpy(quality).to(self.device),
                    ).cpu().numpy().astype(np.float32)

    def score(self, slot: Mapping[str, Any], event_id: int) -> float:
        i = int(event_id)
        deterministic = probabilistic_association(slot, self.event_probs[i] if i < len(self.event_probs) else None)
        if self.model is None or i >= len(self.event_embedding):
            return deterministic
        sem = slot_semantic_vector(slot)[None]
        with torch.no_grad():
            slot_z = self.model.encode_slot(torch.from_numpy(sem).to(self.device)).cpu().numpy()[0]
        learned = float(np.clip(0.5 + 0.5 * np.dot(slot_z, self.event_embedding[i]), 0.0, 1.0))
        return float(0.45 * deterministic + 0.55 * learned)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--db", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--steps", type=int, default=1200)
    tr.add_argument("--batch_size", type=int, default=128)
    tr.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        report = train_grounder(Path(args.db), Path(args.out), args.steps, args.batch_size, args.seed)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
