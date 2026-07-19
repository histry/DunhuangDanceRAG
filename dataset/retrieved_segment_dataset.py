"""Dataset for retrieved segment imitation fine-tuning.

Each sample returns:
    x:    target normalized motion [T,151]
    cond: audio/trajectory plus retrieved_prior_mask/value and segment_mask

The target is still the real Dunhuang motion.  The retrieved prior is only a
conditioning signal over a randomly chosen middle segment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from rag_segment_utils import (
    build_segment_mask,
    crop_or_pad_center,
    load_jsonl,
    load_motion_151,
    normalize_motion,
    normalize_root_to_start,
    random_segment,
)


class RetrievedSegmentDataset(Dataset):
    def __init__(
        self,
        pairs_jsonl: str,
        normalizer=None,
        seq_len: int = 150,
        audio_dim: int = 803,
        segment_min_len: int = 24,
        segment_max_len: int = 60,
        prior_feature_mode: str = "upper",
        protect_width: int = 2,
        seed: int = 1234,
        normalize_root: bool = True,
        include_contact_prior: bool = False,
    ):
        self.rows = load_jsonl(pairs_jsonl)
        if not self.rows:
            raise RuntimeError(f"No rows in pairs_jsonl: {pairs_jsonl}")
        self.normalizer = normalizer
        self.seq_len = int(seq_len)
        self.audio_dim = int(audio_dim)
        self.segment_min_len = int(segment_min_len)
        self.segment_max_len = int(segment_max_len)
        self.prior_feature_mode = str(prior_feature_mode)
        self.protect_width = int(protect_width)
        self.seed = int(seed)
        self.normalize_root = bool(normalize_root)
        self.include_contact_prior = bool(include_contact_prior)
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self):
        return len(self.rows)

    def _load_clip(self, source: str, center: int) -> np.ndarray:
        key = str(source)
        if key not in self._cache:
            motion = load_motion_151(source)
            if self.normalize_root:
                motion = normalize_root_to_start(motion)
            self._cache[key] = motion.astype(np.float32)
        return crop_or_pad_center(self._cache[key], int(center), self.seq_len)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rng = np.random.default_rng(self.seed + idx * 9973)
        seg_start, seg_end = random_segment(
            self.seq_len,
            self.segment_min_len,
            self.segment_max_len,
            rng,
        )

        target = self._load_clip(row["target_source"], int(row["target_center"]))
        prior = self._load_clip(row["prior_source"], int(row["prior_center"]))

        target_norm = normalize_motion(target, self.normalizer)
        prior_norm = normalize_motion(prior, self.normalizer)

        prior_mask = build_segment_mask(
            self.seq_len,
            seg_start,
            seg_end,
            feature_mode=self.prior_feature_mode,
            protect_width=self.protect_width,
            include_contacts=self.include_contact_prior,
        )
        segment_mask = np.zeros((self.seq_len, 1), dtype=np.float32)
        segment_mask[seg_start:seg_end] = 1.0
        if self.protect_width > 0:
            segment_mask[: self.protect_width] = 0.0
            segment_mask[self.seq_len - self.protect_width :] = 0.0

        x = torch.from_numpy(target_norm).float()
        prior_value = torch.from_numpy(prior_norm * prior_mask).float()
        prior_mask_t = torch.from_numpy(prior_mask).float()
        segment_mask_t = torch.from_numpy(segment_mask).float()

        audio = torch.zeros((self.seq_len, self.audio_dim), dtype=torch.float32)
        onset = torch.zeros((self.seq_len, 1), dtype=torch.float32)
        trajectory = x[:, [4, 6]].float()

        cond = {
            "audio": audio,
            "audio_paired": torch.tensor(0.0, dtype=torch.float32),
            "onset": onset,
            "trajectory": trajectory,
            "retrieved_prior_value": prior_value,
            "retrieved_prior_mask": prior_mask_t,
            "retrieved_segment_mask": segment_mask_t,
            "retrieved_prior_weight": torch.tensor(1.0, dtype=torch.float32),
        }
        name = f"{Path(row['target_source']).stem}_{int(row['target_center']):06d}"
        wav = ""
        return x, cond, name, wav
