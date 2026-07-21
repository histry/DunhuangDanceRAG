import csv
import glob
import hashlib
import json
import os
import pickle
import random
import re
import sys
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from torch.utils.data import Dataset

from dataset.quaternion import ax_to_6v
from dataset.preprocess import Normalizer, vectorize_many
from vis import SMPLSkeleton

# ================= 修复 numpy._core 报错的补丁 =================
import numpy.core
sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
sys.modules["numpy._core.umath"] = numpy.core.umath
# ===============================================================

# 151D representation contract:
# [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotations.
CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
TRAJ_ROOT_XZ_IDXS = [ROOT_X_IDX, ROOT_Z_IDX]
SMPL_POS_XZ_IDXS = [0, 2]
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


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


def _require_xz_trajectory_contract(context: str = "") -> None:
    """Fail fast if somebody tries to train/evaluate in image X/Y plane.

    EDGE 151D motion stores root xyz at dimensions 4,5,6.  All trajectory
    conditions and trajectory losses in this project must refer to the physical
    ground plane: X/Z, not image-plane X/Y.  This dataset layer is the first
    place where root/trajectory supervision is constructed, so it enforces the
    contract before any tensors reach the model.
    """
    plane = _env_str("EDGE_TRAJECTORY_PLANE", "xz").lower().replace("-", "_")
    valid_xz = {"xz", "x_z", "ground", "ground_plane", "ground_xz"}
    if plane not in valid_xz:
        where = f" in {context}" if context else ""
        raise RuntimeError(
            f"EDGE trajectory coordinate contract violation{where}: "
            f"EDGE_TRAJECTORY_PLANE={plane!r}. This project only supports "
            "ground-plane X/Z trajectories. Do not use image-plane X/Y."
        )


def _safe_numeric_token_strip(stem: str) -> str:
    """Best-effort fallback: strip common slicing/window suffixes from a stem.

    Some preprocessing pipelines save overlapping clips as separate .pkl files
    even when they come from the same original video.  Splitting by full .pkl
    path can still leak because neighboring clips from the same source video can
    land in train and validation.  We therefore canonicalize names before group
    split when explicit metadata is missing.
    """
    s = str(stem)
    patterns = [
        r"(_win|_window|_slice|_seg|_segment|_clip)?[_-]?\d{4,7}[_-]\d{4,7}$",
        r"(_win|_window|_slice|_seg|_segment|_clip)[_-]?\d+$",
        r"[_-](start|s)\d+[_-](end|e)\d+$",
        r"[_-]f\d+[_-]?t?\d*$",
        r"[_-]\d{6}$",
        r"[_-]\d{5}$",
    ]
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new_s = re.sub(pat, "", s, flags=re.IGNORECASE)
            if new_s != s and new_s:
                s = new_s
                changed = True
    return s


def _stringify_source_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return ""
    return Path(text).stem if ("/" in text or "\\" in text) else Path(text).stem


def infer_original_source_id_from_pkl(path: str, metadata: Optional[dict] = None) -> str:
    """Infer a stable original-video id for file-level split.

    Priority:
    1. explicit metadata inside pkl, e.g. original_filename/source_file/video_name;
    2. parent+canonicalized stem fallback;
    3. full stem as last resort.
    """
    p = Path(path)
    metadata = metadata or {}
    source_keys = (
        "original_filename",
        "orig_filename",
        "source_filename",
        "source_file",
        "source_path",
        "source",
        "video_name",
        "video_id",
        "video",
        "bvh_name",
        "bvh_file",
        "motion_name",
        "motion_id",
        "filename",
        "name",
    )
    for key in source_keys:
        if key in metadata:
            text = _stringify_source_value(metadata.get(key))
            if text:
                return _safe_numeric_token_strip(text)

    return _safe_numeric_token_strip(p.stem)


def _read_pkl_metadata(path: str, load_full_when_needed: bool = True) -> Dict[str, Any]:
    """Read source metadata from a pkl.

    The current Dunhuang files are small enough that loading once in dataset
    construction is acceptable.  This function exists so source-id split uses
    explicit original file names when the preprocessing pipeline preserved them.
    """
    try:
        if not load_full_when_needed:
            return {}
        data = pickle.load(open(path, "rb"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_source_groups(pkl_files: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path in sorted(map(str, pkl_files)):
        meta = _read_pkl_metadata(path)
        sid = infer_original_source_id_from_pkl(path, meta)
        groups.setdefault(sid, []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0])}


def _load_split_manifest(path: str) -> Optional[Dict[str, set]]:
    """Load optional split manifest.

    Supported JSON formats:
      {"train": ["video_a"], "val": ["video_b"]}
      {"train_sources": [...], "val_sources": [...]}

    Supported CSV columns:
      source_id,split
      original_filename,split
      filename,split
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dunhuang split manifest not found: {path}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        train = data.get("train", data.get("train_sources", []))
        val = data.get("val", data.get("valid", data.get("validation", data.get("val_sources", []))))
        return {"train": set(map(str, train)), "val": set(map(str, val))}

    if p.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
        train, val = set(), set()
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                raw_sid = row.get("source_id") or row.get("original_filename") or row.get("filename") or row.get("source")
                split = (row.get("split") or row.get("fold") or "").strip().lower()
                sid = _safe_numeric_token_strip(_stringify_source_value(raw_sid))
                if not sid:
                    continue
                if split in {"train", "tr"}:
                    train.add(sid)
                elif split in {"val", "valid", "validation", "test", "dev"}:
                    val.add(sid)
        return {"train": train, "val": val}

    raise ValueError(f"Unsupported split manifest format: {path}")


def split_source_groups(
    groups: Dict[str, List[str]],
    train: Optional[bool],
    split_ratio: float = 0.9,
    split_seed: int = 42,
    manifest_path: str = "",
    strict: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    all_sources = sorted(groups.keys())
    if train is None:
        files = [f for sid in all_sources for f in groups[sid]]
        return files, {
            "split_name": "all",
            "train_sources": all_sources,
            "val_sources": [],
            "selected_sources": all_sources,
            "num_total_sources": len(all_sources),
            "split_by": "source_id_all",
        }

    if len(all_sources) < 2:
        msg = (
            "Dunhuang source-file split requires at least two original source videos. "
            f"Found {len(all_sources)} source group(s): {all_sources}."
        )
        if strict and not _env_bool("EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT", False):
            raise RuntimeError(msg + " Set EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1 only for smoke tests.")
        selected = all_sources
        return [f for sid in selected for f in groups[sid]], {
            "split_name": "train" if train else "val",
            "train_sources": all_sources,
            "val_sources": all_sources,
            "selected_sources": selected,
            "num_total_sources": len(all_sources),
            "split_by": "single_source_reuse_allowed",
            "warning": msg,
        }

    manifest = _load_split_manifest(manifest_path)
    if manifest is not None:
        train_sources = sorted(set(manifest.get("train", set())) & set(all_sources))
        val_sources = sorted(set(manifest.get("val", set())) & set(all_sources))
        unknown_train = sorted(set(manifest.get("train", set())) - set(all_sources))
        unknown_val = sorted(set(manifest.get("val", set())) - set(all_sources))
        split_by = f"manifest:{manifest_path}"
    else:
        split_ratio = float(np.clip(split_ratio, 0.0, 1.0))
        rng = random.Random(int(split_seed))
        shuffled = list(all_sources)
        rng.shuffle(shuffled)
        split_idx = int(round(len(shuffled) * split_ratio))
        split_idx = max(1, min(len(shuffled) - 1, split_idx))
        train_sources = sorted(shuffled[:split_idx])
        val_sources = sorted(shuffled[split_idx:])
        unknown_train, unknown_val = [], []
        split_by = "original_source_id"

    leakage = sorted(set(train_sources) & set(val_sources))
    if leakage:
        raise RuntimeError(
            "CRITICAL DATA LEAKAGE: train/val source groups overlap: "
            f"{leakage[:20]}"
        )
    if strict and (not train_sources or not val_sources):
        raise RuntimeError(
            "Invalid Dunhuang source split: train or val source list is empty. "
            f"train={len(train_sources)}, val={len(val_sources)}, total={len(all_sources)}"
        )

    selected_sources = train_sources if train else val_sources
    files = [f for sid in selected_sources for f in groups[sid]]
    report = {
        "split_name": "train" if train else "val",
        "split_by": split_by,
        "num_total_sources": len(all_sources),
        "num_train_sources": len(train_sources),
        "num_val_sources": len(val_sources),
        "num_selected_sources": len(selected_sources),
        "train_sources": train_sources,
        "val_sources": val_sources,
        "selected_sources": selected_sources,
        "unknown_manifest_train_sources": unknown_train,
        "unknown_manifest_val_sources": unknown_val,
        "source_group_hash_train": hashlib.sha256("\n".join(train_sources).encode("utf-8")).hexdigest(),
        "source_group_hash_val": hashlib.sha256("\n".join(val_sources).encode("utf-8")).hexdigest(),
    }
    return files, report


def motion_energy_scalar_from_151(motion) -> torch.Tensor:
    """Return a stable normalized energy scalar in [0,1] for one [T,151] motion."""
    if not torch.is_tensor(motion):
        motion = torch.as_tensor(motion, dtype=torch.float32)
    motion = motion.float()
    if motion.ndim != 2 or motion.shape[0] < 2:
        return torch.tensor([0.0], dtype=torch.float32)

    root_xz = motion[:, TRAJ_ROOT_XZ_IDXS]
    lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
    upper_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def rot_indices(joints):
        idx = []
        for j in joints:
            idx.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
        return idx

    root_speed = torch.linalg.norm(root_xz[1:] - root_xz[:-1], dim=-1).mean()
    lower_idx = rot_indices(lower_joints)
    upper_idx = rot_indices(upper_joints)
    lower_energy = torch.sqrt(((motion[1:, lower_idx] - motion[:-1, lower_idx]) ** 2).mean() + 1e-8)
    upper_energy = torch.sqrt(((motion[1:, upper_idx] - motion[:-1, upper_idx]) ** 2).mean() + 1e-8)
    contact = motion[:, CONTACT_SLICE].clamp(0.0, 1.0)
    contact_change = torch.abs(contact[1:] - contact[:-1]).mean()

    raw = 0.35 * root_speed + 0.30 * lower_energy + 0.25 * upper_energy + 0.10 * contact_change
    energy = torch.sigmoid(8.0 * (raw - 0.04))
    return energy.reshape(1).clamp(0.0, 1.0).float()


def _pad_or_trim_feature(feature: np.ndarray, seq_len: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(f"feature must be [T,C], got {feature.shape}")
    if feature.shape[0] > seq_len:
        return feature[:seq_len].astype(np.float32)
    if feature.shape[0] == seq_len:
        return feature.astype(np.float32)
    pad_len = seq_len - feature.shape[0]
    if feature.shape[0] == 0:
        raise ValueError("empty audio feature cannot be padded")
    if feature.shape[0] == 1:
        return np.repeat(feature, seq_len, axis=0).astype(np.float32)
    if feature.shape[0] < 15:
        return np.pad(feature, ((0, pad_len), (0, 0)), mode="edge").astype(np.float32)

    out = feature.copy()
    remaining = pad_len
    while remaining > 0:
        current_pad = min(remaining, out.shape[0] - 1)
        out = np.pad(out, ((0, current_pad), (0, 0)), mode="reflect")
        remaining -= current_pad
    return out.astype(np.float32)


def _resample_feature_rate(
    feature: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    """Resample time-major conditioning features without changing duration."""

    x = np.asarray(feature, dtype=np.float32)
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if x.ndim != 2:
        raise ValueError(f"Expected time-major [T,D] features, got {x.shape}")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("feature FPS values must be positive")
    if len(x) <= 1 or abs(source_fps - target_fps) < 1.0e-8:
        return x.copy()

    target_len = max(
        2,
        int(round(((len(x) - 1) / source_fps) * target_fps)) + 1,
    )
    old_t = np.arange(len(x), dtype=np.float64) / source_fps
    new_t = np.arange(target_len, dtype=np.float64) / target_fps
    new_t = np.minimum(new_t, old_t[-1])
    return np.stack(
        [np.interp(new_t, old_t, x[:, dim]) for dim in range(x.shape[1])],
        axis=-1,
    ).astype(np.float32)


def _onset_from_audio_feature(feature: torch.Tensor, seq_len: int) -> torch.Tensor:
    if feature.ndim != 2:
        return torch.zeros((seq_len, 1), dtype=torch.float32)
    if feature.shape[-1] > 768:
        onset = feature[:, 768:769].clamp_min(0.0)
        return onset / onset.amax().clamp_min(1e-6)
    if feature.shape[-1] >= 35:
        onset = (feature[:, 0:1] + 0.5 * feature[:, -2:-1] + 0.5 * feature[:, -1:]).clamp_min(0.0)
        return onset / onset.amax().clamp_min(1e-6)
    if feature.shape[0] > 1:
        diff = torch.zeros((feature.shape[0], 1), dtype=torch.float32)
        diff[1:, 0] = torch.linalg.norm(feature[1:] - feature[:-1], dim=-1)
        diff[0] = diff[1]
        return diff / diff.amax().clamp_min(1e-6)
    return torch.zeros((seq_len, 1), dtype=torch.float32)


class AISTPPDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool,
        backup_path: str = "data/backup",
        feature_type: str = "hybrid",
        normalizer: Any = None,
        data_len: int = -1,
        seq_len: int = 150,
        include_contacts: bool = True,
        force_reload: bool = False,
        return_traj: bool = False,
        data_fps: int = 30,
        raw_fps: int = 60,
        feature_fps: float = 30.0,
        contact_speed_threshold_mps: float = 0.3,
    ):
        _require_xz_trajectory_contract("AISTPPDataset")
        self.return_traj = return_traj
        self.data_path = data_path
        self.raw_fps = int(raw_fps)
        self.data_fps = int(data_fps)
        self.feature_fps = float(feature_fps)
        self.include_contacts = bool(include_contacts)
        self.contact_speed_threshold_mps = float(contact_speed_threshold_mps)
        if self.raw_fps <= 0 or self.data_fps <= 0 or self.feature_fps <= 0.0:
            raise ValueError("raw_fps, data_fps and feature_fps must be positive")
        if self.data_fps > self.raw_fps or self.raw_fps % self.data_fps != 0:
            raise ValueError(
                "AIST++ stride adapter requires raw_fps to be an integer multiple "
                f"of data_fps, got raw_fps={self.raw_fps}, data_fps={self.data_fps}"
            )
        if self.contact_speed_threshold_mps < 0.0:
            raise ValueError("contact_speed_threshold_mps must be non-negative")
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type
        self.seq_len = seq_len
        self.normalizer = normalizer
        self.data_len = data_len

        split_name = "train" if train else "test"
        contact_tag = "contacts" if self.include_contacts else "no_contacts"
        pickle_name = (
            f"processed_{split_name}_raw{self.raw_fps}_fps{self.data_fps}_"
            f"feat{self.feature_fps:g}_{contact_tag}.pkl"
        )
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        if not train and normalizer is not None:
            normalizer_name = f"normalizer_fps{self.data_fps}_{contact_tag}.pkl"
            pickle.dump(normalizer, open(os.path.join(backup_path, normalizer_name), "wb"))

        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset and applying strict shape clipping...")
            data = self.load_aistpp()
            with open(os.path.join(backup_path, pickle_name), "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        print(f"Loaded {self.name} Dataset With Dimensions: Pos: {data['pos'].shape}, Q: {data['q'].shape}")
        pose_input = self.process_dataset(data["pos"], data["q"])
        self.data = {"pose": pose_input, "filenames": data["filenames"], "wavs": data["wavs"]}
        assert len(pose_input) == len(data["filenames"])
        self.length = len(pose_input)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        raw_feature = _resample_feature_rate(
            np.load(filename_),
            source_fps=self.feature_fps,
            target_fps=self.data_fps,
        )
        feature = torch.from_numpy(_pad_or_trim_feature(raw_feature, self.seq_len)).float()
        pose = self.data["pose"][idx]
        cond = {
            "audio": feature,
            "audio_paired": torch.tensor(1.0, dtype=torch.float32),
            "onset": _onset_from_audio_feature(feature, self.seq_len),
            "energy": motion_energy_scalar_from_151(pose),
        }
        if self.return_traj:
            # Normalized ground-plane X/Z root path from the 151D motion.
            cond["trajectory"] = pose[:, TRAJ_ROOT_XZ_IDXS].float()
        return pose, cond, filename_, self.data["wavs"][idx]

    def load_aistpp(self):
        split_data_path = os.path.join(self.data_path, "train" if self.train else "test")
        motion_path = os.path.join(split_data_path, "motions_sliced")
        sound_path = os.path.join(split_data_path, f"{self.feature_type}_feats")
        wav_path = os.path.join(split_data_path, "wavs_sliced")

        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        motion_dict = {os.path.splitext(os.path.basename(m))[0]: m for m in motions}
        feature_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in features}
        wav_dict = {os.path.splitext(os.path.basename(w))[0]: w for w in wavs}
        common_keys = sorted(list(set(motion_dict.keys()) & set(feature_dict.keys()) & set(wav_dict.keys())))
        print(f"🧩 正在取交集: 匹配到 {len(common_keys)} 个完整的音视频切片对 (动作库:{len(motions)} / 音频库:{len(features)})")

        all_pos, all_q, all_names, all_wavs = [], [], [], []
        required_len = self.seq_len * self.data_stride
        for key in common_keys:
            data = pickle.load(open(motion_dict[key], "rb"))
            pos, q = data["pos"], data["q"]
            if pos.shape[0] < required_len:
                continue
            all_pos.append(pos[:required_len])
            all_q.append(q[:required_len])
            all_names.append(feature_dict[key])
            all_wavs.append(wav_dict[key])

        all_pos = np.array(all_pos)[:, :: self.data_stride, :]
        all_q = np.array(all_q)[:, :: self.data_stride, :]
        return {"pos": all_pos, "q": all_q, "filenames": all_names, "wavs": all_wavs}

    def process_dataset(self, root_pos, local_q):
        smpl = SMPLSkeleton()
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))

        positions = smpl.forward(local_q, root_pos)
        feet = positions[:, :, (7, 8, 10, 11)]
        foot_speed = torch.zeros(feet.shape[:3])
        foot_speed[:, :-1] = (
            (feet[:, 1:] - feet[:, :-1]).norm(dim=-1) * float(self.data_fps)
        )
        if self.include_contacts:
            contacts = (foot_speed < self.contact_speed_threshold_mps).to(local_q)
        else:
            contacts = torch.zeros_like(foot_speed).to(local_q)
        local_q = ax_to_6v(local_q)
        global_pose_vec_input = vectorize_many([contacts, root_pos, local_q]).float().detach()

        if self.train:
            self.normalizer = Normalizer(global_pose_vec_input)
        else:
            assert self.normalizer is not None
        global_pose_vec_input = self.normalizer.normalize(global_pose_vec_input)
        assert not torch.isnan(global_pose_vec_input).any()
        if self.data_len > 0:
            global_pose_vec_input = global_pose_vec_input[: self.data_len]
        return global_pose_vec_input


class OrderedMusicDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        train: bool = False,
        feature_type: str = "hybrid",
        data_name: str = "aist",
        data_fps: int = 30,
    ):
        self.data_path = data_path
        self.data_fps = int(data_fps)
        if self.data_fps <= 0:
            raise ValueError("data_fps must be positive")
        self.feature_type = feature_type
        self.test_list = set(["mLH4", "mKR2", "mBR0", "mLO2", "mJB5", "mWA0", "mJS3", "mMH3", "mHO5", "mPO1"])
        self.train = train
        self.data_name = data_name
        if self.data_name != "aist":
            self.train = True
        self.data = self.load_music()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return None

    def get_batch(self, batch_size, idx=None):
        key = random.choice(self.keys) if idx is None else self.keys[idx]
        seq = self.data[key]
        if len(seq) <= batch_size:
            seq_slice = seq
        else:
            start = random.randint(0, len(seq) - batch_size)
            seq_slice = seq[start : start + batch_size]
        filenames = [os.path.join(self.music_path, x + ".npy") for x in seq_slice]
        features = np.array([np.load(x) for x in filenames])
        return torch.Tensor(features), seq_slice

    def load_music(self):
        if self.feature_type == "baseline":
            feat_dir = f"{self.data_name}_baseline_feats"
        elif self.feature_type == "hybrid":
            feat_dir = f"{self.data_name}_hybrid_feats"
        else:
            feat_dir = f"{self.data_name}_juke_feats/juke_66"
        self.music_path = os.path.join(self.data_path, feat_dir)

        all_names = []
        key_func = lambda x: int(x.split("_")[-1].split("e")[-1])

        def stringintcmp(a, b):
            aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
            ka, kb = key_func(a), key_func(b)
            if aa < bb:
                return -1
            if aa > bb:
                return 1
            if ka < kb:
                return -1
            if ka > kb:
                return 1
            return 0

        for features in glob.glob(os.path.join(self.music_path, "*.npy")):
            all_names.append(os.path.splitext(os.path.basename(features))[0])
        all_names = sorted(all_names, key=cmp_to_key(stringintcmp))

        data_dict = {}
        for name in all_names:
            k = "".join(name.split("_")[:-1])
            if (self.train and k in self.test_list) or ((not self.train) and k not in self.test_list):
                continue
            data_dict[k] = data_dict.get(k, []) + [name]
        self.keys = sorted(list(data_dict.keys()))
        return data_dict


class DummyNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            return x * self.std.to(x.device) + self.mean.to(x.device)
        return x * self.std.numpy() + self.mean.numpy()


class DunhuangDataset(Dataset):
    """Dunhuang motion dataset with strict original-source Train/Val isolation.

    Main fixes:
    - split by original source id BEFORE creating overlapping windows;
    - detect/raise if train/val source ids overlap;
    - compute normalizer only from selected training sources and reuse it for val;
    - enforce X/Z ground-plane trajectory convention everywhere.

    Environment switches:
      EDGE_DUNHUANG_SPLIT_MODE=source_file      # default; only safe formal mode
      EDGE_DUNHUANG_SPLIT_MANIFEST=path.json    # optional explicit source split
      EDGE_DUNHUANG_STRICT_SPLIT=1              # default; fail on empty/invalid split
      EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=0 # set 1 only for smoke tests
      EDGE_DUNHUANG_SPLIT_REPORT_DIR=...        # optional JSON reports
      EDGE_TRAJECTORY_PLANE=xz                  # default; anything else raises
    """

    def __init__(
        self,
        data_path,
        train: Optional[bool] = None,
        seq_len=150,
        audio_dim=803,
        overlap=0.5,
        normalizer=None,
        return_traj=False,
        weak_pairs_path="data/proxy_weak_pairs/weak_pairs.csv",
        use_weak_pairs=True,
        split_ratio=0.9,
        split_seed=42,
        audio_sample_mode="random",
        audio_pairing_mode="proxy",
        paired_audio_missing_policy="error",
        traj_aug_prob=0.0,
        traj_aug_scale_range=(1.0, 1.0),
        traj_aug_rot_deg=0.0,
    ):
        _require_xz_trajectory_contract("DunhuangDataset")
        self.data_path = data_path
        self.train = train
        self.seq_len = int(seq_len)
        self.audio_dim = int(audio_dim)
        self.return_traj = bool(return_traj)
        self.audio_pairing_mode = str(audio_pairing_mode).lower()
        self.paired_audio_missing_policy = paired_audio_missing_policy
        self.overlap = float(overlap)
        self.strict_split = _env_bool("EDGE_DUNHUANG_STRICT_SPLIT", True)

        valid_pairing_modes = {"none", "proxy", "paired"}
        if self.audio_pairing_mode not in valid_pairing_modes:
            raise ValueError(f"audio_pairing_mode must be one of {sorted(valid_pairing_modes)}, got {audio_pairing_mode}")
        if self.audio_pairing_mode == "none":
            use_weak_pairs = False
            audio_sample_mode = "zero"
        if self.audio_pairing_mode == "paired":
            audio_sample_mode = "best"
        self.audio_sample_mode = audio_sample_mode

        self.traj_aug_prob = float(traj_aug_prob) if train else 0.0
        self.traj_aug_scale_range = traj_aug_scale_range
        self.traj_aug_rot_deg = float(traj_aug_rot_deg)

        self.proxy_audios: List[np.ndarray] = []
        self.motion_window_ids: List[str] = []
        self.motion_source_ids: List[str] = []
        self.window_source_paths: List[str] = []
        self.weak_pair_map: Dict[str, List[dict]] = {}
        self.weak_pair_audio_cache: Dict[str, Optional[np.ndarray]] = {}
        if use_weak_pairs:
            self.weak_pair_map = self._load_weak_pairs(weak_pairs_path)

        self._load_proxy_audio_bank()
        self.pkl_files = self._discover_pkl_files(data_path)
        self.all_pkl_files = list(self.pkl_files)
        self.source_groups = build_source_groups(self.pkl_files)
        self.pkl_files, self.split_report = self._split_source_files(
            train=train,
            split_ratio=split_ratio,
            split_seed=split_seed,
        )
        self.selected_source_ids = list(self.split_report.get("selected_sources", []))

        motions_list: List[np.ndarray] = []
        trajs_list: List[np.ndarray] = []

        if not self.pkl_files:
            msg = f"No PKL files selected for Dunhuang split={self.split_report.get('split_name')}"
            if self.strict_split:
                raise RuntimeError(msg)
            print("⚠️ " + msg)
            self.motions = np.zeros((0, self.seq_len, 151), dtype=np.float32)
            self.trajs = np.zeros((0, self.seq_len, 2), dtype=np.float32)
            self.normalizer = normalizer if (normalizer is not None and hasattr(normalizer, "mean")) else None
            return

        smpl = SMPLSkeleton()
        step = max(1, int(round(self.seq_len * (1.0 - self.overlap))))

        for f in self.pkl_files:
            data = pickle.load(open(f, "rb"))
            # ===== 新增：直接读取 151D motion =====
            if "motion" in data:
                motion = np.asarray(data["motion"], dtype=np.float32)

            elif "motion_151" in data:
                motion = np.asarray(data["motion_151"], dtype=np.float32)

            elif "poses" in data:
                motion = np.asarray(data["poses"], dtype=np.float32)

            elif "unit_motions_physical" in data:
                motion = np.asarray(data["unit_motions_physical"], dtype=np.float32)

            else:
                motion = None

            # ===== 151D fallback =====
            if motion is not None:
                _edge_151d_path = locals().get('pkl_path', None) or locals().get('pkl_file', None) or locals().get('file', None) or locals().get('path', None) or locals().get('pkl', None) or 'direct_151d_motion'
                if motion.ndim != 2 or motion.shape[-1] != 151:
                    print(f"⚠️ 跳过 {_edge_151d_path}: invalid 151D motion shape {motion.shape}")
                    continue

                if motion.shape[0] < self.seq_len:
                    print(f"⚠️ 跳过 {_edge_151d_path}: too short {motion.shape[0]}")
                    continue

                motion = motion[: self.seq_len].astype(np.float32)

                motions_list.append(motion)

                trajs_list.append(
                    motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
                )

                self.motion_window_ids.append(
                    Path(_edge_151d_path).stem
                )

                self.motion_source_ids.append(
                    infer_original_source_id_from_pkl(_edge_151d_path, data)
                )

                self.window_source_paths.append(str(_edge_151d_path))

                continue

            # ===== 原始 pos/q 流程 =====
            if "pos" not in data or "q" not in data:
                print(f"⚠️ 跳过 {_edge_151d_path}: missing pos/q")
                continue
            source_id = infer_original_source_id_from_pkl(f, data)
            pos = np.asarray(data["pos"], dtype=np.float32)
            q = np.asarray(data["q"], dtype=np.float32)
            if pos.ndim != 2 or pos.shape[1] != 3 or q.ndim != 2:
                print(f"⚠️ 跳过 {f}: invalid pos/q shapes pos={pos.shape}, q={q.shape}")
                continue

            pos_t = torch.Tensor(pos).unsqueeze(0)
            q_t = torch.Tensor(q).unsqueeze(0)
            q_t_reshaped = q_t.reshape((q_t.shape[0], q_t.shape[1], -1, 3))

            positions = smpl.forward(q_t_reshaped, pos_t)
            feet = positions[:, :, (7, 8, 10, 11)]
            feetv = torch.zeros(feet.shape[:3])
            feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
            contacts = (feetv < 0.01).to(q_t_reshaped)

            q_6v = ax_to_6v(q_t_reshaped)
            motion_t = vectorize_many([contacts, pos_t, q_6v])
            motion = motion_t.squeeze(0).float().detach().numpy()

            # Physical ground-plane trajectory from SMPL root position: X/Z = pos[:, 0], pos[:, 2].
            traj_xz = pos[:, SMPL_POS_XZ_IDXS].astype(np.float32)
            num_frames = motion.shape[0]
            if num_frames < self.seq_len:
                continue

            for start in range(0, num_frames - self.seq_len + 1, step):
                end = start + self.seq_len
                slice_motion = motion[start:end].copy()
                slice_traj = traj_xz[start:end].copy()

                # Localize each window to start at root X/Z = 0.  This keeps root
                # supervision invariant to absolute capture location while preserving
                # the full relative ground-plane path.
                local_start_x = float(slice_motion[0, ROOT_X_IDX])
                local_start_z = float(slice_motion[0, ROOT_Z_IDX])
                slice_motion[:, ROOT_X_IDX] -= local_start_x
                slice_motion[:, ROOT_Z_IDX] -= local_start_z
                slice_traj[:, 0] -= local_start_x
                slice_traj[:, 1] -= local_start_z

                slice_motion, slice_traj = self._augment_motion_traj_physical(slice_motion, slice_traj)
                if _env_bool("EDGE_DUNHUANG_ASSERT_TRAJ_MATCH", True):
                    max_delta = float(np.max(np.abs(slice_motion[:, TRAJ_ROOT_XZ_IDXS] - slice_traj)))
                    if max_delta > 1e-4:
                        raise RuntimeError(
                            f"X/Z trajectory mismatch after preprocessing: file={f}, start={start}, max_delta={max_delta:.6g}"
                        )

                window_id = f"{Path(f).stem}_{start:06d}_{end:06d}"
                motions_list.append(slice_motion.astype(np.float32))
                trajs_list.append(slice_traj.astype(np.float32))
                self.motion_window_ids.append(window_id)
                self.motion_source_ids.append(source_id)
                self.window_source_paths.append(str(f))

        if len(motions_list) == 0:
            msg = (
                f"No valid motion windows for Dunhuang split={self.split_report.get('split_name')}. "
                "This usually means selected source videos are shorter than seq_len."
            )
            if self.strict_split:
                raise RuntimeError(msg)
            print("⚠️ " + msg)
            self.motions = np.zeros((0, self.seq_len, 151), dtype=np.float32)
            self.trajs = np.zeros((0, self.seq_len, 2), dtype=np.float32)
            self.normalizer = normalizer if (normalizer is not None and hasattr(normalizer, "mean")) else None
            return

        self.motions = np.array(motions_list, dtype=np.float32)
        self.trajs = np.array(trajs_list, dtype=np.float32)

        if normalizer is None or not hasattr(normalizer, "mean"):
            if train is False and self.strict_split:
                raise RuntimeError(
                    "Validation DunhuangDataset received no training normalizer. "
                    "Instantiate train dataset first and pass train_dataset.normalizer to val."
                )
            print("⚠️ 未提供有效 Normalizer，将基于当前 Dunhuang split 重新计算统计量。")
            self.normalizer = Normalizer(torch.from_numpy(self.motions))
        else:
            self.normalizer = normalizer

        self.motions = self.normalizer.normalize(torch.from_numpy(self.motions)).numpy().astype(np.float32)
        self.trajs = self._normalize_xz_trajectory(self.trajs).astype(np.float32)

        if _env_bool("EDGE_DUNHUANG_ASSERT_TRAJ_MATCH", True):
            root_xz = self.motions[:, :, TRAJ_ROOT_XZ_IDXS]
            max_delta = float(np.max(np.abs(root_xz - self.trajs))) if len(self.trajs) else 0.0
            if max_delta > 1e-4:
                raise RuntimeError(
                    "Normalized trajectory contract failed: cond['trajectory'] must equal normalized motion root X/Z. "
                    f"max_delta={max_delta:.6g}"
                )

        self._write_split_report_if_requested()
        print(
            f"📦 DunhuangDataset strict source split [{self.split_report.get('split_name')}]: "
            f"sources={len(self.selected_source_ids)}/{self.split_report.get('num_total_sources')}, "
            f"windows={len(self.motions)}, split_by={self.split_report.get('split_by')}, "
            "trajectory_plane=X/Z"
        )

    def _discover_pkl_files(self, data_path: str) -> List[str]:
        candidate_dirs = [data_path, os.path.join(data_path, "processed")]
        found = []
        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                files = sorted(glob.glob(os.path.join(candidate, "*.pkl")))
                if files:
                    found = files
                    self.data_path = candidate
                    break
        if not found:
            found = sorted(glob.glob(os.path.join(str(data_path), "*.pkl")))
        if not found:
            raise FileNotFoundError(f"No Dunhuang .pkl files found under {data_path}")
        return found

    def _split_source_files(self, train, split_ratio=0.9, split_seed=42):
        split_mode = _env_str("EDGE_DUNHUANG_SPLIT_MODE", "source_file").lower()
        if split_mode not in {"source_file", "source", "original_filename", "all"}:
            raise ValueError(
                f"Unsupported EDGE_DUNHUANG_SPLIT_MODE={split_mode!r}. "
                "Use source_file for formal training."
            )
        if split_mode == "all":
            if self.strict_split and train is not None:
                raise RuntimeError(
                    "EDGE_DUNHUANG_SPLIT_MODE=all is forbidden in strict mode because it can leak source videos. "
                    "Use only for debugging with EDGE_DUNHUANG_STRICT_SPLIT=0."
                )
            files = [f for sid in sorted(self.source_groups) for f in self.source_groups[sid]]
            return files, {
                "split_name": "all",
                "split_by": "all_debug",
                "num_total_sources": len(self.source_groups),
                "selected_sources": sorted(self.source_groups),
                "train_sources": sorted(self.source_groups),
                "val_sources": sorted(self.source_groups),
            }
        return split_source_groups(
            self.source_groups,
            train=train,
            split_ratio=split_ratio,
            split_seed=split_seed,
            manifest_path=_env_str("EDGE_DUNHUANG_SPLIT_MANIFEST", ""),
            strict=self.strict_split,
        )

    def _write_split_report_if_requested(self) -> None:
        report_dir = _env_str("EDGE_DUNHUANG_SPLIT_REPORT_DIR", "")
        report_path = _env_str("EDGE_DUNHUANG_SPLIT_REPORT_JSON", "")
        if report_dir and not report_path:
            split = self.split_report.get("split_name", "split")
            report_path = str(Path(report_dir) / f"dunhuang_{split}_source_split_report.json")
        if not report_path:
            return
        payload = dict(self.split_report)
        payload.update(
            {
                "num_windows": int(len(self.motions)),
                "window_ids_head": self.motion_window_ids[:10],
                "window_source_ids_head": self.motion_source_ids[:10],
                "data_path": self.data_path,
                "seq_len": self.seq_len,
                "overlap": self.overlap,
                "trajectory_plane": "xz",
                "trajectory_root_indices_151d": TRAJ_ROOT_XZ_IDXS,
                "smpl_pos_indices_xz": SMPL_POS_XZ_IDXS,
            }
        )
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Dunhuang split/trajectory report saved: {p}")

    def _load_proxy_audio_bank(self):
        rag_db_path = "data/dunhuang_rag_db"
        if not os.path.isdir(rag_db_path):
            return
        for rag_file in sorted(glob.glob(os.path.join(rag_db_path, "*.npy"))):
            try:
                record = np.load(rag_file, allow_pickle=True).item()
                audio_feat = self._validate_audio_feature(record.get("audio_feat", None), source=rag_file)
                if audio_feat is not None:
                    self.proxy_audios.append(audio_feat)
            except Exception as e:
                print(f"⚠️ 跳过损坏的 RAG 文件 {rag_file}: {e}")

    def _normalize_xz_trajectory(self, trajs: np.ndarray) -> np.ndarray:
        if self.normalizer is None or not hasattr(self.normalizer, "mean") or not hasattr(self.normalizer, "std"):
            return trajs.astype(np.float32)
        mean_xz = np.array([self.normalizer.mean[ROOT_X_IDX], self.normalizer.mean[ROOT_Z_IDX]], dtype=np.float32)
        std_xz = np.array([self.normalizer.std[ROOT_X_IDX], self.normalizer.std[ROOT_Z_IDX]], dtype=np.float32)
        return (trajs.astype(np.float32) - mean_xz) / (std_xz + 1e-8)

    def _augment_motion_traj_physical(self, motion, traj):
        """Apply the same geometric transform to motion root X/Z and trajectory X/Z."""
        if self.traj_aug_prob <= 0 or random.random() > self.traj_aug_prob:
            return motion, traj
        motion = motion.copy()
        traj = traj.copy()
        scale_min, scale_max = self.traj_aug_scale_range
        scale = random.uniform(float(scale_min), float(scale_max))
        angle = random.uniform(-self.traj_aug_rot_deg, self.traj_aug_rot_deg) * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        root_xz = motion[:, TRAJ_ROOT_XZ_IDXS].astype(np.float32)
        root_xz = (root_xz @ rot.T) * scale
        traj_xz = (traj.astype(np.float32) @ rot.T) * scale
        motion[:, ROOT_X_IDX] = root_xz[:, 0]
        motion[:, ROOT_Z_IDX] = root_xz[:, 1]
        traj[:, 0] = traj_xz[:, 0]
        traj[:, 1] = traj_xz[:, 1]
        return motion, traj

    def __len__(self):
        return len(self.motions)

    def _resolve_audio_feature_path(self, audio_path):
        if not audio_path:
            return None
        candidates = []
        path = Path(audio_path)
        candidates.append(path if path.suffix == ".npy" else path.with_suffix(".npy"))
        candidates.append(Path("proxy_music") / f"{path.stem}.npy")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    def _validate_audio_feature(self, audio_feat, source=""):
        if audio_feat is None:
            return None
        try:
            audio_feat = np.asarray(audio_feat, dtype=np.float32)
        except Exception as exc:
            print(f"⚠️ 跳过音频特征 {source}: 无法转成 float32 ({exc})")
            return None
        if audio_feat.ndim != 2:
            print(f"⚠️ 跳过音频特征 {source}: 期望 [T, C]，实际 {audio_feat.shape}")
            return None
        if audio_feat.shape[0] <= 0:
            print(f"⚠️ 跳过音频特征 {source}: 时间长度为空 {audio_feat.shape}")
            return None
        if audio_feat.shape[1] != self.audio_dim:
            print(f"⚠️ 跳过音频特征 {source}: 期望 audio_dim={self.audio_dim}，实际 {audio_feat.shape[1]}")
            return None
        if not np.isfinite(audio_feat).all():
            print(f"⚠️ 跳过音频特征 {source}: 包含 NaN/Inf")
            return None
        return audio_feat.astype(np.float32)

    def _get_weak_pair_audio(self, audio_path):
        feature_path = self._resolve_audio_feature_path(audio_path)
        if feature_path is None:
            return None
        if feature_path not in self.weak_pair_audio_cache:
            try:
                loaded = np.load(feature_path)
                self.weak_pair_audio_cache[feature_path] = self._validate_audio_feature(loaded, source=feature_path)
            except Exception as exc:
                print(f"⚠️ 跳过 weak pair 音频特征 {feature_path}: {exc}")
                self.weak_pair_audio_cache[feature_path] = None
        return self.weak_pair_audio_cache[feature_path]

    def _load_weak_pairs(self, weak_pairs_path):
        if not weak_pairs_path or not os.path.isfile(weak_pairs_path):
            return {}
        pair_map: Dict[str, List[dict]] = {}
        with open(weak_pairs_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                window_id = row.get("window_id", "")
                audio_path = row.get("audio_path", "")
                audio_feat = self._get_weak_pair_audio(audio_path)
                if not window_id or audio_feat is None:
                    continue
                try:
                    score = max(float(row.get("score", 1.0)), 1e-6)
                except ValueError:
                    score = 1.0
                pair_map.setdefault(window_id, []).append({"audio_feat": audio_feat, "score": score, "audio_path": audio_path})
        if pair_map:
            print(f"🎵 已加载 weak/proxy music 候选: {len(pair_map)} 个动作窗口。")
        return pair_map

    def _choose_weak_candidate(self, candidates):
        if not candidates:
            return None
        if self.audio_sample_mode == "best":
            return max(candidates, key=lambda candidate: candidate["score"])
        weights = [max(float(candidate.get("score", 1.0)), 1e-6) for candidate in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _sample_audio_feature(self, idx):
        if self.audio_pairing_mode == "none" or self.audio_sample_mode == "zero":
            return None, "zero", 0.0

        window_id = self.motion_window_ids[idx]
        candidates = self.weak_pair_map.get(window_id, [])
        if candidates:
            chosen = self._choose_weak_candidate(candidates)
            if chosen is not None:
                return chosen["audio_feat"], chosen.get("audio_path", "weak_pair"), 1.0

        if self.audio_pairing_mode == "paired":
            if self.paired_audio_missing_policy == "zero":
                return None, "missing_paired_zero", 0.0
            raise RuntimeError(
                f"Strict paired audio requested, but no paired audio candidate for window_id={window_id}. "
                "Either fix weak_pairs_path or set --paired_audio_missing_policy zero for debugging only."
            )

        # Proxy mode fallback: random weak rhythm proxy from RAG bank.
        if self.proxy_audios:
            if self.audio_sample_mode == "best":
                return self.proxy_audios[0], "proxy_bank_best", 0.0
            return random.choice(self.proxy_audios), "proxy_bank_random", 0.0
        return None, "zero_no_proxy", 0.0

    def __getitem__(self, idx):
        pose = torch.from_numpy(self.motions[idx]).float()
        audio_feat, audio_source, paired_flag = self._sample_audio_feature(idx)
        if audio_feat is None:
            feature = torch.zeros((self.seq_len, self.audio_dim), dtype=torch.float32)
        else:
            feature = torch.from_numpy(_pad_or_trim_feature(audio_feat, self.seq_len)).float()
        cond = {
            "audio": feature,
            "audio_paired": torch.tensor(float(paired_flag), dtype=torch.float32),
            "onset": _onset_from_audio_feature(feature, self.seq_len),
            "energy": motion_energy_scalar_from_151(pose),
        }
        if self.return_traj:
            # Strict normalized X/Z ground-plane trajectory, identical to normalized root dims [4,6].
            cond["trajectory"] = torch.from_numpy(self.trajs[idx]).float()
        return pose, cond, self.motion_window_ids[idx], str(audio_source)
