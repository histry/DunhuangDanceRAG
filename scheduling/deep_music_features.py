#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditable CLAP/MSCLAP music semantics for V27/V28 Event-RAG.

Temporal segmentation still uses onset/beat/novelty.  CLAP is used only for
phrase-level semantic query shaping.

Operational guarantees:

- supports LAION-CLAP default checkpoints and explicit local checkpoint paths;
- supports the official music checkpoint path via ``amodel='HTSAT-base'``;
- records package version, backend, checkpoint path, device, success rate, and
  fallback reasons in schedule reports;
- strict mode fails fast when CLAP silently falls back to rule semantics.

Environment variables:

V27_CLAP_CKPT          optional local LAION-CLAP checkpoint path
V27_CLAP_AMODEL        default HTSAT-base when V27_CLAP_CKPT is set, else HTSAT-tiny
V27_CLAP_DEVICE        default cuda:0 if CUDA is available, else cpu
V27_CLAP_ENABLE_FUSION default 0
V27_CLAP_USE_FILELIST  default 0; set 1 to call get_audio_embedding_from_filelist
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


_MODEL_CACHE: Dict[str, Any] = {}
_BACKEND_META: Dict[str, Dict[str, Any]] = {}


def _normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def _projection_matrix(in_dim: int, out_dim: int = 12) -> np.ndarray:
    seed = int(hashlib.sha1(f"v27_music_semantic_{in_dim}_{out_dim}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    mat /= np.maximum(np.linalg.norm(mat, axis=0, keepdims=True), 1e-8)
    return mat


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return str(version(name))
    except Exception:
        return "unknown"


def _audio_fingerprint(path: Path) -> str:
    """Hash audio content so same-name replacements cannot reuse stale features."""
    digest = hashlib.sha256()
    if not path.is_file():
        digest.update(str(path.expanduser().resolve()).encode("utf-8"))
        return "missing_" + digest.hexdigest()[:16]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _phrase_fingerprint(phrases: Sequence[Any]) -> str:
    fields = (
        "start",
        "end",
        "length",
        "music_event",
        "energy",
        "onset",
        "beat_density",
        "tension",
        "calmness",
        "boundary_accent_strength",
    )
    payload = [
        {name: getattr(phrase, name, None) for name in fields}
        for phrase in phrases
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _backend_fingerprint(enabled: bool, model_name: str) -> str:
    payload = {
        "enabled": bool(enabled),
        "model_name": str(model_name).lower(),
        "checkpoint": os.environ.get("V27_CLAP_CKPT", ""),
        "amodel": os.environ.get("V27_CLAP_AMODEL", ""),
        "device": os.environ.get("V27_CLAP_DEVICE", ""),
        "fusion": _bool_env("V27_CLAP_ENABLE_FUSION", False),
        "filelist": _bool_env("V27_CLAP_USE_FILELIST", False),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validated_fps(fps: float) -> float:
    value = float(fps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps!r}")
    return value


def phrase_rule_semantic(phrase: Any, fps: float = 30.0) -> np.ndarray:
    """Return a 12D semantic proxy aligned with hierarchy raw features."""
    fps = _validated_fps(fps)
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    group_map = {
        "calm_flow": 1,
        "release": 1,
        "neutral_flow": 2,
        "build_up": 3,
        "climax": 3,
        "accent": 4,
        "section_change": 5,
    }
    group = int(group_map.get(music_event, 2))
    coarse = np.zeros((6,), dtype=np.float32)
    coarse[np.clip(group, 0, 5)] = 1.0
    energy = float(getattr(phrase, "energy", 0.5))
    onset = float(getattr(phrase, "onset", 0.0))
    beat = float(getattr(phrase, "beat_density", 0.0))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    boundary = float(getattr(phrase, "boundary_accent_strength", 0.0))
    activity = np.clip(0.42 * energy + 0.24 * beat + 0.18 * onset + 0.16 * tension - 0.22 * calm, 0.0, 1.0)
    turn = np.clip(
        0.44 * tension
        + 0.26 * boundary
        + 0.18 * beat
        + (0.18 if music_event in {"climax", "section_change"} else 0.0),
        0.0,
        1.0,
    )
    duration_seconds = float(getattr(phrase, "length", 2.0 * fps)) / fps
    duration = np.clip((duration_seconds - 0.8) / 4.0, 0.0, 1.0)
    style = np.clip(0.50 + 0.30 * calm + 0.20 * tension, 0.0, 1.0)
    quality = np.clip(0.55 + 0.25 * beat + 0.20 * boundary, 0.0, 1.0)
    safety = np.clip(0.68 + 0.20 * calm - 0.15 * onset, 0.0, 1.0)
    return _normalize(np.concatenate([coarse, np.asarray([activity, turn, duration, style, quality, safety], dtype=np.float32)]))


def _deep_mode_success(mode: str) -> bool:
    mode = str(mode)
    return mode in {"laion_clap", "laion_clap_filelist", "msclap", "msclap_file"}


def _load_audio_segment(
    audio_path: Path,
    phrase: Any,
    fps: float,
) -> Tuple[np.ndarray | None, int, str, float, float]:
    fps = _validated_fps(fps)
    try:
        import librosa  # type: ignore
    except Exception as exc:
        return None, 0, f"librosa_unavailable:{exc}", 0.0, 0.0

    start_frame = float(getattr(phrase, "start", 0))
    end_frame = float(getattr(phrase, "end", start_frame + fps))
    start_sec = start_frame / fps
    end_sec = max(start_sec + 0.25, end_frame / fps)
    try:
        y, sr = librosa.load(str(audio_path), sr=48000, mono=True, offset=start_sec, duration=end_sec - start_sec)
    except Exception as exc:
        return None, 0, f"audio_load_failed:{exc}", start_sec, end_sec
    if y.size < 256:
        return None, int(sr), "audio_too_short", start_sec, end_sec
    return np.asarray(y, dtype=np.float32), int(sr), "ok", start_sec, end_sec


def _write_temp_wav(y: np.ndarray, sr: int) -> str:
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"soundfile_unavailable:{exc}") from exc
    tmp = tempfile.NamedTemporaryFile(prefix="v27_clap_phrase_", suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, np.asarray(y, dtype=np.float32), int(sr))
    return tmp.name


def _default_device() -> str:
    env = os.environ.get("V27_CLAP_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _get_laion_clap_model() -> Any:
    ckpt = os.environ.get("V27_CLAP_CKPT", "").strip()
    amodel = os.environ.get("V27_CLAP_AMODEL", "").strip() or ("HTSAT-base" if ckpt else "HTSAT-tiny")
    device = _default_device()
    enable_fusion = _bool_env("V27_CLAP_ENABLE_FUSION", False)
    key = f"laion_clap|ckpt={ckpt}|amodel={amodel}|device={device}|fusion={int(enable_fusion)}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import laion_clap  # type: ignore

    try:
        model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, device=device)
    except TypeError:
        model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)

    if not hasattr(model, "load_ckpt"):
        raise RuntimeError("laion_clap.CLAP_Module has no load_ckpt method")
    if ckpt:
        path = Path(ckpt)
        if not path.is_file():
            raise RuntimeError(f"V27_CLAP_CKPT does not exist: {path}")
        model.load_ckpt(str(path))
    else:
        model.load_ckpt()

    _MODEL_CACHE[key] = model
    _BACKEND_META["laion_clap"] = {
        "backend": "laion_clap",
        "package": "laion-clap",
        "package_version": _package_version("laion-clap"),
        "checkpoint": ckpt or "default_load_ckpt",
        "amodel": amodel,
        "device": device,
        "enable_fusion": bool(enable_fusion),
    }
    return model


def _get_msclap_model() -> Any:
    device = _default_device()
    key = f"msclap|device={device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from msclap import CLAP  # type: ignore

    use_cuda = str(device).startswith("cuda")
    model = CLAP(version="2023", use_cuda=use_cuda)
    _MODEL_CACHE[key] = model
    _BACKEND_META["msclap"] = {
        "backend": "msclap",
        "package": "msclap",
        "package_version": _package_version("msclap"),
        "checkpoint": "package_default",
        "device": device,
    }
    return model


def _try_clap_phrase_embedding(
    audio_path: Path,
    phrase: Any,
    model_name: str,
    fps: float,
) -> Tuple[np.ndarray | None, str]:
    """Best-effort CLAP/MSCLAP phrase embedding with explicit failure mode."""
    y, sr, audio_mode, _start_sec, _end_sec = _load_audio_segment(
        audio_path,
        phrase,
        fps,
    )
    if y is None:
        return None, audio_mode

    name = model_name.lower()
    if name in {"laion_clap", "clap"}:
        try:
            model = _get_laion_clap_model()
            if _bool_env("V27_CLAP_USE_FILELIST", False):
                tmp = _write_temp_wav(y, sr)
                try:
                    emb = model.get_audio_embedding_from_filelist(x=[tmp], use_tensor=False)
                finally:
                    try:
                        Path(tmp).unlink(missing_ok=True)
                    except Exception:
                        pass
                return np.asarray(emb, dtype=np.float32).reshape(-1), "laion_clap_filelist"
            emb = model.get_audio_embedding_from_data(x=y.reshape(1, -1), use_tensor=False)
            return np.asarray(emb, dtype=np.float32).reshape(-1), "laion_clap"
        except Exception as exc:
            return None, f"laion_clap_failed:{type(exc).__name__}:{exc}"

    if name in {"msclap", "microsoft_clap"}:
        try:
            model = _get_msclap_model()
            if hasattr(model, "get_audio_embeddings_from_data"):
                emb = model.get_audio_embeddings_from_data(y.reshape(1, -1))
                return np.asarray(emb, dtype=np.float32).reshape(-1), "msclap"
            tmp = _write_temp_wav(y, sr)
            try:
                emb = model.get_audio_embeddings([tmp])
            finally:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
            return np.asarray(emb, dtype=np.float32).reshape(-1), "msclap_file"
        except Exception as exc:
            return None, f"msclap_failed:{type(exc).__name__}:{exc}"

    return None, f"unsupported_model:{model_name}"


def _meta_from_modes(
    audio_path: str | Path,
    enabled: bool,
    model_name: str,
    phrases: Sequence[Any],
    modes: Sequence[str],
    semantic: np.ndarray,
    fps: float,
) -> Dict[str, Any]:
    deep_success = int(sum(1 for mode in modes if _deep_mode_success(str(mode))))
    fallback = int(len(modes) - deep_success)
    backends = []
    for key in ("laion_clap", "msclap"):
        if key in _BACKEND_META:
            backends.append(dict(_BACKEND_META[key]))
    return {
        "audio": str(audio_path),
        "enabled": bool(enabled),
        "model_name": str(model_name),
        "phrase_fps": float(fps),
        "num_phrases": int(len(phrases)),
        "modes": [str(x) for x in modes],
        "unique_modes": sorted({str(x).split(":")[0] for x in modes}),
        "deep_success_count": deep_success,
        "fallback_count": fallback,
        "deep_success_rate": float(deep_success / max(len(modes), 1)),
        "feature_dim": int(semantic.shape[1]) if semantic.ndim == 2 else 0,
        "backend_meta": backends,
        "env": {
            "V27_CLAP_CKPT": os.environ.get("V27_CLAP_CKPT", ""),
            "V27_CLAP_AMODEL": os.environ.get("V27_CLAP_AMODEL", ""),
            "V27_CLAP_DEVICE": os.environ.get("V27_CLAP_DEVICE", ""),
            "V27_CLAP_ENABLE_FUSION": os.environ.get("V27_CLAP_ENABLE_FUSION", ""),
            "V27_CLAP_USE_FILELIST": os.environ.get("V27_CLAP_USE_FILELIST", ""),
        },
    }


def _assert_deep_success(meta: Dict[str, Any], min_success: float) -> None:
    if not bool(meta.get("enabled", False)):
        raise RuntimeError("V27 deep music strict mode is enabled, but deep music features are disabled.")
    rate = float(meta.get("deep_success_rate", 0.0))
    if rate < float(min_success):
        raise RuntimeError(
            "Deep music semantic extraction did not really run: "
            f"success_rate={rate:.3f}, required={float(min_success):.3f}, "
            f"modes={meta.get('unique_modes')}, backend_meta={meta.get('backend_meta')}. "
            "Install/verify CLAP or disable strict mode."
        )


def phrase_semantic_matrix(
    audio_path: str | Path,
    phrases: Sequence[Any],
    enabled: bool = False,
    model_name: str = "clap",
    cache_dir: str | Path | None = None,
    require_deep: bool = False,
    min_deep_success: float = 0.80,
    fps: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    fps = _validated_fps(fps)
    audio = Path(audio_path)
    cache_path = None
    audio_hash = _audio_fingerprint(audio)
    phrase_hash = _phrase_fingerprint(phrases)
    backend_hash = _backend_fingerprint(enabled, model_name)
    cache_key = (
        f"{audio.stem}_v27_semantic_{model_name}_{len(phrases)}"
        f"_fps{fps:g}"
        f"_audio{audio_hash}_phrases{phrase_hash}_backend{backend_hash}.npz"
    )
    if cache_dir:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        cache_path = cache / cache_key
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=True) as data:
                semantic = np.asarray(data["semantic"], dtype=np.float32)
                meta = json.loads(str(data["meta"].item()))
            cached_fps = meta.get("phrase_fps")
            if cached_fps is not None and np.isclose(
                float(cached_fps),
                fps,
                rtol=0.0,
                atol=1e-9,
            ):
                if require_deep:
                    _assert_deep_success(meta, min_deep_success)
                return semantic, meta

    rows = []
    modes = []
    for phrase in phrases:
        rule = phrase_rule_semantic(phrase, fps=fps)
        if enabled:
            emb, mode = _try_clap_phrase_embedding(
                audio,
                phrase,
                model_name,
                fps=fps,
            )
            if emb is not None and emb.size > 0 and np.all(np.isfinite(emb)):
                proj = _projection_matrix(int(emb.size), 12)
                deep = _normalize(np.asarray(emb, dtype=np.float32).reshape(1, -1) @ proj)
                rows.append(_normalize(0.50 * rule + 0.50 * deep))
                modes.append(mode)
                continue
            modes.append(mode)
        else:
            modes.append("disabled_rule_proxy")
        rows.append(rule)

    semantic = np.stack(rows).astype(np.float32) if rows else np.zeros((0, 12), dtype=np.float32)
    meta = _meta_from_modes(
        audio_path,
        enabled,
        model_name,
        phrases,
        modes,
        semantic,
        fps=fps,
    )
    if require_deep:
        _assert_deep_success(meta, min_deep_success)
    if cache_path is not None:
        np.savez_compressed(cache_path, semantic=semantic, meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object))
    return semantic, meta
