#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime cache and progress telemetry for candidate anatomy evaluation.

The cache is process-local. Its key records Event identity, resampled core
length and FPS, preventing reuse across incompatible temporal contracts.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, TextIO, Tuple

import numpy as np

from contracts.anatomy import event_anatomy_features


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _db_value(
    db: Optional[Mapping[str, Any]],
    key: str,
    event_id: int,
    default: Any,
) -> Any:
    if db is None or key not in db:
        return default
    try:
        value = np.asarray(db[key])[int(event_id)]
    except Exception:
        return default
    return value.item() if isinstance(value, np.generic) else value


@dataclass(frozen=True)
class CandidateAnatomyCacheKey:
    event_id: int
    target_core_frames: int
    fps_millihz: int
    source_frames: int
    sample_frames: int
    evaluation_mode: str

    @classmethod
    def build(
        cls,
        *,
        event_id: int,
        target_core_frames: int,
        fps: float,
        source_frames: int,
        sample_frames: int,
        evaluation_mode: str,
    ) -> "CandidateAnatomyCacheKey":
        return cls(
            event_id=int(event_id),
            target_core_frames=int(target_core_frames),
            fps_millihz=int(round(float(fps) * 1000.0)),
            source_frames=int(source_frames),
            sample_frames=int(sample_frames),
            evaluation_mode=str(evaluation_mode),
        )


class CandidateAnatomyCache:
    """Thread-safe bounded LRU cache with auditable counters."""

    def __init__(self, maximum_entries: Optional[int] = None) -> None:
        self.maximum_entries = max(
            1,
            int(
                _env_int("RETARGET_ANATOMY_CACHE_MAX_ENTRIES", 2048)
                if maximum_entries is None
                else maximum_entries
            ),
        )
        self._items: "OrderedDict[CandidateAnatomyCacheKey, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.static_db_uses = 0
        self.runtime_evaluations = 0

    def get(self, key: CandidateAnatomyCacheKey) -> Optional[Dict[str, Any]]:
        with self._lock:
            self.requests += 1
            value = self._items.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, key: CandidateAnatomyCacheKey, value: Mapping[str, Any]) -> None:
        with self._lock:
            self._items[key] = copy.deepcopy(dict(value))
            self._items.move_to_end(key)
            while len(self._items) > self.maximum_entries:
                self._items.popitem(last=False)
                self.evictions += 1

    def note_static_use(self) -> None:
        with self._lock:
            self.static_db_uses += 1

    def note_runtime_evaluation(self) -> None:
        with self._lock:
            self.runtime_evaluations += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.requests = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.static_db_uses = 0
            self.runtime_evaluations = 0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema": "candidate_anatomy_lru_cache",
                "enabled": _env_bool("RETARGET_ANATOMY_CACHE_ENABLE", True),
                "entries": int(len(self._items)),
                "maximum_entries": int(self.maximum_entries),
                "requests": int(self.requests),
                "hits": int(self.hits),
                "misses": int(self.misses),
                "hit_rate": float(self.hits / max(1, self.requests)),
                "evictions": int(self.evictions),
                "static_db_uses": int(self.static_db_uses),
                "runtime_evaluations": int(self.runtime_evaluations),
            }


CANDIDATE_ANATOMY_CACHE = CandidateAnatomyCache()


_STATIC_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("anatomy_valid", False),
    ("anatomy_hard_valid", False),
    ("anatomy_soft_valid", True),
    ("anatomy_quality", 0.0),
    ("posture_entry", "unknown"),
    ("posture_exit", "unknown"),
    ("posture_mode", "unknown"),
    ("pelvis_height_entry_norm", 0.0),
    ("pelvis_height_exit_norm", 0.0),
    ("pelvis_height_median_norm", 0.0),
    ("body_height_entry_norm", 0.0),
    ("body_height_exit_norm", 0.0),
    ("body_height_median_norm", 0.0),
    ("entry_floor_offset_m", 0.0),
    ("exit_floor_offset_m", 0.0),
    ("torso_compression_ratio_p05", 0.0),
    ("local_angle_violation_ratio", 0.0),
    ("raw_local_angle_violation_ratio", 0.0),
    ("local_angle_severe_ratio", 0.0),
    ("self_collision_severe_ratio", 0.0),
    ("spine_cumulative_angle_p95_deg", 0.0),
)


def static_event_anatomy(
    db: Optional[Mapping[str, Any]], event_id: int
) -> Tuple[Dict[str, Any], bool]:
    feature: Dict[str, Any] = {}
    present = []
    for key, default in _STATIC_FIELDS:
        has = bool(db is not None and key in db)
        present.append(has)
        feature[key] = _db_value(db, key, event_id, default)

    distribution_raw = _db_value(
        db, "posture_distribution_json", event_id, "{}"
    )
    try:
        distribution = json.loads(str(distribution_raw))
        if not isinstance(distribution, dict):
            distribution = {}
    except Exception:
        distribution = {}
    feature["posture_distribution"] = distribution
    feature["anatomy_reasons"] = []
    feature["anatomy_soft_reasons"] = []
    feature["anatomy_valid"] = bool(feature["anatomy_valid"])
    feature["anatomy_hard_valid"] = bool(feature["anatomy_hard_valid"])
    feature["anatomy_soft_valid"] = bool(feature["anatomy_soft_valid"])
    feature["anatomy_quality"] = float(feature["anatomy_quality"])

    explicit_complete = bool(
        _db_value(db, "anatomy_static_complete", event_id, False)
    )
    complete = bool(explicit_complete or all(present))
    return feature, complete


def _stratified_anatomy_sample(
    motion: np.ndarray,
    sample_frames: int,
    fps: float,
) -> np.ndarray:
    x = np.asarray(motion, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D] motion, got {x.shape}")
    target = max(1, int(sample_frames))
    if len(x) <= target:
        return x

    edge = max(1, int(round(6.0 * float(fps) / 30.0)))
    edge = min(edge, max(1, target // 3), len(x))
    first = np.arange(0, edge, dtype=np.int64)
    last = np.arange(max(edge, len(x) - edge), len(x), dtype=np.int64)
    remaining = max(0, target - len(first) - len(last))
    middle = (
        np.linspace(edge, len(x) - edge - 1, remaining, dtype=np.int64)
        if remaining > 0 and len(x) - 2 * edge > 0
        else np.zeros(0, dtype=np.int64)
    )
    index = np.unique(np.concatenate([first, middle, last]))
    if len(index) < target:
        supplement = np.linspace(0, len(x) - 1, target, dtype=np.int64)
        index = np.unique(np.concatenate([index, supplement]))[:target]
    return x[index]


def evaluate_candidate_anatomy(
    *,
    db: Optional[Mapping[str, Any]],
    event_id: int,
    core_motion: np.ndarray,
    fps: float,
    source_frames: int,
    cache: CandidateAnatomyCache = CANDIDATE_ANATOMY_CACHE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return an anatomy decision with static-DB reuse and bounded revalidation."""

    core = np.asarray(core_motion, dtype=np.float32)
    if core.ndim != 2:
        raise ValueError(f"Expected candidate core [T,D], got {core.shape}")
    source_n = max(1, int(source_frames))
    target_n = int(len(core))
    rate = float(fps)
    warp = float(target_n / source_n)
    static_feature, static_complete = static_event_anatomy(db, int(event_id))
    tolerance = max(
        0.0,
        _env_float("RETARGET_ANATOMY_STATIC_WARP_TOLERANCE", 0.02),
    )

    if static_complete and abs(warp - 1.0) <= tolerance:
        cache.note_static_use()
        return copy.deepcopy(static_feature), {
            "schema": "candidate_anatomy_evaluation",
            "mode": "static_event_db",
            "cache_hit": False,
            "static_complete": True,
            "event_id": int(event_id),
            "source_frames": int(source_n),
            "target_core_frames": int(target_n),
            "fps": rate,
            "core_warp": warp,
            "static_warp_tolerance": tolerance,
            "sampled_frames": 0,
            "cache": cache.snapshot(),
        }

    configured_sample = max(
        8,
        _env_int("RETARGET_ANATOMY_REVALIDATION_SAMPLE_FRAMES", 32),
    )
    sample_n = min(target_n, configured_sample)
    mode = (
        "timewarp_stratified_revalidation"
        if static_complete
        else "runtime_stratified_anatomy"
    )
    key = CandidateAnatomyCacheKey.build(
        event_id=int(event_id),
        target_core_frames=target_n,
        fps=rate,
        source_frames=source_n,
        sample_frames=sample_n,
        evaluation_mode=mode,
    )

    if _env_bool("RETARGET_ANATOMY_CACHE_ENABLE", True):
        cached = cache.get(key)
        if cached is not None:
            report = dict(cached.pop("__evaluation_report__", {}))
            report.update({"cache_hit": True, "cache": cache.snapshot()})
            return cached, report

    sample = _stratified_anatomy_sample(core, sample_n, rate)
    started = time.perf_counter()
    feature = event_anatomy_features(sample, fps=rate)
    elapsed = time.perf_counter() - started
    cache.note_runtime_evaluation()
    report = {
        "schema": "candidate_anatomy_evaluation",
        "mode": mode,
        "cache_hit": False,
        "static_complete": bool(static_complete),
        "event_id": int(event_id),
        "source_frames": int(source_n),
        "target_core_frames": int(target_n),
        "fps": rate,
        "core_warp": warp,
        "static_warp_tolerance": tolerance,
        "sampled_frames": int(len(sample)),
        "runtime_seconds": float(elapsed),
    }
    payload = copy.deepcopy(feature)
    payload["__evaluation_report__"] = dict(report)
    if _env_bool("RETARGET_ANATOMY_CACHE_ENABLE", True):
        cache.put(key, payload)
    report["cache"] = cache.snapshot()
    return feature, report


class AnatomyProgressMonitor:
    """Heartbeat logger for long exact-simulation route searches."""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0
        self._state: Dict[str, Any] = {}

    def _emit(self, event: str) -> None:
        if not _env_bool("RETARGET_ANATOMY_PROGRESS_ENABLE", True):
            return
        with self._lock:
            row = dict(self._state)
        row.update(
            {
                "schema": "candidate_route_progress",
                "event": str(event),
                "elapsed_seconds": float(
                    time.perf_counter() - self._started_at
                    if self._started_at
                    else 0.0
                ),
                "cache": CANDIDATE_ANATOMY_CACHE.snapshot(),
            }
        )
        print(
            "[ANATOMY-PROGRESS] "
            + json.dumps(row, ensure_ascii=False, sort_keys=True),
            file=self.stream,
            flush=True,
        )

    def _heartbeat(self) -> None:
        interval = max(
            1.0,
            _env_float(
                "RETARGET_ANATOMY_PROGRESS_HEARTBEAT_SECONDS", 30.0
            ),
        )
        while not self._stop.wait(interval):
            self._emit("heartbeat")

    def start(self, total_slots: int, search: Mapping[str, Any]) -> None:
        self.finish(emit=False)
        with self._lock:
            self._started_at = time.perf_counter()
            self._state = {
                "stage": "route_start",
                "total_slots": int(total_slots),
                "completed_slots": 0,
                "proposal_count": 0,
                "search": dict(search),
            }
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._heartbeat,
                name="candidate-anatomy-progress",
                daemon=True,
            )
            self._thread.start()
        self._emit("route_start")

    def slot_start(
        self,
        slot: int,
        target_frames: int,
        input_states: int,
        candidate_count: int,
    ) -> None:
        with self._lock:
            self._state.update(
                {
                    "stage": "slot_search",
                    "slot": int(slot),
                    "target_frames": int(target_frames),
                    "input_states": int(input_states),
                    "candidate_count": int(candidate_count),
                }
            )
        self._emit("slot_start")

    def candidate_start(
        self,
        *,
        slot: int,
        state_index: int,
        event_id: int,
        candidate_rank: int,
        target_frames: int,
    ) -> float:
        token = time.perf_counter()
        with self._lock:
            self._state.update(
                {
                    "stage": "candidate_exact_simulation",
                    "slot": int(slot),
                    "state_index": int(state_index),
                    "event_id": int(event_id),
                    "candidate_rank": int(candidate_rank),
                    "target_frames": int(target_frames),
                }
            )
        return token

    def candidate_finish(self, token: float, safe: bool) -> None:
        with self._lock:
            self._state["proposal_count"] = int(
                self._state.get("proposal_count", 0)
            ) + 1
            self._state["last_candidate_seconds"] = float(
                time.perf_counter() - float(token)
            )
            self._state["last_candidate_safe"] = bool(safe)
        every = max(
            1,
            _env_int("RETARGET_ANATOMY_PROGRESS_EVERY_PROPOSALS", 25),
        )
        if int(self._state.get("proposal_count", 0)) % every == 0:
            self._emit("proposal_progress")

    def slot_finish(
        self, slot: int, expanded_states: int, retained_states: int
    ) -> None:
        with self._lock:
            self._state.update(
                {
                    "stage": "slot_complete",
                    "slot": int(slot),
                    "completed_slots": int(slot) + 1,
                    "expanded_states": int(expanded_states),
                    "retained_states": int(retained_states),
                }
            )
        self._emit("slot_complete")

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self._state)
        state["cache"] = CANDIDATE_ANATOMY_CACHE.snapshot()
        state["elapsed_seconds"] = float(
            time.perf_counter() - self._started_at if self._started_at else 0.0
        )
        return state

    def finish(self, emit: bool = True) -> None:
        thread = None
        with self._lock:
            if self._thread is None:
                return
            self._state["stage"] = "route_complete"
            self._stop.set()
            thread = self._thread
            self._thread = None
        if emit:
            self._emit("route_complete")
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)


ROUTE_PROGRESS = AnatomyProgressMonitor()
