"""Passive GAR/Paper 1 primitive trace contract.

This module serializes facts produced by the existing retrieval, selection,
generation, and boundary-audit path.  It owns no candidate policy, repair
operator, threshold, aggregate paper metric, or experiment runner.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np


GAR_SELECTION_TRACE_SCHEMA = "gar_selection_trace_v1"
GAR_READINESS_INTERFACE_SCHEMA = "gar_evaluation_readiness_interface_v1"
PLANNED_EVALUATION_HORIZONS = (5, 10, 20, 40)


class TraceContractError(ValueError):
    """Raised when primitive data cannot satisfy the stable trace contract."""


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_ready(item) for item in value), key=str)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_fingerprint(value: Any) -> str:
    """Hash a JSON value independently of dict insertion order and whitespace."""

    encoded = json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{canonical_fingerprint(value)[:24]}"


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _required_text(value: Any, field: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise TraceContractError(f"{field} must be a non-empty stable string")
    return text


def file_sha256(path: str | Path | None) -> Optional[str]:
    """Return a content hash when an applicable artifact exists."""

    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_bundle_fingerprint(
    checkpoints: Mapping[str, str | Path | None],
) -> Optional[str]:
    components = [
        {"role": str(role), "sha256": digest}
        for role, path in sorted(checkpoints.items())
        if (digest := file_sha256(path)) is not None
    ]
    return canonical_fingerprint(components) if components else None


def current_git_commit(repository_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(repository_root)), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TraceContractError("runtime_commit is unavailable") from exc
    return _required_text(result.stdout, "runtime_commit")


def behavior_config_fingerprint(
    config: Mapping[str, Any],
    *,
    runtime_environment: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fingerprint behavior settings while excluding trace and run provenance."""

    behavior = {
        str(key): value
        for key, value in config.items()
        if not str(key).startswith("gar_evaluation_") and not str(key).startswith("_")
    }
    environment = {
        str(key): value
        for key, value in (runtime_environment or {}).items()
        if _is_behavior_environment_key(str(key))
    }
    return canonical_fingerprint(
        {
            "motion_generation_config": behavior,
            "runtime_environment": environment,
        }
    )


_NON_BEHAVIOR_ENVIRONMENT_KEYS = frozenset(
    {
        "GENERATION_PYTHON",
        "GENERATION_SCHEDULE_RUN_ID",
    }
)
_NON_BEHAVIOR_ENVIRONMENT_PREFIXES = (
    "GENERATION_REBUILD_",
    "GENERATION_RETRAIN_",
)
_NON_BEHAVIOR_ENVIRONMENT_SUFFIXES = (
    "_CKPT",
    "_CHECKPOINT",
    "_INDEX",
    "_INDEX_JSON",
    "_INDEX_NPZ",
    "_PATH",
    "_DIR",
    "_ROOT",
    "_START_POSE",
)


def _is_behavior_environment_key(key: str) -> bool:
    """Return whether an environment key can change the evaluated decision path."""

    if key in _NON_BEHAVIOR_ENVIRONMENT_KEYS:
        return False
    if key.startswith(_NON_BEHAVIOR_ENVIRONMENT_PREFIXES):
        return False
    return not key.endswith(_NON_BEHAVIOR_ENVIRONMENT_SUFFIXES)


def _portable_audio_identity(audio: str | Path) -> Mapping[str, str]:
    digest = file_sha256(audio)
    if digest is not None:
        return {"kind": "content_sha256", "value": digest}
    portable = PurePosixPath(str(audio).replace("\\", "/")).name
    return {"kind": "portable_reference", "value": portable}


_SLOT_IDENTITY_FIELDS = (
    "start",
    "end",
    "start_sec",
    "end_sec",
    "t0",
    "t1",
    "audio_start",
    "audio_end",
    "start_frame",
    "end_frame",
    "music_start",
    "music_end",
    "start_seconds",
    "end_seconds",
    "duration",
    "duration_sec",
    "slot_duration",
    "target_frames",
)


def make_sequence_id(audio: str | Path, slots: Sequence[Mapping[str, Any]]) -> str:
    """Identify one evaluation sequence without selected-candidate information."""

    slot_contract = [
        {
            key: _json_ready(slot[key])
            for key in _SLOT_IDENTITY_FIELDS
            if key in slot and slot[key] is not None
        }
        for slot in slots
    ]
    return _stable_id(
        "seq",
        {
            "schema": GAR_SELECTION_TRACE_SCHEMA,
            "audio": _portable_audio_identity(audio),
            "slots": slot_contract,
            "event_count": len(slots),
        },
    )


def make_boundary_id(sequence_id: str, slot_index: int) -> str:
    if int(slot_index) < 1:
        raise TraceContractError("boundary slot_index must be at least 1")
    return _stable_id(
        "bnd",
        {"sequence_id": _required_text(sequence_id, "sequence_id"), "slot_index": int(slot_index)},
    )


def make_evaluation_case_id(sequence_id: str, slot_index: int) -> str:
    if int(slot_index) < 1:
        raise TraceContractError("evaluation slot_index must be at least 1")
    return _stable_id(
        "case",
        {"sequence_id": _required_text(sequence_id, "sequence_id"), "slot_index": int(slot_index)},
    )


@dataclass(frozen=True)
class CandidatePoolEntry:
    candidate_id: str
    retrieval_rank: int
    retrieval_score: Optional[float]
    source_event_id: Optional[str]
    source_recording_id: Optional[str]
    candidate_metadata_fingerprint: str

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        if int(self.retrieval_rank) < 1:
            raise TraceContractError("candidate retrieval_rank must be 1-based")
        if self.retrieval_score is not None and not math.isfinite(float(self.retrieval_score)):
            raise TraceContractError("retrieval_score must be finite or null")
        _required_text(
            self.candidate_metadata_fingerprint,
            "candidate_metadata_fingerprint",
        )


@dataclass(frozen=True)
class GARSlotCandidatePool:
    slot_index: int
    candidate_pool_size: int
    candidate_pool_fingerprint: str
    candidate_pool_manifest: tuple[CandidatePoolEntry, ...]

    def __post_init__(self) -> None:
        if int(self.slot_index) < 0:
            raise TraceContractError("candidate-pool slot_index cannot be negative")
        if int(self.candidate_pool_size) != len(self.candidate_pool_manifest):
            raise TraceContractError("slot candidate_pool_size mismatch")
        if self.candidate_pool_fingerprint != candidate_pool_fingerprint(
            self.candidate_pool_manifest
        ):
            raise TraceContractError("slot candidate_pool_fingerprint mismatch")


def candidate_pool_fingerprint(manifest: Sequence[CandidatePoolEntry | Mapping[str, Any]]) -> str:
    """Hash candidate identity and order only; scores and runtime are excluded."""

    ordered = []
    for raw in manifest:
        row = dataclasses.asdict(raw) if dataclasses.is_dataclass(raw) else dict(raw)
        ordered.append(
            {
                "candidate_id": _required_text(row.get("candidate_id"), "candidate_id"),
                "retrieval_rank": int(row.get("retrieval_rank")),
            }
        )
    expected = list(range(1, len(ordered) + 1))
    observed = [row["retrieval_rank"] for row in ordered]
    if observed != expected:
        raise TraceContractError(
            f"candidate manifest ranks must be contiguous 1-based values: {observed}"
        )
    return canonical_fingerprint(ordered)


def derive_false_safe(
    pre_risk: Optional[float],
    post_risk: Optional[float],
    risk_threshold_value: Optional[float],
) -> Optional[bool]:
    if pre_risk is None or post_risk is None or risk_threshold_value is None:
        return None
    return bool(pre_risk <= risk_threshold_value < post_risk)


def derive_recovered_after_reselection(
    initial_post_safe: Optional[bool],
    final_post_safe: Optional[bool],
    reselection_count: int,
) -> Optional[bool]:
    if initial_post_safe is None or final_post_safe is None:
        return None
    return bool(
        initial_post_safe is False
        and final_post_safe is True
        and int(reselection_count) > 0
    )


@dataclass(frozen=True)
class GARCandidateTrace:
    candidate_id: str
    candidate_rank: int
    retrieval_score: Optional[float]
    evaluation_order: int
    pre_risk: Optional[float]
    pre_risk_components: Optional[Mapping[str, Any]]
    pre_safe: Optional[bool]
    generated: bool
    post_risk: Optional[float]
    post_risk_components: Optional[Mapping[str, Any]]
    post_safe: Optional[bool]
    post_audit_failure_reasons: tuple[str, ...]
    selected_initially: bool
    selected_finally: bool
    rejected_after_post_audit: bool
    reselection_attempt_index: int
    repair_operator_id: str
    generator_id: str
    runtime_ms: Optional[float]
    risk_threshold_value: Optional[float]
    risk_threshold_source: str
    false_safe: Optional[bool]

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        if int(self.candidate_rank) < 1:
            raise TraceContractError("candidate_rank must be 1-based")
        if int(self.evaluation_order) < 1:
            raise TraceContractError("evaluation_order must be 1-based")
        if int(self.reselection_attempt_index) < 0:
            raise TraceContractError("reselection_attempt_index cannot be negative")
        if not self.generated and (
            self.post_risk is not None
            or self.post_safe is not None
            or self.post_risk_components is not None
        ):
            raise TraceContractError("non-generated candidate cannot have post-audit values")
        expected_false_safe = derive_false_safe(
            self.pre_risk, self.post_risk, self.risk_threshold_value
        )
        if self.false_safe != expected_false_safe:
            raise TraceContractError("false_safe does not match complete primitive data")
        for field_name in ("pre_risk", "post_risk", "runtime_ms"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(float(value)):
                raise TraceContractError(f"{field_name} must be finite or null")


@dataclass(frozen=True)
class BoundaryMetrics:
    fk_position_jump: Optional[float] = None
    so3_rotation_geodesic_jump: Optional[float] = None
    velocity_jump: Optional[float] = None
    acceleration_jump: Optional[float] = None
    boundary_jerk_mean: Optional[float] = None
    boundary_jerk_p95: Optional[float] = None
    foot_skate: Optional[float] = None
    penetration_mean: Optional[float] = None
    penetration_max: Optional[float] = None
    contact_discontinuity: Optional[float] = None
    root_velocity_discontinuity: Optional[float] = None
    heading_discontinuity: Optional[float] = None


def _max_present(values: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    parsed = [_optional_float(values.get(key)) for key in keys]
    return max(parsed) if parsed and all(value is not None for value in parsed) else None


def boundary_metrics_from_authoritative_risk(risk: Mapping[str, Any]) -> BoundaryMetrics:
    """Map only semantically matching metrics already computed by production."""

    return BoundaryMetrics(
        fk_position_jump=_max_present(risk, ("entry_fk_jump", "exit_fk_jump")),
        so3_rotation_geodesic_jump=_max_present(
            risk, ("entry_rotation_step_rad", "exit_rotation_step_rad")
        ),
        velocity_jump=_max_present(risk, ("entry_velocity", "exit_velocity")),
        acceleration_jump=_max_present(
            risk, ("entry_acceleration", "exit_acceleration")
        ),
        # Production exposes a maximum jerk, not mean/p95 jerk.
        boundary_jerk_mean=None,
        boundary_jerk_p95=None,
        foot_skate=_optional_float(risk.get("foot_slip_p95")),
        # Existing foot_penetration is mean squared depth (m^2), not mean depth.
        penetration_mean=None,
        penetration_max=_optional_float(risk.get("foot_penetration_max_m")),
        contact_discontinuity=_optional_float(risk.get("contact_switch")),
        root_velocity_discontinuity=None,
        heading_discontinuity=None,
    )


@dataclass(frozen=True)
class GARBoundarySummary:
    candidate_pool_size: int
    candidates_evaluated: int
    initial_candidate_id: Optional[str]
    initial_candidate_rank: Optional[int]
    final_candidate_id: Optional[str]
    final_candidate_rank: Optional[int]
    reselection_count: int
    initial_post_safe: Optional[bool]
    final_post_safe: Optional[bool]
    initial_post_risk: Optional[float]
    final_post_risk: Optional[float]
    recovered_after_reselection: Optional[bool]
    boundary_hard_failure: bool
    hard_failure: bool
    failure_reasons: tuple[str, ...]
    selection_policy_id: str
    repair_operator_id: str
    generator_id: str
    exhaustive_topk_evaluated: bool

    def __post_init__(self) -> None:
        size = int(self.candidate_pool_size)
        if size < 1:
            raise TraceContractError("candidate_pool_size must be positive")
        for name in ("initial_candidate_rank", "final_candidate_rank"):
            rank = getattr(self, name)
            if rank is not None and not 1 <= int(rank) <= size:
                raise TraceContractError(f"{name} must be within the candidate pool")
        expected = derive_recovered_after_reselection(
            self.initial_post_safe,
            self.final_post_safe,
            self.reselection_count,
        )
        if self.recovered_after_reselection != expected:
            raise TraceContractError("recovered_after_reselection is inconsistent")
        if self.exhaustive_topk_evaluated:
            raise TraceContractError(
                "readiness instrumentation cannot assert exhaustive Top-K evaluation"
            )
        if bool(self.hard_failure) != bool(self.boundary_hard_failure):
            raise TraceContractError("hard_failure aliases must agree")
        if bool(self.hard_failure) != bool(self.failure_reasons):
            raise TraceContractError(
                "hard_failure must be derived from authoritative failure_reasons"
            )


@dataclass(frozen=True)
class GARBoundaryTrace:
    schema_version: str
    runtime_commit: str
    config_fingerprint: str
    retrieval_index_fingerprint: str
    generator_id: str
    generator_version: str
    generator_checkpoint_fingerprint: Optional[str]
    generator_config_fingerprint: str
    repair_operator_id: str
    repair_operator_version: str
    repair_config_fingerprint: str
    selection_policy_id: str
    method_variant_id: str
    random_seed: Optional[int]
    sequence_id: str
    slot_index: int
    boundary_id: str
    evaluation_case_id: str
    candidate_pool_fingerprint: str
    candidate_pool_manifest: tuple[CandidatePoolEntry, ...]
    candidate_trace: tuple[GARCandidateTrace, ...]
    risk_threshold_value: Optional[float]
    risk_threshold_source: str
    risk_thresholds: Mapping[str, Any]
    boundary_metrics: BoundaryMetrics
    runtime: Mapping[str, Optional[float]]
    summary: GARBoundarySummary

    def __post_init__(self) -> None:
        if self.schema_version != GAR_SELECTION_TRACE_SCHEMA:
            raise TraceContractError(f"unsupported schema_version={self.schema_version!r}")
        if int(self.slot_index) < 1:
            raise TraceContractError("boundary slot_index must be at least 1")
        if self.candidate_pool_fingerprint != candidate_pool_fingerprint(
            self.candidate_pool_manifest
        ):
            raise TraceContractError("candidate_pool_fingerprint mismatch")
        rank_by_id = {
            entry.candidate_id: entry.retrieval_rank
            for entry in self.candidate_pool_manifest
        }
        if len(rank_by_id) != len(self.candidate_pool_manifest):
            raise TraceContractError("candidate pool contains duplicate candidate IDs")
        pool_size = len(self.candidate_pool_manifest)
        for item in self.candidate_trace:
            if item.candidate_id not in rank_by_id:
                raise TraceContractError("candidate trace references an item outside the pool")
            if int(item.candidate_rank) > pool_size:
                raise TraceContractError("candidate rank exceeds candidate_pool_size")
            if int(item.candidate_rank) != int(rank_by_id[item.candidate_id]):
                raise TraceContractError(
                    "candidate trace rank does not match the stable pool manifest"
                )
        if int(self.summary.candidate_pool_size) != pool_size:
            raise TraceContractError("summary candidate_pool_size mismatch")


@dataclass(frozen=True)
class GARSequenceSummary:
    sequence_id: str
    sequence_event_count: int
    sequence_boundary_count: int
    completed: bool
    hard_failure_count: int
    sequence_runtime_ms: Optional[float]
    boundary_ids: tuple[str, ...]
    generator_id: str
    repair_operator_id: str

    def __post_init__(self) -> None:
        if int(self.sequence_boundary_count) != len(self.boundary_ids):
            raise TraceContractError("sequence boundary count/list mismatch")
        if int(self.sequence_boundary_count) != max(0, int(self.sequence_event_count) - 1):
            raise TraceContractError("sequence event/boundary count mismatch")


@dataclass(frozen=True)
class GARSelectionTrace:
    schema_version: str
    runtime_commit: str
    config_fingerprint: str
    candidate_pool_fingerprint: str
    retrieval_index_fingerprint: str
    generator_id: str
    generator_version: str
    generator_checkpoint_fingerprint: Optional[str]
    generator_config_fingerprint: str
    repair_operator_id: str
    repair_operator_version: str
    repair_config_fingerprint: str
    selection_policy_id: str
    method_variant_id: str
    random_seed: Optional[int]
    capabilities: Mapping[str, bool]
    runtime: Mapping[str, Optional[float]]
    experiment_status: Mapping[str, bool]
    planned_evaluation_horizons: tuple[int, ...]
    sequence: GARSequenceSummary
    candidate_pools: tuple[GARSlotCandidatePool, ...]
    boundaries: tuple[GARBoundaryTrace, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GAR_SELECTION_TRACE_SCHEMA:
            raise TraceContractError(f"unsupported schema_version={self.schema_version!r}")
        expected_boundary_ids = tuple(boundary.boundary_id for boundary in self.boundaries)
        if self.sequence.boundary_ids != expected_boundary_ids:
            raise TraceContractError("sequence boundary IDs do not match boundary traces")
        expected_pool = canonical_fingerprint(
            [
                {
                    "slot_index": pool.slot_index,
                    "candidate_pool_fingerprint": pool.candidate_pool_fingerprint,
                }
                for pool in self.candidate_pools
            ]
        )
        if self.candidate_pool_fingerprint != expected_pool:
            raise TraceContractError("sequence candidate_pool_fingerprint mismatch")
        required_false = (
            "paper1_experiments_implemented",
            "oracle_implemented",
            "statistical_tests_implemented",
            "long_horizon_benchmark_implemented",
            "production_selection_behavior_changed",
        )
        for key in required_false:
            if bool(self.experiment_status.get(key, True)):
                raise TraceContractError(f"readiness-only contract requires {key}=false")
        pool_by_slot = {pool.slot_index: pool for pool in self.candidate_pools}
        if tuple(sorted(pool_by_slot)) != tuple(range(self.sequence.sequence_event_count)):
            raise TraceContractError("candidate pools must cover every sequence slot")
        for boundary in self.boundaries:
            slot_pool = pool_by_slot.get(boundary.slot_index)
            if slot_pool is None or (
                slot_pool.candidate_pool_fingerprint
                != boundary.candidate_pool_fingerprint
            ):
                raise TraceContractError("boundary candidate pool does not match slot pool")
            boundary_provenance = (
                boundary.runtime_commit,
                boundary.config_fingerprint,
                boundary.retrieval_index_fingerprint,
                boundary.generator_id,
                boundary.generator_version,
                boundary.generator_checkpoint_fingerprint,
                boundary.generator_config_fingerprint,
                boundary.repair_operator_id,
                boundary.repair_operator_version,
                boundary.repair_config_fingerprint,
                boundary.selection_policy_id,
                boundary.method_variant_id,
                boundary.random_seed,
            )
            sequence_provenance = (
                self.runtime_commit,
                self.config_fingerprint,
                self.retrieval_index_fingerprint,
                self.generator_id,
                self.generator_version,
                self.generator_checkpoint_fingerprint,
                self.generator_config_fingerprint,
                self.repair_operator_id,
                self.repair_operator_version,
                self.repair_config_fingerprint,
                self.selection_policy_id,
                self.method_variant_id,
                self.random_seed,
            )
            if boundary_provenance != sequence_provenance:
                raise TraceContractError(
                    "boundary provenance does not match its sequence trace"
                )

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_ready(self))

    @staticmethod
    def from_dict(value: Mapping[str, Any]) -> "GARSelectionTrace":
        candidate_pools = []
        for raw_pool in value.get("candidate_pools", []):
            pool = dict(raw_pool)
            manifest = tuple(
                CandidatePoolEntry(**dict(row))
                for row in pool.pop("candidate_pool_manifest", [])
            )
            candidate_pools.append(
                GARSlotCandidatePool(**pool, candidate_pool_manifest=manifest)
            )
        boundaries = []
        for raw_boundary in value.get("boundaries", []):
            boundary = dict(raw_boundary)
            manifest = tuple(
                CandidatePoolEntry(**dict(row))
                for row in boundary.pop("candidate_pool_manifest", [])
            )
            candidates = tuple(
                GARCandidateTrace(
                    **{
                        **dict(row),
                        "post_audit_failure_reasons": tuple(
                            row.get("post_audit_failure_reasons", [])
                        ),
                    }
                )
                for row in boundary.pop("candidate_trace", [])
            )
            metrics = BoundaryMetrics(**dict(boundary.pop("boundary_metrics")))
            summary_value = dict(boundary.pop("summary"))
            summary_value["failure_reasons"] = tuple(
                summary_value.get("failure_reasons", [])
            )
            summary = GARBoundarySummary(**summary_value)
            boundaries.append(
                GARBoundaryTrace(
                    **boundary,
                    candidate_pool_manifest=manifest,
                    candidate_trace=candidates,
                    boundary_metrics=metrics,
                    summary=summary,
                )
            )
        sequence = dict(value.get("sequence", {}))
        sequence["boundary_ids"] = tuple(sequence.get("boundary_ids", []))
        top = dict(value)
        top.pop("boundaries", None)
        top.pop("sequence", None)
        top.pop("candidate_pools", None)
        top["planned_evaluation_horizons"] = tuple(
            top.get("planned_evaluation_horizons", [])
        )
        return GARSelectionTrace(
            **top,
            sequence=GARSequenceSummary(**sequence),
            candidate_pools=tuple(candidate_pools),
            boundaries=tuple(boundaries),
        )


def write_trace(trace: GARSelectionTrace, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            trace.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_trace(path: str | Path) -> GARSelectionTrace:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TraceContractError("trace root must be a JSON object")
    return GARSelectionTrace.from_dict(value)


def _db_value(db: Mapping[str, Any], key: str, index: int) -> Any:
    try:
        value = np.asarray(db[key], dtype=object)[int(index)]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _score_lookup(
    retrieval_row: Mapping[str, Any],
) -> dict[int, Optional[float]]:
    indices = list(retrieval_row.get("candidate_event_indices", []) or [])
    scores = list(retrieval_row.get("candidate_router_probabilities", []) or [])
    return {
        int(index): _optional_float(scores[position]) if position < len(scores) else None
        for position, index in enumerate(indices)
    }


def _candidate_manifest(
    db: Mapping[str, Any],
    event_uids: Sequence[Any],
    candidate_indices: Sequence[int],
    retrieval_row: Mapping[str, Any],
) -> tuple[CandidatePoolEntry, ...]:
    score_by_index = _score_lookup(retrieval_row)
    entries = []
    for rank, raw_index in enumerate(candidate_indices, start=1):
        index = int(raw_index)
        candidate_id = _required_text(event_uids[index], "event_uid")
        recording = _db_value(db, "recording_uids", index)
        source = _db_value(db, "source_uids", index)
        metadata = {
            "candidate_id": candidate_id,
            "source_uid": None if source is None else str(source),
            "recording_uid": None if recording is None else str(recording),
            "source_sequence_id": (
                None
                if _db_value(db, "sequence_ids", index) is None
                else str(_db_value(db, "sequence_ids", index))
            ),
        }
        entries.append(
            CandidatePoolEntry(
                candidate_id=candidate_id,
                retrieval_rank=rank,
                retrieval_score=score_by_index.get(index),
                source_event_id=candidate_id,
                source_recording_id=None if recording is None else str(recording),
                candidate_metadata_fingerprint=canonical_fingerprint(metadata),
            )
        )
    return tuple(entries)


def _rows_by_slot(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row.get("slot", index)): row for index, row in enumerate(rows)}


def _selected_row(
    record: Mapping[str, Any], slot_index: int
) -> tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]:
    assembly = _rows_by_slot(record.get("assembly_report", []) or [])
    boundaries = _rows_by_slot(record.get("boundary_rows", []) or [])
    if slot_index not in assembly:
        raise TraceContractError(f"round has no assembly row for slot {slot_index}")
    return assembly[slot_index], boundaries.get(slot_index)


def _candidate_trace_for_boundary(
    *,
    slot_index: int,
    event_uids: Sequence[Any],
    manifest: Sequence[CandidatePoolEntry],
    round_records: Sequence[Mapping[str, Any]],
    final_round: int,
    repair_operator_id: str,
    generator_id: str,
    risk_threshold_value: Optional[float],
    risk_threshold_source: str,
) -> tuple[GARCandidateTrace, ...]:
    id_by_event_index = {
        int(index): str(event_uids[int(index)]) for index in range(len(event_uids))
    }
    rank_by_id = {entry.candidate_id: entry.retrieval_rank for entry in manifest}
    score_by_id = {entry.candidate_id: entry.retrieval_score for entry in manifest}
    traces = []
    evaluation_order = 0
    first_round = int(round_records[0].get("round", 0))

    for record in round_records:
        attempt = int(record.get("round", 0))
        assembly, post_row = _selected_row(record, slot_index)
        selected_index = int(assembly.get("event_id"))
        selected_id = id_by_event_index.get(selected_index)
        if selected_id not in rank_by_id:
            raise TraceContractError(
                f"selected candidate {selected_id!r} is outside slot {slot_index} pool"
            )
        trials = [dict(item) for item in (assembly.get("candidate_trials", []) or [])]
        if selected_index not in {int(item.get("event_id", -1)) for item in trials}:
            trials.append(
                {
                    "event_id": selected_index,
                    "risk_score": assembly.get("risk_score_predicted"),
                    "risk": assembly.get("risk_predicted"),
                    "safe": assembly.get("safe_predicted"),
                }
            )
        for trial in trials:
            event_index = int(trial.get("event_id", -1))
            candidate_id = id_by_event_index.get(event_index)
            if candidate_id not in rank_by_id:
                raise TraceContractError(
                    f"evaluated candidate {candidate_id!r} is outside slot {slot_index} pool"
                )
            evaluation_order += 1
            generated = bool(event_index == selected_index)
            pre_risk = _optional_float(trial.get("risk_score"))
            pre_components_raw = trial.get("risk")
            pre_components = (
                dict(_json_ready(pre_components_raw))
                if isinstance(pre_components_raw, Mapping)
                else None
            )
            pre_safe = trial.get("safe")
            pre_safe = bool(pre_safe) if isinstance(pre_safe, (bool, np.bool_)) else None
            post_risk = None
            post_components = None
            post_safe = None
            reasons: tuple[str, ...] = ()
            if generated and post_row is not None:
                post_risk = _optional_float(post_row.get("actual_risk_score"))
                raw_post = post_row.get("risk")
                post_components = (
                    dict(_json_ready(raw_post)) if isinstance(raw_post, Mapping) else None
                )
                if "safe" in post_row:
                    post_safe = _optional_bool(post_row.get("safe"))
                reasons = tuple(str(item) for item in post_row.get("failure_reasons", []) or [])
            selected_initially = bool(generated and attempt == first_round)
            selected_finally = bool(generated and attempt == int(final_round))
            traces.append(
                GARCandidateTrace(
                    candidate_id=candidate_id,
                    candidate_rank=int(rank_by_id[candidate_id]),
                    retrieval_score=score_by_id[candidate_id],
                    evaluation_order=evaluation_order,
                    pre_risk=pre_risk,
                    pre_risk_components=pre_components,
                    pre_safe=pre_safe,
                    generated=generated,
                    post_risk=post_risk,
                    post_risk_components=post_components,
                    post_safe=post_safe,
                    post_audit_failure_reasons=reasons,
                    selected_initially=selected_initially,
                    selected_finally=selected_finally,
                    rejected_after_post_audit=bool(
                        generated and post_safe is False and not selected_finally
                    ),
                    reselection_attempt_index=attempt,
                    repair_operator_id=repair_operator_id,
                    generator_id=generator_id,
                    runtime_ms=None,
                    risk_threshold_value=risk_threshold_value,
                    risk_threshold_source=risk_threshold_source,
                    false_safe=derive_false_safe(
                        pre_risk, post_risk, risk_threshold_value
                    ),
                )
            )
    return tuple(traces)


def build_closed_loop_trace(
    *,
    audio: str | Path,
    slots: Sequence[Mapping[str, Any]],
    db: Mapping[str, Any],
    event_uids: Sequence[Any],
    candidate_lists: Sequence[Sequence[int]],
    retrieval_report: Sequence[Mapping[str, Any]],
    round_records: Sequence[Mapping[str, Any]],
    final_round: int,
    runtime_commit: str,
    config_fingerprint: str,
    retrieval_index_fingerprint: str,
    generator_id: str,
    generator_version: str,
    generator_checkpoint_fingerprint: Optional[str],
    generator_config_fingerprint: str,
    repair_operator_id: str,
    repair_operator_version: str,
    repair_config_fingerprint: str,
    selection_policy_id: str,
    method_variant_id: str,
    random_seed: Optional[int],
    risk_threshold_value: Optional[float],
    risk_threshold_source: str,
    risk_thresholds: Mapping[str, Any],
    capabilities: Mapping[str, bool],
    runtime: Mapping[str, Optional[float]],
) -> GARSelectionTrace:
    """Project completed production facts into the passive readiness schema."""

    if not round_records:
        raise TraceContractError("at least one completed selection round is required")
    if len(candidate_lists) != len(slots):
        raise TraceContractError("candidate pool count must match sequence event count")
    if len(retrieval_report) != len(slots):
        raise TraceContractError("retrieval report count must match sequence event count")
    records = tuple(sorted(round_records, key=lambda row: int(row.get("round", 0))))
    final_record = next(
        (row for row in records if int(row.get("round", -1)) == int(final_round)),
        None,
    )
    if final_record is None:
        raise TraceContractError(f"final_round={final_round} is absent from round records")

    sequence_id = make_sequence_id(audio, slots)
    slot_candidate_pools = tuple(
        GARSlotCandidatePool(
            slot_index=slot_index,
            candidate_pool_size=len(manifest),
            candidate_pool_fingerprint=candidate_pool_fingerprint(manifest),
            candidate_pool_manifest=manifest,
        )
        for slot_index in range(len(slots))
        for manifest in (
            _candidate_manifest(
                db,
                event_uids,
                candidate_lists[slot_index],
                retrieval_report[slot_index],
            ),
        )
    )
    boundaries = []
    for slot_index in range(1, len(slots)):
        manifest = slot_candidate_pools[slot_index].candidate_pool_manifest
        pool_fingerprint = slot_candidate_pools[slot_index].candidate_pool_fingerprint
        rank_by_id = {entry.candidate_id: entry.retrieval_rank for entry in manifest}
        initial_assembly, initial_post = _selected_row(records[0], slot_index)
        final_assembly, final_post = _selected_row(final_record, slot_index)
        if initial_post is None or final_post is None:
            raise TraceContractError(
                f"slot {slot_index} lacks authoritative post-generation boundary audit"
            )
        initial_id = str(event_uids[int(initial_assembly.get("event_id"))])
        final_id = str(event_uids[int(final_assembly.get("event_id"))])
        if initial_id not in rank_by_id or final_id not in rank_by_id:
            raise TraceContractError("initial/final candidate is outside the stable pool")
        initial_safe = _optional_bool(initial_post.get("safe"))
        final_safe = _optional_bool(final_post.get("safe"))
        if final_safe is None:
            raise TraceContractError(
                f"slot {slot_index} lacks an authoritative final safe decision"
            )
        final_failure_reasons = tuple(
            str(item) for item in final_post.get("failure_reasons", []) or []
        )
        if final_safe == bool(final_failure_reasons):
            raise TraceContractError(
                "authoritative boundary safe/failure_reasons are inconsistent"
            )
        initial_risk = _optional_float(initial_post.get("actual_risk_score"))
        final_risk = _optional_float(final_post.get("actual_risk_score"))
        # Count every post-audit reselection attempt, including an unsuccessful
        # later round when production ultimately keeps an earlier best payload.
        reselection_count = max(0, len(records) - 1)
        candidate_trace = _candidate_trace_for_boundary(
            slot_index=slot_index,
            event_uids=event_uids,
            manifest=manifest,
            round_records=records,
            final_round=final_round,
            repair_operator_id=repair_operator_id,
            generator_id=generator_id,
            risk_threshold_value=risk_threshold_value,
            risk_threshold_source=risk_threshold_source,
        )
        summary = GARBoundarySummary(
            candidate_pool_size=len(manifest),
            candidates_evaluated=len(candidate_trace),
            initial_candidate_id=initial_id,
            initial_candidate_rank=rank_by_id[initial_id],
            final_candidate_id=final_id,
            final_candidate_rank=rank_by_id[final_id],
            reselection_count=reselection_count,
            initial_post_safe=initial_safe,
            final_post_safe=final_safe,
            initial_post_risk=initial_risk,
            final_post_risk=final_risk,
            recovered_after_reselection=derive_recovered_after_reselection(
                initial_safe, final_safe, reselection_count
            ),
            boundary_hard_failure=not final_safe,
            hard_failure=not final_safe,
            failure_reasons=final_failure_reasons,
            selection_policy_id=selection_policy_id,
            repair_operator_id=repair_operator_id,
            generator_id=generator_id,
            exhaustive_topk_evaluated=False,
        )
        boundary_id = make_boundary_id(sequence_id, slot_index)
        boundaries.append(
            GARBoundaryTrace(
                schema_version=GAR_SELECTION_TRACE_SCHEMA,
                runtime_commit=runtime_commit,
                config_fingerprint=config_fingerprint,
                retrieval_index_fingerprint=retrieval_index_fingerprint,
                generator_id=generator_id,
                generator_version=generator_version,
                generator_checkpoint_fingerprint=generator_checkpoint_fingerprint,
                generator_config_fingerprint=generator_config_fingerprint,
                repair_operator_id=repair_operator_id,
                repair_operator_version=repair_operator_version,
                repair_config_fingerprint=repair_config_fingerprint,
                selection_policy_id=selection_policy_id,
                method_variant_id=method_variant_id,
                random_seed=random_seed,
                sequence_id=sequence_id,
                slot_index=slot_index,
                boundary_id=boundary_id,
                evaluation_case_id=make_evaluation_case_id(sequence_id, slot_index),
                candidate_pool_fingerprint=pool_fingerprint,
                candidate_pool_manifest=manifest,
                candidate_trace=candidate_trace,
                risk_threshold_value=risk_threshold_value,
                risk_threshold_source=risk_threshold_source,
                risk_thresholds=dict(_json_ready(risk_thresholds)),
                boundary_metrics=boundary_metrics_from_authoritative_risk(
                    dict(final_post.get("risk", {}))
                ),
                runtime={
                    "retrieval_runtime_ms": None,
                    "candidate_simulation_runtime_ms": None,
                    "generation_runtime_ms": None,
                    "post_audit_runtime_ms": None,
                    "reselection_runtime_ms": None,
                    "boundary_total_runtime_ms": None,
                },
                summary=summary,
            )
        )

    boundary_tuple = tuple(boundaries)
    sequence_pool_fingerprint = canonical_fingerprint(
        [
            {
                "slot_index": pool.slot_index,
                "candidate_pool_fingerprint": pool.candidate_pool_fingerprint,
            }
            for pool in slot_candidate_pools
        ]
    )
    sequence_runtime = _optional_float(runtime.get("sequence_total_runtime_ms"))
    sequence = GARSequenceSummary(
        sequence_id=sequence_id,
        sequence_event_count=len(slots),
        sequence_boundary_count=max(0, len(slots) - 1),
        completed=True,
        hard_failure_count=sum(
            int(boundary.summary.boundary_hard_failure) for boundary in boundary_tuple
        ),
        sequence_runtime_ms=sequence_runtime,
        boundary_ids=tuple(boundary.boundary_id for boundary in boundary_tuple),
        generator_id=generator_id,
        repair_operator_id=repair_operator_id,
    )
    return GARSelectionTrace(
        schema_version=GAR_SELECTION_TRACE_SCHEMA,
        runtime_commit=runtime_commit,
        config_fingerprint=config_fingerprint,
        candidate_pool_fingerprint=sequence_pool_fingerprint,
        retrieval_index_fingerprint=retrieval_index_fingerprint,
        generator_id=generator_id,
        generator_version=generator_version,
        generator_checkpoint_fingerprint=generator_checkpoint_fingerprint,
        generator_config_fingerprint=generator_config_fingerprint,
        repair_operator_id=repair_operator_id,
        repair_operator_version=repair_operator_version,
        repair_config_fingerprint=repair_config_fingerprint,
        selection_policy_id=selection_policy_id,
        method_variant_id=method_variant_id,
        random_seed=random_seed,
        capabilities={str(key): bool(value) for key, value in capabilities.items()},
        runtime={str(key): _optional_float(value) for key, value in runtime.items()},
        experiment_status={
            "paper1_experiments_implemented": False,
            "oracle_implemented": False,
            "statistical_tests_implemented": False,
            "long_horizon_benchmark_implemented": False,
            "production_selection_behavior_changed": False,
        },
        planned_evaluation_horizons=PLANNED_EVALUATION_HORIZONS,
        sequence=sequence,
        candidate_pools=slot_candidate_pools,
        boundaries=boundary_tuple,
    )
