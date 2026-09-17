"""Auditable multi-seed counterfactual Outcome Bank for repairability learning."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from model.repairability_predictor import FEATURE_NAMES, VIOLATION_FAMILIES


OUTCOME_BANK_RECORD_SCHEMA = "repairability_outcome_record_v1"
OUTCOME_BANK_SUMMARY_SCHEMA = "repairability_outcome_bank_summary_v1"


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reason_families(reasons: Sequence[str]) -> tuple[str, ...]:
    result: set[str] = set()
    for raw in reasons:
        reason = str(raw).strip().lower()
        if "velocity" in reason:
            result.add("velocity")
        elif "acceleration" in reason:
            result.add("acceleration")
        elif "jerk" in reason:
            result.add("jerk")
        elif "rotation" in reason or "angular" in reason:
            result.add("rotation")
        elif "fk" in reason or "position_jump" in reason:
            result.add("fk")
        elif "slip" in reason or "skate" in reason:
            result.add("foot_slip")
        elif "penetration" in reason:
            result.add("penetration")
        elif "contact" in reason or "support" in reason:
            result.add("contact")
        else:
            result.add("physical_other")
    return tuple(name for name in VIOLATION_FAMILIES if name in result)


@dataclass(frozen=True)
class OutcomeRecord:
    schema: str
    sequence_id: str
    boundary_id: str
    evaluation_case_id: str
    group_id: str
    slot_index: int
    candidate_id: str
    candidate_event_index: int
    source_recording_id: Optional[str]
    source_performer_id: Optional[str]
    original_rank: int
    candidate_pool_size: int
    random_seed: int
    features: Mapping[str, float]
    pre_safe: bool
    pre_risk: float
    boundary_safe: bool
    physical_safe: bool
    post_safe: bool
    post_risk: float
    failure_reasons: tuple[str, ...]
    violation_families: tuple[str, ...]
    runtime_ms: float
    runtime_commit: str
    config_fingerprint: str
    candidate_pool_fingerprint: str
    generator_fingerprint: str
    repair_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema != OUTCOME_BANK_RECORD_SCHEMA:
            raise ValueError("unsupported Outcome Bank record schema")
        for name in (
            "sequence_id",
            "boundary_id",
            "evaluation_case_id",
            "group_id",
            "candidate_id",
            "runtime_commit",
            "config_fingerprint",
            "candidate_pool_fingerprint",
            "generator_fingerprint",
            "repair_fingerprint",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"Outcome Bank field {name} cannot be empty")
        if int(self.slot_index) < 1:
            raise ValueError("Outcome Bank slot_index must be a real boundary")
        if int(self.original_rank) < 0 or int(self.original_rank) >= int(
            self.candidate_pool_size
        ):
            raise ValueError("Outcome Bank original rank is outside the pool")
        if set(self.features) != set(FEATURE_NAMES):
            raise ValueError("Outcome Bank feature schema mismatch")
        if not all(math.isfinite(float(value)) for value in self.features.values()):
            raise ValueError("Outcome Bank features must be finite")
        for value_name in ("pre_risk", "post_risk", "runtime_ms"):
            if not math.isfinite(float(getattr(self, value_name))):
                raise ValueError(f"Outcome Bank {value_name} must be finite")
        expected = reason_families(self.failure_reasons)
        if tuple(self.violation_families) != expected:
            raise ValueError("Outcome Bank violation families do not match reasons")
        if bool(self.post_safe) != (len(self.failure_reasons) == 0):
            raise ValueError("Outcome Bank post_safe must be derived from Guard reasons")
        if bool(self.post_safe) != (
            bool(self.boundary_safe) and bool(self.physical_safe)
        ):
            raise ValueError("Outcome Bank post_safe must combine boundary and physical gates")

    @property
    def key(self) -> tuple[str, str, int]:
        return self.evaluation_case_id, self.candidate_id, int(self.random_seed)

    def as_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["features"] = dict(self.features)
        value["failure_reasons"] = list(self.failure_reasons)
        value["violation_families"] = list(self.violation_families)
        return value


def outcome_record_from_mapping(value: Mapping[str, Any]) -> OutcomeRecord:
    return OutcomeRecord(
        schema=str(value.get("schema", "")),
        sequence_id=str(value.get("sequence_id", "")),
        boundary_id=str(value.get("boundary_id", "")),
        evaluation_case_id=str(value.get("evaluation_case_id", "")),
        group_id=str(value.get("group_id", "")),
        slot_index=int(value.get("slot_index", -1)),
        candidate_id=str(value.get("candidate_id", "")),
        candidate_event_index=int(value.get("candidate_event_index", -1)),
        source_recording_id=(
            None
            if value.get("source_recording_id") in (None, "")
            else str(value.get("source_recording_id"))
        ),
        source_performer_id=(
            None
            if value.get("source_performer_id") in (None, "")
            else str(value.get("source_performer_id"))
        ),
        original_rank=int(value.get("original_rank", -1)),
        candidate_pool_size=int(value.get("candidate_pool_size", 0)),
        random_seed=int(value.get("random_seed", -1)),
        features={str(k): float(v) for k, v in dict(value.get("features", {})).items()},
        pre_safe=bool(value.get("pre_safe", False)),
        pre_risk=float(value.get("pre_risk", 0.0)),
        boundary_safe=bool(value.get("boundary_safe", False)),
        physical_safe=bool(value.get("physical_safe", False)),
        post_safe=bool(value.get("post_safe", False)),
        post_risk=float(value.get("post_risk", 0.0)),
        failure_reasons=tuple(str(item) for item in value.get("failure_reasons", ())),
        violation_families=tuple(
            str(item) for item in value.get("violation_families", ())
        ),
        runtime_ms=float(value.get("runtime_ms", 0.0)),
        runtime_commit=str(value.get("runtime_commit", "")),
        config_fingerprint=str(value.get("config_fingerprint", "")),
        candidate_pool_fingerprint=str(value.get("candidate_pool_fingerprint", "")),
        generator_fingerprint=str(value.get("generator_fingerprint", "")),
        repair_fingerprint=str(value.get("repair_fingerprint", "")),
    )


def read_outcome_bank(path: str | Path) -> list[OutcomeRecord]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[OutcomeRecord] = []
    seen: set[tuple[str, str, int]] = set()
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = outcome_record_from_mapping(json.loads(raw))
        except Exception as exc:
            raise ValueError(f"invalid Outcome Bank line {line_number}: {exc}") from exc
        if record.key in seen:
            raise ValueError(f"duplicate Outcome Bank trial key at line {line_number}")
        seen.add(record.key)
        records.append(record)
    return records


class OutcomeBankWriter:
    """Append-only, resumable writer with duplicate and provenance guards."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        existing = read_outcome_bank(self.path)
        self.records_by_key = {record.key: record for record in existing}
        self.keys = {record.key for record in existing}
        self.provenance = {
            (
                record.runtime_commit,
                record.config_fingerprint,
                record.generator_fingerprint,
                record.repair_fingerprint,
            )
            for record in existing
        }
        if len(self.provenance) > 1:
            raise ValueError("Outcome Bank mixes incompatible execution provenance")

    def contains(self, key: tuple[str, str, int]) -> bool:
        return key in self.keys

    def get(self, key: tuple[str, str, int]) -> Optional[OutcomeRecord]:
        return self.records_by_key.get(key)

    def append(self, record: OutcomeRecord) -> bool:
        if record.key in self.keys:
            return False
        provenance = (
            record.runtime_commit,
            record.config_fingerprint,
            record.generator_fingerprint,
            record.repair_fingerprint,
        )
        if self.provenance and provenance not in self.provenance:
            raise ValueError("refusing to mix Outcome Bank execution provenance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    record.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self.keys.add(record.key)
        self.records_by_key[record.key] = record
        self.provenance.add(provenance)
        return True


def summarize_outcome_bank(
    records: Sequence[OutcomeRecord], *, required_seeds: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    groups: Dict[tuple[str, str], list[OutcomeRecord]] = {}
    for record in records:
        groups.setdefault((record.evaluation_case_id, record.candidate_id), []).append(
            record
        )
    expected = None if required_seeds is None else {int(value) for value in required_seeds}
    incomplete: list[Dict[str, Any]] = []
    heterogeneous = 0
    boundary_outcomes: Dict[str, set[bool]] = {}
    for (case_id, candidate_id), rows in groups.items():
        seeds = {int(row.random_seed) for row in rows}
        if expected is not None and seeds != expected:
            incomplete.append(
                {
                    "evaluation_case_id": case_id,
                    "candidate_id": candidate_id,
                    "observed_seeds": sorted(seeds),
                    "expected_seeds": sorted(expected),
                }
            )
        outcomes = {bool(row.post_safe) for row in rows}
        heterogeneous += int(len(outcomes) > 1)
        boundary_outcomes.setdefault(case_id, set()).update(outcomes)
    safe_count = sum(int(record.post_safe) for record in records)
    return {
        "schema": OUTCOME_BANK_SUMMARY_SCHEMA,
        "records": len(records),
        "candidate_seed_groups": len(groups),
        "sequence_count": len({record.sequence_id for record in records}),
        "boundary_count": len({record.evaluation_case_id for record in records}),
        "candidate_count": len({record.candidate_id for record in records}),
        "safe_rate": safe_count / float(max(1, len(records))),
        "seed_sensitive_candidate_groups": heterogeneous,
        "outcome_heterogeneous_boundaries": sum(
            int(len(values) > 1) for values in boundary_outcomes.values()
        ),
        "incomplete_groups": incomplete,
        "complete": not incomplete,
        "fingerprint": canonical_fingerprint(
            [record.as_dict() for record in records]
        ),
    }


def write_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_seed_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a repairability Outcome Bank")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    records = read_outcome_bank(args.bank)
    seeds = _parse_seed_list(args.seeds) if args.seeds else None
    summary = summarize_outcome_bank(records, required_seeds=seeds)
    if args.output:
        write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if args.require_complete and not summary["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
