#!/usr/bin/env python3
"""Resolve one formally published Motion Refiner before generation starts.

The generate-only route must never spend time on scheduler/IK stages and only
then discover that its neural repair asset is absent.  This resolver accepts
an explicit checkpoint or searches the selected trained run for an
unambiguous formal asset.  Every candidate is checked against the current
motion/config contract and its training publication decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

# A path-invoked script receives ``scripts/`` rather than the repository root
# on sys.path.  Keep the standalone preflight identical to ``python -m`` and
# to generate_only.sh, which already exports PYTHONPATH explicitly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from support.event_identity import normalize_event_db_contract
from training.motion_models import (
    MotionGenerationConfig,
    REFINER_MODEL_VERSION,
    _trusted_torch_load,
    assert_motion_checkpoint_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _candidate_paths(run_root: Path, explicit: Optional[Path]) -> List[Path]:
    if explicit is not None:
        return [explicit.resolve()]
    canonical = run_root / "motion_refiner_train_only_refiner.pt"
    found = set()
    if canonical.is_file():
        found.add(canonical.resolve())
    for name in (
        "motion_refiner_train_only_refiner.pt",
        "boundary_refiner.pt",
    ):
        for path in run_root.rglob(name):
            text = path.name.lower()
            if "rejected_validation" in text or "training_snapshot" in text:
                continue
            found.add(path.resolve())
    return sorted(found, key=lambda value: str(value))


def _validate_candidate(
    path: Path,
    cfg: MotionGenerationConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.is_file() or path.stat().st_size <= 0:
        return None, "file_missing_or_empty"
    try:
        payload = _trusted_torch_load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise RuntimeError("payload is not a mapping")
        if str(payload.get("version", "")) != REFINER_MODEL_VERSION:
            raise RuntimeError(
                "model version mismatch: "
                f"checkpoint={payload.get('version')!r}, "
                f"runtime={REFINER_MODEL_VERSION!r}"
            )
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or not state:
            raise RuntimeError("formal state_dict is missing or empty")
        assert_motion_checkpoint_contract(
            dict(payload), cfg, path, "boundary_refiner"
        )
        event_contract = normalize_event_db_contract(
            payload.get("training_event_db_contract")
        )
        if (
            event_contract is None
            or event_contract.get("schema") != "dunhuang_event_db_contract_v2"
            or int(event_contract.get("num_events", -1)) < 1
            or not event_contract.get("ordered_event_uid_sha256")
        ):
            raise RuntimeError("training Event-DB provenance is missing")
        validation = payload.get("validation")
        decision = (
            validation.get("checkpoint_decision")
            if isinstance(validation, Mapping)
            else None
        )
        if not isinstance(decision, Mapping) or decision.get(
            "publish_allowed"
        ) is not True:
            raise RuntimeError(
                "checkpoint was not published by the validation gate"
            )
        if payload.get("formal_checkpoint") is False:
            raise RuntimeError("checkpoint is explicitly marked non-formal")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(path.stat().st_size),
        "model_version": str(payload.get("version")),
        "motion_contract": dict(payload["motion_contract"]),
        "training_event_db_contract": event_contract,
        "checkpoint_decision": dict(decision),
    }, None


def resolve(
    *,
    run_root: Path,
    config: Path,
    explicit: Optional[Path],
    report: Path,
    repository_root: Path,
) -> Dict[str, Any]:
    cfg = MotionGenerationConfig.from_json(config).apply_env()
    candidates = _candidate_paths(run_root, explicit)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for path in candidates:
        row, reason = _validate_candidate(path, cfg)
        if row is None:
            rejected.append({"path": str(path), "reason": str(reason)})
        else:
            accepted.append(row)
    if len(accepted) != 1:
        report.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "generate_only_refiner_binding_v1",
            "ok": False,
            "implementation_commit": _implementation_commit(repository_root),
            "trained_run": str(run_root),
            "config": str(config),
            "explicit_checkpoint": (
                None if explicit is None else str(explicit.resolve())
            ),
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "reason": (
                "no_formal_refiner_checkpoint"
                if not accepted
                else "ambiguous_formal_refiner_checkpoints"
            ),
        }
        report.write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"expected exactly one validated Refiner checkpoint, found "
            f"{len(accepted)}; see {report}"
        )
    binding = {
        "schema": "generate_only_refiner_binding_v1",
        "ok": True,
        "implementation_commit": _implementation_commit(repository_root),
        "trained_run": str(run_root),
        "config": str(config),
        "selection_policy": (
            "explicit_checkpoint" if explicit is not None else
            "canonical_or_unique_validated_checkpoint"
        ),
        "selected": accepted[0],
        "rejected_candidates": rejected,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(binding, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return binding


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--explicit", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        binding = resolve(
            run_root=args.run_root.resolve(),
            config=args.config.resolve(),
            explicit=(None if args.explicit is None else args.explicit),
            report=args.report.resolve(),
            repository_root=REPOSITORY_ROOT,
        )
    except Exception as exc:
        print(f"[FATAL] Refiner checkpoint preflight failed: {exc}", file=sys.stderr)
        return 2
    print(binding["selected"]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
