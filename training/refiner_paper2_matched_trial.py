"""Low-cost matched-candidate evidence utilities for paper 2.

This module is deliberately diagnostic-only.  It never changes candidate
selection, witness generation, projection or authoritative Guard acceptance.
All persisted records bind the implementation, protocol and input contracts;
JSONL append semantics let a server run retain completed candidate audits.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

from training import motion_models as m


SCHEMA = "paper2_matched_candidate_curvature_audit_v1"


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value) -> str:
    """Hash one detached CPU tensor including dtype and shape."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def same_ray_identity_geometry(current, direction, mask, target_rms):
    """Rescale one actual owned-space ray without changing its shape.

    Returns ``rho*sqrt(n)*(u, v)`` represented as an exact-radius anchor and a
    Euclidean unit sphere-tangent direction.  It is intentionally unavailable
    for the G metric: its equivalent-radius calibration is a different
    experiment and must not be mixed into the identity same-ray sweep.
    """
    active = mask.bool()
    count = int(active.sum().detach())
    if count <= 0:
        raise ValueError("same-ray audit requires a nonempty ownership mask")
    radial = current.detach().to(m.torch.float64).masked_fill(~active, 0.0)
    radial_norm = m.torch.linalg.vector_norm(radial[active])
    if not bool(m.torch.isfinite(radial_norm)) or float(radial_norm) <= 0.0:
        raise ValueError("same-ray audit requires a finite nonzero anchor")
    u = radial / radial_norm
    physical = direction.detach().to(m.torch.float64).masked_fill(~active, 0.0)
    physical = physical - m.torch.sum(physical[active] * u[active]) * u
    direction_norm = m.torch.linalg.vector_norm(physical[active])
    if not bool(m.torch.isfinite(direction_norm)) or float(direction_norm) <= 0.0:
        raise ValueError("same-ray audit requires a finite nonzero tangent")
    v = (physical / direction_norm).masked_fill(~active, 0.0)
    radius = float(target_rms) * math.sqrt(float(count))
    anchor = (radius * u).masked_fill(~active, 0.0)
    return anchor, v


def matched_row_audit(
    *,
    first_order_change: Mapping[str, float],
    second_order_change: Mapping[str, float],
    actual_frozen_change: Mapping[str, float],
    row_scale: Mapping[str, float],
    curvature_change=None,
    epsilon=1.0e-15,
):
    """Build strict same-row prediction-to-trial errors.

    The three mappings must describe the same anchor, physical direction,
    angle and frozen scalar row.  Authoritative hard-Guard changes belong in a
    sibling field because a support transition is not the same scalar row.
    """
    names = tuple(actual_frozen_change)
    if set(first_order_change) != set(names):
        raise ValueError("first-order audit rows do not match actual rows")
    if set(second_order_change) != set(names):
        raise ValueError("second-order audit rows do not match actual rows")
    if set(row_scale) != set(names):
        raise ValueError("matched audit requires an explicit scale for every row")
    rows = {}
    for name in names:
        scale = float(row_scale[name])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"invalid frozen contract scale for {name}")
        first = float(first_order_change[name])
        second = float(second_order_change[name])
        actual = float(actual_frozen_change[name])
        if not all(math.isfinite(value) for value in (first, second, actual)):
            raise ValueError(f"nonfinite matched audit row: {name}")
        e1 = abs(first - actual)
        e2 = abs(second - actual)
        rows[name] = {
            "first_order_predicted_change": first,
            "second_order_predicted_change": second,
            "curvature_contribution": (
                float(curvature_change[name])
                if curvature_change is not None else second - first
            ),
            "frozen_witness_trial_change": actual,
            "contract_scale": scale,
            "first_order_absolute_error": e1,
            "second_order_absolute_error": e2,
            "first_order_normalized_error": e1 / scale,
            "second_order_normalized_error": e2 / scale,
            "normalized_error_improvement": (e1 - e2) / scale,
            "log_second_over_first_error": math.log(
                (e2 + float(epsilon)) / (e1 + float(epsilon))
            ),
            "second_order_more_accurate": bool(e2 < e1),
        }
    return rows


class JsonlMechanismRecorder:
    """Deterministic, resumable and quota-limited mechanism record writer."""

    def __init__(
        self,
        *,
        path,
        binding,
        case_uids,
        max_candidates_per_case_budget,
    ):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.binding = dict(binding)
        self.binding_sha256 = canonical_json_sha256(self.binding)
        self.case_uids = frozenset(str(value) for value in case_uids)
        self.limit = int(max_candidates_per_case_budget)
        if not self.case_uids:
            raise ValueError("mechanism audit requires a preregistered case list")
        if not 1 <= self.limit <= 3:
            raise ValueError("candidate quota must be in [1, 3]")
        self.completed = set()
        self.counts = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("binding_sha256") != self.binding_sha256:
                    raise RuntimeError(
                        "existing mechanism audit has a different binding"
                    )
                candidate_id = str(row["candidate_id"])
                self.completed.add(candidate_id)
                key = (
                    str(row["case_uid"]),
                    int(row["budget"]),
                    str(row["candidate_source"]),
                )
                self.counts[key] = self.counts.get(key, 0) + 1

    def accepts_case(self, case_uid):
        return str(case_uid) in self.case_uids

    def candidate_id(
        self,
        *,
        case_uid,
        budget,
        iteration,
        theta_radians,
        backtrack,
        candidate_source,
        constraint_generation_depth,
        direction_sha256,
        support_sha256,
        witness_sha256,
    ):
        return canonical_json_sha256({
            "binding_sha256": self.binding_sha256,
            "case_uid": str(case_uid),
            "budget": int(budget),
            "iteration": int(iteration),
            "theta_radians": float(theta_radians),
            "backtrack": int(backtrack),
            "source": str(candidate_source),
            "constraint_generation_depth": int(constraint_generation_depth),
            "direction_sha256": str(direction_sha256),
            "support_sha256": str(support_sha256),
            "witness_sha256": str(witness_sha256),
        })

    def should_record(
        self, *, candidate_id, case_uid, budget, candidate_source
    ):
        if not self.accepts_case(case_uid) or candidate_id in self.completed:
            return False
        key = (str(case_uid), int(budget), str(candidate_source))
        return self.counts.get(key, 0) < self.limit

    def append(self, record):
        row = {
            "schema": SCHEMA,
            "binding_sha256": self.binding_sha256,
            **record,
        }
        candidate_id = str(row["candidate_id"])
        if candidate_id in self.completed:
            return False
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self.completed.add(candidate_id)
        key = (
            str(row["case_uid"]),
            int(row["budget"]),
            str(row["candidate_source"]),
        )
        self.counts[key] = self.counts.get(key, 0) + 1
        return True

    def audit(self):
        return {
            "schema": SCHEMA,
            "path": str(self.path),
            "binding_sha256": self.binding_sha256,
            "case_uids": sorted(self.case_uids),
            "max_candidates_per_case_budget_source": self.limit,
            "completed_candidate_count": len(self.completed),
            "resumable_jsonl": True,
        }


def merge_numeric_counters(rows):
    result = {}
    for row in rows:
        for name, value in (row or {}).items():
            result[name] = result.get(name, 0) + value
    return result
