#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise transition-risk policy for event routing and local refinement."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from contracts.boundary import transition_multiscale_risk


@dataclass(frozen=True)
class TransitionRiskPolicy:
    intrinsic_previous_weight: float = 0.10
    intrinsic_following_weight: float = 0.10
    pairwise_weight: float = 0.80
    low_threshold: float = 0.35
    high_threshold: float = 0.70
    residual_inpainting_threshold: float = 0.55

    def validate(self) -> None:
        weights = np.asarray(
            [
                self.intrinsic_previous_weight,
                self.intrinsic_following_weight,
                self.pairwise_weight,
            ],
            dtype=np.float64,
        )
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Transition-risk weights must be non-negative and sum to one")
        if not 0.0 < self.low_threshold < self.high_threshold:
            raise ValueError("Transition thresholds must satisfy low < high")


def normalize_pairwise_score(report: Dict[str, Any]) -> float:
    """Map the physical multiscale score to [0,1] without hiding hard rejects."""
    score = float(report.get("score", 0.0))
    # Smooth saturation is preferable to clipping because the raw score is a
    # weighted physical quantity rather than a calibrated probability.
    return float(1.0 - np.exp(-max(score, 0.0)))


def transition_decision(
    previous_motion: np.ndarray,
    following_motion: np.ndarray,
    *,
    previous_intrinsic_prior: Optional[float] = None,
    following_intrinsic_prior: Optional[float] = None,
    bridge_motion: Optional[np.ndarray] = None,
    aligned_residual_risk: Optional[float] = None,
    fps: float = 30.0,
    policy: TransitionRiskPolicy = TransitionRiskPolicy(),
) -> Dict[str, Any]:
    policy.validate()
    bridge = (
        np.asarray(bridge_motion, dtype=np.float32)
        if bridge_motion is not None
        else np.zeros((0, np.asarray(previous_motion).shape[-1]), dtype=np.float32)
    )
    report = transition_multiscale_risk(
        previous_motion, bridge, following_motion, fps=float(fps)
    )
    pairwise = normalize_pairwise_score(report)
    available_values = []
    available_weights = []
    if previous_intrinsic_prior is not None:
        available_values.append(float(np.clip(previous_intrinsic_prior, 0.0, 1.0)))
        available_weights.append(float(policy.intrinsic_previous_weight))
    if following_intrinsic_prior is not None:
        available_values.append(float(np.clip(following_intrinsic_prior, 0.0, 1.0)))
        available_weights.append(float(policy.intrinsic_following_weight))
    available_values.append(pairwise)
    available_weights.append(float(policy.pairwise_weight))
    weights = np.asarray(available_weights, dtype=np.float64)
    weights /= max(float(weights.sum()), 1.0e-8)
    combined = float(np.dot(weights, np.asarray(available_values, dtype=np.float64)))
    if bool(report.get("hard_reject", False)):
        action = "reroute"
        reason = "physical_hard_reject"
    elif combined < policy.low_threshold:
        action = "direct_join"
        reason = "pairwise_risk_below_low_threshold"
    elif combined < policy.high_threshold:
        action = "geodesic_alignment"
        reason = "moderate_pairwise_risk"
    else:
        residual = (
            combined
            if aligned_residual_risk is None
            else float(np.clip(aligned_residual_risk, 0.0, 1.0))
        )
        if aligned_residual_risk is None:
            action = "geodesic_alignment_then_reassess"
            reason = "high_risk_requires_alignment_residual"
        elif residual >= policy.residual_inpainting_threshold:
            action = "contact_guided_masked_inpainting"
            reason = "post_alignment_residual_remains_high"
        else:
            action = "accept_aligned_transition"
            reason = "alignment_reduced_residual"
    return {
        "schema": "dunhuang_pairwise_transition_repair_policy_v1",
        "action": action,
        "reason": reason,
        "combined_risk": combined,
        "pairwise_risk": pairwise,
        "previous_intrinsic_prior": (
            None if previous_intrinsic_prior is None else float(previous_intrinsic_prior)
        ),
        "following_intrinsic_prior": (
            None if following_intrinsic_prior is None else float(following_intrinsic_prior)
        ),
        "aligned_residual_risk": (
            None if aligned_residual_risk is None else float(aligned_residual_risk)
        ),
        "hard_reject": bool(report.get("hard_reject", False)),
        "policy": asdict(policy),
        "physical_report": report,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--following", required=True)
    parser.add_argument("--bridge", default="")
    parser.add_argument("--previous_intrinsic", type=float, default=0.0)
    parser.add_argument("--following_intrinsic", type=float, default=0.0)
    parser.add_argument("--aligned_residual", type=float, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = transition_decision(
        np.load(args.previous, allow_pickle=True),
        np.load(args.following, allow_pickle=True),
        previous_intrinsic_prior=float(args.previous_intrinsic),
        following_intrinsic_prior=float(args.following_intrinsic),
        bridge_motion=(
            np.load(args.bridge, allow_pickle=True) if args.bridge else None
        ),
        aligned_residual_risk=args.aligned_residual,
        fps=float(args.fps),
    )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
