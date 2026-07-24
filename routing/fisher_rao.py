#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically stable probability-simplex operations for Event routing.

The routing decision at one music slot is a categorical distribution over the
slot's feasible Event candidates.  Its intrinsic geometry is the Fisher--Rao
geometry of the open probability simplex, not a Euclidean geometry over raw
logits.  This module intentionally depends only on NumPy so the mathematical
contract can be tested without loading the motion-generation runtime.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


EPS = 1.0e-12


def _as_finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def normalize_probability(
    probability: np.ndarray,
    *,
    axis: int = -1,
    minimum: float = 0.0,
) -> np.ndarray:
    """Validate and normalize a non-negative probability tensor.

    ``minimum`` is applied before the final normalization.  A positive minimum
    keeps values in the open simplex, which is required by logarithmic
    Schrödinger updates.  A zero minimum preserves exact structural zeros.
    """

    array = _as_finite_array(probability, name="probability")
    if np.any(array < 0.0):
        raise ValueError("probability must be non-negative")
    floor = float(minimum)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("minimum probability must be finite and non-negative")
    if floor > 0.0:
        array = np.maximum(array, floor)
    total = np.sum(array, axis=axis, keepdims=True)
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise ValueError("probability has a zero or invalid normalization mass")
    return (array / total).astype(np.float64)


def simplex_softmax(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    mask: Optional[np.ndarray] = None,
    minimum: float = EPS,
) -> np.ndarray:
    """Map finite logits into a categorical probability simplex.

    Masked categories remain exactly zero.  At least one category must be
    feasible in every normalized row.
    """

    values = _as_finite_array(logits, name="logits")
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("temperature must be finite and positive")
    feasible = (
        np.ones(values.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if feasible.shape != values.shape:
        raise ValueError(
            f"mask shape {feasible.shape} does not match logits {values.shape}"
        )
    if np.any(np.sum(feasible, axis=-1) == 0):
        raise ValueError("every simplex row needs at least one feasible category")
    scaled = values / tau
    masked = np.where(feasible, scaled, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    exponent = np.where(feasible, np.exp(masked - maximum), 0.0)
    if minimum > 0.0:
        exponent = np.where(feasible, np.maximum(exponent, float(minimum)), 0.0)
    return normalize_probability(exponent, axis=-1, minimum=0.0)


def fisher_rao_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Fisher--Rao geodesic distance on a categorical simplex.

    Under the square-root embedding ``p -> 2 sqrt(p)``, the open simplex is a
    positive orthant of a sphere with radius two.  Consequently the distance is
    ``2 arccos(sum_i sqrt(p_i q_i))``.
    """

    p = normalize_probability(left, axis=axis, minimum=0.0)
    q = normalize_probability(right, axis=axis, minimum=0.0)
    if p.shape != q.shape:
        raise ValueError(f"simplex shapes differ: {p.shape} versus {q.shape}")
    affinity = np.sum(np.sqrt(p * q), axis=axis)
    return 2.0 * np.arccos(np.clip(affinity, -1.0, 1.0))


def fisher_rao_midpoint(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return the equal-time Fisher--Rao geodesic midpoint."""

    p = normalize_probability(left, minimum=0.0)
    q = normalize_probability(right, minimum=0.0)
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("Fisher--Rao midpoint expects equally shaped vectors")
    root_p = np.sqrt(p)
    root_q = np.sqrt(q)
    affinity = float(np.clip(np.dot(root_p, root_q), -1.0, 1.0))
    angle = float(np.arccos(affinity))
    if angle <= 1.0e-10:
        return p
    sine = float(np.sin(angle))
    midpoint_root = (
        np.sin(0.5 * angle) / sine * root_p
        + np.sin(0.5 * angle) / sine * root_q
    )
    return normalize_probability(np.square(midpoint_root), minimum=0.0)


def categorical_entropy(probability: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Shannon entropy with exact zero probabilities handled safely."""

    p = normalize_probability(probability, axis=axis, minimum=0.0)
    terms = np.zeros_like(p)
    positive = p > 0.0
    terms[positive] = p[positive] * np.log(p[positive])
    return -np.sum(terms, axis=axis)


def categorical_kl(
    probability: np.ndarray,
    reference: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    """KL(probability || reference), failing on unsupported positive mass."""

    p = normalize_probability(probability, axis=axis, minimum=0.0)
    q = normalize_probability(reference, axis=axis, minimum=0.0)
    if p.shape != q.shape:
        raise ValueError(f"simplex shapes differ: {p.shape} versus {q.shape}")
    unsupported = (p > 0.0) & (q <= 0.0)
    if np.any(unsupported):
        raise ValueError("reference has zero mass where probability is positive")
    terms = np.zeros_like(p)
    positive = p > 0.0
    terms[positive] = p[positive] * (
        np.log(p[positive]) - np.log(q[positive])
    )
    return np.sum(terms, axis=axis)


__all__ = [
    "EPS",
    "categorical_entropy",
    "categorical_kl",
    "fisher_rao_distance",
    "fisher_rao_midpoint",
    "normalize_probability",
    "simplex_softmax",
]
