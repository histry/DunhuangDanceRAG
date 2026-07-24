#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discrete multi-marginal Schrödinger routing on a time-expanded graph.

The reference process is a finite, time-inhomogeneous Markov chain whose
transition support is the hard-feasible Event graph.  Iterative proportional
fitting (IPF) finds the KL-nearest path measure matching the requested
slot-wise categorical marginals.  A Viterbi pass decodes the MAP Event path
from the same fitted path measure.

This is a discrete path-space solver.  It must not be confused with a
continuous SDE bridge or with merely adding an entropy bonus to beam search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from routing.fisher_rao import (
    EPS,
    categorical_entropy,
    fisher_rao_distance,
    normalize_probability,
)


NEG_INF = -np.inf


def _logsumexp(
    values: np.ndarray,
    *,
    axis: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    shifted = np.full(array.shape, NEG_INF, dtype=np.float64)
    valid = np.broadcast_to(finite_maximum, array.shape)
    np.subtract(array, maximum, out=shifted, where=valid)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(
        finite_maximum,
        maximum + np.log(np.maximum(total, EPS)),
        NEG_INF,
    )
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _validate_targets(
    target_marginals: Sequence[np.ndarray],
) -> list[np.ndarray]:
    if not target_marginals:
        raise ValueError("at least one target marginal is required")
    targets: list[np.ndarray] = []
    for index, target in enumerate(target_marginals):
        value = np.asarray(target, dtype=np.float64)
        if value.ndim != 1 or value.size == 0:
            raise ValueError(
                f"target marginal {index} must be a non-empty vector"
            )
        targets.append(normalize_probability(value, minimum=EPS))
    return targets


def reference_markov_kernel(
    transition_cost: np.ndarray,
    *,
    feasible: Optional[np.ndarray] = None,
    epsilon: float = 0.35,
) -> np.ndarray:
    """Create a row-stochastic reference transition kernel.

    Hard-infeasible edges remain exactly zero.  A row without a feasible
    outgoing edge is rejected because it cannot define a Markov process.
    """

    cost = np.asarray(transition_cost, dtype=np.float64)
    if cost.ndim != 2 or cost.size == 0:
        raise ValueError("transition cost must be a non-empty matrix")
    if not np.isfinite(cost).all():
        raise ValueError("transition cost contains NaN or Inf")
    temperature = float(epsilon)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Schrödinger epsilon must be finite and positive")
    support = (
        np.ones(cost.shape, dtype=bool)
        if feasible is None
        else np.asarray(feasible, dtype=bool)
    )
    if support.shape != cost.shape:
        raise ValueError(
            f"feasible shape {support.shape} does not match cost {cost.shape}"
        )
    if np.any(np.sum(support, axis=1) == 0):
        bad = np.where(np.sum(support, axis=1) == 0)[0].tolist()
        raise ValueError(f"reference graph has dead outgoing rows: {bad}")
    logits = np.where(support, -cost / temperature, NEG_INF)
    normalizer = _logsumexp(logits, axis=1)[:, None]
    kernel = np.where(support, np.exp(logits - normalizer), 0.0)
    return normalize_probability(kernel, axis=1, minimum=0.0)


@dataclass(frozen=True)
class ChainMarginals:
    log_partition: float
    node: tuple[np.ndarray, ...]
    edge: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SchrodingerResult:
    schema: str
    target_marginals: tuple[np.ndarray, ...]
    node_marginals: tuple[np.ndarray, ...]
    edge_marginals: tuple[np.ndarray, ...]
    posterior_transitions: tuple[np.ndarray, ...]
    reference_transitions: tuple[np.ndarray, ...]
    node_log_potentials: tuple[np.ndarray, ...]
    map_path: tuple[int, ...]
    map_log_probability: float
    path_entropy: float
    iterations: int
    maximum_l1_residual: float
    maximum_fisher_rao_residual: float
    converged: bool


def chain_marginals(
    log_initial: np.ndarray,
    log_transitions: Sequence[np.ndarray],
    node_log_potentials: Sequence[np.ndarray],
) -> ChainMarginals:
    """Exact forward--backward marginals for one finite chain."""

    node_potential = [
        np.asarray(value, dtype=np.float64) for value in node_log_potentials
    ]
    if not node_potential:
        raise ValueError("node potentials must not be empty")
    initial = np.asarray(log_initial, dtype=np.float64)
    if initial.shape != node_potential[0].shape:
        raise ValueError("initial distribution and first node have different shapes")
    transitions = [np.asarray(value, dtype=np.float64) for value in log_transitions]
    if len(transitions) != len(node_potential) - 1:
        raise ValueError("a T-node chain requires T-1 transition matrices")
    for time, transition in enumerate(transitions):
        expected = (node_potential[time].size, node_potential[time + 1].size)
        if transition.shape != expected:
            raise ValueError(
                f"transition {time} shape {transition.shape}, expected {expected}"
            )

    forward = [initial + node_potential[0]]
    for time, transition in enumerate(transitions):
        message = _logsumexp(
            forward[-1][:, None] + transition,
            axis=0,
        )
        forward.append(message + node_potential[time + 1])
    log_partition = float(_logsumexp(forward[-1], axis=0))
    if not np.isfinite(log_partition):
        raise RuntimeError("reference graph has no finite complete path")

    backward = [np.zeros_like(node_potential[-1])]
    for time in range(len(transitions) - 1, -1, -1):
        message = _logsumexp(
            transitions[time]
            + node_potential[time + 1][None]
            + backward[0][None],
            axis=1,
        )
        backward.insert(0, message)

    node = tuple(
        normalize_probability(
            np.exp(forward[time] + backward[time] - log_partition),
            minimum=0.0,
        )
        for time in range(len(node_potential))
    )
    edge = tuple(
        normalize_probability(
            np.exp(
                forward[time][:, None]
                + transition
                + node_potential[time + 1][None]
                + backward[time + 1][None]
                - log_partition
            ).reshape(-1),
            minimum=0.0,
        ).reshape(transition.shape)
        for time, transition in enumerate(transitions)
    )
    return ChainMarginals(log_partition, node, edge)


def viterbi_path(
    log_initial: np.ndarray,
    log_transitions: Sequence[np.ndarray],
    node_log_potentials: Sequence[np.ndarray],
) -> tuple[tuple[int, ...], float]:
    """Exact MAP path for the fitted finite-chain measure."""

    potentials = [
        np.asarray(value, dtype=np.float64) for value in node_log_potentials
    ]
    score = np.asarray(log_initial, dtype=np.float64) + potentials[0]
    backpointers: list[np.ndarray] = []
    for time, transition in enumerate(log_transitions):
        candidate = score[:, None] + np.asarray(transition, dtype=np.float64)
        pointer = np.argmax(candidate, axis=0).astype(np.int64)
        score = candidate[pointer, np.arange(candidate.shape[1])]
        score = score + potentials[time + 1]
        backpointers.append(pointer)
    last = int(np.argmax(score))
    best_score = float(score[last])
    path = [last]
    for pointer in reversed(backpointers):
        path.append(int(pointer[path[-1]]))
    path.reverse()
    return tuple(path), best_score


def _posterior_transitions(
    marginals: ChainMarginals,
) -> tuple[np.ndarray, ...]:
    transitions = []
    for time, edge in enumerate(marginals.edge):
        denominator = marginals.node[time][:, None]
        conditional = np.divide(
            edge,
            denominator,
            out=np.zeros_like(edge),
            where=denominator > EPS,
        )
        row_mass = conditional.sum(axis=1)
        if np.any((marginals.node[time] > EPS) & (row_mass <= 0.0)):
            raise RuntimeError("posterior transition lost reachable row mass")
        active = row_mass > 0.0
        conditional[active] = normalize_probability(
            conditional[active], axis=1, minimum=0.0
        )
        transitions.append(conditional)
    return tuple(transitions)


def multi_marginal_schrodinger(
    target_marginals: Sequence[np.ndarray],
    transition_costs: Sequence[np.ndarray],
    *,
    feasible_masks: Optional[Sequence[np.ndarray]] = None,
    initial_reference: Optional[np.ndarray] = None,
    epsilon: float = 0.35,
    maximum_iterations: int = 300,
    tolerance: float = 1.0e-7,
    damping: float = 1.0,
) -> SchrodingerResult:
    """Fit the KL-nearest feasible path measure to all slot marginals.

    The cyclic IPF update is an information projection on one categorical
    marginal at a time.  A non-converged result is returned explicitly; callers
    decide whether to fall back to a deterministic legacy solver.
    """

    targets = _validate_targets(target_marginals)
    if len(transition_costs) != len(targets) - 1:
        raise ValueError("a T-slot route requires T-1 transition cost matrices")
    masks = (
        [None] * len(transition_costs)
        if feasible_masks is None
        else list(feasible_masks)
    )
    if len(masks) != len(transition_costs):
        raise ValueError("feasible mask count does not match transition count")

    reference = []
    for time, (cost, mask) in enumerate(zip(transition_costs, masks)):
        matrix = np.asarray(cost, dtype=np.float64)
        expected = (targets[time].size, targets[time + 1].size)
        if matrix.shape != expected:
            raise ValueError(
                f"transition cost {time} shape {matrix.shape}, expected {expected}"
            )
        reference.append(
            reference_markov_kernel(matrix, feasible=mask, epsilon=epsilon)
        )
    log_transitions = tuple(
        np.where(kernel > 0.0, np.log(np.maximum(kernel, EPS)), NEG_INF)
        for kernel in reference
    )

    if initial_reference is None:
        initial = np.full(targets[0].shape, 1.0 / targets[0].size)
    else:
        initial = normalize_probability(initial_reference, minimum=EPS)
        if initial.shape != targets[0].shape:
            raise ValueError("initial reference has the wrong candidate count")
    log_initial = np.log(np.maximum(initial, EPS))

    iterations = max(1, int(maximum_iterations))
    tol = float(tolerance)
    rate = float(damping)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("IPF tolerance must be finite and positive")
    if not np.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("IPF damping must be in (0,1]")
    node_log_potentials = [np.zeros_like(target) for target in targets]
    maximum_l1 = np.inf
    maximum_fr = np.inf
    converged = False
    fitted = chain_marginals(
        log_initial, log_transitions, node_log_potentials
    )

    for iteration in range(1, iterations + 1):
        for time, target in enumerate(targets):
            current = fitted.node[time]
            if np.any((target > EPS) & (current <= EPS)):
                raise RuntimeError(
                    "target marginal requests mass outside the reference path support "
                    f"at slot {time}"
                )
            update = np.log(np.maximum(target, EPS)) - np.log(
                np.maximum(current, EPS)
            )
            node_log_potentials[time] += rate * update
            # Potentials are defined up to a per-slot additive constant.
            node_log_potentials[time] -= np.mean(node_log_potentials[time])
            fitted = chain_marginals(
                log_initial, log_transitions, node_log_potentials
            )

        l1_residuals = [
            float(np.sum(np.abs(current - target)))
            for current, target in zip(fitted.node, targets)
        ]
        fr_residuals = [
            float(fisher_rao_distance(current, target))
            for current, target in zip(fitted.node, targets)
        ]
        maximum_l1 = max(l1_residuals)
        maximum_fr = max(fr_residuals)
        if maximum_l1 <= tol:
            converged = True
            break

    map_path, unnormalized_map_score = viterbi_path(
        log_initial,
        log_transitions,
        node_log_potentials,
    )
    map_log_probability = float(
        unnormalized_map_score - fitted.log_partition
    )
    expected_score = float(
        np.dot(fitted.node[0], log_initial + node_log_potentials[0])
    )
    for time, edge in enumerate(fitted.edge):
        finite = np.isfinite(log_transitions[time])
        expected_score += float(
            np.sum(
                edge[finite]
                * (
                    log_transitions[time][finite]
                    + np.broadcast_to(
                        node_log_potentials[time + 1][None],
                        edge.shape,
                    )[finite]
                )
            )
        )
    path_entropy = float(max(0.0, fitted.log_partition - expected_score))

    return SchrodingerResult(
        schema="v46_55_fisher_rao_multi_marginal_graph_sb_v1",
        target_marginals=tuple(targets),
        node_marginals=fitted.node,
        edge_marginals=fitted.edge,
        posterior_transitions=_posterior_transitions(fitted),
        reference_transitions=tuple(reference),
        node_log_potentials=tuple(node_log_potentials),
        map_path=map_path,
        map_log_probability=map_log_probability,
        path_entropy=path_entropy,
        iterations=iteration,
        maximum_l1_residual=float(maximum_l1),
        maximum_fisher_rao_residual=float(maximum_fr),
        converged=bool(converged),
    )


def independent_path_entropy(target_marginals: Sequence[np.ndarray]) -> float:
    """Entropy of an independent per-slot baseline for reporting."""

    return float(
        sum(float(categorical_entropy(target)) for target in target_marginals)
    )


__all__ = [
    "ChainMarginals",
    "SchrodingerResult",
    "chain_marginals",
    "independent_path_entropy",
    "multi_marginal_schrodinger",
    "reference_markov_kernel",
    "viterbi_path",
]
