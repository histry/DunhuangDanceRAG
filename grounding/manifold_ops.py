#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically stable operators for the mixed-curvature grounder.

This module is deliberately independent from ``motion_geometry.product_manifold``.
The latter represents the physical EDGE-151 motion state used by V45/V46/V53;
this file represents the latent retrieval space used by the paper-one grounder.

The latent product contains four factors:

* Lorentz hyperbolic points for explicit event hierarchies;
* unit-sphere points for normalized CLAP-style semantics;
* Gaussian distributions whose covariance uses Bures--Wasserstein geometry;
* Euclidean controls such as duration, energy, and confidence.

All public distances return *squared* distances.  Keeping the squared form
avoids an unnecessary square root in contrastive logits and makes factor
weights directly interpretable.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional for geometry-only tooling
    torch = None
    F = None


EPS = 1.0e-7
FACTOR_NAMES = ("lorentz", "sphere", "gaussian_bw", "euclidean")


def _positive_curvature_np(curvature: Any) -> np.ndarray:
    value = np.asarray(curvature, dtype=np.float64)
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError("Lorentz curvature magnitude must be finite and positive")
    return value


def lorentz_project_np(spatial: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """Lift spatial coordinates to the upper Lorentz hyperboloid.

    The Minkowski signature is ``(-,+,...,+)`` and the returned points satisfy
    ``<x,x>_L = -1 / curvature`` with a positive time coordinate.
    """

    x = np.asarray(spatial, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1 or not np.isfinite(x).all():
        raise ValueError("spatial Lorentz coordinates must be finite and non-empty")
    c = _positive_curvature_np(curvature)
    time = np.sqrt(1.0 / c + np.sum(x * x, axis=-1))
    return np.concatenate([time[..., None], x], axis=-1).astype(np.float32)


def lorentz_inner_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape[-1] != y.shape[-1] or x.shape[-1] < 2:
        raise ValueError("Lorentz points must share a final dimension >= 2")
    return -x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)


def lorentz_distance_sq_np(
    left: np.ndarray,
    right: np.ndarray,
    curvature: float = 1.0,
) -> np.ndarray:
    c = _positive_curvature_np(curvature)
    argument = np.maximum(-c * lorentz_inner_np(left, right), 1.0)
    distance = np.arccosh(argument) / np.sqrt(c)
    return np.maximum(distance * distance, 0.0).astype(np.float32)


def sphere_project_np(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("sphere coordinates must be finite with dimension >= 2")
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norm <= EPS):
        raise ValueError("zero vectors cannot be projected to the unit sphere")
    return (x / norm).astype(np.float32)


def sphere_distance_sq_np(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = sphere_project_np(left).astype(np.float64)
    y = sphere_project_np(right).astype(np.float64)
    dot = np.clip(np.sum(x * y, axis=-1), -1.0, 1.0)
    angle = np.arccos(dot)
    return np.maximum(angle * angle, 0.0).astype(np.float32)


def project_spd_np(
    matrix: np.ndarray,
    minimum_eigenvalue: float = 1.0e-5,
) -> np.ndarray:
    """Symmetrize and eigenvalue-floor one or more covariance matrices."""

    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim < 2
        or value.shape[-1] != value.shape[-2]
        or not np.isfinite(value).all()
    ):
        raise ValueError("SPD input must contain finite square matrices")
    floor = float(minimum_eigenvalue)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_eigenvalue must be finite and positive")
    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, floor)
    projected = (eigenvectors * eigenvalues[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return (0.5 * (projected + np.swapaxes(projected, -1, -2))).astype(
        np.float32
    )


def matrix_sqrt_spd_np(
    matrix: np.ndarray,
    minimum_eigenvalue: float = 1.0e-7,
) -> np.ndarray:
    value = project_spd_np(matrix, minimum_eigenvalue).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[..., None, :]) @ np.swapaxes(
        eigenvectors, -1, -2
    )
    return (0.5 * (root + np.swapaxes(root, -1, -2))).astype(np.float32)


def bures_distance_sq_np(
    left: np.ndarray,
    right: np.ndarray,
    minimum_eigenvalue: float = 1.0e-5,
) -> np.ndarray:
    """Squared Bures distance between broadcast-compatible SPD matrices."""

    a = project_spd_np(left, minimum_eigenvalue).astype(np.float64)
    b = project_spd_np(right, minimum_eigenvalue).astype(np.float64)
    if a.shape[-2:] != b.shape[-2:]:
        raise ValueError("Bures covariance matrices must share their matrix shape")
    a_root = matrix_sqrt_spd_np(a, minimum_eigenvalue).astype(np.float64)
    middle = a_root @ b @ a_root
    middle_root = matrix_sqrt_spd_np(middle, minimum_eigenvalue).astype(np.float64)
    trace = np.trace(a + b - 2.0 * middle_root, axis1=-2, axis2=-1)
    # Round-off can produce tiny negative values for identical matrices.
    return np.maximum(trace, 0.0).astype(np.float32)


def gaussian_wasserstein_distance_sq_np(
    left_mean: np.ndarray,
    left_covariance: np.ndarray,
    right_mean: np.ndarray,
    right_covariance: np.ndarray,
    minimum_eigenvalue: float = 1.0e-5,
) -> np.ndarray:
    mean_left = np.asarray(left_mean, dtype=np.float64)
    mean_right = np.asarray(right_mean, dtype=np.float64)
    if mean_left.shape[-1] != mean_right.shape[-1]:
        raise ValueError("Gaussian means must share their final dimension")
    mean_term = np.sum((mean_left - mean_right) ** 2, axis=-1)
    covariance_term = bures_distance_sq_np(
        left_covariance,
        right_covariance,
        minimum_eigenvalue=minimum_eigenvalue,
    )
    return np.maximum(mean_term + covariance_term, 0.0).astype(np.float32)


def normalized_factor_weights_np(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.shape != (len(FACTOR_NAMES),):
        raise ValueError(
            f"factor weights must contain {len(FACTOR_NAMES)} values in "
            f"{FACTOR_NAMES} order"
        )
    if not np.isfinite(value).all() or np.any(value <= 0.0):
        raise ValueError("product metric weights must be finite and strictly positive")
    return (value / value.sum()).astype(np.float32)


def mixed_product_distance_sq_np(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    weights: Sequence[float] | np.ndarray,
    *,
    curvature: float = 1.0,
    minimum_eigenvalue: float = 1.0e-5,
) -> np.ndarray:
    """Squared distance on the fixed-weight latent product manifold."""

    missing = [
        name
        for name in (
            "lorentz",
            "sphere",
            "gaussian_mean",
            "gaussian_covariance",
            "euclidean",
        )
        if name not in left or name not in right
    ]
    if missing:
        raise KeyError(f"mixed-product factor mappings miss: {missing}")
    factor_weights = normalized_factor_weights_np(weights)
    lorentz = lorentz_distance_sq_np(
        left["lorentz"], right["lorentz"], curvature=curvature
    )
    sphere = sphere_distance_sq_np(left["sphere"], right["sphere"])
    gaussian_per_part = gaussian_wasserstein_distance_sq_np(
        left["gaussian_mean"],
        left["gaussian_covariance"],
        right["gaussian_mean"],
        right["gaussian_covariance"],
        minimum_eigenvalue=minimum_eigenvalue,
    )
    gaussian = np.mean(gaussian_per_part, axis=-1)
    euclidean = np.sum(
        (
            np.asarray(left["euclidean"], dtype=np.float64)
            - np.asarray(right["euclidean"], dtype=np.float64)
        )
        ** 2,
        axis=-1,
    )
    return np.maximum(
        factor_weights[0] * lorentz
        + factor_weights[1] * sphere
        + factor_weights[2] * gaussian
        + factor_weights[3] * euclidean,
        0.0,
    ).astype(np.float32)


def _torch_required() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for differentiable manifold operators")


def lorentz_project_torch(
    spatial: "torch.Tensor", curvature: "torch.Tensor | float"
) -> "torch.Tensor":
    _torch_required()
    c = torch.as_tensor(curvature, dtype=spatial.dtype, device=spatial.device).clamp_min(EPS)
    time = torch.sqrt(1.0 / c + (spatial * spatial).sum(dim=-1))
    return torch.cat([time[..., None], spatial], dim=-1)


def lorentz_inner_torch(left: "torch.Tensor", right: "torch.Tensor") -> "torch.Tensor":
    _torch_required()
    return -left[..., 0] * right[..., 0] + (left[..., 1:] * right[..., 1:]).sum(
        dim=-1
    )


def lorentz_distance_sq_torch(
    left: "torch.Tensor",
    right: "torch.Tensor",
    curvature: "torch.Tensor | float",
) -> "torch.Tensor":
    _torch_required()
    c = torch.as_tensor(curvature, dtype=left.dtype, device=left.device).clamp_min(EPS)
    # acosh'(1) is infinite. Hierarchy matrices always contain a zero-distance
    # diagonal, so use a tiny interior floor for finite backpropagation.
    argument = (-c * lorentz_inner_torch(left, right)).clamp_min(1.0 + EPS)
    distance = torch.acosh(argument) / torch.sqrt(c)
    return (distance * distance).clamp_min(0.0)


def sphere_project_torch(value: "torch.Tensor") -> "torch.Tensor":
    _torch_required()
    return F.normalize(value, dim=-1, eps=EPS)


def sphere_distance_sq_torch(
    left: "torch.Tensor", right: "torch.Tensor"
) -> "torch.Tensor":
    _torch_required()
    x = sphere_project_torch(left)
    y = sphere_project_torch(right)
    # acos has infinite slope at +/-1; keep training distances in the smooth
    # interior while NumPy evaluation retains exact endpoint distances.
    dot = (x * y).sum(dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)
    angle = torch.acos(dot)
    return (angle * angle).clamp_min(0.0)


def project_spd_torch(
    matrix: "torch.Tensor", minimum_eigenvalue: float = 1.0e-5
) -> "torch.Tensor":
    _torch_required()
    original_dtype = matrix.dtype
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    # LAPACK's float32 batched eigh can fail on otherwise valid but tightly
    # clustered spectra. Projection is a contract boundary, so perform it in
    # double precision and return to the model dtype afterwards.
    stable = symmetric.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(stable)
    eigenvalues = eigenvalues.clamp_min(float(minimum_eigenvalue))
    projected = (eigenvectors * eigenvalues.unsqueeze(-2)) @ eigenvectors.transpose(
        -1, -2
    )
    return (
        0.5 * (projected + projected.transpose(-1, -2))
    ).to(original_dtype)


def matrix_sqrt_spd_torch(
    matrix: "torch.Tensor",
    minimum_eigenvalue: float = 1.0e-7,
    iterations: int = 12,
) -> "torch.Tensor":
    """Differentiable SPD square root via scaled Newton--Schulz iteration.

    Avoiding eigenvector derivatives is important here: repeated or nearly
    repeated covariance eigenvalues are common in short motion events and can
    yield NaN gradients through ``torch.linalg.eigh`` even when the forward
    eigendecomposition succeeds.
    """

    _torch_required()
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("SPD square root requires square matrices")
    original_dtype = matrix.dtype
    value = 0.5 * (matrix + matrix.transpose(-1, -2))
    value = value.to(torch.float64)
    dimension = value.shape[-1]
    identity = torch.eye(
        dimension, dtype=value.dtype, device=value.device
    )
    value = value + float(minimum_eigenvalue) * identity
    norm = torch.linalg.matrix_norm(
        value, ord="fro", dim=(-2, -1), keepdim=True
    ).clamp_min(float(minimum_eigenvalue))
    y = value / norm
    z = identity.expand(value.shape)
    for _ in range(max(1, int(iterations))):
        update = 0.5 * (3.0 * identity - z @ y)
        y = y @ update
        z = update @ z
    root = y * torch.sqrt(norm)
    root = 0.5 * (root + root.transpose(-1, -2))
    return root.to(original_dtype)


def bures_distance_sq_torch(
    left: "torch.Tensor",
    right: "torch.Tensor",
    minimum_eigenvalue: float = 1.0e-5,
) -> "torch.Tensor":
    _torch_required()
    if left.shape[-2:] != right.shape[-2:]:
        raise ValueError("Bures covariance matrices must share their matrix shape")
    dimension = left.shape[-1]
    identity = torch.eye(
        dimension, dtype=left.dtype, device=left.device
    )
    a = 0.5 * (left + left.transpose(-1, -2)) + float(
        minimum_eigenvalue
    ) * identity
    b = 0.5 * (right + right.transpose(-1, -2)) + float(
        minimum_eigenvalue
    ) * identity
    a_root = matrix_sqrt_spd_torch(a, minimum_eigenvalue)
    middle_root = matrix_sqrt_spd_torch(
        a_root @ b @ a_root, minimum_eigenvalue
    )
    trace = torch.diagonal(
        a + b - 2.0 * middle_root, dim1=-2, dim2=-1
    ).sum(dim=-1)
    return trace.clamp_min(0.0)


def gaussian_wasserstein_distance_sq_torch(
    left_mean: "torch.Tensor",
    left_covariance: "torch.Tensor",
    right_mean: "torch.Tensor",
    right_covariance: "torch.Tensor",
    minimum_eigenvalue: float = 1.0e-5,
) -> "torch.Tensor":
    _torch_required()
    mean_term = ((left_mean - right_mean) ** 2).sum(dim=-1)
    covariance_term = bures_distance_sq_torch(
        left_covariance,
        right_covariance,
        minimum_eigenvalue=minimum_eigenvalue,
    )
    return (mean_term + covariance_term).clamp_min(0.0)


def mixed_product_distance_sq_torch(
    left: Mapping[str, "torch.Tensor"],
    right: Mapping[str, "torch.Tensor"],
    weights: "torch.Tensor",
    *,
    curvature: "torch.Tensor | float",
    minimum_eigenvalue: float = 1.0e-5,
) -> "torch.Tensor":
    _torch_required()
    if weights.shape[-1] != len(FACTOR_NAMES):
        raise ValueError(f"factor weights must end in {len(FACTOR_NAMES)}")
    factor_weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
    lorentz = lorentz_distance_sq_torch(
        left["lorentz"], right["lorentz"], curvature
    )
    sphere = sphere_distance_sq_torch(left["sphere"], right["sphere"])
    gaussian = gaussian_wasserstein_distance_sq_torch(
        left["gaussian_mean"],
        left["gaussian_covariance"],
        right["gaussian_mean"],
        right["gaussian_covariance"],
        minimum_eigenvalue,
    ).mean(dim=-1)
    euclidean = ((left["euclidean"] - right["euclidean"]) ** 2).sum(dim=-1)
    return (
        factor_weights[..., 0] * lorentz
        + factor_weights[..., 1] * sphere
        + factor_weights[..., 2] * gaussian
        + factor_weights[..., 3] * euclidean
    ).clamp_min(0.0)


__all__ = [
    "EPS",
    "FACTOR_NAMES",
    "bures_distance_sq_np",
    "bures_distance_sq_torch",
    "gaussian_wasserstein_distance_sq_np",
    "gaussian_wasserstein_distance_sq_torch",
    "lorentz_distance_sq_np",
    "lorentz_distance_sq_torch",
    "lorentz_inner_np",
    "lorentz_inner_torch",
    "lorentz_project_np",
    "lorentz_project_torch",
    "matrix_sqrt_spd_np",
    "matrix_sqrt_spd_torch",
    "mixed_product_distance_sq_np",
    "mixed_product_distance_sq_torch",
    "normalized_factor_weights_np",
    "project_spd_np",
    "project_spd_torch",
    "sphere_distance_sq_np",
    "sphere_distance_sq_torch",
    "sphere_project_np",
    "sphere_project_torch",
    "torch",
]
