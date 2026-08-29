import torch

from training.refiner_optimizer import (
    REFINER_UPDATE_PROTOCOL,
    _minimum_norm_simplex_weights,
)


def test_v14_protocol():
    assert (
        REFINER_UPDATE_PROTOCOL
        == "full_cycle_component_guard_pareto_rescue_v7"
    )


def test_minimum_norm_simplex_identity():
    gram = torch.eye(2, dtype=torch.float64)

    weights = _minimum_norm_simplex_weights(gram)

    torch.testing.assert_close(
        weights,
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        rtol=1e-7,
        atol=1e-7,
    )


def test_minimum_norm_simplex_opposite_gradients_detects_zero_hull():
    gram = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    weights = _minimum_norm_simplex_weights(gram)

    norm_sq = float(
        weights @ gram @ weights
    )

    assert abs(norm_sq) < 1e-10
