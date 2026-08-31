import pytest
import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d


def test_v15_2_protocol_contract():
    assert (
        m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
        == "scientific_feasibility_smooth_bottleneck_observable_v8"
    )

    assert (
        d.SCHEMA
        == "refiner_observable_bridge_diagnostic_v15_3_1"
    )

    assert (
        m.SCIENTIFIC_BOTTLENECK_SMOOTH_EPS
        == 1.0e-3
    )


def test_smooth_bottleneck_is_zero_preserving():
    endpoint = torch.zeros(
        4,
        dtype=torch.float64,
    )

    temporal = torch.zeros_like(endpoint)

    joint = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    assert torch.equal(
        joint,
        torch.zeros_like(joint),
    )


def test_smooth_bottleneck_is_symmetric():
    endpoint = torch.tensor(
        [0.002, 0.010, 0.020],
        dtype=torch.float64,
    )

    temporal = torch.tensor(
        [0.011, 0.004, 0.015],
        dtype=torch.float64,
    )

    ab = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    ba = m._joint_scientific_deficit(
        temporal,
        endpoint,
    )

    torch.testing.assert_close(
        ab,
        ba,
        rtol=0,
        atol=1e-14,
    )


def test_smooth_bottleneck_stays_near_hard_max():
    endpoint = torch.tensor(
        [0.002, 0.010, 0.020],
        dtype=torch.float64,
    )

    temporal = torch.tensor(
        [0.011, 0.004, 0.015],
        dtype=torch.float64,
    )

    joint = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    hard = torch.maximum(
        endpoint,
        temporal,
    )

    eps = m.SCIENTIFIC_BOTTLENECK_SMOOTH_EPS

    # Zero-preserving smooth max is never above the hard max and
    # differs from it by at most eps/2.
    assert torch.all(
        joint <= hard + 1e-14
    )

    assert torch.all(
        joint >= hard - eps / 2 - 1e-14
    )


def test_equal_deficits_split_gradient_equally():
    endpoint = torch.tensor(
        [0.01],
        dtype=torch.float64,
        requires_grad=True,
    )

    temporal = torch.tensor(
        [0.01],
        dtype=torch.float64,
        requires_grad=True,
    )

    joint = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    joint.sum().backward()

    torch.testing.assert_close(
        endpoint.grad,
        torch.tensor(
            [0.5],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-12,
    )

    torch.testing.assert_close(
        temporal.grad,
        torch.tensor(
            [0.5],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=1e-12,
    )


def test_cross10_like_near_tie_gives_both_components_gradient():
    # Representative scale from the V15.1 cross-event failure:
    # endpoint ~= 0.01448, temporal ~= 0.01428.
    endpoint = torch.tensor(
        [0.014483388847300108],
        dtype=torch.float64,
        requires_grad=True,
    )

    temporal = torch.tensor(
        [0.014275850264744974],
        dtype=torch.float64,
        requires_grad=True,
    )

    joint = m._joint_scientific_deficit(
        endpoint,
        temporal,
    )

    joint.sum().backward()

    assert endpoint.grad.item() > 0.0
    assert temporal.grad.item() > 0.0

    # Near a tie, neither component should be effectively switched off.
    assert endpoint.grad.item() > 0.30
    assert temporal.grad.item() > 0.30



def test_smooth_bottleneck_rejects_shape_mismatch():
    endpoint = torch.zeros(
        2,
        dtype=torch.float64,
    )

    temporal = torch.zeros(
        3,
        dtype=torch.float64,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        m._joint_scientific_deficit(
            endpoint,
            temporal,
        )
