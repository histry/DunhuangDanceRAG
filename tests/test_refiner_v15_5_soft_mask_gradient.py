import torch

from training import motion_models as m


def test_v15_5_protocol_is_explicit():
    assert (
        m.REFINER_TANGENT_GRADIENT_PROTOCOL
        == "soft_confidence_forward_support_backward_v1"
    )


def test_soft_confidence_forward_is_exact():
    raw = torch.tensor(
        [
            [
                [0.25, -0.50, 1.25],
                [-2.00, 3.00, -4.00],
                [0.75, 0.50, -0.25],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    confidence = torch.tensor(
        [
            [
                [0.0],
                [0.2],
                [1.0],
            ]
        ],
        dtype=torch.float32,
    )

    actual = m._soft_confidence_forward_support_backward(
        raw,
        confidence,
    )

    expected = raw.detach() * confidence

    # V15.5 is a backward-only intervention.
    assert torch.equal(
        actual.detach(),
        expected,
    )


def test_soft_confidence_backward_uses_binary_support():
    raw = torch.tensor(
        [
            [
                [0.25, -0.50, 1.25],
                [-2.00, 3.00, -4.00],
                [0.75, 0.50, -0.25],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    confidence = torch.tensor(
        [
            [
                [0.0],
                [0.2],
                [1.0],
            ]
        ],
        dtype=torch.float64,
    )

    value = m._soft_confidence_forward_support_backward(
        raw,
        confidence,
    )

    value.sum().backward()

    expected_support = torch.broadcast_to(
        (confidence > 0).to(raw.dtype),
        raw.shape,
    )

    assert torch.equal(
        raw.grad,
        expected_support,
    )


def test_soft_confidence_chain_rule_keeps_forward_value():
    raw = torch.tensor(
        [
            [
                [1.0, -2.0, 3.0],
                [4.0, -5.0, 6.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    confidence = torch.tensor(
        [
            [
                [0.25],
                [0.0],
            ]
        ],
        dtype=torch.float64,
    )

    value = m._soft_confidence_forward_support_backward(
        raw,
        confidence,
    )

    loss = value.square().sum()
    loss.backward()

    forward_expected = (
        raw.detach()
        * confidence
    )

    support = torch.broadcast_to(
        (confidence > 0).to(raw.dtype),
        raw.shape,
    )

    gradient_expected = (
        2.0
        * forward_expected
        * support
    )

    assert torch.equal(
        value.detach(),
        forward_expected,
    )

    assert torch.equal(
        raw.grad,
        gradient_expected,
    )


def test_zero_support_has_zero_backward_gradient():
    raw = torch.randn(
        2,
        5,
        24,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )

    confidence = torch.zeros(
        2,
        5,
        24,
        1,
        dtype=torch.float64,
    )

    value = m._soft_confidence_forward_support_backward(
        raw,
        confidence,
    )

    value.sum().backward()

    assert torch.count_nonzero(
        value
    ).item() == 0

    assert torch.count_nonzero(
        raw.grad
    ).item() == 0
