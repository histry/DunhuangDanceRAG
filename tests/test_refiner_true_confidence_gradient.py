"""Regression for V15.5's surrogate-gradient/Armijo mismatch."""
import pytest
import torch

from training import motion_models as m
from training.refiner_optimizer import checked_refiner_step


def shared_parameter_decoder_loss(theta):
    reference = torch.zeros(2, 20, 151, dtype=theta.dtype, device=theta.device)
    reference[..., 7:] = torch.tensor(m.identity6d_np(), dtype=theta.dtype,
                                     device=theta.device).repeat(24)
    reference[..., 5] = .95
    basis = torch.zeros(2, 20, 79, dtype=theta.dtype, device=theta.device)
    basis[..., 4] = 1  # shared root-x proposal, below the cap
    confidence = torch.tensor([.18, 1.0], dtype=theta.dtype,
                              device=theta.device)[:, None, None]
    joint = confidence.expand(2, 20, 24)
    root = confidence.expand(2, 20, 1)
    prediction = m._decode_product_refiner_output(
        reference, theta * basis, joint, root, root,
        m.MotionGenerationConfig(device=str(theta.device)))
    return 2 * prediction[0, :, 4].mean() - prediction[1, :, 4].mean()


def test_decoder_parameter_gradient_matches_forward_finite_difference():
    theta = torch.tensor(.01, dtype=torch.float64, requires_grad=True)
    actual = torch.autograd.grad(shared_parameter_decoder_loss(theta), theta)[0]
    epsilon = 1e-6
    with torch.no_grad():
        numeric = (shared_parameter_decoder_loss(theta + epsilon)
                   - shared_parameter_decoder_loss(theta - epsilon)) / (2 * epsilon)
    assert float(numeric) == pytest.approx(-.64, abs=1e-10)
    torch.testing.assert_close(actual, numeric, rtol=1e-8, atol=1e-10)


def test_checked_optimizer_uses_true_decoder_descent_direction():
    theta = torch.nn.Parameter(torch.tensor(.01, dtype=torch.float64))
    optimizer = torch.optim.AdamW([theta], lr=.001, weight_decay=0)
    loss = shared_parameter_decoder_loss(theta)
    loss.backward()
    report = checked_refiner_step(
        optimizer, loss, lambda: shared_parameter_decoder_loss(theta))
    assert report["optimizer_update_accepted"]
    assert report["adam_directional_derivative"] < 0
    assert not report["used_gradient_rescue"]
    assert report["loss_after"] < report["loss_before"]
    assert float(theta.detach()) > .01


def test_rejected_gradient_protocol_cannot_share_resume_hash(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu")
    actual = m._training_config_sha256(cfg, stage="refiner")
    diffusion = m._training_config_sha256(cfg, stage="diffusion")
    monkeypatch.setattr(m, "REFINER_TANGENT_GRADIENT_PROTOCOL",
                        "soft_confidence_forward_support_backward_v1")
    assert m._training_config_sha256(cfg, stage="refiner") != actual
    assert m._training_config_sha256(cfg, stage="diffusion") == diffusion


def test_decoder_gradcheck_and_zero_support():
    reference = torch.zeros(1, 12, 151, dtype=torch.float64)
    reference[..., 7:] = torch.tensor(m.identity6d_np(), dtype=torch.float64).repeat(24)
    joint = torch.zeros(1, 12, 24, dtype=torch.float64)
    joint[:, 3:9] = .18
    root = joint[..., :1]
    cfg = m.MotionGenerationConfig(device="cpu")
    raw = torch.linspace(.001, .003, 12, dtype=torch.float64).requires_grad_()

    def decode(value):
        output = torch.zeros(1, 12, 79, dtype=value.dtype)
        output[..., 4] = value
        return m._decode_product_refiner_output(
            reference, output, joint, root, root, cfg)[..., 4]

    assert torch.autograd.gradcheck(decode, (raw,), eps=1e-6, atol=1e-8, rtol=1e-5)
    prediction = decode(raw)
    grad = torch.autograd.grad(prediction.sum(), raw)[0]
    assert torch.count_nonzero(grad[:3]) == torch.count_nonzero(grad[9:]) == 0
    assert torch.count_nonzero(prediction[:, :3]) == 0
    assert torch.count_nonzero(prediction[:, 9:]) == 0


def test_decoder_strength_override_changes_refiner_resume_contract(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu")
    monkeypatch.setenv("MOTION_REFINER_CORE_STRENGTH", "0.02")
    before = m._training_config_sha256(cfg, stage="refiner")
    monkeypatch.setenv("MOTION_REFINER_CORE_STRENGTH", "0.5")
    # Same forward weights under a different regional strength are a different
    # optimization problem and must not silently resume the same experiment.
    assert m._training_config_sha256(cfg, stage="refiner") != before


@pytest.mark.parametrize("value", ["nan", "inf", "-0.01", "1.01", "bad"])
def test_invalid_decoder_strength_fails_closed(monkeypatch, value):
    monkeypatch.setenv("MOTION_REFINER_CORE_STRENGTH", value)
    with pytest.raises(ValueError):
        m._refiner_decode_strengths(m.MotionGenerationConfig(device="cpu"))


def test_resolved_decoder_strengths_are_serialized(monkeypatch):
    import dataclasses

    monkeypatch.setenv("MOTION_REFINER_CORE_STRENGTH", "0.07")
    monkeypatch.setenv("MOTION_REFINER_TRANSITION_STRENGTH", "0.8")
    cfg = m.MotionGenerationConfig(device="cpu").apply_env()
    saved = dataclasses.asdict(cfg)
    monkeypatch.delenv("MOTION_REFINER_CORE_STRENGTH")
    monkeypatch.delenv("MOTION_REFINER_TRANSITION_STRENGTH")
    replay = m.MotionGenerationConfig(**saved)
    assert m._refiner_decode_strengths(replay) == {"core": .07, "transition": .8}
