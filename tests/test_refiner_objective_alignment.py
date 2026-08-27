"""Regressions for the SMPL14 V4 repair/clean-identity training mismatch."""

import numpy as np
import pytest

from training import motion_models as m


pytestmark = pytest.mark.skipif(m.torch is None, reason="PyTorch unavailable")


def motion(frames=40, device="cpu"):
    x = np.zeros((1, frames, 151), dtype=np.float32)
    x[..., 7:] = np.tile(m.identity6d_np(), 24)
    x[..., 5] = 0.95
    return m.torch.from_numpy(x).to(device)


def masks(x):
    return (
        x.new_ones(x.shape[:2] + (24,)),
        x.new_ones(x.shape[:2] + (1,)),
        x.new_zeros(x.shape[:2] + (1,)),
    )


def test_clean_temporal_supervision_survives_disabled_repair_tangent_loss():
    clean = motion()
    smooth = clean.clone()
    jitter = clean.clone()
    smooth[..., 4] += 0.001
    jitter[..., 4] += 0.001 * m.torch.tensor([1.0, -1.0] * 20)
    cfg = m.MotionGenerationConfig(
        device="cpu", product_refiner_temporal_supervision_weight=0.0
    )
    steady_loss, _ = m._product_refiner_clean_identity_loss(
        smooth, clean, *masks(clean), cfg
    )
    jitter.requires_grad_(True)
    jitter_loss, terms = m._product_refiner_clean_identity_loss(
        jitter, clean, *masks(clean), cfg
    )
    assert jitter_loss > steady_loss * 2.0
    assert terms["fk_temporal"].item() > 0.0
    jitter_loss.backward()
    assert m.torch.isfinite(jitter.grad).all()
    assert jitter.grad[..., 4].abs().sum() > 0.0


def test_geometry_margin_is_per_window_relative_not_corruption_amplitude():
    clean = motion()
    seam = clean.new_zeros((1, 40, 1))
    seam[:, 12:28] = 1.0
    losses = []
    for amount in (0.005, 0.05):
        bad = clean.clone()
        bad[:, 12:28, 4] += amount
        _, terms = m._product_motion_losses(
            bad, clean, bad, *masks(clean), m.MotionGenerationConfig(),
            seam_mask=seam,
        )
        losses.append(terms["repair_margin"].item())
    assert losses[0] == pytest.approx(0.10, abs=1e-5)
    assert losses[1] == pytest.approx(losses[0], abs=1e-5)


def test_geometry_training_support_matches_unweighted_validation_seam_core():
    clean = motion()
    bad = clean.clone()
    bad[:, 12:28, 4] += 0.05
    seam = clean.new_zeros((1, 40, 1))
    seam[:, 8:32] = 0.35
    seam[:, 12:28] = 1.0
    joint, root, contact = masks(clean)
    cfg = m.MotionGenerationConfig()
    _, first = m._product_motion_losses(
        bad, clean, bad, joint, root, contact, cfg, seam_mask=seam
    )
    _, second = m._product_motion_losses(
        bad, clean, bad, joint * 0.18, root * 0.9, contact, cfg,
        seam_mask=seam,
    )
    expected = float(m.product_log_torch(clean[:, 12:28], bad[:, 12:28]).abs().mean())
    assert first["active_reconstruction"].item() == pytest.approx(expected)
    assert second["active_reconstruction"].item() == pytest.approx(expected)


def test_supported_tangent_filter_does_not_edit_source_or_leak_support():
    clean = motion(frames=48)
    output = clean.new_zeros((1, 48, m.PRODUCT_STATE_DIM))
    output[..., 4] = 0.03
    joint, root, contact = masks(clean)
    joint.zero_()
    root.zero_()
    root[:, 10:38] = 1.0
    cfg = m.MotionGenerationConfig(device="cpu")
    pred = m._decode_product_refiner_output(clean, output, joint, root, contact, cfg)
    raw = clean.clone()
    raw[:, 10:38, 4] += 0.03
    raw_jerk = m.torch.diff(raw[..., 4], n=3, dim=1).abs().max()
    filtered_jerk = m.torch.diff(pred[..., 4], n=3, dim=1).abs().max()
    assert filtered_jerk < raw_jerk * 0.5
    m.torch.testing.assert_close(pred[:, :10], clean[:, :10], rtol=0, atol=0)
    m.torch.testing.assert_close(pred[:, 38:], clean[:, 38:], rtol=0, atol=0)
    assert pred[:, 24, 4].item() == pytest.approx(0.03, abs=1e-6)


def test_zero_refiner_output_preserves_clean_contacts_and_geometry():
    clean = motion()
    clean[:, ::2, :4] = 1.0
    joint, root, contact = masks(clean)
    contact.fill_(1.0)
    pred = m._decode_product_refiner_output(
        clean, clean.new_zeros((1, 40, m.PRODUCT_STATE_DIM)),
        joint, root, contact, m.MotionGenerationConfig(device="cpu"),
    )
    m.torch.testing.assert_close(pred, clean, rtol=0, atol=1e-7)


def test_identity_initialized_contact_head_can_correct_wrong_observation():
    clean = motion()
    bad = clean.clone()
    bad[..., :4] = 1.0
    output = clean.new_zeros((1, 40, m.PRODUCT_STATE_DIM), requires_grad=True)
    joint, root, contact = masks(clean)
    contact.fill_(1.0)
    cfg = m.MotionGenerationConfig(device="cpu")
    pred = m._decode_product_refiner_output(bad, output, joint, root, contact, cfg)
    _, terms = m._product_motion_losses(pred, clean, bad, joint, root, contact, cfg)
    terms["contact"].backward()
    assert m.torch.isfinite(output.grad).all()
    assert output.grad[..., :4].abs().sum() > 0.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seam_fk_errors_match_numpy_and_have_gradients(device):
    if device == "cuda" and not m.torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    clean = motion(device=device)
    pred = clean.clone()
    pred[:, 12:28, 4] += 0.03
    pred.requires_grad_(True)
    seam = clean.new_zeros((1, 40, 1))
    seam[:, 12:28] = 1.0
    cfg = m.MotionGenerationConfig(device=device)
    errors = m._seam_fk_errors_torch(
        m.fk_24_torch(pred), m.fk_24_torch(clean), seam, cfg
    )
    reference = m._seam_fk_temporal_errors_np(
        pred[0].detach().cpu().numpy(), clean[0].cpu().numpy(),
        seam[0].cpu().numpy(), cfg,
    )
    for name, value in errors.items():
        assert value.item() == pytest.approx(reference[name], rel=2e-5, abs=2e-6)
    sum(value.mean() for value in errors.values()).backward()
    assert m.torch.isfinite(pred.grad).all()


def test_short_optimizer_fit_improves_repair_without_modifying_clean_input():
    """A learnability smoke test, not a SMPL14 generalization claim."""
    torch = m.torch
    rng = torch.random.get_rng_state()
    thread_count = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(121)
        clean = motion()
        bad = clean.clone()
        bad[:, 12:28, 4] += 0.03 * torch.sin(torch.linspace(0.1, 3.04, 16))
        seam = clean.new_zeros((1, 40, 1))
        seam[:, 12:28] = 1.0
        joint, root, contact = masks(clean)
        joint.zero_()
        root.zero_()
        root[:, 8:32] = 1.0
        model = m.ProductManifoldTemporalRefiner(hidden=32)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        cfg = m.MotionGenerationConfig(device="cpu")
        condition = clean.new_zeros((2, 32))
        values = []
        for _ in range(81):
            outputs = model(torch.cat([bad, clean]), condition, seam.repeat(2, 1, 1), joint.repeat(2, 1, 1))
            pred = m._decode_product_refiner_output(bad, outputs[:1], joint, root, contact, cfg)
            identity = m._decode_product_refiner_output(clean, outputs[1:], joint, root, contact, cfg)
            repair_loss, terms = m._product_motion_losses(pred, clean, bad, joint, root, contact, cfg, seam_mask=seam)
            identity_loss, _ = m._product_refiner_clean_identity_loss(identity, clean, joint, root, contact, cfg)
            loss = repair_loss + cfg.product_refiner_clean_identity_weight * identity_loss
            values.append(float(terms["active_reconstruction"].detach()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        assert values[-1] < values[0] * 0.97
        assert float(m.product_log_torch(clean, identity).abs().mean().detach()) < 0.005
    finally:
        torch.set_num_threads(thread_count)
        torch.random.set_rng_state(rng)
