"""Zero output is an identity with gradients, not a tolerance exemption."""
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from motion_geometry import product_manifold as p
from motion_geometry.rotations import so3_exp_np, matrix_to_rot6d_np, rot6d_to_matrix_np
from training import motion_models as m
from training import bridge_feasibility as f

torch = m.torch


def reference(prefix=(2, 12), *, gauge=False):
    rng = np.random.default_rng(20260828)
    result = rng.uniform(0, 1, size=prefix+(151,)).astype(np.float32)
    axes = rng.normal(size=prefix+(24, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    angles = np.resize([0.0, 1e-6, 0.3, 1.8, np.pi-1e-5], prefix+(24, 1))
    six = matrix_to_rot6d_np(so3_exp_np(axes*angles), project=False)
    if gauge:
        # The same rotations with non-unit/non-orthogonal, nondegenerate 6D
        # columns. Gram-Schmidt must commute with a left rotation of both.
        six[..., 3:] = 0.8*six[..., 3:] + 0.15*six[..., :3]
        six[..., :3] *= 1.3
    result[..., 7:] = six.reshape(prefix+(144,))
    return result


def require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


@pytest.mark.parametrize("prefix", [(), (7,), (2, 3, 5)])
def test_numpy_zero_is_exact_for_every_batch_prefix(prefix):
    ref = reference(prefix)
    out = p.product_exp_np(ref, np.zeros(prefix+(75,), np.float32))
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(
        p.masked_retract_np(ref, np.zeros(prefix+(75,), np.float32)), ref
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_tangent_keeps_exact_values_and_both_gradients(device, dtype):
    require_device(device)
    ref = torch.tensor(reference((2, 3, 5)), device=device, dtype=dtype, requires_grad=True)
    delta = torch.zeros((2, 3, 5, 75), device=device, dtype=dtype, requires_grad=True)
    out = p.product_exp_torch(ref, delta)
    assert torch.equal(out, ref)
    weights = torch.linspace(-1, 1, out.numel(), device=device, dtype=dtype).reshape_as(out)
    (out*weights).sum().backward()
    torch.testing.assert_close(ref.grad, weights, rtol=0, atol=0)
    assert torch.isfinite(delta.grad).all()
    assert delta.grad[..., :3].abs().sum() > 0
    assert delta.grad[..., 3:].abs().sum() > 0


def test_zero_retraction_passes_numerical_gradient_check():
    ref = torch.tensor(reference((1,)), dtype=torch.float64, requires_grad=True)
    delta = torch.zeros((1, 75), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(p.product_exp_torch, (ref, delta), fast_mode=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonzero_action_matches_body_so3_and_numpy_for_noncanonical_gauge(device):
    require_device(device)
    ref = reference((2, 9), gauge=True)
    delta = np.random.default_rng(7).normal(0, .08, size=(2, 9, 75)).astype(np.float32)
    actual_np = p.product_exp_np(ref, delta)
    actual_torch = p.product_exp_torch(torch.tensor(ref, device=device), torch.tensor(delta, device=device))
    expected = rot6d_to_matrix_np(ref[..., 7:].reshape(2, 9, 24, 6)) @ so3_exp_np(delta[..., 3:].reshape(2, 9, 24, 3))
    for actual in (actual_np, actual_torch.cpu().numpy()):
        np.testing.assert_allclose(rot6d_to_matrix_np(actual[..., 7:].reshape(2, 9, 24, 6)), expected, atol=1e-6)
        np.testing.assert_array_equal(actual[..., :4], ref[..., :4])
    np.testing.assert_allclose(actual_torch.cpu().numpy(), actual_np, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_zero_factors_and_masks_have_no_projection_drift(device):
    require_device(device)
    ref = torch.tensor(reference((2, 6)), device=device)
    delta = torch.zeros((2, 6, 75), device=device)
    delta[:, ::2, 3:6] = .03
    delta.requires_grad_(True)
    mask = torch.ones((2, 6, 24), device=device)
    mask[..., 2] = 0
    result = p.masked_retract_torch(ref, delta, joint_mask=mask, max_rotation_rad=.2)
    assert torch.equal(result[:, 1::2], ref[:, 1::2])
    assert torch.equal(result[..., 13:], ref[..., 13:])
    result.sum().backward()
    assert torch.isfinite(delta.grad).all()
    grad = delta.grad[..., 3:].reshape(2, 6, 24, 3)
    assert torch.count_nonzero(grad[..., 2, :]) == 0
    assert grad[:, 1::2, 1, :].abs().sum() > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_refiner_zero_head_high_jerk_reference_and_real_regression(device):
    require_device(device)
    cfg = m.MotionGenerationConfig(device=device)
    ref_np = reference((1, 120))
    # High-jerk context exercises the already-over-limit, no-regression branch.
    ref_np[..., 4:7] = 0
    ref_np[..., 5] = .95
    ref_np[0, :, 4] = np.where(np.arange(120)%2, -.02, .02)
    ref = torch.tensor(ref_np, device=device)
    output = torch.zeros((1, 120, 79), device=device, requires_grad=True)
    joint = torch.ones((1, 120, 24), device=device)
    root = torch.ones((1, 120, 1), device=device)
    contacts = torch.ones_like(root)
    prediction = m._decode_product_refiner_output(ref, output, joint, root, contacts, cfg)
    assert torch.equal(prediction, ref)
    prediction.sum().backward()
    assert torch.isfinite(output.grad).all()
    assert output.grad[..., 7:].abs().sum() > 0
    gate = m._fixed_support_stage_gate(ref_np[0], prediction.detach().cpu().numpy()[0], cfg)
    assert gate["detail"]["before_extremity_jerk_window_p95_max_mps3"] > 1080
    assert gate["accepted"], gate["reasons"]
    spike = ref_np[0].copy()
    spike[60, 4] += .20
    regression = m._fixed_support_stage_gate(ref_np[0], spike, cfg)
    assert not regression["accepted"]
    assert any("jerk" in reason for reason in regression["reasons"])


def test_zero_edit_failure_stops_before_direct_optimization(tmp_path):
    cfg = m.MotionGenerationConfig(device="cpu")
    b = {"bad":torch.tensor(reference((1, 16))),
         "joint":torch.ones((1, 16, 24)), "root":torch.ones((1, 16, 1)),
         "contact":torch.ones((1, 16, 1)), "seam":torch.ones((1, 16, 1))}
    args = SimpleNamespace(out_dir=str(tmp_path/"blocked"), direct_steps=200)
    def broken_decoder(ref, *args, **kwargs):
        changed = ref.clone()
        changed[:, 8, 4] += .1
        return changed
    with mock.patch.object(m, "_decode_product_refiner_output", side_effect=broken_decoder), \
         mock.patch.object(f, "direct_optimize", side_effect=AssertionError("optimization must not start")):
        code = f.run_foundation(args, cfg, {("seen", "single_recording"):b}, {}, {}, {}, [], {})
    report = json.loads((tmp_path/"blocked"/"foundation_report.json").read_text(encoding="utf8"))
    assert code == 2
    assert report["blocked_before_direct_optimization"]
    assert report["roundtrip"]["exact_identity_count"] == 0
    assert report["direct"] == {}


def test_decoder_and_auditor_implementation_hashes_are_in_fingerprint():
    from training import refiner_bridge_diagnostics as d
    with mock.patch.object(d.common, "_fingerprint", return_value={"implementation_sha256":{}}):
        value = d.fingerprint(None, None)
    assert value["retraction_protocol"] == p.RETRACTION_PROTOCOL
    for name in ("product_manifold", "physical_geometry", "physical_quality"):
        assert len(value["implementation_sha256"][name]) == 64


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_observable_loss_and_audit_share_fk_precision_and_keep_gradients(device):
    require_device(device)
    cfg = m.MotionGenerationConfig(device=device)
    ref_np = reference((1, 48))
    ref = torch.tensor(ref_np, device=device)
    delta = torch.full((1, 48, 75), .001, device=device, requires_grad=True)
    prediction = p.product_exp_torch(ref, delta)
    seam = torch.zeros((1, 48, 1), device=device)
    seam[:, 16:32] = 1
    _, terms = m._observable_refiner_objective(prediction, ref, seam, cfg)
    audit = m._observable_boundary_audit(prediction.detach().cpu().numpy()[0], ref_np[0], seam.cpu().numpy()[0], cfg)
    assert audit["numeric_contract"] == "float64_fk_before_temporal_differences_v1"
    for term, key in (("seam_jerk", "seam_jerk_mps3"), ("seam_acceleration", "seam_acceleration_mps2"),
                      ("seam_velocity", "endpoint_velocity_jump_mps")):
        assert float(terms[term].detach().cpu()) == pytest.approx(audit["after"][key], rel=1e-10, abs=1e-8)
    assert m._observable_boundary_joints_torch(prediction).dtype == torch.float64
    terms["seam_jerk"].backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0
