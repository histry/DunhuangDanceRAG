"""Properties the old nonlocal, absolute-position refiner did not satisfy."""
import numpy as np
import pytest
import torch

from training import motion_models as m


def sample(device):
    frames = 120
    x = torch.zeros((2, frames, 151), device=device)
    x[..., 7:] = torch.as_tensor(np.tile(m.identity6d_np(), 24), device=device)
    x[..., 5] = .95
    x[..., 4] = torch.linspace(0, .3, frames, device=device)
    seam = torch.zeros((2, frames, 1), device=device)
    seam[:, 40:76] = .3
    seam[:, 44:72] = 1
    return x, torch.zeros((2, frames, 32), device=device), seam, seam.expand(-1, -1, 24)


@pytest.fixture(params=['cpu', 'cuda'])
def setup(request):
    device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(123)
    model = m.ProductManifoldTemporalRefiner(hidden=32).to(device)
    # A zero output head would trivially pass all invariance checks.
    with torch.no_grad():
        model.out.weight.normal_(0, .01)
    return model, sample(device)


def test_world_translation_does_not_change_repair_direction(setup):
    model, (x, cond, seam, joint) = setup
    changed = x.clone()
    changed[..., 4] += 10
    changed[..., 6] -= 5
    before = model(x, cond, seam, joint)
    after = model(changed, cond, seam, joint)
    assert before.abs().max() > 1e-4
    torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)
    elevated = x.clone()
    elevated[..., 5] += .1
    torch.testing.assert_close(m._refiner_motion_features(elevated)[..., 5], elevated[..., 5])
    assert not torch.equal(m._refiner_motion_features(x)[..., 5],
                           m._refiner_motion_features(elevated)[..., 5])


def test_distant_context_does_not_change_the_local_seam(setup):
    model, (x, cond, seam, joint) = setup
    changed = x.clone()
    changed[:, :10, 5] += .25
    changed_cond = cond.clone()
    changed_cond[:, :10] = 3
    before = model(x, cond, seam, joint)
    after = model(changed, changed_cond, seam, joint)
    torch.testing.assert_close(before[:, 44:72], after[:, 44:72], atol=0, rtol=0)


def test_cropping_preserved_context_does_not_change_the_seam(setup):
    model, inputs = setup
    full = model(*inputs)
    cropped = model(*(value[:, 10:110] for value in inputs))
    # Same seam, same external anchors, and >16+1 frames of convolution context.
    torch.testing.assert_close(full[:, 44:72], cropped[:, 34:62], atol=1e-6, rtol=1e-5)


def test_gradients_reach_local_motion_but_not_remote_frames(setup):
    model, (x, cond, seam, joint) = setup
    x.requires_grad_(True)
    model(x, cond, seam, joint)[:, 56:60].square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, :10]) == 0
    assert x.grad[:, 40:77].abs().sum() > 0


def test_fresh_model_is_identity_with_trainable_output_head(setup):
    model, inputs = setup
    with torch.no_grad():
        model.out.weight.zero_()
    out = model(*inputs)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert model.out.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.out.weight.grad).all()


def test_input_protocol_is_checked_not_just_stored():
    cfg = m.MotionGenerationConfig()
    contract = m.motion_checkpoint_contract(cfg, 'boundary_refiner')
    assert contract['refiner_input_protocol'] == m.REFINER_INPUT_PROTOCOL
    contract.pop('refiner_input_protocol')
    with pytest.raises(RuntimeError, match='refiner_input_protocol'):
        m.assert_motion_checkpoint_contract({'motion_contract':contract}, cfg, 'old.pt', 'boundary_refiner')


def test_failure_summary_separates_temporal_gain_from_jerk_and_safety():
    from training.refiner_bridge_diagnostics import failure_breakdown
    from tests.test_bridge_feasibility import passing_metrics
    metrics = passing_metrics()
    for row in metrics['windows'] + metrics['cross_event']['windows']:
        row['observable'].update(temporal_gain_only=False, jerk_non_regression=True,
                                 temporal_accepted=False, endpoint_gain=.1, temporal_gain=.01)
    case = metrics['cross_event']['windows'][0]
    case['observable'].update(temporal_gain_only=True, jerk_non_regression=False)
    case['safety'] = {'accepted':False,'reasons':['joint_jerk_mps3_max_regressed']}
    summary = failure_breakdown(metrics)
    assert summary['single_recording/10']['temporal_gain_pass'] == 0
    assert summary['single_recording/10']['jerk_non_regression_pass'] == 8
    assert summary['cross_event/10']['temporal_gain_pass'] == 1
    assert summary['cross_event/10']['jerk_non_regression_pass'] == 7
    assert summary['cross_event/10']['temporal_pass'] == 0
    assert summary['cross_event/10']['physical_failure_reasons'] == {'joint_jerk_mps3_max_regressed':1}


def test_current_decoder_is_exercised_by_diagnostic_evaluation():
    from training import refiner_bridge_diagnostics as d
    cfg = m.MotionGenerationConfig(device='cpu')
    x, cond, seam, joint = sample('cpu')
    x[:, 50:60, 4] += .01  # informative temporal defect, not constant velocity
    batch = m._prepare_refiner_batch(x.numpy(),x.numpy(),seam.numpy(),cond.numpy(),cfg,torch.device('cpu'))
    banks = {('seen',role):batch for role in ('single_recording','cross_event')}
    model = m.ProductManifoldTemporalRefiner(hidden=16)
    metrics = d.evaluate(model,banks,'seen',cfg)
    for row in metrics['windows'] + metrics['cross_event']['windows']:
        assert row['decoder']['raw_tangent_rms'] == 0
        assert row['decoder']['applied_tangent_rms'] == 0
        assert row['decoder']['root_cap_fraction'] == 0
    summary = d.failure_breakdown(metrics)
    assert len(summary) == 2
    assert all(row['temporal_pass'] == 0 for row in summary.values())
