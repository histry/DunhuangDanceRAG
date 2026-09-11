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
    # Exact binary coordinates isolate translation invariance from float32
    # input quantization, which no downstream float64 derivative can undo.
    x[..., 4] = torch.arange(frames, device=device) / 256.0
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


def test_fk_dynamics_features_match_gate_units_and_have_no_hidden_target():
    x, _, seam, _ = sample("cpu")
    features = m._refiner_fk_dynamics_features(x, seam, 30.0)
    assert features.shape == (
        x.shape[0], x.shape[1], m.REFINER_FK_DYNAMICS_FEATURE_DIM
    )
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features[:, :40]) == 0
    shifted = x.clone()
    shifted[..., 4] += 8.0
    shifted[..., 6] -= 3.0
    torch.testing.assert_close(
        features,
        m._refiner_fk_dynamics_features(shifted, seam, 30.0),
        atol=2e-5,
        rtol=2e-5,
    )
    changed = x.clone()
    changed[:, 50, 7:13] += 0.05
    assert not torch.equal(
        features[:, 48:54],
        m._refiner_fk_dynamics_features(changed, seam, 30.0)[:, 48:54],
    )


def test_fk_dynamics_explicitly_cover_third_difference_support_without_config_halo():
    x, _, _, _ = sample("cpu")
    seam = torch.zeros((2, x.shape[1], 1))
    seam[:, 44:72] = 1.0
    changed = x.clone()
    changed[:, 72:75, 7:13] += torch.tensor([0.03, -0.02, 0.01, 0, 0, 0])
    features = m._refiner_fk_dynamics_features(changed, seam, 30.0)
    # Core [44,71] needs three observed frames on either side for jerk.  This
    # contract must hold even when the configured seam carries no soft halo.
    assert features[:, 41:44].abs().sum() > 0
    assert features[:, 72:75].abs().sum() > 0
    assert torch.count_nonzero(features[:, :41]) == 0
    assert torch.count_nonzero(features[:, 75:]) == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_world_fk_features_include_root_motion_and_match_audit_derivatives(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x, _, seam, _ = sample(device)
    x[:, 50:56, 4] += 0.125
    x[:, 53:59, 6] -= 0.0625
    features = m._refiner_fk_dynamics_features(x, seam, 30.0)
    audited = m._observable_boundary_joints_torch(x)
    block = m.NUM_JOINTS * 3
    for order, scale in ((1, 1.0), (2, 10.0), (3, 1000.0)):
        expected = torch.diff(audited, n=order, dim=1) * 30.0**order / scale
        actual = features[:, 44:72, (order - 1)*block:order*block]
        torch.testing.assert_close(
            actual,
            expected[:, 44-order:72-order].flatten(2).to(x.dtype),
            rtol=1e-5,
            atol=1e-6,
        )
    stationary = x.clone()
    stationary[..., 4] = 0
    stationary[..., 6] = 0
    stationary_features = m._refiner_fk_dynamics_features(stationary, seam, 30.0)
    assert not torch.equal(features[..., 2*block:3*block],
                           stationary_features[..., 2*block:3*block])


def test_temporal_objective_waits_for_observable_endpoint_feasibility():
    before = torch.tensor([1.0, 1.0, 0.0])
    proposed = torch.tensor([1.0, 0.985, 0.0], requires_grad=True)
    gate = m._endpoint_feasibility_gate(proposed, before, 0.03)
    torch.testing.assert_close(gate, torch.tensor([0.0, 0.5, 1.0]))
    assert not gate.requires_grad


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


def test_film_uses_observable_anchor_difficulty_without_world_position_leakage():
    x, _, seam, _ = sample("cpu")
    base = m._refiner_film_condition_features(x, seam)
    shifted = x.clone()
    shifted[..., 4] += 12.0
    shifted[..., 6] -= 7.0
    torch.testing.assert_close(
        base,
        m._refiner_film_condition_features(shifted, seam),
        atol=1e-7,
        rtol=1e-6,
    )
    changed = x.clone()
    changed[:, 72, 4] += 0.5
    modified = m._refiner_film_condition_features(changed, seam)
    assert modified[:, 44:72, 0].mean() > base[:, 44:72, 0].mean()
    assert torch.count_nonzero(base[:, :40]) == 0


def test_film_safe_start_is_identity_modulation_and_zero_refiner_output():
    x, cond, seam, joint = sample("cpu")
    model = m.ProductManifoldTemporalRefiner(
        hidden=16,
        film_conditioning=True,
    )
    trace = {}
    output = model(x, cond, seam, joint, film_trace=trace)
    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(trace["gamma"], torch.ones_like(trace["gamma"]))
    assert torch.count_nonzero(trace["beta"]) == 0
    assert trace["condition"].shape == (
        x.shape[0], x.shape[1], m.REFINER_FILM_CONDITION_DIM
    )


def test_film_checkpoint_contract_is_opt_in_and_fail_closed():
    legacy = m.MotionGenerationConfig()
    assert "refiner_film_conditioning" not in m.motion_checkpoint_contract(
        legacy, "boundary_refiner"
    )
    cfg = m.MotionGenerationConfig(product_refiner_film_conditioning=True)
    contract = m.motion_checkpoint_contract(cfg, "boundary_refiner")
    assert contract["refiner_film_conditioning"] == {
        "enabled": True,
        "protocol": m.REFINER_FILM_PROTOCOL,
        "condition_dim": m.REFINER_FILM_CONDITION_DIM,
    }
    broken = dict(contract)
    broken.pop("refiner_film_conditioning")
    with pytest.raises(RuntimeError, match="refiner_film_conditioning"):
        m.assert_motion_checkpoint_contract(
            {"motion_contract": broken}, cfg, "film.pt", "boundary_refiner"
        )
    with pytest.raises(RuntimeError, match="refiner_film_conditioning"):
        m.assert_motion_checkpoint_contract(
            {"motion_contract": contract},
            legacy,
            "film.pt",
            "boundary_refiner",
        )


def test_observable_adapter_is_zero_initialized_and_ownership_local():
    x, cond, seam, joint = sample("cpu")
    model = m.ProductManifoldTemporalRefiner(
        hidden=16,
        observable_adapter=True,
        residual_taper_frames=3,
    )
    trace = {}
    output = model(x, cond, seam, joint, adapter_trace=trace)
    assert torch.count_nonzero(output) == 0
    assert trace["condition"].shape[-1] == m.REFINER_ADAPTER_CONDITION_DIM
    assert trace["tangent"].shape[-1] == 75
    owned = seam >= 0.5
    assert torch.count_nonzero(trace["tangent"][~owned.expand_as(
        trace["tangent"]
    )]) == 0
    assert torch.count_nonzero(output[..., :4]) == 0


def test_observable_adapter_exact_scope_mask_is_case_and_window_local():
    from training import refiner_observable_adapter_probe as probe

    ownership = torch.tensor([
        [[False], [True], [True], [False]],
        [[False], [True], [True], [False]],
    ])
    tangent = torch.ones((2, 4, 75))
    permitted = probe._owned_case_mask(ownership, tangent, case_index=1)
    scoped = tangent.masked_fill(~permitted, 0.0)

    assert torch.count_nonzero(scoped[0]) == 0
    assert torch.count_nonzero(scoped[1, [0, 3]]) == 0
    assert torch.count_nonzero(scoped[1, 1:3]) == 2 * 75


def test_exact_radius_adapter_normalization_is_finite_and_case_local():
    from training import refiner_observable_adapter_probe as probe

    ownership = torch.tensor([
        [[False], [True], [True], [False]],
        [[False], [True], [True], [False]],
    ])
    tangent = torch.zeros((2, 4, 75), requires_grad=True)
    with torch.no_grad():
        tangent[1, 1:3] = 0.25
    samples = [
        {
            "case_index": 0,
            "teacher_kind": "identity_control",
        },
        {
            "case_index": 1,
            "teacher_kind": "exact_projected_direction",
        },
    ]

    normalized, diagnostics = probe._safe_exact_radius_tangent(
        tangent,
        ownership,
        samples,
        1.0e-4,
    )
    permitted = probe._owned_case_mask(ownership, tangent, case_index=1)
    normalized[permitted].sum().backward()

    assert torch.count_nonzero(normalized[0]) == 0
    assert torch.count_nonzero(normalized[1, [0, 3]]) == 0
    assert torch.isfinite(tangent.grad).all()
    torch.testing.assert_close(
        normalized[permitted].square().mean().sqrt(),
        torch.tensor(1.0e-4),
        atol=1.0e-10,
        rtol=1.0e-6,
    )
    assert diagnostics["1"]["radius_equality_resolved"] is True


def test_zero_adapter_normalization_has_finite_gradient_and_unresolved_radius():
    from training import refiner_observable_adapter_probe as probe

    ownership = torch.tensor([[[False], [True], [True], [False]]])
    tangent = torch.zeros((1, 4, 75), requires_grad=True)
    samples = [{
        "case_index": 0,
        "teacher_kind": "exact_projected_direction",
    }]

    normalized, diagnostics = probe._safe_exact_radius_tangent(
        tangent,
        ownership,
        samples,
        1.0e-4,
    )
    normalized.sum().backward()

    assert torch.count_nonzero(normalized) == 0
    assert torch.isfinite(tangent.grad).all()
    assert diagnostics["0"]["normalization_floor_active"] is True
    assert diagnostics["0"]["radius_equality_resolved"] is False


def test_case_isolated_guard_restoration_uses_allowance_scaled_sum():
    from training import refiner_observable_adapter_probe as probe

    first = torch.tensor(0.75, requires_grad=True)
    second = torch.tensor(2.50, requires_grad=True)
    loss, details = probe._fixed_guard_restoration_terms(
        {"first": first, "second": second},
        {"first": 0.0, "second": 2.0},
        {"first": 0.0, "second": 0.0},
        {"first": 1.0, "second": 2.0},
        safety_fraction=0.25,
    )
    loss.backward()

    # (0.75 - 0.25) / 1 + (2.50 - 2.50) / 2 = 0.5.
    torch.testing.assert_close(loss, torch.tensor(0.5))
    torch.testing.assert_close(first.grad, torch.tensor(1.0))
    torch.testing.assert_close(second.grad, torch.tensor(0.0))
    assert details["first"]["absolute_allowance"] == 1.0
    assert details["first"]["final_absolute_limit"] == 1.0
    assert details["first"]["training_safety_limit"] == 0.25
    assert details["first"]["active"] is True


def test_guard_direction_weight_decays_continuously_to_floor():
    from training import refiner_observable_adapter_probe as probe

    safe = probe._continuous_direction_weight(
        torch.tensor(0.0), torch.tensor(0.0), floor=0.1, decay=1.0
    )
    pressured = probe._continuous_direction_weight(
        torch.tensor(3.0), torch.tensor(0.0), floor=0.1, decay=1.0
    )
    severe = probe._continuous_direction_weight(
        torch.tensor(30.0), torch.tensor(0.0), floor=0.1, decay=1.0
    )

    torch.testing.assert_close(safe, torch.tensor(1.0))
    assert 0.1 < float(pressured) < 1.0
    assert abs(float(severe) - 0.1) < 1.0e-6


def test_cross_group_balance_prevents_teacher_count_dominance():
    from training import refiner_observable_adapter_probe as probe

    loss, groups = probe._balanced_cross_group_mean({
        "cross_short": [torch.tensor(1.0)] * 6,
        "cross_long": [torch.tensor(3.0)],
    })

    torch.testing.assert_close(loss, torch.tensor(2.0))
    assert groups == {"cross_short": 1.0, "cross_long": 3.0}


def test_observable_adapter_contract_is_opt_in_and_label_free():
    legacy = m.MotionGenerationConfig()
    assert "refiner_observable_adapter" not in m.motion_checkpoint_contract(
        legacy, "boundary_refiner"
    )
    cfg = m.MotionGenerationConfig(product_refiner_observable_adapter=True)
    contract = m.motion_checkpoint_contract(cfg, "boundary_refiner")
    adapter = contract["refiner_observable_adapter"]
    assert adapter["protocol"] == m.REFINER_OBSERVABLE_ADAPTER_PROTOCOL
    assert adapter["condition_dim"] == m.REFINER_ADAPTER_CONDITION_DIM
    assert adapter["output_tangent_dim"] == 75
    assert adapter["role_label_consumed"] is False


def test_input_protocol_is_checked_not_just_stored():
    cfg = m.MotionGenerationConfig()
    contract = m.motion_checkpoint_contract(cfg, 'boundary_refiner')
    assert contract['refiner_input_protocol'] == m.REFINER_INPUT_PROTOCOL
    assert m.REFINER_INPUT_PROTOCOL.endswith(
        'world_fk_dynamics_condition_path_support_v5'
    )
    model = m.ProductManifoldTemporalRefiner(hidden=16)
    assert model.in_proj.in_channels == (
        m.EDGE_DIM + 32 + 1 + m.NUM_JOINTS
        + m.BOUNDARY_FEATURE_DIM + m.REFINER_FK_DYNAMICS_FEATURE_DIM
        + m.REFINER_CONDITION_PATH_FEATURE_DIM
    )
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
