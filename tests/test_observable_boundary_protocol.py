import inspect
from unittest import mock

import numpy as np
import pytest
import torch

from motion_geometry.boundary_observables import (
    BOUNDARY_FEATURE_DIM, BOUNDARY_PROTOCOL, boundary_features_torch,
    boundary_metrics_torch, observable_gate,
)
from training import motion_models as m


def motion(frames=64, device="cpu"):
    x = torch.zeros((2, frames, 151), device=device)
    x[..., 7:] = torch.as_tensor(np.tile(m.identity6d_np(), 24), device=device)
    x[..., 5] = .95
    x[..., 4] = torch.linspace(0, .4, frames, device=device)
    return x


def mask(x):
    seam = torch.zeros(x.shape[:2] + (1,), device=x.device)
    seam[:, 12:40] = .35
    seam[:, 16:36] = 1
    return seam


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_observables_are_finite_differentiable_and_translation_invariant(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = motion(device=device).requires_grad_()
    seam = mask(x)
    result = boundary_features_torch(x, seam)
    assert result.shape == (2, 64, BOUNDARY_FEATURE_DIM)
    shifted = x.detach().clone()
    shifted[..., 4] += 10
    torch.testing.assert_close(result, boundary_features_torch(shifted, seam), atol=3e-6, rtol=1e-4)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_context_features_use_external_anchors_not_hidden_interior():
    x = motion()
    seam = mask(x)
    first = boundary_features_torch(x, seam)
    changed = x.clone()
    changed[:, 20, 4] += .5
    second = boundary_features_torch(changed, seam)
    # Boundary velocities, phase and other interior frames cannot change.
    torch.testing.assert_close(first[..., -150:], second[..., -150:])
    torch.testing.assert_close(first[:, 21], second[:, 21])
    assert not torch.equal(first[:, 20], second[:, 20])


def test_multiple_seams_and_empty_or_truncated_support():
    x = motion()
    seam = mask(x)
    seam[:, 48:56] = 1
    assert torch.isfinite(boundary_features_torch(x, seam)).all()
    empty = torch.zeros_like(seam)
    assert torch.count_nonzero(boundary_features_torch(x, empty)) == 0
    truncated = torch.ones_like(seam)
    assert torch.count_nonzero(boundary_features_torch(x, truncated)) == 0
    metrics = boundary_metrics_torch(m.fk_24_torch(x), truncated, 30)
    assert not metrics["valid"].any()


def test_default_degradation_is_exact_complete_inference_bridge():
    cfg = m.MotionGenerationConfig(device="cpu")
    clean = motion()[0].numpy()
    clean[:, 4] += .05 * np.sin(np.arange(len(clean)) * .2)
    recipe = {"a": 20, "b": 38}  # no hidden tangent array needed
    damaged, seam = m.degrade_for_refiner(clean, cfg=cfg, recipe=recipe, finalize_contract=False)
    expected = m.reference_motion_inbetween_np(clean[16:20], clean[38:42], 18, cfg, finalize_contract=False)
    np.testing.assert_array_equal(damaged[20:38], expected)
    np.testing.assert_array_equal(damaged[:20], clean[:20])
    np.testing.assert_array_equal(damaged[38:], clean[38:])
    other = clean.copy()
    other[20:38, 4] += 2
    replay, _ = m.degrade_for_refiner(other, cfg=cfg, recipe=recipe, finalize_contract=False)
    np.testing.assert_array_equal(damaged, replay)
    assert np.all(seam[20:38] == 1)


def test_actual_boundary_metrics_not_derivative_distance_to_clean():
    x = motion()
    seam = mask(x)
    bad = x.clone()
    bad[:, 16:36, 4] += .05
    actual = boundary_metrics_torch(m.fk_24_torch(x), seam, 30)
    damaged = boundary_metrics_torch(m.fk_24_torch(bad), seam, 30)
    assert torch.all(actual["endpoint_velocity_jump_mps"] < damaged["endpoint_velocity_jump_mps"])
    cfg = m.MotionGenerationConfig()
    def values(row, i=0):
        return {k: bool(v[i]) if k == "valid" else float(v[i]) for k,v in row.items()}
    unchanged = observable_gate(values(damaged), values(damaged), cfg)
    assert not unchanged["accepted"]
    repaired = observable_gate(values(damaged), values(actual), cfg)
    assert repaired["accepted"]
    assert repaired["hidden_clean_used"] is False


def test_repair_objective_has_no_hidden_clean_argument_and_finite_gradients():
    assert "clean" not in inspect.signature(m._observable_refiner_objective).parameters
    x = motion()
    seam = mask(x)
    prediction = x.clone().requires_grad_()
    loss, terms = m._observable_refiner_objective(prediction, x, seam, m.MotionGenerationConfig())
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert terms["observable_trust_excess"] == 0


def test_cross_event_case_has_no_clean_interior_and_time_local_conditions():
    cfg = m.MotionGenerationConfig()
    x = motion().numpy()
    out, seam, cond = m.make_cross_event_boundary_np(x[0], x[1], np.zeros(32), np.ones(32), cfg)
    assert out.shape == x[0].shape and cond.shape == (64,32)
    core = np.flatnonzero(seam[:, 0] == 1)
    assert np.all(cond[:core[0]] == 0) and np.all(cond[core[-1]+1:] == 1)
    assert np.all((cond[core] > 0) & (cond[core] < 1))


def test_shared_refiner_decode_strength_matches_inference_halo():
    x = motion()
    seam = mask(x)
    joint = torch.ones((2,64,24))
    root = torch.ones((2,64,1))
    a,b,c = m._refiner_decode_masks(joint,root,root,seam,m.MotionGenerationConfig())
    torch.testing.assert_close(b, .02 + .98 * seam)
    torch.testing.assert_close(a[...,0], b[...,0])


def test_old_acceptance_report_cannot_publish_new_models():
    cfg = m.MotionGenerationConfig()
    decision = m._checkpoint_validation_decision({"physical_quality": {"num_windows":16}}, cfg, stage="refiner")
    assert not decision["publish_allowed"]
    assert "missing_or_mismatched_observable_protocol" in decision["reasons"]


def test_failed_observable_candidate_does_not_overwrite_formal_checkpoint(tmp_path):
    formal = tmp_path / "boundary_refiner.pt"
    formal.write_bytes(b"previous-accepted-model")
    decision = m._checkpoint_validation_decision({"physical_quality":{"num_windows":2}},m.MotionGenerationConfig(),stage="refiner")
    path,published = m._save_checkpoint_after_validation({"version":m.REFINER_MODEL_VERSION},formal,decision)
    assert not published and path.name == "boundary_refiner.rejected_validation.pt"
    assert formal.read_bytes() == b"previous-accepted-model"


def test_diffusion_sampling_does_not_use_hidden_target_or_encode_it():
    x = motion()
    seam = mask(x)
    cfg = m.MotionGenerationConfig(device="cpu")
    model = m.TangentDiffusionDenoiser(hidden=16)
    with mock.patch.object(m, "_encode_reference_tangent_state", side_effect=AssertionError("target leakage")):
        with torch.no_grad():
            output = m._sample_diffusion_boundary(model, x, torch.zeros((2,32)), seam,
                seam.expand(-1,-1,24), seam, seam, cfg, 2, 0)
    assert output.shape == x.shape and torch.isfinite(output).all()


def test_repair_loss_is_independent_of_hidden_clean_branch():
    cfg = m.MotionGenerationConfig(device="cpu")
    x = motion()
    seam = mask(x)
    batch = m._prepare_refiner_batch(x.numpy(),x.numpy(),seam.numpy(),np.zeros((2,32),np.float32),cfg,torch.device("cpu"))
    model = m.ProductManifoldTemporalRefiner(hidden=16)
    with torch.no_grad():
        model.out.weight.normal_(0,.0001)
    first = m._refiner_batch_objectives(model,batch,cfg)[0]
    changed = {**batch,"clean":batch["clean"].clone()}
    changed["clean"][:,20:30,4] += .03
    second = m._refiner_batch_objectives(model,changed,cfg)[0]
    torch.testing.assert_close(first,second,atol=0,rtol=0)


def test_clean_reference_metrics_are_not_new_publication_criteria():
    cfg = m.MotionGenerationConfig()
    observable = {"schema":BOUNDARY_PROTOCOL,"num_windows":2,
                  "endpoint":{"pass_rate":1},"temporal":{"pass_rate":1},
                  "physical_non_regression":{"pass_rate":1},
                  "endpoint_informative":2,"temporal_informative":2,
                  "reference_fk_p95_m":.01,"reference_fk_max_m":.02,"reference_product_log_l1":.001}
    metrics = {"reconstruction_product_log_l1":999.,"physical_quality":{
        "num_windows":2,"stage_repair":{"pass_rate":0},"temporal_repair":{"pass_rate":0},
        "fk_position_error_m_p95":999.,"fk_position_error_m_max":999.,
        "clean_input_identity":{"pass_rate":1},"observable_boundary":observable},"cross_event":dict(observable)}
    assert m._checkpoint_validation_decision(metrics,cfg,stage="refiner")["scientific_acceptance"]
    observable["reference_fk_p95_m"] = .9
    bad = m._checkpoint_validation_decision(metrics,cfg,stage="refiner")
    assert not bad["publish_allowed"] and "reference_fk_p95_m_too_high" in bad["reasons"]


def test_bridge_diagnostic_uses_new_positions_without_tangent_targets():
    from training import refiner_bridge_diagnostics as d
    x = motion(frames=120).numpy()
    with mock.patch.object(m,"_refiner_tangent_noise_np",side_effect=AssertionError("historical corruption reached")):
        banks,recipes = d.build_banks(x,np.zeros((2,32),np.float32),["train_a","train_b"],m.MotionGenerationConfig(),torch.device("cpu"))
    assert set(banks) == {(split,role) for split in ("seen","new_position") for role in ("single_recording","cross_event")}
    for role in ("single_recording","cross_event"):
        seen = {(r["window"],r["a"],r["b"]) for r in recipes[f"seen/{role}"]}
        new = {(r["window"],r["a"],r["b"]) for r in recipes[f"new_position/{role}"]}
        assert not seen & new
    assert all(not r["hidden_clean_target"] for r in recipes["seen/cross_event"])


def test_historical_noise_entrypoint_cannot_run_under_new_protocol():
    from training import refiner_noise_refresh_diagnostics as legacy
    with pytest.raises(RuntimeError,match="retired"):
        legacy.main([])


def test_smoke_report_cannot_authorize_full_training(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from training import refiner_bridge_diagnostics as d
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"schema":d.SCHEMA,"fingerprint":{},"completed":True,"published":False,
                               "target_steps":1,"completed_steps":1,"windows":[{}] * 8}),encoding="utf8")
    monkeypatch.setattr(d,"fingerprint",lambda *args:{})
    args = Namespace(config="configs/motion_model.json",check_report=str(path),windows=8)
    with pytest.raises(RuntimeError,match="smoke runs cannot authorize"):
        d.run(args)


def test_diffusion_boundary_features_precomputed_once_per_reverse_process():
    x = motion()
    seam = mask(x)
    model = m.TangentDiffusionDenoiser(hidden=16)
    with mock.patch.object(m,"boundary_features_torch",wraps=m.boundary_features_torch) as features:
        with torch.no_grad():
            m._sample_diffusion_boundary(model,x,torch.zeros((2,32)),seam,seam.expand(-1,-1,24),seam,seam,
                m.MotionGenerationConfig(device="cpu"),4,0)
    assert features.call_count == 1
