"""Optimizer correctness, not a claim that the held-out repair task passes."""
import json
from unittest import mock

import pytest

from contracts.physical_quality import _allowed_after_stage, physical_metric_specs
from motion_geometry.boundary_observables import observable_gate
from tests.test_bridge_feasibility import bank
from training import bridge_feasibility as f
from training import motion_models as m

torch = m.torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_smooth_margin_keeps_target_deadband_and_continuous_bounded_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    baseline = torch.tensor([.01, .28], device=device, dtype=torch.float64, requires_grad=True)
    ratios = torch.tensor([.89, .9, .9000001, .95, 1., 1.2], device=device, dtype=torch.float64, requires_grad=True)
    loss, gap = m._smooth_observable_margin(ratios[:, None]*baseline[None], baseline[None], .1)
    torch.testing.assert_close(loss[:, 0], loss[:, 1], rtol=1e-9, atol=1e-14)
    assert loss[0].eq(0).all() and loss[1].max() < 1e-28
    expected_gap = (ratios.detach()-.9).clamp_min(0)
    torch.testing.assert_close(gap[:, 0], expected_gap, atol=1e-14, rtol=1e-9)
    derivative = torch.autograd.grad(loss[:, 0].sum(), ratios)[0]
    torch.testing.assert_close(derivative, (expected_gap/.1).clamp_max(1), atol=1e-12, rtol=1e-8)
    assert derivative[2] < 2e-6  # no unit-size gradient cliff at the target
    assert baseline.grad is None


def test_smoothing_does_not_relax_actual_acceptance():
    cfg = m.MotionGenerationConfig()
    before = dict(valid=True, endpoint_velocity_jump_mps=1., temporal_energy=1., seam_jerk_mps3=100.)
    for improvement, accepted in ((.029, False), (.031, True)):
        after = {**before, 'endpoint_velocity_jump_mps':1-improvement, 'temporal_energy':1-improvement}
        gate = observable_gate(before, after, cfg)
        assert gate['accepted'] is accepted


@pytest.mark.parametrize("above_limit", [False, True])
def test_support_surrogate_uses_stage_ratio_plus_margin_epsilon_and_ceiling(above_limit):
    cfg = m.MotionGenerationConfig(device='cpu')
    keys = ('foot_skate_mps_p95','foot_skate_mps_max','foot_support_drift_m_p95','foot_support_drift_m_max')
    specs = {s.key:s for s in physical_metric_specs(m.PhysicalQualityLimits.from_environment(),m.StageAcceptancePolicy.from_environment())}
    before, after, expected = {}, {}, []
    for key in keys:
        spec = specs[key]
        value = spec.absolute_limit * (1.2 if above_limit else .5)
        allowed = _allowed_after_stage(value,spec.absolute_limit,spec.stage_ratio,spec.stage_margin)
        epsilon = max(1e-8,abs(value)*1e-6,abs(spec.absolute_limit)*1e-9)
        before[key] = torch.tensor([value,value],dtype=torch.float64)
        after[key] = torch.tensor([allowed+epsilon*.5,allowed+epsilon+.001],dtype=torch.float64,requires_grad=True)
        gap = .001/max(allowed-value,spec.stage_margin,1e-4)
        expected.append(torch.tensor([0., .5*min(gap,1.)**2+gap-min(gap,1.)],dtype=torch.float64))
    joints = torch.zeros((2,8,24,3),dtype=torch.float64)
    with mock.patch.object(m,'_reference_support_statistics_torch',return_value=(before,after,None)):
        loss = m._clean_support_tolerance_loss_torch(joints,joints,joints[...,0,0],cfg,reduction='none',stage_relative=True)
    torch.testing.assert_close(loss,sum(expected)/len(keys),atol=1e-12,rtol=1e-10)
    loss.sum().backward()
    for value in after.values():
        assert value.grad[0] == 0 and value.grad[1] > 0


def test_above_limit_support_boundary_has_no_linear_gradient_cliff():
    cfg=m.MotionGenerationConfig(device='cpu')
    spec=next(s for s in physical_metric_specs(m.PhysicalQualityLimits.from_environment(),
        m.StageAcceptancePolicy.from_environment()) if s.key=='foot_skate_mps_max')
    baseline=torch.tensor([spec.absolute_limit*1.2],dtype=torch.float64)
    epsilon=max(1e-8,float(baseline[0])*1e-6,abs(spec.absolute_limit)*1e-9)
    candidate=(baseline+epsilon+1e-9).requires_grad_(True)
    before={spec.key:baseline}; after={spec.key:candidate}
    joints=torch.zeros((1,8,24,3),dtype=torch.float64)
    with mock.patch.object(m,'_reference_support_statistics_torch',return_value=(before,after,None)):
        loss=m._clean_support_tolerance_loss_torch(joints,joints,joints[...,0,0],cfg,
            reduction='none',stage_relative=True)
    loss.sum().backward()
    assert 0 < candidate.grad.item() < 1e-4


def test_clean_identity_does_not_inherit_the_repair_stage_support_budget():
    b,cfg = bank()
    spec = next(s for s in physical_metric_specs(m.PhysicalQualityLimits.from_environment(),m.StageAcceptancePolicy.from_environment())
                if s.key == 'foot_skate_mps_p95')
    value = spec.absolute_limit*.5
    clean_cap = min(spec.absolute_limit,max(value*spec.stage_ratio,value+spec.stage_margin))
    stage_cap = _allowed_after_stage(value,spec.absolute_limit,spec.stage_ratio,spec.stage_margin)
    assert clean_cap < stage_cap
    before = {spec.key:torch.tensor([value])}
    after = {spec.key:torch.tensor([(clean_cap+stage_cap)*.5])}
    joints = torch.zeros((1,8,24,3))
    with mock.patch.object(m,'_reference_support_statistics_torch',return_value=(before,after,None)):
        clean = m._clean_support_tolerance_loss_torch(joints,joints,joints[...,0,0],cfg)
        repair = m._clean_support_tolerance_loss_torch(joints,joints,joints[...,0,0],cfg,stage_relative=True)
    assert clean > 0 and repair == 0
    original = m._clean_support_tolerance_loss_torch
    with mock.patch.object(m,'_clean_support_tolerance_loss_torch',wraps=original) as support:
        m._observable_refiner_objective(b['bad'],b['bad'],b['seam'],cfg)
        assert support.call_args.kwargs['stage_relative'] is True


def objective(prediction, reference, seam, cfg, **kwargs):
    loss = (prediction[:,20:24,4]-reference[:,20:24,4]-.002).square().mean(1)
    return loss, {k:loss*0 for k in ('endpoint_continuity','temporal_supervision','support_excess','jerk_safety_excess','root_vertical_safety_excess')}


def test_independent_direct_case_gradient_does_not_depend_on_batch_size(tmp_path):
    b, cfg = bank()
    captured = []
    original_step = torch.optim.Adam.step
    def capture(optimizer, *args, **kwargs):
        captured.append(optimizer.param_groups[0]['params'][0].grad.detach().clone())
        return original_step(optimizer,*args,**kwargs)
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=objective), \
         mock.patch.object(torch.optim.Adam,'step',capture):
        f.direct_optimize(b,cfg,1,label='one',log_path=tmp_path/'one.jsonl')
        b2 = {k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in b.items()}
        f.direct_optimize(b2,cfg,1,label='two',log_path=tmp_path/'two.jsonl')
    torch.testing.assert_close(captured[0][0],captured[1][0],atol=1e-10,rtol=1e-6)
    torch.testing.assert_close(captured[1][0],captured[1][1],atol=1e-10,rtol=1e-6)


def test_uphill_adam_proposal_uses_current_descent_gradient(tmp_path):
    b,cfg = bank()
    def uphill(optimizer,*args,**kwargs):
        p = optimizer.param_groups[0]['params'][0]
        with torch.no_grad():
            p.add_(p.grad * .003)
        optimizer.state[p].update(exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=objective), \
         mock.patch.object(torch.optim.Adam,'step',uphill):
        pred, summary = f.direct_optimize(b,cfg,1,label='uphill',log_path=tmp_path/'log.jsonl')
    assert objective(pred,b['bad'],b['seam'],cfg)[0] < objective(b['bad'],b['bad'],b['seam'],cfg)[0]
    assert summary[0]['non_descent_adam_steps'] == 1
    assert summary[0]['gradient_fallback_updates'] == 1
    assert summary[0]['safety_accepted']


def test_zero_gradient_is_not_reported_as_a_successful_edit(tmp_path):
    b,cfg = bank()
    def zero(prediction,*args,**kwargs):
        loss = prediction.flatten(1).sum(1)*0
        return loss, {k:loss for k in ('endpoint_continuity','temporal_supervision','support_excess','jerk_safety_excess')}
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=zero):
        pred,summary = f.direct_optimize(b,cfg,2,label='zero',log_path=tmp_path/'log.jsonl')
    assert torch.equal(pred,b['bad'])
    assert summary[0]['safe_update_count'] == 0 and summary[0]['retained_no_edit']
    assert summary[0]['resolution_limited_trial_count'] > 0


def test_only_achieved_training_targets_stop_a_case_early(tmp_path):
    b,cfg = bank()
    def targeted(prediction, reference, seam, cfg, **kwargs):
        loss, terms = objective(prediction,reference,seam,cfg,**kwargs)
        changed = (prediction != reference).flatten(1).any(1)
        gap = (~changed).to(loss.dtype)*.1
        terms.update(endpoint_relative_gap=gap,temporal_relative_gap=gap,
                     jerk=loss*0,observable_trust_excess=loss*0)
        return loss,terms
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=targeted):
        pred,summary = f.direct_optimize(b,cfg,20,label='target',log_path=tmp_path/'log.jsonl')
    assert summary[0]['target_satisfied'] and summary[0]['attempted_optimizer_steps'] == 1
    assert summary[0]['safe_update_count'] == 1 and not summary[0]['retained_no_edit']

    def not_at_training_target(*args,**kwargs):
        loss,terms = targeted(*args,**kwargs)
        # A 5% deficit to the 10% training target is NOT a stopping criterion,
        # even though its 5% improvement would clear the 3% evaluation gate.
        terms.update(endpoint_relative_gap=loss*0+.05,temporal_relative_gap=loss*0+.05)
        return loss,terms
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=not_at_training_target):
        _,summary = f.direct_optimize(b,cfg,3,label='not-target',log_path=tmp_path/'more.jsonl')
    assert not summary[0]['target_satisfied'] and summary[0]['attempted_optimizer_steps'] == 3

    def at_targets_but_vertical_violation(*args,**kwargs):
        loss,terms=targeted(*args,**kwargs)
        terms['root_vertical_safety_excess']=loss*0+.01
        return loss,terms
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=at_targets_but_vertical_violation):
        _,summary=f.direct_optimize(b,cfg,3,label='root-unsafe',log_path=tmp_path/'root.jsonl')
    assert not summary[0]['target_satisfied'] and summary[0]['attempted_optimizer_steps']==3


def test_stalled_search_is_reported_and_does_not_count_no_edit_as_repair(tmp_path):
    b,cfg = bank()
    def constant(prediction,*args,**kwargs):
        loss = prediction.flatten(1).sum(1)*0+.1
        return loss, {k:loss*0 for k in ('endpoint_continuity','temporal_supervision','support_excess','jerk_safety_excess')}
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=constant):
        pred,summary = f.direct_optimize(b,cfg,200,label='stalled',log_path=tmp_path/'log.jsonl')
    case = summary[0]
    assert case['search_stalled'] and not case['target_satisfied']
    assert case['attempted_optimizer_steps'] == f.DIRECT_STALL_PATIENCE
    assert case['safe_update_count'] == 0 and case['retained_no_edit']
    gate = m._observable_boundary_audit(pred[0].cpu().numpy(),b['bad'][0].cpu().numpy(),b['seam'][0].cpu().numpy(),cfg)
    assert not gate['temporal_accepted']


def test_frozen_case_retains_last_actual_gradient_in_its_report(tmp_path):
    b,cfg = bank()
    b = {k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in b.items()}
    def partial(prediction,reference,seam,cfg,**kwargs):
        loss,terms = objective(prediction,reference,seam,cfg,**kwargs)
        # In full-batch gradient passes case 0 can finish before case 1.
        changed = (prediction != reference).flatten(1).any(1)
        gap = loss*0+.1
        if len(loss)==2 and changed[0]:
            gap[0] = 0
        terms.update(endpoint_relative_gap=gap,temporal_relative_gap=gap,
                     jerk=loss*0,observable_trust_excess=loss*0)
        return loss,terms
    with mock.patch.object(m,'_observable_refiner_objective',side_effect=partial):
        _,summary = f.direct_optimize(b,cfg,3,label='last-gradient',log_path=tmp_path/'log.jsonl')
    assert summary[0]['target_satisfied'] and summary[0]['attempted_optimizer_steps'] == 1
    assert summary[0]['last_pre_update_gradient_norm'] > 0


def test_previous_direct_optimizer_report_cannot_authorize_fitting(tmp_path):
    report = tmp_path/'old.json'
    report.write_text(json.dumps({'schema':'bridge_foundation_feasibility_v3','fingerprint':{},'published':False}))
    with pytest.raises(RuntimeError,match='protocol/config/code mismatch'):
        f.check_foundation_report(report,{},m.MotionGenerationConfig())
