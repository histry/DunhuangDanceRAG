"""Vertical speed/range safety is not implied by average FK jerk improvement."""
from unittest import mock

import numpy as np
import pytest
import torch

from contracts.physical_quality import physical_metric_specs, _allowed_after_stage
from training import motion_models as m
from tests.test_duration_inbetween import motion
from tests.test_bridge_feasibility import bank


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def test_root_statistics_match_independent_audit(device):
    cfg=m.MotionGenerationConfig(device=device)
    x=motion(120)
    x[:,5]=1+.1*np.sin(np.arange(120)*.13)
    expected=m.audit_motion_np(x,cfg)
    actual=m._root_vertical_statistics_torch(torch.as_tensor(x[None],device=device),cfg)
    for key,value in actual.items():
        assert float(value[0])==pytest.approx(expected[key],abs=1e-7,rel=1e-7)


def test_safe_vertical_changes_have_zero_loss_and_no_gradient(device):
    cfg=m.MotionGenerationConfig(device=device)
    x=torch.as_tensor(motion(60)[None],device=device)
    x[:, :,5]+=.01*torch.sin(torch.arange(60,device=device)*.1)
    ref=x.clone().requires_grad_(True)
    pred=x.clone(); pred[:,:,5]=1+(pred[:,:,5]-1)*1.001
    pred.requires_grad_(True)
    loss,_=m._repair_root_vertical_safety_loss_torch(pred,ref,cfg)
    assert loss.item()==0
    loss.sum().backward()
    assert ref.grad is None
    assert torch.count_nonzero(pred.grad)==0


def test_over_limit_input_is_preserved_not_flattened_but_cannot_get_worse(device):
    cfg=m.MotionGenerationConfig(device=device)
    x=torch.as_tensor(motion(60)[None],device=device)
    x[:,:,5]+=2*torch.sin(torch.arange(60,device=device)*.4)
    loss,_=m._repair_root_vertical_safety_loss_torch(x,x,cfg)
    assert loss.item()==0
    ref=x.clone().requires_grad_(True)
    pred=(x*1.01).detach().requires_grad_(True)
    loss,_=m._repair_root_vertical_safety_loss_torch(pred,ref,cfg)
    assert loss.item()>0
    loss.sum().backward()
    assert ref.grad is None and torch.isfinite(pred.grad).all()
    assert pred.grad[:,:,5].abs().sum()>0


@pytest.mark.parametrize('fraction',[0.,.5,.999,1.2])
def test_root_budgets_match_all_stage_registry_ceilings(fraction):
    cfg=m.MotionGenerationConfig(device='cpu')
    before,after,expected={},{},{}
    for spec in physical_metric_specs(m.PhysicalQualityLimits.from_environment(),m.StageAcceptancePolicy.from_environment()):
        if spec.layer!='root_vertical': continue
        value=fraction*spec.absolute_limit
        allowed=_allowed_after_stage(value,spec.absolute_limit,spec.stage_ratio,spec.stage_margin)
        eps=max(1e-8,abs(value)*1e-6,abs(spec.absolute_limit)*1e-9)
        before[spec.key]=torch.tensor([value,value],dtype=torch.float64)
        after[spec.key]=torch.tensor([allowed,allowed+eps+.01],dtype=torch.float64)
        gap=.01/max(1e-4,spec.stage_margin,allowed-value)
        expected[f'repair_{spec.key}_excess']=torch.tensor([0.,.5*min(gap,1.)**2+gap-min(gap,1.)],dtype=torch.float64)
    x=torch.zeros((2,8,151))
    with mock.patch.object(m,'_root_vertical_statistics_torch',side_effect=[before,after]):
        loss,terms=m._repair_root_vertical_safety_loss_torch(x,x,cfg)
    for key in terms: torch.testing.assert_close(terms[key],expected[key],atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(loss,sum(expected.values()),atol=1e-10,rtol=1e-9)


def test_repair_objective_includes_vertical_guard():
    b,cfg=bank()
    x=b['bad']
    initial,_=m._observable_refiner_objective(x,x,b['seam'],cfg,reduction='none')
    with mock.patch.object(m,'_repair_root_vertical_safety_loss_torch',return_value=(torch.tensor([2.]),{})):
        actual,terms=m._observable_refiner_objective(x,x,b['seam'],cfg,reduction='none')
    torch.testing.assert_close(actual-initial,torch.tensor([2.],dtype=actual.dtype))
    assert terms['root_vertical_safety_excess'].item()==2.


def test_uploaded_speed_regressions_are_penalized_without_changing_thresholds():
    # Metrics from 055d85e/19, not a claim to replay unavailable server motions.
    baseline=torch.tensor([1.4640763759613036,1.465709924697876],dtype=torch.float64)
    worse=torch.tensor([1.4666818141937257,1.468364679813385],dtype=torch.float64)
    before={'root_vertical_speed_mps_p95':baseline,'root_vertical_speed_mps_max':baseline*2,
            'root_y_robust_range_m':baseline*.4}
    after={**before,'root_vertical_speed_mps_p95':worse}
    x=torch.zeros((2,8,151))
    with mock.patch.object(m,'_root_vertical_statistics_torch',side_effect=[before,after]), \
         mock.patch.object(m.PhysicalQualityLimits,'from_environment',return_value=m.PhysicalQualityLimits()):
        loss,terms=m._repair_root_vertical_safety_loss_torch(x,x,m.MotionGenerationConfig())
    gap=(worse-baseline-baseline*1e-6)/m.StageAcceptancePolicy.from_environment().root_vertical_speed_p95_margin_mps
    expected=.5*gap.square()
    torch.testing.assert_close(loss,expected)
    assert torch.all(terms['repair_root_vertical_speed_mps_p95_excess']>0)


def test_over_limit_reference_does_not_create_a_linear_gradient_cliff():
    baseline=torch.full((2,),1.5,dtype=torch.float64)
    before={'root_vertical_speed_mps_p95':baseline,'root_vertical_speed_mps_max':baseline*2,
            'root_y_robust_range_m':baseline*.4}
    eps=baseline*1e-6
    candidate=(baseline+eps+torch.tensor([0.,1e-9],dtype=torch.float64)).requires_grad_(True)
    after={**before,'root_vertical_speed_mps_p95':candidate}
    x=torch.zeros((2,8,151))
    with mock.patch.object(m,'_root_vertical_statistics_torch',side_effect=[before,after]), \
         mock.patch.object(m.PhysicalQualityLimits,'from_environment',return_value=m.PhysicalQualityLimits()):
        loss,_=m._repair_root_vertical_safety_loss_torch(x,x,m.MotionGenerationConfig())
    loss.sum().backward()
    assert candidate.grad[0].abs()<1e-9
    assert 0 < candidate.grad[1] < 1e-4
