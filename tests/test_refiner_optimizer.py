import copy
import inspect

import pytest
import torch

from training.refiner_optimizer import checked_refiner_step, REFINER_UPDATE_PROTOCOL


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_state_equal(x,y)
    else:
        assert a==b


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def test_clipped_adam_can_overshoot_but_checked_step_decreases(device):
    p=torch.nn.Parameter(torch.zeros(1,device=device))
    opt=torch.optim.AdamW([p],lr=1.,weight_decay=0.)
    def objective(): return ((10*p-1)**2).sum()
    loss=objective()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p],1.)
    report=checked_refiner_step(opt,loss,objective)
    assert report['first_trial_loss'] > 50*report['loss_before']
    assert report['adam_directional_derivative'] < 0
    assert report['optimizer_update_accepted']
    assert 0 < report['step_scale'] < 1
    assert float(objective().detach()) < float(loss.detach())
    assert report['loss_after'] == float(objective())
    assert report['scientific_acceptance'] is False


def test_full_rejection_restores_parameters_moments_step_and_scale(device):
    p=torch.nn.Parameter(torch.tensor([1.],device=device))
    opt=torch.optim.AdamW([p],lr=.1)
    p.square().sum().backward()
    opt.step()
    opt.param_groups[0]['refiner_trial_scale']=.25
    opt.zero_grad(set_to_none=True)
    loss=p.square().sum(); loss.backward()
    before=p.detach().clone(); state=copy.deepcopy(opt.state_dict())
    report=checked_refiner_step(opt,loss,lambda:loss.detach()+1,max_trials=2)
    assert not report['optimizer_update_accepted']
    assert report['trial_evaluations'] <= 4
    assert report['reason']=='bounded_search_no_descent'
    torch.testing.assert_close(p,before,atol=0,rtol=0)
    assert_state_equal(opt.state_dict(),state)


def test_exception_during_trial_also_rolls_back(device):
    p=torch.nn.Parameter(torch.tensor([1.],device=device))
    opt=torch.optim.AdamW([p],lr=.1)
    loss=p.square().sum(); loss.backward()
    before=p.detach().clone(); state=copy.deepcopy(opt.state_dict())
    def explode(): raise RuntimeError('trial failed')
    with pytest.raises(RuntimeError,match='trial failed'):
        checked_refiner_step(opt,loss,explode)
    torch.testing.assert_close(p,before,atol=0,rtol=0)
    assert_state_equal(opt.state_dict(),state)


def test_uphill_momentum_uses_current_gradient_not_ascent(device):
    p=torch.nn.Parameter(torch.tensor([1.],device=device))
    opt=torch.optim.AdamW([p],lr=.1,weight_decay=0.)
    (-p).sum().backward(); opt.step()  # stale momentum points away from zero
    opt.zero_grad(set_to_none=True)
    def objective(): return p.square().sum()*.01
    loss=objective(); loss.backward()
    report=checked_refiner_step(opt,loss,objective)
    assert report['adam_directional_derivative'] > 0
    assert report['direction']=='current_gradient'
    assert report['used_gradient_rescue'] and report['optimizer_update_accepted']
    assert not opt.state  # stale moments must not be resumed after rescue
    assert float(objective()) < float(loss)


def test_roundoff_sized_adam_gain_uses_meaningful_gradient_rescue(device):
    # Stale momentum has a large component along a highly curved coordinate.
    # Its dot product with the current gradient is negative, but shrinking it
    # indefinitely would accept a near-no-op instead of the useful x direction.
    p=torch.nn.Parameter(torch.zeros(2,device=device))
    opt=torch.optim.AdamW([p],lr=.01,weight_decay=0.)
    (-p[1]).backward(); opt.step()
    with torch.no_grad(): p.zero_()
    opt.param_groups[0]['refiner_trial_scale']=2.0**-11
    opt.zero_grad(set_to_none=True)
    def objective(): return 1+p[0]+.5*p[0].square()+1e6*p[1].square()
    loss=objective(); loss.backward()
    report=checked_refiner_step(opt,loss,objective)
    assert report['adam_directional_derivative'] < 0
    assert report['direction']=='current_gradient'
    assert report['loss_before']-report['loss_after'] > .009
    assert 0 < report['step_scale'] <= 1
    assert p[1]==0 and not opt.state


def test_out_of_range_persisted_scale_is_rejected_and_restored(device):
    p=torch.nn.Parameter(torch.ones(1,device=device))
    opt=torch.optim.AdamW([p],lr=.1)
    opt.param_groups[0]['refiner_trial_scale']=0
    loss=p.square().sum(); loss.backward()
    state=copy.deepcopy(opt.state_dict())
    with pytest.raises(ValueError,match='persisted Refiner trial scale'):
        checked_refiner_step(opt,loss,lambda:p.square().sum())
    assert p==1
    assert_state_equal(opt.state_dict(),state)


def test_curved_objective_can_take_meaningful_step_below_old_scale_floor(device):
    # The old 2^-11 floor rejects every trial here although the optimum is
    # representable and reduces loss by nearly 100%, not a numerical no-op.
    p=torch.nn.Parameter(torch.zeros(1,device=device,dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=1e-3,weight_decay=0.)
    def objective(): return (1e8*p-1).square().sum()
    loss=objective(); loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_([p],1.))
    report=checked_refiner_step(opt,loss,objective,gradient_unscale=max(1.,norm+1e-6))
    assert report['optimizer_update_accepted']
    assert 0 < report['step_scale'] < 2.0**-11
    assert report['loss_after'] < .01
    assert report['trial_evaluations'] <= 24
    first=report['trials'][0]
    assert first['directional_derivative'] == pytest.approx(-2e5,rel=1e-5)
    assert report['trials'][-1]['required_decrease'] <= report['loss_before']-report['loss_after']


def test_tiny_previous_scale_does_not_cap_next_changed_curvature(device):
    p=torch.nn.Parameter(torch.zeros(1,device=device,dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=1e-3,weight_decay=0.)
    def steep(): return (1e8*p-1).square().sum()
    loss=steep(); loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_([p],1.))
    first=checked_refiner_step(opt,loss,steep,gradient_unscale=max(1.,norm+1e-6))
    assert first['optimizer_update_accepted'] and first['step_scale'] < 2e-4
    opt.zero_grad(set_to_none=True)
    def shallow(): return (p-1).square().sum()
    loss=shallow(); loss.backward()
    second=checked_refiner_step(opt,loss,shallow)
    assert second['trials'][0]['scale']==1.
    assert second['optimizer_update_accepted']


def test_tiny_loss_change_is_not_progress_even_when_parameters_change(device):
    p=torch.nn.Parameter(torch.zeros(1,device=device,dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=1e-13,weight_decay=0.)
    def objective(): return (1+p).sum()
    loss=objective(); loss.backward()
    state=copy.deepcopy(opt.state_dict())
    report=checked_refiner_step(opt,loss,objective)
    assert not report['optimizer_update_accepted']
    assert report['insufficient_decrease_trials'] > 0
    assert report['trial_evaluations'] <= 24
    assert p==0
    assert_state_equal(opt.state_dict(),state)


def test_representable_one_in_fifty_million_decrease_is_progress(device):
    p=torch.nn.Parameter(torch.zeros(1,device=device,dtype=torch.float64))
    opt=torch.optim.SGD([p],lr=2e-8)
    def objective(): return (1+p).sum()
    loss=objective(); loss.backward()
    report=checked_refiner_step(opt,loss,objective)
    assert report['optimizer_update_accepted']
    assert report['loss_before']-report['loss_after'] == pytest.approx(2e-8)
    assert report['minimum_loss_decrease'] == pytest.approx(1e-8)


def test_resume_replays_scaled_update_and_optimizer_state(device):
    def create():
        parameter=torch.nn.Parameter(torch.zeros(1,device=device))
        optimizer=torch.optim.AdamW([parameter],lr=.3,weight_decay=.01)
        return parameter,optimizer
    def step(p,opt):
        opt.zero_grad(set_to_none=True)
        def objective(): return ((10*p-1)**2).sum()
        loss=objective(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p],1.)
        return checked_refiner_step(opt,loss,objective)
    p,opt=create(); step(p,opt)
    state=copy.deepcopy(opt.state_dict()); weights=p.detach().clone()
    expected=step(p,opt)
    q,resumed=create()
    with torch.no_grad(): q.copy_(weights)
    resumed.load_state_dict(state)
    actual=step(q,resumed)
    assert actual==expected
    torch.testing.assert_close(p,q,atol=0,rtol=0)
    assert_state_equal(opt.state_dict(),resumed.state_dict())


def test_invalid_loss_and_gradient_do_not_mutate(device):
    p=torch.nn.Parameter(torch.ones(1,device=device))
    opt=torch.optim.AdamW([p],lr=.1)
    loss=p.square().sum(); loss.backward()
    with pytest.raises(FloatingPointError):
        checked_refiner_step(opt,loss*float('nan'),lambda:loss)
    p.grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        checked_refiner_step(opt,loss,lambda:loss)
    assert float(p)==1 and not opt.state


def test_nonfinite_trials_never_replace_finite_state(device):
    p=torch.nn.Parameter(torch.ones(1,device=device))
    opt=torch.optim.AdamW([p],lr=.1)
    loss=p.square().sum(); loss.backward()
    report=checked_refiner_step(opt,loss,lambda:loss.detach()*float('nan'),max_trials=2)
    assert not report['optimizer_update_accepted']
    assert report['nonfinite_trials']==report['trial_evaluations']
    assert report['nonfinite_trials']>0
    assert float(p)==1 and not opt.state


def test_zero_gradient_does_not_apply_weight_decay(device):
    p=torch.nn.Parameter(torch.ones(1,device=device))
    opt=torch.optim.AdamW([p],lr=.1,weight_decay=1.)
    loss=(p*0).sum(); loss.backward()
    report=checked_refiner_step(opt,loss,lambda:loss)
    assert report['reason']=='zero_gradient'
    assert report['trial_evaluations']==0 and float(p)==1 and not opt.state


def test_diagnostic_and_formal_training_share_checked_update():
    from training import motion_models as m
    from training import refiner_bridge_diagnostics as d
    for fn in (m.train_refiner,d.run):
        source=inspect.getsource(fn)
        assert 'checked_refiner_step(' in source
        assert '_refiner_guarded_total_batch_loss(' in source
    assert REFINER_UPDATE_PROTOCOL == m.REFINER_UPDATE_PROTOCOL


@pytest.mark.parametrize('protocol_name',['REFINER_UPDATE_PROTOCOL',
    'REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL','REFINER_REPAIR_SAFETY_PROTOCOL','REFINER_INPUT_PROTOCOL'])
def test_update_protocol_invalidates_refiner_resume_hash_not_diffusion(monkeypatch,protocol_name):
    from training import motion_models as m
    cfg=m.MotionGenerationConfig()
    a=m._training_config_sha256(cfg,stage='refiner')
    b=m._training_config_sha256(cfg,stage='diffusion')
    monkeypatch.setattr(m,protocol_name,'old_protocol')
    assert m._training_config_sha256(cfg,stage='refiner') != a
    assert m._training_config_sha256(cfg,stage='diffusion') == b


def test_counts_cover_unprinted_updates_without_implying_scientific_acceptance():
    from training.refiner_optimizer import record_update, validate_update_summary
    summary={}
    row={'optimizer_update_accepted':True,'used_gradient_rescue':False,
         'trial_evaluations':3,'nonfinite_trials':0,'loss_before':1.,'loss_after':.5}
    record_update(summary,row)
    record_update(summary,{**row,'optimizer_update_accepted':False,'loss_after':1.})
    assert summary['attempted_steps']==2 and summary['accepted_steps']==1
    assert summary['retained_steps']==1 and summary['trial_evaluations']==6
    assert summary['accepted_non_descent_steps']==0
    validate_update_summary(summary, 2)
    with pytest.raises(RuntimeError, match='inconsistent'):
        validate_update_summary(summary, 3)
    with pytest.raises(RuntimeError, match='non-descent'):
        validate_update_summary({**summary,'accepted_non_descent_steps':1}, 2)


def test_real_refiner_objective_replays_same_batch_and_keeps_finite_gradients(device):
    from training import motion_models as m
    from tests.test_bridge_feasibility import bank
    torch.manual_seed(42)
    batch,cfg=bank(device=device)
    original={key:value.detach().clone() for key,value in batch.items()}
    model=m.ProductManifoldTemporalRefiner(hidden=16).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
    def objective(): return m._refiner_total_batch_loss(model,batch,cfg)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss=objective()
        loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        report=checked_refiner_step(opt,loss,objective,gradient_unscale=max(1.,norm+1e-6))
        with torch.no_grad(): after=float(objective())
        assert after <= float(loss)
        assert report['loss_after'] == after
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for key in batch:
        torch.testing.assert_close(batch[key],original[key],atol=0,rtol=0)


def test_report_without_complete_update_accounting_cannot_authorize_pilot(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    from training import refiner_bridge_diagnostics as d
    path=tmp_path/'incomplete.json'
    path.write_text(json.dumps({'schema':d.SCHEMA,'fingerprint':{},'completed':True,
        'published':False,'target_steps':400,'completed_steps':400,'windows':[{}]*8}),encoding='utf8')
    monkeypatch.setattr(d,'fingerprint',lambda *args:{})
    with pytest.raises(RuntimeError,match='optimizer update protocol'):
        d.run(Namespace(config='configs/motion_model.json',check_report=str(path),windows=8))


def test_group_guard_rejects_scalar_descent_that_regresses_subgroup():
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    opt=torch.optim.SGD([p],lr=.1)
    def objective():
        value=(p-1).square().sum()
        return value, {'single_short': p.square().sum()}
    loss,_=objective(); loss.backward()
    before=p.detach().clone(); state=copy.deepcopy(opt.state_dict())
    report=checked_refiner_step(
        opt,loss,objective,
        group_guard_before={'single_short':0.0},
        group_guard_relative_tolerance=0.0,
        group_guard_absolute_tolerance=0.0,
        max_trials=3,
    )
    assert not report['optimizer_update_accepted']
    assert report['group_guard_rejected_trials']>0
    torch.testing.assert_close(p,before,atol=0,rtol=0)
    assert_state_equal(opt.state_dict(),state)


def test_group_guard_accepts_non_regressing_subgroup():
    p=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    opt=torch.optim.SGD([p],lr=.1)
    def objective():
        value=(p-1).square().sum()
        return value, {'single_short': -p.sum()}
    loss,_=objective(); loss.backward()
    report=checked_refiner_step(
        opt,loss,objective,
        group_guard_before={'single_short':0.0},
        group_guard_relative_tolerance=0.0,
        group_guard_absolute_tolerance=0.0,
    )
    assert report['optimizer_update_accepted']
    assert report['group_guard_after']['single_short']<=0.0
