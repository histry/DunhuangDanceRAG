import inspect

import pytest
import torch

from training import refiner_bridge_diagnostics as d


def test_fixed_fit_uses_all_seen_cases_and_never_reads_probe():
    class SeenOnly(dict):
        def __getitem__(self, key):
            assert key[0] == 'seen', 'held-out position leaked into fitting'
            return super().__getitem__(key)
    def role(offset):
        x=torch.arange(offset,offset+16)[:,None,None].float()
        return {'clean':x,'bad':x+100,'cond':x+200,'clean_cond':x+300}
    banks=SeenOnly({('seen','single_recording'):role(0),('seen','cross_event'):role(16)})
    batch=d.fixed_fit_bank(banks)
    torch.testing.assert_close(batch['clean'].flatten(),torch.arange(32).float())
    torch.testing.assert_close(batch['bad'].flatten(),torch.arange(100,132).float())
    assert torch.bincount(batch['group']).tolist()==[8,8,8,8]
    assert len(batch['clean_cond'])==32


def test_fixed_bank_rejects_missing_or_unpaired_role_cases():
    a={'clean':torch.zeros(16,1,1)}
    for count in (15,14):
        with pytest.raises(ValueError,match='paired'):
            d.fixed_fit_bank({('seen','single_recording'):a,
                              ('seen','cross_event'):{'clean':torch.zeros(count,1,1)}})


def test_fit_contract_counts_examples_not_just_iterations():
    contract=d.fit_bank_contract(8)
    assert contract['cases_per_update']==32
    assert contract['cases_per_role_width']==8
    assert contract['gradient_scope']==contract['line_search_scope']=='complete_seen_bank'
    assert contract['probe_used_for_updates'] is False
    source=inspect.getsource(d.run)
    assert 'balanced_indices(' not in source
    assert 'batch = train' in source
    assert 'fit_bank_contract(args.windows)' in source
    assert d.PROBE_SCOPE == 'unfitted_local_motion_context_within_train_windows'
    assert '"probe_scope":PROBE_SCOPE' in source


def test_fixed_bank_stall_is_not_counted_as_400_steps_or_pilot_acceptance():
    for reason in ('bounded_search_no_descent','zero_gradient'):
        assert d.fixed_bank_stalled({'optimizer_update_accepted':False,'reason':reason})
    assert not d.fixed_bank_stalled({'optimizer_update_accepted':True,
                                     'reason':'same_batch_loss_decreased'})
    from training import motion_models as m
    assert 'fixed_bank_stalled' not in inspect.getsource(m.train_refiner)


def test_stalled_report_cannot_authorize_pilot(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    report=tmp_path/'stalled.json'
    report.write_text(json.dumps({'schema':d.SCHEMA,'fingerprint':{},'completed':True,
        'published':False,'stopped_early':True,'completed_steps':150,'target_steps':400}))
    monkeypatch.setattr(d,'fingerprint',lambda *args:{})
    with pytest.raises(RuntimeError,match='optimization stalled'):
        d.run(Namespace(config='configs/motion_model.json',check_report=str(report)))


def test_complete_step_count_without_complete_fit_bank_cannot_authorize_pilot(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL
    report=tmp_path/'subset.json'
    report.write_text(json.dumps({'schema':d.SCHEMA,'fingerprint':{},'completed':True,
        'published':False,'target_steps':400,'completed_steps':400,'windows':[{}]*8,
        'fit_bank':{**d.fit_bank_contract(8),'cases_per_update':8},
        'optimizer_updates':{'protocol':REFINER_UPDATE_PROTOCOL,'attempted_steps':400,
            'accepted_steps':400,'retained_steps':0,'trial_evaluations':400,
            'accepted_non_descent_steps':0}}))
    monkeypatch.setattr(d,'fingerprint',lambda *args:{})
    with pytest.raises(RuntimeError,match='complete predefined TRAIN fit bank'):
        d.run(Namespace(config='configs/motion_model.json',check_report=str(report),windows=8))


def test_portable_bank_and_optimizer_state_are_diagnostic_only(tmp_path):
    from training import motion_models as m
    model=torch.nn.Linear(1,1)
    optimizer=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,1)).sum().backward(); optimizer.step()
    batch={'clean':torch.arange(32.).reshape(32,1,1)}
    report={'fingerprint':{'test':'exact'},'windows':[], 'fit_bank':d.fit_bank_contract(8)}
    report['fit_bank_artifact']=d.save_fit_bank(tmp_path,batch,report,m.MotionGenerationConfig())
    d.save_diagnostic_state(tmp_path,model,optimizer,report,19)
    bank=m._trusted_torch_load(tmp_path/'fit_bank.pt',map_location='cpu')
    state=m._trusted_torch_load(tmp_path/'diagnostic_state.pt',map_location='cpu')
    assert bank['train_only'] and not bank['formal_checkpoint'] and not state['publish_allowed']
    assert set(bank['batch'])=={'clean'}
    assert bank['batch']['clean'].device.type=='cpu'
    assert state['completed_steps']==19
    assert report['fit_bank_artifact']['sha256']==d.common.file_sha256(tmp_path/'fit_bank.pt')
    assert state['optimizer_state_dict']['state']
    assert 'training_resume' not in state and 'version' not in state


def test_unlogged_stall_records_gradients_exact_state_and_return_code(tmp_path,monkeypatch):
    import json
    import numpy as np
    from argparse import Namespace
    from training import motion_models as m
    from training import bridge_feasibility as f
    from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL
    cfg=m.MotionGenerationConfig(device='cpu',window_len=8)
    monkeypatch.setattr(m.MotionGenerationConfig,'from_json',lambda path:cfg)
    monkeypatch.setattr(m.MotionGenerationConfig,'apply_env',lambda self:self)
    db={'paths':np.array(['ignored']*8),'source_uids':np.array([str(i) for i in range(8)]),
        'source_formats':np.array(['chang_e_official_smpl']*8)}
    monkeypatch.setattr(m,'load_db',lambda path:db)
    monkeypatch.setattr(m,'_training_db_contract',lambda *a:{})
    monkeypatch.setattr(m,'_validate_source_disjoint',lambda *a:{'verified':True})
    monkeypatch.setattr(m,'load_motion_window',lambda *a,**k:np.zeros((8,151)))
    monkeypatch.setattr(m,'_descriptor_values_in_training_coordinates',lambda *a:np.zeros((8,32)))
    banks={('seen',role):{'clean':torch.zeros(16,1,1)} for role in ('single_recording','cross_event')}
    monkeypatch.setattr(d,'build_banks',lambda *a:(banks,{}))
    monkeypatch.setattr(d,'fingerprint',lambda *a:{})
    monkeypatch.setattr(d.common,'file_sha256',lambda path:'test-digest')
    monkeypatch.setattr(f,'check_foundation_report',lambda *a:None)
    monkeypatch.setattr(f,'group_decisions',lambda *a:{'group':{'passed':False}})
    monkeypatch.setattr(m,'ProductManifoldTemporalRefiner',lambda:torch.nn.Linear(1,1))
    def objective(model,batch,cfg):
        r=sum(p.square().sum() for p in model.parameters())
        return r,r*0,{'endpoint':r},{}
    monkeypatch.setattr(m,'_refiner_batch_objectives',objective)
    monkeypatch.setattr(m,'_refiner_gradient_diagnostics',lambda *a:{'recorded':True})
    monkeypatch.setattr(m,'_refiner_component_gradients',lambda *a:{'recorded':True})
    calls=[]
    def step(*a,**k):
        calls.append(1)
        accepted=len(calls)==1
        return {'protocol':REFINER_UPDATE_PROTOCOL,'optimizer_update_accepted':accepted,
                'reason':'same_batch_loss_decreased' if accepted else 'bounded_search_no_descent',
                'used_gradient_rescue':not accepted,'trial_evaluations':1,'nonfinite_trials':0,
                'loss_before':1.,'loss_after':.9 if accepted else 1.}
    monkeypatch.setattr(m,'checked_refiner_step',step)
    monkeypatch.setattr(d,'evaluate',lambda *a:{})
    monkeypatch.setattr(d,'failure_breakdown',lambda *a:{})
    monkeypatch.setattr(m,'_checkpoint_validation_decision',lambda *a,**k:
                        {'scientific_acceptance':False,'reasons':['not_ready'],'observed':{}})
    out=tmp_path/'diagnostic'
    result=d.run(Namespace(config='unused',check_report=None,out_dir=str(out),steps=400,
        eval_every=200,windows=8,foundation_report=str(tmp_path/'foundation.json'),db='train',val_db='val'))
    report=json.loads((out/'diagnostic_report.json').read_text())
    logs=[json.loads(row) for row in (out/'gradients.jsonl').read_text().splitlines()]
    state=m._trusted_torch_load(out/'diagnostic_state.pt',map_location='cpu')
    assert result==2 and len(calls)==2
    assert report['stopped_early'] and report['completed_steps']==2 and not report['diagnostic_ready']
    assert logs[-1]['step']==2 and logs[-1]['gradient']['recorded'] and logs[-1]['component_gradients']['recorded']
    assert state['completed_steps']==2 and not state['formal_checkpoint']
    assert len((out/'optimizer_updates.jsonl').read_text().splitlines())==2
