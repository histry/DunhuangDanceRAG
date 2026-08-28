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
