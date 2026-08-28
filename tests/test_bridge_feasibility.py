import copy
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from training import bridge_feasibility as f
from training import motion_models as m
from tests.test_duration_inbetween import motion

torch = m.torch


def bank(frames=48,device="cpu"):
    cfg = m.MotionGenerationConfig(device=device)
    original = motion(frames)[None]
    bad = original.copy()
    bad[:,16:32,4] += .005*np.sin(np.linspace(0,np.pi,16))
    seam = np.zeros((1,frames,1),np.float32)
    seam[:,16:32] = 1
    return m._prepare_refiner_batch(original,bad,seam,np.zeros((1,frames,32),np.float32),cfg,torch.device(device)),cfg


def test_direct_control_cannot_read_clean_or_run_network_and_backtracks(tmp_path):
    b,cfg = bank()
    b["clean"].fill_(float("nan"))  # not an available repair target
    before = m._observable_refiner_objective(b["bad"],b["bad"],b["seam"],cfg)[0]
    with mock.patch.object(m.ProductManifoldTemporalRefiner,"forward",side_effect=AssertionError("network used")):
        prediction,trace = f.direct_optimize(b,cfg,2,label="test",log_path=tmp_path/"log.jsonl")
    after = m._observable_refiner_objective(prediction,b["bad"],b["seam"],cfg)[0]
    assert after <= before + 1e-6
    assert torch.isfinite(prediction).all()
    assert 0 <= trace[0]["joint_cap_fraction"] <= 1
    assert not list(tmp_path.glob("*.pt"))


def test_observable_normalization_does_not_downweight_quiet_windows():
    b,cfg = bank()
    x = b["bad"].expand(2,-1,-1)
    seam = b["seam"].expand(2,-1,-1)
    base = torch.tensor([.01,.28],dtype=torch.float64)
    before = {"endpoint_velocity_jump_mps":base,"temporal_energy":base,"seam_jerk_mps3":base*100,
              "seam_acceleration_mps2":base,"valid":torch.ones(2,dtype=torch.bool)}
    after = {**before,"endpoint_velocity_jump_mps":base*1.1,"temporal_energy":base*1.1}
    with mock.patch.object(m,"boundary_metrics_torch",side_effect=[after,before]):
        losses,terms = m._observable_refiner_objective(x,x,seam,cfg,reduction="none")
    torch.testing.assert_close(terms["endpoint_continuity"],torch.full_like(base,.2))
    torch.testing.assert_close(terms["temporal_supervision"],torch.full_like(base,.2))
    assert losses.shape == (2,)


def test_component_gradient_conflicts_are_measured_not_assumed():
    model = torch.nn.Linear(1,1,bias=False)
    p = model.weight.sum()
    terms = {"endpoint_continuity":p,"temporal_supervision":-p,"jerk":p*0,"support_excess":p*0}
    r = m._refiner_component_gradients(model,terms,m.MotionGenerationConfig())
    assert r["pairs"]["endpoint/temporal"]["cosine"] == pytest.approx(-1)
    assert r["pairs"]["endpoint/temporal"]["conflicting"]
    assert r["pairs"]["endpoint/support"]["cosine"] is None
    assert model.weight.grad is None
    p.backward()  # diagnostic retained the actual training graph


def test_fixed_support_does_not_disappear_when_candidate_starts_sliding():
    cfg = m.MotionGenerationConfig(device="cpu")
    ref = motion(48)
    pred = ref.copy()
    pred[:,4] += np.arange(48)*.03
    r = m._fixed_support_stage_gate(ref,pred,cfg)
    assert not r["accepted"]
    assert r["support_comparison"]["frames"]>0
    assert r["support_comparison"]["after"]["foot_skate_mps_p95"]>.8
    assert r["support_comparison"]["after"]["foot_support_drift_m_max"]>1
    assert r["support_comparison"]["before"]["foot_support_drift_m_max"]==0
    assert "independent_support_diagnostic" in r


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_support_statistics_share_loss_audit_and_gradients(device):
    if device=="cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not installed")
    b,cfg = bank(device=device)
    ref = m.fk_24_torch(b["bad"]).detach()
    pred = ref.clone()
    pred[:,20:28,list(m.DEFAULT_FOOT_JOINTS),0] += .02
    pred.requires_grad_(True)
    before,after,static = m._reference_support_statistics_torch(pred,ref,b["bad"][...,:4],cfg)
    assert static.shape==(1,48,4)
    after["foot_support_drift_m_max"].sum().backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum()>0
    if device=="cuda":
        cpu = m._reference_support_statistics_torch(pred.detach().cpu(),ref.cpu(),b["bad"][...,:4].cpu(),cfg)
        for key in before:
            torch.testing.assert_close(before[key].cpu(),cpu[0][key],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(after[key].cpu(),cpu[1][key],rtol=1e-5,atol=1e-6)


def test_loss_per_case_mean_matches_scalar_reduction():
    b,cfg = bank()
    vector,terms = m._observable_refiner_objective(b["bad"],b["bad"],b["seam"],cfg,reduction="none")
    scalar,average = m._observable_refiner_objective(b["bad"],b["bad"],b["seam"],cfg)
    torch.testing.assert_close(vector.mean(),scalar)
    for key in terms:
        torch.testing.assert_close(terms[key].mean(),average[key])


def test_balanced_sampling_and_group_failure_cannot_hide_in_mean():
    idx = f.balanced_indices(16,np.random.default_rng(1))
    assert len(set(idx))==8
    for offset in (0,16):
        for parity in (0,1):
            assert sum(i in range(offset+parity,offset+16,2) for i in idx)==2
    metrics = passing_metrics()
    for row in metrics["cross_event"]["windows"]:
        if row["width"]==28:
            row["observable"]["temporal_accepted"] = False
    groups = f.group_decisions(metrics,m.MotionGenerationConfig())
    assert not groups["cross_event/28"]["passed"]
    assert groups["cross_event/10"]["passed"]


def passing_metrics():
    def row(width):
        return {"width":width,"observable":{"endpoint_accepted":True,"temporal_accepted":True,
            "reference_fidelity_accepted":True,"physical_non_regression":{"accepted":True}},
            "safety":{"accepted":True}}
    return {"windows":[row(w) for w in (10,28) for _ in range(8)],
            "cross_event":{"windows":[row(w) for w in (10,28) for _ in range(8)]}}


def test_smoke_incomplete_groups_and_bad_roundtrip_never_authorize_fitting():
    cfg = m.MotionGenerationConfig()
    report = {"completed":True,"direct_steps":200,"windows":[{}]*8,
              "direct":{s:passing_metrics() for s in ("seen","new_position")},
              "interpolation_vs_ik":{f"{s}/{r}":{"cases":16,"preserved":16}
                    for s in ("seen","new_position") for r in ("single_recording","cross_event")},
              "roundtrip":{"rejected_count":0,"cases":64,"exact_identity_count":64}}
    assert f.foundation_decision(report,cfg)["ready_for_network_diagnostic"]
    bad = copy.deepcopy(report); bad["direct_steps"]=2
    assert not f.foundation_decision(bad,cfg)["ready_for_network_diagnostic"]
    bad = copy.deepcopy(report); bad["roundtrip"]["rejected_count"]=1
    assert not f.foundation_decision(bad,cfg)["ready_for_network_diagnostic"]
    bad = copy.deepcopy(report); bad["roundtrip"]["exact_identity_count"]=63
    assert "no_edit_decode_not_identity" in f.foundation_decision(bad,cfg)["reasons"]
    bad = copy.deepcopy(report); bad["interpolation_vs_ik"]={}
    assert not f.foundation_decision(bad,cfg)["ready_for_network_diagnostic"]
    bad = copy.deepcopy(report); del bad["direct"]["new_position"]
    assert not f.foundation_decision(bad,cfg)["ready_for_network_diagnostic"]


def test_v8_shell_dependency_order_and_no_asset_retraining():
    script = (Path(__file__).resolve().parents[1]/"scripts/train_refiner_v8.sh").read_text(encoding="utf8")
    assert script.index('"$MODE" == foundation') < script.index('"$MODE" == diagnose')
    assert '--foundation_report "$FOUNDATION"' in script
    assert '--check_report "$FIT_DIR/diagnostic_report.json"' in script
    assert script.index('--check_report') < script.index('"$MODE" == pilot')
    assert script.index('SOURCE_DISJOINT_PILOT_ACCEPTED') < script.index('train-diffusion')
    for name in ("RETARGET_CACHE","EVENT_DB"):
        assert f"RETARGET_CLEAN_REBUILD_{name}=0" in script
    for name in ("ROUTER","DURATION","PLANNER"):
        assert f"RETARGET_CLEAN_RETRAIN_{name}=0" in script
