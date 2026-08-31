from argparse import Namespace
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from training import motion_models as m
from training import refiner_safe_start_diagnostics as s
from training.refiner_optimizer import checked_refiner_step as real_checked_step
from tests.test_refiner_group_gradient_audit import bank_tensor, frozen  # noqa: F401


def small_factory(monkeypatch):
    original = m.ProductManifoldTemporalRefiner
    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", lambda **kw: original(hidden=4, **kw))


def add_probe(frozen):
    path, bank, state, report = frozen
    roles = {role: {key: value[start:start + 16].clone() for key, value in bank["anchor"].items() if key != "group"}
             for role, start in (("single_recording", 0), ("cross_event", 16))}
    probe = {"schema": "refiner_local_context_probe_bank_v1", "probe_only": True,
             "updates_forbidden": True, "formal_checkpoint": False, "publish_allowed": False,
             "fingerprint": bank["fingerprint"], "config": bank["config"], "windows": bank["windows"], "banks": roles}
    torch.save(probe, path / "probe_bank.pt")
    descriptor = {"file": "probe_bank.pt", "sha256": s.a.file_sha256(path / "probe_bank.pt"),
                  "cases": 32, "probe_only": True, "updates_forbidden": True}
    state["probe_bank_artifact"] = report["probe_bank_artifact"] = descriptor
    torch.save(state, path / "diagnostic_state.pt")
    (path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    return probe


def args_for(frozen, tmp_path_factory, **changes):
    args = Namespace(state_dir=str(frozen[0]), out_dir=str(tmp_path_factory.mktemp("safe_start") / "pair"),
                     legacy_core_strength=.02, legacy_transition_strength=1., device="cpu", seed=42, preflight_only=False)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def test_paired_seed_changes_only_output_weight_and_preserves_rng(monkeypatch):
    small_factory(monkeypatch)
    before = torch.get_rng_state().clone()
    cfg = m.MotionGenerationConfig()
    states = s.paired_initial_states(cfg, 42)
    assert torch.equal(before, torch.get_rng_state())
    assert states.keys() == s.ARMS.keys()
    zero, gaussian = states.values()
    assert zero.keys() == gaussian.keys()
    assert [key for key in zero if not torch.equal(zero[key], gaussian[key])] == ["out.weight"]
    assert torch.count_nonzero(zero["out.weight"]) == 0
    assert gaussian["out.weight"].std().item() == pytest.approx(1e-5, rel=.2)
    again = s.paired_initial_states(cfg, 42)
    assert s.state_hash(again["A1_gaussian"]) == s.state_hash(gaussian)


@pytest.mark.parametrize("std", [-1, float("nan"), float("inf")])
def test_invalid_initialization_is_rejected(std):
    with pytest.raises(ValueError, match="initialization std"):
        m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=std)


def test_gaussian_unlocks_true_trunk_derivative_without_changing_decoder():
    torch.manual_seed(42)
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = {k: v[:2].double() if v.is_floating_point() else v[:2] for k, v in bank_tensor(cfg).items()}
    model = m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=1e-5).double()

    def objective():
        pred, _ = m._refiner_batch_outputs(model, batch, cfg)
        return pred[..., 4].mean()

    gradient = torch.autograd.grad(objective(), model.in_proj.weight)[0]
    assert gradient.norm() > 0
    index = int(gradient.abs().argmax())
    parameter = model.in_proj.weight
    original = parameter.detach().clone()
    eps = 1e-5
    with torch.no_grad():
        parameter.flatten()[index] += eps
        plus = objective()
        parameter.copy_(original)
        parameter.flatten()[index] -= eps
        minus = objective()
        parameter.copy_(original)
    torch.testing.assert_close(gradient.flatten()[index], (plus - minus) / (2 * eps), rtol=2e-5, atol=1e-11)
    with torch.no_grad():
        model.out.weight.zero_()
    assert torch.count_nonzero(torch.autograd.grad(objective(), model.in_proj.weight)[0]) == 0


def test_initial_safety_requires_every_case_and_both_branches(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = bank_tensor(cfg)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    calls = {"repair": 0, "clean": 0}

    def stage(*args, **kwargs):
        calls["repair"] += 1
        return {"accepted": calls["repair"] != 32, "reasons": []}

    def clean(acc, *args):
        calls["clean"] += 1
        acc["clean_identity_gates"].append({"accepted": calls["clean"] != 1, "reasons": []})

    monkeypatch.setattr(m, "_fixed_support_stage_gate", stage)
    monkeypatch.setattr(m, "_observable_reference_fidelity", lambda *args: ({}, True))
    monkeypatch.setattr(m, "_record_validation_clean_identity_prediction", clean)
    result = s.initial_safety(model, [("first", part), ("last", part)], cfg)
    assert not result["passed"] and result["checked_cases"] == 64
    assert calls == {"repair": 64, "clean": 64}
    assert result["failure_counts"] == {"clean/identity_rejection": 1, "repair/physical_rejection": 1}


def test_real_initial_safety_rejects_unsafe_candidate():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = {k: v[:1].clone() for k, v in bank_tensor(cfg).items()}
    clean, _ = m.enforce_edge151_contract_np(part["clean"][0].numpy(), cfg, derive_contact=True, project_rot=True)
    part["clean"][0] = torch.as_tensor(clean)
    part["bad"][0] = part["clean"][0]
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    assert s.initial_safety(model, [("valid", part)], cfg)["passed"]
    # A large vertical edit under an enlarged experimental cap must not pass
    # merely because the layer weights are finite. Production caps stay intact.
    with torch.no_grad():
        model.out.bias[5] = 10
    unsafe_cfg = dataclasses.replace(cfg, product_refiner_root_cap_m=1.)
    assert not s.initial_safety(model, [("unsafe", part)], unsafe_cfg)["passed"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_measured_update_is_after_real_transaction_rollback(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = m.MotionGenerationConfig(device=device, window_len=60)
    part = bank_tensor(cfg)
    part["bad"][..., 4] = .01 * torch.linspace(0, 8, 60).sin()
    bank = {"anchor": part, "context_reservoir": {str(i): part for i in range(5)},
            "transaction_schedule": [[0, 1, 2, 3, 4]]}
    model = m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=1e-5).to(device)
    initial = s.d._cpu_tree(model.state_dict())

    def reject(optimizer, loss, closure, **kwargs):
        return real_checked_step(optimizer, loss,
                                 lambda: (loss.detach() + 1, kwargs["group_guard_before"]), max_trials=1, **kwargs)

    monkeypatch.setattr(s, "checked_refiner_step", reject)
    result = s.train_arm(model, initial, bank, cfg, "A1_gaussian", tmp_path, "test", steps=1)
    assert result["optimizer_updates"]["retained_steps"] == 1
    row = result["final_training_transaction"]
    assert row["cases"] == 192 and row["context_indices"] == [0, 1, 2, 3, 4]
    assert row["scopes"]["shared_trunk"]["gradient_norm_before_clip"] > 0
    assert all(value["actual_update_norm"] == value["displacement_from_initial_norm"] == 0 for value in row["layers"].values())
    snapshot = m._trusted_torch_load(tmp_path / "diagnostic_latest.pt", map_location="cpu")
    assert snapshot["version"] != m.REFINER_MODEL_VERSION
    assert not snapshot["formal_checkpoint"] and not snapshot["publish_allowed"] and not snapshot["pilot_allowed"]
    with pytest.raises(RuntimeError, match="rejects a non-product refiner checkpoint"):
        m._cached_inference_model("boundary_refiner", tmp_path / "diagnostic_latest.pt", cfg)


def test_real_final_scoring_keeps_all_groups_and_rejects_no_repair(frozen):
    add_probe(frozen)
    path, bank, state, _ = frozen
    cfg = m.MotionGenerationConfig(device="cpu")
    probe, _ = s.load_probe(path, state, bank, cfg)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    result = s.evaluate_final(model, bank, probe, cfg)
    assert not result["diagnostic_gates_passed"]
    assert set(result["group_decisions"]) == {"seen", "new_position"}
    assert all(len(groups) == 4 and all(row["cases"] == 8 for row in groups.values())
               for groups in result["group_decisions"].values())


def test_preflight_failure_forbids_both_optimizers_and_probe(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)
    monkeypatch.setattr(s, "initial_safety", lambda model, *args: {"passed": not bool(model.out.weight.count_nonzero())})
    monkeypatch.setattr(s, "train_arm", lambda *args: pytest.fail("unsafe initialization started training"))
    monkeypatch.setattr(s, "load_probe", lambda *args: pytest.fail("unsafe initialization loaded probe"))
    args = args_for(frozen, tmp_path_factory)
    assert s.run(args) == 2
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert report["completed"] and not report["diagnostic_completed"] and not report["preflight_passed"]
    assert not report["probe_loaded"] and not report["pilot_allowed"]
    assert all(entry["completed_steps"] == 0 for entry in report["arms"].values())


def test_preflight_only_needs_no_probe_and_performs_no_updates(frozen, tmp_path_factory, monkeypatch):
    small_factory(monkeypatch)
    monkeypatch.setattr(s, "initial_safety", lambda *args: {"passed": True})
    monkeypatch.setattr(s, "train_arm", lambda *args: pytest.fail("preflight-only started training"))
    monkeypatch.setattr(s, "load_probe", lambda *args: pytest.fail("preflight-only loaded probe"))
    args = args_for(frozen, tmp_path_factory, preflight_only=True)
    assert not (frozen[0] / "probe_bank.pt").exists()
    assert s.run(args) == 0
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert report["completed"] and report["preflight_passed"] and not report["diagnostic_completed"]
    assert not report["probe_loaded"] and not report["pilot_allowed"]
    assert all(entry["completed_steps"] == 0 for entry in report["arms"].values())


def test_source_changed_during_preflight_forbids_training(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)

    def preflight(*args):
        with (frozen[0] / "diagnostic_report.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return {"passed": True}

    monkeypatch.setattr(s, "initial_safety", preflight)
    monkeypatch.setattr(s, "train_arm", lambda *args: pytest.fail("changed source started training"))
    monkeypatch.setattr(s, "load_probe", lambda *args: pytest.fail("changed source loaded probe"))
    args = args_for(frozen, tmp_path_factory)
    with pytest.raises(RuntimeError, match="frozen source changed"):
        s.run(args)
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert not report["completed"] and not report["probe_loaded"]
    assert all(entry["completed_steps"] == 0 for entry in report["arms"].values())


def test_probe_is_only_opened_after_both_fixed_final_states(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)
    monkeypatch.setattr(s, "initial_safety", lambda *args: {"passed": True})
    events = []

    def train(model, initial, bank, cfg, arm, destination, experiment_hash):
        assert not any(event == "probe" for event in events)
        events.append(arm)
        s.save_state(destination / "diagnostic_latest.pt", model, arm, s.STEPS, experiment_hash)
        return {"completed_steps": s.STEPS, "optimizer_updates": {},
                "final_state_sha256": s.state_hash(model.state_dict()),
                "final_checkpoint_sha256": s.a.file_sha256(destination / "diagnostic_latest.pt")}

    original = m._trusted_torch_load

    def load(path, **kwargs):
        if path.name == "probe_bank.pt":
            assert events == list(s.ARMS)
            events.append("probe")
        return original(path, **kwargs)

    monkeypatch.setattr(s, "train_arm", train)
    monkeypatch.setattr(m, "_trusted_torch_load", load)
    monkeypatch.setattr(s, "evaluate_final", lambda *args: {"diagnostic_gates_passed": True, "failure_breakdown": {}})
    args = args_for(frozen, tmp_path_factory)
    before = torch.get_rng_state().clone()
    assert s.run(args) == 0
    assert events == [*s.ARMS, "probe"] and torch.equal(before, torch.get_rng_state())
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert report["diagnostic_completed"] and report["probe_loaded"]
    assert not any(report[k] for k in ("scientific_acceptance", "publish_allowed", "pilot_allowed", "source_weights_used_for_initialization"))
    with pytest.raises(FileExistsError):
        s.run(args)


@pytest.mark.parametrize("failure", ["hash", "config", "train_flag", "width"])
def test_bad_probe_artifact_fails_closed(frozen, failure):
    probe = add_probe(frozen)
    path, bank, state, report = frozen
    if failure == "hash":
        with (path / "probe_bank.pt").open("ab") as handle:
            handle.write(b"tampered")
    else:
        if failure == "config":
            probe["config"] = {**probe["config"], "lr": 99.}
        elif failure == "train_flag":
            probe["updates_forbidden"] = False
        else:
            probe["banks"]["single_recording"]["seam"][0].zero_()
        torch.save(probe, path / "probe_bank.pt")
        state["probe_bank_artifact"]["sha256"] = s.a.file_sha256(path / "probe_bank.pt")
        (path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        s.load_probe(path, state, bank, m.MotionGenerationConfig())
