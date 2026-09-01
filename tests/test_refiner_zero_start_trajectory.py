from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from training import motion_models as m
from training import refiner_safe_start_diagnostics as safe
from training import refiner_zero_start_trajectory as z
from training.refiner_optimizer import checked_refiner_step as real_checked_step
from tests.test_refiner_group_gradient_audit import bank_tensor, frozen  # noqa: F401
from tests.test_refiner_safe_start_diagnostics import add_probe


def small_factory(monkeypatch):
    original = m.ProductManifoldTemporalRefiner
    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", lambda **kwargs: original(hidden=4, **kwargs))


def args_for(frozen, tmp_path_factory, **changes):
    args = Namespace(state_dir=str(frozen[0]),
                     out_dir=str(tmp_path_factory.mktemp("zero_trajectory") / "run"),
                     legacy_core_strength=.02, legacy_transition_strength=1.,
                     device="cpu", seed=42)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def synthetic_bank(cfg):
    part = bank_tensor(cfg)
    part["bad"][..., 4] = .01 * torch.linspace(0, 8, cfg.window_len).sin()
    return {"anchor": part, "context_reservoir": {str(i): part for i in range(5)},
            "transaction_schedule": [[0, 1, 2, 3, 4]]}


def test_fresh_zero_state_is_exact_deterministic_and_preserves_rng(monkeypatch):
    small_factory(monkeypatch)
    cfg = m.MotionGenerationConfig(device="cpu")
    before = torch.get_rng_state().clone()
    state = z.fresh_zero_state(cfg, 42)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.count_nonzero(state["out.weight"]) == 0
    assert torch.count_nonzero(state["out.bias"]) == 0
    assert safe.state_hash(state) == safe.state_hash(z.fresh_zero_state(cfg, 42))
    with pytest.raises(ValueError, match="initialization seed"):
        z.fresh_zero_state(cfg, -1)


def test_fixed_contract_constants_and_full_transaction(frozen):
    _, bank, _, _ = frozen
    cfg = m.MotionGenerationConfig(device="cpu")
    assert z.STEPS == 400
    assert z.SNAPSHOT_STEPS == (0, 1, 2, 3, 5, 10, 25, 50, 100, 200, 300, 400)
    rows = safe.train_banks(bank, cfg)
    assert len(rows) == 1 + len(bank["context_reservoir"])
    batch = z.a.materialize_transaction(bank, cfg, 0)
    assert len(batch["group"]) == 192
    assert [int((batch["group"] == i).sum()) for i in range(4)] == [48] * 4


def test_step_zero_detail_has_true_zero_trunk_vjp_and_no_state_change(tmp_path, monkeypatch):
    small_factory(monkeypatch)
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    bank = synthetic_bank(cfg)
    initial = z.fresh_zero_state(cfg, 42)
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps)
    model.load_state_dict(initial)
    batch = z.a.materialize_transaction(bank, cfg, 0)
    before = safe.state_hash(model.state_dict())
    detail = z.detailed_snapshot(model, batch, cfg, initial, 0)
    assert before == safe.state_hash(model.state_dict())
    assert detail["head"]["weight_is_exactly_zero"]
    assert detail["head_transport"]["hidden_gradient_norm"] == 0
    assert detail["head_transport"]["output_gradient_norm"] > 0
    assert detail["head_transport"]["vjp_absolute_error_norm"] == 0
    assert set(detail["repair_tangent"]) == {"raw", "after_mask", "after_smoothing", "after_taper", "applied"}
    assert all(row["gradient"]["norm"] == 0 for name, row in detail["parameters"].items()
               if not name.startswith("out."))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_two_step_trajectory_measures_preclip_gradient_and_retained_update(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    small_factory(monkeypatch)
    cfg = m.MotionGenerationConfig(device=device, window_len=60)
    bank = synthetic_bank(cfg)
    initial = z.fresh_zero_state(cfg, 42)
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
    model.load_state_dict(initial)
    result = z.train_trajectory(model, initial, bank, cfg, tmp_path, "test", steps=2)
    rows = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and result["optimizer_summary"]["attempted_steps"] == 2
    first = rows[0]
    assert first["cases"] == 192 and set(first["cases_per_group"].values()) == {48}
    assert first["gradient_before_clipping"]["verified_pre_clip"]
    assert first["gradient_before_clipping"]["shared_trunk_norm"] == 0
    assert first["gradient_before_clipping"]["output_head_norm"] > 0
    assert first["optimizer"]["retained"]
    assert not first["head"]["weight_is_exactly_zero"]
    assert result["trajectory"]["first_nonzero_head_step"] == 1
    assert result["trajectory"]["first_nonzero_trunk_gradient_step"] == 2
    assert result["trajectory"]["optimizer_steps"]["attempted"] == 2
    assert result["trajectory"]["optimizer_steps"]["accepted"] == 2
    assert set(result["snapshot_artifacts"]) == {"0", "1", "2"}
    assert (tmp_path / "diagnostic_latest.pt").is_file()


def test_rejected_step_is_exactly_rolled_back(tmp_path, monkeypatch):
    small_factory(monkeypatch)
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    bank = synthetic_bank(cfg)
    initial = z.fresh_zero_state(cfg, 42)
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps)
    model.load_state_dict(initial)

    def reject(optimizer, loss, closure, **kwargs):
        return real_checked_step(optimizer, loss,
                                 lambda: (loss.detach() + 1, kwargs["group_guard_before"]),
                                 max_trials=1, **kwargs)

    monkeypatch.setattr(z, "checked_refiner_step", reject)
    result = z.train_trajectory(model, initial, bank, cfg, tmp_path, "test", steps=1)
    row = json.loads((tmp_path / "updates.jsonl").read_text())
    assert row["optimizer"]["rolled_back"] and not row["optimizer"]["retained"]
    assert all(scope["actual_update_norm"] == 0
               for scope in row["actual_retained_update_after_rollback"].values())
    assert all(item["actual_update_norm"] == 0 for item in row["parameter_updates"].values())
    assert safe.state_hash(model.state_dict()) == safe.state_hash(initial)
    assert result["trajectory"]["optimizer_steps"]["rolled_back"] == 1


def test_trajectory_summary_separates_path_length_and_displacement():
    rows = []
    for step in range(1, 401):
        rows.append({
            "step": step,
            "optimizer": {"retained": step != 7},
            "head": {"weight_is_exactly_zero": step < 2},
            "gradient_before_clipping": {"shared_trunk_norm": 0 if step == 1 else 2.,
                                           "trunk_to_head_norm_ratio": .5},
            "actual_retained_update_after_rollback": {
                "shared_trunk": {"actual_update_norm": 0 if step == 7 else 3.,
                                  "displacement_from_initial_norm": 4.},
                "output_head": {"actual_update_norm": 0 if step == 7 else 6.,
                                "displacement_from_initial_norm": 8.},
            },
        })
    optimizer = {"attempted_steps": 400, "accepted_steps": 399, "retained_steps": 1}
    summary = z.trajectory_summary(rows, optimizer)
    assert summary["first_nonzero_head_step"] == 2
    assert summary["first_nonzero_trunk_gradient_step"] == 2
    assert summary["cumulative_retained_movement"]["shared_trunk"] == 1197
    assert summary["final_displacement_from_initial"]["shared_trunk"] == 4
    assert summary["optimizer_steps"] == {
        "attempted": 400, "accepted": 399, "retained": 399, "rolled_back": 1,
        "accepted_rate": 399 / 400, "rolled_back_rate": 1 / 400,
        "legacy_optimizer_counter_note": "optimizer_summary.retained_steps means rejected steps restored by rollback"}


def test_preflight_failure_has_zero_optimizer_and_never_loads_probe(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)
    monkeypatch.setattr(safe, "initial_safety", lambda *args: {"passed": False, "checked_cases": 2976})
    monkeypatch.setattr(z, "train_trajectory", lambda *args: pytest.fail("failed preflight started optimizer"))
    monkeypatch.setattr(safe, "load_probe", lambda *args: pytest.fail("failed preflight loaded probe"))
    args = args_for(frozen, tmp_path_factory)
    assert z.run(args) == 2
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert report["completed"] and not report["diagnostic_completed"]
    assert report["optimizer_steps"] == 0 and not report["probe_loaded"]
    assert not report["pilot_allowed"] and not report["publish_allowed"]


def test_probe_opens_only_after_fixed_step_400_state(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)
    monkeypatch.setattr(safe, "initial_safety", lambda model, rows, cfg: {
        "passed": True, "checked_cases": sum(len(batch["group"]) for _, batch in rows)})
    events = []

    def train(model, initial, bank, cfg, destination, experiment_hash):
        assert torch.count_nonzero(model.out.weight) == 0
        events.append("train_400")
        z.save_state(Path(destination) / "diagnostic_latest.pt", model, 400, experiment_hash)
        return {"completed_steps": 400, "optimizer_summary": {"attempted_steps": 400},
                "trajectory": {}, "snapshot_artifacts": {}, "final_training_transaction": {},
                "final_state_sha256": safe.state_hash(model.state_dict()),
                "final_checkpoint_sha256": z.a.file_sha256(Path(destination) / "diagnostic_latest.pt")}

    original_load = m._trusted_torch_load

    def tracked(path, **kwargs):
        if Path(path).name == "probe_bank.pt":
            assert events == ["train_400"]
            events.append("probe")
        return original_load(path, **kwargs)

    monkeypatch.setattr(z, "train_trajectory", train)
    monkeypatch.setattr(m, "_trusted_torch_load", tracked)
    monkeypatch.setattr(safe, "evaluate_final", lambda *args: {
        "diagnostic_gates_passed": True, "failure_breakdown": {}})
    args = args_for(frozen, tmp_path_factory)
    before = torch.get_rng_state().clone()
    assert z.run(args) == 0
    assert events == ["train_400", "probe"] and torch.equal(before, torch.get_rng_state())
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert report["historical_comparison_is_descriptive_only"]
    assert report["fresh_initialization"] and not report["source_weights_used_for_initialization"]
    assert report["completed_steps"] == report["optimizer_steps"] == 400
    assert report["probe_loaded"] and report["diagnostic_completed"]
    assert not any(report[key] for key in ("scientific_acceptance", "publish_allowed", "pilot_allowed"))


def test_source_mutation_after_preflight_fails_closed(frozen, tmp_path_factory, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)

    def mutate(*args):
        with (frozen[0] / "diagnostic_report.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return {"passed": True, "checked_cases": 2976}

    monkeypatch.setattr(safe, "initial_safety", mutate)
    monkeypatch.setattr(z, "train_trajectory", lambda *args: pytest.fail("mutated source trained"))
    args = args_for(frozen, tmp_path_factory)
    with pytest.raises(RuntimeError, match="frozen source changed"):
        z.run(args)
    report = json.loads((Path(args.out_dir) / "report.json").read_text())
    assert not report["completed"] and report["optimizer_steps"] == 0 and not report["probe_loaded"]
    assert report["error"]["type"] == "RuntimeError"
    assert "frozen source changed" in report["error"]["message"]


def test_artifact_is_rejected_by_formal_inference_and_resume(tmp_path):
    cfg = m.MotionGenerationConfig(device="cpu")
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    path = tmp_path / "diagnostic_latest.pt"
    z.save_state(path, model, 400, "test")
    with pytest.raises(RuntimeError, match="rejects a non-product refiner checkpoint"):
        m._cached_inference_model("boundary_refiner", path, cfg)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="schema mismatch"):
        m._load_training_resume_snapshot(
            path, stage="refiner", model=model, optimizer=optimizer, target_steps=400,
            cfg=cfg, training_contract={}, validation_contract={}, device="cpu")


def test_true_gradient_finite_difference_has_no_surrogate_path():
    torch.manual_seed(5)
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = {key: (value[:2].double() if value.is_floating_point() else value[:2])
             for key, value in bank_tensor(cfg).items()}
    model = m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=1e-5).double()

    def objective():
        repair, clean, _, _ = m._refiner_batch_objectives(model, batch, cfg)
        return repair + cfg.product_refiner_clean_identity_weight * clean

    gradient = torch.autograd.grad(objective(), model.in_proj.weight)[0]
    index = int(gradient.abs().argmax())
    original = model.in_proj.weight.detach().clone()
    epsilon = 1e-5
    with torch.no_grad():
        model.in_proj.weight.flatten()[index] += epsilon
        plus = objective()
        model.in_proj.weight.copy_(original)
        model.in_proj.weight.flatten()[index] -= epsilon
        minus = objective()
        model.in_proj.weight.copy_(original)
    torch.testing.assert_close(gradient.flatten()[index], (plus - minus) / (2 * epsilon),
                               rtol=5e-4, atol=1e-9)


@pytest.mark.parametrize("failure", ["hash", "config"])
def test_probe_contract_is_reused_and_fails_closed(frozen, failure):
    probe = add_probe(frozen)
    path, bank, state, report = frozen
    if failure == "hash":
        with (path / "probe_bank.pt").open("ab") as handle:
            handle.write(b"changed")
    else:
        probe["config"] = {**probe["config"], "lr": 9.}
        torch.save(probe, path / "probe_bank.pt")
        state["probe_bank_artifact"]["sha256"] = z.a.file_sha256(path / "probe_bank.pt")
        report["probe_bank_artifact"] = state["probe_bank_artifact"]
        (path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        safe.load_probe(path, state, bank, m.MotionGenerationConfig())
