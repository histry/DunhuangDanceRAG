from argparse import Namespace
import json
from pathlib import Path
import warnings

import pytest
import torch

from training import motion_models as m
from training import refiner_final_failure_audit as f
from training import refiner_group_gradient_audit as a
from training import refiner_safe_start_diagnostics as safe
from training import refiner_zero_start_trajectory as z
from tests.test_refiner_group_gradient_audit import bank_tensor, frozen  # noqa: F401
from tests.test_refiner_safe_start_diagnostics import add_probe, small_factory


def transaction(cfg, *, copies=1):
    part = bank_tensor(cfg)
    batch = {key: torch.cat([value] * copies) for key, value in part.items()}
    batch["bad"][..., 4] = .01 * torch.linspace(0, 8, cfg.window_len).sin()
    return batch


def evaluation_stub():
    labels = ("single_recording/10", "single_recording/28", "cross_event/10", "cross_event/28")
    groups = {split: {label: {"cases": 8, "required": 6, "temporal_required": 6,
                                      "endpoint": 0, "temporal": 0,
                                      "physical_non_regression": 8, "passed": False}
                      for label in labels} for split in ("seen", "new_position")}
    decisions = {split: {"scientific_acceptance": False,
                         "thresholds": {"min_clean_identity_rate": .75}}
                 for split in ("seen", "new_position")}
    return {"group_decisions": groups, "decisions": decisions,
            "diagnostic_gates_passed": False, "failure_breakdown": {}}


def case_rows():
    rows = []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                for index in range(8):
                    rows.append({
                        "split": split, "role": role, "width": width,
                        "group": f"{role}/{width}", "case_index": index,
                        "endpoint": {"before": 2., "after": 1.99, "absolute_improvement": .01,
                                     "relative_improvement": .005,
                                     "required_relative_improvement": .03, "accepted": False},
                        "temporal": {"before": 3., "after": 2.99, "absolute_improvement": .01,
                                     "relative_improvement": .003,
                                     "required_relative_improvement": .03, "accepted": False},
                        "geometry": {"reference_fidelity": {"accepted": True},
                                     "repair_toward_hidden_clean": ({"available": True, "accepted": False,
                                         "before": 1., "after": .99, "absolute_improvement": .01,
                                         "relative_improvement": .01,
                                         "required_relative_improvement": .03}
                                         if role == "single_recording" else {"available": False})},
                        "physical": {"accepted": True, "before": {key: 1. for key in f.PHYSICAL_KEYS},
                                     "after": {key: 1. for key in f.PHYSICAL_KEYS}},
                        "clean_identity": {"accepted": True, "product_log_l1": 0., "contact_l1": 0.},
                        "failure_reasons": ["endpoint/observable_endpoint_not_improved",
                                            "temporal/observable_temporal_not_improved"],
                    })
    return rows


def make_trajectory(frozen, destination, monkeypatch):
    add_probe(frozen)
    small_factory(monkeypatch)
    source, bank, state, _ = frozen
    _, _, cfg, metadata = a.load_frozen_source(
        source, a.LEGACY_COMMIT, legacy_core_strength=.02, legacy_transition_strength=1.)
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps)
    experiment = {"schema": z.SCHEMA, "runtime_commit": f.TRAJECTORY_COMMIT,
                  "source": metadata, "config": {}}
    experiment_hash = f._canonical_hash(experiment)
    destination.mkdir()
    (destination / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    z.save_state(destination / "diagnostic_latest.pt", model, 400, experiment_hash)
    update = {"objective_before": {"repair": 1., "clean": 0., "training_total": 1.},
              "optimizer": {"retained": True, "rolled_back": False},
              "head": {"blocks": {"contact": {"parameter_norm": 0., "actual_update_norm": 0.}}}}
    with (destination / "updates.jsonl").open("w", encoding="utf-8") as handle:
        for step in range(1, 401):
            handle.write(json.dumps({"step": step, **update}) + "\n")
    report = {"schema": z.SCHEMA, "completed": True, "diagnostic_completed": True,
              "completed_steps": 400, "optimizer_steps": 400, "probe_loaded": True,
              "fresh_initialization": True, "source_weights_used_for_initialization": False,
              "historical_comparison_is_descriptive_only": True,
              "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False,
              "experiment": experiment, "experiment_sha256": experiment_hash,
              "final_checkpoint_sha256": a.file_sha256(destination / "diagnostic_latest.pt"),
              "final_state_sha256": safe.state_hash(model.state_dict()),
              "probe_sha256": state["probe_bank_artifact"]["sha256"],
              "trajectory": {"optimizer_steps": {"attempted": 400, "accepted": 400,
                                                   "retained": 400, "rolled_back": 0}},
              "final": evaluation_stub()}
    (destination / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_tensor_and_mask_stats_distinguish_exact_zero_and_nonzero():
    zero = torch.zeros(2, 3)
    nonzero = torch.tensor([[0., 1., 0.], [2., 0., 0.]])
    assert f.tensor_stats(zero)["exactly_zero"]
    stats = f.mask_stats(nonzero)
    assert stats["numel"] == 6 and stats["nonzero_count"] == 2
    assert stats["nonzero_fraction"] == pytest.approx(1 / 3)
    assert stats["max"] == 2 and not stats["exactly_zero"]


def test_contact_root_joint_gradient_slices_use_exact_rows():
    weight = torch.arange(79 * 2.).reshape(79, 2, 1)
    bias = torch.arange(79.)
    result = f._block_stats(weight, bias)
    assert result["contact"]["row_range"] == [0, 4]
    assert result["root"]["combined"]["numel"] == 3 * 3
    assert result["joint"]["combined"]["numel"] == 72 * 3
    assert result["contact"]["weight"]["abs_max"] == weight[:4].abs().max()


def test_true_contact_gradients_are_real_read_only_and_restore_mode_grad_and_hook(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = transaction(cfg)
    model = m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=1e-5)
    model.train()
    model.in_proj.weight.grad = torch.ones_like(model.in_proj.weight)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hooks = len(model.out._forward_hooks)
    result = f.true_contact_gradients(model, batch, cfg)
    assert set(result) >= {"repair_objective", "clean_identity_objective", "training_total"}
    assert set(result["training_total"]["head_parameter_gradients"]) == {"contact", "root", "joint"}
    assert model.training and len(model.out._forward_hooks) == hooks
    assert torch.equal(model.in_proj.weight.grad, torch.ones_like(model.in_proj.weight))
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_zero_contact_mask_blocks_true_contact_parameter_gradient():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = transaction(cfg)
    batch["contact"].zero_()
    batch["clean_contact"].zero_()
    model = m.ProductManifoldTemporalRefiner(hidden=4, output_init_std=1e-5)
    result = f.true_contact_gradients(model, batch, cfg)
    for objective in ("repair_objective", "clean_identity_objective", "training_total"):
        assert result[objective]["head_parameter_gradients"]["contact"]["combined"]["exactly_zero"]


def test_nonzero_contact_mask_exposes_objective_connectivity():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = transaction(cfg)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    with torch.no_grad():
        model.out.bias[0] = .1
    result = f.true_contact_gradients(model, batch, cfg)
    assert not result["training_total"]["head_parameter_gradients"]["contact"]["combined"]["exactly_zero"]


def test_decoder_zero_and_nonzero_mask_controls_and_finite_difference():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    batch = transaction(cfg)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    result = f.contact_decoder_jacobian(model, batch, cfg)
    assert result["zero_mask_control"]["exactly_zero"]
    assert not result["nonzero_mask_control"]["exactly_zero"]
    assert result["finite_difference_control"]["autograd_finite_difference_agree"]
    assert len(result["actual_paths"]) == 8


def test_contact_mask_audit_keeps_four_groups_two_branches_and_controls():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    train = transaction(cfg, copies=6)
    part = bank_tensor(cfg)
    banks = {(split, role): {key: value[offset:offset + 16] for key, value in part.items() if key != "group"}
             for split in ("seen", "new_position")
             for role, offset in (("single_recording", 0), ("cross_event", 16))}
    result = f.contact_mask_audit(train, banks, cfg)
    assert len(result["rows"]) == 24
    assert all(set(row) >= {"effective_decoder_mask", "effective_root_mask", "effective_joint_mask"}
               for row in result["rows"])
    assert {row["branch"] for row in result["rows"]} == {"repair", "clean"}


def test_case_summary_preserves_all_64_cases_and_failure_cross_tabs():
    result = f.summarize_cases(case_rows(), evaluation_stub())
    assert set(result["group_table"]) == {"seen", "new_position"}
    assert all(len(groups) == 4 for groups in result["group_table"].values())
    assert result["failure_reason_by_split"]["endpoint/observable_endpoint_not_improved"] == {
        "seen": 32, "new_position": 32}
    row = result["group_table"]["seen"]["single_recording/10"]
    assert row["cases"] == 8 and not row["passed"]
    assert row["metrics"]["endpoint"]["before"] == 2
    answers = f.answer_scientific_questions(case_rows(), result["group_table"])
    assert answers["primary_endpoint_or_temporal"]["answer"] == "endpoint_and_temporal_tied"
    assert answers["physical_safety_failure"]["answer"] is False


def test_real_case_attribution_has_four_groups_and_forbids_cross_hidden_clean():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = bank_tensor(cfg)
    banks = {(split, role): {key: value[offset:offset + 16] for key, value in part.items() if key != "group"}
             for split in ("seen", "new_position")
             for role, offset in (("single_recording", 0), ("cross_event", 16))}
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    rows = f.case_failure_attribution(model, banks, cfg)
    assert len(rows) == 64 and all(row["case_index"] in range(16) for row in rows)
    assert all(not row["geometry"]["repair_toward_hidden_clean"]["available"]
               for row in rows if row["role"] == "cross_event")
    assert all(row["geometry"]["reference_fidelity"]["hidden_clean_used"] is False for row in rows)
    assert model.training


def test_trajectory_history_requires_exact_400_order_and_records_evidence_limit(tmp_path):
    path = tmp_path / "updates.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for step in range(1, 401):
            handle.write(json.dumps({"step": step, "optimizer": {"retained": True, "rolled_back": False},
                "head": {"blocks": {"contact": {"parameter_norm": 0., "actual_update_norm": 0.}}}}) + "\n")
    result = f.trajectory_contact_history(path)
    assert result["all_steps_retained"] and result["rollback_steps"] == 0
    assert result["contact_parameter_exact_zero_all_steps"]
    assert not result["contact_gradient_recorded_per_block"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 401, "optimizer": {"retained": True, "rolled_back": False},
            "head": {"blocks": {"contact": {"parameter_norm": 0., "actual_update_norm": 0.}}}}) + "\n")
    with pytest.raises(ValueError, match="exactly 400"):
        f.trajectory_contact_history(path)


def test_trajectory_loader_verifies_experiment_checkpoint_and_state_hashes(frozen, tmp_path, monkeypatch):
    report = make_trajectory(frozen, tmp_path / "trajectory", monkeypatch)
    _, _, hashes, loaded, experiment, checkpoint = f._load_trajectory(
        tmp_path / "trajectory", f.TRAJECTORY_COMMIT)
    assert loaded["experiment_sha256"] == f._canonical_hash(experiment)
    assert hashes["diagnostic_latest.pt"] == report["final_checkpoint_sha256"]
    assert safe.state_hash(checkpoint["model_state_dict"]) == report["final_state_sha256"]


@pytest.mark.parametrize("failure", ["commit", "experiment", "checkpoint"])
def test_trajectory_loader_fails_closed_on_provenance_mutation(frozen, tmp_path, monkeypatch, failure):
    directory = tmp_path / "trajectory"
    make_trajectory(frozen, directory, monkeypatch)
    if failure == "commit":
        expected = "wrong"
    elif failure == "experiment":
        expected = f.TRAJECTORY_COMMIT
        experiment = json.loads((directory / "experiment.json").read_text())
        experiment["unexpected"] = True
        (directory / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    else:
        expected = f.TRAJECTORY_COMMIT
        with (directory / "diagnostic_latest.pt").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        f._load_trajectory(directory, expected)


def _stub_contact_results(monkeypatch):
    zero = {"exactly_zero": True}
    nonzero = {"exactly_zero": False}
    blocks = {name: {"combined": zero} for name in ("contact", "root", "joint")}
    objective = {"value": 1., "head_parameter_gradients": blocks}
    gradients = {"repair_objective": objective, "clean_identity_objective": objective,
                 "training_total": objective, "current_terms": {}}
    monkeypatch.setattr(f, "case_failure_attribution", lambda *args: case_rows())
    monkeypatch.setattr(f, "contact_mask_audit", lambda *args: {
        "rows": [], "all_effective_masks_exactly_zero": True})
    monkeypatch.setattr(f, "true_contact_gradients", lambda *args: gradients)
    monkeypatch.setattr(f, "contact_decoder_jacobian", lambda *args, **kwargs: {
        "actual_paths": [{"all_positions": zero}], "zero_mask_control": zero,
        "nonzero_mask_control": nonzero, "finite_difference_control": {}})


def test_run_is_create_only_read_only_zero_optimizer_and_provenance_complete(
        frozen, tmp_path, monkeypatch):
    directory = tmp_path / "trajectory"
    traj = make_trajectory(frozen, directory, monkeypatch)
    _stub_contact_results(monkeypatch)
    monkeypatch.setattr(safe, "evaluate_final", lambda *args: evaluation_stub())
    output = tmp_path.parent / f"{tmp_path.name}_audit" / "report.json"
    before = {name: a.file_sha256(frozen[0] / name) for name in
              ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")}
    args = Namespace(state_dir=str(frozen[0]), trajectory_dir=str(directory), output=str(output),
                     device="cpu", expected_trajectory_commit=f.TRAJECTORY_COMMIT,
                     legacy_core_strength=.02, legacy_transition_strength=1.)
    assert f.run(args) == 0
    result = json.loads(output.read_text())
    assert result["optimizer"] is None and result["optimizer_steps"] == 0
    assert result["model_state_unchanged"] and result["checkpoint_selection_performed"] is False
    assert result["contact_connectivity"]["exact_zero_origin"]["classification"] == "mask_zero"
    assert result["provenance"]["trajectory_experiment_sha256"] == traj["experiment_sha256"]
    assert not any(result[key] for key in ("scientific_acceptance", "publish_allowed", "pilot_allowed"))
    assert before == {name: a.file_sha256(frozen[0] / name) for name in before}
    with pytest.raises(FileExistsError):
        f.run(args)


def test_source_mutation_during_audit_fails_closed(frozen, tmp_path, monkeypatch):
    directory = tmp_path / "trajectory"
    make_trajectory(frozen, directory, monkeypatch)
    _stub_contact_results(monkeypatch)
    monkeypatch.setattr(safe, "evaluate_final", lambda *args: evaluation_stub())
    original = f.case_failure_attribution

    def mutate(*args):
        with (frozen[0] / "diagnostic_report.json").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return original(*args)

    monkeypatch.setattr(f, "case_failure_attribution", mutate)
    output = tmp_path.parent / f"{tmp_path.name}_mutation_audit" / "report.json"
    args = Namespace(state_dir=str(frozen[0]), trajectory_dir=str(directory),
                     output=str(output), device="cpu",
                     expected_trajectory_commit=f.TRAJECTORY_COMMIT,
                     legacy_core_strength=.02, legacy_transition_strength=1.)
    with pytest.raises(RuntimeError, match="source artifact changed"):
        f.run(args)
    assert not Path(args.output).exists()


def test_diagnostic_checkpoint_remains_forbidden_for_formal_inference(tmp_path):
    cfg = m.MotionGenerationConfig(device="cpu")
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    path = tmp_path / "diagnostic_latest.pt"
    z.save_state(path, model, 400, "experiment")
    with pytest.raises(RuntimeError, match="rejects a non-product refiner checkpoint"):
        m._cached_inference_model("boundary_refiner", path, cfg)


def test_vjp_scalar_conversion_warning_is_removed():
    source = Path(z.__file__).read_text(encoding="utf-8")
    assert "vjp_error = (hidden_gradient - expected).detach().double().norm().item()" in source
    with warnings.catch_warnings(record=True) as caught:
        value = (torch.ones(1, requires_grad=True) - 1).detach().double().norm().item()
    assert value == 0 and not caught
