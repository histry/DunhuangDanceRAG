"""Contract tests for the fixed SECDR diagnostic intervention.

The server run is the execution gate.  These tests encode the immutable
architecture, support/q math, decision tree, and reporting boundary.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import motion_models as m
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_support_extent_direction_rotation_intervention as audit


def _row(split, width, index, *, rcsp=0.2, secdr=0.3, temporal_rcsp=False, temporal_secdr=True):
    return {
        "split": split,
        "role": "cross_event",
        "width": width,
        "identity": f"{split}/cross_event/{width}/{index}",
        "G_rcsp": rcsp,
        "G_secdr": secdr,
        "temporal_pass_base": False,
        "temporal_pass_rcsp": temporal_rcsp,
        "temporal_pass_secdr": temporal_secdr,
        "endpoint_acceptance_base": True,
        "endpoint_acceptance_rcsp": True,
        "endpoint_acceptance_secdr": True,
        "jerk_non_regression_base": True,
        "jerk_non_regression_rcsp": True,
        "jerk_non_regression_secdr": True,
        "overall_acceptance_base": False,
        "overall_acceptance_rcsp": True,
        "overall_acceptance_secdr": True,
        "applied_action_norm_rcsp": 1.0,
        "applied_action_norm_secdr": 1.0,
        "gain_per_action_norm_rcsp": 0.1,
        "gain_per_action_norm_secdr": 0.2,
        "safety_non_regression": True,
        "binary_support_identical": True,
        "temporal_alignment_rcsp": {"cosine_to_negative_gradient": 0.2},
        "temporal_alignment_secdr": {"cosine_to_negative_gradient": 0.4},
        "rotation_one_minus_cosine": 0.1,
        "predecoder_norm_preservation_error": {"root_max_abs_error": 0.0, "joint_max_abs_error": 0.0},
    }


def _efficacy(value):
    return {"supported": value}


def _mechanism(value):
    return {"supported": value}


def _integrated_secdr_fixture():
    cfg = m.MotionGenerationConfig(device="cpu")
    frames = cfg.window_len
    clean = torch.zeros(2, frames, 151)
    clean[..., 7:] = torch.tensor(m.identity6d_np()).repeat(24)
    clean[..., 5] = 0.95
    bad = clean.clone()
    phase = torch.linspace(0.0, 1.0, frames)
    bad[..., 4] += 0.02 * phase.sin()
    seam = torch.zeros(2, frames, 1)
    seam[0, 30:40] = 1.0
    seam[1, 30:58] = 1.0
    joint = seam.expand(2, frames, 24).clone() * 0.18
    root = seam.clone() * 0.18
    batch = {
        "clean": clean,
        "bad": bad,
        "seam": seam,
        "cond": torch.zeros(2, frames, 32),
        "group": torch.tensor([1, 3]),
        "joint": joint,
        "root": root,
        "contact": root.clone(),
        "clean_joint": joint.clone(),
        "clean_root": root.clone(),
        "clean_contact": root.clone(),
    }
    batch = audit.attach_train_role_ids(batch)
    base = m.ProductManifoldTemporalRefiner(hidden=4, fps=cfg.fps)
    rcsp_model = rcsp.FrozenBaseRCSPModel(base)
    with torch.no_grad():
        rcsp_model.adapter.cross_adapter.bias.fill_(0.02)
    model = audit.SECDRModel(rcsp_model, 0.25, 0.5)
    calibration = {"s_min": 0.0, "s_max": 1.0}
    return cfg, batch, model, calibration


def test_schema_is_exact():
    assert audit.SCHEMA == "refiner_support_extent_conditioned_direction_rotation_intervention_v1"


def test_frozen_parent_is_exact():
    assert audit.IMPLEMENTATION_PARENT_COMMIT == "83aa72a59508f4bba21d684648a55a63b13141ab"


def test_phase21_commit_is_exact():
    assert audit.FROZEN_PHASE21_COMMIT == "c461ba44689103cd0690488267e3bd42507ad7ab"


def test_bctr_commit_is_exact():
    assert audit.FROZEN_BCTR_COMMIT == "b0cd4437cfb0144046b1408397cc5dad72471cf9"


def test_fixed_case_counts_are_exact():
    assert audit.PRIMARY_CASES == 32
    assert audit.FINAL_CASES == 64
    assert audit.CASES_PER_GROUP == 8
    assert audit.TRAIN_EXPECTED_CROSS_CASES == 96


def test_fixed_step_budget_is_exact():
    assert audit.STEPS == 400


def test_geometry_dimension_is_root_plus_joints():
    assert audit.ROOT_DIM + audit.JOINT_DIM == audit.GEOMETRY_DIM


def test_rotator_parameter_count_is_exact():
    assert audit.ROTATOR_PARAMETER_COUNT == 5193


def test_rotator_has_only_root_and_joint_maps():
    assert set(audit.TangentDirectionRotator().state_dict()) == {"root.weight", "joint.weight"}


def test_rotator_is_bias_free():
    rotator = audit.TangentDirectionRotator()
    assert rotator.root.bias is None
    assert rotator.joint.bias is None


def test_rotator_weight_shapes_are_exact():
    rotator = audit.TangentDirectionRotator()
    assert tuple(rotator.root.weight.shape) == (3, 3)
    assert tuple(rotator.joint.weight.shape) == (72, 72)


def test_rotator_is_zero_initialized():
    rotator = audit.TangentDirectionRotator()
    assert bool((rotator.root.weight == 0).all())
    assert bool((rotator.joint.weight == 0).all())
    assert rotator.parameter_count == 5193


def test_support_extent_returns_binary_support_and_active_frames():
    joint = torch.zeros((2, 4, 24))
    root = torch.zeros((2, 4, 1))
    joint[0, 1, 0] = 0.2
    root[1, 2, 0] = 0.4
    support, active, fraction = audit.support_extent_fraction(joint, root)
    assert support.shape == (2, 4, 75)
    assert active.tolist() == [1, 1]
    assert fraction.tolist() == pytest.approx([0.25, 0.25])


def test_support_extent_uses_strictly_positive_weights():
    joint = torch.zeros((1, 4, 24))
    root = torch.zeros((1, 4, 1))
    joint[0, 0, 0] = -1.0
    root[0, 1, 0] = 1.0
    _support, active, _fraction = audit.support_extent_fraction(joint, root)
    assert active.item() == 1


def test_support_extent_rejects_no_active_frame():
    with pytest.raises(ValueError, match="no active frame"):
        audit.support_extent_fraction(torch.zeros((1, 3, 24)), torch.zeros((1, 3, 1)))


def test_support_extent_rejects_invalid_fraction_layout():
    with pytest.raises(ValueError):
        audit.support_extent_fraction(torch.ones((1, 3, 24)), torch.ones((2, 3, 1)))


@pytest.mark.parametrize("value, expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (-1.0, 0.0), (2.0, 1.0)])
def test_support_extent_q_clips_to_unit_interval(value, expected):
    assert audit.support_extent_q(value, 0.0, 1.0) == pytest.approx(expected)


def test_support_extent_q_is_monotonic():
    values = torch.tensor([0.1, 0.2, 0.3])
    q = audit.support_extent_q(values, 0.1, 0.3)
    assert bool((q[1:] >= q[:-1]).all())


def test_support_extent_q_rejects_degenerate_range():
    with pytest.raises(ValueError, match="range"):
        audit.support_extent_q(0.5, 1.0, 1.0)


def test_effective_q_bypasses_single_recording():
    roles = torch.tensor([0, 1, 0, 1])
    fractions = torch.tensor([0.2, 0.2, 0.8, 0.8])
    q = audit.effective_conditioner_q(roles, fractions, 0.2, 0.8)
    assert q.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_effective_q_rejects_misaligned_roles():
    with pytest.raises(ValueError, match="aligned"):
        audit.effective_conditioner_q(torch.zeros(2, dtype=torch.long), torch.zeros(1), 0.0, 1.0)


def test_zero_rotator_is_exact_identity_for_action():
    rotator = audit.TangentDirectionRotator()
    action = torch.randn((2, 5, 75))
    support = torch.ones_like(action)
    q = torch.tensor([0.0, 1.0])
    result, _details = rotator(action, support, q)
    assert torch.equal(result[0], action[0])
    assert torch.equal(result[1], action[1])


def test_zero_initialized_rotator_has_nonzero_eligible_gradient():
    rotator = audit.TangentDirectionRotator()
    action = torch.linspace(-0.4, 0.6, 2 * 3 * 75).reshape(2, 3, 75)
    support = torch.ones_like(action)
    q = torch.ones(2)
    result, _details = rotator(action, support, q)
    assert torch.equal(result, action)
    target = action.clone()
    target[..., :3] += 0.07
    target[..., 3:] -= 0.03
    loss = (result - target).square().mean()
    loss.backward()
    root_gradient = rotator.root.weight.grad
    joint_gradient = rotator.joint.weight.grad
    assert root_gradient is not None
    assert joint_gradient is not None
    assert bool(torch.isfinite(root_gradient).all())
    assert bool(torch.isfinite(joint_gradient).all())
    total = math.sqrt(
        float(root_gradient.detach().double().norm()) ** 2
        + float(joint_gradient.detach().double().norm()) ** 2
    )
    assert total > audit.GRADIENT_NUMERICAL_TOL
    for parameter in rotator.parameters():
        parameter.grad = None


def test_integrated_secdr_objective_has_zero_start_gradient_without_step():
    _cfg, batch, model, calibration = _integrated_secdr_fixture()
    repair, clean, _terms, _identity_terms = audit.secdr_batch_objectives(
        model, batch, _cfg, calibration, capture_details=True
    )
    loss = repair + _cfg.product_refiner_clean_identity_weight * clean
    loss.backward()
    root_gradient = model.rotator.root.weight.grad
    joint_gradient = model.rotator.joint.weight.grad
    assert root_gradient is not None
    assert joint_gradient is not None
    assert bool(torch.isfinite(root_gradient).all())
    assert bool(torch.isfinite(joint_gradient).all())
    assert max(
        float(root_gradient.detach().double().norm()),
        float(joint_gradient.detach().double().norm()),
    ) > audit.GRADIENT_NUMERICAL_TOL
    for parameter in model.rotator.parameters():
        parameter.grad = None
    model.clear_last_details()


def test_zero_start_preflight_fails_closed_without_a_gradient(monkeypatch):
    _cfg, batch, model, calibration = _integrated_secdr_fixture()

    def zero_objective(*_args, **_kwargs):
        return torch.zeros(()), torch.zeros(()), {}, {}

    monkeypatch.setattr(audit, "secdr_batch_objectives", zero_objective)
    result = audit.zero_start_trainability_preflight(model, batch, _cfg, calibration)
    assert result["passed"] is False
    assert result["optimizer_constructed"] is False
    assert result["optimizer_steps"] == 0
    assert result["parameter_update_attempted"] is False
    assert result["any_gradient_nonzero"] is False
    assert result["parameters_unchanged_after_probe"] is True
    assert result["gradients_cleared_after_probe"] is True


def test_zero_action_remains_zero():
    rotator = audit.TangentDirectionRotator()
    action = torch.zeros((1, 2, 75))
    result, _details = rotator(action, torch.ones_like(action), torch.ones(1))
    assert torch.equal(result, action)


def test_support_is_reapplied_after_rotation():
    rotator = audit.TangentDirectionRotator()
    with torch.no_grad():
        rotator.root.weight[0, 1] = 1.0
    action = torch.ones((1, 2, 75))
    support = torch.zeros_like(action)
    support[..., 0] = 1.0
    result, _details = rotator(action, support, torch.ones(1))
    assert bool((result[..., 1:] == 0).all())


def test_root_and_joint_norms_are_preserved():
    rotator = audit.TangentDirectionRotator()
    with torch.no_grad():
        rotator.root.weight[0, 1] = 0.2
        rotator.joint.weight[0, 1] = 0.1
    action = torch.randn((2, 3, 75))
    support = torch.ones_like(action)
    result, _details = rotator(action, support, torch.ones(2))
    assert torch.allclose(result[..., :3].norm(dim=-1), action[..., :3].norm(dim=-1), atol=1e-6)
    assert torch.allclose(result[..., 3:].norm(dim=-1), action[..., 3:].norm(dim=-1), atol=1e-6)


def test_scope_rows_supports_seen_new_and_widths():
    rows = [_row("seen", 10, 0), _row("seen", 28, 1), _row("new_position", 10, 2), _row("new_position", 28, 3)]
    assert len(audit._scope_rows(rows, "overall")) == 4
    assert len(audit._scope_rows(rows, "seen")) == 2
    assert len(audit._scope_rows(rows, "new")) == 2
    assert len(audit._scope_rows(rows, "width10")) == 2
    assert len(audit._scope_rows(rows, "width28")) == 2


def test_scope_rows_supports_each_frozen_group():
    rows = [_row("seen", 10, 0), _row("seen", 28, 1), _row("new_position", 10, 2), _row("new_position", 28, 3)]
    for group in audit.GROUP_ORDER:
        assert len(audit._scope_rows(rows, group)) == 1


def test_scope_rows_rejects_unknown_scope():
    with pytest.raises(ValueError, match="unknown"):
        audit._scope_rows([], "unknown")


def test_width_gap_uses_rcsp_and_secdr_medians():
    rows = [_row("seen", width, index, rcsp=(0.2 if width == 10 else 0.4), secdr=(0.3 if width == 10 else 0.4)) for width in (10, 28) for index in range(8)]
    gap = audit.width_gap(rows, "seen")
    assert gap["gap_current"] == pytest.approx(0.2)
    assert gap["gap_secdr"] == pytest.approx(0.1)
    assert gap["gap_shrink_fraction"] == pytest.approx(0.5)


def test_width_gap_is_null_for_zero_current_gap():
    rows = [_row("seen", width, index, rcsp=0.2, secdr=0.2) for width in (10, 28) for index in range(8)]
    assert audit.width_gap(rows, "seen")["gap_shrink_fraction"] is None


def test_summary_contains_required_fields():
    rows = [_row(split, width, index) for split in ("seen", "new_position") for width in (10, 28) for index in range(8)]
    summary = audit.make_summaries(rows)
    assert set(audit.SUMMARY_SCOPES) == set(summary)
    for scope in audit.SUMMARY_SCOPES:
        assert "median_G_rcsp" in summary[scope]
        assert "median_G_secdr" in summary[scope]
        assert "median_temporal_alignment_cosine_secdr" in summary[scope]
        assert "median_gain_per_action_norm_secdr" in summary[scope]


def test_summary_requires_exactly_32_primary_rows():
    with pytest.raises(ValueError, match="32"):
        audit.make_summaries([])


def test_efficacy_requires_all_preregistered_conditions():
    rows = [_row(split, width, index) for split in ("seen", "new_position") for width in (10, 28) for index in range(8)]
    result = audit._efficacy(rows, "seen")
    assert set(result["conditions"]) == {
        "gap_reduced", "width28_gain_strictly_improved", "width10_gain_non_degraded",
        "width10_temporal_pass_non_decreased", "width10_endpoint_non_decreased",
        "width28_endpoint_non_decreased", "safety_non_regression", "original_jerk_non_regression",
    }


def test_efficacy_result_is_boolean():
    rows = [_row("seen", width, index) for width in (10, 28) for index in range(8)]
    assert isinstance(audit._efficacy(rows, "seen")["supported"], bool)


def test_mechanism_requires_width28_cases():
    rows = [_row("seen", 28, index) for index in range(8)]
    result = audit._mechanism(rows, "seen", {"total": 1.0})
    assert result["width"] == 28
    assert result["cases"] == 8


def test_mechanism_records_defined_cases():
    rows = [_row("seen", 28, index) for index in range(8)]
    rows[0]["temporal_alignment_rcsp"]["cosine_to_negative_gradient"] = None
    result = audit._mechanism(rows, "seen", {"total": 1.0})
    assert result["defined_paired_cases"] == 7


def test_full_decision_is_exact():
    result = audit.adjudicate({"seen": _efficacy(True), "new": _efficacy(True)}, {"seen": _mechanism(True), "new": _mechanism(True)})
    assert result["result"] == "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_SUPPORTED"
    assert result["next_action"] == "freeze_secdr_candidate_and_enter_joint_evidence_synthesis"


def test_partial_decision_is_exact():
    result = audit.adjudicate({"seen": _efficacy(True), "new": _efficacy(False)}, {"seen": _mechanism(True), "new": _mechanism(True)})
    assert result["result"] == "PARTIAL_WIDTH_CONDITIONED_DIRECTION_INTERVENTION"
    assert result["next_action"] == "retain_partial_direction_evidence_and_enter_joint_evidence_synthesis"


def test_mechanism_only_decision_is_exact():
    result = audit.adjudicate({"seen": _efficacy(False), "new": _efficacy(False)}, {"seen": _mechanism(True), "new": _mechanism(True)})
    assert result["result"] == "DIRECTION_MECHANISM_WITHOUT_SUFFICIENT_EFFICACY"
    assert result["next_action"] == "reject_as_solution_and_enter_joint_evidence_synthesis"


def test_not_supported_decision_is_exact():
    result = audit.adjudicate({"seen": _efficacy(False), "new": _efficacy(False)}, {"seen": _mechanism(False), "new": _mechanism(False)})
    assert result["result"] == "WIDTH_CONDITIONED_DIRECTION_INTERVENTION_NOT_SUPPORTED"
    assert result["next_action"] == "reject_secdr_and_enter_joint_evidence_synthesis"


@pytest.mark.parametrize("field", ["no_further_intervention_search", "causal_root_cause_proven", "scientific_acceptance", "publish_allowed", "pilot_allowed"])
def test_decision_safety_flags_are_fixed(field):
    result = audit.adjudicate({"seen": _efficacy(False), "new": _efficacy(False)}, {"seen": _mechanism(False), "new": _mechanism(False)})
    expected = True if field == "no_further_intervention_search" else False
    assert result[field] is expected


def test_forward_has_no_width_argument():
    parameters = set(inspect.signature(audit.SECDRModel.forward).parameters)
    assert "width" not in parameters


def test_rotator_forward_has_no_width_argument():
    parameters = set(inspect.signature(audit.TangentDirectionRotator.forward).parameters)
    assert "width" not in parameters


def test_training_scope_names_only_rotator_weights():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "model.rotator.parameters()" in source
    assert "self.base.parameters()" in source
    assert "self.rcsp.adapter.parameters()" in source
    assert "update_zero" not in source


def test_no_width_head_or_width_loss_is_declared():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "width_head" not in source
    assert "direction_cosine_loss_added" in source


def test_bctr_is_not_recomputed_by_secdr():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"bctr_recomputed": False' in source
    assert '"bctr_used_for_candidate_evaluation": False' in source


def test_no_production_inference_entrypoint_is_called():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "run_normal_generation" not in source
    assert "formal_inference" not in source


def test_report_keeps_scientific_flags_false():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"scientific_acceptance": False' in source
    assert '"publish_allowed": False' in source
    assert '"pilot_allowed": False' in source


def test_optimizer_budget_is_not_a_sweep():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "for step in range(1, STEPS + 1)" in source
    assert "best_checkpoint" not in source
    assert "alpha_sweep" not in source


def test_server_script_requires_explicit_lineage_inputs():
    script = Path(__file__).parents[1] / "scripts/run_refiner_support_extent_direction_rotation_intervention.sh"
    source = script.read_text(encoding="utf-8")
    assert "PHASE21_REPORT" in source
    assert "BCTR_REPORT" in source
    assert "BCTR_CORRECTION_REPORT" in source
    assert "PREVIOUS_SECDR_REPORT" in source
    assert "latest-artifact discovery" in source.lower()


def test_report_schema_is_present_in_documentation():
    doc = Path(__file__).parents[1] / "docs/refiner_support_extent_conditioned_direction_rotation_intervention.md"
    assert audit.SCHEMA in doc.read_text(encoding="utf-8") or "SECDR" in doc.read_text(encoding="utf-8")
