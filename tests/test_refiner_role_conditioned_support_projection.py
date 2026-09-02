import inspect
from pathlib import Path

import pytest
import torch

from training import motion_models as m
from training import refiner_role_conditioned_support_projection_experiment as r
from tests.test_refiner_group_gradient_audit import bank_tensor


def tiny_batch(*, cases=4, frames=60):
    cfg = m.MotionGenerationConfig(device="cpu", window_len=frames)
    source = bank_tensor(cfg)
    batch = {
        key: value[:cases].clone()
        for key, value in source.items()
    }
    batch["group"] = torch.tensor([0, 1, 2, 3][:cases], dtype=torch.long)
    phase = torch.linspace(0, 6, frames)
    batch["bad"][..., 4] = 0.02 * phase.sin()
    return cfg, batch


def nonzero_base(hidden=4):
    model = m.ProductManifoldTemporalRefiner(hidden=hidden)
    with torch.no_grad():
        model.out.weight.normal_(0, 1.0e-4)
        model.out.bias.normal_(0, 1.0e-5)
    return model


def test_schema_steps_roles_and_geometry_contract_are_fixed():
    assert r.SCHEMA == "refiner_role_conditioned_support_projection_experiment_v2"
    assert r.STEPS == 400
    assert r.GEOMETRY_DIM == 75
    assert r.ROLE_MAPPING == {"single_recording": 0, "cross_event": 1}
    assert r.SUPPORT_PROJECTION_PROTOCOL.endswith("binary_projection_v1")


def test_base_is_frozen_and_only_two_adapters_require_grad():
    wrapper = r.FrozenBaseRCSPModel(nonzero_base())
    scope = r.validate_parameter_scope(wrapper)
    assert scope["base_trainable_parameters"] == 0
    assert scope["adapter_parameters"] == 2 * 75 * (4 + 1)
    assert scope["single_adapter_parameters"] == scope["cross_adapter_parameters"]
    assert all(not parameter.requires_grad for parameter in wrapper.base.parameters())
    assert all(parameter.requires_grad for parameter in wrapper.adapter.parameters())
    assert set(scope["trainable_parameter_names"]) == {
        "adapter.single_adapter.weight",
        "adapter.single_adapter.bias",
        "adapter.cross_adapter.weight",
        "adapter.cross_adapter.bias",
    }


def test_both_role_adapters_are_exactly_zero_initialized():
    wrapper = r.FrozenBaseRCSPModel(nonzero_base())
    result = r.validate_zero_initialization(wrapper)
    assert result["all_parameters_exactly_zero"]
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in wrapper.adapter.parameters()
    )


def test_train_role_ids_come_from_explicit_group_contract_not_width():
    groups = torch.tensor([0, 1, 2, 3, 1, 2])
    assert r.role_ids_from_train_groups(groups).tolist() == [0, 0, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="group ids"):
        r.attach_train_role_ids({"seam": torch.zeros(2, 60, 1)})


def test_final_role_ids_use_explicit_role_and_ignore_width():
    metadata = [
        {"role": "single_recording", "width": 28},
        {"role": "cross_event", "width": 10},
    ]
    assert r.role_ids_from_metadata(metadata, "cpu").tolist() == [0, 1]
    metadata[0]["role"] = "unknown"
    with pytest.raises(ValueError, match="unknown explicit final role"):
        r.role_ids_from_metadata(metadata, "cpu")


def test_role_heads_are_independent_and_mixed_batch_routes_explicitly():
    adapter = r.RoleConditionedSupportProjectedAdapter(hidden_dim=2)
    with torch.no_grad():
        adapter.single_adapter.bias.fill_(1.0)
        adapter.cross_adapter.bias.fill_(2.0)
    hidden = torch.zeros(2, 2, 3)
    joint = torch.ones(2, 3, 24)
    root = torch.ones(2, 3, 1)
    projected, details = adapter(hidden, torch.tensor([0, 1]), joint, root)
    assert torch.equal(projected[0], torch.ones_like(projected[0]))
    assert torch.equal(projected[1], torch.full_like(projected[1], 2.0))
    assert torch.equal(details["adapter_raw"], projected)


def test_binary_projection_does_not_repeat_soft_confidence():
    adapter = r.RoleConditionedSupportProjectedAdapter(hidden_dim=1)
    with torch.no_grad():
        adapter.single_adapter.bias.fill_(1.0)
    hidden = torch.zeros(2, 1, 2)
    role = torch.zeros(2, dtype=torch.long)
    joint = torch.stack((torch.full((2, 24), 0.1), torch.full((2, 24), 0.9)))
    root = torch.stack((torch.full((2, 1), 0.1), torch.full((2, 1), 0.9)))
    projected, _ = adapter(hidden, role, joint, root)
    assert torch.equal(projected[0], projected[1])
    assert torch.equal(projected, torch.ones_like(projected))


def test_projection_blocks_all_outside_support_coordinates_exactly():
    adapter = r.RoleConditionedSupportProjectedAdapter(hidden_dim=1)
    with torch.no_grad():
        adapter.single_adapter.bias.fill_(3.0)
    hidden = torch.zeros(1, 1, 2)
    joint = torch.ones(1, 2, 24)
    root = torch.ones(1, 2, 1)
    root[:, 0] = 0
    joint[:, 1, 5] = 0
    projected, details = adapter(hidden, torch.zeros(1, dtype=torch.long), joint, root)
    support = details["binary_support"]
    assert torch.count_nonzero(projected * (1 - support)) == 0
    assert torch.count_nonzero(projected[0, 0, :3]) == 0
    assert torch.count_nonzero(projected[0, 1, 3 + 5 * 3 : 3 + 6 * 3]) == 0


def test_adapter_is_geometry_only_and_contact_stays_bit_identical():
    cfg, batch = tiny_batch(cases=2)
    base = nonzero_base()
    wrapper = r.FrozenBaseRCSPModel(base)
    with torch.no_grad():
        wrapper.adapter.single_adapter.bias.fill_(0.25)
        wrapper.adapter.cross_adapter.bias.fill_(-0.25)
        batch = r.attach_train_role_ids(batch)
        r.rcsp_batch_outputs(wrapper, batch, cfg, capture_details=True)
    details = wrapper.last_details
    assert details["raw_base"].shape[-1] == 79
    assert details["adapter_projected"].shape[-1] == 75
    assert torch.equal(details["raw_adapted"][..., :4], details["raw_base"][..., :4])


def test_zero_adapter_has_exact_train_raw_decoded_and_metric_parity():
    cfg, batch = tiny_batch(cases=4)
    base = nonzero_base()
    wrapper = r.FrozenBaseRCSPModel(base)
    result = r.train_initial_parity(base, wrapper, batch, cfg)
    assert result["verified"]
    assert result["raw_max_abs_difference"] == 0
    assert result["decoded_motion_max_abs_difference"] == 0
    assert result["clean_decoded_motion_max_abs_difference"] == 0
    assert set(result["scientific_metric_max_abs_difference"]) == {
        "temporal_scientific_deficit",
        "endpoint_scientific_deficit",
    }
    assert all(value == 0 for value in result["scientific_metric_max_abs_difference"].values())


def test_backward_populates_only_adapter_gradients():
    cfg, batch = tiny_batch(cases=4)
    wrapper = r.FrozenBaseRCSPModel(nonzero_base())
    batch = r.attach_train_role_ids(batch)
    repair, clean, _terms, _ = r.rcsp_batch_objectives(wrapper, batch, cfg)
    (repair + cfg.product_refiner_clean_identity_weight * clean).backward()
    assert all(parameter.grad is None for parameter in wrapper.base.parameters())
    assert all(parameter.grad is not None for parameter in wrapper.adapter.parameters())


def test_checked_step_closure_updates_or_rolls_back_adapter_without_touching_base():
    cfg, batch = tiny_batch(cases=4)
    wrapper = r.FrozenBaseRCSPModel(nonzero_base())
    batch = r.attach_train_role_ids(batch)
    base_hash = r.safe.state_hash(wrapper.base.state_dict())
    optimizer = torch.optim.AdamW(wrapper.adapter.parameters(), lr=cfg.lr, weight_decay=1e-4)
    repair, clean, terms, _ = r.rcsp_batch_objectives(wrapper, batch, cfg)
    loss = repair + cfg.product_refiner_clean_identity_weight * clean
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_norm = float(torch.nn.utils.clip_grad_norm_(wrapper.adapter.parameters(), 1.0))
    update = r.checked_refiner_step(
        optimizer,
        loss,
        lambda: r.rcsp_guarded_total_batch_loss(wrapper, batch, cfg),
        gradient_unscale=max(1.0, clip_norm + 1e-6),
        group_guard_before=m._refiner_group_repair_losses(terms, require_all=True),
        group_guard_relative_tolerance=cfg.product_refiner_group_guard_relative_tolerance,
        group_guard_absolute_tolerance=cfg.product_refiner_group_guard_absolute_tolerance,
    )
    assert update["protocol"] == m.REFINER_UPDATE_PROTOCOL
    assert isinstance(update["optimizer_update_accepted"], bool)
    assert r.safe.state_hash(wrapper.base.state_dict()) == base_hash
    assert all(parameter.grad is None for parameter in wrapper.base.parameters())


def test_invalid_or_missing_explicit_route_fails_closed():
    cfg, batch = tiny_batch(cases=1)
    wrapper = r.FrozenBaseRCSPModel(nonzero_base())
    with pytest.raises(RuntimeError, match="requires explicit role"):
        wrapper(batch["bad"], batch["cond"], batch["seam"], batch["joint"])
    adapter = r.RoleConditionedSupportProjectedAdapter(hidden_dim=1)
    with pytest.raises(ValueError, match="role_id values"):
        adapter(
            torch.zeros(1, 1, 2),
            torch.tensor([2]),
            torch.ones(1, 2, 24),
            torch.ones(1, 2, 1),
        )


def test_support_statistics_require_exact_zero_projected_escape():
    rows = []
    for index, role in enumerate(("single_recording", "cross_event")):
        rows.append(
            {
                "split": "seen",
                "role": role,
                "width": 10,
                "group": f"{role}/10",
                "case_index": index,
                "bank_case_index": index,
                "adapter_raw_norm": 2.0,
                "adapter_projected_norm": 1.0,
                "projection_retention_ratio": 0.5,
                "adapter_energy_outside_support_before_projection": 3.0,
                "projected_outside_support_max": 0.0,
            }
        )
    # Add the missing fixed scopes; summary accepts any non-empty cells only when present.
    expanded = []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                for case_index in range(8):
                    expanded.append({**rows[0], "split": split, "role": role, "width": width,
                                     "group": f"{role}/{width}", "case_index": case_index,
                                     "bank_case_index": case_index})
    result = r.support_projection_summary(expanded)
    assert result["summary"]["overall"]["projected_outside_support_max"] == 0
    expanded[0]["projected_outside_support_max"] = 1e-8
    with pytest.raises(RuntimeError, match="escaped correction"):
        r.support_projection_summary(expanded)


def synthetic_case_rows(*, rcsp=False):
    rows = []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                for index in range(2):
                    rescued = rcsp and role == "single_recording" and index == 0
                    rows.append(
                        {
                            "split": split,
                            "role": role,
                            "width": width,
                            "temporal_gate_pass": rescued,
                            "endpoint_gate_pass": rescued,
                            "physical_pass": True,
                            "geometry_pass": True,
                            "clean_pass": True,
                            "all_diagnostic_conditions": rescued,
                            "temporal_scientific_deficit": 0.8 if rcsp else 1.0,
                            "endpoint_scientific_deficit": 0.9 if rcsp else 1.0,
                            "temporal_repair_gain": 0.2 if rcsp else 0.0,
                            "endpoint_repair_gain": 0.1 if rcsp else 0.0,
                        }
                    )
    return rows


def test_baseline_comparison_reports_base_rcsp_and_signed_deltas():
    base = r.fixed_final_summary(synthetic_case_rows())
    rcsp = r.fixed_final_summary(synthetic_case_rows(rcsp=True))
    comparison = r.baseline_comparison(base, rcsp)
    assert set(comparison["overall"]) >= {"BASE", "RCSP", "delta_rcsp_minus_base"}
    assert comparison["overall"]["delta_rcsp_minus_base"][
        "temporal_scientific_deficit_mean"
    ] < 0
    assert comparison["baseline_retrained"] is False


def test_scientific_answer_can_describe_single_rescue_without_accepting_science():
    base_rows = synthetic_case_rows()
    rcsp_rows = synthetic_case_rows(rcsp=True)
    base = r.fixed_final_summary(base_rows)
    rcsp = r.fixed_final_summary(rcsp_rows)
    comparison = r.baseline_comparison(base, rcsp)
    answer = r.scientific_answers(
        {**base, "case_level": base_rows},
        {**rcsp, "case_level": rcsp_rows},
        comparison,
    )
    assert answer["role_conditioned_direction_rescue"] == (
        "SUPPORTED_BY_DIAGNOSTIC_EXPERIMENT"
    )
    assert answer["any_single_recording_case_crossed_temporal_gate"]
    assert "does not prove a root cause" in answer["claim_boundary"]


def test_scientific_answer_separates_deficit_improvement_from_width_gate_rescue():
    base_rows = synthetic_case_rows()
    rcsp_rows = []
    for row in base_rows:
        rescued = row["role"] == "cross_event" and row["width"] == 10
        rcsp_rows.append(
            {
                **row,
                "temporal_gate_pass": rescued,
                "endpoint_gate_pass": rescued,
                "all_diagnostic_conditions": rescued,
                "temporal_scientific_deficit": 0.8,
                "endpoint_scientific_deficit": 0.9,
                "temporal_repair_gain": 0.2,
                "endpoint_repair_gain": 0.1,
            }
        )
    base = r.fixed_final_summary(base_rows)
    rcsp = r.fixed_final_summary(rcsp_rows)
    comparison = r.baseline_comparison(base, rcsp)
    answer = r.scientific_answers(
        {**base, "case_level": base_rows},
        {**rcsp, "case_level": rcsp_rows},
        comparison,
    )
    assert all(answer["group_descriptive_improvement"].values())
    assert answer["temporal_gate_rescue_width_pattern"] == "WIDTH_10_ONLY"
    assert answer["temporal_gate_pass_delta_by_width"] == {"10": 4, "28": 0}
    assert answer["temporal_gate_pass_delta_by_role"] == {
        "single_recording": 0,
        "cross_event": 4,
    }
    assert answer["role_conditioned_direction_rescue"] == (
        "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
    )


def test_direction_summary_uses_medians_and_all_required_scopes_are_declared():
    rows = [
        {
            "projected_adapter_delta_vs_negative_temporal_gradient_cosine": value,
            "adapted_total_action_vs_negative_temporal_gradient_cosine": value / 2,
        }
        for value in (-0.5, 0.5, 1.0)
    ]
    summary = r._direction_summary(rows)
    assert summary[
        "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
    ] == pytest.approx(0.5)
    assert r.FINAL_BLOCK_ORDER == (
        ("seen", "single_recording"),
        ("seen", "cross_event"),
        ("new_position", "single_recording"),
        ("new_position", "cross_event"),
    )


def test_cli_has_no_alpha_seed_width_or_step_tuning_knobs():
    source = inspect.getsource(r.main)
    for forbidden in ("--alpha", "--seed", "--width", "--steps", "--resume"):
        assert forbidden not in source
    assert "--expected-main-commit" in source
    assert "--expected-trajectory-commit" in source


def test_report_contract_keeps_production_science_publish_and_pilot_false():
    source = inspect.getsource(r.run)
    assert '"production_model_modified": False' in source
    assert '"production_inference_modified": False' in source
    assert '"checkpoint_selection_performed": False' in source
    assert '"scale_selection_performed": False' in source
    assert '"scientific_acceptance": False' in source
    assert '"publish_allowed": False' in source
    assert '"pilot_allowed": False' in source
    assert '"held_out_new_position_used_for_training": False' in source
    assert '"all_frozen_reservoir_banks"' in source


def test_shell_wrapper_forbids_dirty_wrong_commit_overwrite_and_pilot():
    source = (Path(__file__).parents[1] / "scripts" /
              "run_refiner_role_conditioned_support_projection.sh").read_text()
    assert 'git status --porcelain' in source
    assert 'git rev-parse HEAD' in source and 'git rev-parse origin/main' in source
    assert 'test ! -e "$RESULT_DIR"' in source
    assert "--device cuda" in source
    assert "No production edit" in source
    assert "Pilot" in source


def test_production_refiner_source_has_no_rcsp_adapter_dependency():
    production = Path(m.__file__).read_text(encoding="utf-8")
    assert "RoleConditionedSupportProjectedAdapter" not in production
    assert "FrozenBaseRCSPModel" not in production
