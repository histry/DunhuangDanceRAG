from argparse import Namespace
import inspect
import json

import pytest
import torch

from training import motion_models as m
from training import refiner_temporal_action_alignment_audit as t
from tests.test_refiner_group_gradient_audit import bank_tensor


def small_batch(*, cases=2, frames=60):
    cfg = m.MotionGenerationConfig(device="cpu", window_len=frames)
    source = bank_tensor(cfg)
    batch = {key: value[:cases].clone() for key, value in source.items() if key != "group"}
    phase = torch.linspace(0, 6, frames)
    batch["bad"][..., 4] = .02 * phase.sin()
    return cfg, batch


def final_report(*, endpoint_failures=50):
    remaining = endpoint_failures
    metrics = {}
    for split in ("seen", "new_position"):
        single, cross = [], []
        for role, rows in (("single_recording", single), ("cross_event", cross)):
            for index in range(16):
                endpoint_failed = remaining > 0
                remaining -= endpoint_failed
                observable = {
                    "temporal_accepted": False,
                    "endpoint_accepted": not endpoint_failed,
                    "reference_fidelity_accepted": True,
                }
                row = {"observable": observable, "width": 10 if index % 2 == 0 else 28}
                if role == "single_recording":
                    observable["physical_non_regression"] = {"accepted": True}
                    row["clean_identity"] = {"accepted": True}
                else:
                    row["safety"] = {"accepted": True}
                rows.append(row)
        metrics[split] = {"windows": single, "cross_event": {"windows": cross}}
    assert remaining == 0
    return {"metrics": metrics}


def synthetic_rows(cases=4):
    rows = []
    base = {
        "action_norm": 2.0, "gradient_norm": 1.0,
        "cosine_to_negative_gradient": .5, "directional_derivative": -1.0,
        "local_descent": True, "local_ascent": False, "local_flat": False,
        "exact_zero_action": False, "exact_zero_gradient": False,
    }
    objective = {
        "objective_value": 1.0,
        "spaces": {space: dict(base) for space in t.SPACES},
        "blocks": {block: {space: dict(base) for space in t.SPACES}
                   for block in t.GEOMETRY_BLOCKS},
        "gradient_outside_decoder_support": {"norm": 0.0, "exactly_zero": True},
    }
    for index in range(cases):
        rows.append({"split": "seen", "role": "single_recording", "width": 10,
                     "group": "single_recording/10", "case_index": index,
                     "active_geometry_fraction": .25,
                     "zero_origin": {name: objective for name in t.OBJECTIVES},
                     "current_output": {name: objective for name in t.OBJECTIVES}})
    return rows


def test_schema_and_reviewed_baseline_are_pinned():
    assert t.SCHEMA == "refiner_temporal_action_alignment_audit_v1"
    assert t.REVIEWED_MAIN_BASELINE == "2dc0fcb9606cda7ad44fc3ae9fea15ef13fca1e6"
    assert t.TRAJECTORY_COMMIT == "b2d71e1fa92cb2a6723810060722c0edea7a3a99"


def test_layout_excludes_contact_and_partitions_all_75_coordinates():
    blocks = t.GEOMETRY_BLOCKS
    assert blocks["root_translation"] == (0, 1, 2)
    flattened = [value for indices in blocks.values() for value in indices]
    assert len(flattened) == len(set(flattened)) == 75
    assert set(flattened) == set(range(75))


def test_extremity_block_uses_actual_canonical_joint_indices():
    assert tuple(t.EXTREMITY_JOINTS) == (7, 8, 10, 11, 20, 21, 22, 23)
    assert len(t.GEOMETRY_BLOCKS["extremity_joints"]) == 24
    assert len(t.GEOMETRY_BLOCKS["body_joints"]) == 48


@pytest.mark.parametrize(
    "gradient, cosine, directional, descent, ascent",
    [([-1.0, 0.0], 1.0, -1.0, True, False),
     ([1.0, 0.0], -1.0, 1.0, False, True),
     ([0.0, 1.0], 0.0, 0.0, False, False)],
)
def test_alignment_direction_signs(gradient, cosine, directional, descent, ascent):
    result = t.alignment_stats(torch.tensor([[1.0, 0.0]]), torch.tensor([gradient]))[0]
    assert result["cosine_to_negative_gradient"] == pytest.approx(cosine)
    assert result["directional_derivative"] == pytest.approx(directional)
    assert result["local_descent"] is descent and result["local_ascent"] is ascent


@pytest.mark.parametrize("zero", ["action", "gradient"])
def test_zero_vector_cosine_is_null(zero):
    action, gradient = torch.ones(1, 2), torch.ones(1, 2)
    (action if zero == "action" else gradient).zero_()
    result = t.alignment_stats(action, gradient)[0]
    assert result["cosine_to_negative_gradient"] is None
    assert result[f"exact_zero_{zero}"]


def test_nonfinite_alignment_fails_closed():
    with pytest.raises(FloatingPointError, match="nonfinite"):
        t.alignment_stats(torch.tensor([[float("nan")]]), torch.ones(1, 1))


@pytest.mark.parametrize("space", t.SPACES)
def test_space_vectors_have_the_declared_mask_semantics(space):
    action = torch.tensor([[[2.0, 3.0] + [0.0] * 73]])
    gradient = torch.tensor([[[5.0, 7.0] + [0.0] * 73]])
    mask = torch.tensor([[[.25, 0.0] + [0.0] * 73]])
    a_value, g_value = t._space_vectors(action, gradient, mask, space)
    if space == "raw_all_geometry":
        assert a_value[0, 0, :2].tolist() == [2.0, 3.0]
        assert g_value[0, 0, :2].tolist() == [5.0, 7.0]
    elif space == "raw_supported_geometry":
        assert a_value[0, 0, :2].tolist() == [2.0, 0.0]
        assert g_value[0, 0, :2].tolist() == [5.0, 0.0]
    else:
        assert a_value[0, 0, :2].tolist() == [.5, 0.0]
        assert g_value[0, 0, :2].tolist() == [1.25, 0.0]


def test_outside_support_gradient_is_measured_separately():
    action = torch.ones(1, 1, 75)
    gradient = torch.zeros_like(action)
    gradient[..., 1] = 4
    mask = torch.zeros_like(action)
    mask[..., 0] = .5
    row = t._objective_alignment(action, gradient, mask, torch.ones(1))[0]
    assert row["gradient_outside_decoder_support"]["norm"] == 4
    assert not row["gradient_outside_decoder_support"]["exactly_zero"]


def test_effective_mask_has_75_geometry_channels_and_excludes_contact():
    cfg, batch = small_batch(cases=2)
    mask = t._effective_geometry_mask(batch, cfg)
    assert mask.shape == (2, cfg.window_len, 75)
    assert torch.equal(mask[..., :3], m._refiner_decode_masks(
        batch["joint"], batch["root"], batch["contact"], batch["seam"], cfg)[1].expand(-1, -1, 3))


def test_zero_origin_uses_exact_zero_action_and_identity_decoder():
    cfg, batch = small_batch(cases=1)
    action = torch.randn(1, cfg.window_len, 75)
    result = t.zero_origin_point(action, batch, cfg)
    assert result["raw_geometry_origin_exact_zero"]
    assert result["decoded_origin_exact_identity"]
    assert set(result["gradients"]) == set(t.OBJECTIVES)


def test_current_output_uses_one_production_forward_and_has_strict_parity():
    cfg, batch = small_batch(cases=1)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-4)
    result = t.production_current_point(model, batch, cfg)
    assert result["production_forward_parity"]["verified"]
    assert result["production_forward_parity"]["model_forward_calls"] == 1
    assert result["action"].shape == (1, cfg.window_len, 75)
    assert all(value.shape == result["action"].shape for value in result["gradients"].values())


def test_audit_batch_restores_modes_grads_hooks_and_state():
    cfg, batch = small_batch(cases=1)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-4)
    model.train()
    model.in_proj.weight.grad = torch.ones_like(model.in_proj.weight)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    hooks = len(model.out._forward_hooks)
    metadata = [{"split": "train", "role": "single", "width": 10,
                 "group": "single_short", "case_index": 0}]
    from training import refiner_final_failure_audit as failure
    with failure.preserve_model_runtime(model):
        result = t.audit_batch(model, batch, cfg, metadata)
    assert model.training and len(model.out._forward_hooks) == hooks
    assert torch.equal(model.in_proj.weight.grad, torch.ones_like(model.in_proj.weight))
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert len(result["rows"]) == 1


def test_train_metadata_requires_exact_192_and_48_per_group():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = bank_tensor(cfg)
    batch = {key: torch.cat([value] * 6) for key, value in part.items()}
    rows = t.train_metadata(batch)
    assert len(rows) == 192
    assert all(sum(row["group"] == group for row in rows) == 48 for group in m.REFINER_GROUP_LABELS)
    with pytest.raises(ValueError, match="48-case"):
        t.train_metadata({**batch, "group": batch["group"][:-1]})


def test_combine_final_banks_has_fixed_64_and_eight_per_cell():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = {key: value[:16] for key, value in bank_tensor(cfg).items() if key != "group"}
    banks = {(split, role): part for split in ("seen", "new_position")
             for role in ("single_recording", "cross_event")}
    batch, rows = t.combine_final_banks(banks)
    assert batch["clean"].shape[0] == len(rows) == 64
    cells = {(split, role, width): sum(row["split"] == split and row["role"] == role
                                      and row["width"] == width for row in rows)
             for split in ("seen", "new_position")
             for role in ("single_recording", "cross_event") for width in (10, 28)}
    assert set(cells.values()) == {8}


def test_combine_final_banks_rejects_bad_width_or_count():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = {key: value[:15] for key, value in bank_tensor(cfg).items() if key != "group"}
    banks = {(split, role): part for split in ("seen", "new_position")
             for role in ("single_recording", "cross_event")}
    with pytest.raises(ValueError, match="16 cases"):
        t.combine_final_banks(banks)


def test_fixed_final_failure_facts_are_verified_exactly():
    result = t.validate_confirmed_final_failures(final_report())
    assert result["temporal_failed"] == 64 and result["endpoint_failed"] == 50
    assert result["physical_failed"] == result["reference_fidelity_failed"] == 0
    assert result["clean_identity_cases"] == 32


def test_changed_fixed_final_failure_facts_fail_closed():
    with pytest.raises(RuntimeError, match="facts changed"):
        t.validate_confirmed_final_failures(final_report(endpoint_failures=49))


def test_summary_preserves_groups_spaces_blocks_and_sign_counts():
    result = t.summarize(synthetic_rows(), ("split", "group"))
    group = result["seen/single_recording/10"]
    metric = group["points"]["zero_origin"]["temporal"]
    assert group["cases"] == 4
    assert metric["spaces"]["raw_supported_geometry"]["positive_cosine_cases"] == 4
    assert set(metric["blocks"]) == set(t.GEOMETRY_BLOCKS)
    assert metric["soft_masked_cosine_median"] == .5


def test_scientific_answers_are_automatic_but_do_not_recommend_architecture():
    rows = synthetic_rows()
    answers = t.scientific_answers(rows, rows)
    assert answers["train_temporal_gradient_present"]["answer"]
    assert answers["final_model_action_vs_zero_origin_temporal_descent"]["answer"].startswith("mostly_aligned")
    assert "architecture_recommendation" not in answers
    assert "multi_head_recommendation" not in answers
    assert "does not" in answers["interpretation_boundary"]


@pytest.mark.parametrize("objective_name", t.OBJECTIVES)
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_fixed_h_synthetic_finite_difference_matches_autograd(objective_name, alpha):
    coefficient = 2.0 if objective_name == "temporal" else 3.0
    action = torch.tensor([[.5, -1.0]], dtype=torch.float64)
    objective = lambda value: coefficient * value.square().sum()
    result = t.finite_difference_directional_check(objective, action, alpha)
    assert result["h"] == 1e-3 and result["alpha"] == alpha
    assert result["autograd"] == pytest.approx(result["finite_difference"], abs=1e-10)


def test_finite_difference_step_cannot_be_tuned_or_probe_selected():
    with pytest.raises(ValueError, match="fixed"):
        t.finite_difference_directional_check(lambda value: value.square().sum(),
                                              torch.ones(1), 0.0, h=1e-2)


def test_source_contains_no_optimizer_or_update_api():
    source = inspect.getsource(t)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "optimizer_steps\": 0" in source
    assert "parameter_update_performed\": False" in source
    assert "checkpoint_selection\": False" in source


def test_output_report_is_create_only(tmp_path):
    output = tmp_path / "report.json"
    from training import refiner_final_failure_audit as failure
    failure._exclusive_json(output, {"schema": t.SCHEMA})
    with pytest.raises(FileExistsError):
        failure._exclusive_json(output, {"schema": t.SCHEMA})


def test_cli_requires_runtime_commit_contract():
    signature = inspect.signature(t.run)
    assert list(signature.parameters) == ["args"]
    parser_source = inspect.getsource(t.main)
    assert "--expected-main-commit" in parser_source
    assert "required=True" in parser_source


def test_cpu_synthetic_alignment_is_finite():
    action = torch.randn(3, 4, 75)
    gradient = torch.randn_like(action)
    result = t.alignment_stats(action, gradient)
    assert len(result) == 3
    assert all(math_value == math_value for row in result for math_value in
               (row["action_norm"], row["gradient_norm"], row["directional_derivative"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_alignment_matches_cpu():
    action = torch.randn(2, 3, 75)
    gradient = torch.randn_like(action)
    cpu, cuda = t.alignment_stats(action, gradient), t.alignment_stats(action.cuda(), gradient.cuda())
    for left, right in zip(cpu, cuda):
        for key in ("action_norm", "gradient_norm", "cosine_to_negative_gradient",
                    "directional_derivative"):
            assert left[key] == pytest.approx(right[key], rel=1e-12, abs=1e-12)
        assert {key: value for key, value in left.items() if key not in {
            "action_norm", "gradient_norm", "cosine_to_negative_gradient", "directional_derivative"
        }} == {key: value for key, value in right.items() if key not in {
            "action_norm", "gradient_norm", "cosine_to_negative_gradient", "directional_derivative"
        }}
