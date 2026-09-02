import inspect
import json
from pathlib import Path

import pytest
import torch

from training import refiner_single_direction_decomposition_audit as d
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from tests.test_refiner_role_conditioned_support_projection import nonzero_base, tiny_batch


def block_row(signed=1.0, cosine=0.5):
    row = {"gradient_norm": 2.0}
    for action in d.ACTION_NAMES:
        row.update(
            {
                f"{action}_action_norm": 1.0,
                f"{action}_cosine": cosine if signed != 0 else None,
                f"{action}_signed_dot": signed,
                f"{action}_positive_contribution_sum": max(signed, 0.0),
                f"{action}_negative_contribution_sum": min(signed, 0.0),
                f"{action}_absolute_contribution_sum": abs(signed),
                f"{action}_cancellation_ratio": 0.0 if signed != 0 else None,
            }
        )
    return row


def synthetic_rows():
    rows = []
    for split, role in rcsp.FINAL_BLOCK_ORDER:
        for width in (10, 28):
            for case_index in range(8):
                anatomy_time = {
                    f"{anatomy}/{time}": block_row()
                    for anatomy in d.ANATOMY_NAMES
                    for time in d.TIME_NAMES
                }
                anatomy = {name: block_row() for name in d.ANATOMY_NAMES}
                temporal = {name: block_row() for name in d.TIME_NAMES}
                rows.append(
                    {
                        "split": split,
                        "role": role,
                        "width": width,
                        "case_index": case_index,
                        "bank_case_index": case_index,
                        "whole": block_row(),
                        "anatomy": anatomy,
                        "temporal": temporal,
                        "anatomy_time": anatomy_time,
                    }
                )
    return rows


def test_schema_reviewed_parent_and_fixed_case_contract():
    assert d.SCHEMA == "refiner_single_direction_decomposition_audit_v1"
    assert d.REVIEWED_MAIN_BASELINE == "c2ceea1bfc449e51f697577ba2ec2dce9a70d699"
    assert d.FINAL_CASES == 64
    assert d.FINAL_CHUNK_SIZE == 8


def test_authoritative_anatomy_partition_is_complete_and_disjoint():
    partition = d.ANATOMY_PARTITION
    assert partition["partition_source"].startswith("motion_geometry.physical.EXTREMITY_JOINTS")
    assert partition["num_joints"] == 24
    assert partition["root_coordinate_indices"] == [0, 1, 2]
    assert partition["extremity_joint_indices"] == [7, 8, 10, 11, 20, 21, 22, 23]
    assert partition["body_joint_indices"] == [
        0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19
    ]
    coordinates = [
        coordinate
        for name in d.ANATOMY_NAMES
        for coordinate in partition["coordinates"][name]
    ]
    assert len(coordinates) == 75
    assert set(coordinates) == set(range(75))


def test_active_indices_are_split_deterministically_without_width_input():
    active = [2, 4, 5, 9, 12, 20, 21, 25, 30, 41]
    result = d.temporal_partition(active)
    assert result == {
        "early": [2, 4, 5, 9],
        "center": [12, 20, 21],
        "late": [25, 30, 41],
    }
    assert [value for name in d.TIME_NAMES for value in result[name]] == active
    assert "width" not in inspect.signature(d.temporal_partition).parameters
    with pytest.raises(ValueError, match="active_frame_count"):
        d.temporal_partition([])


def test_contribution_stats_signed_parts_cosine_and_cancellation_are_exact():
    action = torch.tensor([1.0, -1.0, 2.0])
    gradient = torch.tensor([-2.0, -3.0, 1.0])
    result = d.contribution_stats(action, gradient)
    # action * -gradient = [2, -3, -2]
    assert result["positive_contribution_sum"] == pytest.approx(2.0)
    assert result["negative_contribution_sum"] == pytest.approx(-5.0)
    assert result["signed_dot"] == pytest.approx(-3.0)
    assert result["absolute_contribution_sum"] == pytest.approx(7.0)
    assert result["cancellation_ratio"] == pytest.approx(1 - 3 / 7)


def test_zero_action_or_gradient_has_null_cosine_and_zero_absolute_has_null_cancellation():
    zero_action = d.contribution_stats(torch.zeros(3), torch.ones(3))
    zero_gradient = d.contribution_stats(torch.ones(3), torch.zeros(3))
    assert zero_action["cosine_to_negative_gradient"] is None
    assert zero_action["cancellation_ratio"] is None
    assert zero_gradient["cosine_to_negative_gradient"] is None
    assert zero_gradient["cancellation_ratio"] is None


def test_case_decomposition_uses_binary_support_and_builds_all_nine_blocks():
    frames = 12
    support = torch.zeros(frames, 75)
    support[[1, 3, 4, 7, 9, 11], :3] = 1
    gradient = torch.zeros(frames, 75)
    gradient[support.bool()] = -1
    actions = {name: support.clone() for name in d.ACTION_NAMES}
    row = d.decompose_case(
        actions,
        gradient,
        support,
        {"split": "seen", "role": "single_recording", "width": 10},
    )
    assert row["active_frame_indices"] == [1, 3, 4, 7, 9, 11]
    assert row["first_active_frame"] == 1
    assert row["last_active_frame"] == 11
    assert row["active_frame_count"] == 6
    assert row["temporal_frame_indices"] == {
        "early": [1, 3], "center": [4, 7], "late": [9, 11]
    }
    assert len(row["anatomy_time"]) == 9
    assert set(row["anatomy_time"]) == {
        f"{anatomy}/{time}" for anatomy in d.ANATOMY_NAMES for time in d.TIME_NAMES
    }
    assert row["gradient_outside_binary_support_max"] == 0


def test_gradient_outside_binary_support_fails_closed():
    support = torch.zeros(3, 75)
    support[1, 0] = 1
    gradient = torch.zeros_like(support)
    gradient[0, 0] = 1e-5
    actions = {name: torch.zeros_like(support) for name in d.ACTION_NAMES}
    with pytest.raises(RuntimeError, match="escaped decoder support"):
        d.decompose_case(actions, gradient, support, {})


def test_summary_has_64_cases_32_per_role_and_eight_per_group():
    scopes, summary, anatomy, temporal, anatomy_time = d.build_summaries(synthetic_rows())
    assert summary["overall"]["cases"] == 64
    assert summary["role:single_recording"]["cases"] == 32
    assert summary["role:cross_event"]["cases"] == 32
    assert all(len(values) == 8 for name, values in scopes.items() if name.startswith("group:"))
    assert len(scopes["group:new_position/single_recording/28"]) == 8
    assert set(anatomy) == set(d.ANATOMY_NAMES)
    assert set(temporal) == set(d.TIME_NAMES)
    assert len(anatomy_time) == 9


def test_summary_records_defined_positive_negative_and_undefined_cases():
    result = d.summarize_blocks([block_row(1), block_row(-1), block_row(0, None)])
    adapter = result["actions"]["adapter"]
    assert adapter["defined_cosine_cases"] == 2
    assert adapter["positive_signed_dot_cases"] == 1
    assert adapter["negative_signed_dot_cases"] == 1
    assert adapter["zero_or_undefined_cases"] == 1


def test_source_comparison_is_unpaired_and_detects_majority_sign_shift():
    rows = synthetic_rows()
    for row in rows:
        if row["role"] == "single_recording" and row["width"] == 10:
            sign = 1.0 if row["split"] == "seen" else -1.0
            row["anatomy_time"]["root/early"] = block_row(sign)
    scopes, *_ = d.build_summaries(rows)
    result = d.source_conditioned_comparison(scopes)
    comparison = result["comparison"]["10"]["root/early"]
    assert comparison["stable_sign_shift"]
    assert comparison["left_negative_cases"] == 0
    assert comparison["right_negative_cases"] == 8
    assert result["case_pairing_performed"] is False


def test_width_comparison_is_direction_only_and_never_changes_normalization():
    rows = synthetic_rows()
    for row in rows:
        if row["split"] == "seen" and row["role"] == "single_recording":
            row["anatomy_time"]["body/late"] = block_row(1 if row["width"] == 10 else -1)
    scopes, *_ = d.build_summaries(rows)
    result = d.width_conditioned_comparison(scopes)
    assert result["comparison"]["seen"]["body/late"]["stable_sign_shift"]
    assert result["normalization_or_threshold_investigated"] is False
    assert result["case_pairing_performed"] is False


def parameter_report(rows):
    parameter_rows = []
    for source in ("train_transaction_0", "seen", "new_position"):
        for width in (10, 28):
            parameter_rows.append(
                {
                    "key": f"{source}/single_recording/{width}",
                    "role": "single_recording",
                    "parameter_gradient_norm": 2.0,
                    "learned_displacement_vs_negative_gradient_cosine": 0.01,
                }
            )
    return {"parameter_gradient_rows": parameter_rows, "rows": rows}


def test_parameter_bridge_keeps_train_parameter_only_and_adds_no_fake_correlation():
    rows = synthetic_rows()
    scopes, *_ = d.build_summaries(rows)
    result = d.parameter_to_action_bridge(parameter_report(rows), scopes)
    assert len(result["rows"]) == 6
    train = [row for row in result["rows"] if row["condition"].startswith("train_")]
    final = [row for row in result["rows"] if not row["condition"].startswith("train_")]
    assert all(row["parameter_evidence_only"] for row in train)
    assert all("whole_adapter_cosine" not in row for row in train)
    assert all("whole_adapter_cosine" in row for row in final)
    assert result["paired_correlation_computed"] is False
    assert result["case_selection_performed"] is False


def test_scientific_classification_uses_sign_and_majority_case_consistency():
    rows = synthetic_rows()
    for row in rows:
        if row["role"] == "single_recording":
            row["anatomy_time"]["extremity/late"] = block_row(-1)
            row["anatomy"]["extremity"] = block_row(-1)
    scopes, summary, anatomy, temporal, anatomy_time = d.build_summaries(rows)
    source = d.source_conditioned_comparison(scopes)
    width = d.width_conditioned_comparison(scopes)
    new_rows = scopes["group:new_position/single_recording/28"]
    answers = d.scientific_answers(
        summary,
        anatomy,
        temporal,
        anatomy_time,
        source,
        width,
        {
            "rows": new_rows,
            "all_single_rows": scopes["role:single_recording"],
            "all_cross_rows": scopes["role:cross_event"],
        },
        True,
    )
    assert answers["anatomical_cancellation_observed"]
    assert answers["anatomical_stable_signs_by_action"]["adapter"] == {
        "root": 1,
        "body": 1,
        "extremity": -1,
    }
    assert answers["localized_negative_block_observed"]
    assert answers["localized_negative_blocks_by_action"]["adapter"]["extremity/late"]
    assert answers["localized_negative_blocks_by_action"]["total"]["extremity/late"]
    assert answers["new_position_single_28_localized_ascent"]
    assert "SINGLE_LOCALIZED_DIRECTION_MISMATCH" in answers["supported_descriptive_mechanisms"]
    assert answers["single_direction_decomposition"] == "MULTIPLE_MECHANISMS_SUPPORTED"
    assert "cannot prove" in answers["claim_boundary"]


def test_adapter_checkpoint_path_must_come_from_report_and_remain_in_rcsp_dir(tmp_path):
    checkpoint = tmp_path / "custom_adapter_name.pt"
    checkpoint.write_bytes(b"adapter")
    report = {
        "parameter_update_scope": {
            "adapter_checkpoint": {
                "path": str(checkpoint),
                "sha256": d._file_sha256(checkpoint),
            }
        }
    }
    assert d._adapter_path_from_report(tmp_path, report) == checkpoint.resolve()
    outside = tmp_path.parent / "outside.pt"
    outside.write_bytes(b"outside")
    report["parameter_update_scope"]["adapter_checkpoint"] = {
        "path": str(outside), "sha256": d._file_sha256(outside)
    }
    with pytest.raises(ValueError, match="outside"):
        d._adapter_path_from_report(tmp_path, report)


def test_adapter_checkpoint_must_remain_diagnostic_only():
    checkpoint = {
        "schema": d.parameter_audit.RCSP_SOURCE_SCHEMA,
        "completed_steps": d.rcsp.STEPS,
        "formal_checkpoint": False,
        "production_model_modified": False,
        "checkpoint_selection_performed": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "resume_allowed": False,
    }
    assert d._validate_adapter_checkpoint(checkpoint) is checkpoint
    checkpoint["resume_allowed"] = True
    with pytest.raises(ValueError, match="diagnostic-only"):
        d._validate_adapter_checkpoint(checkpoint)


def test_parameter_attribution_lineage_is_fixed_to_c2_baseline(tmp_path):
    path = tmp_path / "parameter_report.json"
    report = {
        "schema": d.PARAMETER_ATTRIBUTION_SCHEMA,
        "completed": True,
        "provenance": {
            "runtime_commit": d.REVIEWED_MAIN_BASELINE,
            "rcsp_commit": d.RCSP_SOURCE_COMMIT,
            "rcsp_sha256": {"report.json": "rcsp-hash"},
        },
        "optimizer_steps": 0,
        "gradient_protocol": {"parameter_update_performed": False},
        "production_model_modified": False,
        "pilot_allowed": False,
        "parameter_gradient_rows": [{}] * 12,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = d._validate_parameter_attribution(path, "rcsp-hash")
    assert loaded[2]["provenance"]["runtime_commit"] == d.REVIEWED_MAIN_BASELINE
    report["optimizer_steps"] = 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="read-only contract"):
        d._validate_parameter_attribution(path, "rcsp-hash")


def complete_fixed_batch():
    cfg, source = tiny_batch(cases=4)
    indices = []
    metadata = []
    mapping = {
        ("single_recording", 10): 0,
        ("single_recording", 28): 1,
        ("cross_event", 10): 2,
        ("cross_event", 28): 3,
    }
    for split, role in rcsp.FINAL_BLOCK_ORDER:
        for width in (10, 28):
            indices.extend([mapping[(role, width)]] * 8)
            metadata.extend(
                {
                    "split": split,
                    "role": role,
                    "width": width,
                    "case_index": case_index,
                    "bank_case_index": case_index,
                }
                for case_index in range(8)
            )
    index = torch.tensor(indices)
    batch = {key: value.index_select(0, index) for key, value in source.items() if key != "group"}
    return cfg, batch, metadata


def test_full_64_evaluation_reuses_authoritative_action_gradient_and_keeps_parameters_frozen():
    cfg, batch, metadata = complete_fixed_batch()
    model = rcsp.FrozenBaseRCSPModel(nonzero_base())
    with torch.no_grad():
        model.adapter.single_adapter.bias.fill_(1e-4)
        model.adapter.cross_adapter.bias.fill_(2e-4)
    source_direction = rcsp.direction_alignment(model, batch, metadata, cfg)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    base_hash = d.safe.state_hash(model.base.state_dict())
    adapter_hash = d.safe.state_hash(model.adapter.state_dict())
    expected_bank_index = source_direction["case_level"][0]["bank_case_index"]
    source_direction["case_level"][0]["bank_case_index"] = -1
    with pytest.raises(RuntimeError, match="row identity mismatch"):
        d.evaluate_decomposition(
            model,
            batch,
            metadata,
            cfg,
            {"direction_alignment": source_direction},
        )
    source_direction["case_level"][0]["bank_case_index"] = expected_bank_index
    result = d.evaluate_decomposition(
        model,
        batch,
        metadata,
        cfg,
        {"direction_alignment": source_direction},
    )
    assert len(result["case_level"]) == 64
    assert result["direction_parity"]["verified"]
    assert all(row["gradient_outside_binary_support_max"] == 0 for row in result["case_level"])
    assert d.safe.state_hash(model.base.state_dict()) == base_hash
    assert d.safe.state_hash(model.adapter.state_dict()) == adapter_hash
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cli_source_and_shell_forbid_training_selection_and_production_changes():
    cli = inspect.getsource(d.main)
    for forbidden in ("--steps", "--alpha", "--width", "--seed", "--resume"):
        assert forbidden not in cli
    source = Path(d.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"architecture_selection_performed": False' in source
    assert '"production_model_modified": False' in source
    assert '"pilot_allowed": False' in source
    for heading in (
        "SOURCE-CONDITIONED COMPARISON",
        "WIDTH-CONDITIONED DIRECTION COMPARISON",
        "NEW_POSITION/SINGLE/28 8 CASES",
        "PARAMETER-TO-ACTION BRIDGE",
    ):
        assert heading in source
    shell = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_refiner_single_direction_decomposition.sh"
    ).read_text(encoding="utf-8")
    assert "git status --porcelain" in shell
    assert "git rev-parse origin/main" in shell
    assert "No optimizer" in shell
    assert "Pilot remains forbidden" in shell
