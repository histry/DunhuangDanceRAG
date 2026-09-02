import inspect
from pathlib import Path

import pytest
import torch

from training import refiner_cross_width_normalization_audit as a


def test_schema_baseline_and_primary_contract():
    assert a.SCHEMA == "refiner_cross_width_normalization_audit_v1"
    assert a.FROZEN_ARTIFACT_COMMIT == "a9fbff524e46b0e13ab5e902f09c608e43cfb40f"
    assert a.PARENT_COMMIT == "a33b17a78909bdf7125aa690d672f3991b7e5867"
    assert a.PRIMARY_CASES == 32
    assert a.GROUP_ORDER == (
        "seen/cross_event/10", "seen/cross_event/28",
        "new_position/cross_event/10", "new_position/cross_event/28",
    )


def test_temporal_partition_is_ordered_and_has_no_width_parameter():
    active = [2, 4, 5, 9, 12, 20, 21, 25, 30, 41]
    assert a.temporal_partition(active) == {
        "early": [2, 4, 5, 9], "center": [12, 20, 21], "late": [25, 30, 41]
    }
    assert "width" not in inspect.signature(a.temporal_partition).parameters
    with pytest.raises(ValueError, match="active_frame_count"):
        a.temporal_partition([])


def test_effective_weight_count_and_zero_denominator():
    assert a.effective_weight_count(torch.tensor([1.0, 2.0])) == pytest.approx(9.0 / 5.0)
    assert a.effective_weight_count(torch.zeros(3)) is None


def test_authoritative_temporal_reduction_reconstructs_boundary_metric():
    joints = torch.zeros(1, 8, 24, 3, dtype=torch.float64)
    joints[0, :, 0, 0] = torch.arange(8, dtype=torch.float64) ** 2
    seam = torch.zeros(1, 8, 1)
    seam[:, 2:5] = 1
    result = a.temporal_reduction_parity(joints, seam, 30.0)
    assert result["verified"]
    assert result["max_abs_error"] == pytest.approx(0.0)


def test_normalization_inventory_is_explicit_and_has_no_width_dependency():
    inventory = a.normalization_inventory()
    assert "sum(v2 * S2)" in inventory["temporal_metric"]["symbolic_formula"]
    assert inventory["temporal_metric"]["active_frame_count_used"] is False
    assert inventory["temporal_metric"]["valid_derivative_sample_count_used"] is True
    assert inventory["decoder_width_dependency"]["width_explicitly_used_in_objective"] is False
    assert inventory["gate"]["width_dependent"] is False


def test_temporal_distribution_preserves_authoritative_mass():
    joints = torch.zeros(1, 8, 24, 3, dtype=torch.float64)
    joints[0, :, 0, 0] = torch.arange(8, dtype=torch.float64) ** 2
    seam = torch.zeros(1, 8, 1)
    seam[:, 2:5] = 1
    components = a.authoritative_temporal_components(joints, seam, 30.0)
    distribution = a._temporal_distribution(components, [2, 3, 4], 8)
    assert distribution["total_mass"] == pytest.approx(float(components["temporal_energy"][0]))
    assert set(distribution["thirds"]) == {"early", "center", "late"}


def test_repair_distribution_signs_and_zero_denominator_efficiency():
    left = {"aligned_stencil_start_contribution": [2.0, 1.0], "thirds": {"early": 2.0, "center": 1.0, "late": 0.0}}
    right = {"aligned_stencil_start_contribution": [1.0, 3.0], "thirds": {"early": 1.0, "center": 3.0, "late": 0.0}}
    result = a._repair_distribution(left, right, 4)
    assert result["repair_total"] == pytest.approx(-1.0)
    assert result["positive_repair_mass"] == pytest.approx(1.0)
    assert result["negative_repair_mass"] == pytest.approx(-2.0)
    assert a._ratio(1.0, 0.0) is None


def _synthetic_row(split, width, value):
    return {
        "split": split, "role": "cross_event", "width": width,
        "bank_case_index": value, "temporal_deficit_base": 2.0,
        "temporal_deficit_rcsp": 1.0, "repair_gain": 1.0,
        "gate_pass_base": False, "gate_pass_rcsp": width == 10,
        "gate_margin_base": -0.2 if width == 28 else -0.1,
        "gate_margin_rcsp": -0.1 if width == 28 else 0.1,
        "raw_temporal_numerator": {"seam_acceleration": {"BASE": 10.0, "RCSP": 8.0}, "seam_jerk": {"BASE": 20.0, "RCSP": 16.0}},
        "temporal_denominator": {"seam_acceleration": {"BASE": 3.0 + width / 10, "RCSP": 3.0 + width / 10}, "seam_jerk": {"BASE": 4.0 + width / 10, "RCSP": 4.0 + width / 10}},
        "normalized_temporal_subterms": {"seam_acceleration": {"BASE": 2.0, "RCSP": 1.5}, "seam_jerk": {"BASE": 3.0, "RCSP": 2.0}},
        "active_statistics": {"binary_active_frame_count": width, "binary_active_coordinate_count": width * 3},
        "effective_weight_statistics": {"total_weight_sum": width * 2.0, "effective_weight_count": width},
        "support_retention_ratio": 0.7, "adapter_direction_cosine": 0.2 if width == 10 else 0.1,
        "total_direction_cosine": 0.2,
        "finite_action_efficiency": {"RCSP": {"final_tangent": {"G_over_action_norm": 1.0, "G_over_action_energy": 1.0}}},
        "decoder_stage_statistics": {"RCSP": {"final_tangent": {"l2_norm": 1.0, "energy_sum": 1.0}}},
        "BASE": {"temporal_error_distribution": {"stats": {"effective_temporal_support": 2.0}}},
        "RCSP": {"temporal_error_distribution": {"stats": {"effective_temporal_support": 3.0}}},
        "temporal_repair_distribution": {"repair_effective_support": 3.0},
    }


def test_group_summary_requires_four_cross_groups_and_eight_cases():
    rows = []
    for split in ("seen", "new_position"):
        for width in (10, 28):
            rows.extend(_synthetic_row(split, width, index) for index in range(8))
    summary = a.cross_group_summary(rows)
    assert set(summary) == set(a.GROUP_ORDER)
    assert all(row["cases"] == 8 for row in summary.values())
    assert summary["seen/cross_event/10"]["temporal_pass_count_rcsp"] == 8


def test_json_is_create_only():
    path = Path(__file__).parents[1] / ".phase2_create_only_test.json"
    try:
        a._exclusive_json(path, {"schema": a.SCHEMA})
        with pytest.raises(FileExistsError):
            a._exclusive_json(path, {"schema": a.SCHEMA})
    finally:
        if path.exists():
            path.unlink()


def test_shell_contract_is_read_only_and_exact_main():
    source = Path(__file__).parents[1] / "scripts" / "audit_refiner_cross_width_normalization.sh"
    text = source.read_text(encoding="utf-8")
    assert "git status --porcelain" in text
    assert "git rev-parse origin/main" in text
    assert "a9fbff524e46b0e13ab5e902f09c608e43cfb40f" in text
    assert "No optimizer" in text
    assert "Pilot remains forbidden" in text


def test_source_has_no_update_or_selection_controls():
    source = Path(a.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert '"optimizer_steps": 0' in source
    assert '"parameter_update_performed": False' in source
    assert '"production_model_modified": False' in source
    assert '"pilot_allowed": False' in source


def test_pairing_contract_is_unpaired_by_default():
    assert a.ZERO_DENOMINATOR is None
    assert "pair_key" in Path(a.__file__).read_text(encoding="utf-8")
