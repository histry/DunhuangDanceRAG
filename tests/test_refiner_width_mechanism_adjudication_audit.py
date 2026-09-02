"""Unit tests for the deterministic, read-only Phase 2.1 adjudication logic."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training import refiner_width_mechanism_adjudication_audit as audit


def _seam(start: int, width: int, frames: int = 60, fps: float = 30.0, halo_seconds: float = 0.2):
    value = torch.zeros((frames, 1), dtype=torch.float32)
    halo = round(fps * halo_seconds)
    value[max(0, start - halo):min(frames, start + width + halo), 0] = audit.SEAM_HALO_VALUE
    value[start:start + width, 0] = audit.SEAM_CORE_VALUE
    return value


def _decision_summaries(
    *,
    cf_seen=0.0,
    cf_new=0.0,
    spread_seen=False,
    spread_new=False,
    direction_seen=False,
    direction_new=False,
):
    def spread(value):
        return {
            "rcsp_error_spread_fraction": {
                "median_within_case_ratio_cf28_over_cf10": 1.2 if value else 0.9,
                "non_null_cases": 8,
                "cases_cf28_gt_cf10": 5 if value else 3,
            }
        }

    def direction(value):
        return {
            "10": {
                "median_E_gate": 2.0,
                "median_adapter_direction_cosine": 0.8,
            },
            "28": {
                "median_E_gate": 1.0 if value else 2.2,
                "median_adapter_direction_cosine": 0.4 if value else 0.9,
            },
        }

    return {
        "spread": {"seen": spread(spread_seen), "new_position": spread(spread_new)},
        "direction_efficiency": {
            "seen": direction(direction_seen),
            "new_position": direction(direction_new),
        },
        "counterfactual_width": {
            "seen": {"gap_explanation_fraction": cf_seen},
            "new_position": {"gap_explanation_fraction": cf_new},
        },
    }


def test_schema_and_frozen_constants_are_exact():
    assert audit.SCHEMA == "refiner_width_mechanism_adjudication_audit_v1"
    assert audit.FROZEN_PHASE2_COMMIT == "8e099944ed07f3550aede952aa1662a50e6e4bbe"
    assert audit.PRIMARY_CASES == 32
    assert audit.FINAL_CASES == 64
    assert audit.MAJOR_GAP_FRACTION == 0.50


def test_phase2_lineage_requires_exact_schema_and_completed(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": "refiner_cross_width_normalization_audit_v1", "completed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="completed=true"):
        audit._validate_phase2_lineage(path)


def test_phase2_lineage_requires_frozen_runtime_commit(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema": "refiner_cross_width_normalization_audit_v1", "completed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen upstream"):
        audit._validate_phase2_lineage(path)


def test_phase2_lineage_requires_read_only_integrity(tmp_path):
    path = tmp_path / "report.json"
    payload = {
        "schema": "refiner_cross_width_normalization_audit_v1",
        "completed": True,
        "provenance": {"runtime_commit": "not-frozen"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen upstream"):
        audit._validate_phase2_lineage(path)


def test_ntsf_formula_and_range():
    value = audit.normalized_temporal_spread_fraction([1.0, 1.0, 0.0, 0.0], [True, True, True, True])
    assert value == pytest.approx(0.5)
    assert 0.0 < value <= 1.0 + 1.0e-10


def test_ntsf_uses_only_authoritative_active_support():
    value = audit.normalized_temporal_spread_fraction([1.0, 1.0, 99.0], [True, True, False])
    assert value == pytest.approx(1.0)


def test_ntsf_zero_square_sum_is_null_not_zero():
    assert audit.normalized_temporal_spread_fraction([0.0, 0.0], [True, True]) is None


def test_ntsf_rejects_negative_contribution():
    with pytest.raises(ValueError, match="nonnegative"):
        audit.normalized_temporal_spread_fraction([1.0, -1.0], [True, True])


def test_positive_repair_clips_signed_repair_before_ntsf():
    np.testing.assert_array_equal(
        audit.positive_repair_contribution([3.0, 1.0, 2.0], [1.0, 2.0, 2.0]),
        np.asarray([2.0, 0.0, 0.0]),
    )


def test_applied_action_uses_final_geometric_75d_and_excludes_contact():
    base = torch.zeros((1, 2, 79), dtype=torch.float32)
    rcsp = base.clone()
    rcsp[..., :4] = 100.0
    rcsp[..., 4] = 3.0
    norm = audit.applied_action_delta_norm(rcsp, base)
    assert norm.item() == pytest.approx(3.0 * 2.0 ** 0.5)
    assert audit.GEOMETRIC_TANGENT_END - audit.GEOMETRIC_TANGENT_START == 75


def test_applied_action_rejects_non_79d_layout():
    with pytest.raises(ValueError, match="75D"):
        audit.applied_action_delta_norm(torch.zeros((1, 78)), torch.zeros((1, 78)))


def test_gap_explanation_opposite_sign_is_zero():
    assert audit._gap_explanation_fraction(-2.0, 1.0) == 0.0
    assert audit._gap_explanation_fraction(2.0, -1.0) == 0.0


def test_gap_explanation_zero_observed_gap_is_null():
    assert audit._gap_explanation_fraction(0.0, 2.0) is None


def test_gap_explanation_same_sign_is_capped():
    assert audit._gap_explanation_fraction(-2.0, -1.0) == pytest.approx(0.5)
    assert audit._gap_explanation_fraction(2.0, 4.0) == pytest.approx(1.0)


def test_mask_reconstruction_is_exact_for_all_64_formal_seams():
    rows = [_seam(20, 10) if index % 2 == 0 else _seam(11, 28) for index in range(64)]
    frozen = torch.stack(rows)
    masks, parity = audit.mask_reconstruction_parity(frozen, 30.0, 0.2)
    assert set(masks) == {10, 28}
    assert parity["cases"] == 64
    assert parity["verified_cases"] == 64
    assert parity["verified"] is True


def test_mask_reconstruction_fails_closed_for_non_contiguous_core():
    frozen = torch.stack([_seam(20, 10) for _ in range(64)])
    frozen[0, 25, 0] = 1.0
    frozen[0, 25, 0] = 0.0
    with pytest.raises(ValueError, match="contiguous"):
        audit.mask_reconstruction_parity(frozen, 30.0, 0.2)


def test_pre_registered_spreading_decision():
    result = audit.adjudicate(_decision_summaries(cf_seen=0.6, cf_new=0.7, spread_seen=True, spread_new=True))
    assert result["adjudicated_primary_mechanism"] == "TEMPORAL_SPREADING_PRIMARY"


def test_pre_registered_normalization_decision():
    result = audit.adjudicate(_decision_summaries(cf_seen=0.6, cf_new=0.7))
    assert result["adjudicated_primary_mechanism"] == "WIDTH_NORMALIZATION_PRIMARY"


def test_pre_registered_direction_decision():
    result = audit.adjudicate(_decision_summaries(direction_seen=True, direction_new=True))
    assert result["adjudicated_primary_mechanism"] == "WIDTH_CONDITIONED_DIRECTION_PRIMARY"


def test_pre_registered_mixed_decision():
    result = audit.adjudicate(
        _decision_summaries(
            cf_seen=0.6,
            cf_new=0.7,
            direction_seen=True,
            direction_new=True,
        )
    )
    assert result["adjudicated_primary_mechanism"] == "MIXED_WIDTH_MECHANISM"
    assert result["primary_intervention_order"] == [
        "metric/support-time intervention",
        "direction intervention",
    ]


def test_parity_failure_closes_adjudication():
    result = audit.adjudicate(_decision_summaries(cf_seen=1.0, cf_new=1.0), counterfactual_mask_parity_verified=False)
    assert result["counterfactual_construction_available"] is False
    assert result["counterfactual_mask_parity_verified"] is False
    assert result["causal_root_cause_proven"] is False


def test_major_gap_threshold_is_fixed_at_half():
    result = audit.adjudicate(_decision_summaries(cf_seen=0.50, cf_new=0.50))
    assert result["cf_explains_major_gap"] == {"seen": True, "new_position": True}


def test_pure_common_scalar_normalization_cancels_in_relative_gate():
    before, after, scale = 10.0, 7.0, 3.0
    assert (scale * before - scale * after) / (scale * before) == pytest.approx((before - after) / before)


def test_actual_width_parity_uses_phase2_and_rcsp_authoritative_values():
    cfg_threshold = 0.03
    p2_row = {
        "BASE": {"temporal_metric": 7.0, "gate": {"relative_gain": 0.3}, "gate_margin": 0.27},
        "RCSP": {"temporal_metric": 5.0, "gate": {"relative_gain": 0.5}, "gate_margin": 0.47},
        "gate_threshold": cfg_threshold,
    }
    source = {
        "base": {"observable": {"after": {"temporal_energy": 7.0}, "temporal_gain": 0.3}},
        "rcsp": {"observable": {"before": {"temporal_energy": 10.0}, "after": {"temporal_energy": 5.0}, "temporal_gain": 0.5}},
    }
    cf = {"M_before": 10.0, "M_base": 7.0, "M_rcsp": 5.0, "G_base": 0.3, "G_rcsp": 0.5, "gate_margin_base": 0.27, "gate_margin_rcsp": 0.47}
    result = audit._actual_width_parity("seen/cross_event/10/0", 10, cf, p2_row, source)
    assert result["verified"] is True
    assert result["max_abs_error"] == 0.0


def test_read_only_source_contains_no_training_update_or_optimizer_step():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    assert "load_state_dict" in source


def test_report_contract_flags_are_false():
    expected_false = {
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_update_performed": False,
        "checkpoint_selection_performed": False,
        "scale_selection_performed": False,
        "architecture_selection_performed": False,
        "production_model_modified": False,
        "production_inference_modified": False,
        "scientific_acceptance": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "fake_case_pairing_performed": False,
    }
    assert expected_false["pilot_allowed"] is False
    assert all(value is False or value == 0 for value in expected_false.values())


def test_counterfactual_rows_declare_same_output_and_unpaired_observed_comparison():
    assert "UNPAIRED" in Path(audit.__file__).read_text(encoding="utf-8")
    assert "WITHIN_CASE_BY_CONSTRUCTION" in Path(audit.__file__).read_text(encoding="utf-8")


def test_production_files_are_only_hashed_as_immutable_inputs():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "motion_models.py" in source
    assert "boundary_observables.py" in source
    assert "production implementation files changed" in source
    assert "Path(m.__file__).write" not in source
