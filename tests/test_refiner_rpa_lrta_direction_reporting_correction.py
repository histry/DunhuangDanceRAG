from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training import refiner_rpa_lrta_direction_reporting_correction as correction


def test_schema_parent_and_primary_space_are_frozen():
    assert correction.SCHEMA == (
        "refiner_rpa_lrta_direction_reporting_correction_v1"
    )
    assert correction.CORRECTION_PARENT_COMMIT == (
        "ca8c313bea7870b81cbe4e02ab6ba7c39741764d"
    )
    assert correction.PRIMARY_SPACE == "raw_all_geometry"


def test_formal_artifact_hashes_are_frozen():
    assert correction.EXPECTED_RPA_REPORT_SHA256 == (
        "08fd36d5bd504a16cb5f18348358e8e236008e0758481e7c5372dddca0c6808e"
    )
    assert correction.EXPECTED_RPA_ADAPTER_SHA256 == (
        "2b6a7ae7d08721bcff5b174403a7137ec7494c2f871a21ffc5a63bdc7be70110"
    )
    assert correction.EXPECTED_RPA_UPDATES_SHA256 == (
        "aedcf96068976ead5988d055af248b067e641849214365ba9fea3fdee35f0a86"
    )


def test_alignment_rows_uses_correct_cosine_sign():
    action = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    gradient = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float64)
    row = correction._alignment_rows(action, gradient)[0]
    assert row["action_norm"] == pytest.approx(1.0)
    assert row["gradient_norm"] == pytest.approx(1.0)
    assert row["directional_derivative"] == pytest.approx(-1.0)
    assert row["signed_descent_dot"] == pytest.approx(1.0)
    assert row["cosine_to_negative_gradient"] == pytest.approx(1.0)
    assert row["local_descent"] is True


def test_alignment_rows_zero_gradient_is_undefined():
    action = torch.ones(1, 2, 3)
    gradient = torch.zeros_like(action)
    row = correction._alignment_rows(action, gradient)[0]
    assert row["cosine_to_negative_gradient"] is None
    assert row["exact_zero_gradient"] is True
    assert row["exact_zero_action"] is False


def test_alignment_rows_zero_action_is_undefined():
    action = torch.zeros(1, 2, 3)
    gradient = torch.ones_like(action)
    row = correction._alignment_rows(action, gradient)[0]
    assert row["cosine_to_negative_gradient"] is None
    assert row["exact_zero_action"] is True
    assert row["exact_zero_gradient"] is False


@pytest.mark.parametrize("space", correction.SPACES)
def test_space_vectors_keep_layout(space):
    action = torch.arange(150.0).reshape(1, 2, 75)
    gradient = torch.flip(action, dims=(-1,))
    mask = torch.zeros_like(action)
    mask[..., :20] = 0.5
    a_value, g_value = correction._space_vectors(
        action,
        gradient,
        mask,
        space,
    )
    assert a_value.shape == action.shape
    assert g_value.shape == gradient.shape


def test_primary_space_is_raw_all_not_support_selected():
    action = torch.ones(1, 1, 75)
    gradient = torch.ones_like(action)
    mask = torch.zeros_like(action)

    raw_a, raw_g = correction._space_vectors(
        action,
        gradient,
        mask,
        "raw_all_geometry",
    )
    supported_a, supported_g = correction._space_vectors(
        action,
        gradient,
        mask,
        "raw_supported_geometry",
    )

    assert torch.equal(raw_a, action)
    assert torch.equal(raw_g, gradient)
    assert torch.equal(supported_a, torch.zeros_like(action))
    assert torch.equal(supported_g, torch.zeros_like(gradient))


def _synthetic_summary(rcsp_cosine, rpa_cosine, defined=4):
    return {
        "primary": {
            "defined_cosine_RCSP": defined,
            "defined_cosine_RPA": defined,
            "median_cosine_RCSP": rcsp_cosine,
            "median_cosine_RPA": rpa_cosine,
        }
    }


def test_condition_improved_requires_strict_median_gain():
    assert correction._condition_improved(_synthetic_summary(0.1, 0.2)) is True
    assert correction._condition_improved(_synthetic_summary(0.2, 0.2)) is False
    assert correction._condition_improved(_synthetic_summary(0.3, 0.2)) is False


def test_condition_improved_rejects_undefined_cosines():
    assert correction._condition_improved(
        _synthetic_summary(None, None, defined=0)
    ) is False


def _formal_report_for_decision():
    return {
        "decision": {
            "conditions": {
                "A_single_seen_rescue": False,
                "B_single_new_rescue": False,
                "C_cross28_seen_effectiveness": True,
                "D_cross28_new_effectiveness": True,
                "E_no_temporal_regression": True,
                "F_no_endpoint_regression": False,
                "G_no_safety_regression": False,
                "H_single_direction_improved": False,
                "I_cross28_direction_improved": False,
            },
            "total_temporal_newly_rescued_vs_RCSP": 1,
            "net_gate_improvement": True,
        }
    }


@pytest.mark.parametrize(
    "corrected_h,corrected_i",
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_corrected_hi_can_never_reopen_frozen_candidate(
    corrected_h,
    corrected_i,
):
    result = correction._decision_with_corrected_hi(
        _formal_report_for_decision(),
        corrected_h,
        corrected_i,
    )
    assert result["result"] == "RPA_LRTA_NOT_SUPPORTED"
    assert result["decision_invariant"] is True


def test_direction_interpretation_mechanism_present_but_method_rejected():
    summary = _synthetic_summary(0.1, 0.2)
    result = correction._interpret_direction(True, False, summary, summary)
    assert result["classification"] == (
        "RPA_DIRECTION_MECHANISM_PRESENT_BUT_METHOD_REMAINS_UNSUPPORTED"
    )
    assert result["formal_candidate_decision"] == "RPA_LRTA_NOT_SUPPORTED"
    assert result["pilot_authorized"] is False


def test_direction_interpretation_unresolved_if_target_cosines_undefined():
    defined = _synthetic_summary(0.1, 0.2)
    undefined = _synthetic_summary(None, None, defined=0)
    result = correction._interpret_direction(False, False, undefined, defined)
    assert result["classification"] == (
        "RPA_DIRECTION_REPORTING_REMAINS_UNRESOLVED"
    )


def test_no_optimizer_or_training_entry_points_in_correction_source():
    source = Path(correction.__file__).read_text(encoding="utf-8")
    assert "torch.optim." not in source
    assert "checked_refiner_step(" not in source
    assert ".backward()" not in source
