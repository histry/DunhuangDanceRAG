"""Behavior checks for the development diagnostic adapter. Not executed locally."""
from pathlib import Path

import numpy as np
import pytest

from training.generation_stage_diagnostics import (
    _apply_verified_solution_windows,
    _array,
    _boundary_nonregression,
    _merge_frame_windows,
    summarize_report,
)


def test_absent_diffusion_transaction_is_unknown_not_success():
    report = {"config": {"diffusion_enable": True},
              "closed_loop": {"env": {"BOUNDARY_USE_DIFFUSION": "0"}}}
    result = summarize_report(report)
    assert result["switches"]["BOUNDARY_USE_DIFFUSION"] == "0"
    assert result["transactions"]["diffusion"]["accepted"] is None


def test_rollback_preserves_candidate_and_selected_evidence_separately():
    result = summarize_report({"stage_reports": {
        "pre_refine_audit": {"joint_jerk_mps3_max": 2100},
        "final_audit": {"joint_jerk_mps3_max": 2100},
        "boundary_refiner_transaction": {"accepted": False, "rolled_back": True,
                                         "reasons": ["no_meaningful_repair_gain"]},
        "lower_body_ik_true_ik": {
            "audit_after_candidate": {"joint_jerk_mps3_max": 5093},
            "local_transactions": {"attempted": 2, "accepted": 0, "rejected": 2,
                "transactions": [{"relative_reasons": ["jerk"]}, {"relative_reasons": ["jerk", "skate"]}]},
        },
    }})
    assert result["metrics"]["ik_candidate"]["joint_jerk_mps3_max"] == 5093
    assert result["metrics"]["final_audit"]["joint_jerk_mps3_max"] == 2100
    assert result["ik_rejections"]["relative_reasons"] == {"jerk": 2, "skate": 1}
    assert result["transactions"]["refiner"]["rolled_back"] is True


def test_round_counts_are_not_conflated_with_assembly_decisions():
    result = summarize_report({
        "closed_loop": {"rounds": [{"round": 0, "unsafe_boundaries": 36}, {"round": 1, "unsafe_boundaries": 35}]},
        "stage_reports": {"closed_loop_concat": [{"decision": "accepted_best_unsafe_fallback"}]},
    })
    assert len(result["rounds"]) == 2
    assert result["assembly_decisions"]["accepted_best_unsafe_fallback"] == 1


def _solution_row(root: Path, case_id: str, span, reference, changed_local):
    directory = root / case_id
    directory.mkdir()
    returned = reference.copy()
    returned[changed_local, 0] = 1.0
    artifacts = {
        "reference": _array(directory, "reference", reference),
        "returned_motion": _array(directory, "returned_motion", returned),
        "returned_action": _array(
            directory,
            "returned_action",
            np.ones((len(reference), 75), dtype=np.float32),
        ),
        "proposal_action": _array(
            directory,
            "proposal_action",
            np.zeros((len(reference), 75), dtype=np.float32),
        ),
    }
    return {
        "case_id": case_id,
        "solution_artifacts": {
            "bundle_sha256": "bundle-sha",
            "frame_span": list(span),
            "artifacts": artifacts,
        },
    }


def test_replay_rejects_overlapping_actual_edit_frames(tmp_path):
    full = np.zeros((6, 3), dtype=np.float32)
    first = _solution_row(tmp_path, "a", [0, 4], full[0:4], 2)
    second = _solution_row(tmp_path, "b", [2, 6], full[2:6], 0)
    with pytest.raises(ValueError, match="edit regions overlap"):
        _apply_verified_solution_windows(
            full, [first, second], "bundle-sha", 1.0e-7
        )


def test_replay_rejects_reference_from_a_different_capture(tmp_path):
    full = np.zeros((5, 3), dtype=np.float32)
    stale = full[1:4].copy()
    stale[0, 1] = 2.0
    row = _solution_row(tmp_path, "stale", [1, 4], stale, 1)
    with pytest.raises(ValueError, match="solution reference mismatch"):
        _apply_verified_solution_windows(full, [row], "bundle-sha", 1.0e-7)


def test_v9_merges_physical_and_boundary_windows():
    assert _merge_frame_windows(
        [[10, 20], [22, 30], [70, 90]],
        frames=100,
        gap=2,
    ) == [[10, 30], [70, 90]]


def test_v9_boundary_guard_is_strictly_nonregressing():
    before = [{
        "slot": 1,
        "actual_boundary_jerk_mps3": 500.0,
        "actual_foot_slip_p95_mps": 0.30,
    }]
    improved = [{
        "slot": 1,
        "actual_boundary_jerk_mps3": 490.0,
        "actual_foot_slip_p95_mps": 0.25,
    }]
    regressed = [{
        "slot": 1,
        "actual_boundary_jerk_mps3": 490.0,
        "actual_foot_slip_p95_mps": 0.31,
    }]
    assert _boundary_nonregression(before, improved)["accepted"] is True
    decision = _boundary_nonregression(before, regressed)
    assert decision["accepted"] is False
    assert decision["reasons"] == [
        "slot_1:actual_foot_slip_p95_mps_regressed"
    ]
