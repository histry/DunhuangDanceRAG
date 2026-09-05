"""Behavior checks for the report-only adapter. Not executed locally."""
from training.generation_stage_diagnostics import summarize_report


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
