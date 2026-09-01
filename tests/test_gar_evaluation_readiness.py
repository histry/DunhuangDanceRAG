from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.gar_evaluation_readiness import (
    GAR_SELECTION_TRACE_SCHEMA,
    CandidatePoolEntry,
    GARCandidateTrace,
    TraceContractError,
    behavior_config_fingerprint,
    boundary_metrics_from_authoritative_risk,
    build_closed_loop_trace,
    candidate_pool_fingerprint,
    derive_false_safe,
    derive_recovered_after_reselection,
    make_boundary_id,
    make_evaluation_case_id,
    make_sequence_id,
    read_trace,
    write_trace,
)
from training.motion_models import MotionGenerationConfig
from routing.boundary_closed_loop import gar_evaluation_trace_context


ROOT = Path(__file__).resolve().parents[1]


def _risk(total: float) -> dict[str, float]:
    return {
        "total": total,
        "entry_velocity": 0.10,
        "exit_velocity": 0.20,
        "entry_acceleration": 0.30,
        "exit_acceleration": 0.40,
        "boundary_joint_jerk_max": 10.0,
        "entry_fk_jump": 0.001,
        "exit_fk_jump": 0.002,
        "entry_fk_jump_max_m": 0.003,
        "exit_fk_jump_max_m": 0.004,
        "entry_rotation_step_rad": 0.01,
        "exit_rotation_step_rad": 0.02,
        "foot_slip": 0.01,
        "foot_slip_p95": 0.02,
        "foot_slip_max": 0.03,
        "foot_penetration": 0.0001,
        "foot_penetration_max_m": 0.005,
        "contact_switch": 0.25,
    }


def _fixture_inputs() -> dict:
    slots = [
        {"start_seconds": 0.0, "end_seconds": 2.0, "target_frames": 60},
        {"start_seconds": 2.0, "end_seconds": 4.0, "target_frames": 60},
    ]
    db = {
        "event_uids": ["evt-a", "evt-b", "evt-c"],
        "source_uids": ["src-a", "src-b", "src-c"],
        "recording_uids": ["rec-a", "rec-b", "rec-c"],
        "sequence_ids": ["take-a", "take-b", "take-c"],
    }
    retrieval_report = [
        {
            "candidate_event_indices": [0, 1],
            "candidate_router_probabilities": [0.8, 0.2],
        },
        {
            "candidate_event_indices": [1, 2],
            "candidate_router_probabilities": [0.7, 0.3],
        },
    ]
    round0 = {
        "round": 0,
        "assembly_report": [
            {"slot": 0, "event_id": 0, "candidate_trials": []},
            {
                "slot": 1,
                "event_id": 1,
                "candidate_rank": 0,
                "risk_score_predicted": 0.30,
                "risk_predicted": _risk(0.30),
                "safe_predicted": True,
                "candidate_trials": [
                    {
                        "event_id": 1,
                        "rank": 0,
                        "risk_score": 0.30,
                        "risk": _risk(0.30),
                        "safe": True,
                    },
                    {
                        "event_id": 2,
                        "rank": 1,
                        "risk_score": 0.45,
                        "risk": _risk(0.45),
                        "safe": True,
                    },
                ],
            },
        ],
        "boundary_rows": [
            {
                "slot": 1,
                "curr_event_id": 1,
                "actual_risk_score": 0.90,
                "safe": False,
                "failure_reasons": ["boundary_joint_jerk_max_mps3_too_high"],
                "risk": _risk(0.90),
            }
        ],
        "selected_pairs": [[0, 0], [1, 0]],
    }
    # The production loop renumbers the remaining candidate to internal rank 0
    # after banning evt-b. The readiness trace must still report pool rank 2.
    round1 = {
        "round": 1,
        "assembly_report": [
            {"slot": 0, "event_id": 0, "candidate_trials": []},
            {
                "slot": 1,
                "event_id": 2,
                "candidate_rank": 0,
                "risk_score_predicted": 0.40,
                "risk_predicted": _risk(0.40),
                "safe_predicted": True,
                "candidate_trials": [
                    {
                        "event_id": 2,
                        "rank": 0,
                        "risk_score": 0.40,
                        "risk": _risk(0.40),
                        "safe": True,
                    }
                ],
            },
        ],
        "boundary_rows": [
            {
                "slot": 1,
                "curr_event_id": 2,
                "actual_risk_score": 0.20,
                "safe": True,
                "failure_reasons": [],
                "risk": _risk(0.20),
            }
        ],
        "selected_pairs": [[0, 0], [2, 0]],
    }
    return {
        "audio": "fixtures/song.wav",
        "slots": slots,
        "db": db,
        "event_uids": db["event_uids"],
        "candidate_lists": [[0, 1], [1, 2]],
        "retrieval_report": retrieval_report,
        "round_records": [round0, round1],
        "final_round": 1,
        "runtime_commit": "a" * 40,
        "config_fingerprint": "b" * 64,
        "retrieval_index_fingerprint": "c" * 64,
        "generator_id": "edge151_motion_generation_pipeline",
        "generator_version": "edge151_refiner_diffusion_ik_pipeline_v1",
        "generator_checkpoint_fingerprint": None,
        "generator_config_fingerprint": "d" * 64,
        "repair_operator_id": "so3_endpoint_velocity_bridge",
        "repair_operator_version": "so3_endpoint_velocity_bridge_v1",
        "repair_config_fingerprint": "e" * 64,
        "selection_policy_id": "current_policy_v1",
        "method_variant_id": "current_boundary_closed_loop",
        "random_seed": 42,
        "risk_threshold_value": None,
        "risk_threshold_source": "BoundaryContinuityLimits.from_environment",
        "risk_thresholds": {"boundary_joint_jerk_max_mps3": 650.0},
        "capabilities": {
            "candidate_simulation_enabled": True,
            "adaptive_transition_enabled": True,
            "post_audit_enabled": True,
            "reselection_enabled": True,
        },
        "runtime": {
            "retrieval_runtime_ms": 1.0,
            "candidate_simulation_runtime_ms": 2.0,
            "generation_runtime_ms": 3.0,
            "post_audit_runtime_ms": 4.0,
            "reselection_runtime_ms": 0.5,
            "sequence_total_runtime_ms": 11.0,
        },
    }


def _trace():
    return build_closed_loop_trace(**_fixture_inputs())


def test_readiness_config_is_disabled_by_default() -> None:
    cfg = MotionGenerationConfig()
    assert cfg.gar_evaluation_trace_enable is False
    assert cfg.gar_evaluation_method_variant_id == "current_boundary_closed_loop"


def test_candidate_rank_is_strictly_one_based() -> None:
    with pytest.raises(TraceContractError, match="1-based"):
        CandidatePoolEntry("evt-a", 0, 0.8, "evt-a", "rec-a", "f" * 64)


def test_candidate_rank_uses_original_pool_after_reselection() -> None:
    boundary = _trace().boundaries[0]
    final = [item for item in boundary.candidate_trace if item.selected_finally]
    assert len(final) == 1
    assert final[0].candidate_id == "evt-c"
    assert final[0].candidate_rank == 2
    assert boundary.summary.final_candidate_rank == 2


def test_candidate_rank_cannot_exceed_pool_size() -> None:
    value = _trace().to_dict()
    value["boundaries"][0]["candidate_trace"][0]["candidate_rank"] = 3
    with pytest.raises(TraceContractError, match="exceeds"):
        type(_trace()).from_dict(value)


def test_candidate_rank_must_match_manifest_identity() -> None:
    value = _trace().to_dict()
    value["boundaries"][0]["candidate_trace"][0]["candidate_rank"] = 2
    with pytest.raises(TraceContractError, match="stable pool manifest"):
        type(_trace()).from_dict(value)


def test_candidate_pool_fingerprint_is_stable() -> None:
    manifest = [
        CandidatePoolEntry("evt-a", 1, 0.8, "evt-a", "rec-a", "a" * 64),
        CandidatePoolEntry("evt-b", 2, 0.2, "evt-b", "rec-b", "b" * 64),
    ]
    assert candidate_pool_fingerprint(manifest) == candidate_pool_fingerprint(
        [dict(reversed(list(dataclasses.asdict(row).items()))) for row in manifest]
    )


def test_candidate_order_changes_pool_fingerprint() -> None:
    first = [
        CandidatePoolEntry("evt-a", 1, 0.8, None, None, "a" * 64),
        CandidatePoolEntry("evt-b", 2, 0.2, None, None, "b" * 64),
    ]
    second = [
        CandidatePoolEntry("evt-b", 1, 0.2, None, None, "b" * 64),
        CandidatePoolEntry("evt-a", 2, 0.8, None, None, "a" * 64),
    ]
    assert candidate_pool_fingerprint(first) != candidate_pool_fingerprint(second)


def test_candidate_identity_changes_pool_fingerprint() -> None:
    first = [CandidatePoolEntry("evt-a", 1, 0.8, None, None, "a" * 64)]
    second = [CandidatePoolEntry("evt-z", 1, 0.8, None, None, "a" * 64)]
    assert candidate_pool_fingerprint(first) != candidate_pool_fingerprint(second)


def test_runtime_and_score_do_not_change_pool_fingerprint() -> None:
    first = [CandidatePoolEntry("evt-a", 1, 0.8, None, None, "a" * 64)]
    second = [CandidatePoolEntry("evt-a", 1, 0.1, None, None, "z" * 64)]
    assert candidate_pool_fingerprint(first) == candidate_pool_fingerprint(second)


def test_pre_and_post_risk_are_independent_fields() -> None:
    generated = [item for item in _trace().boundaries[0].candidate_trace if item.generated]
    assert generated[0].pre_risk == 0.30
    assert generated[0].post_risk == 0.90
    assert generated[-1].pre_risk == 0.40
    assert generated[-1].post_risk == 0.20


def test_missing_post_risk_remains_null() -> None:
    simulated_only = [
        item for item in _trace().boundaries[0].candidate_trace if not item.generated
    ]
    assert simulated_only
    assert simulated_only[0].post_risk is None
    assert simulated_only[0].post_safe is None


def test_missing_generated_post_safe_remains_null() -> None:
    inputs = _fixture_inputs()
    inputs["round_records"][0]["boundary_rows"][0]["safe"] = None
    trace = build_closed_loop_trace(**inputs)
    generated = [
        item
        for item in trace.boundaries[0].candidate_trace
        if item.generated and item.reselection_attempt_index == 0
    ]
    assert generated[0].post_safe is None
    assert trace.boundaries[0].summary.initial_post_safe is None
    assert trace.boundaries[0].summary.recovered_after_reselection is None


def test_missing_pre_risk_serializes_as_null() -> None:
    item = GARCandidateTrace(
        candidate_id="evt-a",
        candidate_rank=1,
        retrieval_score=None,
        evaluation_order=1,
        pre_risk=None,
        pre_risk_components=None,
        pre_safe=None,
        generated=False,
        post_risk=None,
        post_risk_components=None,
        post_safe=None,
        post_audit_failure_reasons=(),
        selected_initially=False,
        selected_finally=False,
        rejected_after_post_audit=False,
        reselection_attempt_index=0,
        repair_operator_id="repair",
        generator_id="generator",
        runtime_ms=None,
        risk_threshold_value=None,
        risk_threshold_source="multi_threshold_contract",
        false_safe=None,
    )
    assert dataclasses.asdict(item)["pre_risk"] is None


def test_false_safe_requires_complete_scalar_data() -> None:
    assert derive_false_safe(None, 0.8, 0.5) is None
    assert derive_false_safe(0.2, None, 0.5) is None
    assert derive_false_safe(0.2, 0.8, None) is None
    assert derive_false_safe(0.2, 0.8, 0.5) is True
    assert derive_false_safe(0.8, 0.9, 0.5) is False


def test_current_multi_threshold_contract_keeps_false_safe_null() -> None:
    assert all(
        item.false_safe is None for item in _trace().boundaries[0].candidate_trace
    )


def test_recovered_after_reselection_logic() -> None:
    boundary = _trace().boundaries[0]
    assert boundary.summary.reselection_count == 1
    assert boundary.summary.initial_post_safe is False
    assert boundary.summary.final_post_safe is True
    assert boundary.summary.recovered_after_reselection is True


def test_no_reselection_never_reports_recovered() -> None:
    assert derive_recovered_after_reselection(False, True, 0) is False
    assert derive_recovered_after_reselection(True, True, 1) is False
    assert derive_recovered_after_reselection(None, True, 1) is None


def test_sequence_boundary_and_case_ids_are_stable_and_paired() -> None:
    slots = _fixture_inputs()["slots"]
    left = make_sequence_id("server/path/song.wav", slots)
    right = make_sequence_id("windows\\path\\song.wav", copy.deepcopy(slots))
    assert left == right
    assert make_boundary_id(left, 1) == make_boundary_id(right, 1)
    assert make_evaluation_case_id(left, 1) == make_evaluation_case_id(right, 1)


def test_trace_projection_does_not_mutate_selection_inputs() -> None:
    inputs = _fixture_inputs()
    before = copy.deepcopy(inputs)
    trace = build_closed_loop_trace(**inputs)
    assert inputs == before
    assert trace.boundaries[0].summary.initial_candidate_id == "evt-b"
    assert trace.boundaries[0].summary.final_candidate_id == "evt-c"


def test_runtime_values_do_not_participate_in_selection() -> None:
    first = _fixture_inputs()
    second = copy.deepcopy(first)
    second["runtime"] = {key: 999999.0 for key in second["runtime"]}
    a = build_closed_loop_trace(**first).boundaries[0].summary
    b = build_closed_loop_trace(**second).boundaries[0].summary
    assert a == b


def test_generator_and_repair_metadata_do_not_participate_in_selection() -> None:
    first = _fixture_inputs()
    second = copy.deepcopy(first)
    second["generator_id"] = "same_behavior_named_for_comparison"
    second["repair_operator_id"] = "same_repair_named_for_comparison"
    a = build_closed_loop_trace(**first).boundaries[0].summary
    b = build_closed_loop_trace(**second).boundaries[0].summary
    assert (a.initial_candidate_id, a.final_candidate_id) == (
        b.initial_candidate_id,
        b.final_candidate_id,
    )


def test_boundary_trace_carries_complete_sequence_provenance() -> None:
    trace = _trace()
    boundary = trace.boundaries[0]
    assert boundary.generator_version == trace.generator_version
    assert boundary.generator_config_fingerprint == trace.generator_config_fingerprint
    assert boundary.repair_operator_version == trace.repair_operator_version
    assert boundary.method_variant_id == trace.method_variant_id


def test_trace_serialization_round_trip_preserves_nulls(tmp_path: Path) -> None:
    trace = _trace()
    path = write_trace(trace, tmp_path / "trace.json")
    loaded = read_trace(path)
    assert loaded.to_dict() == trace.to_dict()
    text = path.read_text(encoding="utf-8")
    assert '"risk_threshold_value": null' in text
    assert '"post_risk": null' in text


def test_existing_metrics_map_without_placeholder_values() -> None:
    metrics = boundary_metrics_from_authoritative_risk(_risk(0.2))
    assert metrics.fk_position_jump == 0.002
    assert metrics.so3_rotation_geodesic_jump == 0.02
    assert metrics.velocity_jump == 0.20
    assert metrics.acceleration_jump == 0.40
    assert metrics.foot_skate == 0.02
    assert metrics.penetration_max == 0.005
    assert metrics.contact_discontinuity == 0.25


def test_unimplemented_metrics_remain_null() -> None:
    metrics = boundary_metrics_from_authoritative_risk(_risk(0.2))
    assert metrics.boundary_jerk_mean is None
    assert metrics.boundary_jerk_p95 is None
    assert metrics.penetration_mean is None
    assert metrics.root_velocity_discontinuity is None
    assert metrics.heading_discontinuity is None


def test_behavior_config_fingerprint_excludes_trace_only_fields() -> None:
    first = {
        "seed": 42,
        "gar_evaluation_trace_enable": False,
        "gar_evaluation_method_variant_id": "a",
    }
    second = {
        "seed": 42,
        "gar_evaluation_trace_enable": True,
        "gar_evaluation_method_variant_id": "b",
    }
    assert behavior_config_fingerprint(first) == behavior_config_fingerprint(second)


def test_behavior_config_fingerprint_excludes_transient_generation_provenance() -> None:
    config = {"seed": 42}
    first_environment = {
        "GENERATION_FPS": "30",
        "GENERATION_PYTHON": "/env-a/bin/python",
        "GENERATION_ROUTER_CKPT": "/run-a/router.pt",
        "GENERATION_INDEX_JSON": "/run-a/index.json",
        "GENERATION_SCHEDULE_RUN_ID": "temporary-a",
        "GENERATION_REBUILD_EVENT_DB": "1",
    }
    second_environment = {
        "GENERATION_FPS": "30",
        "GENERATION_PYTHON": "/env-b/bin/python",
        "GENERATION_ROUTER_CKPT": "/run-b/router.pt",
        "GENERATION_INDEX_JSON": "/run-b/index.json",
        "GENERATION_SCHEDULE_RUN_ID": "temporary-b",
        "GENERATION_REBUILD_EVENT_DB": "0",
    }
    assert behavior_config_fingerprint(
        config, runtime_environment=first_environment
    ) == behavior_config_fingerprint(config, runtime_environment=second_environment)


def test_behavior_config_fingerprint_keeps_behavior_environment() -> None:
    config = {"seed": 42}
    assert behavior_config_fingerprint(
        config, runtime_environment={"GENERATION_FPS": "30"}
    ) != behavior_config_fingerprint(
        config, runtime_environment={"GENERATION_FPS": "60"}
    )


def test_candidate_metadata_fingerprint_uses_stable_db_identity() -> None:
    manifest = _trace().boundaries[0].candidate_pool_manifest
    assert manifest[0].candidate_id == "evt-b"
    assert manifest[0].source_recording_id == "rec-b"
    assert len(manifest[0].candidate_metadata_fingerprint) == 64


def test_candidate_pool_manifest_covers_first_slot_without_fabricating_boundary() -> None:
    trace = _trace()
    assert [pool.slot_index for pool in trace.candidate_pools] == [0, 1]
    assert trace.candidate_pools[0].candidate_pool_size == 2
    assert trace.sequence.sequence_boundary_count == 1


def test_boundary_failure_reasons_come_from_final_authoritative_audit() -> None:
    summary = _trace().boundaries[0].summary
    assert summary.hard_failure is False
    assert summary.boundary_hard_failure is False
    assert summary.failure_reasons == ()


def test_unsafe_final_boundary_preserves_reason_codes() -> None:
    inputs = _fixture_inputs()
    inputs["round_records"] = inputs["round_records"][:1]
    inputs["final_round"] = 0
    summary = build_closed_loop_trace(**inputs).boundaries[0].summary
    assert summary.hard_failure is True
    assert summary.failure_reasons == (
        "boundary_joint_jerk_max_mps3_too_high",
    )


def test_sequence_trace_contains_primitive_runtime_not_boundary_estimates() -> None:
    trace = _trace()
    assert trace.sequence.sequence_runtime_ms == 11.0
    assert trace.runtime["generation_runtime_ms"] == 3.0
    assert all(value is None for value in trace.boundaries[0].runtime.values())
    assert all(item.runtime_ms is None for item in trace.boundaries[0].candidate_trace)


def test_experiment_status_is_explicitly_false() -> None:
    status = _trace().experiment_status
    assert status == {
        "paper1_experiments_implemented": False,
        "oracle_implemented": False,
        "statistical_tests_implemented": False,
        "long_horizon_benchmark_implemented": False,
        "production_selection_behavior_changed": False,
    }


def test_schema_has_no_selection_or_aggregate_algorithm_definitions() -> None:
    source = (ROOT / "evaluation" / "gar_evaluation_readiness.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden = {
        "oracle_select",
        "unsafe_boundary_rate",
        "failure_recovery_rate",
        "false_safe_rate",
        "sequence_success_rate",
        "wilcoxon",
        "spearman",
        "delta_interpolator",
    }
    assert function_names.isdisjoint(forbidden)


def test_trace_builder_is_called_after_production_rounds() -> None:
    source = (ROOT / "routing" / "boundary_closed_loop.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    generate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_closed_loop"
    )
    calls = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    trace_line = min(
        node.lineno for node in calls if node.func.id == "build_closed_loop_trace"
    )
    selection_line = min(
        node.lineno
        for node in calls
        if node.func.id == "assemble_closed_loop_reference"
    )
    audit_line = min(node.lineno for node in calls if node.func.id == "audit_boundaries")
    assert trace_line > selection_line
    assert trace_line > audit_line


def test_readiness_trace_uses_v1_schema_and_stable_pairing_fields() -> None:
    trace = _trace()
    boundary = trace.boundaries[0]
    assert trace.schema_version == GAR_SELECTION_TRACE_SCHEMA
    assert boundary.schema_version == GAR_SELECTION_TRACE_SCHEMA
    assert boundary.sequence_id == trace.sequence.sequence_id
    assert boundary.boundary_id in trace.sequence.boundary_ids
    assert boundary.evaluation_case_id.startswith("case_")
    assert boundary.candidate_pool_fingerprint


def test_runtime_context_records_base_selection_policy(monkeypatch) -> None:
    monkeypatch.delenv("GROUNDING_GLOBAL_ROUTE_ENABLE", raising=False)
    context = gar_evaluation_trace_context(
        SimpleNamespace(
            REFINER_MODEL_VERSION="refiner-v",
            DIFFUSION_MODEL_VERSION="diffusion-v",
        ),
        SimpleNamespace(refiner=None, diffusion=None),
        MotionGenerationConfig(refiner_enable=False, diffusion_enable=False),
        {"event_uids": ["evt-a", "evt-b"]},
        [{"slot": 0, "event_id": 0}],
    )
    assert context["selection_policy_id"].startswith("boundary_closed_loop_")
    assert context["method_variant_id"] == "current_boundary_closed_loop"


def test_runtime_context_records_actual_graph_sb_gar_policy(monkeypatch) -> None:
    monkeypatch.setenv("GROUNDING_GLOBAL_ROUTE_ENABLE", "1")
    context = gar_evaluation_trace_context(
        SimpleNamespace(
            REFINER_MODEL_VERSION="refiner-v",
            DIFFUSION_MODEL_VERSION="diffusion-v",
        ),
        SimpleNamespace(refiner=None, diffusion=None),
        MotionGenerationConfig(refiner_enable=False, diffusion_enable=False),
        {"event_uids": ["evt-a", "evt-b"]},
        [
            {
                "slot": 0,
                "event_id": 0,
                "method": "viability-aware Routing Budget",
                "risk_predicted": {"event_geometry_grounding": {}},
            }
        ],
    )
    assert context["selection_policy_id"].startswith("fisher_rao_graph_sb_")
    assert context["method_variant_id"] == "current_geometry_aware_routing"
