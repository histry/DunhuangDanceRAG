from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from evaluation.repairability_outcome_bank import (
    OUTCOME_BANK_RECORD_SCHEMA,
    OutcomeBankWriter,
    OutcomeRecord,
    reason_families,
)
from model.repairability_predictor import (
    FEATURE_NAMES,
    REPAIRABILITY_CHECKPOINT_SCHEMA,
    VIOLATION_FAMILIES,
    RepairabilityPredictor,
    RepairabilityRanker,
    proposal_feature_mapping,
)


def feature_fixture():
    return proposal_feature_mapping(
        router_probability=0.5,
        original_rank=1,
        pool_size=4,
        target_frames=120,
        transition_frames=20,
        core_frames=100,
        core_warp=1.1,
        pre_safe=True,
        pre_risk_score=0.25,
        risk={"foot_slip": 0.01, "entry_fk_jump": 0.02},
    )


def test_feature_schema_is_complete_and_finite():
    features = feature_fixture()
    assert tuple(features) == FEATURE_NAMES
    assert np.isfinite(np.asarray(list(features.values()))).all()
    assert features["original_rank_fraction"] == pytest.approx(1.0 / 3.0)


def test_outcome_record_derives_authoritative_safety_from_reasons():
    record = OutcomeRecord(
        schema=OUTCOME_BANK_RECORD_SCHEMA,
        sequence_id="seq",
        boundary_id="boundary",
        evaluation_case_id="case",
        group_id="recording-group",
        slot_index=1,
        candidate_id="event",
        candidate_event_index=3,
        source_recording_id="recording",
        source_performer_id="performer",
        original_rank=0,
        candidate_pool_size=2,
        random_seed=7,
        features=feature_fixture(),
        pre_safe=True,
        pre_risk=0.2,
        boundary_safe=False,
        physical_safe=True,
        activity_safe=True,
        schedule_safe=True,
        post_safe=False,
        post_risk=0.8,
        failure_reasons=("foot_slip_p95_exceeded",),
        violation_families=reason_families(("foot_slip_p95_exceeded",)),
        runtime_ms=10.0,
        runtime_commit="commit",
        config_fingerprint="config",
        candidate_pool_fingerprint="pool",
        generator_fingerprint="generator",
        repair_fingerprint="repair",
    )
    assert record.violation_families == ("foot_slip",)
    with pytest.raises(ValueError, match="post_safe"):
        dataclasses.replace(record, post_safe=True)


def test_outcome_bank_rejects_mixed_candidate_pools(tmp_path):
    record = OutcomeRecord(
        schema=OUTCOME_BANK_RECORD_SCHEMA,
        sequence_id="seq",
        boundary_id="boundary",
        evaluation_case_id="case",
        group_id="recording-group",
        slot_index=1,
        candidate_id="event-a",
        candidate_event_index=3,
        source_recording_id="recording",
        source_performer_id="performer",
        original_rank=0,
        candidate_pool_size=2,
        random_seed=7,
        features=feature_fixture(),
        pre_safe=True,
        pre_risk=0.2,
        boundary_safe=True,
        physical_safe=True,
        activity_safe=True,
        schedule_safe=True,
        post_safe=True,
        post_risk=0.1,
        failure_reasons=(),
        violation_families=(),
        runtime_ms=10.0,
        runtime_commit="commit",
        config_fingerprint="config",
        candidate_pool_fingerprint="pool-a",
        generator_fingerprint="generator",
        repair_fingerprint="repair",
    )
    writer = OutcomeBankWriter(tmp_path / "bank.jsonl")
    assert writer.append(record) is True
    changed_pool = dataclasses.replace(
        record,
        candidate_id="event-b",
        candidate_event_index=4,
        random_seed=8,
        candidate_pool_fingerprint="pool-b",
    )
    with pytest.raises(ValueError, match="candidate pools"):
        writer.append(changed_pool)


def test_ranker_abstains_on_indistinguishable_candidates():
    model = RepairabilityPredictor(architecture="linear")
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = {
        "schema": REPAIRABILITY_CHECKPOINT_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "violation_families": list(VIOLATION_FAMILIES),
        "model_config": {
            "architecture": "linear",
            "hidden_dims": [64, 64, 32],
            "dropout": 0.1,
        },
        "state_dict": model.state_dict(),
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_std": [1.0] * len(FEATURE_NAMES),
        "post_risk_mean": 0.0,
        "post_risk_std": 1.0,
        "calibration_temperature": 1.0,
        "selection": {"minimum_probability_spread": 0.02},
        "promotion": {"rank_authorized": True},
    }
    ranker = RepairabilityRanker(checkpoint)
    first = ranker.predict(feature_fixture())
    second = ranker.predict(feature_fixture())
    assert ranker.should_abstain([first, second]) is True
