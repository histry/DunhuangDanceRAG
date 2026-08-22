#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json

import numpy as np

from evaluation.audit_heading import main, validate_formal_heading_contract
from support.event_identity import make_event_db_contract


def test_current_formal_heading_contract_is_accepted() -> None:
    result = validate_formal_heading_contract(
        {
            "heading_contract": {
                "schema": "formal_boundary_aligned_heading_v1",
                "authoritative_reference": "motion_ref_path",
                "event_turn_budget_source": "generation_event_db",
            }
        }
    )
    assert result["ok"] is True
    assert result["reasons"] == []


def test_missing_or_legacy_heading_contract_fails_closed() -> None:
    missing = validate_formal_heading_contract({})
    legacy = validate_formal_heading_contract(
        {
            "event_heading_planner": {"legacy": True},
            "heading_contract": {
                "schema": "legacy_event_heading",
                "authoritative_reference": "motion_ref_path",
                "event_turn_budget_source": "generation_event_db",
            },
        }
    )
    assert missing["ok"] is False
    assert "missing_formal_heading_contract" in missing["reasons"]
    assert legacy["ok"] is False
    assert any("invalid_formal_heading_schema" in value for value in legacy["reasons"])


def test_current_formal_heading_report_passes_functional_audit(tmp_path) -> None:
    frames = 12
    motion = np.zeros((frames, 151), dtype=np.float32)
    identity_rot6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:151] = np.tile(identity_rot6d, 24)

    motion_path = tmp_path / "motion.npy"
    reference_path = tmp_path / "motion.motion_ref.npy"
    db_path = tmp_path / "generation_db.npz"
    report_path = tmp_path / "generation.report.json"
    output_path = tmp_path / "heading.audit.json"
    np.save(motion_path, motion)
    np.save(reference_path, motion)

    event_uids = np.asarray(["event-0"], dtype=object)
    np.savez(
        db_path,
        paths=np.asarray(["event.npy"], dtype=object),
        event_uids=event_uids,
        canonical_fps=np.asarray([30.0], dtype=np.float32),
        event_turn_intents=np.asarray(["none"], dtype=object),
        event_yaw_budget_rad=np.asarray([np.pi], dtype=np.float32),
        event_stage_delta_yaw_rad=np.asarray([0.0], dtype=np.float32),
    )
    report_path.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "event_db_contract": make_event_db_contract(event_uids),
                "motion_ref_path": str(reference_path),
                "heading_contract": {
                    "schema": "formal_boundary_aligned_heading_v1",
                    "authoritative_reference": "motion_ref_path",
                    "event_turn_budget_source": "generation_event_db",
                },
                "stage_reports": {
                    "closed_loop_concat": [{"event_id": 0, "core_span": [0, frames]}]
                },
                "slots": [{}],
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--motion",
            str(motion_path),
            "--report",
            str(report_path),
            "--db",
            str(db_path),
            "--out",
            str(output_path),
            "--fps",
            "30",
        ]
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert result["ok"] is True
    assert result["schema"] == "formal_boundary_aligned_heading_audit_v1"
