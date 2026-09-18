"""Lifecycle contracts for the generate-only Motion Refiner asset."""

import os
import subprocess
import sys
from pathlib import Path

import scripts.resolve_generation_refiner as resolver


def _formal_payload():
    return {
        "version": resolver.REFINER_MODEL_VERSION,
        "state_dict": {"weight": object()},
        "motion_contract": {"schema": "test"},
        "training_event_db_contract": {
            "schema": "dunhuang_event_db_contract_v2",
            "num_events": 3,
            "ordered_event_uid_sha256": "a" * 64,
        },
        "validation": {
            "checkpoint_decision": {"publish_allowed": True},
        },
    }


def test_explicit_refiner_checkpoint_is_never_replaced_by_directory_scan(tmp_path):
    explicit = tmp_path / "chosen.pt"
    explicit.write_bytes(b"chosen")
    (tmp_path / "motion_refiner_train_only_refiner.pt").write_bytes(
        b"canonical"
    )
    assert resolver._candidate_paths(tmp_path, explicit) == [
        explicit.resolve()
    ]


def test_unique_nested_formal_name_can_be_discovered(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "formal" / "boundary_refiner.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"formal")
    assert resolver._candidate_paths(tmp_path, None) == [
        checkpoint.resolve()
    ]


def test_rejected_validation_checkpoint_is_not_a_generation_asset(tmp_path):
    rejected = (
        tmp_path
        / "checkpoints"
        / "motion_refiner_train_only_refiner.rejected_validation.pt"
    )
    rejected.parent.mkdir(parents=True)
    rejected.write_bytes(b"rejected")
    assert resolver._candidate_paths(tmp_path, None) == []


def test_checkpoint_without_formal_publication_decision_is_rejected(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "boundary_refiner.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _formal_payload()
    payload["validation"]["checkpoint_decision"]["publish_allowed"] = False
    monkeypatch.setattr(
        resolver,
        "_trusted_torch_load",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        resolver,
        "assert_motion_checkpoint_contract",
        lambda *_args, **_kwargs: None,
    )
    accepted, reason = resolver._validate_candidate(checkpoint, object())
    assert accepted is None
    assert "not published" in reason


def test_formal_checkpoint_binding_records_hash_and_provenance(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "boundary_refiner.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _formal_payload()
    monkeypatch.setattr(
        resolver,
        "_trusted_torch_load",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        resolver,
        "assert_motion_checkpoint_contract",
        lambda *_args, **_kwargs: None,
    )
    accepted, reason = resolver._validate_candidate(checkpoint, object())
    assert reason is None
    assert accepted["path"] == str(checkpoint.resolve())
    assert len(accepted["sha256"]) == 64
    assert accepted["training_event_db_contract"]["num_events"] == 3


def test_path_invoked_resolver_bootstraps_repository_imports(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "resolve_generation_refiner.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--run-root" in completed.stdout
