"""Fast fail-closed checks for the g1f4 M-v1 calibration contract."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from training import refiner_v15_15g1f4_policies as policies


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preregistration():
    return {
        "schema": policies.PREREGISTRATION_SCHEMA,
        "status": "protocol_frozen_not_numerically_calibrated",
        "metric_definition": {
            "beta_identity_regularizer": 1.0,
            "group_weights_v1": {
                "lambda_boundary": 1.0,
                "lambda_jerk": 1.0,
                "lambda_science": 1.0,
            },
        },
        "train_only_calibration": {"rho_E": 1.0e-4},
        "fairness_contract": {"metric_radius_tolerance": 1.0e-12},
    }


def _calibration(prereg_sha):
    observation = {
        "case_uid": "txn:1",
        "calibration_case_uid": "txn:1",
        "selected_method": "geodesic_joint_sqp_k2",
        "calibration_source_stage": "post_composite_selector_post_projector",
        "projected_candidate": True,
        "adapter_incumbent_locked": False,
        "projector_backtracking_factor": 1.0,
        "euclidean_rms": 1.0e-4,
        "metric_rms_at_rho_E": 2.0e-4,
        "metric_to_euclidean_ratio": 2.0,
        "task_space_norm_Jd_W": 0.0,
        "anchor_metric": {"metric_shell_dtype": "float64"},
    }
    return {
        "schema": policies.CALIBRATION_SCHEMA,
        "status": "train_only_numerically_calibrated",
        "implementation_commit": COMMIT,
        "coordinate": "owned_physical_tangent",
        "rho_E": 1.0e-4,
        "alpha": 2.0,
        "rho_G": 2.0e-4,
        "beta_identity_regularizer": 1.0,
        "group_weights_v1": {
            "lambda_boundary": 1.0,
            "lambda_jerk": 1.0,
            "lambda_science": 1.0,
        },
        "metric_radius_tolerance": 1.0e-12,
        "calibration_split": "train",
        "development_or_held_out_consumed": False,
        "calibration_source": "final_composite_selected_projected_train_corrections",
        "selection_stage": "post_composite_selector",
        "projection_stage": "accepted_projected_candidate",
        "adapter_incumbent_cases_excluded": True,
        "identity_cases_excluded": True,
        "raw_preselector_variants_used": False,
        "projected_tangent_used": True,
        "metric_shell_dtype": "float64",
        "observation_count": 1,
        "observations": [observation],
        "preregistered_contract_sha256": prereg_sha,
    }


def _bound_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPECTED_COMMIT", COMMIT)
    prereg = tmp_path / "prereg.json"
    prereg_sha = _write_json(prereg, _preregistration())
    calibration = tmp_path / "calibration.json"
    calibration_sha = _write_json(calibration, _calibration(prereg_sha))
    return prereg, prereg_sha, calibration, calibration_sha


def _load_metric(prereg, calibration, calibration_sha):
    return policies.build_metric_operator(
        policies.ANCHOR_KINEMATIC_METRIC,
        calibration_path=calibration,
        calibration_sha256=calibration_sha,
        preregistration_path=prereg,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("preregistered_contract_sha256", "bad"),
        ("implementation_commit", "bad"),
        ("rho_E", 2.0e-4),
        ("beta_identity_regularizer", 2.0),
        ("group_weights_v1", {"lambda_boundary": 2.0, "lambda_jerk": 1.0, "lambda_science": 1.0}),
        ("calibration_split", "development"),
        ("calibration_source", "pre_selector"),
    ],
)
def test_metric_loader_rejects_contract_mismatch(
    tmp_path, monkeypatch, field, value,
):
    prereg, _, calibration, _ = _bound_paths(tmp_path, monkeypatch)
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload[field] = value
    calibration_sha = _write_json(calibration, payload)
    with pytest.raises(policies.MetricCalibrationRequired):
        _load_metric(prereg, calibration, calibration_sha)


def test_metric_loader_rejects_expected_sha_mismatch(tmp_path, monkeypatch):
    prereg, _, calibration, _ = _bound_paths(tmp_path, monkeypatch)
    with pytest.raises(policies.MetricCalibrationRequired):
        _load_metric(prereg, calibration, "0" * 64)


@pytest.mark.parametrize("method", ["identity", "adapter"])
def test_metric_loader_rejects_identity_or_adapter_calibration_observation(
    tmp_path, monkeypatch, method,
):
    prereg, _, calibration, _ = _bound_paths(tmp_path, monkeypatch)
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload["observations"][0]["selected_method"] = method
    calibration_sha = _write_json(calibration, payload)
    with pytest.raises(policies.MetricCalibrationRequired):
        _load_metric(prereg, calibration, calibration_sha)


def test_missing_metric_row_scale_fails_closed(tmp_path):
    operator = policies.MetricOperator(
        mode=policies.IDENTITY_METRIC,
        calibration={
            "beta_identity_regularizer": 1.0,
            "group_weights_v1": {
                "lambda_boundary": 1.0,
                "lambda_jerk": 1.0,
                "lambda_science": 1.0,
            },
            "rho_E": 1.0e-4,
        },
        calibration_output=str(tmp_path / "unused.json"),
        case_uid="txn:1",
    )
    value = torch.zeros((1, 2), dtype=torch.float32)
    with pytest.raises(policies.MetricKernelUnavailable):
        operator.bind_anchor(
            current=value,
            mask=torch.ones_like(value, dtype=torch.bool),
            taper=torch.ones_like(value),
            gradients={"guard::boundary": torch.ones_like(value)},
            row_base_names={"guard::boundary": "boundary"},
            row_scales={},
            floor=1.0e-8,
        )


def test_float64_metric_shell_normalization_and_geodesic():
    mask = torch.ones((1, 2), dtype=torch.bool)
    kernel = policies.AnchorMetricKernel(
        case_uid="txn:1",
        mask=mask,
        rows=torch.empty((0, 1, 2), dtype=torch.float64),
        beta=1.0,
        trace_scale=1.0,
        rho_g=1.0e-4,
        row_names=(), row_groups=(), row_scales=(), anchor_sha256="test",
    )
    current, resolved = kernel.normalize(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32), 1.0e-4
    )
    assert resolved and current.dtype == torch.float64
    assert abs(kernel.rms(current) - 1.0e-4) <= 1.0e-12
    trial, accepted, audit = kernel.geodesic_update(
        current, torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        0.25, 1.0e-12, direction_is_tangent=True,
    )
    assert accepted and trial.dtype == torch.float64
    assert audit["metric_shell_dtype"] == "float64"
    assert abs(kernel.rms(trial) - kernel.rms(current)) <= 1.0e-12


def test_identity_mode_and_post_selector_eligibility():
    identity = policies.build_metric_operator(policies.IDENTITY_METRIC)
    marker = object()
    assert identity.execute("ignored", lambda: marker) is marker
    audit = {
        "raw_audit": {"passed": True},
        "projector_result": {"projection_backtracking_factor": 1.0},
        "effective_projected_candidate": True,
    }
    assert not policies.is_post_selector_projected_calibration_candidate(
        selected_method="adapter", selected_audit=audit,
        adapter_incumbent_locked=True,
    )
    assert not policies.is_post_selector_projected_calibration_candidate(
        selected_method="identity", selected_audit=audit,
        adapter_incumbent_locked=False,
    )
    assert policies.is_post_selector_projected_calibration_candidate(
        selected_method="geodesic_joint_sqp_k2", selected_audit=audit,
        adapter_incumbent_locked=False,
    )
