#!/usr/bin/env python
"""Fast, CPU-only fail-closed contract checks for g1f4 M-v1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from training import refiner_v15_15g1f4_policies as policies


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preregistration() -> dict:
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


def calibration(prereg_sha: str) -> dict:
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


class G1F4MContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.previous_commit = os.environ.get("EXPECTED_COMMIT")
        os.environ["EXPECTED_COMMIT"] = COMMIT
        self.addCleanup(self._restore_commit)
        self.prereg = Path(self.tmp.name) / "prereg.json"
        self.prereg_sha = write_json(self.prereg, preregistration())
        self.calibration = Path(self.tmp.name) / "calibration.json"
        self.calibration_sha = write_json(
            self.calibration, calibration(self.prereg_sha)
        )

    def _restore_commit(self):
        if self.previous_commit is None:
            os.environ.pop("EXPECTED_COMMIT", None)
        else:
            os.environ["EXPECTED_COMMIT"] = self.previous_commit

    def load(self, sha=None):
        return policies.build_metric_operator(
            policies.ANCHOR_KINEMATIC_METRIC,
            calibration_path=self.calibration,
            calibration_sha256=self.calibration_sha if sha is None else sha,
            preregistration_path=self.prereg,
        )

    def mutate_and_expect_rejection(self, mutate):
        payload = json.loads(self.calibration.read_text(encoding="utf-8"))
        mutate(payload)
        self.calibration_sha = write_json(self.calibration, payload)
        with self.assertRaises(policies.MetricCalibrationRequired):
            self.load()

    def test_01_calibration_sha_mismatch(self):
        with self.assertRaises(policies.MetricCalibrationRequired):
            self.load("0" * 64)

    def test_02_to_10_loader_binding_rejections(self):
        mutations = {
            "prereg_sha": lambda p: p.__setitem__("preregistered_contract_sha256", "bad"),
            "implementation": lambda p: p.__setitem__("implementation_commit", "bad"),
            "rho_e": lambda p: p.__setitem__("rho_E", 2.0e-4),
            "beta": lambda p: p.__setitem__("beta_identity_regularizer", 2.0),
            "weights": lambda p: p.__setitem__("group_weights_v1", {"lambda_boundary": 2.0, "lambda_jerk": 1.0, "lambda_science": 1.0}),
            "non_train": lambda p: p.__setitem__("calibration_split", "development"),
            "source": lambda p: p.__setitem__("calibration_source", "pre_selector"),
            "identity_observation": lambda p: p["observations"][0].__setitem__("selected_method", "identity"),
            "adapter_observation": lambda p: p["observations"][0].__setitem__("selected_method", "adapter"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = calibration(self.prereg_sha)
                mutate(payload)
                self.calibration_sha = write_json(self.calibration, payload)
                with self.assertRaises(policies.MetricCalibrationRequired):
                    self.load()

    def test_11_missing_metric_scale_fails_closed(self):
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
            calibration_output=str(Path(self.tmp.name) / "unused.json"),
            case_uid="txn:1",
        )
        value = torch.zeros((1, 2), dtype=torch.float32)
        with self.assertRaises(policies.MetricKernelUnavailable):
            operator.bind_anchor(
                current=value, mask=torch.ones_like(value, dtype=torch.bool),
                taper=torch.ones_like(value),
                gradients={"guard::boundary": torch.ones_like(value)},
                row_base_names={"guard::boundary": "boundary"},
                row_scales={}, floor=1.0e-8,
            )

    def test_12_and_13_float64_normalize_and_geodesic(self):
        mask = torch.ones((1, 2), dtype=torch.bool)
        kernel = policies.AnchorMetricKernel(
            case_uid="txn:1", mask=mask,
            rows=torch.empty((0, 1, 2), dtype=torch.float64),
            beta=1.0, trace_scale=1.0, rho_g=1.0e-4,
            row_names=(), row_groups=(), row_scales=(), anchor_sha256="test",
        )
        current, resolved = kernel.normalize(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32), 1.0e-4
        )
        self.assertTrue(resolved)
        self.assertEqual(current.dtype, torch.float64)
        self.assertLessEqual(abs(kernel.rms(current) - 1.0e-4), 1.0e-12)
        trial, accepted, audit = kernel.geodesic_update(
            current, torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            0.25, 1.0e-12, direction_is_tangent=True,
        )
        self.assertTrue(accepted)
        self.assertEqual(trial.dtype, torch.float64)
        self.assertEqual(audit["metric_shell_dtype"], "float64")
        self.assertLessEqual(abs(kernel.rms(trial) - kernel.rms(current)), 1.0e-12)

    def test_14_identity_path_is_unchanged(self):
        marker = object()
        operator = policies.build_metric_operator(policies.IDENTITY_METRIC)
        self.assertIs(operator.execute("ignored", lambda: marker), marker)

    def test_15_selector_eligibility(self):
        audit = {
            "raw_audit": {"passed": True},
            "projector_result": {"projection_backtracking_factor": 1.0},
            "effective_projected_candidate": True,
        }
        self.assertFalse(policies.is_post_selector_projected_calibration_candidate(
            selected_method="adapter", selected_audit=audit,
            adapter_incumbent_locked=True,
        ))
        self.assertFalse(policies.is_post_selector_projected_calibration_candidate(
            selected_method="identity", selected_audit=audit,
            adapter_incumbent_locked=False,
        ))
        self.assertTrue(policies.is_post_selector_projected_calibration_candidate(
            selected_method="geodesic_joint_sqp_k2", selected_audit=audit,
            adapter_incumbent_locked=False,
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
