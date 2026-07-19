#!/usr/bin/env python3
from __future__ import annotations

import unittest
import numpy as np

from contracts.gravity import identity6d_np, rot6d_to_matrix_np
from contracts.anatomy import (
    anatomy_metrics_np,
    evaluate_anatomy_contract,
    geodesic_c2_bridge_np,
)


def identity_motion(T=40):
    x = np.zeros((T, 151), dtype=np.float32)
    x[:, 7:151] = identity6d_np((T, 24)).reshape(T, -1)
    # Put the root high enough for feet to remain around zero.
    x[:, 5] = 0.93
    return x


class TestV4652Anatomy(unittest.TestCase):
    def test_identity_contract(self):
        x = identity_motion()
        metrics = anatomy_metrics_np(x)
        ok, reasons = evaluate_anatomy_contract(metrics)
        self.assertTrue(ok, reasons)
        self.assertLess(metrics["bone_length_drift_max"], 1e-5)

    def test_degenerate_rot6d_is_detected_or_projected(self):
        x = identity_motion()
        x[10:20, 7:13] = 0.0
        metrics = anatomy_metrics_np(x)
        # The canonical converter projects degeneracy to identity. This test
        # ensures the result remains finite and the FK contract does not explode.
        self.assertEqual(metrics["nonfinite_count"], 0)
        self.assertTrue(np.isfinite(metrics["anatomy_quality"]))

    def test_penetration_uses_absolute_stage_floor(self):
        x = identity_motion(100)
        # One frame penetrates the y=0 stage floor by 5 cm.  A percentile-centered
        # floor would hide or distort this value; the world-floor metric must not.
        x[0, 5] -= 0.05
        metrics = anatomy_metrics_np(x)
        self.assertEqual(metrics["floor_reference_mode"], "stage_zero")
        self.assertAlmostEqual(metrics["stage_floor_y"], 0.0, places=6)
        self.assertLess(metrics["foot_penetration_min_m"], -0.04)

    def test_geodesic_bridge(self):
        a = identity_motion(4)
        b = identity_motion(4)
        b[:, 4] = 0.6
        bridge = geodesic_c2_bridge_np(a, b, 24)
        self.assertEqual(bridge.shape, (24, 151))
        mats = rot6d_to_matrix_np(bridge[:, 7:151].reshape(24, 24, 6))
        det = np.linalg.det(mats)
        self.assertLess(float(np.max(np.abs(det - 1.0))), 1e-4)
        self.assertTrue(np.isfinite(bridge).all())


if __name__ == "__main__":
    unittest.main()
