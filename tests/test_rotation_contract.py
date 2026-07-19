#!/usr/bin/env python3
import unittest
import numpy as np

from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    so3_geodesic_np,
    so3_log_np,
)


class RotationContractTest(unittest.TestCase):
    def test_column_concat_roundtrip(self):
        rng = np.random.default_rng(20260717)
        q, _ = np.linalg.qr(rng.normal(size=(256, 3, 3)))
        det = np.linalg.det(q)
        q[det < 0, :, -1] *= -1.0
        six = matrix_to_rot6d_np(q.astype(np.float32))
        rec = rot6d_to_matrix_np(six)
        err = so3_geodesic_np(q.astype(np.float32), rec)
        self.assertLess(float(err.max()), 1.0e-4)

    def test_log_exp_roundtrip(self):
        rng = np.random.default_rng(42)
        v = rng.normal(size=(128, 3)).astype(np.float32)
        v *= (rng.uniform(0.0, 2.5, size=(128, 1)) / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)).astype(np.float32)
        r = so3_exp_np(v)
        rec = so3_exp_np(so3_log_np(r))
        err = so3_geodesic_np(r, rec)
        self.assertLess(float(err.max()), 2.0e-4)


if __name__ == "__main__":
    unittest.main()
