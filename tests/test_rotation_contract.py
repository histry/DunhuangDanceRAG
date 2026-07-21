#!/usr/bin/env python3
import unittest
import numpy as np

import motion_geometry.rotations as rotation_module
from motion_geometry.rotations import (
    CANONICAL_ROT6D_LAYOUT,
    ROT6D_LAYOUT_PYTORCH3D_ROW,
    convert_motion_rot6d_layout_np,
    convert_rot6d_layout_np,
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    rot6d_to_matrix_layout_np,
    so3_exp_np,
    so3_geodesic_np,
    so3_log_np,
    so3_log_torch,
)
from motion_geometry.heading import resample_motion_so3, root_yaw_np


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

    def test_near_pi_log_exp_roundtrip(self):
        axes = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, -0.7, 0.4]],
            dtype=np.float32,
        )
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        angles = np.asarray([np.pi, np.pi - 1.0e-6, np.pi - 5.0e-5], dtype=np.float32)
        rotations = so3_exp_np(axes * angles[:, None])
        recovered = so3_exp_np(so3_log_np(rotations))
        error = so3_geodesic_np(rotations, recovered)
        self.assertLess(float(error.max()), 2.0e-4)

    def test_near_pi_numpy_fallback_log_exp_roundtrip(self):
        axes = np.asarray(
            [[1.0, 0.0, 0.0], [0.3, -0.4, 0.5]],
            dtype=np.float32,
        )
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        angles = np.asarray([np.pi, np.pi - 1.0e-6], dtype=np.float32)
        rotations = so3_exp_np(axes * angles[:, None])
        scipy_rotation = rotation_module.Rotation
        try:
            rotation_module.Rotation = None
            recovered = so3_exp_np(rotation_module.so3_log_np(rotations))
        finally:
            rotation_module.Rotation = scipy_rotation
        error = so3_geodesic_np(rotations, recovered)
        self.assertLess(float(error.max()), 3.0e-4)

    def test_torch_near_pi_log_is_not_zero(self):
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is not installed in the lightweight test runtime")
        rotations = torch.as_tensor(
            so3_exp_np(
                np.asarray(
                    [[np.pi, 0.0, 0.0], [0.0, np.pi, 0.0]],
                    dtype=np.float32,
                )
            )
        )
        value = so3_log_torch(rotations)
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool((torch.linalg.vector_norm(value, dim=-1) > 3.0).all()))

    def test_legacy_row_adapter_preserves_physical_rotation(self):
        rng = np.random.default_rng(17)
        q, _ = np.linalg.qr(rng.normal(size=(64, 3, 3)))
        q[np.linalg.det(q) < 0, :, -1] *= -1.0
        canonical = matrix_to_rot6d_np(q.astype(np.float32))
        legacy = convert_rot6d_layout_np(
            canonical,
            CANONICAL_ROT6D_LAYOUT,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
        )
        recovered = rot6d_to_matrix_layout_np(
            legacy,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
        )
        error = so3_geodesic_np(q.astype(np.float32), recovered)
        self.assertLess(float(error.max()), 1.0e-4)

    def test_motion_layout_roundtrip_preserves_non_rotation_channels(self):
        motion = np.zeros((3, 151), dtype=np.float32)
        motion[:, :7] = np.arange(21, dtype=np.float32).reshape(3, 7)
        rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (3, 24, 3, 3)).copy()
        rotations[1, 0] = so3_exp_np(np.asarray([0.3, -0.5, 0.7], dtype=np.float32))
        motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(3, 144)
        legacy = convert_motion_rot6d_layout_np(
            motion,
            CANONICAL_ROT6D_LAYOUT,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
        )
        restored = convert_motion_rot6d_layout_np(
            legacy,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
            CANONICAL_ROT6D_LAYOUT,
        )
        np.testing.assert_array_equal(restored[:, :7], motion[:, :7])
        original_matrix = rot6d_to_matrix_np(motion[:, 7:151].reshape(3, 24, 6))
        restored_matrix = rot6d_to_matrix_np(restored[:, 7:151].reshape(3, 24, 6))
        self.assertLess(float(so3_geodesic_np(original_matrix, restored_matrix).max()), 1.0e-4)

    def test_heading_resampling_uses_column_contract(self):
        motion = np.zeros((2, 151), dtype=np.float32)
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (2, 24, 3, 3),
        ).copy()
        rotations[1, 0] = so3_exp_np(
            np.asarray([0.0, np.pi / 2.0, 0.0], dtype=np.float32)
        )
        motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(2, 144)
        result = resample_motion_so3(
            motion,
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        )
        yaw = root_yaw_np(result)
        self.assertAlmostEqual(float(yaw[1]), np.pi / 4.0, places=4)
        self.assertAlmostEqual(float(yaw[-1]), np.pi / 2.0, places=4)

    def test_legacy_and_canonical_resampling_are_physically_equivalent(self):
        rng = np.random.default_rng(19)
        tangent = rng.normal(size=(5, 24, 3)).astype(np.float32)
        tangent *= 0.4 / np.maximum(
            np.linalg.norm(tangent, axis=-1, keepdims=True),
            1.0e-6,
        )
        canonical = np.zeros((5, 151), dtype=np.float32)
        canonical[:, :7] = rng.normal(size=(5, 7)).astype(np.float32)
        canonical[:, 7:151] = matrix_to_rot6d_np(so3_exp_np(tangent)).reshape(5, 144)
        legacy = convert_motion_rot6d_layout_np(
            canonical,
            CANONICAL_ROT6D_LAYOUT,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
        )
        tau = np.linspace(0.0, 1.0, 9, dtype=np.float32)
        canonical_result = resample_motion_so3(canonical, tau)
        legacy_result = resample_motion_so3(
            legacy,
            tau,
            rot6d_layout=ROT6D_LAYOUT_PYTORCH3D_ROW,
        )
        legacy_as_canonical = convert_motion_rot6d_layout_np(
            legacy_result,
            ROT6D_LAYOUT_PYTORCH3D_ROW,
            CANONICAL_ROT6D_LAYOUT,
        )
        np.testing.assert_allclose(
            canonical_result[:, :7],
            legacy_as_canonical[:, :7],
            atol=1.0e-6,
        )
        expected = rot6d_to_matrix_np(canonical_result[:, 7:151].reshape(-1, 24, 6))
        actual = rot6d_to_matrix_np(legacy_as_canonical[:, 7:151].reshape(-1, 24, 6))
        self.assertLess(float(so3_geodesic_np(expected, actual).max()), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
