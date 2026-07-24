import unittest

import numpy as np

from events.intrinsic_geometry import (
    GAUSSIAN_BODY_PARTS,
    GAUSSIAN_FEATURE_DIM,
    _bodypart_gaussian_statistics,
)
from motion_geometry.rotations import so3_exp_np
from motion_geometry.smpl24 import NUM_JOINTS


class BodyPartGaussianGeometryTests(unittest.TestCase):
    def test_statistics_are_spd_and_have_stable_contract(self):
        rng = np.random.default_rng(29)
        frames = 48
        omega = rng.normal(
            0.0, 0.4, size=(frames - 1, NUM_JOINTS, 3)
        ).astype(np.float32)
        alpha = np.diff(omega, axis=0).astype(np.float32) * 30.0
        mean, covariance, samples = _bodypart_gaussian_statistics(
            omega,
            alpha,
            shrinkage=0.25,
            minimum_eigenvalue=1.0e-4,
        )
        parts = len(GAUSSIAN_BODY_PARTS)
        self.assertEqual(mean.shape, (parts, GAUSSIAN_FEATURE_DIM))
        self.assertEqual(
            covariance.shape,
            (parts, GAUSSIAN_FEATURE_DIM, GAUSSIAN_FEATURE_DIM),
        )
        self.assertEqual(samples.shape, (parts,))
        self.assertTrue(np.all(samples == frames - 2))
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(
            covariance,
            np.swapaxes(covariance, -1, -2),
            atol=1.0e-6,
        )
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(covariance).min()), 0.000099
        )

    def test_degenerate_clip_uses_explicit_floor_covariance(self):
        omega = np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
        alpha = np.zeros((0, NUM_JOINTS, 3), dtype=np.float32)
        mean, covariance, samples = _bodypart_gaussian_statistics(
            omega, alpha, minimum_eigenvalue=0.01
        )
        np.testing.assert_allclose(mean, 0.0)
        np.testing.assert_array_equal(samples, 0)
        expected = np.broadcast_to(
            np.eye(GAUSSIAN_FEATURE_DIM, dtype=np.float32) * 0.01,
            covariance.shape,
        )
        np.testing.assert_allclose(covariance, expected, atol=1.0e-7)

    def test_rotation_signal_produces_nonzero_mean(self):
        frames = 24
        tangent = np.zeros((frames, NUM_JOINTS, 3), dtype=np.float32)
        tangent[:, 0, 1] = np.linspace(0.0, 0.6, frames)
        rotations = so3_exp_np(tangent)
        # Construct physical derivatives exactly as the production path does.
        from motion_geometry.rotations import (
            angular_acceleration_np,
            angular_velocity_np,
        )

        omega = angular_velocity_np(rotations, fps=30.0)
        alpha = angular_acceleration_np(rotations, fps=30.0)
        mean, _, _ = _bodypart_gaussian_statistics(omega, alpha)
        self.assertGreater(float(np.linalg.norm(mean[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
