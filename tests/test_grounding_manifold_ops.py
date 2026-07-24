import unittest

import numpy as np

from grounding.manifold_ops import (
    bures_distance_sq_np,
    gaussian_wasserstein_distance_sq_np,
    lorentz_distance_sq_np,
    lorentz_inner_np,
    lorentz_project_np,
    mixed_product_distance_sq_np,
    normalized_factor_weights_np,
    project_spd_np,
    sphere_distance_sq_np,
    sphere_project_np,
)


class GroundingManifoldOperatorTests(unittest.TestCase):
    def test_lorentz_projection_satisfies_hyperboloid_constraint(self):
        rng = np.random.default_rng(11)
        spatial = rng.normal(size=(9, 6)).astype(np.float32)
        curvature = 0.7
        points = lorentz_project_np(spatial, curvature)
        np.testing.assert_allclose(
            lorentz_inner_np(points, points),
            -1.0 / curvature,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        self.assertTrue(np.all(points[:, 0] > 0.0))

    def test_lorentz_distance_is_symmetric_and_zero_on_identity(self):
        rng = np.random.default_rng(12)
        left = lorentz_project_np(rng.normal(size=(7, 5)), 1.2)
        right = lorentz_project_np(rng.normal(size=(7, 5)), 1.2)
        np.testing.assert_allclose(
            lorentz_distance_sq_np(left, right, 1.2),
            lorentz_distance_sq_np(right, left, 1.2),
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            lorentz_distance_sq_np(left, left, 1.2), 0.0, atol=2.0e-5
        )

    def test_sphere_distance_matches_quarter_circle(self):
        left = sphere_project_np(np.asarray([[1.0, 0.0, 0.0]]))
        right = sphere_project_np(np.asarray([[0.0, 1.0, 0.0]]))
        np.testing.assert_allclose(
            sphere_distance_sq_np(left, right),
            (np.pi / 2.0) ** 2,
            atol=1.0e-6,
        )

    def test_spd_projection_floors_eigenvalues(self):
        matrix = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
        projected = project_spd_np(matrix, minimum_eigenvalue=0.05)
        eigenvalues = np.linalg.eigvalsh(projected)
        self.assertGreaterEqual(float(eigenvalues.min()), 0.04999)
        np.testing.assert_allclose(projected, projected.T, atol=1.0e-7)

    def test_bures_and_gaussian_wasserstein_contracts(self):
        covariance_a = np.asarray(
            [[[1.2, 0.15], [0.15, 0.8]], [[0.9, 0.05], [0.05, 1.4]]],
            dtype=np.float32,
        )
        covariance_b = np.asarray(
            [[[0.7, -0.1], [-0.1, 1.1]], [[1.3, 0.2], [0.2, 0.75]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            bures_distance_sq_np(covariance_a, covariance_a),
            0.0,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            bures_distance_sq_np(covariance_a, covariance_b),
            bures_distance_sq_np(covariance_b, covariance_a),
            atol=2.0e-5,
        )
        zero = np.zeros((2, 2), dtype=np.float32)
        shifted = np.asarray([[1.0, -2.0], [0.5, 0.5]], dtype=np.float32)
        expected_mean_term = np.sum(shifted**2, axis=-1)
        np.testing.assert_allclose(
            gaussian_wasserstein_distance_sq_np(
                zero, covariance_a, shifted, covariance_a
            ),
            expected_mean_term,
            atol=2.0e-5,
        )

    def test_mixed_product_distance_uses_global_positive_weights(self):
        rng = np.random.default_rng(19)
        covariance = np.broadcast_to(
            np.eye(3, dtype=np.float32), (4, 5, 3, 3)
        ).copy()
        factors = {
            "lorentz": lorentz_project_np(rng.normal(size=(4, 4))),
            "sphere": sphere_project_np(rng.normal(size=(4, 6))),
            "gaussian_mean": rng.normal(size=(4, 5, 3)).astype(np.float32),
            "gaussian_covariance": covariance,
            "euclidean": rng.normal(size=(4, 2)).astype(np.float32),
        }
        weights = normalized_factor_weights_np([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        distance = mixed_product_distance_sq_np(
            factors, factors, weights, curvature=1.0
        )
        np.testing.assert_allclose(distance, 0.0, atol=3.0e-5)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            normalized_factor_weights_np([1.0, 1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
