import unittest
import numpy as np

from motion_geometry.rotations import project_to_so3_np, so3_geodesic_np


class RotationFusionTests(unittest.TestCase):
    def test_matrix_average_projection_is_proper_rotation(self):
        a = np.eye(3, dtype=np.float32)
        b = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        r = project_to_so3_np(0.5 * a + 0.5 * b)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=5)
        self.assertLess(float(so3_geodesic_np(a, r)), np.pi / 2)
        self.assertLess(float(so3_geodesic_np(r, b)), np.pi / 2)


if __name__ == "__main__":
    unittest.main()
