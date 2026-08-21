#!/usr/bin/env python3
import unittest
import numpy as np

from contracts.gravity import identity6d_np
from motion_geometry.smpl24 import MOTION_DIM, NUM_JOINTS, ROT6D_START, ROT6D_END
from retargeting.official_smpl_source_preprocess import (
    HardCutPolicy,
    hard_cut_analysis,
)


def synthetic_motion(frames=120):
    x = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    x[:, ROT6D_START:ROT6D_END] = identity6d_np(
        (frames, NUM_JOINTS)
    ).reshape(frames, -1)
    x[:, 4] = np.linspace(0.0, 0.30, frames, dtype=np.float32)
    x[:, 5] = 1.0
    return x


class OfficialSmplHardCutTest(unittest.TestCase):
    def test_smooth_motion_is_not_cut(self):
        x = synthetic_motion()
        result = hard_cut_analysis(x, fps=60.0, policy=HardCutPolicy())
        self.assertEqual(result["cut_boundaries"], [])

    def test_root_teleport_creates_boundary(self):
        x = synthetic_motion()
        x[60:, 4] += 1.0
        result = hard_cut_analysis(x, fps=60.0, policy=HardCutPolicy())
        self.assertIn(60, result["cut_boundaries"])

    def test_nonfinite_frame_is_isolated(self):
        x = synthetic_motion()
        x[40, 4] = np.nan
        result = hard_cut_analysis(x, fps=60.0, policy=HardCutPolicy())
        self.assertIn(40, result["cut_boundaries"])
        self.assertIn(41, result["cut_boundaries"])


if __name__ == "__main__":
    unittest.main()
