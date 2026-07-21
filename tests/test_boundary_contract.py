#!/usr/bin/env python3
import unittest
import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np, so3_exp_np
from contracts.boundary import (
    build_frame_joint_risk_mask,
    tangent_masked_merge,
    transition_multiscale_risk,
)


def identity_motion(frames: int = 40) -> np.ndarray:
    x = np.zeros((frames, 151), dtype=np.float32)
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 24, 3, 3))
    x[:, 7:151] = matrix_to_rot6d_np(eye).reshape(frames, -1)
    x[:, 5] = 0.95
    x[:, 0:4] = 1.0
    return x


class BoundaryContractTest(unittest.TestCase):
    def test_mask_and_tangent_merge(self):
        ref = identity_motion(60)
        proposal = ref.copy()
        rot = np.broadcast_to(np.eye(3, dtype=np.float32), (60, 24, 3, 3)).copy()
        rot[25:35, 18] = so3_exp_np(np.asarray([0.0, 0.0, 0.4], np.float32))
        proposal[:, 7:151] = matrix_to_rot6d_np(rot).reshape(60, -1)
        seam = np.zeros((60, 1), dtype=np.float32)
        seam[20:40] = 1.0
        masks = build_frame_joint_risk_mask(proposal, seam)
        self.assertEqual(np.asarray(masks["joint"]).shape, (60, 24))
        merged = tangent_masked_merge(ref, proposal, masks)
        self.assertEqual(merged.shape, ref.shape)
        self.assertTrue(np.isfinite(merged).all())
        # Frames outside the editable seam remain exactly preserved.
        self.assertTrue(np.allclose(merged[:15], ref[:15], atol=1e-6))

    def test_transition_report(self):
        a = identity_motion(20)
        b = identity_motion(20)
        b[:, 5] += 0.05
        report = transition_multiscale_risk(a, np.zeros((0, 151), np.float32), b)
        self.assertIn("parts", report)
        self.assertIn("root_y_gap_m", report)
        self.assertTrue(np.isfinite(float(report["score"])))

    def test_physical_windows_scale_with_fps(self):
        a30 = identity_motion(30)
        a60 = identity_motion(60)
        report30 = transition_multiscale_risk(
            a30, np.zeros((0, 151), np.float32), a30, fps=30.0
        )
        report60 = transition_multiscale_risk(
            a60, np.zeros((0, 151), np.float32), a60, fps=60.0
        )
        self.assertEqual(report30["tangent_window_frames"], 8)
        self.assertEqual(report60["tangent_window_frames"], 16)
        self.assertAlmostEqual(
            report30["tangent_window_seconds"],
            report60["tangent_window_seconds"],
        )


if __name__ == "__main__":
    unittest.main()
