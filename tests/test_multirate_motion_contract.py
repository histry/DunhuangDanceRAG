import pickle
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from motion_geometry.physical import motion_physical_metrics_np
from motion_geometry.resampling import resample_edge151_np
from motion_geometry.rotations import matrix_to_rot6d_np, rot6d_to_matrix_np, so3_exp_np
from motion_geometry.smpl24 import MOTION_DIM, OFFSETS, PARENTS, skeleton_contract
from retargeting.smpl_adapter import load_smpl24_parameters
from routing.boundary_closed_loop import risk_safe
from support.common import motion_descriptor_raw


def identity_motion(frames: int, fps: float) -> np.ndarray:
    motion = np.zeros((frames, MOTION_DIM), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)[None, None]
    motion[:, 7:151] = np.broadcast_to(
        matrix_to_rot6d_np(identity), (frames, 24, 6)
    ).reshape(frames, -1)
    time = np.arange(frames, dtype=np.float32) / float(fps)
    motion[:, 4] = 0.15 * time
    motion[:, :4] = 1.0
    return motion


class MultirateMotionContractTests(unittest.TestCase):
    def test_smpl24_contract_is_complete(self):
        contract = skeleton_contract()
        self.assertEqual(OFFSETS.shape, (24, 3))
        self.assertEqual(PARENTS.shape, (24,))
        self.assertEqual(contract["motion_dim"], 151)
        self.assertEqual(len(contract["sha256"]), 64)

    def test_edge_resampling_stays_on_so3_and_keeps_contacts_binary(self):
        motion = identity_motion(31, 30.0)
        angles = np.linspace(0.0, np.pi - 1.0e-4, len(motion), dtype=np.float32)
        matrices = so3_exp_np(np.stack([np.zeros_like(angles), angles, np.zeros_like(angles)], axis=-1))
        motion[:, 7:13] = matrix_to_rot6d_np(matrices)
        motion[:, 0] = np.arange(len(motion)) % 2
        out = resample_edge151_np(motion, source_fps=30.0, target_fps=60.0)
        decoded = rot6d_to_matrix_np(out[:, 7:151].reshape(len(out), 24, 6))
        should_be_identity = np.swapaxes(decoded, -1, -2) @ decoded
        self.assertEqual(out.shape, (61, 151))
        self.assertLess(float(np.max(np.abs(should_be_identity - np.eye(3)))), 1.0e-5)
        self.assertTrue(set(np.unique(out[:, :4])).issubset({0.0, 1.0}))
        np.testing.assert_allclose(out[[0, -1], 4:7], motion[[0, -1], 4:7], atol=1.0e-6)

    def test_aistplusplus_fields_and_scaling_are_preserved(self):
        poses = np.zeros((61, 72), dtype=np.float32)
        trans = np.zeros((61, 3), dtype=np.float32)
        trans[:, 0] = np.linspace(0.0, 0.5, 61)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pkl"
            with path.open("wb") as stream:
                pickle.dump(
                    {
                        "smpl_poses": poses,
                        "smpl_trans": trans,
                        "smpl_scaling": np.asarray([1.15], dtype=np.float32),
                    },
                    stream,
                )
            motion, report = load_smpl24_parameters(path, target_fps=30.0)
        self.assertEqual(motion.shape, (31, 151))
        self.assertEqual(report["source_format"], "aistplusplus_smpl")
        self.assertEqual(report["source_fps"], 60.0)
        self.assertAlmostEqual(report["smpl_scaling"], 1.15, places=5)
        self.assertEqual(report["smpl_scaling_mode"], "canonical_body")

    def test_physical_speed_is_stable_between_30_and_60_fps(self):
        motion30 = identity_motion(61, 30.0)
        motion60 = identity_motion(121, 60.0)
        report30 = motion_physical_metrics_np(motion30, fps=30.0)
        report60 = motion_physical_metrics_np(motion60, fps=60.0)
        self.assertAlmostEqual(
            report30["joint_velocity_mps_p95"],
            report60["joint_velocity_mps_p95"],
            places=4,
        )
        self.assertAlmostEqual(
            report30["foot_skate_mps_p95"],
            report60["foot_skate_mps_p95"],
            places=4,
        )

    def test_scheduler_descriptor_uses_physical_time(self):
        motion30 = identity_motion(61, 30.0)
        motion60 = identity_motion(121, 60.0)
        angle30 = np.linspace(0.0, 1.2, len(motion30), dtype=np.float32)
        angle60 = np.linspace(0.0, 1.2, len(motion60), dtype=np.float32)
        rot30 = so3_exp_np(
            np.stack([np.zeros_like(angle30), angle30, np.zeros_like(angle30)], axis=-1)
        )
        rot60 = so3_exp_np(
            np.stack([np.zeros_like(angle60), angle60, np.zeros_like(angle60)], axis=-1)
        )
        motion30[:, 7:13] = matrix_to_rot6d_np(rot30)
        motion60[:, 7:13] = matrix_to_rot6d_np(rot60)
        descriptor30 = motion_descriptor_raw(motion30, fps=30.0)
        descriptor60 = motion_descriptor_raw(motion60, fps=60.0)
        np.testing.assert_allclose(
            descriptor30,
            descriptor60,
            rtol=2.0e-3,
            atol=2.0e-3,
        )

    def test_boundary_gate_reads_only_explicit_si_thresholds(self):
        risk = {
            "boundary_joint_jerk_max": 700.0,
            "exit_fk_jump": 0.010,
            "exit_rotation_step_rad": 0.040,
            "foot_slip": 0.040,
            "foot_penetration": 0.0001,
        }
        environment = {
            "V46_46_MAX_BOUNDARY_JERK_MPS3": "800.0",
            "V46_46_MAX_EXIT_FK_JUMP_M": "0.015",
            "V46_46_MAX_EXIT_ROT_RAD": "0.08",
            "V46_46_MAX_FOOT_SLIP_MPS": "0.06",
            "V46_46_MAX_FOOT_PENETRATION_M2": "0.001",
            # This ambiguous legacy value must not change the SI gate.
            "V46_46_MAX_BOUNDARY_JERK": "1.0",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            self.assertTrue(risk_safe(risk))

        environment["V46_46_MAX_BOUNDARY_JERK_MPS3"] = "600.0"
        with mock.patch.dict(os.environ, environment, clear=False):
            self.assertFalse(risk_safe(risk))


if __name__ == "__main__":
    unittest.main()
