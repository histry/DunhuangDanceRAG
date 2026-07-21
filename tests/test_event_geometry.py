#!/usr/bin/env python3
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motion_geometry.rotations import matrix_to_rot6d_np
from events.intrinsic_geometry import augment_database


def motion(frames: int, shift: float) -> np.ndarray:
    x = np.zeros((frames, 151), dtype=np.float32)
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 24, 3, 3))
    x[:, 7:151] = matrix_to_rot6d_np(eye).reshape(frames, -1)
    x[:, 4] = np.linspace(0.0, shift, frames)
    x[:, 5] = 0.95
    x[:, :4] = 1.0
    return x


class EventGeometryTest(unittest.TestCase):
    def test_schema_preserving_augmentation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = []
            for i in range(4):
                p = root / f"event_{i}.npy"
                np.save(p, motion(30 + i, 0.02 * i))
                paths.append(str(p))
            db = root / "events.npz"
            np.savez_compressed(
                db,
                paths=np.asarray(paths, dtype=object),
                source_uids=np.asarray(["a", "a", "b", "b"], dtype=object),
                posture_mode=np.asarray(["standing"] * 4, dtype=object),
                event_families=np.asarray(["pose", "pose", "flow", "flow"], dtype=object),
                motion_stage_roles=np.asarray(["intro", "development", "development", "resolution"], dtype=object),
                anatomy_quality=np.asarray([0.9, 0.8, 0.85, 0.75], dtype=np.float32),
                event_quality_scores=np.asarray([0.8, 0.7, 0.75, 0.65], dtype=np.float32),
                desc=np.zeros((4, 32), dtype=np.float32),
                canonical_fps=np.full(4, 30.0, dtype=np.float32),
            )
            rep = augment_database(db)
            with self.assertRaisesRegex(RuntimeError, "FPS mismatch"):
                augment_database(db, fps=60.0)
            with np.load(db, allow_pickle=True) as obj:
                self.assertIn("v46_53_geometry_desc_z", obj.files)
                self.assertIn("v46_53_shared_embedding", obj.files)
                self.assertEqual(obj["v46_53_geometry_desc_z"].shape[0], 4)
            self.assertTrue(rep["ok"])


if __name__ == "__main__":
    unittest.main()
