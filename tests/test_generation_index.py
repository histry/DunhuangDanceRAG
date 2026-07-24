import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scheduling.build_generation_index import build_generation_index
from scheduling.index_io import load_shared_index
from motion_geometry.smpl24 import skeleton_contract_json


def identity_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 5] = 1.0
    rot = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:] = np.tile(rot, 24)
    return motion


class GenerationIndexTests(unittest.TestCase):
    def test_builder_preserves_generation_order_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, frames in enumerate((24, 30)):
                path = root / f"event_{index}.npy"
                np.save(path, identity_motion(frames))
                paths.append(str(path))
            db_path = root / "events_aesd.npz"
            np.savez_compressed(
                db_path,
                paths=np.asarray(paths, dtype=object),
                source_uids=np.asarray(["source_a", "source_b"], dtype=object),
                source_files=np.asarray(["a.bvh", "b.bvh"], dtype=object),
                starts=np.asarray([0, 10]),
                ends=np.asarray([24, 40]),
                frames=np.asarray([24, 30]),
                event_families=np.asarray(["pose", "turn"], dtype=object),
                dance_keys=np.asarray(["d0", "d1"], dtype=object),
                posture_entry=np.asarray(["standing", "half_squat"], dtype=object),
                posture_exit=np.asarray(["standing", "standing"], dtype=object),
                posture_mode=np.asarray(["standing", "half_squat"], dtype=object),
                aesd_event_semantics=np.asarray(["pose_hold", "turning_climax"], dtype=object),
                canonical_fps=np.asarray([30.0, 30.0], dtype=np.float32),
                source_start_seconds=np.asarray([0.0, 10.0 / 30.0], dtype=np.float64),
                source_end_seconds=np.asarray([24.0 / 30.0, 40.0 / 30.0], dtype=np.float64),
                skeleton_contract_json=np.asarray(skeleton_contract_json(), dtype=object),
            )
            json_path = root / "index.json"
            npz_path = root / "index.npz"
            report = build_generation_index(db_path, json_path, npz_path)
            metadata, arrays, items = load_shared_index(json_path, npz_path)
            try:
                self.assertEqual(report["num_events"], 2)
                self.assertEqual(len(set(arrays["event_uids"].tolist())), 2)
                self.assertEqual(items[0]["source_uid"], "source_a")
                self.assertEqual(metadata["event_db_contract"], report["event_db_contract"])
                self.assertEqual(metadata["canonical_fps_values"], [30.0])
                self.assertEqual(
                    metadata["natural_duration_units"],
                    "frames_at_canonical_fps",
                )
                self.assertEqual(str(metadata["rot6d_layout"]), "column")
            finally:
                arrays.close()

    def test_builder_rejects_missing_posture_endpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_path = root / "event.npy"
            np.save(motion_path, identity_motion(24))
            db_path = root / "events_aesd.npz"
            np.savez_compressed(
                db_path,
                paths=np.asarray([str(motion_path)], dtype=object),
                source_uids=np.asarray(["source_a"], dtype=object),
                starts=np.asarray([0]),
                ends=np.asarray([24]),
                frames=np.asarray([24]),
                canonical_fps=np.asarray([30.0], dtype=np.float32),
                skeleton_contract_json=np.asarray(
                    skeleton_contract_json(),
                    dtype=object,
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "posture state fields"):
                build_generation_index(
                    db_path,
                    root / "index.json",
                    root / "index.npz",
                )

    def test_builder_rejects_database_without_canonical_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_path = root / "event.npy"
            np.save(motion_path, identity_motion(24))
            db_path = root / "events_aesd.npz"
            np.savez_compressed(
                db_path,
                paths=np.asarray([str(motion_path)], dtype=object),
                source_uids=np.asarray(["source_a"], dtype=object),
                starts=np.asarray([0]),
                ends=np.asarray([24]),
                frames=np.asarray([24]),
                skeleton_contract_json=np.asarray(
                    skeleton_contract_json(),
                    dtype=object,
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "no canonical_fps contract"):
                build_generation_index(
                    db_path,
                    root / "index.json",
                    root / "index.npz",
                )


if __name__ == "__main__":
    unittest.main()
