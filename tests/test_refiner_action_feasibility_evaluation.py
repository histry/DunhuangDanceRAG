import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.motion_models import MotionGenerationConfig
from training.refiner_action_feasibility_evaluation import (
    MANIFEST_SCHEMA,
    load_case_manifest,
)


def _write_case(root: Path, case_id: str, recording: str, split: str = "dev") -> None:
    frames = 8
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 7:] = np.tile(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), 24
    )
    np.save(root / f"{case_id}-reference.npy", motion)
    np.save(root / f"{case_id}-seam.npy", np.r_[np.zeros(2), np.ones(4), np.zeros(2)])
    np.save(root / f"{case_id}-joint.npy", np.zeros((frames, 24), dtype=np.float32))
    np.save(root / f"{case_id}-root.npy", np.zeros(frames, dtype=np.float32))
    np.save(root / f"{case_id}-contact.npy", np.zeros((frames, 4), dtype=np.float32))


class RefinerActionFeasibilityEvaluationTests(unittest.TestCase):
    def test_manifest_loads_explicit_development_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "case-a", "recording-a")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": MANIFEST_SCHEMA,
                        "cases": [
                            {
                                "case_id": "case-a",
                                "role": "single_recording",
                                "width": 4,
                                "position_stratum": "seen",
                                "split": "dev",
                                "source_uid": "source-a",
                                "recording_uid": "recording-a",
                                "reference_path": "case-a-reference.npy",
                                "seam_path": "case-a-seam.npy",
                                "joint_mask_path": "case-a-joint.npy",
                                "root_mask_path": "case-a-root.npy",
                                "contact_mask_path": "case-a-contact.npy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cases, info = load_case_manifest(root / "manifest.json", MotionGenerationConfig(device="cpu"))
            self.assertEqual(len(cases), 1)
            self.assertEqual(info["recordings"], 1)

    def test_test_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "case-a", "recording-a", split="test")
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "cases": [{
                    "case_id": "case-a", "role": "single_recording", "width": 4,
                    "position_stratum": "seen", "split": "test",
                    "recording_uid": "recording-a", "reference_path": "case-a-reference.npy",
                    "seam_path": "case-a-seam.npy", "joint_mask_path": "case-a-joint.npy",
                    "root_mask_path": "case-a-root.npy", "contact_mask_path": "case-a-contact.npy",
                }],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_case_manifest(path, MotionGenerationConfig(device="cpu"))

    def test_recording_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_case(root, "case-a", "recording-shared")
            _write_case(root, "case-b", "recording-shared")
            rows = []
            for case_id, split in (("case-a", "dev"), ("case-b", "train")):
                rows.append({
                    "case_id": case_id, "role": "single_recording", "width": 4,
                    "position_stratum": "seen", "split": split,
                    "recording_uid": "recording-shared",
                    "reference_path": f"{case_id}-reference.npy",
                    "seam_path": f"{case_id}-seam.npy",
                    "joint_mask_path": f"{case_id}-joint.npy",
                    "root_mask_path": f"{case_id}-root.npy",
                    "contact_mask_path": f"{case_id}-contact.npy",
                })
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema": MANIFEST_SCHEMA, "cases": rows}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_case_manifest(path, MotionGenerationConfig(device="cpu"))


if __name__ == "__main__":
    unittest.main()
