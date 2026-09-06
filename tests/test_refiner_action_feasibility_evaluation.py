import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.motion_models import MotionGenerationConfig
from training.refiner_action_feasibility_evaluation import (
    MANIFEST_SCHEMA,
    _iteration_summary,
    _sha256,
    _verified_proposal_source,
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
    def test_proposal_requires_the_exact_checkpoint_and_array_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "refiner.pt"
            checkpoint.write_bytes(b"verified-checkpoint")
            proposal = root / "proposal.npy"
            np.save(proposal, np.zeros((8, 151), dtype=np.float32))
            row = {
                "proposal_motion_path": str(proposal),
                "metadata": {
                    "proposal_checkpoint_sha256": _sha256(checkpoint),
                    "proposal_motion_sha256": _sha256(proposal),
                },
            }
            source = _verified_proposal_source(
                row,
                root / "manifest.json",
                {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            )
            self.assertEqual(source["kind"], "motion")
            row["metadata"]["proposal_checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                _verified_proposal_source(
                    row,
                    root / "manifest.json",
                    {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
                )

    def test_iteration_summary_removes_large_probe_payloads(self):
        result = _iteration_summary({
            "iteration": 2,
            "accepted_phase": "temporal",
            "trial_diagnostics": [{"large": [1, 2, 3]}],
            "finite_difference_reachability": [{"probe": 1}, {"probe": 2}],
            "direction_diagnostics": [{"direction": 1}],
        })
        self.assertEqual(result["trial_diagnostics_count"], 1)
        self.assertEqual(result["finite_difference_reachability_count"], 2)
        self.assertNotIn("trial_diagnostics", result)

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
