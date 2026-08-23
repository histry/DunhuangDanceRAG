import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motion_geometry.smpl24 import skeleton_contract
from support.event_identity import make_event_db_contract
from support.scheduler_checkpoint_contracts import (
    assert_scheduler_checkpoint_contract,
    scheduler_training_contract,
)


class SchedulerTrainingContractTests(unittest.TestCase):
    def _fixture(self, root: Path, dataset_contract: dict):
        index_json = root / "event_index.json"
        index_npz = root / "duration_index.npz"
        dataset = root / "router_training.npz"
        index_json.write_text("{}", encoding="utf-8")
        np.savez(index_npz, placeholder=np.zeros((1,), dtype=np.float32))
        np.savez(
            dataset,
            fps=np.asarray(30.0, dtype=np.float32),
            event_db_contract_json=np.asarray(
                json.dumps(dataset_contract, sort_keys=True), dtype=object
            ),
        )
        return index_json, index_npz, dataset

    def test_training_contract_binds_dataset_to_ordered_event_db(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, expected)
            contract = scheduler_training_contract(
                role="router",
                fps=30.0,
                index_metadata={
                    "canonical_fps_values": [30.0],
                    "event_db_contract": expected,
                    "skeleton_contract": skeleton_contract(),
                },
                index_json=index_json,
                index_npz=index_npz,
                dataset=dataset,
            )
        self.assertEqual(expected, contract["event_db_contract"])
        self.assertEqual(30.0, contract["fps"])

    def test_stale_dataset_event_order_is_rejected(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        stale = make_event_db_contract(["evt_b", "evt_a"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, stale)
            with self.assertRaisesRegex(RuntimeError, "event DB contract mismatch"):
                scheduler_training_contract(
                    role="router",
                    fps=30.0,
                    index_metadata={
                        "canonical_fps_values": [30.0],
                        "event_db_contract": expected,
                        "skeleton_contract": skeleton_contract(),
                    },
                    index_json=index_json,
                    index_npz=index_npz,
                    dataset=dataset,
                )

    def test_runtime_rejects_descriptor_index_content_drift(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, expected)
            contract = scheduler_training_contract(
                role="router",
                fps=30.0,
                index_metadata={
                    "canonical_fps_values": [30.0],
                    "event_db_contract": expected,
                    "skeleton_contract": skeleton_contract(),
                },
                index_json=index_json,
                index_npz=index_npz,
                dataset=dataset,
            )
            checkpoint = {"fps": 30.0, "scheduler_contract": contract}
            index_json.write_text('{"changed": true}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "index JSON hash mismatch"):
                assert_scheduler_checkpoint_contract(
                    checkpoint,
                    role="router",
                    runtime_fps=30.0,
                    event_db_contract=expected,
                    index_json=index_json,
                    index_npz=index_npz,
                )

    def test_cross_commit_reuse_accepts_unchanged_role_model_implementation(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, expected)
            contract = scheduler_training_contract(
                role="router",
                fps=30.0,
                index_metadata={
                    "canonical_fps_values": [30.0],
                    "event_db_contract": expected,
                    "skeleton_contract": skeleton_contract(),
                },
                index_json=index_json,
                index_npz=index_npz,
                dataset=dataset,
            )
            checkpoint_code = dict(contract["code_provenance"])
            checkpoint_code.update(
                commit="1" * 40,
                worktree_clean=True,
                status_sha256="clean",
                status_entries=0,
            )
            contract["code_provenance"] = checkpoint_code
            checkpoint = {"fps": 30.0, "scheduler_contract": contract}
            runtime_code = {
                **checkpoint_code,
                "commit": "2" * 40,
            }
            with patch(
                "support.scheduler_checkpoint_contracts.repository_code_provenance",
                return_value=runtime_code,
            ), patch(
                "support.scheduler_checkpoint_contracts._changed_runtime_model_files",
                return_value=[],
            ) as changed_files:
                validated = assert_scheduler_checkpoint_contract(
                    checkpoint,
                    role="router",
                    runtime_fps=30.0,
                    event_db_contract=expected,
                    index_json=index_json,
                    index_npz=index_npz,
                )
        compatibility = validated["runtime_code_compatibility"]
        self.assertTrue(compatibility["ok"])
        self.assertTrue(compatibility["cross_commit_reuse"])
        self.assertEqual(
            "role_scoped_model_implementation_unchanged",
            compatibility["policy"],
        )
        changed_files.assert_called_once_with(
            role="router",
            checkpoint_commit="1" * 40,
            runtime_commit="2" * 40,
        )

    def test_cross_commit_reuse_rejects_changed_role_model_implementation(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, expected)
            contract = scheduler_training_contract(
                role="duration",
                fps=30.0,
                index_metadata={
                    "canonical_fps_values": [30.0],
                    "event_db_contract": expected,
                    "skeleton_contract": skeleton_contract(),
                },
                index_json=index_json,
                index_npz=index_npz,
                dataset=dataset,
                model_rot6d_layout="pytorch3d_row",
            )
            checkpoint_code = dict(contract["code_provenance"])
            checkpoint_code.update(
                commit="1" * 40,
                worktree_clean=True,
                status_sha256="clean",
                status_entries=0,
            )
            contract["code_provenance"] = checkpoint_code
            checkpoint = {"fps": 30.0, "scheduler_contract": contract}
            runtime_code = {
                **checkpoint_code,
                "commit": "2" * 40,
            }
            with patch(
                "support.scheduler_checkpoint_contracts.repository_code_provenance",
                return_value=runtime_code,
            ), patch(
                "support.scheduler_checkpoint_contracts._changed_runtime_model_files",
                return_value=["model/duration_predictor.py"],
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "runtime model implementation changed since training"
                ):
                    assert_scheduler_checkpoint_contract(
                        checkpoint,
                        role="duration",
                        runtime_fps=30.0,
                        event_db_contract=expected,
                        index_json=index_json,
                        index_npz=index_npz,
                    )

    def test_cross_commit_reuse_rejects_dirty_runtime(self):
        expected = make_event_db_contract(["evt_a", "evt_b"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_json, index_npz, dataset = self._fixture(root, expected)
            contract = scheduler_training_contract(
                role="planner",
                fps=30.0,
                index_metadata={
                    "canonical_fps_values": [30.0],
                    "event_db_contract": expected,
                    "skeleton_contract": skeleton_contract(),
                },
                index_json=index_json,
                index_npz=index_npz,
                dataset=dataset,
            )
            checkpoint_code = dict(contract["code_provenance"])
            checkpoint_code.update(
                commit="1" * 40,
                worktree_clean=True,
                status_sha256="clean",
                status_entries=0,
            )
            contract["code_provenance"] = checkpoint_code
            checkpoint = {"fps": 30.0, "scheduler_contract": contract}
            runtime_code = {
                **checkpoint_code,
                "commit": "2" * 40,
                "worktree_clean": False,
                "status_sha256": "dirty",
                "status_entries": 1,
            }
            with patch(
                "support.scheduler_checkpoint_contracts.repository_code_provenance",
                return_value=runtime_code,
            ), patch(
                "support.scheduler_checkpoint_contracts._changed_runtime_model_files"
            ) as changed_files:
                with self.assertRaisesRegex(RuntimeError, "requires clean training"):
                    assert_scheduler_checkpoint_contract(
                        checkpoint,
                        role="planner",
                        runtime_fps=30.0,
                        event_db_contract=expected,
                        index_json=index_json,
                        index_npz=index_npz,
                    )
        changed_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
