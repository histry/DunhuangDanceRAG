import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
