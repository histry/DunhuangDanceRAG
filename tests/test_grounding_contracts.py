import json
import unittest

import numpy as np

from grounding.model import _geometry_for_checkpoint


def database(fps: float = 60.0) -> dict:
    return {
        "v46_53_geometry_desc": np.asarray(
            [[12.0, 14.0], [8.0, 6.0]], dtype=np.float32
        ),
        # Deliberately incompatible split-local values.  They must never be
        # consumed by validation/test embedding.
        "v46_53_geometry_desc_z": np.asarray(
            [[99.0, 99.0], [-99.0, -99.0]], dtype=np.float32
        ),
        "v46_53_geometry_schema_version": np.asarray(
            "geometry_v2", dtype=object
        ),
        "v46_53_geometry_fps": np.asarray(fps, dtype=np.float32),
        "skeleton_contract_json": np.asarray(
            json.dumps({"schema": "smpl24"}, sort_keys=True), dtype=object
        ),
    }


def checkpoint(fps: float = 60.0) -> dict:
    skeleton = json.dumps({"schema": "smpl24"}, sort_keys=True)
    return {
        "geometry_contract": {
            "geometry_schema": "geometry_v2",
            "fps": fps,
            "skeleton_contract_json": skeleton,
            "geometry_dim": 2,
        },
        "geometry_train_mean": np.asarray([[10.0, 10.0]], dtype=np.float32),
        "geometry_train_std": np.asarray([[2.0, 4.0]], dtype=np.float32),
    }


class GroundingContractTests(unittest.TestCase):
    def test_eval_split_uses_training_statistics(self):
        transformed = _geometry_for_checkpoint(database(), checkpoint())
        np.testing.assert_allclose(
            transformed,
            np.asarray([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32),
        )

    def test_cross_rate_grounder_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fps"):
            _geometry_for_checkpoint(database(fps=60.0), checkpoint(fps=30.0))

    def test_legacy_grounder_without_training_contract_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no train-split geometry contract"):
            _geometry_for_checkpoint(database(), {})


if __name__ == "__main__":
    unittest.main()
