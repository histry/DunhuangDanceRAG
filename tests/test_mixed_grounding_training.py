import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grounding import mixed_curvature as mixed
from grounding.paired_data import SCHEMA as PAIRED_SCHEMA


@unittest.skipIf(mixed.torch is None, "PyTorch is not installed")
class MixedGroundingTrainingSmokeTests(unittest.TestCase):
    def test_two_epoch_training_checkpoint_and_retrieval_report(self):
        rng = np.random.default_rng(41)
        rows = 16
        bodyparts = 5
        gaussian_dim = 3
        covariance = np.broadcast_to(
            np.eye(gaussian_dim, dtype=np.float32),
            (rows, bodyparts, gaussian_dim, gaussian_dim),
        ).copy()
        # Avoid a completely identical target population while retaining SPD.
        covariance *= rng.uniform(
            0.8, 1.2, size=(rows, bodyparts, 1, 1)
        ).astype(np.float32)
        payload = {
            "schema": np.asarray(PAIRED_SCHEMA, dtype=object),
            "clap": rng.normal(size=(rows, 12)).astype(np.float32),
            "temporal": rng.normal(size=(rows, 16, 12)).astype(np.float32),
            "motion_geometry": rng.normal(size=(rows, 10)).astype(np.float32),
            "bodypart_flow": rng.normal(
                size=(rows, bodyparts, 8)
            ).astype(np.float32),
            "gaussian_mean": rng.normal(
                size=(rows, bodyparts, gaussian_dim)
            ).astype(np.float32),
            "gaussian_covariance": covariance,
            "controls": rng.uniform(size=(rows, 4)).astype(np.float32),
            "quality": rng.uniform(0.4, 1.0, size=rows).astype(np.float32),
            "pair_ids": np.repeat(np.arange(rows // 2), 2).astype(np.int64),
            "family_ids": (np.arange(rows) % 3).astype(np.int64),
            "source_ids": np.repeat(np.arange(4), rows // 4).astype(np.int64),
            "event_indices": np.arange(rows, dtype=np.int64),
            "event_uids": np.asarray(
                [f"evt_smoke_{index}" for index in range(rows)], dtype=object
            ),
            "supervision": np.asarray(
                ["synthetic_test"] * rows, dtype=object
            ),
            "event_db_contract_json": np.asarray(
                json.dumps({"schema": "smoke", "num_events": rows}),
                dtype=object,
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "paired.npz"
            checkpoint = root / "mixed.pt"
            np.savez_compressed(dataset, **payload)
            environment = {
                "V46_53_MIXED_GROUNDER_CUDA": "0",
                "V46_53_MIXED_HIDDEN": "32",
                "V46_53_MIXED_LORENTZ_DIM": "4",
                "V46_53_MIXED_SPHERE_DIM": "8",
            }
            with patch.dict(os.environ, environment, clear=False):
                report = mixed.train_mixed_grounder(
                    dataset,
                    checkpoint,
                    epochs=2,
                    batch_size=4,
                    seed=9,
                    validation_ratio=0.25,
                    patience=2,
                )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(report["ok"])
            self.assertGreaterEqual(report["best_epoch"], 1)
            self.assertIn("audio_to_motion", report["validation_retrieval"])
            self.assertIn(
                "R@1", report["validation_retrieval"]["audio_to_motion"]
            )
            loaded = mixed._load_torch_checkpoint(checkpoint)
            self.assertEqual(loaded["schema"], mixed.SCHEMA)
            self.assertEqual(loaded["config"]["hidden_dim"], 32)


if __name__ == "__main__":
    unittest.main()
