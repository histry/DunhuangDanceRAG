import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from events import build_pipeline


class MixedBuildPipelineTests(unittest.TestCase):
    def test_train_split_is_embedded_immediately_after_mixed_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "train"
            out_dir.mkdir(parents=True)
            np.savez_compressed(
                out_dir / "events.npz",
                paths=np.asarray(["event.npy"], dtype=object),
            )
            paired = root / "paired.npz"
            paired.write_bytes(b"paired")
            checkpoint = root / "mixed.pt"
            environment = {
                "V46_53_GROUNDER_ENABLE": "1",
                "V46_53_TRAIN_GROUNDER_ON_BUILD": "1",
                "V46_53_GROUNDER_ARCHITECTURE": "mixed",
                "V46_53_GROUNDER_PAIRED_DATASET": str(paired),
                "V46_53_GROUNDER_CKPT": str(checkpoint),
            }
            with patch.dict(os.environ, environment, clear=False), patch(
                "events.build_pipeline.v52.main", return_value=0
            ), patch(
                "events.build_pipeline.augment_database",
                return_value={"num_events": 1, "geometry_dim": 112},
            ), patch(
                "grounding.mixed_curvature.train_mixed_grounder",
                return_value={"schema": "train", "ok": True},
            ) as train, patch(
                "grounding.mixed_curvature.embed_database_mixed",
                return_value={"schema": "embed", "ok": True},
            ) as embed:
                result = build_pipeline.main(["--out_db", str(out_dir)])

            self.assertEqual(result, 0)
            train.assert_called_once()
            embed.assert_called_once_with(out_dir / "events.npz", checkpoint)
            report = json.loads(
                (out_dir / "events.v46_53.build.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(report["grounding"]["train_db_embedded"])
            self.assertEqual(
                report["grounding"]["embedding"]["schema"], "embed"
            )


if __name__ == "__main__":
    unittest.main()
