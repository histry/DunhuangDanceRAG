import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_multirate_branches import (
    _preflight_rate_specific_checkpoints,
    main,
)


class MultirateBuilderTests(unittest.TestCase):
    def test_execute_preflight_rejects_cross_rate_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {"execute": True}
            for fps in (30, 60):
                for prefix in ("router_ckpt", "planner_ckpt", "duration_ckpt"):
                    path = root / f"{prefix}_{fps}.pt"
                    path.write_bytes(b"checkpoint")
                    values[f"{prefix}_{fps}"] = str(path)
            args = argparse.Namespace(**values)

            with patch(
                "scripts.build_multirate_branches._load_checkpoint_contract",
                return_value={"config": {"fps": 30.0}},
            ):
                with self.assertRaisesRegex(RuntimeError, "FPS mismatch"):
                    _preflight_rate_specific_checkpoints(args)

    def test_plan_is_source_disjoint_and_uses_complete_event_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            output = root / "output"
            result = main(
                [
                    "--source_dirs",
                    str(source),
                    "--output_root",
                    str(output),
                    "--base_config",
                    str(Path("configs/motion_model.json").resolve()),
                ]
            )
            self.assertEqual(result, 0)
            manifest = json.loads(
                (output / "multirate_build_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["schema"],
                "dunhuang_multirate_build_plan_v2_source_disjoint",
            )
            for branch in manifest["branches"].values():
                commands = branch["commands"]
                modules = [
                    command[command.index("-m") + 1]
                    for command in commands
                    if "-m" in command
                ]
                self.assertIn("data_pipeline.split_sources", modules)
                self.assertEqual(modules.count("events.build_pipeline"), 3)
                self.assertNotIn("events.build_database", modules)
                self.assertIn("train", branch["event_dbs"])
                self.assertIn("val", branch["event_dbs"])
                self.assertIn("test", branch["event_dbs"])
                self.assertEqual(
                    branch["training_event_db"],
                    branch["event_dbs"]["train"],
                )


if __name__ == "__main__":
    unittest.main()
