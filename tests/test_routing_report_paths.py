import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class RoutingReportPathTests(unittest.TestCase):
    @staticmethod
    def _module():
        try:
            import torch  # noqa: F401
            import routing.heading_closed_loop_impl2 as implementation
        except Exception as exc:
            raise unittest.SkipTest(f"routing runtime dependencies unavailable: {exc}")
        return implementation

    def test_plain_report_json_resolves_sibling_motion(self):
        implementation = self._module()
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            report = root / "report.json"
            motion = root / "motion.npy"
            report.write_text(json.dumps({}), encoding="utf-8")
            np.save(motion, np.zeros((2, 151), dtype=np.float32))
            self.assertEqual(
                motion.resolve(),
                implementation._resolve_motion_path(report, {}),
            )

    def test_report_suffix_resolves_matching_npy(self):
        implementation = self._module()
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            report = root / "dance.report.json"
            motion = root / "dance.npy"
            report.write_text(json.dumps({}), encoding="utf-8")
            np.save(motion, np.zeros((2, 151), dtype=np.float32))
            self.assertEqual(
                motion.resolve(),
                implementation._resolve_motion_path(report, {}),
            )

    def test_explicit_motion_path_wins_over_json_path(self):
        implementation = self._module()
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            report = root / "report.json"
            explicit = root / "chosen.npy"
            fallback = root / "motion.npy"
            report.write_text(json.dumps({"motion": str(fallback)}), encoding="utf-8")
            np.save(explicit, np.zeros((2, 151), dtype=np.float32))
            np.save(fallback, np.ones((2, 151), dtype=np.float32))
            self.assertEqual(
                explicit.resolve(),
                implementation._resolve_motion_path(
                    report,
                    {"motion": str(fallback)},
                    explicit_motion_path=explicit,
                ),
            )

    def test_runtime_fps_uses_matching_config_and_scheduler_environment(self):
        implementation = self._module()
        with tempfile.TemporaryDirectory() as root_dir:
            config = Path(root_dir) / "motion.json"
            config.write_text(json.dumps({"fps": 60.0}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"V46_51_FPS": "60", "V46_FPS": "60"},
                clear=False,
            ):
                self.assertEqual(
                    implementation._runtime_fps(["--config", str(config)]),
                    60.0,
                )

    def test_runtime_fps_rejects_config_environment_mismatch(self):
        implementation = self._module()
        with tempfile.TemporaryDirectory() as root_dir:
            config = Path(root_dir) / "motion.json"
            config.write_text(json.dumps({"fps": 60.0}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"V46_51_FPS": "30"},
                clear=False,
            ):
                os.environ.pop("V46_FPS", None)
                with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                    implementation._runtime_fps(["--config", str(config)])


if __name__ == "__main__":
    unittest.main()
