import importlib
import os
import unittest
from unittest.mock import patch


class V241AblationContractTests(unittest.TestCase):
    def test_presets_have_explicit_zero_control_and_small_research_weights(self):
        mod = importlib.import_module("run_v24_ablation")
        self.assertEqual(mod.PRESETS["Z"], (0.0, 0.0))
        self.assertEqual(mod.PRESETS["A"], (0.005, 0.0005))
        self.assertEqual(mod.PRESETS["B"], (0.010, 0.0010))
        self.assertEqual(mod.PRESETS["C"], (0.020, 0.0020))

    def test_formal_ablation_requires_authoritative_experiment_environment(self):
        mod = importlib.import_module("run_v24_ablation")
        clean = {
            key: value
            for key, value in os.environ.items()
            if key not in {"EXPERIMENT_CONFIG_LOADED", "EXPERIMENT_ACTIVE_PROFILE"}
        }
        with patch.dict(os.environ, clean, clear=True):
            with self.assertRaisesRegex(RuntimeError, "source configs/experiment.env"):
                mod._require_research_environment()

    def test_research_environment_snapshot_is_accepted(self):
        mod = importlib.import_module("run_v24_ablation")
        with patch.dict(
            os.environ,
            {
                "EXPERIMENT_CONFIG_LOADED": "1",
                "EXPERIMENT_ACTIVE_PROFILE": "research",
                "RETARGET_CLEAN_ITERATIONS": "280",
                "RETARGET_CLEAN_LEARNING_RATE": "0.018",
            },
            clear=False,
        ):
            snapshot = mod._require_research_environment()
        self.assertEqual(snapshot["RETARGET_CLEAN_ITERATIONS"], "280")
        self.assertEqual(snapshot["RETARGET_CLEAN_LEARNING_RATE"], "0.018")


if __name__ == "__main__":
    unittest.main()
