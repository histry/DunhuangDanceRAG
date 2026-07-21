import os
import unittest
from unittest.mock import patch

from support.checkpoint_contracts import assert_checkpoint_fps


class CheckpointFpsContractTests(unittest.TestCase):
    def test_matching_rate_is_accepted(self):
        value = assert_checkpoint_fps(
            {"config": {"fps": 60.0}},
            role="test",
            runtime_fps=60.0,
        )
        self.assertEqual(value, 60.0)

    def test_mismatched_rate_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "FPS mismatch"):
            assert_checkpoint_fps(
                {"config": {"fps": 30.0}},
                role="test",
                runtime_fps=60.0,
            )

    def test_missing_rate_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "has no FPS contract"):
                assert_checkpoint_fps(
                    {"config": {}},
                    role="test",
                    runtime_fps=30.0,
                )

    def test_legacy_override_is_restricted_to_30fps(self):
        with patch.dict(
            os.environ,
            {"DUNHUANG_ALLOW_LEGACY_30FPS_CHECKPOINTS": "1"},
            clear=True,
        ):
            self.assertEqual(
                assert_checkpoint_fps(
                    {"config": {}},
                    role="test",
                    runtime_fps=30.0,
                ),
                30.0,
            )
            with self.assertRaisesRegex(RuntimeError, "has no FPS contract"):
                assert_checkpoint_fps(
                    {"config": {}},
                    role="test",
                    runtime_fps=60.0,
                )

    def test_invalid_declared_rate_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid FPS metadata"):
            assert_checkpoint_fps(
                {"config": {"fps": float("nan")}},
                role="test",
                runtime_fps=30.0,
            )


if __name__ == "__main__":
    unittest.main()
