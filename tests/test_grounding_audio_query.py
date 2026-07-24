import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grounding.audio_query import enrich_schedule_audio
from grounding.model import GroundingRuntime


class GroundingAudioQueryTests(unittest.TestCase):
    def test_schedule_is_enriched_without_losing_existing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "music.wav"
            audio.write_bytes(b"test")
            schedule = root / "schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "slots": [
                            {"slot_id": 7, "start_sec": 0.0, "end_sec": 1.0},
                            {"slot_id": 8, "start_sec": 1.0, "end_sec": 2.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "enriched.json"
            clap = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            )
            temporal = np.linspace(
                0.0, 1.0, 120 * 12, dtype=np.float32
            ).reshape(120, 12)
            with patch(
                "grounding.audio_query.phrase_deep_embedding_matrix",
                return_value=(
                    clap,
                    {"deep_success_rate": 1.0, "projection": "none"},
                ),
            ), patch(
                "grounding.audio_query.extract_audio_features",
                return_value=(temporal, {"duration_sec": 2.0}),
            ):
                report = enrich_schedule_audio(
                    audio, schedule, output, temporal_frames=16
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["slots"], 2)
            self.assertEqual(payload["slots"][0]["slot_id"], 7)
            self.assertEqual(len(payload["slots"][0]["clap_embedding"]), 3)
            self.assertEqual(len(payload["slots"][0]["temporal_features"]), 16)
            self.assertEqual(len(payload["slots"][0]["temporal_features"][0]), 12)

    def test_mixed_runtime_can_fail_closed_on_missing_audio(self):
        runtime = GroundingRuntime.__new__(GroundingRuntime)
        runtime.event_probs = np.asarray([[0.5, 0.5]], dtype=np.float32)
        runtime.model = None
        runtime.event_embedding = np.zeros((1, 1), dtype=np.float32)

        class MissingAudioRuntime:
            def score(self, slot, event_id):
                return None

        runtime.mixed_runtime = MissingAudioRuntime()
        with patch.dict(
            os.environ,
            {"V46_53_MIXED_REQUIRE_RUNTIME_AUDIO": "1"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "grounding.audio_query"):
                runtime.score({"music_semantic_probs": [0.5, 0.5]}, 0)

    def test_mixed_strict_mode_rejects_missing_checkpoint_at_startup(self):
        with patch.dict(
            os.environ,
            {
                "V46_53_GROUNDER_ARCHITECTURE": "mixed",
                "V46_53_MIXED_REQUIRE_RUNTIME_AUDIO": "1",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires an existing"
            ):
                GroundingRuntime(
                    {"aesd_music_alignment_probs": np.ones((1, 2))},
                    "definitely_missing_mixed_grounder.pt",
                )


if __name__ == "__main__":
    unittest.main()
