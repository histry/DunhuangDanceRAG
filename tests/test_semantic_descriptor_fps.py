import json
import tempfile
import unittest
from pathlib import Path

from events.semantic_descriptor import (
    build_descriptor_object,
    parse_descriptor_file,
)


def final_descriptor(fps=None):
    value = {
        "usage": "generate_schedule",
        "is_final_schedule": True,
        "slot_source": "v21_router_v26_planner",
        "slots": [
            {
                "slot_id": 0,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "target_frames": 30,
                "music_semantic_top_label": "lyrical_flow",
            }
        ],
    }
    if fps is not None:
        value["fps"] = fps
    return value


class SemanticDescriptorFpsTests(unittest.TestCase):
    def _write(self, root: Path, value) -> Path:
        path = root / "schedule.mssd.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_final_descriptor_requires_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), final_descriptor())
            with self.assertRaisesRegex(RuntimeError, "has no FPS contract"):
                parse_descriptor_file(path, require_final_schedule=True, fps=30.0)

    def test_final_descriptor_rejects_runtime_rate_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), final_descriptor(30.0))
            with self.assertRaisesRegex(RuntimeError, "FPS contract mismatch"):
                parse_descriptor_file(path, require_final_schedule=True, fps=60.0)

    def test_final_descriptor_accepts_matching_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), final_descriptor(30.0))
            slots, features, meta = parse_descriptor_file(
                path,
                require_final_schedule=True,
                fps=30.0,
            )
            self.assertEqual(meta["fps"], 30.0)
            self.assertEqual(slots[0]["target_frames"], 30)
            self.assertEqual(features.shape, (1, 32))

    def test_final_descriptor_builder_requires_fps(self):
        with self.assertRaisesRegex(RuntimeError, "must declare fps"):
            build_descriptor_object(
                "song.wav",
                [],
                {
                    "usage": "generate_schedule",
                    "is_final_schedule": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
