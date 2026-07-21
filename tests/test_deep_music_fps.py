import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scheduling import deep_music_features as features


def phrase_at_rate(fps: float, *, start_seconds: float = 1.0) -> SimpleNamespace:
    start = int(round(start_seconds * fps))
    length = int(round(2.0 * fps))
    return SimpleNamespace(
        start=start,
        end=start + length,
        length=length,
        music_event="build_up",
        energy=0.7,
        onset=0.4,
        beat_density=0.6,
        tension=0.5,
        calmness=0.1,
        boundary_accent_strength=0.3,
    )


class DeepMusicFpsTests(unittest.TestCase):
    def test_rule_semantics_are_invariant_to_equivalent_frame_rates(self):
        at_30 = features.phrase_rule_semantic(phrase_at_rate(30.0), fps=30.0)
        at_60 = features.phrase_rule_semantic(phrase_at_rate(60.0), fps=60.0)
        np.testing.assert_allclose(at_30, at_60, rtol=0.0, atol=1e-7)

    def test_cache_is_rate_and_phrase_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            audio = cache / "song.wav"
            audio.write_bytes(b"same scientific test audio")
            matrix_30, meta_30 = features.phrase_semantic_matrix(
                audio,
                [phrase_at_rate(30.0)],
                enabled=False,
                cache_dir=cache,
                fps=30.0,
            )
            matrix_60, meta_60 = features.phrase_semantic_matrix(
                audio,
                [phrase_at_rate(60.0)],
                enabled=False,
                cache_dir=cache,
                fps=60.0,
            )
            np.testing.assert_allclose(matrix_30, matrix_60, rtol=0.0, atol=1e-7)
            self.assertEqual(meta_30["phrase_fps"], 30.0)
            self.assertEqual(meta_60["phrase_fps"], 60.0)
            caches = list(cache.glob("*_v27_semantic_*.npz"))
            self.assertEqual(len(caches), 2)
            for path in caches:
                with np.load(path, allow_pickle=True) as data:
                    meta = json.loads(str(data["meta"].item()))
                    self.assertIn(meta["phrase_fps"], (30.0, 60.0))

    def test_deep_projection_preserves_all_twelve_dimensions(self):
        phrase = phrase_at_rate(30.0)
        embedding = np.arange(1, 9, dtype=np.float32)
        with patch.object(
            features,
            "_try_clap_phrase_embedding",
            return_value=(embedding, "laion_clap"),
        ):
            semantic, meta = features.phrase_semantic_matrix(
                "not_read.wav",
                [phrase],
                enabled=True,
                fps=30.0,
            )
        projection = features._projection_matrix(embedding.size, 12)
        deep = features._normalize(embedding.reshape(1, -1) @ projection)
        rule = features.phrase_rule_semantic(phrase, fps=30.0)
        expected = features._normalize(0.5 * rule + 0.5 * deep)
        np.testing.assert_allclose(semantic[0], expected, rtol=0.0, atol=1e-7)
        self.assertGreater(float(np.std(semantic[0])), 0.01)
        self.assertEqual(meta["deep_success_count"], 1)

    def test_invalid_rate_fails(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            features.phrase_rule_semantic(phrase_at_rate(30.0), fps=0.0)


if __name__ == "__main__":
    unittest.main()
