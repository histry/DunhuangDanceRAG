import unittest

import numpy as np

from scheduling.music_event_calibration import build_phrase_query


class MusicQueryPhysicalTimeTests(unittest.TestCase):
    def test_duration_component_is_rate_invariant(self):
        features_30 = np.full((60, 12), 0.4, dtype=np.float32)
        features_60 = np.full((120, 12), 0.4, dtype=np.float32)
        query_30, event_30 = build_phrase_query(
            features_30, 0, len(features_30), fps=30.0
        )
        query_60, event_60 = build_phrase_query(
            features_60, 0, len(features_60), fps=60.0
        )
        np.testing.assert_allclose(query_30, query_60, atol=1.0e-7)
        self.assertEqual(event_30, event_60)

    def test_nonpositive_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            build_phrase_query(np.zeros((2, 12), dtype=np.float32), 0, 2, fps=0.0)

    def test_nonfinite_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            build_phrase_query(
                np.zeros((2, 12), dtype=np.float32),
                0,
                2,
                fps=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
