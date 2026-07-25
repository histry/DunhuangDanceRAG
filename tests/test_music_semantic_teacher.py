#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

import numpy as np

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS
from training.music_semantic_teacher import weak_semantic_distribution


class MusicSemanticTeacherTests(unittest.TestCase):
    @staticmethod
    def _window(overrides=None):
        row = np.zeros((48, 12), dtype=np.float32)
        defaults = {
            0: 0.5,   # energy
            1: 0.2,   # onset
            2: 0.1,   # beat
            4: 0.5,   # arousal
            5: 0.0,   # delta
            6: 0.4,   # tension
            7: 0.5,   # calm
            8: 0.3,   # novelty
            9: 0.4,   # brightness
            10: 0.2,  # section
            11: 0.2,  # accent
        }
        defaults.update(dict(overrides or {}))
        for index, value in defaults.items():
            row[:, int(index)] = float(value)
        return row

    def test_distribution_is_strictly_positive_and_normalized(self):
        probability = weak_semantic_distribution(self._window(), "neutral_flow")
        self.assertEqual(probability.shape, (len(MUSIC_SEMANTIC_LABELS),))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue((probability > 0.0).all())
        self.assertAlmostEqual(float(probability.sum()), 1.0, places=6)

    def test_calm_window_prefers_calm_or_pose(self):
        probability = weak_semantic_distribution(
            self._window({0: 0.15, 1: 0.05, 2: 0.02, 4: 0.18, 6: 0.12, 7: 0.90, 8: 0.08, 11: 0.05}),
            "calm_flow",
        )
        top = MUSIC_SEMANTIC_LABELS[int(np.argmax(probability))]
        self.assertIn(top, {"calm_meditative", "pose_hold"})

    def test_percussive_window_prefers_percussive_or_footwork(self):
        probability = weak_semantic_distribution(
            self._window({0: 0.82, 1: 0.90, 2: 0.86, 4: 0.78, 6: 0.72, 7: 0.08, 8: 0.42, 11: 0.92}),
            "accent",
        )
        top = MUSIC_SEMANTIC_LABELS[int(np.argmax(probability))]
        self.assertIn(top, {"percussive_accent", "footwork_flow", "turning_climax"})


if __name__ == "__main__":
    unittest.main()
