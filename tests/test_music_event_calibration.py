#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from scheduling.music_event_calibration import classify_phrase_event


def _window(
    arousal_start: float,
    arousal_end: float,
    tension_start: float,
    tension_end: float,
    calm: float = 0.68,
    frames: int = 90,
) -> np.ndarray:
    features = np.zeros((frames, 12), dtype=np.float32)

    features[:, 0] = 0.30
    features[:, 1] = 0.18
    features[:, 2] = 0.05
    features[:, 3] = 0.45
    features[:, 4] = np.linspace(
        arousal_start,
        arousal_end,
        frames,
        dtype=np.float32,
    )
    features[:, 5] = np.gradient(features[:, 4])
    features[:, 6] = np.linspace(
        tension_start,
        tension_end,
        frames,
        dtype=np.float32,
    )
    features[:, 7] = calm
    features[:, 8] = 0.20
    features[:, 9] = 0.30
    features[:, 10] = 0.10
    features[:, 11] = 0.10

    return features


class MusicEventCalibrationTest(unittest.TestCase):
    def test_positive_trend_overrides_calm_state(self) -> None:
        window = _window(
            arousal_start=0.20,
            arousal_end=0.36,
            tension_start=0.34,
            tension_end=0.38,
        )

        event, _ = classify_phrase_event(window)
        self.assertEqual(event, "build_up")

    def test_negative_trend_overrides_calm_state(self) -> None:
        window = _window(
            arousal_start=0.38,
            arousal_end=0.22,
            tension_start=0.40,
            tension_end=0.32,
        )

        event, _ = classify_phrase_event(window)
        self.assertEqual(event, "release")

    def test_dominant_positive_trend_wins_conflicting_evidence(
        self,
    ) -> None:
        # Arousal rises more strongly than tension falls.
        window = _window(
            arousal_start=0.20,
            arousal_end=0.36,
            tension_start=0.44,
            tension_end=0.34,
        )

        event, _ = classify_phrase_event(window)
        self.assertEqual(event, "build_up")

    def test_stable_low_dynamic_window_remains_calm(self) -> None:
        window = _window(
            arousal_start=0.27,
            arousal_end=0.28,
            tension_start=0.34,
            tension_end=0.35,
        )

        event, _ = classify_phrase_event(window)
        self.assertEqual(event, "calm_flow")


if __name__ == "__main__":
    unittest.main()
