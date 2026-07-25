#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from events.semantic_descriptor import (
    MUSIC_SEMANTIC_LABELS,
    class_prior_adjustment,
    event_probs_from_fields,
    semantic_distribution_diagnostics,
)


class AESDSemanticCalibrationTests(unittest.TestCase):
    def test_aerial_family_has_nonzero_aerial_probability(self):
        probabilities = event_probs_from_fields(
            dance_key="unknown",
            event_family="aerial_curve",
            music_alignment_label="unknown",
            energy_label="moderate",
            rhythm_label="lyrical",
            locomotion_label="floating_leaning",
            support_label="low_contact_flight_like",
            quality=0.9,
            semantic_confidence=0.9,
        )
        aerial = MUSIC_SEMANTIC_LABELS.index("aerial_curve")
        self.assertGreater(float(probabilities[aerial]), 0.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_grouped_evidence_does_not_collapse_distribution(self):
        probabilities = event_probs_from_fields(
            dance_key="sogdian_whirl",
            event_family="turning_flow",
            music_alignment_label="turning_climax",
            energy_label="high",
            rhythm_label="accented",
            locomotion_label="turning_travel",
            support_label="alternating_or_pivot_support",
            quality=0.8,
            semantic_confidence=0.8,
        )
        diagnostics = semantic_distribution_diagnostics(probabilities)
        self.assertTrue(0.0 <= diagnostics["normalized_entropy"] <= 1.0)
        self.assertGreater(np.count_nonzero(probabilities > 0.01), 1)

    def test_prior_adjustment_is_normalized(self):
        probabilities = np.asarray([0.01, 0.05, 0.05, 0.05, 0.08, 0.70, 0.05, 0.01])
        prior = np.asarray([0.02, 0.20, 0.10, 0.05, 0.10, 0.45, 0.06, 0.02])
        adjusted = class_prior_adjustment(probabilities, prior, alpha=0.35)
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=6)
        self.assertLess(adjusted[5], probabilities[5] / probabilities.sum())


if __name__ == "__main__":
    unittest.main()
