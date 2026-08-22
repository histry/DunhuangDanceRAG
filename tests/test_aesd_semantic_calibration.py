#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from events.semantic_descriptor import (
    class_prior_adjustment,
    semantic_distribution_diagnostics,
)


class AESDSemanticCalibrationTests(unittest.TestCase):
    def test_distribution_diagnostics_are_normalized(self):
        probabilities = np.asarray(
            [0.03, 0.07, 0.12, 0.08, 0.10, 0.31, 0.19, 0.10],
            dtype=np.float32,
        )
        diagnostics = semantic_distribution_diagnostics(probabilities)
        self.assertTrue(0.0 <= diagnostics["normalized_entropy"] <= 1.0)
        self.assertEqual(diagnostics["top_label"], "turning_climax")

    def test_prior_adjustment_is_normalized(self):
        probabilities = np.asarray([0.01, 0.05, 0.05, 0.05, 0.08, 0.70, 0.05, 0.01])
        prior = np.asarray([0.02, 0.20, 0.10, 0.05, 0.10, 0.45, 0.06, 0.02])
        adjusted = class_prior_adjustment(probabilities, prior, alpha=0.35)
        self.assertTrue(np.isfinite(adjusted).all())
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=6)
        self.assertLess(adjusted[5], probabilities[5] / probabilities.sum())


if __name__ == "__main__":
    unittest.main()
