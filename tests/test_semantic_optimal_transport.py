#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from grounding.semantic_optimal_transport import (
    jensen_shannon_divergence,
    sparse_sinkhorn,
)


class SemanticOptimalTransportTests(unittest.TestCase):
    def test_js_divergence_identity_and_separation(self):
        probabilities = np.eye(3, dtype=np.float32)
        distance = jensen_shannon_divergence(probabilities, probabilities)
        self.assertEqual(distance.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(distance), 0.0, atol=1.0e-7))
        self.assertTrue(np.all(distance[np.triu_indices(3, 1)] > 0.5))

    def test_sparse_sinkhorn_rows_normalize(self):
        candidates = np.asarray([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        costs = np.asarray(
            [[0.05, 0.80], [0.10, 0.60], [0.20, 0.15]], dtype=np.float32
        )
        marginal = np.asarray([0.35, 0.30, 0.35], dtype=np.float32)
        plan, report = sparse_sinkhorn(
            candidates,
            costs,
            marginal,
            epsilon=0.15,
            iterations=300,
        )
        self.assertEqual(plan.shape, costs.shape)
        self.assertTrue(np.isfinite(plan).all())
        self.assertTrue(np.allclose(plan.sum(axis=1), 1.0, atol=1.0e-5))
        self.assertGreater(report["active_events"], 1)


if __name__ == "__main__":
    unittest.main()
