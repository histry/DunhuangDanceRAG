#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from events.semantic_descriptor import MUSIC_SEMANTIC_LABELS
from routing.diversity_constrained_retrieval import (
    select_diverse_candidates,
    selection_audit,
)


class DiversityConstrainedRetrievalTests(unittest.TestCase):
    def test_source_and_family_caps(self):
        count = 30
        scores = np.linspace(1.0, 0.0, count)
        sources = np.asarray([f"source_{index // 5}" for index in range(count)], dtype=object)
        families = np.asarray([f"family_{index // 4}" for index in range(count)], dtype=object)
        semantics = np.asarray(
            ["turning_climax"] * 15 + ["lyrical_flow"] * 15, dtype=object
        )
        music = np.zeros(len(MUSIC_SEMANTIC_LABELS), dtype=np.float32)
        music[MUSIC_SEMANTIC_LABELS.index("turning_climax")] = 0.55
        music[MUSIC_SEMANTIC_LABELS.index("lyrical_flow")] = 0.45
        selected = select_diverse_candidates(
            scores,
            sources,
            families,
            semantics,
            music,
            top_k=12,
            source_cap=3,
            family_cap=4,
        )
        audit = selection_audit(selected, sources, families, semantics)
        self.assertEqual(len(selected), 12)
        self.assertGreaterEqual(audit["source_coverage"], 4)
        self.assertLessEqual(max(audit["source_histogram"].values()), 3)


if __name__ == "__main__":
    unittest.main()
