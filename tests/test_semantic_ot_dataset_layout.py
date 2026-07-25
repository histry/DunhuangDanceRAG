#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

import numpy as np

from grounding.semantic_ot_grounder import _phrase_layout


class SemanticOTDatasetLayoutTests(unittest.TestCase):
    def test_grouped_candidate_layout(self):
        payload = {
            "phrase_ids": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "teacher_pair_weight": np.asarray(
                [0.5, 0.3, 0.2, 0.4, 0.35, 0.25], dtype=np.float32
            ),
        }
        phrases, candidates = _phrase_layout(payload)
        self.assertEqual(candidates, 3)
        self.assertTrue(np.array_equal(phrases, np.asarray([0, 1])))

    def test_rejects_interleaved_phrases(self):
        payload = {
            "phrase_ids": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "teacher_pair_weight": np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        }
        with self.assertRaises(RuntimeError):
            _phrase_layout(payload)


if __name__ == "__main__":
    unittest.main()
