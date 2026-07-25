#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.audit_semantic_alignment import audit_alignment
from grounding.semantic_optimal_transport import SCHEMA


class SemanticAlignmentAuditTests(unittest.TestCase):
    def test_audit_explicitly_disclaims_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic_ot.npz"
            np.savez_compressed(
                path,
                schema=np.asarray(SCHEMA, dtype=object),
                phrase_ids=np.asarray([0, 0, 1, 1], dtype=np.int64),
                teacher_music_probs=np.asarray(
                    [[0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0]] * 2
                    + [[0.1, 0.1, 0.1, 0.1, 0.6, 0, 0, 0]] * 2,
                    dtype=np.float32,
                ),
                teacher_action_probs=np.asarray(
                    [[0.6, 0.2, 0.1, 0.1, 0, 0, 0, 0], [0.5, 0.2, 0.2, 0.1, 0, 0, 0, 0],
                     [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.4, 0.2, 0, 0]],
                    dtype=np.float32,
                ),
                teacher_pair_weight=np.asarray([0.7, 0.3, 0.6, 0.4], dtype=np.float32),
                teacher_js_divergence=np.asarray([0.1, 0.2, 0.15, 0.25], dtype=np.float32),
                candidate_rank=np.asarray([0, 1, 0, 1], dtype=np.int64),
                source_ids=np.asarray([0, 1, 0, 2], dtype=np.int64),
                family_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
                teacher_entropy=np.asarray([0.2, 0.2, 0.3, 0.3], dtype=np.float32),
                teacher_margin=np.asarray([0.5, 0.5, 0.4, 0.4], dtype=np.float32),
                is_ground_truth_pair=np.zeros(4, dtype=np.bool_),
            )
            report = audit_alignment(path, top_k=2)
            self.assertFalse(report["is_ground_truth_pair"])
            self.assertFalse(
                report["metric_contract"]["retrieval_accuracy_against_human_pairs"]
            )
            self.assertTrue(np.isfinite(report["weighted_mssd_aesd_js_divergence"]))


if __name__ == "__main__":
    unittest.main()
