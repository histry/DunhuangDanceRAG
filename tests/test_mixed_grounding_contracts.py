import unittest

import numpy as np

from grounding.mixed_curvature import (
    apply_normalization,
    fit_train_normalization,
    retrieval_metrics,
    source_disjoint_split,
)
from tests.test_paired_grounding_data import valid_payload


class MixedGroundingContractTests(unittest.TestCase):
    def test_source_split_has_no_group_leakage(self):
        source_ids = np.repeat(np.arange(6), 4)
        train, validation = source_disjoint_split(source_ids, 0.34, seed=7)
        train_sources = set(source_ids[train].tolist())
        validation_sources = set(source_ids[validation].tolist())
        self.assertTrue(train_sources)
        self.assertTrue(validation_sources)
        self.assertFalse(train_sources & validation_sources)

    def test_source_pair_audio_components_do_not_cross_split(self):
        source_ids = np.asarray(
            [0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64
        )
        pair_ids = np.asarray(
            [10, 11, 10, 12, 20, 21, 20, 22], dtype=np.int64
        )
        audio_ids = np.asarray(
            [100, 101, 100, 102, 200, 201, 200, 202],
            dtype=np.int64,
        )
        train, validation = source_disjoint_split(
            source_ids,
            0.5,
            seed=7,
            pair_ids=pair_ids,
            audio_group_ids=audio_ids,
        )
        for values in (source_ids, pair_ids, audio_ids):
            self.assertFalse(
                set(values[train].tolist())
                & set(values[validation].tolist())
            )

    def test_single_identity_component_is_rejected(self):
        source_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
        pair_ids = np.asarray([5, 6, 5, 7], dtype=np.int64)
        audio_ids = np.asarray([50, 60, 50, 70], dtype=np.int64)
        with self.assertRaisesRegex(RuntimeError, "one connected component"):
            source_disjoint_split(
                source_ids,
                0.5,
                seed=7,
                pair_ids=pair_ids,
                audio_group_ids=audio_ids,
            )

    def test_train_only_normalization_centres_training_rows(self):
        payload = valid_payload(rows=8)
        payload["motion_geometry"] = np.arange(
            8 * 12, dtype=np.float32
        ).reshape(8, 12)
        payload["bodypart_flow"] = np.arange(
            8 * 5 * 8, dtype=np.float32
        ).reshape(8, 5, 8)
        payload["temporal"] = np.arange(
            8 * 64 * 12, dtype=np.float32
        ).reshape(8, 64, 12)
        payload["gaussian_mean"] = np.arange(
            8 * 5 * 3, dtype=np.float32
        ).reshape(8, 5, 3)
        training = np.asarray([0, 1, 2, 3], dtype=np.int64)
        normalization = fit_train_normalization(payload, training)
        transformed = apply_normalization(payload, normalization)
        np.testing.assert_allclose(
            transformed["motion_geometry"][training].mean(axis=0),
            0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            transformed["temporal"][training].mean(axis=(0, 1)),
            0.0,
            atol=1.0e-5,
        )
        # Held-out rows are deliberately not re-centred with their own stats.
        self.assertGreater(
            float(np.abs(transformed["motion_geometry"][4:]).mean()), 0.1
        )

    def test_single_source_split_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "two distinct"):
            source_disjoint_split(np.zeros(8, dtype=np.int64), 0.2, seed=1)

    def test_multi_positive_retrieval_metrics(self):
        pair_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray(
            [
                [0.9, 0.8, 0.1, 0.0],
                [0.8, 0.9, 0.0, 0.1],
                [0.1, 0.0, 0.9, 0.8],
                [0.0, 0.1, 0.8, 0.9],
            ],
            dtype=np.float32,
        )
        metrics = retrieval_metrics(scores, pair_ids, pair_ids)
        self.assertEqual(metrics["R@1"], 1.0)
        self.assertEqual(metrics["mAP"], 1.0)
        self.assertEqual(metrics["queries"], 4)


if __name__ == "__main__":
    unittest.main()
