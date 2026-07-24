import unittest

import numpy as np

from grounding.paired_data import (
    DEFAULT_TEMPORAL_FRAMES,
    TEMPORAL_DIM,
    _resample_sequence,
    validate_paired_payload,
)


def valid_payload(rows: int = 4) -> dict:
    bodyparts = 5
    gaussian_dim = 3
    covariance = np.broadcast_to(
        np.eye(gaussian_dim, dtype=np.float32),
        (rows, bodyparts, gaussian_dim, gaussian_dim),
    ).copy()
    return {
        "clap": np.ones((rows, 16), dtype=np.float32),
        "temporal": np.ones(
            (rows, DEFAULT_TEMPORAL_FRAMES, TEMPORAL_DIM), dtype=np.float32
        ),
        "motion_geometry": np.ones((rows, 12), dtype=np.float32),
        "bodypart_flow": np.ones((rows, bodyparts, 8), dtype=np.float32),
        "gaussian_mean": np.zeros(
            (rows, bodyparts, gaussian_dim), dtype=np.float32
        ),
        "gaussian_covariance": covariance,
        "controls": np.ones((rows, 4), dtype=np.float32),
        "quality": np.ones(rows, dtype=np.float32),
        "pair_ids": np.arange(rows, dtype=np.int64),
        "audio_group_ids": np.arange(rows, dtype=np.int64),
        "family_ids": np.arange(rows, dtype=np.int64) % 2,
        "source_ids": np.arange(rows, dtype=np.int64) % 2,
        "event_indices": np.arange(rows, dtype=np.int64),
        "event_uids": np.asarray([f"evt_{i}" for i in range(rows)], dtype=object),
    }


class PairedGroundingDataTests(unittest.TestCase):
    def test_valid_payload_contract(self):
        dimensions = validate_paired_payload(valid_payload())
        self.assertEqual(dimensions["rows"], 4)
        self.assertEqual(dimensions["clap_dim"], 16)
        self.assertEqual(dimensions["bodypart_count"], 5)
        self.assertEqual(dimensions["control_dim"], 4)

    def test_contract_rejects_non_spd_covariance(self):
        payload = valid_payload()
        payload["gaussian_covariance"][0, 0, 0, 0] = -1.0
        with self.assertRaisesRegex(RuntimeError, "strictly SPD"):
            validate_paired_payload(payload)

    def test_contract_rejects_row_misalignment(self):
        payload = valid_payload()
        payload["quality"] = payload["quality"][:-1]
        with self.assertRaisesRegex(RuntimeError, "row-aligned"):
            validate_paired_payload(payload)

    def test_temporal_resampling_preserves_endpoints(self):
        sequence = np.stack(
            [
                np.linspace(0.0, 1.0, 5, dtype=np.float32)
                for _ in range(TEMPORAL_DIM)
            ],
            axis=-1,
        )
        result = _resample_sequence(sequence, 17)
        self.assertEqual(result.shape, (17, TEMPORAL_DIM))
        np.testing.assert_allclose(result[0], sequence[0])
        np.testing.assert_allclose(result[-1], sequence[-1])


if __name__ == "__main__":
    unittest.main()
