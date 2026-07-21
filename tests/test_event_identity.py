import unittest

import numpy as np

from support.event_identity import (
    event_uid_from_item,
    event_uids_from_generation_db,
    make_event_db_contract,
    stable_event_uid,
)


class EventIdentityTests(unittest.TestCase):
    def test_uid_is_root_independent(self):
        left = stable_event_uid(
            source_uid="female_lotus_1",
            source_file="/old/root/female_lotus_1.bvh",
            start=12,
            end=72,
            frames=60,
        )
        right = stable_event_uid(
            source_uid="female_lotus_1",
            source_file=r"D:\new\root\female_lotus_1.bvh",
            start=12,
            end=72,
            frames=60,
        )
        self.assertEqual(left, right)

    def test_generation_contract_is_order_sensitive(self):
        db = {
            "paths": np.asarray(["a.npy", "b.npy"], dtype=object),
            "source_uids": np.asarray(["a", "b"], dtype=object),
            "source_files": np.asarray(["a.bvh", "b.bvh"], dtype=object),
            "starts": np.asarray([0, 10]),
            "ends": np.asarray([10, 20]),
            "frames": np.asarray([10, 10]),
            "canonical_fps": np.asarray([30.0, 30.0]),
        }
        uids = event_uids_from_generation_db(db)
        normal = make_event_db_contract(uids)
        reversed_contract = make_event_db_contract(list(reversed(uids)))
        self.assertNotEqual(
            normal["ordered_event_uid_sha256"],
            reversed_contract["ordered_event_uid_sha256"],
        )

    def test_uid_is_independent_of_target_fps(self):
        at_30 = stable_event_uid(
            source_uid="aist_gBR_sBM_cAll_d04_mBR0_ch01",
            source_file="aist_gBR_sBM_cAll_d04_mBR0_ch01.pkl",
            start=30,
            end=90,
            frames=60,
            source_fps=30.0,
        )
        at_60 = stable_event_uid(
            source_uid="aist_gBR_sBM_cAll_d04_mBR0_ch01",
            source_file="aist_gBR_sBM_cAll_d04_mBR0_ch01.pkl",
            start=60,
            end=180,
            frames=120,
            source_fps=60.0,
        )
        self.assertEqual(at_30, at_60)

    def test_explicit_source_uid_is_independent_of_source_filename(self):
        left = stable_event_uid(
            source_uid="source-0042",
            source_file="original_capture.bvh",
            start=30,
            end=90,
            source_fps=30.0,
        )
        right = stable_event_uid(
            source_uid="source-0042",
            source_file="renamed_for_public_release.bvh",
            start=60,
            end=180,
            source_fps=60.0,
        )
        self.assertEqual(left, right)

    def test_generation_db_without_time_or_fps_fails_closed(self):
        db = {
            "paths": np.asarray(["a.npy"], dtype=object),
            "source_uids": np.asarray(["a"], dtype=object),
            "starts": np.asarray([0]),
            "ends": np.asarray([30]),
            "frames": np.asarray([30]),
        }
        with self.assertRaisesRegex(RuntimeError, "physical time fields"):
            event_uids_from_generation_db(db)

    def test_item_with_physical_time_does_not_need_frame_rate(self):
        uid = event_uid_from_item(
            {
                "source_uid": "a",
                "source_start": 30,
                "source_end": 90,
                "source_start_seconds": 1.0,
                "source_end_seconds": 3.0,
            }
        )
        expected = stable_event_uid(
            source_uid="a",
            start=60,
            end=180,
            source_fps=60.0,
        )
        self.assertEqual(uid, expected)

    def test_item_without_time_or_fps_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "explicit FPS contract"):
            event_uid_from_item(
                {
                    "source_uid": "a",
                    "source_start": 0,
                    "source_end": 30,
                },
                position=7,
            )


if __name__ == "__main__":
    unittest.main()
