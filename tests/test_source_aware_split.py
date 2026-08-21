#!/usr/bin/env python3
import unittest

from data_pipeline.split_sources import recording_group_records


class SourceAwareSplitTest(unittest.TestCase):
    def test_segments_keep_one_source_and_recording_identity(self):
        records = [
            {
                "source_uid": "female_36pose_1",
                "recording_uid": "female_36pose_sequence",
                "performer_group": "female",
                "dance_key": "thirty_six_postures",
            },
            {
                "source_uid": "female_36pose_1",
                "recording_uid": "female_36pose_sequence",
                "performer_group": "female",
                "dance_key": "thirty_six_postures",
            },
            {
                "source_uid": "female_36pose_2",
                "recording_uid": "female_36pose_sequence",
                "performer_group": "female",
                "dance_key": "thirty_six_postures",
            },
        ]
        units = recording_group_records(records)
        self.assertEqual(len(units), 1)
        self.assertEqual(
            units[0]["source_uids"],
            ["female_36pose_1", "female_36pose_2"],
        )
        self.assertEqual(units[0]["num_performer_tracks"], 2)
        self.assertEqual(units[0]["num_segments"], 3)


if __name__ == "__main__":
    unittest.main()
