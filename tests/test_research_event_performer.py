import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from tools.research_augment_event_performer import augment_events_npz
except ImportError:
    from events.augment_performer_metadata import augment_events_npz


class ResearchEventPerformerMetadataTest(unittest.TestCase):
    def test_metadata_is_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.npz"
            np.savez_compressed(
                path,
                paths=np.asarray([
                    "/x/female_pose_event.npy",
                    "/x/male_drum_event.npy",
                ], dtype=object),
                source_uids=np.asarray([
                    "female_pose",
                    "male_drum",
                ], dtype=object),
            )
            report = augment_events_npz(str(path), require_known=True)
            db = np.load(path, allow_pickle=True)
            self.assertEqual(
                list(db["performer_groups"]),
                ["female", "male"],
            )
            self.assertEqual(report["performer_group_histogram"]["female"], 1)


if __name__ == "__main__":
    unittest.main()
