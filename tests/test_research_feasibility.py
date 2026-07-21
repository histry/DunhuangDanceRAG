import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from tools.research_feasibility_contract import (
        FeasibilityPolicy,
        duration_feasible,
        sanitize_slot,
    )
except ImportError:
    from routing.feasibility_contract import (
        FeasibilityPolicy,
        duration_feasible,
        sanitize_slot,
    )


class ResearchFeasibilityTest(unittest.TestCase):
    def test_duration_contract(self):
        policy = FeasibilityPolicy()
        self.assertTrue(duration_feasible(45, 59, False, policy, tier=1))
        self.assertFalse(duration_feasible(39, 59, False, policy, tier=1))
        self.assertTrue(duration_feasible(39, 59, True, policy, tier=2))

    def test_legacy_scheduler_event_identity_is_provenance_only(self):
        slot = {
            "event_id": "old_event",
            "event_index": 585,
            "family_id": "46:9",
            "target_frames": 59,
            "duration": 1.0,
            "music_semantic_top_label": "calm_meditative",
        }
        clean = sanitize_slot(slot, 30.0)
        self.assertNotIn("event_id", clean)
        self.assertNotIn("event_index", clean)
        self.assertEqual(clean["target_frames"], 59)
        self.assertAlmostEqual(clean["duration"], 59 / 30.0)
        self.assertFalse(clean["scheduler_event_identity_authoritative"])
        self.assertEqual(
            clean["scheduler_event_provenance"]["event_id"],
            "old_event",
        )

    def test_stable_uid_is_authoritative_only_after_contract_alignment(self):
        slot = {
            "event_id": "old_event",
            "event_index": 585,
            "v26_event_uid": "evt_0123456789abcdef",
            "target_frames": 90,
        }
        unaligned = sanitize_slot(slot, 30.0)
        self.assertFalse(unaligned["scheduler_event_identity_authoritative"])
        self.assertNotIn("event_uid", unaligned)
        self.assertEqual(
            unaligned["scheduler_event_provenance"]["event_uid"],
            "evt_0123456789abcdef",
        )

        aligned = sanitize_slot(slot, 30.0, aligned_event_db=True)
        self.assertTrue(aligned["scheduler_event_identity_authoritative"])
        self.assertEqual(aligned["event_uid"], "evt_0123456789abcdef")
        self.assertEqual(aligned["v26_event_uid"], "evt_0123456789abcdef")
        self.assertNotIn("event_id", aligned)


if __name__ == "__main__":
    unittest.main()
