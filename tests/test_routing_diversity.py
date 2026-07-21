import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from routing.diversity import diversity_assessment, select_safe_diverse_proposal


class RoutingDiversityTests(unittest.TestCase):
    def setUp(self):
        self.db = {
            "event_uids": np.asarray(["e0", "e1", "e2", "e3"], dtype=object),
            "source_uids": np.asarray(["s0", "s0", "s1", "s2"], dtype=object),
            "event_families": np.asarray(["f0", "f1", "f1", "f2"], dtype=object),
            "dance_keys": np.asarray(["d0", "d1", "d1", "d2"], dtype=object),
        }

    def test_exact_event_cooldown_is_hard(self):
        result = diversity_assessment(self.db, 0, [2, 0])
        self.assertFalse(result["hard_valid"])
        self.assertIn("event_uid_cooldown", result["hard_reasons"])

    def test_safe_primary_is_preserved(self):
        rows = [
            (SimpleNamespace(event_id=2, safe=True, risk_score=0.3), {"heading_detail": {}}),
            (SimpleNamespace(event_id=3, safe=True, risk_score=0.1), {"heading_detail": {}}),
        ]
        selected, _extra, decision = select_safe_diverse_proposal(
            rows, db=self.db, selected_event_ids=[], primary_event_id=2
        )
        self.assertEqual(selected.event_id, 2)
        self.assertEqual(decision, "preserved_primary_safe")

    def test_source_run_reselects(self):
        rows = [
            (SimpleNamespace(event_id=1, safe=True, risk_score=0.01), {"heading_detail": {}}),
            (SimpleNamespace(event_id=2, safe=True, risk_score=0.2), {"heading_detail": {}}),
        ]
        with patch.dict(os.environ, {"V46_54_MAX_SOURCE_RUN": "1"}, clear=False):
            selected, _extra, decision = select_safe_diverse_proposal(
                rows, db=self.db, selected_event_ids=[0], primary_event_id=1
            )
        self.assertEqual(selected.event_id, 2)
        self.assertEqual(decision, "reselected_heading_physics_diverse")

    def test_cooldown_is_not_silently_relaxed_when_pool_is_exhausted(self):
        rows = [
            (SimpleNamespace(event_id=0, safe=True, risk_score=0.01), {"heading_detail": {}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "exhausted candidates"):
            select_safe_diverse_proposal(
                rows,
                db=self.db,
                selected_event_ids=[0],
                primary_event_id=0,
            )

    def test_physical_safety_is_never_relaxed(self):
        rows = [
            (SimpleNamespace(event_id=2, safe=False, risk_score=0.01), {"heading_detail": {}}),
            (SimpleNamespace(event_id=3, safe=False, risk_score=0.02), {"heading_detail": {}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "physically_safe=0"):
            select_safe_diverse_proposal(
                rows,
                db=self.db,
                selected_event_ids=[],
                primary_event_id=2,
            )


if __name__ == "__main__":
    unittest.main()
