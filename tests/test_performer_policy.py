import os
import unittest
import numpy as np

from routing.performer_policy import resolve_candidate_policy, performer_switch_penalty


class PerformerPolicyTest(unittest.TestCase):
    def setUp(self):
        self.db = {
            "paths": np.asarray(["a", "b", "c", "d"], dtype=object),
            "genders": np.asarray(["female", "male", "female", "male"], dtype=object),
            "event_quality_scores": np.asarray([0.8, 0.7, 0.9, 0.6], dtype=np.float32),
        }

    def test_fixed_female(self):
        old = os.environ.get("PERFORMER_GROUP")
        os.environ["PERFORMER_GROUP"] = "female"
        try:
            rows, report = resolve_candidate_policy([[0, 1], [2, 3]], self.db)
            self.assertEqual(rows, [[0], [2]])
            self.assertEqual(report["resolved"], "female")
        finally:
            if old is None:
                os.environ.pop("PERFORMER_GROUP", None)
            else:
                os.environ["PERFORMER_GROUP"] = old

    def test_switch_penalty(self):
        old = os.environ.get("PERFORMER_GROUP")
        os.environ["PERFORMER_GROUP"] = "mixed"
        try:
            self.assertGreater(performer_switch_penalty(self.db, 0, 1, {}), 0.0)
            self.assertEqual(performer_switch_penalty(self.db, 0, 2, {}), 0.0)
        finally:
            if old is None:
                os.environ.pop("PERFORMER_GROUP", None)
            else:
                os.environ["PERFORMER_GROUP"] = old


if __name__ == "__main__":
    unittest.main()
