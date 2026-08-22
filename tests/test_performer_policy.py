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

    def test_solo_filter_and_fixed_source_identity(self):
        self.db.update(
            {
                "source_uids": np.asarray(["a", "b", "a", "b"], dtype=object),
                "solo_compatible": np.asarray([True, False, True, False]),
            }
        )
        names = (
            "PERFORMER_GROUP",
            "PERFORMER_IDENTITY_MODE",
            "PERFORMER_REQUIRE_SOLO_COMPATIBLE",
        )
        old = {name: os.environ.get(name) for name in names}
        os.environ.update(
            {
                "PERFORMER_GROUP": "female",
                "PERFORMER_IDENTITY_MODE": "fixed_source",
                "PERFORMER_REQUIRE_SOLO_COMPATIBLE": "1",
            }
        )
        try:
            rows, report = resolve_candidate_policy([[0, 1], [2, 3]], self.db)
            self.assertEqual(rows, [[0], [2]])
            self.assertEqual(report["resolved_identity"], "a")
            self.assertTrue(report["same_source_track_guaranteed"])
            self.assertEqual(report["excluded_unreviewed_pair_candidates"], 2)
        finally:
            for name, value in old.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_fixed_dancer_rejects_unverified_identity(self):
        self.db.update(
            {
                "dancer_ids": np.asarray(["", "", "", ""], dtype=object),
                "dancer_id_statuses": np.asarray(["unverified"] * 4, dtype=object),
            }
        )
        old = os.environ.get("PERFORMER_IDENTITY_MODE")
        os.environ["PERFORMER_IDENTITY_MODE"] = "fixed_dancer"
        try:
            with self.assertRaisesRegex(RuntimeError, "verified non-empty"):
                resolve_candidate_policy([[0, 1], [2, 3]], self.db)
        finally:
            if old is None:
                os.environ.pop("PERFORMER_IDENTITY_MODE", None)
            else:
                os.environ["PERFORMER_IDENTITY_MODE"] = old


if __name__ == "__main__":
    unittest.main()
