import json
import tempfile
import unittest
from pathlib import Path

from evaluation.audit_event_database import (
    load_split_membership_contract,
    split_membership_reasons,
)


class EventDatabaseSplitAuditTests(unittest.TestCase):
    def _manifest(
        self,
        root: Path,
        *,
        val_recordings=None,
        protocol="category_covered_source_disjoint",
    ) -> Path:
        val_recordings = val_recordings or ["recording_a", "recording_b"]
        payload = {
            "schema": (
                "category_covered_recording_disjoint_cache_split_"
                "v5_manifest_audited"
            ),
            "ok": True,
            "reasons": [],
            "split_protocol": protocol,
            "target_counts": {
                "train": 4,
                "val": len(val_recordings),
                "test": 2,
            },
            "splits": {
                "train": {
                    "source_uids": ["source_t1", "source_t2"],
                    "recording_uids": [
                        "recording_t1",
                        "recording_t2",
                        "recording_t3",
                        "recording_t4",
                    ],
                },
                "val": {
                    "source_uids": [
                        value.replace("recording", "source")
                        for value in val_recordings
                    ],
                    "recording_uids": val_recordings,
                },
                "test": {
                    "source_uids": ["source_c", "source_d"],
                    "recording_uids": ["recording_c", "recording_d"],
                },
            },
        }
        path = root / "source_split_manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_exact_manifest_membership_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = load_split_membership_contract(
                self._manifest(Path(tmp)),
                "val",
            )
        self.assertEqual(contract["expected_num_recording_uids"], 2)
        self.assertEqual(
            split_membership_reasons(
                ["source_a", "source_b", "source_a"],
                ["recording_a", "recording_b", "recording_a"],
                contract,
            ),
            [],
        )

    def test_manifest_membership_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = load_split_membership_contract(
                self._manifest(Path(tmp)),
                "val",
            )
        reasons = split_membership_reasons(
            ["source_a"],
            ["recording_a"],
            contract,
        )
        self.assertTrue(any("source_split_membership_mismatch" in x for x in reasons))
        self.assertTrue(
            any("recording_split_membership_mismatch" in x for x in reasons)
        )

    def test_loto_single_group_is_allowed_only_by_exact_manifest_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract = load_split_membership_contract(
                self._manifest(
                    Path(tmp),
                    val_recordings=["recording_a"],
                    protocol="leave_one_theme_out",
                ),
                "val",
            )
        self.assertEqual(contract["expected_num_recording_uids"], 1)
        self.assertEqual(
            split_membership_reasons(
                ["source_a"],
                ["recording_a"],
                contract,
            ),
            [],
        )

    def test_ordinary_single_group_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._manifest(
                Path(tmp),
                val_recordings=["recording_a"],
            )
            with self.assertRaisesRegex(RuntimeError, "at least two"):
                load_split_membership_contract(manifest, "val")


if __name__ == "__main__":
    unittest.main()
