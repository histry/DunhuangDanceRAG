#!/usr/bin/env python3
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from data_pipeline import split_sources
from data_pipeline.split_sources import (
    assign_records_category_covered,
    exact_split_counts,
    leave_one_theme_out_assignment,
    recording_group_records,
    source_record,
)


class SourceAwareSplitTest(unittest.TestCase):
    def _committed_official_manifest_units(self):
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "motion"
            / "smpl_official_14"
            / "sources.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = []
        for source in manifest["sources"]:
            if not bool(source["solo_compatible"]):
                continue
            rows.append(
                {
                    "source_uid": source["source_id"],
                    "recording_uid": source["recording_uid"],
                    "performer_group": source["performer_group"],
                    "dance_key": source["dance_category"],
                    "theme_label_status": source["theme_label_status"],
                    "dancer_id": source["dancer_id"],
                    "dancer_id_status": source["dancer_id_status"],
                }
            )
        return manifest, recording_group_records(rows)

    def _write_report(self, root, report):
        motion = root / "segment_000.npy"
        motion.write_bytes(b"placeholder")
        motion.with_suffix(".retarget.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return motion

    def test_explicit_official_smpl_format_never_calls_bvh_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = self._write_report(
                root,
                {
                    "ok": True,
                    "source_gate_ok": True,
                    "source_format": "chang_e_official_smpl",
                    "source_metadata": {
                        "source_format": "chang_e_official_smpl",
                        "source_id": "female_36pose_1",
                        "recording_uid": "female_36pose_sequence",
                        "recording_performer_count": 2,
                        "solo_compatibility": "requires_manual_review",
                        "solo_compatible": False,
                        "solo_review_status": "pending_manual_review",
                        "performer_group": "female",
                        "dance_category": "thirty_six_postures",
                    },
                },
            )

            self.assertFalse(hasattr(split_sources, "motion_api"))
            record = source_record(root, motion)

        self.assertEqual(record["source_uid"], "female_36pose_1")
        self.assertEqual(record["recording_uid"], "female_36pose_sequence")
        self.assertEqual(
            record["semantic"]["source_format"],
            "chang_e_official_smpl",
        )

    def test_non_official_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = self._write_report(
                root,
                {
                    "ok": True,
                    "source_gate_ok": True,
                    "source_format": "legacy_bvh",
                    "source_used": "female_take_01.bvh",
                    "source_metadata": {
                        "source_id": "must_not_select_formal_branch",
                    },
                },
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "accepts only source_format=chang_e_official_smpl",
            ):
                source_record(root, motion)

    def test_direct_official_contract_without_format_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion = self._write_report(
                root,
                {
                    "ok": True,
                    "source_gate_ok": True,
                    "source_metadata": {"source_id": "female_lotus"},
                    "source_preprocess_contract": {
                        "direct_official_smpl": True,
                    },
                },
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "must declare source_format=chang_e_official_smpl",
            ):
                source_record(root, motion)

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

    def _official_recording_units(self):
        rows = []
        specs = [
            ("female_36", "thirty_six_postures", "female", "confirmed"),
            ("female_feitian", "flying_apsaras", "female", "confirmed"),
            ("female_lotus", "lotus_steps", "female", "confirmed"),
            ("female_meditation", "revelation_meditation", "female", "confirmed"),
            ("male_36", "thirty_six_postures", "male", "confirmed"),
            ("male_drum", "lei_gong_drum", "male", "confirmed"),
            ("male_meditation", "revelation_meditation", "male", "confirmed"),
            ("male_pipa_1", "pipa_behind_back", "male", "confirmed"),
            ("male_pipa_2", "pipa_behind_back", "male", "confirmed"),
            ("male_ribbon_1", "unknown", "male", "pending_official_confirmation"),
            ("male_ribbon_2", "unknown", "male", "pending_official_confirmation"),
        ]
        for uid, theme, gender, status in specs:
            rows.append(
                {
                    "source_uid": uid,
                    "recording_uid": uid,
                    "performer_group": gender,
                    "dance_key": theme,
                    "theme_label_status": status,
                    "dancer_id": None,
                    "dancer_id_status": "unverified",
                }
            )
        return recording_group_records(rows)

    def test_category_covered_split_keeps_unique_themes_in_training(self):
        units = self._official_recording_units()
        target = exact_split_counts(len(units), 0.67, 0.165, 0.165)
        assignment = assign_records_category_covered(units, target, seed=20260718)
        self.assertEqual(target, {"train": 7, "val": 2, "test": 2})
        train_themes = {
            row["dance_key"]
            for row in units
            if assignment[row["source_uid"]] == "train"
            and row["theme_label_status"] == "confirmed"
        }
        all_themes = {
            row["dance_key"]
            for row in units
            if row["theme_label_status"] == "confirmed"
        }
        self.assertEqual(train_themes, all_themes)
        for unique_theme in ("flying_apsaras", "lotus_steps", "lei_gong_drum"):
            unit = next(row for row in units if row["dance_key"] == unique_theme)
            self.assertEqual(assignment[unit["source_uid"]], "train")

    def test_formal_solo_split_has_two_groups_and_repeatable_theme_per_eval(self):
        units = [
            row
            for row in self._official_recording_units()
            if row["dance_key"]
            not in {"thirty_six_postures", "lei_gong_drum"}
        ]
        target = exact_split_counts(len(units), 0.50, 0.25, 0.25)
        assignment = assign_records_category_covered(
            units,
            target,
            seed=20260718,
        )
        self.assertEqual(target, {"train": 4, "val": 2, "test": 2})
        confirmed_counts = Counter(
            row["dance_key"]
            for row in units
            if row["theme_label_status"] == "confirmed"
        )
        repeatable = {
            theme for theme, count in confirmed_counts.items() if count >= 2
        }
        train_themes = {
            row["dance_key"]
            for row in units
            if assignment[row["source_uid"]] == "train"
            and row["theme_label_status"] == "confirmed"
        }
        self.assertEqual(train_themes, set(confirmed_counts))
        for split in ("val", "test"):
            split_rows = [
                row
                for row in units
                if assignment[row["source_uid"]] == split
            ]
            self.assertEqual(len(split_rows), 2)
            self.assertTrue(
                any(
                    row["theme_label_status"] == "confirmed"
                    and row["dance_key"] in repeatable
                    for row in split_rows
                )
            )

    def test_committed_smpl14_manifest_satisfies_formal_solo_protocol(self):
        manifest, units = self._committed_official_manifest_units()
        self.assertEqual(manifest["num_sources"], 14)
        self.assertEqual(manifest["num_recording_groups"], 11)
        self.assertEqual(len(units), 8)

        target = exact_split_counts(len(units), 0.50, 0.25, 0.25)
        assignment = assign_records_category_covered(
            units,
            target,
            seed=20260718,
        )
        counts = Counter(assignment.values())
        self.assertEqual(counts, Counter({"train": 4, "val": 2, "test": 2}))

        confirmed_counts = Counter(
            row["dance_key"]
            for row in units
            if row["theme_label_status"] == "confirmed"
        )
        repeatable = {
            theme for theme, count in confirmed_counts.items() if count >= 2
        }
        for split in ("val", "test"):
            split_rows = [
                row
                for row in units
                if assignment[row["source_uid"]] == split
            ]
            self.assertTrue(
                any(
                    row["theme_label_status"] == "confirmed"
                    and row["dance_key"] in repeatable
                    for row in split_rows
                )
            )

    def test_unique_theme_holdout_is_explicit_zero_shot_protocol(self):
        units = self._official_recording_units()
        assignment = leave_one_theme_out_assignment(
            units,
            heldout_theme="lei_gong_drum",
            seed=9,
        )
        heldout = [
            row for row in units if row["dance_key"] == "lei_gong_drum"
        ]
        self.assertTrue(heldout)
        self.assertTrue(
            all(assignment[row["source_uid"]] == "test" for row in heldout)
        )


if __name__ == "__main__":
    unittest.main()
