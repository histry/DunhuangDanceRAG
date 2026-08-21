#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_pipeline import split_sources
from data_pipeline.split_sources import (
    assign_records_category_covered,
    exact_split_counts,
    leave_one_theme_out_assignment,
    recording_group_records,
    source_record,
)


class SourceAwareSplitTest(unittest.TestCase):
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
                        "performer_group": "female",
                        "dance_category": "thirty_six_postures",
                    },
                },
            )

            with mock.patch.object(
                split_sources.motion_api,
                "parse_change_bvh_semantics",
                side_effect=AssertionError(
                    "BVH parser entered formal SMPL path"
                ),
            ):
                record = source_record(root, motion)

        self.assertEqual(record["source_uid"], "female_36pose_1")
        self.assertEqual(record["recording_uid"], "female_36pose_sequence")
        self.assertEqual(
            record["semantic"]["source_format"],
            "chang_e_official_smpl",
        )

    def test_non_official_format_stays_on_legacy_bvh_parser(self):
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
            legacy_semantic = {
                "source_uid": "female_take_01",
                "recording_uid": "female_take_01",
                "performer_group": "female",
                "dance_key": "legacy",
            }

            with mock.patch.object(
                split_sources.motion_api,
                "parse_change_bvh_semantics",
                return_value=legacy_semantic,
            ) as parser:
                record = source_record(root, motion)

        parser.assert_called_once_with("female_take_01.bvh")
        self.assertEqual(record["source_uid"], "female_take_01")

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
