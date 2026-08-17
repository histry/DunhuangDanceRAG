import unittest
from unittest.mock import patch

from scheduling.schedule_hard_constraints import (
    ScheduleHardConstraintError,
    assert_schedule_hard_constraints,
    audit_schedule_hard_constraints,
    final_selection_constraint_rows,
)
from scheduling.validate_schedule import audit_contract


def schedule_rows():
    rows = []
    for index, (source, event, motion_event) in enumerate(
        (
            ("source_a", "event_1", "pose_hold"),
            ("source_b", "event_2", "turning_climax"),
            ("source_c", "event_3", "footwork_flow"),
            ("source_d", "event_4", "arm_flourish"),
        )
    ):
        rows.append(
            {
                "slot": index,
                "start_frame": index * 25,
                "end_frame": (index + 1) * 25,
                "target_frames": 25,
                "duration_seconds": 25.0 / 30.0,
                "allocated_phrase_total": 25,
                "allocated_content_len": 20,
                "source_uid": source,
                "event_uid": event,
                "motion_event": motion_event,
                # This deliberately conflicts with motion_event in most rows.
                "music_event": "pose_hold",
                "music_alignment_label": "pose_hold",
            }
        )
    return rows


class ScheduleHardConstraintTests(unittest.TestCase):
    def test_boundary_values_pass_and_ignore_music_labels(self):
        report = audit_schedule_hard_constraints(schedule_rows())

        self.assertTrue(report["ok"])
        self.assertTrue(report["music_label_independent"])
        self.assertEqual(report["metrics"]["pose_hold_ratio"], 0.25)
        self.assertEqual(report["metrics"]["single_source_ratio"], 0.25)
        self.assertEqual(report["metrics"]["unique_event_count"], 4)
        self.assertEqual(report["metrics"]["core_frame_ratio"], 0.80)

    def test_pose_hold_ratio_is_a_hard_failure(self):
        rows = schedule_rows()
        rows[1]["motion_event"] = "pose_hold"
        report = audit_schedule_hard_constraints(rows)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("pose_hold_ratio_exceeded" in reason for reason in report["reasons"])
        )

    def test_single_source_ratio_is_a_hard_failure(self):
        rows = schedule_rows()
        rows[1]["source_uid"] = "source_a"
        report = audit_schedule_hard_constraints(rows)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "single_source_ratio_exceeded" in reason
                for reason in report["reasons"]
            )
        )

    def test_distinct_performer_tracks_cannot_hide_one_recording_share(self):
        rows = schedule_rows()
        rows[0]["recording_uid"] = "shared_recording"
        rows[1]["recording_uid"] = "shared_recording"
        rows[2]["recording_uid"] = "recording_c"
        rows[3]["recording_uid"] = "recording_d"

        report = audit_schedule_hard_constraints(rows)

        self.assertFalse(report["ok"])
        self.assertEqual(report["metrics"]["single_source_ratio"], 0.25)
        self.assertEqual(report["metrics"]["single_recording_ratio"], 0.50)
        self.assertTrue(
            any(
                "single_recording_ratio_exceeded" in reason
                for reason in report["reasons"]
            )
        )

    def test_recording_share_has_its_own_limit(self):
        rows = schedule_rows()
        rows[0]["recording_uid"] = "shared_recording"
        rows[1]["recording_uid"] = "shared_recording"

        report = audit_schedule_hard_constraints(
            rows,
            max_single_source_ratio=0.60,
            max_single_recording_ratio=0.40,
        )

        self.assertFalse(report["ok"])
        self.assertEqual(report["limits"]["max_single_recording_ratio"], 0.40)
        self.assertTrue(
            any(
                "single_recording_ratio_exceeded" in reason
                for reason in report["reasons"]
            )
        )

    def test_unique_event_count_is_a_hard_failure(self):
        rows = schedule_rows()
        rows[3]["event_uid"] = "event_1"
        report = audit_schedule_hard_constraints(rows)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "unique_event_count_below_minimum" in reason
                for reason in report["reasons"]
            )
        )

    def test_core_frame_ratio_is_a_hard_failure(self):
        rows = schedule_rows()
        for row in rows:
            row["allocated_content_len"] = 15
        report = audit_schedule_hard_constraints(rows)
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "core_frame_ratio_below_minimum" in reason
                for reason in report["reasons"]
            )
        )

    def test_short_schedule_does_not_relax_unique_event_minimum(self):
        report = audit_schedule_hard_constraints(
            schedule_rows()[:2],
            max_pose_hold_ratio=0.50,
            max_single_source_ratio=0.50,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "unique_event_count_below_minimum:2<4" in reason
                for reason in report["reasons"]
            )
        )

    def test_assertion_exposes_complete_failure_report(self):
        rows = schedule_rows()
        rows[1]["source_uid"] = "source_a"
        with self.assertRaises(ScheduleHardConstraintError) as captured:
            assert_schedule_hard_constraints(rows)
        self.assertIn("metrics", captured.exception.report)
        self.assertIn("single_source_ratio_exceeded", str(captured.exception))

    def test_final_schedule_validator_recomputes_hard_constraints(self):
        rows = schedule_rows()
        rows[1]["source_uid"] = "source_a"
        descriptor = {
            "usage": "generate_schedule",
            "is_final_schedule": True,
            "slot_source": "v21_router_v26_planner",
            "total_target_frames": 100,
            "slots": rows,
        }
        info = {
            "path": "synthetic.wav",
            "sha256": "abc",
            "target_frames": 100,
            "duration_seconds": 100.0 / 30.0,
        }
        with patch("scheduling.validate_schedule.audio_info", return_value=info):
            report = audit_contract(
                audio="synthetic.wav",
                schedule=descriptor,
                fps=30.0,
                require_fresh=False,
                require_raw_report=False,
                max_frame_error=0,
                max_seconds_error=1.0e-6,
            )

        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                reason.startswith("schedule_hard_constraint:")
                and "single_source_ratio_exceeded" in reason
                for reason in report["reasons"]
            )
        )

    def test_final_closed_loop_selection_is_resolved_from_event_db(self):
        db = {
            "event_uids": ["event_1", "event_2"],
            "source_uids": ["source_a", "source_b"],
            "recording_uids": ["recording_a", "recording_b"],
            "aesd_event_semantics": ["pose_hold", "turning_climax"],
        }
        rows = final_selection_constraint_rows(
            db,
            [
                {"slot": 0, "event_id": 1, "target_frames": 25, "core_frames": 20},
                {"slot": 1, "event_id": 0, "target_frames": 25, "core_frames": 20},
            ],
        )
        self.assertEqual(rows[0]["event_uid"], "event_2")
        self.assertEqual(rows[0]["source_uid"], "source_b")
        self.assertEqual(rows[0]["recording_uid"], "recording_b")
        self.assertEqual(rows[0]["motion_event"], "turning_climax")
        self.assertEqual(rows[0]["allocated_content_len"], 20)

    def test_unknown_recording_identity_falls_back_to_source(self):
        db = {
            "event_uids": ["event_1"],
            "source_uids": ["source_a"],
            "recording_uids": ["unknown"],
            "aesd_event_semantics": ["footwork_flow"],
        }
        rows = final_selection_constraint_rows(
            db,
            [{"event_id": 0, "target_frames": 25, "core_frames": 20}],
        )

        self.assertEqual(rows[0]["recording_uid"], "source_a")


if __name__ == "__main__":
    unittest.main()
