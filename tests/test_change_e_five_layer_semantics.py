#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from events.build_semantics import build_semantics
from motion_geometry.rotations import matrix_to_rot6d_np
from training import motion_models


def formal_meta(theme: str) -> dict:
    return {
        "source_format": "chang_e_official_smpl",
        "source_id": f"source_{theme}",
        "source_uid": f"source_{theme}",
        "recording_uid": f"recording_{theme}",
        "sequence_id": f"sequence_{theme}",
        "dancer_id": None,
        "dancer_id_status": "unverified",
        "performer_track_id": 1,
        "recording_performer_count": 1,
        "solo_compatibility": "single_track_recording",
        "solo_compatible": True,
        "solo_review_status": "not_required_single_track",
        "sequence_index": 1,
        "performer_group": "female",
        "dance_key": theme,
        "dance_category": theme,
        "theme_label_status": "confirmed",
        "source_context": ["drum"] if theme == "lei_gong_drum" else [],
        "manifest_sha256": "a" * 64,
    }


class ChangeEFiveLayerSemanticsTests(unittest.TestCase):
    def test_formal_solo_metadata_survives_semantic_resolution(self):
        resolved = motion_models.official_smpl_semantics_from_metadata(
            formal_meta("lotus_steps")
        )
        self.assertTrue(resolved["solo_compatible"])
        self.assertEqual(resolved["recording_performer_count"], 1)
        self.assertEqual(resolved["solo_compatibility"], "single_track_recording")

    def test_theme_never_creates_a_local_action_without_descriptor(self):
        meditation = motion_models.strong_action_semantics_from_meta(
            formal_meta("revelation_meditation")
        )
        drum = motion_models.strong_action_semantics_from_meta(
            formal_meta("lei_gong_drum")
        )
        self.assertEqual(meditation["event_family"], "unknown")
        self.assertEqual(drum["event_family"], "unknown")
        self.assertEqual(drum["local_action_labels"], ["unknown"])
        self.assertFalse(drum["source_context_is_local_action_truth"])

    def test_non_official_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "official SMPL"):
            motion_models.strong_action_semantics_from_meta(
                {"source_format": "legacy_bvh", "dance_key": "lei_gong_drum"}
            )

    def test_same_window_has_same_local_semantics_across_themes(self):
        desc = np.zeros(32, dtype=np.float32)
        desc[0] = 3.0
        desc[5] = 0.18
        desc[10] = 0.5
        desc[15] = 1.8
        desc[17] = 1.4
        desc[31] = 2.0
        outputs = [
            motion_models.strong_action_semantics_from_meta(
                formal_meta(theme), desc
            )
            for theme in ("revelation_meditation", "lei_gong_drum", "flying_apsaras")
        ]
        self.assertTrue(all(row["event_family"] == "turn_spin" for row in outputs))
        self.assertEqual(
            outputs[0]["local_action_scores_json"],
            outputs[1]["local_action_scores_json"],
        )
        self.assertTrue(all(row["music_alignment_label"] == "unknown" for row in outputs))
        self.assertTrue(
            all(not row["music_compatibility_is_ground_truth"] for row in outputs)
        )

    def test_floorwork_is_an_independent_local_action(self):
        desc = np.zeros(32, dtype=np.float32)
        desc[0] = 3.0
        desc[5] = 0.12
        desc[10] = 0.65
        desc[25] = 0.92
        desc[31] = 2.0
        result = motion_models.strong_action_semantics_from_meta(
            formal_meta("lotus_steps"), desc
        )
        self.assertEqual(result["event_family"], "floorwork")
        self.assertIn("floorwork", result["local_action_labels"])

    def test_formal_event_writer_cannot_reach_bvh_parser(self):
        cfg = motion_models.MotionGenerationConfig()
        frames = max(int(cfg.min_event_frames), 60)
        clip = np.zeros((frames, 151), dtype=np.float32)
        identity = matrix_to_rot6d_np(np.eye(3, dtype=np.float32)[None, None])
        clip[:, 7:151] = np.broadcast_to(identity, (frames, 24, 6)).reshape(
            frames, -1
        )
        meta = formal_meta("lotus_steps")
        meta.update(
            {
                "source_file": "C:/authoritative/female_lotus.npz",
                "source_asset": "C:/authoritative/female_lotus.npz",
                "source_start_seconds": 0.0,
                "source_end_seconds": frames / float(cfg.fps),
                "label": "lotus_steps",
                "parent_label": "lotus_steps",
            }
        )
        lists = {name: [] for name in (
            "descs", "entries", "exits", "c0s", "c1s",
            "music_feats", "music_masks", "meta",
        )}
        self.assertFalse(hasattr(motion_models, "parse_change_bvh_semantics"))
        with tempfile.TemporaryDirectory() as directory:
            motion_models.add_event_to_db_lists(
                clip=clip,
                event_idx=0,
                out_path=Path(directory) / "event.npy",
                cfg=cfg,
                source=meta["source_uid"],
                matched_audio=None,
                st=0,
                base_meta=meta,
                **lists,
            )
        self.assertEqual(
            lists["meta"][0]["source_format"], "chang_e_official_smpl"
        )
        self.assertTrue(lists["meta"][0]["solo_compatible"])

    def test_formal_aesd_reads_weak_compatibility_instead_of_theme_votes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.npz"
            target = root / "events_aesd.npz"
            compatibility = '{"lyrical_flow": 0.7, "pose_hold": 0.3}'
            np.savez_compressed(
                source,
                event_semantics_schema_version=np.asarray(
                    "chang_e_five_layer_event_semantics_v2", dtype=object
                ),
                paths=np.asarray(["a.npy", "b.npy"], dtype=object),
                source_groups=np.asarray(["a", "b"], dtype=object),
                source_uids=np.asarray(["a", "b"], dtype=object),
                recording_uids=np.asarray(["ra", "rb"], dtype=object),
                starts=np.asarray([0, 0], dtype=np.int32),
                ends=np.asarray([60, 60], dtype=np.int32),
                canonical_fps=np.asarray([30.0, 30.0], dtype=np.float32),
                dance_keys=np.asarray(
                    ["lei_gong_drum", "revelation_meditation"], dtype=object
                ),
                event_families=np.asarray(
                    ["upper_body_gesture", "upper_body_gesture"], dtype=object
                ),
                music_alignment_labels=np.asarray(["unknown", "unknown"], dtype=object),
                music_compatibility_scores_json=np.asarray(
                    [compatibility, compatibility], dtype=object
                ),
                desc=np.zeros((2, 32), dtype=np.float32),
                entry=np.zeros((2, 144), dtype=np.float32),
                exit=np.zeros((2, 144), dtype=np.float32),
                contact_entry=np.zeros((2, 4), dtype=np.float32),
                contact_exit=np.zeros((2, 4), dtype=np.float32),
            )
            report = build_semantics(source, target, prior_alpha=0.0)
            with np.load(target, allow_pickle=True) as data:
                probabilities = data["aesd_raw_music_alignment_probs"]
        np.testing.assert_allclose(probabilities[0], probabilities[1])
        self.assertEqual(
            report["semantic_evidence_source"],
            "formal_local_kinematic_weak_compatibility",
        )


if __name__ == "__main__":
    unittest.main()
