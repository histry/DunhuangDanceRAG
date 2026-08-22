from __future__ import annotations

import unittest
import tempfile
import json
import wave
from pathlib import Path

import numpy as np

from contracts.heading import (
    canonicalize_event_entry_heading_np,
    enforce_event_heading_contract,
    infer_turn_intent,
    unwrap_root_yaw_np,
)
from evaluation.validate_formal_route import validate_formal_route_report
from evaluation.audit_formal_single_person_db import audit_single_person_db
from motion_geometry.rotations import matrix_to_rot6d_np
from routing.boundary_closed_loop import (
    compute_condition,
    formal_ctsr_schedule_contract,
)
from routing.global_path import _slot_event_association
from training.music_corpus import audio_sha256
from scheduling.music_phrase_segmentation import whole_song_features


def sustained_turn_motion(frames: int = 91, degrees: float = 120.0) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    rotations = np.tile(identity, (frames, 24, 1, 1))
    angles = np.radians(np.linspace(0.0, degrees, frames))
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotations[:, 0, 0, 0] = cosine
    rotations[:, 0, 0, 2] = sine
    rotations[:, 0, 2, 0] = -sine
    rotations[:, 0, 2, 2] = cosine
    motion[:, 7:151] = matrix_to_rot6d_np(rotations).reshape(frames, 144)
    return motion


class FormalFailClosedContractTests(unittest.TestCase):
    @staticmethod
    def _formal_slot() -> dict:
        return {
            "router_architecture": "ctsr_weak_temporal_v1",
            "router_supervision_source": "semantic_ot_teacher",
            "router_compatibility_is_ground_truth": False,
            "action_compatibility_is_ground_truth": False,
            "hierarchy_semantic_contract": (
                "semantic_ot_teacher_x_weak_motion_local_action"
            ),
            "formal_candidate_contract": "ctsr_weak_scheduler_siblings_v1",
            "formal_candidate_event_uids": ["event-a", "event-b"],
            "formal_candidate_router_probabilities": [0.8, 0.2],
        }

    def test_formal_closed_loop_rejects_legacy_candidate_retrieval(self) -> None:
        slot = self._formal_slot()
        self.assertTrue(formal_ctsr_schedule_contract([slot]))
        slot.pop("formal_candidate_event_uids")
        with self.assertRaisesRegex(RuntimeError, "empty CTSR candidate"):
            formal_ctsr_schedule_contract([slot])

    def test_formal_repair_condition_tracks_reselected_motion(self) -> None:
        class Runtime:
            @staticmethod
            def build_frame_local_conditioning(
                features,
                _report,
                total_frames,
                descriptor_mean,
                descriptor_std,
            ):
                del descriptor_mean, descriptor_std
                return np.repeat(features, int(total_frames), axis=0)

        descriptors = np.stack(
            [np.full(32, 0.25, dtype=np.float32), np.full(32, 0.75, dtype=np.float32)]
        )
        condition = compute_condition(
            Runtime(),
            np.zeros((1, 32), dtype=np.float32),
            [
                {
                    "event_id": 1,
                    "target_frames": 2,
                    "conditioning_contract": "selected_event_motion_descriptor_v1",
                }
            ],
            total_frames=2,
            db={
                "desc": descriptors,
                "desc_mean": np.zeros(32, dtype=np.float32),
                "desc_std": np.ones(32, dtype=np.float32),
            },
        )
        np.testing.assert_allclose(condition, 0.75)

    def test_graph_sb_formal_unary_uses_router_probability_not_grounder(self) -> None:
        slot = self._formal_slot()
        slot["formal_candidate_router_probabilities"] = [0.2, 0.8]
        value, source = _slot_event_association(
            slot,
            {"event_uids": np.asarray(["event-a", "event-b"], dtype=object)},
            1,
        )
        self.assertAlmostEqual(value, 1.0)
        self.assertEqual(source, "ctsr_candidate_probability")

    def test_unproven_router_npy_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "song.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes(b"\x00\x00" * 8000)
            digest = audio_sha256(audio)
            cache = root / "cache"
            cache.mkdir()
            feature_path = cache / f"song_{digest[:16]}_whole_song_fps8_8.npy"
            np.save(feature_path, np.zeros((8, 12), dtype=np.float32))
            feature_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "cache_schema": "music_12d_content_addressed_cache_v2",
                        "audio_sha256": digest,
                        "extractor": {
                            "backend": "wave_fallback",
                            "backend_version": None,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "not proven to use Librosa"):
                whole_song_features(audio, fps=8.0, cache_dir=cache, require_librosa=True)

    def test_motion_evident_turn_is_not_dropped_without_theme_prior(self) -> None:
        motion = sustained_turn_motion()
        inferred = infer_turn_intent(motion, {}, fps=30.0)
        self.assertEqual(inferred["intent"], "turn")
        self.assertTrue(inferred["motion_turn_evidence"])
        _corrected, report = enforce_event_heading_contract(motion, {}, fps=30.0)
        self.assertTrue(report["valid"])
        self.assertNotIn(report["reason"], {"drop_reset_or_drift", "non_turn_yaw_exceeds_hard_budget"})

    def test_event_entry_heading_anchors_physical_first_frame(self) -> None:
        # The first 0.15 seconds already contain a fast turn, so their circular
        # reference is deliberately different from the physical first frame.
        motion = sustained_turn_motion(frames=31, degrees=180.0)
        before = unwrap_root_yaw_np(motion)

        canonical, report = canonicalize_event_entry_heading_np(
            motion,
            fps=30.0,
        )
        after = unwrap_root_yaw_np(canonical)

        self.assertEqual(
            report["entry_anchor_contract"],
            "physical_first_frame_yaw_zero_v1",
        )
        self.assertGreater(
            abs(float(report["entry_heading_window_reference_deg"])),
            5.0,
        )
        self.assertLess(abs(float(np.degrees(after[0]))), 1.0e-3)
        self.assertLess(abs(float(report["entry_heading_after_deg"])), 1.0e-3)
        self.assertAlmostEqual(
            float(after[-1] - after[0]),
            float(before[-1] - before[0]),
            places=4,
        )

    def test_formal_graph_sb_acceptance(self) -> None:
        slot = self._formal_slot()
        report = {
            "graph_route_graph_sb_route": {
                "solver": "fisher_rao_graph_sb",
                "fallback_used": False,
                "unary_semantic_contract": "ctsr_candidate_probability",
                "schrodinger": {"converged": True},
                "trace": [
                    {
                        "candidates": [
                            {
                                "event_id": 3,
                                "association_source": "ctsr_candidate_probability",
                            }
                        ]
                    }
                ],
            },
            "slots": [slot],
            "selected_event_indices_final": [3],
            "stage_reports": {
                "retrieval": [
                    {
                        "routing_policy": "formal_ctsr_scheduler_locked_candidates",
                        "candidate_event_indices": [3, 4],
                    }
                ],
                "neural_music_conditioning": {
                    "conditioning_contract": "selected_event_motion_descriptor_v1",
                    "categorical_music_label_used_as_body_semantics": False,
                },
            },
        }
        self.assertTrue(validate_formal_route_report(report)["ok"])

    def test_legacy_fallback_is_rejected(self) -> None:
        report = {
            "event_geometry_global_route": {
                "solver": "legacy_beam",
                "fallback_used": True,
            }
        }
        with self.assertRaisesRegex(RuntimeError, "fallback_used=true"):
            validate_formal_route_report(report)

    def test_single_person_db_audit_rejects_unreviewed_pair_event(self) -> None:
        db = {
            "paths": np.asarray(["event.npy"], dtype=object),
            "solo_compatible": np.asarray([False]),
            "solo_compatibilities": np.asarray(["requires_manual_review"], dtype=object),
            "solo_review_statuses": np.asarray(["pending_manual_review"], dtype=object),
            "recording_performer_counts": np.asarray([2]),
            "dancer_ids": np.asarray([""], dtype=object),
            "dancer_id_statuses": np.asarray(["unverified"], dtype=object),
            "source_uids": np.asarray(["source"], dtype=object),
            "dance_categories": np.asarray(["thirty_six_postures"], dtype=object),
        }
        result = audit_single_person_db(db)
        self.assertFalse(result["ok"])
        self.assertFalse(result["same_dancer_claim_supported"])


if __name__ == "__main__":
    unittest.main()
