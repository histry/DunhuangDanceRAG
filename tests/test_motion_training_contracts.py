import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import training.motion_models as motion_runtime

from motion_geometry.resampling import blend_edge151_geodesic_np
from motion_geometry.rotations import (
    matrix_to_rot6d_np,
    rot6d_to_matrix_np,
    so3_exp_np,
    so3_geodesic_np,
)
from motion_geometry.smpl24 import skeleton_contract
from support.event_identity import make_event_db_contract
from training.motion_models import (
    MotionGenerationConfig,
    _finalize_generation_outputs,
    _new_validation_physical_accumulator,
    _record_validation_physical_prediction,
    _summarize_validation_physical_metrics,
    _descriptor_values_in_training_coordinates,
    _training_db_contract,
    _validate_source_disjoint,
    assert_motion_checkpoint_contract,
    build_frame_local_conditioning,
    identity6d_np,
    load_db,
    motion_checkpoint_contract,
    parse_args,
)


def _database(count=2, fps=30.0, sources=None):
    sources = sources or [f"source_{index}" for index in range(count)]
    uids = np.asarray([f"evt_{index}" for index in range(count)], dtype=object)
    raw = np.stack([
        np.linspace(float(index), float(index) + 1.0, 32, dtype=np.float32)
        for index in range(count)
    ])
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True) + 1.0e-6
    return {
        "paths": np.asarray([f"motion_{index}.npy" for index in range(count)], dtype=object),
        "desc": raw,
        "desc_z": ((raw - mean) / std).astype(np.float32),
        "desc_mean": mean.astype(np.float32),
        "desc_std": std.astype(np.float32),
        "canonical_fps": np.full(count, fps, dtype=np.float32),
        "skeleton_contract_json": np.asarray(
            json.dumps(skeleton_contract(), sort_keys=True), dtype=object
        ),
        "event_uids": uids,
        "event_db_contract_json": np.asarray(
            json.dumps(make_event_db_contract(uids), sort_keys=True), dtype=object
        ),
        "source_uids": np.asarray(sources, dtype=object),
    }


class MotionTrainingContractTests(unittest.TestCase):
    @staticmethod
    def _identity_motion(frames=60):
        motion = np.zeros((frames, 151), dtype=np.float32)
        motion[:, 7:151] = np.tile(identity6d_np(), 24)[None]
        return motion

    def test_training_db_rejects_false_fps_contract(self):
        cfg = MotionGenerationConfig()
        cfg.fps = 60.0
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _training_db_contract(_database(fps=30.0), cfg, "test")

    def test_motion_config_uses_normalized_pipeline_fps(self):
        with mock.patch.dict(os.environ, {"MOTION_FPS": "60"}, clear=False):
            cfg = MotionGenerationConfig().apply_env()
        self.assertEqual(cfg.fps, 60.0)

    def test_validation_sources_must_be_disjoint(self):
        train = _database(sources=["a", "b"])
        validation = _database(sources=["b", "c"])
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            _validate_source_disjoint(train, validation)

    def test_validation_descriptors_use_training_statistics(self):
        train = _database()
        validation = _database()
        validation["desc"] = validation["desc"] + 10.0
        aligned = _descriptor_values_in_training_coordinates(validation, train)
        split_local = (
            validation["desc"] - validation["desc"].mean(axis=0, keepdims=True)
        ) / (validation["desc"].std(axis=0, keepdims=True) + 1.0e-6)
        self.assertFalse(np.allclose(aligned, split_local))

    def test_database_relative_paths_do_not_depend_on_cwd(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as other:
            root = Path(td)
            motion = np.zeros((2, 151), dtype=np.float32)
            np.save(root / "motion.npy", motion)
            payload = _database(count=1, sources=["only"])
            payload["paths"] = np.asarray(["motion.npy"], dtype=object)
            db_path = root / "events.npz"
            np.savez_compressed(db_path, **payload)
            previous = Path.cwd()
            try:
                os.chdir(other)
                loaded = load_db(db_path)
            finally:
                os.chdir(previous)
            self.assertEqual(Path(loaded["paths"][0]), (root / "motion.npy").resolve())

    def test_geodesic_commit_is_stable_near_pi(self):
        reference = np.zeros((1, 151), dtype=np.float32)
        proposal = np.zeros_like(reference)
        identity = np.eye(3, dtype=np.float32)
        near_pi = so3_exp_np(np.asarray([0.0, np.pi - 1.0e-5, 0.0], dtype=np.float32))
        reference[:, 7:151] = matrix_to_rot6d_np(
            np.repeat(identity[None, None], 24, axis=1)
        ).reshape(1, -1)
        proposal[:, 7:151] = matrix_to_rot6d_np(
            np.repeat(near_pi[None, None], 24, axis=1)
        ).reshape(1, -1)
        midpoint = blend_edge151_geodesic_np(reference, proposal, 0.5)
        matrix = rot6d_to_matrix_np(midpoint[:, 7:151].reshape(1, 24, 6))
        angle = so3_geodesic_np(identity, matrix)[0]
        self.assertTrue(np.isfinite(midpoint).all())
        self.assertTrue(np.allclose(angle, (np.pi - 1.0e-5) * 0.5, atol=2.0e-4))

    def test_checkpoint_must_match_generation_event_database(self):
        cfg = MotionGenerationConfig()
        cfg._event_db_contract = make_event_db_contract(["evt_a"])
        checkpoint = {
            "motion_contract": motion_checkpoint_contract(cfg, "boundary_refiner"),
            "training_event_db_contract": make_event_db_contract(["evt_b"]),
        }
        with self.assertRaisesRegex(RuntimeError, "checkpoint/Generation"):
            assert_motion_checkpoint_contract(
                checkpoint, cfg, "refiner.pt", "boundary_refiner"
            )

    def test_motion_refiner_motion_cli_accepts_source_disjoint_validation_db(self):
        refiner = parse_args([
            "train-refiner", "--db", "train.npz", "--val_db", "val.npz", "--out", "out.pt"
        ])
        diffusion = parse_args([
            "train-diffusion", "--db", "train.npz", "--val_db", "val.npz", "--out", "out.pt"
        ])
        self.assertEqual(refiner.val_db, "val.npz")
        self.assertEqual(diffusion.val_db, "val.npz")

    def test_motion_training_cli_requires_validation_db(self):
        with self.assertRaises(SystemExit):
            parse_args(["train-refiner", "--db", "train.npz", "--out", "out.pt"])
        with self.assertRaises(SystemExit):
            parse_args(["train-diffusion", "--db", "train.npz", "--out", "out.pt"])

    def test_frame_local_conditioning_interpolates_transition(self):
        features = np.stack(
            [np.zeros(32, dtype=np.float32), np.ones(32, dtype=np.float32)]
        )
        reports = [
            {"target_frames": 3, "transition_span": None},
            {"target_frames": 3, "transition_span": [3, 5]},
        ]
        condition = build_frame_local_conditioning(
            features,
            reports,
            total_frames=6,
            descriptor_mean=np.zeros(32, dtype=np.float32),
            descriptor_std=np.ones(32, dtype=np.float32),
        )
        self.assertEqual(condition.shape, (6, 32))
        self.assertTrue(np.allclose(condition[:4], 0.0))
        self.assertTrue(np.allclose(condition[4:], 1.0))
        self.assertGreater(float(np.std(condition)), 0.0)

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_neural_models_accept_frame_local_conditioning(self):
        torch = motion_runtime.torch
        frames = 8
        condition = torch.randn(1, frames, 32)
        motion = torch.zeros(1, frames, 151)
        seam = torch.zeros(1, frames, 1)
        refiner = motion_runtime.TemporalRefiner()
        refined = refiner(motion, condition, seam)
        self.assertEqual(tuple(refined.shape), (1, frames, 151))

        denoiser = motion_runtime.DiffusionDenoiser()
        denoised = denoiser(
            motion,
            motion,
            condition,
            seam,
            torch.zeros(1, dtype=torch.long),
        )
        self.assertEqual(tuple(denoised.shape), (1, frames, 151))

    def test_generation_has_one_explicit_public_entrypoint(self):
        source = (Path(__file__).parents[1] / "training" / "motion_models.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(source.count("\ndef generate("), 1)
        self.assertNotIn("_stage_guard_orig_generate", source)
        self.assertNotIn("_energy_stability_orig_generate", source)
        self.assertNotIn("_physics_stability_orig_generate", source)
        self.assertNotIn("PHYSICS_STABILITY_MSA_MAX_CORRECTION_VEL_MPF", source)

    def test_validation_summary_contains_deployment_physical_metrics(self):
        cfg = MotionGenerationConfig()
        motion = self._identity_motion(60)
        accumulator = _new_validation_physical_accumulator()
        _record_validation_physical_prediction(accumulator, motion, motion, cfg)
        summary = _summarize_validation_physical_metrics(accumulator)
        self.assertEqual(summary["num_windows"], 1)
        self.assertIn("fk_position_error_m_p95", summary)
        self.assertIn("foot_skate_mps_p95", summary["worst_window"])
        self.assertIn("joint_jerk_mps3_p95", summary["worst_window"])
        self.assertIn("joint_rotation_step_rad_p95", summary["worst_window"])
        self.assertIn("root_horizontal_net_displacement_m", summary["worst_window"])

    def test_final_physical_gate_rejects_without_writing_accepted_output(self):
        cfg = MotionGenerationConfig()
        motion = self._identity_motion(60)
        motion[:, 4] = np.where(np.arange(60) % 2 == 0, 0.0, 2.0)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = Namespace(
                out=str(root / "motion.npy"),
                json=str(root / "report.json"),
                audio=None,
                render_output=None,
                render_script="rendering/render_motion.py",
            )
            np.save(root / "motion.npy", self._identity_motion(10))
            rc = _finalize_generation_outputs(
                args,
                cfg,
                motion,
                self._identity_motion(60),
                np.zeros((60, 1), dtype=np.float32),
                {},
            )
            self.assertEqual(rc, 2)
            self.assertFalse((root / "motion.npy").exists())
            self.assertTrue((root / "motion.rejected.npy").exists())
            self.assertEqual(len(list(root.glob("motion.preexisting_*.npy"))), 1)
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["generation_status"], "rejected_by_final_physical_gate")
            self.assertFalse(report["accepted_output_written"])
            self.assertFalse(report["final_physical_gate"]["ok"])
            self.assertIsNotNone(report["preexisting_accepted_output_quarantined"])


if __name__ == "__main__":
    unittest.main()
