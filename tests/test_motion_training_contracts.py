import json
import inspect
import os
import tempfile
import unittest
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
    _checkpoint_validation_decision,
    _new_validation_physical_accumulator,
    _record_validation_physical_prediction,
    _summarize_validation_physical_metrics,
    _descriptor_values_in_training_coordinates,
    _training_db_contract,
    _validate_source_disjoint,
    assert_motion_checkpoint_contract,
    build_frame_local_conditioning,
    degrade_for_refiner,
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

    def test_motion_audit_reuses_one_fk_for_contacts_and_metrics(self):
        motion = self._identity_motion(60)
        original_fk = motion_runtime.fk_24_np
        with mock.patch.object(
            motion_runtime,
            "fk_24_np",
            wraps=original_fk,
        ) as training_fk, mock.patch(
            "motion_geometry.physical.fk24_np",
            side_effect=AssertionError("physical audit repeated FK"),
        ):
            report = motion_runtime.audit_motion_np(motion)

        self.assertEqual(training_fk.call_count, 1)
        self.assertEqual(report["frames"], len(motion))

    def test_formal_event_duration_bounds_are_explicit_and_overridable(self):
        cfg = MotionGenerationConfig.from_json("configs/motion_model.json")
        self.assertEqual(cfg.min_event_frames, 45)
        self.assertEqual(cfg.max_event_frames, 120)
        self.assertGreater(cfg.max_event_frames, cfg.min_event_frames)
        with mock.patch.dict(
            os.environ,
            {"MOTION_MAX_EVENT_FRAMES": "180"},
            clear=False,
        ):
            overridden = MotionGenerationConfig.from_json(
                "configs/motion_model.json"
            ).apply_env()
        self.assertEqual(overridden.max_event_frames, 180)

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
        joint_mask = torch.ones(1, frames, 24)
        refiner = motion_runtime.ProductManifoldTemporalRefiner()
        refined = refiner(motion, condition, seam, joint_mask)
        self.assertEqual(tuple(refined.shape), (1, frames, 79))

        denoiser = motion_runtime.TangentDiffusionDenoiser()
        denoised = denoiser(
            torch.zeros(1, frames, 79),
            motion,
            condition,
            seam,
            joint_mask,
            torch.zeros(1, dtype=torch.long),
        )
        self.assertEqual(tuple(denoised.shape), (1, frames, 79))

    def test_preloaded_pool_reads_fixed_event_once(self):
        cfg = MotionGenerationConfig()
        cfg.window_len = 60
        motion = self._identity_motion(60)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "event.npy"
            np.save(path, motion)
            original_load = np.load
            with mock.patch.object(
                motion_runtime.np,
                "load",
                wraps=original_load,
            ) as load_mock:
                pool = motion_runtime.PreloadedMotionWindowPool.preload(
                    [path, path],
                    cfg.window_len,
                    cfg,
                    mode="refiner",
                )
                first = pool.sample(0)
                second = pool.sample(1)
            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(pool.unique_paths, 1)
            self.assertEqual(len(pool.fixed_windows), 2)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(first.flags.writeable)

    def test_preloaded_refiner_pool_preserves_random_crop(self):
        cfg = MotionGenerationConfig()
        cfg.window_len = 60
        motion = self._identity_motion(80)
        motion[:, 4] = np.arange(80, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "long_event.npy"
            np.save(path, motion)
            pool = motion_runtime.PreloadedMotionWindowPool.preload(
                [path],
                cfg.window_len,
                cfg,
                mode="refiner",
            )
            self.assertIsNone(pool.fixed_windows[0])
            with mock.patch.object(
                motion_runtime.random,
                "randint",
                side_effect=[0, 20],
            ):
                first = pool.sample(0)
                second = pool.sample(0)
            self.assertFalse(np.array_equal(first[:, 4], second[:, 4]))
            self.assertEqual(first.shape, (60, 151))
            self.assertEqual(second.shape, (60, 151))

    def test_training_bridge_can_defer_redundant_contract_finalization(self):
        cfg = MotionGenerationConfig()
        previous = self._identity_motion(4)
        following = self._identity_motion(4)
        following[:, 4] = 0.25
        finalized = motion_runtime.reference_motion_inbetween_np(
            previous,
            following,
            20,
            cfg,
        )
        deferred = motion_runtime.reference_motion_inbetween_np(
            previous,
            following,
            20,
            cfg,
            finalize_contract=False,
        )
        manually_finalized, _ = motion_runtime.enforce_edge151_contract_np(
            deferred,
            cfg,
            source_hint="test_deferred_training_bridge",
            derive_contact=True,
            project_rot=True,
        )
        self.assertTrue(
            np.allclose(finalized, manually_finalized, atol=2.0e-6)
        )

    def test_parallel_batch_risk_masks_match_serial_order(self):
        cfg = MotionGenerationConfig()
        motion = self._identity_motion(30)
        motion[8:13, 4] = np.linspace(0.0, 0.2, 5, dtype=np.float32)
        batch = np.stack([motion, motion[::-1].copy()])
        seam = np.zeros((2, 30, 1), dtype=np.float32)
        seam[0, 8:13] = 1.0
        seam[1, 17:22] = 1.0
        serial = motion_runtime._risk_masks_for_batch_np(batch, seam, cfg)
        with motion_runtime.ThreadPoolExecutor(max_workers=2) as executor:
            parallel = motion_runtime._risk_masks_for_batch_np(
                batch,
                seam,
                cfg,
                executor=executor,
            )
        for expected, actual in zip(serial, parallel):
            self.assertTrue(np.array_equal(expected, actual))

    def test_motion_training_hot_loops_use_preload_and_one_batch_mask_call(self):
        refiner_source = inspect.getsource(motion_runtime.train_refiner)
        diffusion_source = inspect.getsource(motion_runtime.train_diffusion)
        for source in (refiner_source, diffusion_source):
            self.assertIn("PreloadedMotionWindowPool.preload", source)
            self.assertEqual(source.count("_risk_masks_for_batch_np("), 1)
        self.assertNotIn("np.load", diffusion_source)

    def test_training_progress_is_unbuffered_and_reports_eta(self):
        with mock.patch("builtins.print") as print_mock:
            motion_runtime._emit_training_progress(
                "[test]",
                0,
                10,
                motion_runtime.time.perf_counter() - 1.0,
                loss=1.0,
            )
        self.assertTrue(print_mock.call_args.kwargs["flush"])
        output = print_mock.call_args.args[0]
        self.assertIn("step=1/10", output)
        self.assertIn("steps_per_second=", output)
        self.assertIn("eta_minutes=", output)

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_physics_loss_contract_remains_reachable(self):
        torch = motion_runtime.torch
        cfg = MotionGenerationConfig()
        motion = torch.from_numpy(self._identity_motion(8))[None]
        total, terms = motion_runtime._world_space_physics_losses(
            motion,
            motion,
            cfg,
        )
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertEqual(
            set(terms),
            {
                "fk",
                "foot",
                "support",
                "penetration",
                "acceleration",
                "jerk",
                "endpoint_continuity",
                "seam_velocity",
                "seam_acceleration",
                "seam_jerk",
                "relative_temporal",
            },
        )

    def test_validation_summary_contains_deployment_physical_metrics(self):
        cfg = MotionGenerationConfig()
        motion = self._identity_motion(60)
        accumulator = _new_validation_physical_accumulator()
        _record_validation_physical_prediction(
            accumulator,
            motion,
            motion,
            cfg,
            degraded=motion,
        )
        summary = _summarize_validation_physical_metrics(accumulator)
        self.assertEqual(summary["num_windows"], 1)
        self.assertEqual(summary["schema"], "motion_checkpoint_stage_validation_v3")
        self.assertIn("fk_position_error_m_p95", summary)
        self.assertIn("foot_skate_mps_p95", summary["worst_window"])
        self.assertIn("joint_jerk_mps3_p95", summary["worst_window"])
        self.assertIn("joint_rotation_step_rad_p95", summary["worst_window"])
        self.assertIn("root_horizontal_net_displacement_m", summary["worst_window"])
        self.assertEqual(summary["stage_repair"]["pass_rate"], 1.0)
        self.assertEqual(summary["temporal_repair"]["pass_rate"], 1.0)
        self.assertEqual(
            summary["stage_repair"]["geometry_repair_gain"]["observed"][
                "mean"
            ],
            1.0,
        )
        self.assertEqual(
            summary["stage_repair"][
                "prediction_product_log_l1_to_clean"
            ]["mean"],
            0.0,
        )
        self.assertEqual(summary["clean_reference_fidelity"]["pass_rate"], 1.0)
        self.assertEqual(
            summary["clean_physical_non_regression"]["pass_rate"], 1.0
        )
        self.assertFalse(
            summary["final_generation_gate_diagnostic"]["checkpoint_criterion"]
        )

    def test_degradation_is_local_smooth_and_stays_on_product_manifold(self):
        cfg = MotionGenerationConfig()
        clean = self._identity_motion(120)
        clean[:, 4] = np.linspace(0.0, 0.6, len(clean), dtype=np.float32)
        with mock.patch.object(motion_runtime.random, "randint", side_effect=[24, 60]):
            np.random.seed(20260824)
            degraded, seam = degrade_for_refiner(
                clean,
                cfg=cfg,
                finalize_contract=False,
            )

        inactive = seam[:, 0] == 0.0
        np.testing.assert_array_equal(degraded[inactive, 4:], clean[inactive, 4:])
        self.assertGreater(float(np.max(np.abs(degraded - clean))), 0.0)
        matrices = rot6d_to_matrix_np(degraded[:, 7:151].reshape(-1, 24, 6))
        determinants = np.linalg.det(matrices)
        np.testing.assert_allclose(determinants, 1.0, atol=2.0e-5)
        self.assertTrue(np.isfinite(degraded).all())

    def test_refiner_receptive_field_covers_formal_training_seam(self):
        cfg = MotionGenerationConfig()
        model = motion_runtime.ProductManifoldTemporalRefiner()
        maximum_seam = int(
            round(cfg.transition_train_max_seconds * cfg.fps)
        )
        self.assertEqual(model.temporal_dilations, (1, 2, 5))
        self.assertGreaterEqual(
            model.temporal_receptive_field_frames,
            maximum_seam + 1,
        )

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_refiner_loss_is_normalized_on_corrupted_seam_and_requires_gain(self):
        torch = motion_runtime.torch
        cfg = MotionGenerationConfig(device="cpu")
        clean = self._identity_motion(60)
        clean_t = torch.from_numpy(clean[None])
        degraded_t = clean_t.clone()
        degraded_t[:, 20:40, motion_runtime.ROOT_X_IDX] += 0.05
        seam = torch.zeros((1, 60, 1), dtype=torch.float32)
        seam[:, 20:40] = 1.0
        joint = torch.zeros((1, 60, 24), dtype=torch.float32)
        root = torch.ones((1, 60, 1), dtype=torch.float32)
        contact = torch.zeros((1, 60, 1), dtype=torch.float32)

        _, unchanged = motion_runtime._product_motion_losses(
            degraded_t,
            clean_t,
            degraded_t,
            joint,
            root,
            contact,
            cfg,
            seam_mask=seam,
        )
        _, repaired = motion_runtime._product_motion_losses(
            clean_t,
            clean_t,
            degraded_t,
            joint,
            root,
            contact,
            cfg,
            seam_mask=seam,
        )
        harmed_t = clean_t.clone()
        harmed_t[:, 0:10, motion_runtime.ROOT_X_IDX] += 0.02
        _, harmed = motion_runtime._product_motion_losses(
            harmed_t,
            clean_t,
            degraded_t,
            joint,
            root,
            contact,
            cfg,
            seam_mask=seam,
        )

        self.assertGreater(
            unchanged["active_reconstruction"].item(),
            unchanged["reconstruction"].item(),
        )
        self.assertGreater(unchanged["repair_margin"].item(), 0.0)
        self.assertAlmostEqual(
            unchanged["repair_margin"].item(),
            cfg.product_refiner_training_target_repair_gain,
            places=5,
        )
        self.assertAlmostEqual(repaired["active_reconstruction"].item(), 0.0)
        self.assertAlmostEqual(repaired["repair_margin"].item(), 0.0)
        self.assertAlmostEqual(repaired["clean_preservation"].item(), 0.0)
        self.assertGreater(harmed["clean_preservation"].item(), 0.0)

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_refiner_soft_mask_scales_before_applied_tangent_cap(self):
        torch = motion_runtime.torch
        cfg = MotionGenerationConfig(device="cpu")
        clean = torch.from_numpy(self._identity_motion(12))[None]
        output = torch.zeros((1, 12, motion_runtime.PRODUCT_STATE_DIM))
        mask_value = 0.18
        target_angle = 0.10
        output[..., 4 + 3] = target_angle / mask_value
        joint = torch.zeros((1, 12, 24))
        joint[..., 0] = mask_value
        root = torch.zeros((1, 12, 1))
        contact = torch.zeros((1, 12, 1))

        prediction = motion_runtime._decode_product_refiner_output(
            clean, output, joint, root, contact, cfg
        )
        applied = motion_runtime.product_log_torch(clean, prediction)
        angle = torch.linalg.vector_norm(
            applied[..., 3:6], dim=-1
        ).mean()

        self.assertAlmostEqual(float(angle), target_angle, places=5)

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_clean_identity_loss_rejects_refiner_edits(self):
        torch = motion_runtime.torch
        cfg = MotionGenerationConfig(device="cpu")
        clean = torch.from_numpy(self._identity_motion(30))[None]
        joint = torch.ones((1, 30, 24))
        root = torch.ones((1, 30, 1))
        contact = torch.ones((1, 30, 1))
        exact, _ = motion_runtime._product_refiner_clean_identity_loss(
            clean, clean, joint, root, contact, cfg
        )
        harmed = clean.clone()
        harmed[:, 8:22, motion_runtime.ROOT_X_IDX] += 0.02
        changed, _ = motion_runtime._product_refiner_clean_identity_loss(
            harmed, clean, joint, root, contact, cfg
        )

        self.assertAlmostEqual(float(exact), 0.0, places=7)
        self.assertGreater(float(changed), 0.0)

    def test_transition_bridge_mix_is_fail_closed(self):
        cfg = MotionGenerationConfig()
        cfg.transition_bridge_mix = 0.0
        clean = self._identity_motion(120)
        with self.assertRaisesRegex(ValueError, "transition_bridge_mix"):
            with mock.patch.object(
                motion_runtime.random,
                "randint",
                side_effect=[24, 60],
            ):
                degrade_for_refiner(
                    clean,
                    cfg=cfg,
                    finalize_contract=False,
                )

    def test_authentic_event_travel_is_not_a_checkpoint_failure(self):
        cfg = MotionGenerationConfig()
        motion = self._identity_motion(60)
        motion[:, 4] = np.linspace(0.0, 1.0, len(motion), dtype=np.float32)
        accumulator = _new_validation_physical_accumulator()
        _record_validation_physical_prediction(
            accumulator,
            motion,
            motion,
            cfg,
            degraded=motion,
        )
        summary = _summarize_validation_physical_metrics(accumulator)

        self.assertEqual(summary["stage_repair"]["pass_rate"], 1.0)
        self.assertEqual(summary["clean_reference_fidelity"]["pass_rate"], 1.0)
        self.assertEqual(
            summary["clean_physical_non_regression"]["pass_rate"], 1.0
        )
        self.assertEqual(
            summary["final_generation_gate_diagnostic"]["prediction"]["pass_rate"],
            0.0,
        )

    def test_unchanged_degraded_input_is_not_counted_as_repair(self):
        cfg = MotionGenerationConfig()
        clean = self._identity_motion(120)
        with mock.patch.object(motion_runtime.random, "randint", side_effect=[24, 60]):
            np.random.seed(20260824)
            degraded, _ = degrade_for_refiner(
                clean,
                cfg=cfg,
                finalize_contract=False,
            )
        accumulator = _new_validation_physical_accumulator()
        _record_validation_physical_prediction(
            accumulator,
            degraded,
            clean,
            cfg,
            degraded=degraded,
        )
        summary = _summarize_validation_physical_metrics(accumulator)

        self.assertEqual(summary["stage_repair"]["pass_rate"], 0.0)
        self.assertIn(
            "no_meaningful_geometry_repair_gain",
            summary["stage_repair"]["failure_reasons"],
        )

    def test_checkpoint_decision_ignores_final_gate_diagnostic(self):
        cfg = MotionGenerationConfig()
        metrics = {
            "reconstruction_product_log_l1": 0.01,
            "physical_quality": {
                "num_windows": 16,
                "fk_position_error_m_p95": 0.01,
                "fk_position_error_m_max": 0.02,
                "stage_repair": {"pass_rate": 1.0},
                "temporal_repair": {"pass_rate": 1.0},
                "clean_reference_fidelity": {"pass_rate": 1.0},
                "clean_physical_non_regression": {"pass_rate": 0.0},
                "clean_input_identity": {"pass_rate": 1.0},
                "final_generation_gate_diagnostic": {
                    "prediction": {"pass_rate": 0.0},
                },
            },
        }
        decision = _checkpoint_validation_decision(metrics, cfg, stage="refiner")

        self.assertTrue(decision["scientific_acceptance"])
        self.assertTrue(decision["publish_allowed"])
        self.assertFalse(decision["final_generation_gate_used"])
        self.assertTrue(decision["clean_non_regression_diagnostic_only"])

        metrics["physical_quality"]["stage_repair"]["pass_rate"] = 0.0
        rejected = _checkpoint_validation_decision(metrics, cfg, stage="refiner")
        self.assertFalse(rejected["scientific_acceptance"])
        self.assertFalse(rejected["publish_allowed"])

        metrics["physical_quality"]["stage_repair"]["pass_rate"] = 1.0
        metrics["physical_quality"]["temporal_repair"]["pass_rate"] = 0.0
        rejected_temporal = _checkpoint_validation_decision(
            metrics, cfg, stage="refiner"
        )
        self.assertIn(
            "temporal_repair_rate_too_low",
            rejected_temporal["reasons"],
        )

        metrics["physical_quality"]["temporal_repair"]["pass_rate"] = 1.0
        metrics["physical_quality"]["clean_input_identity"]["pass_rate"] = 0.0
        rejected_identity = _checkpoint_validation_decision(
            metrics, cfg, stage="refiner"
        )
        self.assertIn(
            "clean_identity_rate_too_low", rejected_identity["reasons"]
        )

    @unittest.skipIf(motion_runtime.torch is None, "PyTorch unavailable")
    def test_rejected_checkpoint_is_quarantined_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "boundary_refiner.pt"
            saved, published = motion_runtime._save_checkpoint_after_validation(
                {"version": "test_candidate"},
                requested,
                {"publish_allowed": False},
            )

            self.assertFalse(published)
            self.assertFalse(requested.exists())
            self.assertEqual(saved.name, "boundary_refiner.rejected_validation.pt")
            self.assertTrue(saved.is_file())

if __name__ == "__main__":
    unittest.main()
