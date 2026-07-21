import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
    V46Config,
    _descriptor_values_in_training_coordinates,
    _training_db_contract,
    _validate_source_disjoint,
    assert_motion_checkpoint_contract,
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
    def test_training_db_rejects_false_fps_contract(self):
        cfg = V46Config()
        cfg.fps = 60.0
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _training_db_contract(_database(fps=30.0), cfg, "test")

    def test_motion_config_uses_normalized_pipeline_fps(self):
        with mock.patch.dict(os.environ, {"V46_FPS": "60"}, clear=False):
            cfg = V46Config().apply_env()
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
        cfg = V46Config()
        cfg._event_db_contract = make_event_db_contract(["evt_a"])
        checkpoint = {
            "motion_contract": motion_checkpoint_contract(cfg, "v45_refiner"),
            "training_event_db_contract": make_event_db_contract(["evt_b"]),
        }
        with self.assertRaisesRegex(RuntimeError, "checkpoint/Generation"):
            assert_motion_checkpoint_contract(
                checkpoint, cfg, "refiner.pt", "v45_refiner"
            )

    def test_v45_v46_cli_accepts_source_disjoint_validation_db(self):
        refiner = parse_args([
            "train-refiner", "--db", "train.npz", "--val_db", "val.npz", "--out", "out.pt"
        ])
        diffusion = parse_args([
            "train-diffusion", "--db", "train.npz", "--val_db", "val.npz", "--out", "out.pt"
        ])
        self.assertEqual(refiner.val_db, "val.npz")
        self.assertEqual(diffusion.val_db, "val.npz")


if __name__ == "__main__":
    unittest.main()
