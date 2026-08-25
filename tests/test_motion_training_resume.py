import json
import random
from argparse import Namespace
from unittest import mock

import numpy as np
import pytest

from support.event_identity import make_event_db_contract
from training import motion_models as models
from motion_geometry.smpl24 import skeleton_contract


pytestmark = pytest.mark.skipif(
    models.torch is None,
    reason="PyTorch unavailable",
)


def _database_contract(label: str):
    return {
        "event_db_contract": make_event_db_contract(
            [f"{label}_event_0", f"{label}_event_1"]
        )
    }


def _write_training_db(root, label: str):
    paths = []
    for index in range(2):
        motion = np.zeros((16, 151), dtype=np.float32)
        motion[:, 7:151] = np.tile(models.identity6d_np(), 24)
        motion[:, models.ROOT_X_IDX] = np.linspace(
            0.0,
            0.02 + index * 0.01,
            len(motion),
            dtype=np.float32,
        )
        motion_path = root / f"{label}_{index}.npy"
        np.save(motion_path, motion)
        paths.append(str(motion_path))
    desc = np.stack(
        [
            np.linspace(index, index + 1.0, 32, dtype=np.float32)
            for index in range(2)
        ]
    )
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1.0e-6
    event_uids = np.asarray(
        [f"{label}_event_{index}" for index in range(2)], dtype=object
    )
    db_path = root / f"{label}.npz"
    np.savez_compressed(
        db_path,
        paths=np.asarray(paths, dtype=object),
        desc=desc,
        desc_z=((desc - mean) / std).astype(np.float32),
        desc_mean=mean.astype(np.float32),
        desc_std=std.astype(np.float32),
        canonical_fps=np.full(2, 30.0, dtype=np.float32),
        skeleton_contract_json=np.asarray(
            json.dumps(skeleton_contract(), sort_keys=True), dtype=object
        ),
        event_uids=event_uids,
        event_db_contract_json=np.asarray(
            json.dumps(make_event_db_contract(event_uids), sort_keys=True),
            dtype=object,
        ),
        source_uids=np.asarray(
            [f"{label}_source_{index}" for index in range(2)], dtype=object
        ),
    )
    return db_path


@pytest.mark.parametrize(
    ("stage", "diffusion_steps"),
    [("refiner", None), ("diffusion", 50)],
)
def test_training_snapshot_roundtrip_restores_model_optimizer_and_rng(
    tmp_path,
    stage,
    diffusion_steps,
):
    torch = models.torch
    cfg = models.MotionGenerationConfig(device="cpu")
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss = model(torch.ones((2, 4))).square().mean()
    loss.backward()
    optimizer.step()
    expected_parameters = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    training_contract = _database_contract("train")
    validation_contract = _database_contract("validation")
    path = tmp_path / f"{stage}.training_snapshot.pt"

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    payload = models._save_training_resume_snapshot(
        path,
        stage=stage,
        model=model,
        optimizer=optimizer,
        completed_steps=3,
        target_steps=10,
        elapsed_seconds=12.5,
        cfg=cfg,
        training_contract=training_contract,
        validation_contract=validation_contract,
        diffusion_steps=diffusion_steps,
    )
    expected_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    random.seed(201)
    np.random.seed(202)
    torch.manual_seed(203)

    completed, elapsed, resume = models._load_training_resume_snapshot(
        path,
        stage=stage,
        model=model,
        optimizer=optimizer,
        target_steps=10,
        cfg=cfg,
        training_contract=training_contract,
        validation_contract=validation_contract,
        device="cpu",
        diffusion_steps=diffusion_steps,
    )
    actual_rng = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
    )

    assert payload["formal_checkpoint"] is False
    assert payload["publication_state"] == "resume_only_not_validated"
    assert payload["checkpoint_decision"]["publish_allowed"] is False
    assert "state_dict" not in payload
    inference_role = (
        "boundary_refiner" if stage == "refiner" else "motion_diffusion"
    )
    with pytest.raises(RuntimeError, match="Formal generation rejects"):
        models._cached_inference_model(inference_role, path, cfg)
    assert completed == 3
    assert elapsed == pytest.approx(12.5)
    assert resume["resumed"] is True
    assert actual_rng == pytest.approx(expected_rng)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[key])


def test_training_snapshot_rejects_event_database_mismatch(tmp_path):
    torch = models.torch
    cfg = models.MotionGenerationConfig(device="cpu")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    snapshot_path = tmp_path / "refiner.training_snapshot.pt"
    training_contract = _database_contract("train")
    validation_contract = _database_contract("validation")
    models._save_training_resume_snapshot(
        snapshot_path,
        stage="refiner",
        model=model,
        optimizer=optimizer,
        completed_steps=1,
        target_steps=4,
        elapsed_seconds=1.0,
        cfg=cfg,
        training_contract=training_contract,
        validation_contract=validation_contract,
    )

    with pytest.raises(RuntimeError, match="event DB contract mismatch"):
        models._load_training_resume_snapshot(
            snapshot_path,
            stage="refiner",
            model=model,
            optimizer=optimizer,
            target_steps=4,
            cfg=cfg,
            training_contract=_database_contract("different_train"),
            validation_contract=validation_contract,
            device="cpu",
        )


def test_snapshot_options_fail_closed_instead_of_overwriting(tmp_path):
    cfg = models.MotionGenerationConfig(device="cpu")
    checkpoint = tmp_path / "refiner.pt"
    snapshot = tmp_path / "refiner.training_snapshot.pt"
    snapshot.write_bytes(b"existing recovery state")
    args = Namespace(
        snapshot_path=str(snapshot),
        resume_snapshot=None,
        snapshot_every=20,
    )

    with pytest.raises(RuntimeError, match="already exists"):
        models._resolve_training_snapshot_options(args, cfg, checkpoint)


def test_motion_training_cli_exposes_resume_only_snapshot_contract():
    for command in ("train-refiner", "train-diffusion"):
        argv = [
            command,
            "--db",
            "train.npz",
            "--val_db",
            "val.npz",
            "--out",
            "model.pt",
            "--snapshot_path",
            "model.resume.pt",
            "--resume_snapshot",
            "model.resume.pt",
            "--snapshot_every",
            "25",
        ]
        parsed = models.parse_args(argv)
        assert parsed.snapshot_path == "model.resume.pt"
        assert parsed.resume_snapshot == "model.resume.pt"
        assert parsed.snapshot_every == 25


@pytest.mark.parametrize("stage", ["refiner", "diffusion"])
def test_formal_training_entrypoint_resumes_and_preserves_recovery_until_pipeline_completion(
    tmp_path,
    stage,
):
    torch = models.torch
    train_path = _write_training_db(tmp_path, "train")
    validation_path = _write_training_db(tmp_path, "validation")
    config_path = tmp_path / "motion_config.json"
    config_path.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "window_len": 16,
                "hop_len": 8,
                "batch_size": 2,
                "device": "cpu",
                "gpu_preprocessing": False,
                "checkpoint_validation_fail_closed": False,
                "training_snapshot_interval_steps": 1,
                "diffusion_steps": 2,
            }
        ),
        encoding="utf-8",
    )
    cfg = models.MotionGenerationConfig.from_json(config_path).apply_env()
    train_db = models.load_db(train_path)
    validation_db = models.load_db(validation_path)
    train_contract = models._training_db_contract(train_db, cfg, "test train")
    validation_contract = models._training_db_contract(
        validation_db, cfg, "test validation"
    )
    if stage == "refiner":
        model = models.ProductManifoldTemporalRefiner(151, 32)
        diffusion_steps = None
    else:
        model = models.TangentDiffusionDenoiser(models.PRODUCT_STATE_DIM, 32)
        diffusion_steps = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    snapshot_path = tmp_path / f"{stage}.resume.pt"
    output_path = tmp_path / f"{stage}.formal.pt"
    models._save_training_resume_snapshot(
        snapshot_path,
        stage=stage,
        model=model,
        optimizer=optimizer,
        completed_steps=1,
        target_steps=2,
        elapsed_seconds=0.5,
        cfg=cfg,
        training_contract=train_contract,
        validation_contract=validation_contract,
        diffusion_steps=diffusion_steps,
    )
    args = Namespace(
        config=str(config_path),
        db=str(train_path),
        val_db=str(validation_path),
        out=str(output_path),
        steps=2,
        diffusion_steps=diffusion_steps,
        snapshot_path=str(snapshot_path),
        resume_snapshot=str(snapshot_path),
        snapshot_every=1,
    )
    accepted = {
        "scientific_acceptance": True,
        "publish_allowed": True,
        "fail_closed_enforced": True,
        "reasons": [],
    }
    evaluator = (
        "_evaluate_refiner_validation"
        if stage == "refiner"
        else "_evaluate_diffusion_validation"
    )
    trainer = models.train_refiner if stage == "refiner" else models.train_diffusion
    with mock.patch.object(models, evaluator, return_value={"test": True}), mock.patch.object(
        models,
        "_checkpoint_validation_decision",
        return_value=accepted,
    ):
        assert trainer(args) == 0

    assert output_path.is_file()
    assert snapshot_path.is_file()
    snapshot = models._trusted_torch_load(snapshot_path, map_location="cpu")
    assert snapshot["formal_checkpoint"] is False
    assert snapshot["completed_steps"] == 2
    checkpoint = models._trusted_torch_load(output_path, map_location="cpu")
    assert checkpoint["training_resume"]["resumed"] is True
    assert checkpoint["training_resume"]["completed_steps"] == 1
    assert checkpoint["training_snapshot"]["resume_only"] is True
