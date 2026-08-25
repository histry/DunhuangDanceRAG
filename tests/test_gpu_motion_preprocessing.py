import inspect
import json
import os
import random
from argparse import Namespace
from unittest import mock

import numpy as np
import pytest

from training import motion_models as models
from motion_geometry.smpl24 import skeleton_contract
from support.event_identity import make_event_db_contract


pytestmark = pytest.mark.skipif(
    models.torch is None,
    reason="PyTorch unavailable",
)


def _risk_fixture(batch: int = 6):
    cfg = models.MotionGenerationConfig(device="cpu")
    cfg.window_len = 120
    base = np.zeros((cfg.window_len, 151), dtype=np.float32)
    base[:, 7:151] = np.tile(models.identity6d_np(), 24)[None]
    base[:, models.ROOT_X_IDX] = np.linspace(
        0.0,
        0.15,
        cfg.window_len,
        dtype=np.float32,
    )
    random.seed(20260824)
    np.random.seed(20260824)
    rows = [models.degrade_for_refiner(base, cfg=cfg) for _ in range(batch)]
    motion = np.stack([value for value, _ in rows])
    seam = np.stack([value for _, value in rows])
    return cfg, motion, seam


def _write_training_db(root, label: str, source_prefix: str):
    motions = []
    for index in range(2):
        motion = np.zeros((36, 151), dtype=np.float32)
        motion[:, 7:151] = np.tile(models.identity6d_np(), 24)
        motion[:, models.ROOT_X_IDX] = np.linspace(
            0.0,
            0.03 + 0.01 * index,
            len(motion),
            dtype=np.float32,
        )
        path = root / f"{label}_{index}.npy"
        np.save(path, motion)
        motions.append(str(path))
    desc = np.stack(
        [np.linspace(index, index + 1.0, 32, dtype=np.float32) for index in range(2)]
    )
    mean = desc.mean(axis=0, keepdims=True)
    std = desc.std(axis=0, keepdims=True) + 1.0e-6
    event_uids = np.asarray([f"{label}_event_{index}" for index in range(2)], dtype=object)
    source_uids = np.asarray(
        [f"{source_prefix}_{index}" for index in range(2)], dtype=object
    )
    db_path = root / f"{label}.npz"
    np.savez_compressed(
        db_path,
        paths=np.asarray(motions, dtype=object),
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
        source_uids=source_uids,
        recording_uids=source_uids,
    )
    return db_path


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_risk_masks_match_numpy_contract(device):
    torch = models.torch
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg, motion, seam = _risk_fixture()
    with mock.patch.dict(
        os.environ,
        {
            "PHYSICAL_PEAK_JERK_MASK_ENABLE": "1",
            "PHYSICAL_PEAK_JERK_THRESHOLD_MPS3": "1400",
            "PHYSICAL_PEAK_JERK_PERCENTILE": "99.5",
            "PHYSICAL_PEAK_JERK_RADIUS_FRAMES": "4",
            "PHYSICAL_PEAK_JERK_PARENT_DEPTH": "2",
        },
        clear=False,
    ):
        expected = models._risk_masks_for_batch_np(motion, seam, cfg)
        actual = tuple(
            value.detach().cpu().numpy()
            for value in models._risk_masks_for_batch_torch(
                torch.from_numpy(motion).to(device),
                torch.from_numpy(seam).to(device),
                cfg,
            )
        )

    for cpu_value, torch_value in zip(expected, actual):
        np.testing.assert_allclose(cpu_value, torch_value, atol=2.0e-5)
        np.testing.assert_array_equal(cpu_value > 0.0, torch_value > 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_batch_contract_matches_numpy_training_contract(device):
    torch = models.torch
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg, clean_batch, _ = _risk_fixture(batch=4)
    random.seed(20260825)
    np.random.seed(20260825)
    raw_batch = np.stack(
        [
            models.degrade_for_refiner(
                clean,
                cfg=cfg,
                finalize_contract=False,
            )[0]
            for clean in clean_batch
        ]
    )
    expected = np.stack(
        [
            models.enforce_edge151_contract_np(
                motion,
                cfg,
                derive_contact=True,
                project_rot=True,
            )[0]
            for motion in raw_batch
        ]
    )
    actual = (
        models._enforce_internal_batch_contract_torch(
            torch.from_numpy(raw_batch).to(device),
            cfg,
        )
        .cpu()
        .numpy()
    )

    np.testing.assert_allclose(expected, actual, atol=2.0e-6)
    np.testing.assert_array_equal(expected[..., :4], actual[..., :4])


def test_gpu_preprocessing_is_explicit_and_overridable():
    cfg = models.MotionGenerationConfig.from_json("configs/motion_model.json")
    assert cfg.gpu_preprocessing is True
    assert cfg.diffusion_noise_batch_max_mib == 256.0
    with mock.patch.dict(
        os.environ,
        {"MOTION_GPU_PREPROCESSING": "0"},
        clear=False,
    ):
        cfg.apply_env()
    assert cfg.gpu_preprocessing is False


def test_cuda_device_resolution_handles_indexed_and_invalid_devices():
    # ``apply_env`` intentionally gives MOTION_DEVICE precedence over the
    # dataclass constructor.  Pin each scenario here so an operator's exported
    # MOTION_DEVICE (for example the formal ``cuda`` setting) cannot silently
    # change what this test exercises.
    with mock.patch.dict(
        os.environ,
        {"MOTION_DEVICE": "cuda:0"},
        clear=False,
    ), mock.patch.object(models.torch.cuda, "is_available", return_value=False):
        cfg = models.MotionGenerationConfig().apply_env()
    assert cfg.device == "cpu"

    with mock.patch.dict(
        os.environ,
        {"MOTION_DEVICE": "cuda:3"},
        clear=False,
    ), mock.patch.object(
        models.torch.cuda,
        "is_available",
        return_value=True,
    ), mock.patch.object(models.torch.cuda, "device_count", return_value=1):
        with pytest.raises(ValueError, match="only 1 CUDA device"):
            models.MotionGenerationConfig().apply_env()

    with mock.patch.dict(
        os.environ,
        {"MOTION_DEVICE": "not-a-device"},
        clear=False,
    ):
        with pytest.raises(ValueError, match="Invalid MOTION_DEVICE"):
            models.MotionGenerationConfig().apply_env()


def test_ik_best_state_has_no_per_iteration_host_sync():
    source = inspect.getsource(models._stage_guard_orig_true_lower_body_ik)
    loop_start = source.index("for it in range(int(cfg.ik_iters)):")
    loop_end = source.index(
        "best_loss = float(best_loss_device.cpu())",
        loop_start,
    )
    loop_source = source[loop_start:loop_end]

    assert ".cpu()" not in loop_source
    assert "best_motion_device = torch.where" in loop_source


def test_diffusion_noise_is_bounded_to_current_window_batch():
    source = inspect.getsource(
        models._stage_guard_orig_apply_diffusion_model
    )
    assert "noise_bank" not in source
    assert "_diffusion_noise_batch_torch" in source
    assert "noise_window_cap" in source


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seeded_diffusion_noise_is_batch_partition_invariant(device):
    torch = models.torch
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    kwargs = {
        "base_seed": 20260824,
        "steps": 3,
        "window": 12,
        "dtype": torch.float32,
        "device": device,
    }
    together = models._diffusion_noise_batch_torch(range(5), **kwargs)
    partitioned = torch.cat(
        [
            models._diffusion_noise_batch_torch(range(2), **kwargs),
            models._diffusion_noise_batch_torch(range(2, 5), **kwargs),
        ],
        dim=0,
    )
    assert torch.equal(together, partitioned)
    assert together.device.type == device


def test_diffusion_noise_memory_cap_is_enforced_and_validated():
    torch = models.torch
    cfg = models.MotionGenerationConfig(diffusion_noise_batch_max_mib=4.0)
    cap = models._diffusion_noise_window_cap(
        cfg,
        steps=50,
        window=120,
        dtype=torch.float32,
    )
    bytes_per_window = 50 * 120 * models.PRODUCT_STATE_DIM * 4
    assert cap == (4 * 1024 ** 2) // bytes_per_window
    cfg.diffusion_noise_batch_max_mib = 1.0
    with pytest.raises(ValueError, match="smaller than one diffusion window"):
        models._diffusion_noise_window_cap(
            cfg,
            steps=50,
            window=120,
            dtype=torch.float32,
        )
    cfg.diffusion_noise_batch_max_mib = 0.0
    with pytest.raises(ValueError, match="positive finite"):
        models._diffusion_noise_window_cap(
            cfg,
            steps=50,
            window=120,
            dtype=torch.float32,
        )
    cfg.diffusion_noise_batch_max_mib = 4.0
    with pytest.raises(ValueError, match="positive steps and window"):
        models._diffusion_noise_window_cap(
            cfg,
            steps=0,
            window=120,
            dtype=torch.float32,
        )


def test_cuda_refiner_training_step_keeps_masks_and_physics_on_device():
    torch = models.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = models.MotionGenerationConfig(device="cuda")
    cfg.window_len = 30
    clean = np.zeros((2, cfg.window_len, 151), dtype=np.float32)
    clean[..., 7:151] = np.tile(models.identity6d_np(), 24)
    random.seed(20260826)
    np.random.seed(20260826)
    rows = [
        models.degrade_for_refiner(
            item,
            cfg=cfg,
            finalize_contract=False,
        )
        for item in clean
    ]
    bad = torch.from_numpy(np.stack([value for value, _ in rows])).cuda()
    seam = torch.from_numpy(np.stack([value for _, value in rows])).cuda()
    clean_t = torch.from_numpy(clean).cuda()
    bad = models._enforce_internal_batch_contract_torch(bad, cfg)
    joint, root, contact = models._risk_masks_for_batch_torch(
        bad,
        seam,
        cfg,
    )
    model = models.ProductManifoldTemporalRefiner(151, 32).cuda()
    output = model(
        bad,
        torch.zeros((2, 32), device="cuda"),
        seam,
        joint,
    )
    prediction = models._decode_product_refiner_output(
        bad,
        output,
        joint,
        root,
        contact,
        cfg,
    )
    loss, _ = models._product_motion_losses(
        prediction,
        clean_t,
        bad,
        joint,
        root,
        contact,
        cfg,
        seam_mask=seam,
    )
    loss.backward()

    assert loss.is_cuda
    assert joint.is_cuda and root.is_cuda and contact.is_cuda
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cuda_formal_training_entrypoints_complete_one_step(tmp_path):
    torch = models.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    train_db = _write_training_db(tmp_path, "train", "train_source")
    validation_db = _write_training_db(tmp_path, "validation", "val_source")
    config_path = tmp_path / "motion_config.json"
    config_path.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "window_len": 30,
                "hop_len": 15,
                "batch_size": 2,
                "device": "cuda",
                "gpu_preprocessing": True,
                "refiner_train_steps": 1,
                "diffusion_train_steps": 1,
                "diffusion_steps": 2,
                # This smoke test proves CUDA reachability, not scientific
                # convergence after a single optimizer step.
                "checkpoint_validation_fail_closed": False,
            }
        ),
        encoding="utf-8",
    )
    refiner_path = tmp_path / "refiner.pt"
    diffusion_path = tmp_path / "diffusion.pt"

    # Formal operators commonly export fail-closed=1 before running tests.
    # Isolate this one-step CUDA reachability smoke test from that ambient
    # policy: one optimizer step is intentionally not a convergence claim.
    with mock.patch.dict(
        os.environ,
        {"MOTION_CHECKPOINT_VALIDATION_FAIL_CLOSED": "0"},
        clear=False,
    ):
        assert models.train_refiner(
            Namespace(
                config=str(config_path),
                db=str(train_db),
                val_db=str(validation_db),
                out=str(refiner_path),
                steps=1,
            )
        ) == 0
        assert models.train_diffusion(
            Namespace(
                config=str(config_path),
                db=str(train_db),
                val_db=str(validation_db),
                out=str(diffusion_path),
                steps=1,
                diffusion_steps=2,
            )
        ) == 0

    refiner = models._trusted_torch_load(refiner_path, map_location="cpu")
    diffusion = models._trusted_torch_load(diffusion_path, map_location="cpu")
    assert refiner["version"] == "product_manifold_boundary_refiner_v2"
    assert diffusion["version"] == "reference_tangent_motion_diffusion_v2"
    assert refiner["validation"]["source_disjoint"]["overlap"] == []
    assert diffusion["validation"]["source_disjoint"]["overlap"] == []
