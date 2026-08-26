import inspect

import numpy as np
import pytest

from motion_geometry.rotations import matrix_to_rot6d_np
from training import motion_models as models


pytestmark = pytest.mark.skipif(models.torch is None, reason="PyTorch unavailable")


def _identity_batch(frames: int = 12):
    motion = np.zeros((1, frames, 151), dtype=np.float32)
    matrices = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, 24, 3, 3)
    )
    motion[0, :, 7:151] = matrix_to_rot6d_np(matrices).reshape(frames, -1)
    motion[0, :, :4] = 1.0
    motion[0, :, 5] = 0.95
    return models.torch.from_numpy(motion)


def test_batch_fk_and_world_physics_losses_have_finite_gradients():
    clean = _identity_batch()
    prediction = clean.clone()
    prediction[:, 6, 4] += 0.10
    prediction.requires_grad_(True)
    cfg = models.MotionGenerationConfig(device="cpu")

    joints = models.fk_24_torch(prediction)
    total, terms = models._world_space_physics_losses(prediction, clean, cfg)
    total.backward()

    assert joints.shape == (1, 12, 24, 3)
    assert terms["fk"].item() > 0.0
    assert terms["acceleration"].item() > 0.0
    assert terms["jerk"].item() > 0.0
    assert prediction.grad is not None
    assert models.torch.isfinite(prediction.grad).all()


def test_world_physics_loss_is_zero_at_clean_target():
    clean = _identity_batch()
    total, terms = models._world_space_physics_losses(
        clean,
        clean,
        models.MotionGenerationConfig(device="cpu"),
    )

    assert total.item() == pytest.approx(0.0, abs=1.0e-8)
    assert all(value.item() == pytest.approx(0.0, abs=1.0e-8) for value in terms.values())


def test_seam_world_temporal_losses_are_local_and_differentiable():
    clean = _identity_batch(frames=16)
    prediction = clean.clone()
    prediction[:, 5, 4] += 0.08
    prediction.requires_grad_(True)
    seam = models.torch.zeros((1, 16, 1), dtype=prediction.dtype)
    seam[:, 5:11] = 1.0
    cfg = models.MotionGenerationConfig(device="cpu")

    total, terms = models._world_space_physics_losses(
        prediction,
        clean,
        cfg,
        seam_mask=seam,
    )
    total.backward()

    assert terms["endpoint_continuity"].item() > 0.0
    assert terms["seam_velocity"].item() > 0.0
    assert terms["seam_acceleration"].item() > 0.0
    assert terms["seam_jerk"].item() > 0.0
    assert prediction.grad is not None
    assert models.torch.isfinite(prediction.grad).all()


def test_refiner_relative_target_does_not_require_exact_clean_interior():
    clean = _identity_batch(frames=16)
    degraded = clean.clone()
    degraded[:, 5:11, 4] += 0.05
    prediction = clean.clone()
    prediction[:, 5:11, 4] += 0.045
    seam = models.torch.zeros((1, 16, 1), dtype=clean.dtype)
    seam[:, 5:11] = 1.0
    joint_mask = models.torch.ones((1, 16, 24), dtype=clean.dtype)
    root_mask = models.torch.ones((1, 16, 1), dtype=clean.dtype)
    contact_mask = models.torch.zeros((1, 16, 4), dtype=clean.dtype)
    cfg = models.MotionGenerationConfig(
        device="cpu",
        product_refiner_training_target_repair_gain=0.10,
    )

    _, terms = models._product_motion_losses(
        prediction,
        clean,
        degraded,
        joint_mask,
        root_mask,
        contact_mask,
        cfg,
        seam_mask=seam,
    )

    assert terms["active_reconstruction"].item() > 0.0
    assert terms["repair_margin"].item() == pytest.approx(0.0, abs=1.0e-7)


def test_temporal_repair_gate_accepts_improvement_and_rejects_noop():
    clean = _identity_batch(frames=24)[0].numpy()
    degraded = clean.copy()
    degraded[7:17, 4] += np.sin(
        np.linspace(0.0, np.pi, 10, dtype=np.float32)
    ) * 0.08
    prediction = clean + 0.5 * (degraded - clean)
    seam = np.zeros((24, 1), dtype=np.float32)
    seam[7:17] = 1.0
    cfg = models.MotionGenerationConfig(device="cpu")

    repaired = models._temporal_repair_gate_np(
        prediction,
        degraded,
        clean,
        seam,
        cfg,
    )
    noop = models._temporal_repair_gate_np(
        degraded,
        degraded,
        clean,
        seam,
        cfg,
    )

    assert repaired["accepted"] is True
    assert repaired["detail"]["jerk_non_regression"] is True
    assert noop["accepted"] is False
    assert "no_meaningful_temporal_repair_gain" in noop["reasons"]


def test_fk_reuses_offsets_without_per_joint_device_synchronization():
    motion = _identity_batch()
    models._FK_OFFSETS_TORCH_CACHE.clear()
    first = models.fk_24_torch(motion)
    assert len(models._FK_OFFSETS_TORCH_CACHE) == 1
    cached_offset_id = id(next(iter(models._FK_OFFSETS_TORCH_CACHE.values())))
    second = models.fk_24_torch(motion)

    assert id(next(iter(models._FK_OFFSETS_TORCH_CACHE.values()))) == cached_offset_id
    assert models.torch.equal(first, second)
    assert ".item()" not in inspect.getsource(models.fk_24_torch)


def test_batched_physics_fk_matches_two_independent_calls():
    clean = _identity_batch()
    prediction = clean.clone()
    prediction[:, 5, 4] += 0.03
    separate_prediction = models.fk_24_torch(prediction)
    separate_clean = models.fk_24_torch(clean)
    combined = models.fk_24_torch(
        models.torch.cat([prediction, clean], dim=0)
    )

    assert models.torch.equal(combined[:1], separate_prediction)
    assert models.torch.equal(combined[1:], separate_clean)
