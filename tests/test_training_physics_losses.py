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
