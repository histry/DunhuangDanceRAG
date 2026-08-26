import tempfile
from pathlib import Path

import numpy as np
import pytest

from training import motion_models as models


pytestmark = pytest.mark.skipif(
    models.torch is None,
    reason="PyTorch unavailable",
)


def _identity_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 7:151] = np.tile(models.identity6d_np(), 24)
    motion[:, 4] = np.linspace(0.0, 0.12, frames, dtype=np.float32)
    return motion


def _config(
    batch_size: int,
    device: str = "cpu",
) -> models.MotionGenerationConfig:
    cfg = models.MotionGenerationConfig(device=device)
    cfg.window_len = 30
    cfg.hop_len = 15
    cfg.inference_window_batch_size = batch_size
    cfg.diffusion_steps = 2
    return cfg


def test_refiner_window_batching_matches_single_window_execution():
    torch = models.torch
    cfg = _config(1)
    motion = _identity_motion(75)
    condition = np.zeros((len(motion), 32), dtype=np.float32)
    seam = np.zeros((len(motion), 1), dtype=np.float32)
    seam[24:40] = 1.0
    torch.manual_seed(123)
    model = models.ProductManifoldTemporalRefiner(151, 32)
    models._INFERENCE_MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "refiner.pt"
        torch.save(
            {
                "version": "product_manifold_boundary_refiner_v4",
                "state_dict": model.state_dict(),
                "motion_contract": models.motion_checkpoint_contract(
                    cfg,
                    "boundary_refiner",
                ),
            },
            checkpoint_path,
        )
        single = models._stage_guard_orig_apply_refiner_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            cfg,
        )
        batched_cfg = _config(4)
        batched = models._stage_guard_orig_apply_refiner_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            batched_cfg,
        )

    assert np.allclose(single, batched, atol=3.0e-6)
    assert len(models._INFERENCE_MODEL_CACHE) == 1


@pytest.mark.parametrize(
    "legacy_version",
    [
        "product_manifold_boundary_refiner_v1",
        "product_manifold_boundary_refiner_v2",
        "product_manifold_boundary_refiner_v3",
    ],
)
def test_formal_refiner_rejects_legacy_checkpoint(legacy_version):
    torch = models.torch
    cfg = _config(1)
    motion = _identity_motion(30)
    condition = np.zeros((len(motion), 32), dtype=np.float32)
    seam = np.ones((len(motion), 1), dtype=np.float32)
    model = models.ProductManifoldTemporalRefiner(151, 32)
    models._INFERENCE_MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "legacy_refiner.pt"
        torch.save(
            {
                "version": legacy_version,
                "state_dict": model.state_dict(),
                "motion_contract": models.motion_checkpoint_contract(
                    cfg,
                    "boundary_refiner",
                ),
            },
            checkpoint_path,
        )
        with pytest.raises(RuntimeError, match="non-product refiner"):
            models._stage_guard_orig_apply_refiner_model(
                motion,
                condition,
                seam,
                str(checkpoint_path),
                cfg,
            )


def test_diffusion_window_batching_preserves_seeded_result():
    torch = models.torch
    cfg = _config(1)
    motion = _identity_motion(75)
    condition = np.zeros((len(motion), 32), dtype=np.float32)
    seam = np.zeros((len(motion), 1), dtype=np.float32)
    seam[24:40] = 1.0
    torch.manual_seed(321)
    model = models.TangentDiffusionDenoiser(models.PRODUCT_STATE_DIM, 32)
    models._INFERENCE_MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "diffusion.pt"
        torch.save(
            {
                "version": "reference_tangent_motion_diffusion_v2",
                "diffusion_steps": 2,
                "state_dict": model.state_dict(),
                "motion_contract": models.motion_checkpoint_contract(
                    cfg,
                    "motion_diffusion",
                ),
            },
            checkpoint_path,
        )
        models._cached_inference_model(
            "motion_diffusion",
            checkpoint_path,
            cfg,
        )
        torch.manual_seed(999)
        single = models._stage_guard_orig_apply_diffusion_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            cfg,
        )
        batched_cfg = _config(4)
        torch.manual_seed(999)
        batched = models._stage_guard_orig_apply_diffusion_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            batched_cfg,
        )

    assert np.allclose(single, batched, atol=5.0e-5)
    assert len(models._INFERENCE_MODEL_CACHE) == 1


def test_cuda_diffusion_gpu_preprocessing_preserves_seeded_batching():
    torch = models.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = _config(1, device="cuda")
    motion = _identity_motion(75)
    condition = np.zeros((len(motion), 32), dtype=np.float32)
    seam = np.zeros((len(motion), 1), dtype=np.float32)
    seam[24:40] = 1.0
    torch.manual_seed(654)
    model = models.TangentDiffusionDenoiser(models.PRODUCT_STATE_DIM, 32)
    models._INFERENCE_MODEL_CACHE.clear()

    with tempfile.TemporaryDirectory() as td:
        checkpoint_path = Path(td) / "diffusion_cuda.pt"
        torch.save(
            {
                "version": "reference_tangent_motion_diffusion_v2",
                "diffusion_steps": 2,
                "state_dict": model.state_dict(),
                "motion_contract": models.motion_checkpoint_contract(
                    cfg,
                    "motion_diffusion",
                ),
            },
            checkpoint_path,
        )
        models._cached_inference_model(
            "motion_diffusion",
            checkpoint_path,
            cfg,
        )
        torch.manual_seed(777)
        single = models._stage_guard_orig_apply_diffusion_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            cfg,
        )
        batched_cfg = _config(4, device="cuda")
        torch.manual_seed(777)
        batched = models._stage_guard_orig_apply_diffusion_model(
            motion,
            condition,
            seam,
            str(checkpoint_path),
            batched_cfg,
        )

    assert np.allclose(single, batched, atol=8.0e-5)
    assert len(models._INFERENCE_MODEL_CACHE) == 1


def test_guarded_transaction_defers_bounded_kbo_to_outer_gate(monkeypatch):
    cfg = _config(1)
    reference = _identity_motion(30)
    candidate = reference.copy()
    candidate[:, models.ROOT_X_IDX] += 0.01
    seam = np.ones((len(reference), 1), dtype=np.float32)
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs.get("stage"))
        return False, ["synthetic_reject"], {}

    monkeypatch.setattr(models, "_kinematic_barrier_oracle", reject)
    monkeypatch.setattr(
        models,
        "_apply_guarded_stage_prior",
        lambda motion, *args, **kwargs: (motion, {}),
    )

    deferred = models._bounded_residual_update(
        candidate,
        reference,
        seam,
        cfg,
        stage="refiner",
        validate_barrier=False,
    )
    assert calls == []
    assert not np.array_equal(deferred, reference)

    guarded = models._bounded_residual_update(
        candidate,
        reference,
        seam,
        cfg,
        stage="refiner",
    )
    assert calls == ["refiner_bounded_residual"]
    assert np.array_equal(guarded, reference)
