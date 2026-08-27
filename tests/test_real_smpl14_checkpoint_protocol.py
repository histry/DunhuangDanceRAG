import random
from pathlib import Path

import numpy as np
import pytest

from motion_geometry.product_manifold import product_exp_np, product_log_np

from retargeting.smpl_adapter import (
    CHANG_E_POSE_LAYOUT,
    load_smpl24_parameters,
)
from training.motion_models import (
    MotionGenerationConfig,
    _checkpoint_validation_decision,
    _new_validation_physical_accumulator,
    _record_validation_physical_prediction,
    _summarize_validation_physical_metrics,
    degrade_for_refiner,
)
from training import motion_models as models


ROOT = Path(__file__).resolve().parents[1]
SMPL14 = ROOT / "assets" / "motion" / "smpl_official_14"


def _real_smpl14_files():
    paths = sorted(SMPL14.glob("*.npz"))
    if not paths:
        pytest.skip("real SMPL14 assets are not installed in this checkout")
    assert len(paths) == 14, f"partial SMPL14 installation: {len(paths)}/14"
    return paths


def _central_real_window(source: Path, target: Path, frames: int):
    with np.load(source, allow_pickle=True) as data:
        poses = np.asarray(data["poses"])
        trans = np.asarray(data["trans"])
        start = max(0, (len(poses) - frames) // 2)
        stop = min(len(poses), start + frames)
        payload = {
            "poses": poses[start:stop],
            "trans": trans[start:stop],
            "mocap_framerate": np.asarray(
                data.get("mocap_framerate", 30.0)
            ),
        }
        if "betas" in data:
            payload["betas"] = np.asarray(data["betas"])
    np.savez(target, **payload)
    motion, _ = load_smpl24_parameters(
        target,
        target_fps=30.0,
        pose_layout=CHANG_E_POSE_LAYOUT,
    )
    assert len(motion) == stop - start
    return motion


def test_real_smpl14_clean_degradation_and_checkpoint_rejection(tmp_path):
    """Exercise the formal checkpoint protocol on all installed SMPL14 files."""

    cfg = MotionGenerationConfig.from_json(ROOT / "configs" / "motion_model.json")
    ideal = _new_validation_physical_accumulator()
    no_repair = _new_validation_physical_accumulator()
    exact_product_repair = _new_validation_physical_accumulator()
    outside_seam_changes = []

    for index, source in enumerate(_real_smpl14_files()):
        clean = _central_real_window(
            source,
            tmp_path / source.name,
            int(cfg.window_len),
        )
        random.seed(20260824 + index)
        np.random.seed(20260824 + index)
        degraded, seam = degrade_for_refiner(clean, cfg=cfg)

        _record_validation_physical_prediction(
            ideal,
            clean,
            clean,
            cfg,
            degraded=degraded,
            seam_mask=seam,
        )
        _record_validation_physical_prediction(
            no_repair,
            degraded,
            clean,
            cfg,
            degraded=degraded,
            seam_mask=seam,
        )
        reconstructed = product_exp_np(
            degraded,
            product_log_np(degraded, clean),
        )
        _record_validation_physical_prediction(
            exact_product_repair,
            reconstructed,
            clean,
            cfg,
            degraded=degraded,
            seam_mask=seam,
        )
        inactive = seam[:, 0] == 0.0
        outside_seam_changes.append(
            float(np.max(np.abs((degraded - clean)[inactive, 4:])))
        )

    ideal_summary = _summarize_validation_physical_metrics(ideal)
    no_repair_summary = _summarize_validation_physical_metrics(no_repair)
    exact_product_summary = _summarize_validation_physical_metrics(
        exact_product_repair
    )

    assert ideal_summary["stage_repair"]["pass_rate"] == 1.0
    assert ideal_summary["temporal_repair"]["pass_rate"] == 1.0
    assert ideal_summary["clean_reference_fidelity"]["pass_rate"] == 1.0
    assert (
        ideal_summary["clean_physical_non_regression"]["pass_rate"] == 1.0
    )
    assert no_repair_summary["stage_repair"]["pass_rate"] == 0.0
    assert no_repair_summary["temporal_repair"]["pass_rate"] == 0.0
    assert exact_product_summary["stage_repair"]["pass_rate"] == 1.0
    assert exact_product_summary["temporal_repair"]["pass_rate"] == 1.0
    assert (
        exact_product_summary["clean_reference_fidelity"]["pass_rate"]
        == 1.0
    )
    assert max(outside_seam_changes) < 2.0e-6
    assert not ideal_summary["final_generation_gate_diagnostic"][
        "checkpoint_criterion"
    ]

    rejected = _checkpoint_validation_decision(
        {
            "reconstruction_product_log_l1": 0.0,
            "physical_quality": no_repair_summary,
        },
        cfg,
        stage="refiner",
    )
    assert rejected["scientific_acceptance"] is False
    assert rejected["publish_allowed"] is False
    assert "stage_repair_rate_too_low" in rejected["reasons"]

    # The strict multi-metric clean comparison is retained as a diagnostic,
    # but product-manifold round trips must not be rejected after the bounded
    # clean-reference fidelity gate and the real clean-input identity gate pass.
    exact_product_summary["clean_physical_non_regression"]["pass_rate"] = 0.0
    exact_product_summary["clean_input_identity"] = {"pass_rate": 1.0}
    accepted = _checkpoint_validation_decision(
        {
            "reconstruction_product_log_l1": 0.0,
            "physical_quality": exact_product_summary,
        },
        cfg,
        stage="refiner",
    )
    assert accepted["scientific_acceptance"] is True
    assert accepted["publish_allowed"] is True


def test_real_smpl14_zero_refiner_preserves_authentic_clean_motion(tmp_path):
    if models.torch is None:
        pytest.skip("PyTorch unavailable")
    cfg = MotionGenerationConfig(device="cpu")
    accumulator = _new_validation_physical_accumulator()
    for source in _real_smpl14_files():
        clean = _central_real_window(source, tmp_path / source.name, 120)
        # Match load_motion_window(): the adapter's provisional contacts are
        # re-derived under the training configuration before the model sees it.
        clean, _ = models.enforce_edge151_contract_np(
            clean, cfg, derive_contact=True, project_rot=True,
        )
        tensor = models.torch.from_numpy(clean[None])
        prediction = models._decode_product_refiner_output(
            tensor, tensor.new_zeros((1, len(clean), 79)),
            tensor.new_ones((1, len(clean), 24)),
            tensor.new_ones((1, len(clean), 1)),
            tensor.new_ones((1, len(clean), 1)), cfg,
        )
        models._record_validation_clean_identity_prediction(
            accumulator, prediction[0].numpy(), clean, cfg,
        )
    gates = accumulator["clean_identity_gates"]
    assert len(gates) == 14
    assert all(gate["accepted"] for gate in gates), [gate["reasons"] for gate in gates]
