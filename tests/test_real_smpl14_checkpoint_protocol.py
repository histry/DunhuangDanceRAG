import random
from pathlib import Path

import numpy as np
import pytest

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
        )
        _record_validation_physical_prediction(
            no_repair,
            degraded,
            clean,
            cfg,
            degraded=degraded,
        )
        inactive = seam[:, 0] == 0.0
        outside_seam_changes.append(
            float(np.max(np.abs((degraded - clean)[inactive, 4:])))
        )

    ideal_summary = _summarize_validation_physical_metrics(ideal)
    no_repair_summary = _summarize_validation_physical_metrics(no_repair)

    assert ideal_summary["stage_repair"]["pass_rate"] == 1.0
    assert ideal_summary["clean_reference_fidelity"]["pass_rate"] == 1.0
    assert (
        ideal_summary["clean_physical_non_regression"]["pass_rate"] == 1.0
    )
    assert no_repair_summary["stage_repair"]["pass_rate"] == 0.0
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
