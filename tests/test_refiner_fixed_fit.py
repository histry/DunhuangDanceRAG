import json
from argparse import Namespace
from unittest import mock

import numpy as np
import pytest

from training import motion_models as m
from training import refiner_diagnostics as d
from tests.test_motion_training_resume import _write_training_db


pytestmark = pytest.mark.skipif(m.torch is None, reason="PyTorch unavailable")


def motion(batch=1, frames=40):
    x = np.zeros((batch, frames, 151), dtype=np.float32)
    x[..., 7:] = np.tile(m.identity6d_np(), 24)
    x[..., 5] = 0.95
    return m.torch.from_numpy(x)


def clean_loss(pred, clean):
    joint = pred.new_ones(pred.shape[:2] + (24,))
    root = pred.new_ones(pred.shape[:2] + (1,))
    return m._product_refiner_clean_identity_loss(pred, clean, joint, root, root, m.MotionGenerationConfig(device="cpu"))


def test_clean_safe_region_keeps_hard_constraints_zero_but_noop_prior_is_active():
    clean = motion()
    pred = clean.clone()
    pred[..., 4] += 0.002
    pred[..., :4] += 0.01
    pred.requires_grad_(True)
    loss, terms = clean_loss(pred, clean)
    assert terms["reconstruction"] > 0
    # Formal safety dead-bands are unchanged, but V12 intentionally adds a
    # small always-on no-op prior for every nonzero unnecessary edit.
    assert terms["geometry_excess"].item() == pytest.approx(0, abs=1e-10)
    assert terms["contact_excess"].item() == pytest.approx(0, abs=1e-10)
    assert terms["noop"] > 0
    assert loss > 0
    loss.backward()
    assert pred.grad.abs().sum() > 0


def test_exact_clean_identity_still_has_zero_loss_and_zero_gradient():
    clean = motion()
    pred = clean.clone().requires_grad_(True)
    loss, terms = clean_loss(pred, clean)
    assert terms["noop"].item() == pytest.approx(0, abs=1e-12)
    assert loss.item() == pytest.approx(0, abs=1e-12)
    loss.backward()
    assert pred.grad.abs().max() == 0


def test_clean_tolerance_is_per_window_not_diluted_by_batch():
    clean = motion(batch=4)
    pred = clean.clone()
    pred[0, :, 4] += 0.6
    pred.requires_grad_(True)
    loss, terms = clean_loss(pred, clean)
    assert terms["geometry_excess"] > 0
    loss.backward()
    assert pred.grad[0, :, 4].abs().sum() > 0
    assert pred.grad[1:].abs().max() == 0


def test_small_geometry_jitter_still_triggers_high_frequency_constraint():
    clean = motion()
    pred = clean.clone()
    pred[..., 4] += m.torch.tensor([0.001, -0.001] * 20)
    pred.requires_grad_(True)
    loss, terms = clean_loss(pred, clean)
    assert terms["geometry_excess"] == 0
    assert terms["jerk_max_excess"] > 0
    loss.backward()
    assert m.torch.isfinite(pred.grad).all()
    assert pred.grad[..., 4].abs().sum() > 0


def test_clean_foot_speed_excess_is_penalized_despite_small_geometry_error():
    clean = motion()
    pred = clean.clone()
    pred[..., 4] += m.torch.linspace(0.0, 0.08, pred.shape[1])
    pred.requires_grad_(True)
    loss, terms = clean_loss(pred, clean)
    assert terms["geometry_excess"] == 0
    assert terms["support_excess"] > 0
    loss.backward()
    assert m.torch.isfinite(pred.grad).all()
    assert pred.grad[..., 4].abs().sum() > 0


def test_masked_speed_quantiles_have_safe_empty_support_gradients():
    speed = m.torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]], requires_grad=True)
    mask = m.torch.tensor([[[True, False], [True, False]], [[False, False], [False, False]]])
    p95, maximum = m._masked_speed_stats_torch(speed, mask)
    assert p95.tolist() == pytest.approx([2.9, 0.0])
    assert maximum.tolist() == pytest.approx([3.0, 0.0])
    (p95.sum() + maximum.sum()).backward()
    assert m.torch.isfinite(speed.grad).all()
    assert speed.grad[1].abs().sum() == 0


def test_masked_speed_statistics_do_not_hide_nonfinite_supported_values():
    speed = m.torch.tensor([[[float("inf"), 0.0]]])
    mask = m.torch.tensor([[[True, False]]])
    p95, maximum = m._masked_speed_stats_torch(speed, mask)
    assert not m.torch.isfinite(p95).all()
    assert not m.torch.isfinite(maximum).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jerk_constraint_matches_reference_budget(device, monkeypatch):
    if device == "cuda" and not m.torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = m.StageAcceptancePolicy()
    monkeypatch.setattr(m.StageAcceptancePolicy, "from_environment", lambda: policy)
    t = m.torch.arange(40, dtype=m.torch.float64, device=device)
    joints = m.torch.zeros((2, 40, 24, 3), dtype=m.torch.float64, device=device)
    joints[..., 0] = (0.01 * m.torch.sin(t * 1.2))[None, :, None]
    candidate = (joints * 1.3).requires_grad_(True)
    cfg = m.MotionGenerationConfig(device=device)
    loss, terms = m._clean_jerk_tolerance_loss_torch(candidate, joints, cfg)
    base = np.linalg.norm(np.diff(joints.cpu().numpy(), n=3, axis=1) * 30**3, axis=-1).reshape(2, -1)
    pred = np.linalg.norm(np.diff(candidate.detach().cpu().numpy(), n=3, axis=1) * 30**3, axis=-1).reshape(2, -1)
    for label, ratio, margin in (("p95", 1.1, 25.0), ("max", 1.02, 40.0)):
        a = base.max(1) if label == "max" else np.percentile(base, 95, axis=1)
        b = pred.max(1) if label == "max" else np.percentile(pred, 95, axis=1)
        allowed = np.maximum(a * ratio, a + margin)
        expected = np.mean(np.maximum(b - allowed, 0) / np.maximum(allowed - a, 1))
        assert terms[f"jerk_{label}_excess"].item() == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert m.torch.isfinite(candidate.grad).all()


def test_gradient_conflict_measurement_does_not_mutate_optimizer_gradients():
    model = m.torch.nn.Linear(2, 1, bias=False)
    repair = model.weight.sum()
    clean = -model.weight.sum()
    report = m._refiner_gradient_diagnostics(model, repair, clean, 1.0)
    assert report["gradient_cosine"] == pytest.approx(-1)
    assert report["conflicting"]
    assert report["combined_to_sum_norm_ratio"] == pytest.approx(0)
    assert model.weight.grad is None
    (repair + 0.5 * clean).backward()
    m.torch.testing.assert_close(model.weight.grad, m.torch.full_like(model.weight, 0.5))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_and_extremity_jerk_statistics_match_audit(device):
    if device == "cuda" and not m.torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from motion_geometry.physical import _window_percentile_max, EXTREMITY_JOINTS
    rng = np.random.default_rng(71)
    values = rng.uniform(0, 100, (2, 117, 24))
    values[:, 85:99, 20] += 350.0
    actual = m._clean_jerk_statistics_torch(m.torch.as_tensor(values, device=device), 30)
    for index, row in enumerate(values):
        extremes = row[:, list(EXTREMITY_JOINTS)]
        expected = {
            "p95": np.percentile(row, 95), "max": row.max(),
            "window_p95": _window_percentile_max(row, fps=30),
            "extremity_p95": np.percentile(extremes, 95),
            "extremity_window_p95": _window_percentile_max(extremes, fps=30),
        }
        for key, value in expected.items():
            assert actual[key][index].item() == pytest.approx(value, rel=1e-6)


def test_zero_clean_gradient_is_not_reported_as_conflict():
    model = m.torch.nn.Linear(2, 1, bias=False)
    repair = model.weight.sum()
    report = m._refiner_gradient_diagnostics(model, repair, 0 * repair, 0.5)
    assert report["gradient_cosine"] is None
    assert report["clean_gradient_norm_weighted"] == 0
    assert report["combined_to_sum_norm_ratio"] == pytest.approx(1)
    assert not report["conflicting"]


def test_fixed_selection_is_deterministic_and_source_balanced():
    sources = np.asarray(["a"] * 20 + ["b"] * 30 + ["c"] * 40)
    selected = d.fixed_indices(sources, 8)
    assert selected == d.fixed_indices(sources, 8)
    assert len(set(selected)) == 8
    assert set(sources[selected]) == {"a", "b", "c"}


def test_real_fit_decision_cannot_publish_even_when_fit_passes():
    with mock.patch.object(m, "_checkpoint_validation_decision", return_value={
        "scientific_acceptance": True, "publish_allowed": True, "reasons": [],
        "observed": {}, "thresholds": {},
    }):
        decision = d.fit_decision({}, m.MotionGenerationConfig())
    assert decision["fit_passed"]
    assert not decision["publish_allowed"]
    assert not decision["scientific_acceptance"]


def test_fixed_fit_runner_reuses_frozen_real_contract_and_never_fits_val(tmp_path, monkeypatch):
    train_path = _write_training_db(tmp_path, "train")
    val_path = _write_training_db(tmp_path, "validation")
    original_load = m.load_db

    def load(path):
        db = original_load(path)
        db["source_formats"] = np.asarray(["chang_e_official_smpl"] * len(db["paths"]))
        return db

    monkeypatch.setattr(m, "load_db", load)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"device": "cpu", "window_len": 40}), encoding="utf-8")
    args = Namespace(config=str(cfg_path), db=str(train_path), val_db=str(val_path),
                     out_dir=str(tmp_path / "diagnostic"), check_report=None,
                     steps=2, windows=2, eval_every=1, gradient_every=1)
    original_window = m.load_motion_window
    paths_read = []

    def window(path, *pos, **kw):
        paths_read.append(str(path))
        return original_window(path, *pos, **kw)

    monkeypatch.setattr(m, "load_motion_window", window)
    count_threads = m.torch.get_num_threads()
    m.torch.set_num_threads(2)
    try:
        code = d.run(args)
    finally:
        m.torch.set_num_threads(count_threads)
    report = json.loads((tmp_path / "diagnostic/diagnostic_report.json").read_text())
    assert code in (0, 2)
    assert report["completed"]
    assert len(paths_read) == 2  # not reloaded or re-corrupted per step
    assert all("validation" not in path for path in paths_read)
    assert len(report["gradient_history"]) == 2
    assert (tmp_path / "diagnostic/fit_step_000001.json").is_file()
    assert report["last"]["step"] == 2
    assert not report["published"]
    assert not report["used_for_formal_checkpoint_selection"]
    assert not (tmp_path / "diagnostic/boundary_refiner.pt").exists()
    weights = m._trusted_torch_load(tmp_path / "diagnostic/diagnostic_weights.pt", map_location="cpu")
    assert weights["version"] != m.REFINER_MODEL_VERSION
    assert weights["formal_checkpoint"] is False
    args.check_report = str(tmp_path / "diagnostic/diagnostic_report.json")
    report["fit_gate"]["fit_passed"] = False
    (tmp_path / "diagnostic/diagnostic_report.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="has not passed"):
        d.check_report(args, m.MotionGenerationConfig.from_json(cfg_path).apply_env())


def test_diagnostic_guard_rejects_stale_revision_and_changed_event(tmp_path, monkeypatch):
    cfg = m.MotionGenerationConfig()
    event = tmp_path / "event.npy"
    np.save(event, np.zeros((4, 151), dtype=np.float32))
    metrics = {
        "num_windows": 1, "reconstruction_product_log_l1": 0.001,
        "physical_quality": {
            "num_windows": 1, "stage_repair": {"pass_rate": 1},
            "temporal_repair": {"pass_rate": 1}, "clean_input_identity": {"pass_rate": 1},
            "fk_position_error_m_p95": 0.01, "fk_position_error_m_max": 0.02,
        },
    }
    fingerprint = {"code_revision": "current"}
    observable = {"schema":m.BOUNDARY_PROTOCOL,"num_windows":1,
                  "endpoint":{"pass_rate":1},"temporal":{"pass_rate":1},
                  "physical_non_regression":{"pass_rate":1},
                  "endpoint_informative":1,"temporal_informative":1,
                  "reference_fk_p95_m":.01,"reference_fk_max_m":.02,"reference_product_log_l1":.001}
    metrics["physical_quality"]["observable_boundary"] = observable
    metrics["cross_event"] = observable
    monkeypatch.setattr(d, "_fingerprint", lambda args, config: fingerprint)
    report = {
        "schema": d.SCHEMA, "role": "training_fit_diagnostic_only", "completed": True,
        "published": False, "used_for_formal_checkpoint_selection": False,
        "fingerprint": fingerprint, "fit_gate": {"fit_passed": True},
        "best": {"metrics": metrics}, "windows": [{"path": str(event), "sha256": d.file_sha256(event)}],
    }
    path = tmp_path / "report.json"
    args = Namespace(check_report=str(path), windows=1)
    path.write_text(json.dumps(report), encoding="utf-8")
    assert d.check_report(args, cfg) == 0
    fingerprint = {"code_revision": "new"}
    with pytest.raises(RuntimeError, match="mismatch"):
        d.check_report(args, cfg)
    fingerprint = report["fingerprint"]
    np.save(event, np.ones((4, 151), dtype=np.float32))
    with pytest.raises(RuntimeError, match="event changed"):
        d.check_report(args, cfg)
