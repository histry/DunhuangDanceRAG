import json
import random
from argparse import Namespace

import numpy as np
import pytest

from training import motion_models as m
from training import refiner_diagnostics as fixed
from training import refiner_factor_diagnostics as factors
from training import refiner_trace
from tests.test_motion_training_resume import _write_training_db


pytestmark = pytest.mark.skipif(m.torch is None, reason="PyTorch unavailable")


def motion(frames=120):
    x = np.zeros((frames, 151), dtype=np.float32)
    x[:, 7:] = np.tile(m.identity6d_np(), 24)
    x[:, 5] = .95
    x[:, 4] = .05 * np.sin(np.linspace(0, 4, frames))
    return x


def test_recipes_pair_modes_and_isolate_noise_from_position():
    cfg = m.MotionGenerationConfig(device="cpu")
    random.seed(71)
    np.random.seed(72)
    py_state, np_state = random.getstate(), np.random.get_state()
    rows = factors.make_recipes(cfg, 8, 42)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert [sum(r["split"] == s for r in rows) for s in factors.SPLITS] == [32, 64, 16]
    again = factors.make_recipes(cfg, 8, 42)
    assert [r["tangent_sha256"] for r in rows] == [r["tangent_sha256"] for r in again]
    for row in rows:
        width = row["b"] - row["a"]
        starts = set()
        for center in range(max(2, cfg.window_len // 5), max(3, 4 * cfg.window_len // 5) + 1):
            a = max(1, center - width // 2)
            b = min(cfg.window_len - 1, a + width)
            starts.add(max(1, b - width))
        assert row["a"] in starts
        reference = next(r for r in rows if r["split"] == "fit_seen" and r["window_index"] == row["window_index"] and r["recipe_id"] == row["recipe_id"])
        if row["split"] == "probe_unseen_noise":
            assert (row["a"], row["b"]) == (reference["a"], reference["b"])
            assert row["tangent_sha256"] != reference["tangent_sha256"]
        if row["split"] == "probe_unseen_position":
            assert row["b"] - row["a"] == reference["b"] - reference["a"]
            assert row["a"] != reference["a"]
            assert row["tangent_sha256"] == reference["tangent_sha256"]


def test_modes_have_same_seam_but_distinct_geometry_and_ignore_noise_for_bridge():
    cfg = m.MotionGenerationConfig(device="cpu")
    clean = motion()
    rows = factors.make_recipes(cfg, 1, 42)
    recipe = rows[0]
    original = clean.copy()
    outputs = {mode: m.degrade_for_refiner(clean, cfg=cfg, mode=mode, recipe=recipe, finalize_contract=False)
               for mode in factors.MODES}
    np.testing.assert_array_equal(clean, original)
    for _, seam in outputs.values():
        np.testing.assert_array_equal(seam, outputs["mixed"][1])
    assert not np.allclose(outputs["bridge_only"][0], outputs["mixed"][0])
    assert not np.allclose(outputs["tangent_only"][0], outputs["mixed"][0])
    new_noise = next(r for r in rows if r["split"] == "probe_unseen_noise")
    bridge, _ = m.degrade_for_refiner(clean, cfg=cfg, mode="bridge_only", recipe=new_noise, finalize_contract=False)
    np.testing.assert_array_equal(bridge, outputs["bridge_only"][0])
    outside = np.ones(len(clean), bool)
    outside[recipe["a"]:recipe["b"]] = False
    for damaged, _ in outputs.values():
        np.testing.assert_array_equal(damaged[outside], clean[outside])
    with pytest.raises(ValueError, match="invalid seam"):
        m.degrade_for_refiner(clean, cfg=cfg, recipe={**recipe, "b": 999})


def test_default_mixed_replays_with_captured_recipe_without_rng_side_effects(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu")
    clean = motion()
    saved = []
    original = m._refiner_tangent_noise_np
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        saved.append(result.copy())
        return result
    monkeypatch.setattr(m, "_refiner_tangent_noise_np", capture)
    random.seed(19)
    np.random.seed(20)
    default, seam = m.degrade_for_refiner(clean, cfg=cfg, mode="mixed")
    core = np.flatnonzero(seam[:, 0] >= .5)
    recipe = {"a": int(core[0]), "b": int(core[-1]) + 1, "tangent": saved[0]}
    state = (random.getstate(), np.random.get_state())
    replay, replay_seam = m.degrade_for_refiner(clean, cfg=cfg, recipe=recipe, mode="mixed")
    np.testing.assert_array_equal(default, replay)
    np.testing.assert_array_equal(seam, replay_seam)
    assert random.getstate() == state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1][1])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trace_does_not_change_default_outputs_or_gradients(device):
    if device == "cuda" and not m.torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    t = m.torch
    cfg = m.MotionGenerationConfig(device=device)
    x = t.from_numpy(motion(40)[None]).to(device)
    output = (t.randn(1, 40, 79, device=device) * .01).requires_grad_(True)
    joint = t.zeros(1, 40, 24, device=device)
    joint[:, 10:25] = .18
    root = joint[..., :1]
    plain = m._decode_product_refiner_output(x, output, joint, root, root, cfg)
    trace = {}
    observed = m._decode_product_refiner_output(x, output, joint, root, root, cfg, trace=trace)
    assert t.equal(plain, observed)
    grad_a = t.autograd.grad(plain.square().sum(), output, retain_graph=True)[0]
    grad_b = t.autograd.grad(observed.square().sum(), output)[0]
    assert t.equal(grad_a, grad_b)
    assert all(not trace[key].requires_grad for key in refiner_trace.STAGES)
    for variant in factors.VARIANTS:
        counterfactual = m._decode_product_refiner_output(x, output, joint, root, root, cfg, diagnostic_variant=variant)
        assert t.equal(counterfactual[:, :10], x[:, :10])
        assert t.equal(counterfactual[:, 25:], x[:, 25:])


def test_cap_fraction_measures_post_mask_vectors_not_raw_output():
    cfg = m.MotionGenerationConfig(device="cpu", product_refiner_residual_smoothing_passes=0,
                                   product_refiner_residual_taper_frames=0)
    x = m.torch.from_numpy(motion(40)[None])
    output = m.torch.ones(1, 40, 79)
    mask = m.torch.ones(1, 40, 24) * .001
    root = mask[..., :1]
    trace = {}
    m._decode_product_refiner_output(x, output, mask, root, root, cfg, trace=trace)
    summary = refiner_trace.summarize_window(refiner_trace.detached_numpy(trace), 0, x[0].numpy(), x[0].numpy(), np.ones((40, 1)))
    assert summary["root_m"]["cap"]["clipped_fraction"] == 0
    assert summary["rotation_rad"]["cap"]["clipped_fraction"] == 0
    mask.fill_(1)
    m._decode_product_refiner_output(x, output, mask, root, root, cfg, trace=trace)
    summary = refiner_trace.summarize_window(refiner_trace.detached_numpy(trace), 0, x[0].numpy(), x[0].numpy(), np.ones((40, 1)))
    assert summary["rotation_rad"]["cap"]["clipped_fraction"] == 1


def test_trivial_corruption_is_not_a_success():
    row = {"geometry": {"accepted": True, "detail": {"degraded_product_log_l1_to_clean": 0}},
           "temporal": {"accepted": True, "detail": {"degraded": dict.fromkeys(
               ("seam_velocity_error", "seam_acceleration_error", "seam_jerk_error", "endpoint_velocity_error"), 0)}}}
    result = factors.informative_rates([row])
    assert result["geometry"] == {"informative": 0, "trivial": 1, "passed": 0, "rate": None}
    assert result["temporal"]["rate"] is None


def test_trace_distinguishes_seam_dilation_from_zero_mask_protection():
    cfg = m.MotionGenerationConfig(device="cpu")
    x = m.torch.from_numpy(motion(40)[None])
    output = m.torch.ones(1, 40, 79) * .02
    joint = m.torch.zeros(1, 40, 24)
    joint[:, 7:23] = .18
    root = joint[..., :1]
    trace = {}
    m._decode_product_refiner_output(x, output, joint, root, root, cfg, trace=trace)
    seam = np.zeros((40, 1), dtype=np.float32)
    seam[10:20] = 1
    summary = refiner_trace.summarize_window(refiner_trace.detached_numpy(trace), 0, x[0].numpy(), x[0].numpy(), seam)
    for label in ("root_m", "rotation_rad"):
        assert summary[label]["mask_by_scope"]["outside"]["max"] > 0
        assert summary[label]["stages"]["applied"]["outside"]["max"] > 0
        assert summary[label]["applied_where_mask_zero"]["max"] < 1e-7


def cohort_fixture(tmp_path, monkeypatch):
    train, val = _write_training_db(tmp_path, "train"), _write_training_db(tmp_path, "validation")
    original = m.load_db
    def load(path):
        db = original(path)
        db["source_formats"] = np.asarray(["chang_e_official_smpl"] * len(db["paths"]))
        return db
    monkeypatch.setattr(m, "load_db", load)
    monkeypatch.setenv("MOTION_DEVICE", "cpu")
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"device": "cpu", "window_len": 40}), encoding="utf-8")
    cfg = m.MotionGenerationConfig.from_json(cfg_path).apply_env()
    db = load(train)
    clean = np.stack([m.load_motion_window(path, 40, cfg, random_crop=False) for path in db["paths"]])
    cond = m._descriptor_values_in_training_coordinates(db, db)
    directory = tmp_path / "cohort"
    directory.mkdir()
    np.savez_compressed(directory / "fixed_training_batch.npz", clean=clean, cond=cond)
    report = {"schema": fixed.SCHEMA, "role": "training_fit_diagnostic_only",
              "fingerprint": {"train_db_sha256": fixed.file_sha256(train), "validation_db_sha256": fixed.file_sha256(val),
                              "config_sha256": m._training_config_sha256(cfg, stage="refiner"), "code_revision": "old_revision"},
              "fixed_batch_sha256": fixed.file_sha256(directory / "fixed_training_batch.npz"),
              "windows": [{"event_index": i, "path": str(path), "sha256": fixed.file_sha256(path),
                           "source_uid": str(db["source_uids"][i])} for i, path in enumerate(db["paths"])]}
    (directory / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    return cfg, cfg_path, train, val, db, directory


def test_cohort_content_and_training_identity_are_checked(tmp_path, monkeypatch):
    cfg, _, train, val, db, directory = cohort_fixture(tmp_path, monkeypatch)
    clean, cond, rows, provenance = factors.load_cohort(directory, db, cfg, train, val, 2)
    assert clean.shape == (2, 40, 151)
    assert provenance["original_code_revision"] == "old_revision"
    report = json.loads((directory / "diagnostic_report.json").read_text())
    report["windows"][0]["path"] = str(tmp_path / "validation_0.npy")
    (directory / "diagnostic_report.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="training event"):
        factors.load_cohort(directory, db, cfg, train, val, 2)


def test_reference_snapshot_is_read_only_and_checks_contracts(tmp_path, monkeypatch):
    cfg, _, train, val, db, _ = cohort_fixture(tmp_path, monkeypatch)
    train_contract = m._training_db_contract(db, cfg, "test train")
    val_contract = m._training_db_contract(m.load_db(val), cfg, "test validation metadata")
    model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32)
    optimizer = m.torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    path = tmp_path / "reference.training_snapshot.pt"
    monkeypatch.setattr(m, "_training_code_revision", lambda: "old_reference_revision")
    payload = m._save_training_resume_snapshot(
        path, stage="refiner", model=model, optimizer=optimizer,
        completed_steps=1000, target_steps=8000, elapsed_seconds=20,
        cfg=cfg, training_contract=train_contract, validation_contract=val_contract,
    )
    before = fixed.file_sha256(path)
    monkeypatch.setattr(m, "_training_code_revision", lambda: "new_diagnostic_revision")
    loaded, info = factors.load_reference_snapshot(path, cfg, train_contract, val_contract, "cpu")
    assert info["code_revision"] == "old_reference_revision" and info["read_only"]
    assert not info["used_to_initialize_experiments"]
    assert fixed.file_sha256(path) == before
    for key, value in model.state_dict().items():
        assert m.torch.equal(value, loaded.state_dict()[key])
    with pytest.raises(RuntimeError):
        factors.load_reference_snapshot(path, cfg, val_contract, train_contract, "cpu")
    payload["formal_checkpoint"] = True
    m._atomic_torch_save(payload, path)
    with pytest.raises(RuntimeError, match="training snapshot"):
        factors.load_reference_snapshot(path, cfg, train_contract, val_contract, "cpu")


def test_runner_isolated_fresh_models_and_no_validation_motion(tmp_path, monkeypatch):
    cfg, config, train, val, db, directory = cohort_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "load_motion_window", lambda *a, **k: pytest.fail("no event/validation motion reload permitted"))
    original_model = m.ProductManifoldTemporalRefiner
    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", lambda *a: original_model(*a, hidden=16))
    # Actual first-case audit/decoder in every bank, keeping this fixture small.
    original_evaluate = factors.evaluate_bank
    calls = []
    def evaluate(model, bank, config, **kwargs):
        batch, rows = bank
        calls.append((kwargs["label"], len(rows)))
        return original_evaluate(model, (factors.select_batch(batch, slice(0, 1)), rows[:1]), config, **kwargs)
    monkeypatch.setattr(factors, "evaluate_bank", evaluate)
    args = Namespace(config=str(config), db=str(train), val_db=str(val), fixed_fit_dir=str(directory),
                     reference_snapshot=None, out_dir=str(tmp_path / "experiment"), windows=2,
                     steps=1, eval_every=1, counterfactual_windows=1)
    threads = m.torch.get_num_threads()
    m.torch.set_num_threads(2)
    try:
        assert factors.run(args) == 0
    finally:
        m.torch.set_num_threads(threads)
    report = json.loads((tmp_path / "experiment/factor_report.json").read_text())
    assert report["completed"] and not report["published"]
    assert not report["scientific_acceptance"]
    assert report["selection_policy"] == "fixed_final_step_not_best_unseen_score"
    assert set(report["modes"]) == set(factors.MODES)
    assert len({v["initial_weights_sha256"] for v in report["modes"].values()}) == 1
    assert not any("bridge_only" in label and "probe_unseen_noise" in label for label, count in calls)
    assert set(count for _, count in calls) == {4, 8, 16}
    for mode in factors.MODES:
        payload = m._trusted_torch_load(tmp_path / "experiment" / mode / "diagnostic_final.pt", map_location="cpu")
        assert payload["version"] != m.REFINER_MODEL_VERSION
        assert not payload["formal_checkpoint"]
        with pytest.raises(RuntimeError, match="Formal generation rejects"):
            m._cached_inference_model("boundary_refiner", tmp_path / "experiment" / mode / "diagnostic_final.pt", cfg)
    with pytest.raises(FileExistsError):
        factors.run(args)
