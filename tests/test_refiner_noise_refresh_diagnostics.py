import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from training import motion_models as m
from training import refiner_diagnostics as fixed
from training import refiner_factor_diagnostics as factors
from training import refiner_noise_refresh_diagnostics as refresh
from tests.test_refiner_factor_diagnostics import cohort_fixture, motion


pytestmark = pytest.mark.skipif(m.torch is None, reason="PyTorch unavailable")


def factor_fixture(tmp_path, monkeypatch):
    cfg, config, train, val, db, cohort = cohort_fixture(tmp_path, monkeypatch)
    args = Namespace(config=str(config), db=str(train), val_db=str(val), fixed_fit_dir=str(cohort),
                     factor_dir=str(tmp_path / "factor"), out_dir=str(tmp_path / "refresh"),
                     windows=2, steps=1, eval_every=1)
    clean, cond, windows, _ = factors.load_cohort(cohort, db, cfg, train, val, 2)
    directory = tmp_path / "factor"
    directory.mkdir()
    recipes = factors.make_recipes(cfg, 2, cfg.seed)
    np.savez_compressed(directory / "clean_cohort.npz", clean=clean, cond=cond)
    np.savez_compressed(directory / "recipes.npz", **{f"tangent_{row['case_id']}": row["tangent"] for row in recipes})
    m.torch.manual_seed(cfg.seed)
    model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32)
    m._atomic_torch_save({"version": "refiner_factor_diagnostic_only_v1", "formal_checkpoint": False,
                          "publish_allowed": False, "model_state_dict": model.state_dict()},
                         directory / "diagnostic_initial_weights.pt")
    report = {"schema": factors.SCHEMA, "completed": True, "published": False,
              "role": "training_corruption_factor_diagnostic_only", "fingerprint": fixed._fingerprint(args, cfg),
              "windows": windows, "corruption_severity": .06,
              "recipes": [refresh.metadata(row) for row in recipes],
              "bank_files_sha256": {name: fixed.file_sha256(directory / name) for name in ("clean_cohort.npz", "recipes.npz")},
              "initial_weights_sha256": fixed.file_sha256(directory / "diagnostic_initial_weights.pt")}
    report["fingerprint"]["factor_code_sha256"] = {
        "product_manifold.py": fixed.file_sha256(Path(m.__file__).parents[1] / "motion_geometry/product_manifold.py")}
    (directory / "factor_report.json").write_text(json.dumps(report), encoding="utf-8")
    return args, cfg, clean, cond, windows, recipes


def test_new_probes_are_disjoint_and_position_comparison_changes_only_position():
    cfg = m.MotionGenerationConfig(device="cpu")
    old = factors.make_recipes(cfg, 8, cfg.seed)
    random.seed(123)
    np.random.seed(124)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    rows = refresh.make_evaluation_recipes(old, cfg, refresh.NoiseStream(cfg, old))
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert [sum(row["split"] == split for row in rows) for split in refresh.SPLITS] == [32, 64, 16]
    previous = {row["case_id"]: row for row in old}
    by_id = {row["case_id"]: row for row in rows}
    for row in rows:
        if row["split"] == "anchor_fixed_noise":
            origin = previous[int(row["case_id"].split("_")[1])]
            assert row["noise_seed"] == origin["noise_seed"]
            np.testing.assert_array_equal(row["tangent"], origin["tangent"])
            assert (row["a"], row["b"]) == (origin["a"], origin["b"])
        else:
            assert row["noise_seed"] not in {r["noise_seed"] for r in old}
            assert row["tangent_sha256"] not in {r["tangent_sha256"] for r in old}
        if row["split"] == "probe_unseen_position":
            paired = by_id[row["paired_noise_case_id"]]
            np.testing.assert_array_equal(row["tangent"], paired["tangent"])
            assert row["window_index"] == paired["window_index"]
            assert row["a"] != paired["a"] and row["b"] - row["a"] == paired["b"] - paired["a"]
            assert (row["a"], row["b"]) not in {(r["a"], r["b"]) for r in old if r["window_index"] == row["window_index"]}
            assert row["a"] not in {r["a"] for r in old if r["window_index"] == row["window_index"]}
            assert row["a"] + row["b"] not in {r["a"] + r["b"] for r in old if r["window_index"] == row["window_index"]}
            assert row["a"] in factors.allowed_seam_starts(cfg, row["b"] - row["a"])


def test_refresh_plan_reproducible_unique_and_shared_without_changing_seams(tmp_path):
    cfg = m.MotionGenerationConfig(device="cpu")
    old = factors.make_recipes(cfg, 2, cfg.seed)
    plans = []
    for _ in range(2):
        stream = refresh.NoiseStream(cfg, old)
        probes = refresh.make_evaluation_recipes(old, cfg, stream)
        fit = [row for row in probes if row["split"] == "anchor_fixed_noise"]
        order = refresh.training_order(len(fit), 5, cfg.seed)
        plans.append(refresh.make_refresh_plan(fit, order, stream))
    a, b = ([row for batch in plan for row in batch] for plan in plans)
    assert [refresh.metadata(row) for row in a] == [refresh.metadata(row) for row in b]
    assert len({row["noise_seed"] for row in a}) == 40
    assert len({row["tangent_sha256"] for row in a}) == 40
    assert not {row["noise_seed"] for row in a} & {row["noise_seed"] for row in probes}
    assert not {row["tangent_sha256"] for row in a} & {row["tangent_sha256"] for row in probes}
    for batch, indices in zip(plans[0], order):
        for row, index in zip(batch, indices):
            base = fit[index]
            assert (row["window_index"], row["a"], row["b"]) == (base["window_index"], base["a"], base["b"])
            expected = m._refiner_tangent_noise_np(row["b"] - row["a"], .06, cfg, rng=np.random.default_rng(row["noise_seed"]))
            np.testing.assert_array_equal(row["tangent"], expected)
    refresh.save_noise_archive(tmp_path / "noise.npz", a)
    with np.load(tmp_path / "noise.npz", allow_pickle=False) as saved:
        for i, row in enumerate(a):
            np.testing.assert_array_equal(saved["tangent"][saved["offsets"][i]:saved["offsets"][i+1]], row["tangent"])


def test_seed_collisions_are_rejected_before_drawing(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu")
    old = factors.make_recipes(cfg, 1, cfg.seed)
    stream = refresh.NoiseStream(cfg, old)
    blocked = old[0]["noise_seed"]
    original = factors.private_seed
    monkeypatch.setattr(factors, "private_seed", lambda *parts: blocked if parts[-1] == 0 else original(*parts))
    row = stream.draw(20, "test")
    assert row["noise_seed"] != blocked


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_and_online_batch_share_same_preprocessing_and_recompute_masks(device, monkeypatch):
    if device == "cuda" and not m.torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("MOTION_GPU_PREPROCESSING", "1")
    cfg = m.MotionGenerationConfig(device=device)
    clean, cond, windows = motion()[None], np.zeros((1, 32), np.float32), [{"source_uid": "train_only"}]
    old = factors.make_recipes(cfg, 1, cfg.seed)
    stream = refresh.NoiseStream(cfg, old)
    recipes = refresh.make_evaluation_recipes(old, cfg, stream)
    fit = [row for row in recipes if row["split"] == "anchor_fixed_noise"]
    banks = factors.build_banks(clean, cond, windows, recipes, cfg, device, modes=("mixed",), splits=("anchor_fixed_noise",))
    cached = factors.select_batch(banks[("mixed", "anchor_fixed_noise")][0], [1, 0])
    online, _ = factors.prepare_bank(clean, cond, windows, [fit[1], fit[0]], cfg, device, "mixed")
    for key in cached:
        m.torch.testing.assert_close(cached[key], online[key], rtol=0, atol=0)
    name = "_risk_masks_for_batch_torch" if device == "cuda" else "_risk_masks_for_batch_np"
    original = getattr(m, name)
    observed_inputs = []
    def capture(values, *args, **kwargs):
        observed_inputs.append(values.detach().cpu().numpy().copy() if m.torch.is_tensor(values) else values.copy())
        return original(values, *args, **kwargs)
    monkeypatch.setattr(m, name, capture)
    fresh = {**fit[0], **stream.draw(fit[0]["b"] - fit[0]["a"], "new_batch")}
    updated, _ = factors.prepare_bank(clean, cond, windows, [fresh], cfg, device, "mixed")
    assert not m.torch.equal(updated["bad"][0], online["bad"][1])
    assert m.torch.equal(updated["seam"][0], online["seam"][1])
    assert len(observed_inputs) == 1
    np.testing.assert_array_equal(observed_inputs[0][0], updated["bad"][0].cpu().numpy())
    # Risk thresholds may legitimately give equal masks for two weak noises.
    # Verify recomputation from new input rather than demanding unequal values.


def test_no_edit_is_exact_identity_not_zero_logits(monkeypatch):
    cfg = m.MotionGenerationConfig(device="cpu")
    old = factors.make_recipes(cfg, 1, cfg.seed)
    clean, _ = m.enforce_edge151_contract_np(motion(), cfg, derive_contact=True, project_rot=True)
    bank = factors.prepare_bank(clean[None], np.zeros((1, 32), np.float32), [{}], [old[0]], cfg, "cpu", "mixed")
    before = {key: value.clone() for key, value in bank[0].items()}
    original = m._record_validation_physical_prediction
    def checked(acc, pred, clean, config, **kwargs):
        np.testing.assert_array_equal(pred, kwargs["degraded"])
        return original(acc, pred, clean, config, **kwargs)
    monkeypatch.setattr(m, "_record_validation_physical_prediction", checked)
    metrics = refresh.evaluate_no_edit(bank, cfg, "test")
    assert metrics["informative_repair"]["geometry"]["passed"] == 0
    assert metrics["informative_repair"]["temporal"]["passed"] == 0
    assert metrics["windows"][0]["clean_identity"]["accepted"]
    assert metrics["windows"][0]["direction_cosine"] is None
    for key, value in before.items():
        assert m.torch.equal(value, bank[0][key])


@pytest.mark.parametrize("different_cases", [False, True])
def test_paired_counts_exclude_trivial_targets_and_reject_unpaired_cases(tmp_path, different_cases):
    row = {"case_id": "case_0", "geometry": {"accepted": True, "detail": {"degraded_product_log_l1_to_clean": 0}},
           "temporal": {"accepted": True, "detail": {"degraded": dict.fromkeys(
               ("seam_velocity_error", "seam_acceleration_error", "seam_jerk_error", "endpoint_velocity_error"), 0)}},
           "clean_identity": {"accepted": True}}
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text(json.dumps({"windows": [row]}))
    right.write_text(json.dumps({"windows": [{**row, "case_id": "wrong_case" if different_cases else "case_0"}]}))
    report = {"no_edit": {mode: {split: {} for split in refresh.SPLITS} for mode in refresh.MODES},
              "experiments": {mode: {arm: {"history": [{"results": {
                  split: {"report": str(left if arm == "fixed_noise" else right)} for split in refresh.SPLITS}}]}
                  for arm in refresh.ARMS} for mode in refresh.MODES}}
    if different_cases:
        with pytest.raises(RuntimeError, match="identical cases"):
            refresh.compare_arms(report)
    else:
        result = refresh.compare_arms(report)["modes"]["mixed"]["probe_unseen_noise"]["paired_outcomes"]
        for kind in ("geometry", "temporal"):
            assert result[kind]["both_pass"] == 0 and result[kind]["trivial_excluded"] == 1
        assert result["clean_identity"]["both_pass"] == 1


@pytest.mark.parametrize("change", ["config", "mask", "archive", "trained_weights"])
def test_parent_replay_contracts_fail_closed(tmp_path, monkeypatch, change):
    args, cfg, clean, cond, windows, _ = factor_fixture(tmp_path, monkeypatch)
    path = tmp_path / "factor/factor_report.json"
    report = json.loads(path.read_text())
    if change == "config":
        report["fingerprint"]["config_sha256"] = "wrong"
    elif change == "mask":
        report["fingerprint"]["mask_and_physical_environment"]["GROUNDING_MASK_DILATE"] = "100"
    elif change == "archive":
        np.savez_compressed(tmp_path / "factor/clean_cohort.npz", clean=clean + 1, cond=cond)
    else:
        initial = tmp_path / "factor/diagnostic_initial_weights.pt"
        payload = m._trusted_torch_load(initial, map_location="cpu")
        next(iter(payload["model_state_dict"].values())).add_(.1)
        m._atomic_torch_save(payload, initial)
        report["initial_weights_sha256"] = fixed.file_sha256(initial)
    path.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        refresh.load_factor_inputs(args.factor_dir, args, cfg, clean, cond, windows)


def test_runner_pairs_four_fresh_models_without_validation_motion_or_promotion(tmp_path, monkeypatch):
    original_model = m.ProductManifoldTemporalRefiner
    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", lambda *a: original_model(*a, hidden=16))
    args, cfg, _, _, _, _ = factor_fixture(tmp_path, monkeypatch)
    parent_hashes = {str(path): fixed.file_sha256(path) for path in (tmp_path / "factor").iterdir()}
    monkeypatch.setattr(m, "load_motion_window", lambda *a, **k: pytest.fail("no motion reloading, especially validation"))
    # Exercise actual metrics/decoder for one paired case per bank, not hundreds of synthetic repeats.
    original_eval, original_no_edit = factors.evaluate_bank, refresh.evaluate_no_edit
    seen = []
    def evaluate(model, bank, cfg, **kwargs):
        seen.append((kwargs["label"], len(bank[1])))
        return original_eval(model, (factors.select_batch(bank[0], slice(0, 1)), bank[1][:1]), cfg, **kwargs)
    def no_edit(bank, cfg, label):
        return original_no_edit((factors.select_batch(bank[0], slice(0, 1)), bank[1][:1]), cfg, label)
    monkeypatch.setattr(factors, "evaluate_bank", evaluate)
    monkeypatch.setattr(refresh, "evaluate_no_edit", no_edit)
    threads = m.torch.get_num_threads()
    m.torch.set_num_threads(2)
    try:
        assert refresh.run(args) == 0
    finally:
        m.torch.set_num_threads(threads)
    out = Path(args.out_dir)
    report = json.loads((out / "noise_refresh_report.json").read_text())
    assert report["completed"] and not report["scientific_acceptance"] and not report["published"]
    assert report["isolation"]["refresh_probe_seed_overlap"] == 0
    assert not report["isolation"]["validation_motion_read"]
    assert set(count for _, count in seen) == {4, 8, 16}
    runs = [report["experiments"][mode][arm] for mode in refresh.MODES for arm in refresh.ARMS]
    assert len(runs) == 4 and all(row["completed"] for row in runs)
    assert len({row["initial_weights_sha256"] for row in runs}) == 1
    assert len({row["training_order_sha256"] for row in runs}) == 1
    for mode in refresh.MODES:
        for arm in refresh.ARMS:
            with pytest.raises(RuntimeError, match="Formal generation rejects"):
                m._cached_inference_model("boundary_refiner", out / mode / arm / "diagnostic_latest.pt", cfg)
    assert all(fixed.file_sha256(path) == digest for path, digest in parent_hashes.items())
    with pytest.raises(FileExistsError):
        refresh.run(args)
