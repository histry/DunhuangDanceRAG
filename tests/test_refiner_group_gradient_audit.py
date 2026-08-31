import dataclasses
import json
from argparse import Namespace

import pytest
import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import refiner_group_gradient_audit as a


def bank_tensor(cfg):
    n, frames = 32, cfg.window_len
    clean = torch.zeros(n, frames, 151)
    clean[..., 7:] = torch.tensor(m.identity6d_np()).repeat(24)
    clean[..., 5] = .95
    group = torch.tensor([0, 1] * 8 + [2, 3] * 8)
    seam = torch.zeros(n, frames, 1)
    for i, label in enumerate(group):
        seam[i, 30:30 + (10 if label % 2 == 0 else 28)] = 1
    joint = seam.expand(n, frames, 24).clone() * .18
    root = seam.clone() * .18
    contact = root.clone()
    return {"clean": clean, "bad": clean.clone(), "seam": seam,
            "cond": torch.zeros(n, frames, 32), "group": group,
            "joint": joint, "root": root, "contact": contact,
            "clean_joint": joint.clone(), "clean_root": root.clone(), "clean_contact": contact.clone()}


@pytest.fixture
def frozen(tmp_path):
    cfg = m.MotionGenerationConfig(device="cpu")
    contract = d.fit_bank_contract(8, cfg)
    count = contract["context_reservoir_cycle_length"]
    fingerprint = {"code_revision": a.LEGACY_COMMIT, "model_version": m.REFINER_MODEL_VERSION,
                   "refiner_input_protocol": m.REFINER_INPUT_PROTOCOL,
                   "observable_objective_protocol": m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL,
                   "refiner_batch_aggregation_protocol": m.REFINER_BATCH_AGGREGATION_PROTOCOL,
                   "repair_safety_protocol": m.REFINER_REPAIR_SAFETY_PROTOCOL,
                   "refiner_update_protocol": m.REFINER_UPDATE_PROTOCOL,
                   "fit_protocol": d.FIT_PROTOCOL, "context_reservoir_protocol": d.CONTEXT_RESERVOIR_PROTOCOL,
                   "stage_acceptance_policy": dataclasses.asdict(m.StageAcceptancePolicy.from_environment()),
                   "mask_and_physical_environment": {}}
    windows = [{"test_window": i} for i in range(8)]
    config = dataclasses.asdict(cfg)
    config.pop("refiner_core_strength")
    config.pop("refiner_transition_strength")
    part = bank_tensor(cfg)
    bank = {"schema": "refiner_train_safe_start_context_reservoir_v4", "train_only": True,
            "formal_checkpoint": False, "publish_allowed": False, "fingerprint": fingerprint,
            "windows": windows, "contract": contract, "config": config, "anchor": part,
            "context_reservoir": {str(i): part for i in range(count)},
            "transaction_schedule": [[(i + j) % count for j in range(5)] for i in range(count)]}
    torch.save(bank, tmp_path / "fit_bank.pt")
    artifact = {"file": "fit_bank.pt", "sha256": a.file_sha256(tmp_path / "fit_bank.pt")}
    state = {"schema": "refiner_diagnostic_state_v1", "formal_checkpoint": False,
             "publish_allowed": False, "completed_steps": 400, "fingerprint": fingerprint,
             "fit_bank_artifact": artifact, "model_state_dict": {}}
    report = {"schema": "refiner_observable_bridge_diagnostic_v15_4_1", "fingerprint": fingerprint,
              "completed_steps": 400, "target_steps": 400, "completed": True,
              "stopped_early": False, "published": False, "fit_bank": contract,
              "fit_bank_artifact": artifact, "windows": windows}
    torch.save(state, tmp_path / "diagnostic_state.pt")
    (tmp_path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    return tmp_path, bank, state, report


def load(source, **kwargs):
    return a.load_transaction(source, a.LEGACY_COMMIT, 0,
                              legacy_core_strength=.02, legacy_transition_strength=1., **kwargs)


def test_exact_train_transaction_does_not_load_probe(frozen, monkeypatch):
    path, _, _, _ = frozen
    original = m._trusted_torch_load
    opened = []

    def tracked(file, **kwargs):
        opened.append(file.name)
        return original(file, **kwargs)

    monkeypatch.setattr(m, "_trusted_torch_load", tracked)
    _, batch, cfg, metadata = load(path)
    assert set(opened) == {"fit_bank.pt", "diagnostic_state.pt"}
    assert batch["clean"].shape == (192, 120, 151)
    assert [(batch["group"] == i).sum().item() for i in range(4)] == [48] * 4
    assert metadata["context_indices"] == [0, 1, 2, 3, 4]
    assert metadata["decoder_strength_evidence"].startswith("explicit_legacy")
    assert cfg.refiner_core_strength == .02


def test_loader_shape_contract_matches_production_preprocessing():
    cfg = m.MotionGenerationConfig(device="cpu")
    source = bank_tensor(cfg)
    actual = m._prepare_refiner_batch(
        source["clean"].numpy(), source["bad"].numpy(), source["seam"].numpy(),
        source["cond"].numpy(), cfg, "cpu")
    actual["group"] = source["group"]
    a._validate_bank(actual, cfg)


@pytest.mark.parametrize("failure", ["hash", "fingerprint", "schedule", "width", "probe", "source", "v15_5"])
def test_bad_or_incompatible_artifacts_fail_closed(frozen, failure):
    path, bank, state, report = frozen
    if failure == "hash":
        with (path / "fit_bank.pt").open("ab") as handle:
            handle.write(b"modified")
    elif failure == "fingerprint":
        state["fingerprint"] = {**state["fingerprint"], "extra": "not in report"}
    elif failure in {"schedule", "width", "probe"}:
        if failure == "schedule":
            bank["transaction_schedule"][0][1] = 0
        elif failure == "width":
            bank["anchor"]["seam"][0].zero_()
        else:
            bank["train_only"] = False
        torch.save(bank, path / "fit_bank.pt")
        report["fit_bank_artifact"]["sha256"] = a.file_sha256(path / "fit_bank.pt")
    elif failure == "source":
        # The fixture shares this dict between state/bank/report.
        state["fingerprint"]["code_revision"] = "wrong-commit"
        torch.save(bank, path / "fit_bank.pt")
        report["fit_bank_artifact"]["sha256"] = a.file_sha256(path / "fit_bank.pt")
    else:
        report["schema"] = "refiner_observable_bridge_diagnostic_v15_5"
    torch.save(state, path / "diagnostic_state.pt")
    (path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_legacy_strengths_cannot_be_silently_guessed(frozen):
    with pytest.raises(ValueError, match="supply both"):
        a.load_transaction(frozen[0], a.LEGACY_COMMIT, 0)


@pytest.mark.parametrize("anchor_has_clean_cond", [False, True])
def test_mixed_clean_condition_fields_cannot_be_dropped(frozen, anchor_has_clean_cond):
    path, bank, state, report = frozen
    # Preserve each individual bank's valid schema but make its optional clean
    # conditioning inconsistent. Previously the anchor keys silently won.
    bank["anchor"] = dict(bank["anchor"])
    bank["context_reservoir"] = {k: dict(v) for k, v in bank["context_reservoir"].items()}
    targets = [bank["anchor"]] if anchor_has_clean_cond else list(bank["context_reservoir"].values())
    for part in targets:
        part["clean_cond"] = part["cond"] + 1
    torch.save(bank, path / "fit_bank.pt")
    report["fit_bank_artifact"]["sha256"] = a.file_sha256(path / "fit_bank.pt")
    torch.save(state, path / "diagnostic_state.pt")
    (path / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identical fields"):
        load(path)


def test_same_direction_clean_tasks_can_still_oppose_repair():
    # Clean-to-clean cosine=+1 is NOT evidence about clean-vs-repair cosine.
    repair, clean = torch.tensor([1., 0.]), torch.tensor([-2., 0.])
    report = a._clean_repair_pair(repair, .5 * clean)
    assert report["cosine"] == -1
    assert report["combined_norm"] == 0
    assert report["weighted_clean_to_repair_ratio"] == 1
    report = a._clean_repair_pair(repair, 0 * clean)
    assert report["cosine"] is None and report["weighted_clean_to_repair_ratio"] == 0


def test_saved_policy_and_strengths_override_and_restore_caller_environment(monkeypatch):
    monkeypatch.setenv("PHYSICAL_STAGE_JERK_MAX_RATIO", "999")
    monkeypatch.setenv("MOTION_REFINER_CORE_STRENGTH", "0.9")
    with pytest.raises(RuntimeError):
        with a.frozen_environment({"mask_and_physical_environment": {}}, {"core": .02, "transition": 1.}):
            assert m.StageAcceptancePolicy.from_environment().jerk_max_ratio == 1.02
            assert m._refiner_decode_strengths(m.MotionGenerationConfig()) == {"core": .02, "transition": 1.}
            raise RuntimeError("test cleanup")
    assert m.StageAcceptancePolicy.from_environment().jerk_max_ratio == 999
    assert m._refiner_decode_strengths(m.MotionGenerationConfig())["core"] == .9


def test_known_conflicts_zero_gradients_and_read_only_state(monkeypatch):
    model = torch.nn.Module()
    model.trunk = torch.nn.Linear(1, 1, bias=False)
    model.out = torch.nn.Linear(1, 1, bias=False)
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 7
    original = {k: v.clone() for k, v in model.state_dict().items()}
    batch = {"group": torch.arange(4).repeat_interleave(48)}
    slopes = [(1., 0.), (-1., 0.), (0., 1.), (0., 0.)]

    def objective(net, batch, cfg, *, group_objectives):
        for label, (x, y) in zip(a.GROUPS, slopes):
            repair = (x * net.trunk.weight + y * net.out.weight).sum()
            group_objectives[label] = {
                "repair_objective": repair, "training_total": repair,
                "endpoint_deficit_mean": repair, "temporal_deficit_mean": repair * 0,
                "clean_identity": repair * 0,
            }

    monkeypatch.setattr(m, "_refiner_batch_objectives", objective)
    result = a.compute_geometry(model, batch, m.MotionGenerationConfig())
    matrix = result["geometry"]["repair_objective"]["all_parameters"]["cosine"]
    assert matrix[0][1] == -1 and matrix[0][2] == 0
    assert all(value is None for value in matrix[3])
    assert result["mean_training_gradient_norm"] == .25
    assert model.training
    assert all(torch.equal(p.grad, torch.ones_like(p) * 7) for p in model.parameters())
    assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())


def test_real_objective_group_gradient_reconstructs_full_transaction():
    cfg = m.MotionGenerationConfig(device="cpu", window_len=60)
    part = bank_tensor(cfg)
    batch = {k: torch.cat([v] * 6) for k, v in part.items()}
    # Different reference severities expose the bug where a per-group forward
    # silently recomputes the full-transaction normalization floor.
    amplitudes = torch.tensor([1e-5, .002, .02, .2])[batch["group"]]
    phase = torch.linspace(0, 8, cfg.window_len)
    batch["bad"][..., 4] = amplitudes[:, None] * phase.sin()
    torch.manual_seed(42)
    model = m.ProductManifoldTemporalRefiner(hidden=4)
    # Nonzero output exposes clean-identity and the physical branches too.
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-5)
    grouped = {}
    repair, protection, _, _ = m._refiner_batch_objectives(model, batch, cfg, group_objectives=grouped)
    full = torch.autograd.grad(repair + cfg.product_refiner_clean_identity_weight * protection,
                               list(model.parameters()), retain_graph=True)
    recomposed = torch.autograd.grad(sum(g["training_total"] for g in grouped.values()) / 4,
                                     list(model.parameters()))
    for actual, expected in zip(recomposed, full):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)


def test_cli_refuses_to_write_inside_frozen_artifacts(frozen):
    path = frozen[0]
    with pytest.raises(FileExistsError):
        a.run(Namespace(output=str(path / "new.json"), state_dir=str(path)))


def test_end_to_end_output_is_nonpublishing_and_rng_is_restored(frozen, tmp_path_factory, monkeypatch):
    path = frozen[0]
    output = tmp_path_factory.mktemp("gradient_audit_output") / "report.json"

    def model_factory(**kwargs):
        torch.rand(5)  # model initialization must not change the caller RNG
        return torch.nn.Module()

    def geometry(model, batch, cfg):
        assert batch["group"].numel() == 192
        return {"group_order": list(a.GROUPS)}

    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", model_factory)
    monkeypatch.setattr(a, "compute_geometry", geometry)
    before = torch.get_rng_state().clone()
    args = Namespace(output=str(output), state_dir=str(path), expected_source_commit=a.LEGACY_COMMIT,
                     transaction_index=0, device="cpu", legacy_core_strength=.02, legacy_transition_strength=1.)
    assert a.run(args) == 0
    assert torch.equal(before, torch.get_rng_state())
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["completed"] and report["optimizer_steps"] == 0
    assert not report["pilot_allowed"] and not report["publish_allowed"] and not report["probe_loaded"]
    with pytest.raises(FileExistsError):
        a.run(args)
