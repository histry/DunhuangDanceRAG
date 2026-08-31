import json
from argparse import Namespace

import pytest
import torch

from training import motion_models as m
from training import refiner_group_gradient_audit as groups
from training import refiner_parameter_gradient_audit as layers
from tests.test_refiner_group_gradient_audit import bank_tensor, frozen  # noqa: F401


def sample(device):
    cfg = m.MotionGenerationConfig(device=device, window_len=60)
    part = bank_tensor(cfg)
    batch = {key: torch.cat([value] * 6).to(device) for key, value in part.items()}
    amplitude = torch.tensor([.002, .008, .01, .03], device=device)[batch["group"]]
    phase = torch.linspace(0, 8, cfg.window_len, device=device)
    batch["bad"][..., 4] += amplitude[:, None] * phase.sin()
    model = m.ProductManifoldTemporalRefiner(hidden=4).to(device)
    return model, batch, cfg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_head_blocks_hidden_vjp_but_not_head_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    model, batch, cfg = sample(device)
    # A preexisting mixed mode and .grad must not be changed by observation.
    model.net[0].eval()
    modes = {name: module.training for name, module in model.named_modules()}
    saved = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.full_like(p, 7)
    result = layers.compute(model, batch, cfg)
    detail = result["layer_details"]
    full = detail["gradients"]["full_transaction"]["training_total"]
    assert detail["head"]["weight_is_exactly_zero"]
    assert full["scopes"]["shared_trunk"]["gradient_norm"] == 0
    assert full["scopes"]["output_head"]["gradient_norm"] > 0
    assert full["head_transport"]["hidden_gradient_norm"] == 0
    assert full["head_transport"]["output_gradient_norm"] > 0
    assert full["head_transport"]["vjp_absolute_error_norm"] == 0
    assert full["parameters"]["out.weight"]["gradient_to_parameter_norm_ratio"] is None
    assert full["parameters"]["in_proj.weight"]["connected"]
    assert detail["group_reconstruction"]["verified"]
    assert all(torch.equal(saved[n], p) and torch.equal(p.grad, torch.full_like(p, 7))
               for n, p in model.named_parameters())
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert all(not module._forward_hooks for module in model.modules())
    for branch in ("repair", "clean"):
        assert detail["decoder"]["single_short"][branch]["raw"]["norm"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonzero_head_true_gradient_matches_plain_audit_and_analytic_vjp(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(43)
    model, batch, cfg = sample(device)
    with torch.no_grad():
        model.out.weight.normal_(0, 1e-5)
    plain = groups.compute_geometry(model, batch, cfg)
    detailed = layers.compute(model, batch, cfg)
    assert detailed["losses"] == plain["losses"]
    # CUDA convolution reductions need not be bit-identical across backward
    # traversals requesting different intermediate targets.
    for component in groups.COMPONENTS:
        for scope in ("all_parameters", "shared_trunk", "output_head"):
            left, right = detailed["geometry"][component][scope], plain["geometry"][component][scope]
            torch.testing.assert_close(torch.tensor(left["norms"], dtype=torch.float64),
                                       torch.tensor(right["norms"], dtype=torch.float64), rtol=1e-5, atol=1e-8)
            for row_left, row_right in zip(left["cosine"], right["cosine"]):
                for actual, expected in zip(row_left, row_right):
                    if expected is None:
                        assert actual is None
                    else:
                        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8)
    full = detailed["layer_details"]["gradients"]["full_transaction"]["training_total"]
    assert full["scopes"]["shared_trunk"]["gradient_norm"] > 0
    transport = full["head_transport"]
    assert transport["vjp_relative_error"] < 1e-5
    assert transport["hidden_to_output_gradient_norm_ratio"] <= (
        detailed["layer_details"]["head"]["weight_spectral_norm"] * (1 + 1e-5))
    assert full["parameters"]["out.weight"]["gradient_to_parameter_norm_ratio"] > 0
    # The repair objective may reach only repair activations, even though both
    # branches share the same parameters and forward call.
    repair = detailed["layer_details"]["gradients"]["single_short"]["repair_objective"]
    assert repair["activations"]["clean"]["out.output"]["norm"] == 0
    assert repair["activations"]["repair"]["out.output"]["norm"] > 0


def test_hook_cleanup_on_failed_objective(monkeypatch):
    model, batch, cfg = sample("cpu")
    original = m._refiner_batch_objectives

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure after forward")

    monkeypatch.setattr(m, "_refiner_batch_objectives", broken)
    with pytest.raises(RuntimeError, match="injected failure"):
        layers.compute(model, batch, cfg)
    assert model.training
    assert all(not module._forward_hooks for module in model.modules())


def test_reparameterization_changes_gradient_norms_without_changing_function():
    # Exact linear counterexample to "head/trunk gradient ratio proves learning
    # starvation". Both networks implement z=3*x; coordinates alone differ.
    def measure(scale):
        a = torch.tensor(2. * scale, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(1.5 / scale, dtype=torch.float64, requires_grad=True)
        h = a * 4
        z = w * h
        ga, gw, gh = torch.autograd.grad((z - 1).square(), (a, w, h))
        return z.detach(), abs(ga / gw), gh * h.detach()
    z1, ratio1, product1 = measure(1.)
    z2, ratio2, product2 = measure(100.)
    torch.testing.assert_close(z1, z2, rtol=0, atol=0)
    torch.testing.assert_close(product1, product2)
    torch.testing.assert_close(ratio1 / ratio2, torch.tensor(10000., dtype=torch.float64))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_layer_measurements_fail_closed(value):
    with pytest.raises(FloatingPointError):
        layers.tensor_stats(torch.tensor([value]))


def test_layer_run_preserves_frozen_artifacts_rng_and_nonpublication(frozen, tmp_path_factory, monkeypatch, capsys):
    path, _, state, _ = frozen
    model_type = m.ProductManifoldTemporalRefiner
    model = model_type(hidden=4)
    state["model_state_dict"] = model.state_dict()
    torch.save(state, path / "diagnostic_state.pt")
    original_hashes = {name: groups.file_sha256(path / name)
                       for name in ("diagnostic_state.pt", "diagnostic_report.json", "fit_bank.pt")}
    opened = []
    loader = m._trusted_torch_load

    def track(file, **kwargs):
        opened.append(file.name)
        return loader(file, **kwargs)

    monkeypatch.setattr(m, "_trusted_torch_load", track)
    monkeypatch.setattr(m, "ProductManifoldTemporalRefiner", lambda **kwargs: model_type(hidden=4, **kwargs))
    output = tmp_path_factory.mktemp("layer_audit") / "report.json"
    args = Namespace(output=str(output), state_dir=str(path), expected_source_commit=groups.LEGACY_COMMIT,
                     transaction_index=0, device="cpu", legacy_core_strength=.02, legacy_transition_strength=1.)
    rng = torch.get_rng_state().clone()
    assert groups.run(args, compute=layers.compute, schema=layers.SCHEMA, audit_file=layers.__file__) == 0
    assert torch.equal(rng, torch.get_rng_state())
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == layers.SCHEMA and result["completed"]
    assert result["optimizer_steps"] == 0
    assert not any(result[k] for k in ("probe_loaded", "pilot_allowed", "publish_allowed", "scientific_acceptance"))
    assert result["layer_details"]["group_reconstruction"]["verified"]
    assert set(opened) == {"diagnostic_state.pt", "fit_bank.pt"}
    assert original_hashes == {name: groups.file_sha256(path / name) for name in original_hashes}
    layers.print_summary(result)
    assert "FULL TRANSACTION HEAD VJP" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        groups.run(args, compute=layers.compute, schema=layers.SCHEMA)
