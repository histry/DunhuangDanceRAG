"""Frozen TRAIN layer/activation audit. No optimizer, initialization change or pilot.

Parameter-gradient norms depend on coordinates and activation scale. This report
measures those scales and the actual head VJP; it never diagnoses training history
or authorizes a new architecture from one frozen transaction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from training import motion_models as m
from training import refiner_group_gradient_audit as a


SCHEMA = "refiner_frozen_train_parameter_gradient_audit_v1"


def tensor_stats(value):
    value = value.detach().double()
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise FloatingPointError("empty or nonfinite layer audit tensor")
    norm, maximum, zero_fraction = torch.stack([
        value.norm(), value.abs().max(), (value == 0).double().mean(),
    ]).cpu().tolist()
    return {"numel": value.numel(), "norm": norm, "rms": norm / math.sqrt(value.numel()),
            "abs_max": maximum, "zero_fraction": zero_fraction}


class LayerObserver:
    """Observation-only forward hooks, with VJPs from ordinary autograd.grad.

    The model executes ONCE on concatenated [192 repair, 192 clean] cases.
    Selection below is for reporting only, never a separate loss forward.
    """

    def __init__(self, model, batch, cfg):
        self.model, self.batch, self.cfg = model, batch, cfg
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.trace, self.activations = {}, {}
        self.target_names = []
        self.parameter_stats = {n: {**tensor_stats(p), "shape": list(p.shape)} for n, p in self.named}
        self.report = {"parameters": self.parameter_stats, "forward_activations": {},
                       "gradients": {}, "decoder": {}, "head": {},
                       "notes": [
                           "Gradient/parameter ratios are not optimizer updates; zero denominator is null.",
                           "Norms, RMS and gradient shares depend on parameterization; no starvation verdict.",
                           "Full-transaction gradients are measured directly and checked against group means.",
                           "Activation gradient RMS uses only the named group's cases in each branch, all frames.",
                           "No initial checkpoint or accepted-update history is reconstructed.",
                       ]}

    @contextmanager
    def capture(self):
        handles = []

        def hook(name):
            def observe(module, inputs, output):
                if name + ".output" in self.activations:
                    raise RuntimeError("layer audit requires exactly one model forward")
                if not torch.is_tensor(output) or output.ndim != 3:
                    raise ValueError("layer audit requires [case, channel, frame] tensors")
                self.activations[name + ".output"] = output
                if name in {"in_proj", "out"}:
                    self.activations[name + ".input"] = inputs[0]
            return observe

        try:
            for name, module in self.model.named_modules():
                if name in {"in_proj", "out"} or (name.startswith("net.") and not list(module.children())):
                    handles.append(module.register_forward_hook(hook(name)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.activations.clear()
            self.trace.clear()

    def _select(self, group, branch):
        ids = self.batch["group"]
        selected = torch.ones_like(ids, dtype=torch.bool) if group == "full_transaction" else ids == a.GROUPS.index(group)
        empty = torch.zeros_like(selected)
        return torch.cat([selected, empty] if branch == "repair" else [empty, selected])

    def targets(self):
        if not {"in_proj.input", "out.input", "out.output"} <= self.activations.keys():
            raise ValueError("missing refiner layer observations")
        count = 2 * self.batch["group"].numel()
        if any(value.shape[0] != count for value in self.activations.values()):
            raise ValueError("activation case count does not match repair+clean transaction")
        self.target_names = [n for n, value in self.activations.items() if value.requires_grad]
        for group in (*a.GROUPS, "full_transaction"):
            self.report["forward_activations"][group] = {
                branch: {n: tensor_stats(value[self._select(group, branch)])
                         for n, value in self.activations.items()}
                for branch in ("repair", "clean")}
            selected = self._select(group, "repair")[:count // 2]
            self.report["decoder"][group] = {
                branch: {key: tensor_stats(trace[key][selected])
                         for key in ("raw", "after_mask", "after_smoothing", "after_taper", "applied")
                         if key in trace}
                for branch, trace in self.trace.items() if branch in {"repair", "clean"}}
        weight = self.model.out.weight.detach().cpu().double().squeeze(-1)
        self.report["head"] = {
            "weight": self.parameter_stats["out.weight"], "bias": self.parameter_stats["out.bias"],
            "weight_spectral_norm": float(torch.linalg.matrix_norm(weight, ord=2)),
            "weight_is_exactly_zero": bool((weight == 0).all()),
            "channel_blocks": {
                label: tensor_stats(weight[lo:hi])
                for label, lo, hi in (("contact", 0, 4), ("root", 4, 7), ("joint", 7, 79))},
            "initialization": "production zero head is unchanged; frozen weights are measured, not reinitialized",
        }
        return [self.activations[n] for n in self.target_names]

    def record(self, group, component, parameter_gradients, activation_gradients):
        by_name = dict(zip(self.target_names, activation_gradients))
        parameters = {}
        norm_squares = {"shared_trunk": 0.0, "output_head": 0.0}
        counts = dict.fromkeys(norm_squares, 0)
        for (name, parameter), gradient in zip(self.named, parameter_gradients):
            stats = tensor_stats(torch.zeros_like(parameter) if gradient is None else gradient)
            pn = self.parameter_stats[name]["norm"]
            parameters[name] = {**stats, "connected": gradient is not None,
                                "gradient_to_parameter_norm_ratio": stats["norm"] / pn if pn else None}
            scope = "output_head" if name.startswith("out.") else "shared_trunk"
            norm_squares[scope] += stats["norm"] ** 2
            counts[scope] += parameter.numel()
        norm_square = sum(norm_squares.values())
        scopes = {scope: {"numel": counts[scope], "gradient_norm": math.sqrt(square),
                          "gradient_rms": math.sqrt(square / counts[scope]),
                          "squared_norm_share": square / norm_square if norm_square else None}
                  for scope, square in norm_squares.items()}
        activations = {}
        for branch in ("repair", "clean"):
            selected = self._select(group, branch)
            activations[branch] = {}
            for name in self.target_names:
                value = self.activations[name][selected]
                gradient = by_name[name]
                sliced = torch.zeros_like(value) if gradient is None else gradient[selected]
                stats = tensor_stats(sliced)
                # This product is invariant to reciprocal scalar rescaling of
                # an activation and its gradient; it is not an Adam step size.
                activations[branch][name] = {
                    **stats, "connected": gradient is not None,
                    "gradient_times_activation_rms": tensor_stats(sliced * value.detach())["rms"]}
        output = self.activations["out.output"]
        hidden = self.activations["out.input"]
        gz, gh = by_name["out.output"], by_name["out.input"]
        gz = torch.zeros_like(output) if gz is None else gz
        gh = torch.zeros_like(hidden) if gh is None else gh
        with torch.no_grad():
            expected = F.conv_transpose1d(gz, self.model.out.weight)
            # W^T dL/dz has the same [case, hidden, frame] coordinates as dL/dh.
            error = tensor_stats(gh - expected)["norm"]
            hn, zn = tensor_stats(gh)["norm"], tensor_stats(gz)["norm"]
        self.report["gradients"].setdefault(group, {})[component] = {
            "parameters": parameters, "scopes": scopes, "activations": activations,
            "head_transport": {"hidden_gradient_norm": hn, "output_gradient_norm": zn,
                               "hidden_to_output_gradient_norm_ratio": hn / zn if zn else None,
                               "vjp_absolute_error_norm": error,
                               "vjp_relative_error": error / hn if hn else None}}

    def finish(self, total, targets, expected):
        gradients = torch.autograd.grad(total, targets, retain_graph=False, allow_unused=True)
        size = len(self.named)
        actual = torch.cat([(torch.zeros_like(p) if g is None else g).detach().cpu().double().flatten()
                            for (_, p), g in zip(self.named, gradients[:size])])
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5,
                                   msg="group gradients do not reconstruct the full TRAIN objective")
        self.report["group_reconstruction"] = {
            "verified": True, "absolute_error_norm": float((actual - expected).norm()),
            "actual_full_gradient_norm": float(actual.norm()), "rtol": 2e-4, "atol": 1e-5}
        self.record("full_transaction", "training_total", gradients[:size], gradients[size:])


def compute(model, batch, cfg):
    return a.compute_geometry(model, batch, cfg, observer=LayerObserver(model, batch, cfg))


def print_summary(result):
    """Emit the next review inputs automatically; do not require another run."""
    detail = result["layer_details"]
    full = detail["gradients"]["full_transaction"]["training_total"]
    print("[LAYER AUDIT] No starvation verdict; single frozen TRAIN transaction.")
    print("HEAD:", json.dumps(detail["head"], allow_nan=False))
    print("GROUP RECONSTRUCTION:", json.dumps(detail["group_reconstruction"]))
    print("FULL TRANSACTION SCOPES:", json.dumps(full["scopes"]))
    print("FULL TRANSACTION HEAD VJP:", json.dumps(full["head_transport"]))
    print("PARAMETER: count / parameter_norm / parameter_rms / gradient_norm / gradient_rms / grad_to_param")
    for name, stat in full["parameters"].items():
        p = detail["parameters"][name]
        ratio = stat["gradient_to_parameter_norm_ratio"]
        ratio = "NA" if ratio is None else f"{ratio:.6e}"
        print(f"{name}: {p['numel']} / {p['norm']:.6e} / {p['rms']:.6e} / "
              f"{stat['norm']:.6e} / {stat['rms']:.6e} / {ratio}")
    print("ACTIVATION (full TRAIN, repair branch): activation_rms / gradient_rms / grad_times_activation_rms")
    for name, grad in full["activations"]["repair"].items():
        act = detail["forward_activations"]["full_transaction"]["repair"][name]
        print(f"{name}: {act['rms']:.6e} / {grad['rms']:.6e} / {grad['gradient_times_activation_rms']:.6e}")
    for group in a.GROUPS:
        record = detail["gradients"][group]["training_total"]
        print(f"GROUP {group}:", json.dumps({
            "scopes": record["scopes"], "head_transport": record["head_transport"],
            "clean_vs_repair": {scope: table[group] for scope, table in result["clean_vs_repair"].items()},
            "repair_tangent_rms": {key: value["rms"] for key, value in detail["decoder"][group]["repair"].items()},
        }, allow_nan=False))
    print("[STOP] optimizer_steps=0 probe_loaded=false pilot_allowed=false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transaction-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-core-strength", type=float)
    parser.add_argument("--legacy-transition-strength", type=float)
    args = parser.parse_args()
    status = a.run(args, compute=compute, schema=SCHEMA, audit_file=__file__)
    print_summary(json.loads(Path(args.output).read_text(encoding="utf-8")))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
