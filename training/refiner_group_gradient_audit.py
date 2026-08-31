"""Read-only gradient geometry on one frozen TRAIN reservoir transaction.

No optimizer, training step, probe loading, checkpoint selection or pilot gate.
The negative V15.5 binary-backward experiment is not a compatible source.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import dataclasses
import json
import os
from pathlib import Path

import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training.refiner_diagnostics import file_sha256


SCHEMA = "refiner_frozen_train_group_gradient_audit_v1"
LEGACY_COMMIT = "6e73e0eda9f349d3a611864f4719b22807ee5952"
GROUPS = m.REFINER_GROUP_LABELS
STRENGTH_ENV = ("MOTION_REFINER_CORE_STRENGTH", "MOTION_REFINER_TRANSITION_STRENGTH")
ENV_PREFIXES = ("GROUNDING_", "PHYSICAL_", "CONTACT_")
COMPONENTS = ("repair_objective", "endpoint_deficit_mean", "temporal_deficit_mean",
              "clean_identity", "training_total")


@contextmanager
def frozen_environment(fingerprint, strengths):
    """Restore recorded physical policy; never apply the caller's overrides."""
    recorded = fingerprint["mask_and_physical_environment"]
    if not isinstance(recorded, dict) or any(
        not k.startswith(ENV_PREFIXES) or not isinstance(v, str)
        for k, v in recorded.items()
    ):
        raise ValueError("invalid frozen physical environment")
    keys = {k for k in os.environ if k.startswith(ENV_PREFIXES)} | set(recorded) | set(STRENGTH_ENV)
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(recorded)
        for key, field in zip(STRENGTH_ENV, ("core", "transition")):
            os.environ[key] = str(strengths[field])
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _validate_bank(batch, cfg):
    required = {"clean", "bad", "seam", "cond", "joint", "root", "contact",
                "clean_joint", "clean_root", "clean_contact", "group"}
    if set(batch) not in (required, required | {"clean_cond"}):
        raise ValueError("TRAIN bank fields mismatch; probe/unknown tensors forbidden")
    frames = int(cfg.window_len)
    for key, tensor in batch.items():
        if not torch.is_tensor(tensor) or tensor.shape[0] != 32:
            raise ValueError(f"invalid 32-case TRAIN bank tensor: {key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite TRAIN tensor: {key}")
    shapes = {"clean": (32, frames, 151), "bad": (32, frames, 151),
              "seam": (32, frames, 1), "group": (32,)}
    for key in ("joint", "clean_joint"):
        shapes[key] = (32, frames, 24)
    for key in ("root", "clean_root"):
        shapes[key] = (32, frames, 1)
    for key in ("contact", "clean_contact"):
        # A frame-level contact confidence is broadcast to the four logits.
        shapes[key] = (32, frames, 1)
    if any(tuple(batch[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError("TRAIN bank shape mismatch")
    for key in ("cond", "clean_cond"):
        if key in batch and tuple(batch[key].shape) not in {(32, 32), (32, frames, 32)}:
            raise ValueError("TRAIN conditioning shape mismatch")
    expected = torch.tensor([0, 1] * 8 + [2, 3] * 8, device=batch["group"].device)
    if not torch.equal(batch["group"], expected):
        raise ValueError("TRAIN role/width order mismatch")
    widths = (batch["seam"][..., 0] >= .5).sum(1)
    if not torch.equal(widths, torch.where(expected % 2 == 0, 10, 28)):
        raise ValueError("TRAIN group labels do not match seam widths")


def load_frozen_source(source, expected_commit, *,
                       legacy_core_strength=None, legacy_transition_strength=None):
    """Validate artifact provenance once; never load the probe artifact."""
    source = Path(source)
    report_path, state_path, bank_path = (source / name for name in
        ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt"))
    source_hashes = {p.name: file_sha256(p) for p in (report_path, state_path, bank_path)}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = m._trusted_torch_load(state_path, map_location="cpu")
    bank = m._trusted_torch_load(bank_path, map_location="cpu")
    fingerprint = state["fingerprint"]
    if fingerprint != report["fingerprint"] or fingerprint != bank["fingerprint"]:
        raise ValueError("state/report/TRAIN bank fingerprint mismatch")
    if fingerprint.get("code_revision") != expected_commit:
        raise ValueError("unexpected frozen source commit")
    legacy = report.get("schema") == "refiner_observable_bridge_diagnostic_v15_4_1"
    if legacy:
        if expected_commit != LEGACY_COMMIT or fingerprint.get("refiner_tangent_gradient_protocol") is not None:
            raise ValueError("not the reviewed V15.4.1 true-gradient source")
    elif (report.get("schema") != d.SCHEMA or fingerprint.get("refiner_tangent_gradient_protocol")
          != m.REFINER_TANGENT_GRADIENT_PROTOCOL):
        raise ValueError("incompatible backward protocol; V15.5 is rejected")
    expected_protocols = {
        "model_version": m.REFINER_MODEL_VERSION,
        "refiner_input_protocol": m.REFINER_INPUT_PROTOCOL,
        "observable_objective_protocol": m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL,
        "refiner_batch_aggregation_protocol": m.REFINER_BATCH_AGGREGATION_PROTOCOL,
        "repair_safety_protocol": m.REFINER_REPAIR_SAFETY_PROTOCOL,
        "refiner_update_protocol": m.REFINER_UPDATE_PROTOCOL,
        "fit_protocol": d.FIT_PROTOCOL,
        "context_reservoir_protocol": d.CONTEXT_RESERVOIR_PROTOCOL,
    }
    if any(fingerprint.get(k) != v for k, v in expected_protocols.items()):
        raise ValueError("frozen model/objective/safety/reservoir protocol mismatch")
    if (state.get("schema") != "refiner_diagnostic_state_v1"
            or state.get("formal_checkpoint") is not False or state.get("publish_allowed") is not False
            or state.get("completed_steps") != 400 or report.get("completed_steps") != 400
            or report.get("target_steps") != 400 or report.get("completed") is not True
            or report.get("stopped_early") is not False or report.get("published") is not False):
        raise ValueError("requires a complete, unpublished 400-step diagnostic state")
    if (bank.get("schema") != "refiner_train_safe_start_context_reservoir_v4"
            or bank.get("train_only") is not True or bank.get("formal_checkpoint") is not False
            or bank.get("publish_allowed") is not False):
        raise ValueError("requires the TRAIN-only reservoir artifact, never probe_bank.pt")
    artifact = state["fit_bank_artifact"]
    if (artifact != report["fit_bank_artifact"] or artifact.get("file") != "fit_bank.pt"
            or artifact.get("sha256") != file_sha256(bank_path)):
        raise ValueError("TRAIN artifact hash/descriptor mismatch")
    if bank.get("windows") != report.get("windows") or len(bank["windows"]) != 8:
        raise ValueError("frozen TRAIN windows mismatch")
    config = dict(bank["config"])
    if legacy:
        if legacy_core_strength is None or legacy_transition_strength is None:
            raise ValueError("legacy artifact omitted decoder strengths: supply both explicitly")
        strengths = {"core": float(legacy_core_strength), "transition": float(legacy_transition_strength)}
        strength_evidence = "explicit_legacy_values_not_recorded_in_source_fingerprint"
    else:
        strengths = fingerprint["refiner_decode_strengths"]
        if any(config.get(f"refiner_{k}_strength") != strengths[k] for k in ("core", "transition")):
            raise ValueError("serialized config and decoder strength fingerprint mismatch")
        strength_evidence = "source_fingerprint_and_config"
    config.update(refiner_core_strength=strengths["core"], refiner_transition_strength=strengths["transition"])
    cfg = m.MotionGenerationConfig(**config)
    with frozen_environment(fingerprint, strengths):
        m._refiner_decode_strengths(cfg)
        if dataclasses.asdict(m.StageAcceptancePolicy.from_environment()) != fingerprint["stage_acceptance_policy"]:
            raise ValueError("frozen physical environment does not reproduce stage policy")
    contract = d.fit_bank_contract(8, cfg)
    if bank.get("contract") != contract or report.get("fit_bank") != contract:
        raise ValueError("TRAIN reservoir contract mismatch")
    reservoir = bank["context_reservoir"]
    count = len(reservoir)
    if set(reservoir) != {str(i) for i in range(count)} or count != contract["context_reservoir_cycle_length"]:
        raise ValueError("TRAIN reservoir indices/count mismatch")
    expected_schedule = ([list(range(5))] if count == 5 else
                         [[(i + j) % count for j in range(5)] for i in range(count)])
    if [list(row) for row in bank["transaction_schedule"]] != expected_schedule:
        raise ValueError("TRAIN transaction schedule mismatch")
    metadata = {
        "source_commit": expected_commit, "source_schema": report["schema"],
        "source_completed_steps": 400,
        "decoder_strengths": strengths, "decoder_strength_evidence": strength_evidence,
        "source_sha256": source_hashes,
        "source_directory": str(source.resolve()),
    }
    return state, bank, cfg, metadata


def materialize_transaction(bank, cfg, transaction_index):
    """Build only the current validated 192-case TRAIN transaction."""
    schedule = bank["transaction_schedule"]
    if not 0 <= transaction_index < len(schedule):
        raise ValueError("transaction index outside frozen schedule")
    parts = [bank["anchor"]] + [bank["context_reservoir"][str(i)] for i in schedule[transaction_index]]
    for part in parts:
        _validate_bank(part, cfg)
    if any(set(part) != set(parts[0]) for part in parts[1:]):
        raise ValueError("TRAIN banks must have identical fields; cannot drop clean_cond")
    batch = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
    if batch["group"].numel() != 192:
        raise ValueError("frozen transaction must have exactly 192 cases")
    return batch


def load_transaction(source, expected_commit, transaction_index, *,
                     legacy_core_strength=None, legacy_transition_strength=None):
    state, bank, cfg, metadata = load_frozen_source(
        source, expected_commit, legacy_core_strength=legacy_core_strength,
        legacy_transition_strength=legacy_transition_strength)
    batch = materialize_transaction(bank, cfg, transaction_index)
    metadata.update(transaction_index=transaction_index,
                    context_indices=list(bank["transaction_schedule"][transaction_index]),
                    cases=192, cases_per_group=48)
    return state, batch, cfg, metadata


def _cosine_table(vectors):
    norms = [float(g.norm()) for g in vectors]
    return {"norms": norms, "cosine": [
        [None if norms[i] == 0 or norms[j] == 0 else
         max(-1.0, min(1.0, float(a.dot(b)) / (norms[i] * norms[j])))
         for j, b in enumerate(vectors)] for i, a in enumerate(vectors)]}


def compute_geometry(model, batch, cfg, *, observer=None):
    """Unclipped true parameter gradients; autograd.grad never populates .grad.

    Equal 48-case groups make the mean of training_total gradients equal to
    the full 192-case objective gradient (including group CVaR and clean loss).
    Endpoint/temporal means are diagnostics, not an additive CVaR decomposition.
    """
    if batch["group"].shape != (192,) or any(int((batch["group"] == i).sum()) != 48 for i in range(4)):
        raise ValueError("gradient geometry requires four 48-case TRAIN groups")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    states = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    modes = {module: module.training for module in model.modules()}
    scopes = {"all_parameters": [True] * len(named),
              "shared_trunk": [not n.startswith("out.") for n, _ in named],
              "output_head": [n.startswith("out.") for n, _ in named]}
    vectors = {component: {scope: [] for scope in scopes} for component in COMPONENTS}
    values = {}
    try:
        model.eval()
        with torch.enable_grad(), (nullcontext() if observer is None else observer.capture()):
            grouped = {}
            kwargs = {} if observer is None else {"trace": observer.trace}
            full_objectives = m._refiner_batch_objectives(model, batch, cfg, group_objectives=grouped, **kwargs)
            if set(grouped) != set(GROUPS):
                raise ValueError("incomplete full-transaction group objectives")
            targets = parameters + ([] if observer is None else observer.targets())
            for group, label in enumerate(GROUPS):
                objectives = grouped[label]
                if set(objectives) != set(COMPONENTS):
                    raise ValueError("incomplete group objective components")
                values[label] = {}
                for component, objective in objectives.items():
                    if not bool(torch.isfinite(objective)):
                        raise FloatingPointError(f"nonfinite {label}/{component} objective")
                    gradients = (torch.autograd.grad(objective, targets, retain_graph=True, allow_unused=True)
                                 if objective.requires_grad else [None] * len(targets))
                    if observer is not None:
                        observer.record(label, component, gradients[:len(parameters)], gradients[len(parameters):])
                    flat = [(torch.zeros_like(p) if g is None else g).detach().cpu().double().reshape(-1)
                            for p, g in zip(parameters, gradients)]
                    if not all(bool(torch.isfinite(g).all()) for g in flat):
                        raise FloatingPointError(f"nonfinite {label}/{component} gradient")
                    for scope, included in scopes.items():
                        items = [g for g, keep in zip(flat, included) if keep]
                        vectors[component][scope].append(torch.cat(items) if items else torch.empty(0, dtype=torch.float64))
                    values[label][component] = float(objective.detach())
                del objectives, objective, gradients
            if observer is not None:
                total = full_objectives[0] + cfg.product_refiner_clean_identity_weight * full_objectives[1]
                expected = torch.stack(vectors["training_total"]["all_parameters"]).mean(0)
                observer.finish(total, targets, expected)
        if any(not torch.equal(states[name], tensor) for name, tensor in model.state_dict().items()):
            raise RuntimeError("read-only gradient audit changed model state")
    finally:
        for module, mode in modes.items():
            module.training = mode
    result = {
        "group_order": list(GROUPS), "losses": values,
        "geometry": {key: {scope: _cosine_table(v) for scope, v in by_scope.items()}
                     for key, by_scope in vectors.items()},
        "mean_training_gradient_norm": float(torch.stack(vectors["training_total"]["all_parameters"]).mean(0).norm()),
        "component_note": "endpoint/temporal are unweighted deficit means, not additive CVaR contributions",
        "before_clipping": True, "zero_gradient_cosine": None,
    }
    weight = float(cfg.product_refiner_clean_identity_weight)
    if not 0 <= weight < float("inf"):
        raise ValueError("clean loss weight must be finite and nonnegative")
    result["clean_loss_weight"] = weight
    # Within-group clean-to-clean cosines cannot answer clean-vs-repair conflict.
    result["clean_vs_repair"] = {}
    for scope in scopes:
        repairs = vectors["repair_objective"][scope]
        cleans = vectors["clean_identity"][scope]
        result["clean_vs_repair"][scope] = {
            label: _clean_repair_pair(repair, clean * weight)
            for label, repair, clean in zip(GROUPS, repairs, cleans)
        }
        result["clean_vs_repair"][scope]["full_transaction"] = _clean_repair_pair(
            torch.stack(repairs).mean(0), weight * torch.stack(cleans).mean(0))
    if observer is not None:
        result["layer_details"] = observer.report
    return result


def _clean_repair_pair(repair, weighted_clean):
    table = _cosine_table([repair, weighted_clean])
    rn, cn = table["norms"]
    combined = float((repair + weighted_clean).norm())
    return {"repair_norm": rn, "weighted_clean_norm": cn,
            "weighted_clean_to_repair_ratio": cn / rn if rn else None,
            "cosine": table["cosine"][0][1], "combined_norm": combined,
            "combined_to_sum_norm_ratio": combined / (rn + cn) if rn + cn else None}


def run(args, *, compute=None, schema=SCHEMA, audit_file=__file__):
    output, source = Path(args.output).resolve(), Path(args.state_dir).resolve()
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError("write a new audit JSON outside the frozen source directory")
    state, batch, cfg, metadata = load_transaction(
        source, args.expected_source_commit, args.transaction_index,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg.device = str(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), frozen_environment(state["fingerprint"], metadata["decoder_strengths"]):
        model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        result = (compute or compute_geometry)(model, batch, cfg)
    for name, digest in metadata["source_sha256"].items():
        if file_sha256(source / name) != digest:
            raise RuntimeError("frozen artifact changed during read-only audit")
    result.update(schema=schema, completed=True, source=metadata,
                  runtime_commit=m._training_code_revision(), device=str(device),
                  tangent_gradient_protocol=m.REFINER_TANGENT_GRADIENT_PROTOCOL,
                  implementation_sha256={"audit": file_sha256(audit_file),
                      "group_audit": file_sha256(__file__), "motion_models": file_sha256(m.__file__)},
                  optimizer_steps=0, probe_loaded=False, scientific_acceptance=False,
                  publish_allowed=False, pilot_allowed=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    print(json.dumps({"stage": "group_gradient_audit_complete", "schema": schema, "output": str(output),
                      "optimizer_steps": 0, "probe_loaded": False, "pilot_allowed": False}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transaction-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-core-strength", type=float)
    parser.add_argument("--legacy-transition-strength", type=float)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
