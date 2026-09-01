"""Fresh exact-zero Refiner optimization trajectory; never Pilot or formal training.

One fresh A0 model, the frozen TRAIN reservoir, and exactly 400 checked updates.
Every gradient is ordinary autograd from the unchanged scalar training objective.
The probe remains unopened until the final state and hashes are fixed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import refiner_group_gradient_audit as a
from training import refiner_parameter_gradient_audit as layers
from training import refiner_safe_start_diagnostics as safe
from training.refiner_optimizer import checked_refiner_step, record_update, validate_update_summary


SCHEMA = "refiner_zero_start_trajectory_v1"
MODEL_VERSION = "refiner_zero_start_trajectory_diagnostic_only_v1"
ARM = "A0_zero"
STEPS = 400
SNAPSHOT_STEPS = (0, 1, 2, 3, 5, 10, 25, 50, 100, 200, 300, 400)
TRAJECTORY_STEPS = (1, 2, 5, 10, 25, 50, 100, 200, 300, 400)
EARLY_GRADIENT_STEPS = (1, 2, 3, 5, 10)
LAYER_PREFIXES = ("in_proj", "net.0", "net.1", "net.3", "net.4", "net.6", "net.7")
HEAD_BLOCKS = (("contact", 0, 4), ("root", 4, 7), ("joint", 7, 79))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def fresh_zero_state(cfg, seed):
    """Create a CPU state without consuming caller CPU/CUDA RNG streams."""
    if not 0 <= int(seed) < 2**32:
        raise ValueError("initialization seed must be in [0,2**32)")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        model = m.ProductManifoldTemporalRefiner(fps=cfg.fps, output_init_std=0.0)
        state = d._cpu_tree(model.state_dict())
    if bool((state["out.weight"] != 0).any()) or bool((state["out.bias"] != 0).any()):
        raise RuntimeError("fresh A0 output head is not exactly zero")
    return state


def save_state(path, model, step, experiment_hash):
    m._atomic_torch_save({
        "schema": SCHEMA,
        "version": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "formal_checkpoint": False,
        "publish_allowed": False,
        "pilot_allowed": False,
        "resume_allowed": False,
        "arm": ARM,
        "completed_steps": int(step),
        "target_steps": STEPS,
        "experiment_sha256": experiment_hash,
        "model_state_dict": d._cpu_tree(model.state_dict()),
    }, path)


def _scope_parameter_norms(model):
    squares = {"shared_trunk": 0.0, "output_head": 0.0}
    counts = Counter()
    for name, parameter in model.named_parameters():
        scope = "output_head" if name.startswith("out.") else "shared_trunk"
        value = parameter.detach().double()
        squares[scope] += float((value * value).sum())
        counts[scope] += parameter.numel()
    return {scope: {"numel": counts[scope], "parameter_norm": math.sqrt(square),
                    "parameter_rms": math.sqrt(square / counts[scope])}
            for scope, square in squares.items()}


def _aggregate_layers(model, measurements):
    current = dict(model.named_parameters())
    result = {}
    for prefix in LAYER_PREFIXES:
        names = [name for name in current if name == prefix or name.startswith(prefix + ".")]
        if not names:
            raise RuntimeError(f"missing trajectory layer {prefix}")
        rows = [measurements[name] for name in names]
        parameter_norm = _norm(float(current[name].detach().double().norm()) for name in names)
        numel = sum(current[name].numel() for name in names)
        result[prefix] = {
            "parameters": names,
            "numel": numel,
            "parameter_norm": parameter_norm,
            "parameter_rms": parameter_norm / math.sqrt(numel),
            "gradient_norm_before_clip": _norm(row["gradient_norm_before_clip"] for row in rows),
            "gradient_rms_before_clip": math.sqrt(sum(row["gradient_norm_before_clip"] ** 2 for row in rows) / numel),
            "actual_update_norm": _norm(row["actual_update_norm"] for row in rows),
            "displacement_from_initial_norm": _norm(row["displacement_from_initial_norm"] for row in rows),
            "true_gradient_dot_actual_update": sum(row["true_gradient_dot_actual_update"] for row in rows),
        }
    return result


def _head_evolution(model, before):
    weight = model.out.weight.detach().double()
    bias = model.out.bias.detach().double()
    old_weight = before["out.weight"].double()
    old_bias = before["out.bias"].double()
    result = {
        "weight": {**layers.tensor_stats(weight), "shape": list(weight.shape),
                   "spectral_norm": float(torch.linalg.matrix_norm(weight.squeeze(-1).cpu(), ord=2))},
        "bias": {**layers.tensor_stats(bias), "shape": list(bias.shape)},
        "weight_is_exactly_zero": bool((weight == 0).all()),
        "blocks": {},
    }
    for label, lo, hi in HEAD_BLOCKS:
        parameter = torch.cat([weight[lo:hi].reshape(-1), bias[lo:hi].reshape(-1)])
        update = torch.cat([(weight - old_weight)[lo:hi].reshape(-1),
                            (bias - old_bias)[lo:hi].reshape(-1)])
        result["blocks"][label] = {
            "parameter_norm": float(parameter.norm()),
            "parameter_rms": float(parameter.norm()) / math.sqrt(parameter.numel()),
            "actual_update_norm": float(update.norm()),
            "numel": parameter.numel(),
        }
    return result


def _compact_optimizer(update):
    keys = (
        "protocol", "optimizer_update_accepted", "direction", "step_scale", "reason",
        "loss_before", "loss_after", "minimum_loss_decrease", "armijo_factor",
        "trial_evaluations", "loss_rejected_trials", "nonfinite_trials",
        "insufficient_decrease_trials", "used_gradient_rescue", "adam_directional_derivative",
        "group_guard_enabled", "group_guard_before", "group_guard_after",
        "group_guard_rejected_trials", "group_guard_last_violations", "gradient_unscale",
    )
    result = {key: update.get(key) for key in keys}
    result.update(retained=bool(update["optimizer_update_accepted"]),
                  rolled_back=not bool(update["optimizer_update_accepted"]))
    return result


def compact_step(model, before, gradients, initial, update, repair, clean, total, terms,
                 batch, bank, step, clip_norm):
    measured = safe.step_measurements(model, before, gradients, initial)
    scopes = measured["scopes"]
    parameter_norms = _scope_parameter_norms(model)
    scopes.update({scope: {**row, **parameter_norms[scope]}
                   for scope, row in scopes.items()})
    trunk, head = scopes["shared_trunk"], scopes["output_head"]
    total_gradient = math.sqrt(trunk["gradient_norm_before_clip"] ** 2
                               + head["gradient_norm_before_clip"] ** 2)
    if not math.isclose(total_gradient, float(clip_norm), rel_tol=2e-5, abs_tol=1e-7):
        raise RuntimeError("saved raw gradients do not reconstruct the pre-clipping norm")
    row = {
        "step": int(step),
        "state_position": "after_checked_step",
        "transaction_index": (step - 1) % len(bank["transaction_schedule"]),
        "context_indices": list(bank["transaction_schedule"][(step - 1) % len(bank["transaction_schedule"])]),
        "cases": len(batch["group"]),
        "cases_per_group": {a.GROUPS[i]: int((batch["group"] == i).sum()) for i in range(4)},
        "objective_before": {
            "repair": float(repair.detach()),
            "clean": float(clean.detach()),
            "training_total": float(total.detach()),
            "group_repair_terms": {key: float(value.detach()) for key, value in terms.items()
                                   if key.startswith("group_")},
        },
        "optimizer": _compact_optimizer(update),
        "gradient_before_clipping": {
            "verified_pre_clip": True,
            "clip_norm_before": float(clip_norm),
            "total_norm": total_gradient,
            "shared_trunk_norm": trunk["gradient_norm_before_clip"],
            "shared_trunk_rms": trunk["gradient_rms_before_clip"],
            "output_head_norm": head["gradient_norm_before_clip"],
            "output_head_rms": head["gradient_rms_before_clip"],
            "trunk_to_head_norm_ratio": _ratio(trunk["gradient_norm_before_clip"],
                                                head["gradient_norm_before_clip"]),
        },
        "actual_retained_update_after_rollback": scopes,
        "parameter_updates": measured["layers"],
        "head": _head_evolution(model, before),
        "tracked_layers": _aggregate_layers(model, measured["layers"]),
        "measurement_note": "Raw autograd is pre-clipping; updates are measured after checked-step acceptance/rollback and include AdamW decay.",
    }
    return row


@contextmanager
def _head_activations(model):
    observed = {}

    def hook(module, inputs, output):
        if observed:
            raise RuntimeError("detailed trajectory snapshot requires exactly one model forward")
        observed["hidden"] = inputs[0]
        observed["output"] = output

    handle = model.out.register_forward_hook(hook)
    try:
        yield observed
    finally:
        handle.remove()


def detailed_snapshot(model, batch, cfg, initial, step, compact=None):
    """Read-only post-step layer/VJP/decoder detail on that step's TRAIN batch."""
    states = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = {module: module.training for module in model.modules()}
    named = list(model.named_parameters())
    trace = {}
    try:
        model.eval()
        with torch.enable_grad(), _head_activations(model) as activation:
            repair, clean, terms, _ = m._refiner_batch_objectives(model, batch, cfg, trace=trace)
            total = repair + cfg.product_refiner_clean_identity_weight * clean
            targets = [parameter for _, parameter in named] + [activation["hidden"], activation["output"]]
            gradients = torch.autograd.grad(total, targets, allow_unused=True)
        parameter_gradients = gradients[:len(named)]
        hidden_gradient, output_gradient = gradients[-2:]
        hidden_gradient = torch.zeros_like(activation["hidden"]) if hidden_gradient is None else hidden_gradient
        output_gradient = torch.zeros_like(activation["output"]) if output_gradient is None else output_gradient
        expected = F.conv_transpose1d(output_gradient, model.out.weight)
        hn, zn = float(hidden_gradient.double().norm()), float(output_gradient.double().norm())
        vjp_error = (hidden_gradient - expected).detach().double().norm().item()
        records = {}
        for (name, parameter), gradient in zip(named, parameter_gradients):
            gradient = torch.zeros_like(parameter) if gradient is None else gradient
            current = parameter.detach().double()
            displacement = current - initial[name].double()
            records[name] = {
                "parameter": layers.tensor_stats(current),
                "gradient": layers.tensor_stats(gradient),
                "displacement_from_initial_norm": float(displacement.norm()),
                "actual_update_norm": (0.0 if compact is None else
                                       compact["parameter_updates"][name]["actual_update_norm"]),
                "training_step_preupdate_gradient_dot_actual_update": (None if compact is None else
                    compact["parameter_updates"][name]["true_gradient_dot_actual_update"]),
            }
        decoder = {key: layers.tensor_stats(value) for key, value in trace["repair"].items()
                   if key in {"raw", "after_mask", "after_smoothing", "after_taper", "applied"}}
        result = {
            "schema": "refiner_zero_start_detailed_snapshot_v1",
            "step": int(step),
            "state_position": "initialization" if step == 0 else "after_checked_step",
            "gradient_position": "same_checkpoint_state_on_named_train_transaction",
            "objective": {"repair": float(repair.detach()), "clean": float(clean.detach()),
                          "training_total": float(total.detach())},
            "parameters": records,
            "head": (compact["head"] if compact is not None else
                     _head_evolution(model, {name: parameter.detach() for name, parameter in named})),
            "head_transport": {
                "hidden_gradient_norm": hn,
                "output_gradient_norm": zn,
                "hidden_to_output_gradient_norm_ratio": _ratio(hn, zn),
                "vjp_absolute_error_norm": vjp_error,
                "vjp_relative_error": _ratio(vjp_error, hn),
            },
            "repair_tangent": decoder,
            "optimizer_step": compact,
            "formal_checkpoint": False,
            "publish_allowed": False,
            "pilot_allowed": False,
        }
        if step == 0 and (result["head"]["weight"]["norm"] != 0
                          or any(record["gradient"]["norm"] != 0
                                 for name, record in records.items() if not name.startswith("out."))):
            raise RuntimeError("exact-zero step-0 trunk gradient contract failed")
        return result
    finally:
        for module, mode in modes.items():
            module.training = mode
        if any(not torch.equal(states[name], value) for name, value in model.state_dict().items()):
            raise RuntimeError("detailed trajectory snapshot changed model state")


def trajectory_summary(rows, optimizer_summary):
    by_step = {row["step"]: row for row in rows}
    first_head = next((row["step"] for row in rows
                       if row["optimizer"]["retained"] and not row["head"]["weight_is_exactly_zero"]), None)
    first_gradient = next((row["step"] for row in rows
                           if row["gradient_before_clipping"]["shared_trunk_norm"] > 0), None)
    first_update = next((row["step"] for row in rows
                         if row["actual_retained_update_after_rollback"]["shared_trunk"]["actual_update_norm"] > 0), None)
    trajectory = {}
    for step in TRAJECTORY_STEPS:
        if step not in by_step:
            continue
        row = by_step[step]
        trunk = row["actual_retained_update_after_rollback"]["shared_trunk"]
        head = row["actual_retained_update_after_rollback"]["output_head"]
        trajectory[str(step)] = {
            "trunk_to_head_gradient_norm_ratio": row["gradient_before_clipping"]["trunk_to_head_norm_ratio"],
            "trunk_to_head_actual_update_norm_ratio": _ratio(trunk["actual_update_norm"], head["actual_update_norm"]),
            "trunk_to_head_displacement_norm_ratio": _ratio(trunk["displacement_from_initial_norm"],
                                                             head["displacement_from_initial_norm"]),
        }
    accepted = optimizer_summary["accepted_steps"]
    rolled_back = optimizer_summary["retained_steps"]  # historical optimizer name means rejected/rolled back
    return {
        "zero_detection": "exact floating-point comparison; no epsilon or scientific threshold",
        "first_nonzero_head_step": first_head,
        "first_nonzero_trunk_gradient_step": first_gradient,
        "first_nonzero_trunk_update_step": first_update,
        "early_trunk_gradient_norms": {str(step): by_step[step]["gradient_before_clipping"]["shared_trunk_norm"]
                                       for step in EARLY_GRADIENT_STEPS if step in by_step},
        "head_trunk_trajectory": trajectory,
        "cumulative_retained_movement": {
            scope: sum(row["actual_retained_update_after_rollback"][scope]["actual_update_norm"] for row in rows)
            for scope in ("shared_trunk", "output_head")},
        "final_displacement_from_initial": {
            scope: rows[-1]["actual_retained_update_after_rollback"][scope]["displacement_from_initial_norm"]
            for scope in ("shared_trunk", "output_head")},
        "optimizer_steps": {
            "attempted": optimizer_summary["attempted_steps"],
            "accepted": accepted,
            "retained": accepted,
            "rolled_back": rolled_back,
            "accepted_rate": accepted / optimizer_summary["attempted_steps"],
            "rolled_back_rate": rolled_back / optimizer_summary["attempted_steps"],
            "legacy_optimizer_counter_note": "optimizer_summary.retained_steps means rejected steps restored by rollback",
        },
    }


def train_trajectory(model, initial_cpu, bank, cfg, destination, experiment_hash, *, steps=STEPS):
    if not 1 <= int(steps) <= STEPS:
        raise ValueError("trajectory updates must be within the fixed 400-step diagnostic budget")
    initial = {name: value.to(cfg.device) for name, value in initial_cpu.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    optimizer_summary = {}
    rows = []
    destination = Path(destination)
    snapshots = destination / "snapshots"
    snapshots.mkdir()
    updates_path = destination / "updates.jsonl"
    updates_path.touch(exist_ok=False)
    model.train()
    initial_batch = {key: value.to(cfg.device) for key, value in a.materialize_transaction(bank, cfg, 0).items()}
    detail0 = detailed_snapshot(model, initial_batch, cfg, initial, 0)
    _exclusive_json(snapshots / "step_000.json", detail0)
    snapshot_files = {"0": {"file": "snapshots/step_000.json",
                             "sha256": a.file_sha256(snapshots / "step_000.json")}}
    started = time.perf_counter()
    for step in range(1, int(steps) + 1):
        index = (step - 1) % len(bank["transaction_schedule"])
        batch = {key: value.to(cfg.device) for key, value in a.materialize_transaction(bank, cfg, index).items()}
        repair, clean, terms, _ = m._refiner_batch_objectives(model, batch, cfg)
        loss = repair + cfg.product_refiner_clean_identity_weight * clean
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        update = checked_refiner_step(
            optimizer, loss,
            lambda: m._refiner_guarded_total_batch_loss(model, batch, cfg, require_all_groups=True),
            gradient_unscale=max(1.0, clip_norm + 1e-6),
            group_guard_before=m._refiner_group_repair_losses(terms, require_all=True),
            group_guard_relative_tolerance=cfg.product_refiner_group_guard_relative_tolerance,
            group_guard_absolute_tolerance=cfg.product_refiner_group_guard_absolute_tolerance)
        record_update(optimizer_summary, update)
        row = compact_step(model, before, gradients, initial, update, repair, clean, loss, terms,
                           batch, bank, step, clip_norm)
        if step == 1 and row["gradient_before_clipping"]["shared_trunk_norm"] != 0:
            raise RuntimeError("fresh exact-zero head did not block the true step-1 trunk gradient")
        if not update["optimizer_update_accepted"] and any(
                item["actual_update_norm"] != 0 for item in row["actual_retained_update_after_rollback"].values()):
            raise RuntimeError("rolled-back trajectory step changed parameters")
        rows.append(row)
        with updates_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if step in SNAPSHOT_STEPS:
            detail = detailed_snapshot(model, batch, cfg, initial, step, compact=row)
            path = snapshots / f"step_{step:03d}.json"
            _exclusive_json(path, detail)
            snapshot_files[str(step)] = {"file": f"snapshots/{path.name}", "sha256": a.file_sha256(path)}
            save_state(destination / "diagnostic_latest.pt", model, step, experiment_hash)
            print(json.dumps({"stage": "zero_start_trajectory", "step": step,
                              "gradient": row["gradient_before_clipping"],
                              "actual_update": row["actual_retained_update_after_rollback"],
                              "head": row["head"], "optimizer_updates": optimizer_summary,
                              "elapsed_seconds": time.perf_counter() - started}, allow_nan=False), flush=True)
    if int(steps) not in SNAPSHOT_STEPS:
        save_state(destination / "diagnostic_latest.pt", model, int(steps), experiment_hash)
    validate_update_summary(optimizer_summary, int(steps))
    expected_snapshots = {str(step) for step in SNAPSHOT_STEPS if step <= int(steps)}
    if set(snapshot_files) != expected_snapshots:
        raise RuntimeError("trajectory detailed snapshot set is incomplete")
    return {
        "completed_steps": int(steps),
        "optimizer_summary": optimizer_summary,
        "trajectory": trajectory_summary(rows, optimizer_summary),
        "snapshot_artifacts": snapshot_files,
        "final_training_transaction": rows[-1],
        "final_state_sha256": safe.state_hash(model.state_dict()),
        "final_checkpoint_sha256": a.file_sha256(destination / "diagnostic_latest.pt"),
    }


def run(args):
    source, output = Path(args.state_dir).resolve(), Path(args.out_dir).resolve()
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError("use a new trajectory directory outside the frozen source")
    state, bank, cfg, metadata = a.load_frozen_source(
        source, a.LEGACY_COMMIT, legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    rows = safe.train_banks(bank, cfg)
    if not (source / "probe_bank.pt").is_file():
        raise FileNotFoundError("final evaluation requires the original probe_bank.pt; never regenerate it")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    # The source config fixes the only initialization seed; there is no CLI
    # seed knob that could turn this diagnostic into a seed search.
    seed = int(cfg.seed)
    experiment = {
        "schema": SCHEMA, "arm": ARM, "output_init_std": 0.0, "initialization_seed": seed,
        "steps": STEPS, "cases_per_step": 192, "cases_per_group": 48,
        "snapshot_steps": list(SNAPSHOT_STEPS), "transaction_schedule": bank["transaction_schedule"],
        "training_math": {
            "tangent_gradient_protocol": m.REFINER_TANGENT_GRADIENT_PROTOCOL,
            "optimizer_protocol": m.REFINER_UPDATE_PROTOCOL,
            "decoder_strengths": metadata["decoder_strengths"],
            "decoder_strength_evidence": metadata["decoder_strength_evidence"],
            "unchanged_true_autograd": True, "probe_selection": False,
        },
        "source": metadata, "config": dataclasses.asdict(cfg),
        "frozen_physical_environment": state["fingerprint"]["mask_and_physical_environment"],
        "stage_acceptance_policy": state["fingerprint"]["stage_acceptance_policy"],
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "dtype": "float32", "device": str(device)},
        "runtime_commit": m._training_code_revision(),
        "implementation_sha256": {Path(path).name: a.file_sha256(path) for path in (
            __file__, safe.__file__, layers.__file__, m.__file__, a.__file__, d.__file__,
            d.boundary_observables.__file__, d.product_manifold.__file__, d.physical.__file__,
            d.physical_quality.__file__, Path(__file__).with_name("bridge_feasibility.py"),
            Path(__file__).with_name("refiner_optimizer.py"))},
    }
    experiment_hash = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    report = {
        "schema": SCHEMA, "experiment": experiment, "experiment_sha256": experiment_hash,
        "completed": False, "diagnostic_completed": False, "completed_steps": 0,
        "optimizer_steps": 0, "probe_loaded": False,
        "fresh_initialization": True, "source_weights_used_for_initialization": False,
        "historical_comparison_is_descriptive_only": True,
        "historical_reference": {"source_schema": metadata["source_schema"],
                                 "completed_steps": metadata["source_completed_steps"],
                                 "source_commit": metadata["source_commit"],
                                 "claim": "provenance/reference only; not a matched initialization comparison"},
        "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False,
        "next_action": "review_zero_start_trajectory_not_pilot",
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    m.save_json(experiment, output / "experiment.json")
    m.save_json(report, report_path)

    def check_source():
        for name, digest in metadata["source_sha256"].items():
            if a.file_sha256(source / name) != digest:
                raise RuntimeError("frozen source changed during trajectory diagnostic")

    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    try:
        with torch.random.fork_rng(devices=cuda_devices), a.frozen_environment(
                state["fingerprint"], metadata["decoder_strengths"]):
            initial = fresh_zero_state(cfg, seed)
            model = m.ProductManifoldTemporalRefiner(fps=cfg.fps, output_init_std=0.0).to(device)
            model.load_state_dict(initial, strict=True)
            save_state(output / "diagnostic_initial.pt", model, 0, experiment_hash)
            report.update(initial_state_sha256=safe.state_hash(initial),
                          initial_trunk_sha256=safe.state_hash(initial, trunk_only=True),
                          initial_checkpoint_sha256=a.file_sha256(output / "diagnostic_initial.pt"),
                          preflight=safe.initial_safety(model, rows, cfg))
            report["preflight_passed"] = report["preflight"]["passed"]
            m.save_json(report, report_path)
            if not report["preflight_passed"]:
                check_source()
                report.update(completed=True, diagnostic_completed=False, optimizer_steps=0,
                              next_action="review_exact_zero_initial_safety_no_pilot")
                m.save_json(report, report_path)
                print(json.dumps({"stage": "zero_start_preflight_complete", "passed": False,
                                  "optimizer_steps": 0, "probe_loaded": False,
                                  "pilot_allowed": False, "report": str(report_path)}), flush=True)
                return 2
            check_source()
            trained = train_trajectory(model, initial, bank, cfg, output, experiment_hash)
            report.update(trained, completed_steps=STEPS, optimizer_steps=STEPS)
            m.save_json(report, report_path)
            check_source()
            final_path = output / "diagnostic_latest.pt"
            if a.file_sha256(final_path) != trained["final_checkpoint_sha256"]:
                raise RuntimeError("final trajectory checkpoint changed before probe evaluation")
            snapshot = m._trusted_torch_load(final_path, map_location="cpu")
            if (snapshot.get("schema") != SCHEMA or snapshot.get("version") != MODEL_VERSION
                    or snapshot.get("arm") != ARM or snapshot.get("completed_steps") != STEPS
                    or snapshot.get("experiment_sha256") != experiment_hash
                    or snapshot.get("formal_checkpoint") is not False
                    or snapshot.get("publish_allowed") is not False
                    or snapshot.get("pilot_allowed") is not False
                    or snapshot.get("resume_allowed") is not False
                    or safe.state_hash(snapshot["model_state_dict"]) != trained["final_state_sha256"]):
                raise RuntimeError("not the fixed final zero-start trajectory state")
            final_model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
            final_model.load_state_dict(snapshot["model_state_dict"], strict=True)
            probe, probe_hash = safe.load_probe(source, state, bank, cfg)
            report.update(probe_loaded=True, probe_sha256=probe_hash,
                          final=safe.evaluate_final(final_model, bank, probe, cfg))
            if a.file_sha256(source / "probe_bank.pt") != probe_hash:
                raise RuntimeError("frozen probe changed during final evaluation")
            del model, final_model
        check_source()
        report.update(completed=True, diagnostic_completed=True)
        m.save_json(report, report_path)
        print(json.dumps({"stage": "zero_start_trajectory_complete", "report": str(report_path),
                          "trajectory": report["trajectory"],
                          "diagnostic_gates_passed": report["final"]["diagnostic_gates_passed"],
                          "scientific_acceptance": False, "publish_allowed": False,
                          "pilot_allowed": False}, allow_nan=False), flush=True)
        return 0
    except BaseException as error:
        updates = output / "updates.jsonl"
        if updates.is_file():
            partial_steps = sum(bool(line.strip()) for line in updates.read_text(encoding="utf-8").splitlines())
            report.update(completed_steps=partial_steps, optimizer_steps=partial_steps)
        report.update(completed=False, error={"type": type(error).__name__, "message": str(error)})
        m.save_json(report, report_path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
