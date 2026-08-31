"""Paired initialization diagnostic, never a Pilot or formal checkpoint.

Fresh identical trunks, fixed sigma, fixed TRAIN reservoir schedule, 400 updates
per arm. All unique TRAIN banks must pass initial safety before either arm can
train. The held-out local-context artifact is opened only after BOTH arms finish.
"""
from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from training import motion_models as m
from training import refiner_bridge_diagnostics as d
from training import refiner_group_gradient_audit as a
from training.bridge_feasibility import group_decisions
from training.refiner_optimizer import checked_refiner_step, record_update, validate_update_summary


SCHEMA = "refiner_paired_safe_start_diagnostic_v1"
MODEL_VERSION = "refiner_safe_start_diagnostic_only_v1"
ARMS = {"A0_zero": 0.0, "A1_gaussian": 1e-5}
STEPS = 400


def state_hash(state, *, trunk_only=False):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if trunk_only and name.startswith("out."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def paired_initial_states(cfg, seed):
    """CPU initialization gives the two arms exactly the same trunk tensors."""
    if not 0 <= seed < 2**32:
        raise ValueError("initialization seed must be in [0,2**32)")
    states = {}
    # Seed the CPU generator only, without modifying any CUDA generator.
    with torch.random.fork_rng(devices=[]):
        for arm, std in ARMS.items():
            torch.random.default_generator.manual_seed(seed)
            model = m.ProductManifoldTemporalRefiner(fps=cfg.fps, output_init_std=std)
            states[arm] = d._cpu_tree(model.state_dict())
    if len({state_hash(state, trunk_only=True) for state in states.values()}) != 1:
        raise RuntimeError("paired initialization changed the trunk")
    if not bool((states["A0_zero"]["out.weight"] == 0).all()):
        raise RuntimeError("baseline head is not zero")
    if not bool((states["A1_gaussian"]["out.weight"] != 0).any()):
        raise RuntimeError("Gaussian head is still zero")
    return states


def train_banks(bank, cfg):
    """Validate the entire TRAIN reservoir, including contexts beyond index 0."""
    rows = [("anchor", bank["anchor"])] + [
        (f"context_{i}", bank["context_reservoir"][str(i)]) for i in range(len(bank["context_reservoir"]))]
    if not 1 <= len(bank["transaction_schedule"]) <= STEPS:
        raise ValueError("400 steps must cover at least one complete TRAIN cycle")
    for _, part in rows:
        a._validate_bank(part, cfg)
        if set(part) != set(bank["anchor"]):
            raise ValueError("TRAIN reservoir fields differ, including clean_cond")
    return rows


def initial_safety(model, rows, cfg):
    """Same existing per-case physical/fidelity/clean gates; no repair-gain demand.

    A small random output is not assumed safe. Every bank/case is checked, with
    no sigma search, retry, mask change, mean-only gate or probe consultation.
    """
    result = {"passed": True, "checked_cases": 0, "banks": [], "failure_counts": {}}
    failures = Counter()
    model.eval()
    for name, cpu_batch in rows:
        batch = {key: value.to(cfg.device) for key, value in cpu_batch.items()}
        with torch.no_grad():
            prediction, clean_prediction = m._refiner_batch_outputs(model, batch, cfg)
        arrays = [v.detach().cpu().numpy() for v in
                  (prediction, clean_prediction, batch["bad"], batch["clean"])]
        rejected = []
        for index, (pred, ident, reference, clean) in enumerate(zip(*arrays)):
            safety = m._fixed_support_stage_gate(reference, pred, cfg)
            _, fidelity = m._observable_reference_fidelity(reference, pred, cfg)
            acc = m._new_validation_physical_accumulator()
            m._record_validation_clean_identity_prediction(acc, ident, clean, cfg)
            identity = acc["clean_identity_gates"][-1]
            reasons = [f"repair/{r}" for r in safety.get("reasons", [])]
            if not safety["accepted"] and not reasons:
                reasons.append("repair/physical_rejection")
            if not fidelity:
                reasons.append("repair/reference_geometry_budget_exceeded")
            reasons.extend(f"clean/{r}" for r in identity.get("reasons", []))
            if not identity["accepted"] and not identity.get("reasons"):
                reasons.append("clean/identity_rejection")
            result["checked_cases"] += 1
            if reasons:
                rejected.append({"case": index, "group": a.GROUPS[int(cpu_batch["group"][index])],
                                 "reasons": reasons})
                failures.update(reasons)
        result["banks"].append({"bank": name, "cases": len(cpu_batch["clean"]), "rejected": rejected})
        result["passed"] = result["passed"] and not rejected
        print(json.dumps({"stage": "safe_start_preflight", "bank": name, "rejected": len(rejected),
                          "checked_cases": result["checked_cases"]}), flush=True)
    result["failure_counts"] = dict(failures)
    return result


def step_measurements(model, before, gradients, initial):
    """Actual retained updates, including decay, AFTER checked-step rollback.

    Raw gradients alone do not measure Adam learning. Record per-layer actual
    updates, displacement from initialization and the true g dot delta as well.
    """
    layers, sums = {}, {scope: Counter() for scope in ("shared_trunk", "output_head")}
    for name, parameter in model.named_parameters():
        old = before[name].double()
        gradient = gradients[name].double()
        delta = parameter.detach().double() - old
        drift = parameter.detach().double() - initial[name].double()
        pn, gn, un, dn, dot = torch.stack([
            old.norm(), gradient.norm(), delta.norm(), drift.norm(), (gradient * delta).sum()]).cpu().tolist()
        row = {"numel": parameter.numel(), "parameter_norm_before": pn,
               "gradient_norm_before_clip": gn, "gradient_rms_before_clip": gn / math.sqrt(parameter.numel()),
               "actual_update_norm": un, "actual_update_to_parameter_ratio": un / pn if pn else None,
               "displacement_from_initial_norm": dn, "true_gradient_dot_actual_update": dot}
        if not all(math.isfinite(v) for v in (pn, gn, un, dn, dot)):
            raise FloatingPointError("nonfinite layer/update measurement")
        layers[name] = row
        scope = "output_head" if name.startswith("out.") else "shared_trunk"
        sums[scope].update(numel=parameter.numel(), parameter_squared=pn**2, gradient_squared=gn**2,
                           update_squared=un**2, displacement_squared=dn**2, dot=dot)
    scopes = {scope: {"numel": v["numel"], "gradient_norm_before_clip": math.sqrt(v["gradient_squared"]),
                      "gradient_rms_before_clip": math.sqrt(v["gradient_squared"] / v["numel"]),
                      "actual_update_norm": math.sqrt(v["update_squared"]),
                      "displacement_from_initial_norm": math.sqrt(v["displacement_squared"]),
                      "true_gradient_dot_actual_update": v["dot"]}
              for scope, v in sums.items()}
    return {"layers": layers, "scopes": scopes,
            "update_note": "Includes AdamW decay; nonzero displacement alone is not task learning."}


def save_state(path, model, arm, step, experiment_hash):
    m._atomic_torch_save({"version": MODEL_VERSION, "formal_checkpoint": False,
                         "publish_allowed": False, "pilot_allowed": False,
                         "arm": arm, "completed_steps": step, "experiment_sha256": experiment_hash,
                         "model_state_dict": d._cpu_tree(model.state_dict())}, path)


def train_arm(model, initial, bank, cfg, arm, destination, experiment_hash, *, steps=STEPS):
    """Fixed budget. No resume, early-stop selection, LR change or probe reads."""
    if not 1 <= steps <= STEPS:
        raise ValueError("diagnostic updates must be bounded by the fixed 400-step budget")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    initial = {key: value.to(cfg.device) for key, value in initial.items()}
    summary = {}
    started = time.perf_counter()
    final = None
    for step in range(1, steps + 1):
        index = (step - 1) % len(bank["transaction_schedule"])
        batch = {key: value.to(cfg.device) for key, value in a.materialize_transaction(bank, cfg, index).items()}
        repair, clean, terms, _ = m._refiner_batch_objectives(model, batch, cfg)
        loss = repair + cfg.product_refiner_clean_identity_weight * clean
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
        update = checked_refiner_step(
            optimizer, loss, lambda: m._refiner_guarded_total_batch_loss(model, batch, cfg, require_all_groups=True),
            gradient_unscale=max(1.0, norm + 1e-6),
            group_guard_before=m._refiner_group_repair_losses(terms, require_all=True),
            group_guard_relative_tolerance=cfg.product_refiner_group_guard_relative_tolerance,
            group_guard_absolute_tolerance=cfg.product_refiner_group_guard_absolute_tolerance)
        record_update(summary, update)
        final = {"step": step, "transaction_index": index,
                 "context_indices": list(bank["transaction_schedule"][index]),
                 "cases": len(batch["group"]), "repair_before": float(repair.detach()),
                 "clean_before": float(clean.detach()), "total_before": float(loss.detach()),
                 "group_terms_before": {k: float(v.detach()) for k, v in terms.items() if k.startswith("group_")},
                 "clip_norm_before": norm, "optimizer_update": update,
                 **step_measurements(model, before, gradients, initial)}
        if not update["optimizer_update_accepted"] and any(
            row["actual_update_norm"] != 0 for row in final["layers"].values()):
            raise RuntimeError("rejected optimizer update changed model parameters")
        with (destination / "updates.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(final, allow_nan=False) + "\n")
        if step == 1 or step % 25 == 0 or step == steps:
            save_state(destination / "diagnostic_latest.pt", model, arm, step, experiment_hash)
            print(json.dumps({"stage": "safe_start_fit", "arm": arm, "step": step,
                              "total_before": final["total_before"], "scopes": final["scopes"],
                              "optimizer_updates": summary, "elapsed_seconds": time.perf_counter() - started},
                             allow_nan=False), flush=True)
    validate_update_summary(summary, steps)
    return {"completed_steps": steps, "optimizer_updates": summary, "final_training_transaction": final,
            "final_state_sha256": state_hash(model.state_dict()),
            "final_checkpoint_sha256": a.file_sha256(destination / "diagnostic_latest.pt")}


def load_probe(source, state, bank, cfg):
    """Only call after both fixed final states exist; never return a fit bank."""
    report = json.loads((source / "diagnostic_report.json").read_text(encoding="utf-8"))
    descriptor = state.get("probe_bank_artifact")
    if (not isinstance(descriptor, dict) or descriptor != report.get("probe_bank_artifact")
            or descriptor.get("file") != "probe_bank.pt" or descriptor.get("cases") != 32
            or descriptor.get("probe_only") is not True or descriptor.get("updates_forbidden") is not True):
        raise ValueError("missing or mismatched frozen probe descriptor")
    path = source / "probe_bank.pt"
    digest = a.file_sha256(path)
    if digest != descriptor.get("sha256"):
        raise ValueError("probe artifact checksum mismatch")
    probe = m._trusted_torch_load(path, map_location="cpu")
    if (probe.get("schema") != "refiner_local_context_probe_bank_v1"
            or probe.get("probe_only") is not True or probe.get("updates_forbidden") is not True
            or probe.get("formal_checkpoint") is not False or probe.get("publish_allowed") is not False
            or probe.get("fingerprint") != bank["fingerprint"] or probe.get("config") != bank["config"]
            or probe.get("windows") != bank["windows"]):
        raise ValueError("probe/source/config contract mismatch")
    roles = probe["banks"]
    if set(roles) != {"single_recording", "cross_event"}:
        raise ValueError("probe role set mismatch")
    parts = [roles[role] for role in ("single_recording", "cross_event")]
    if any(set(part) != set(parts[0]) or any(v.shape[0] != 16 for v in part.values()) for part in parts):
        raise ValueError("probe role tensor layouts differ")
    combined = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    if "group" in combined:
        raise ValueError("unexpected fit group labels in probe artifact")
    combined["group"] = bank["anchor"]["group"].clone()
    a._validate_bank(combined, cfg)
    if a.file_sha256(path) != digest:
        raise RuntimeError("probe artifact changed during load")
    return roles, digest


def evaluate_final(model, bank, probe, cfg):
    """Use the existing four-group endpoint/temporal/physical/clean decisions."""
    model.eval()
    banks = {("seen", role): {key: value[offset:offset + 16].to(cfg.device)
                              for key, value in bank["anchor"].items() if key != "group"}
             for role, offset in (("single_recording", 0), ("cross_event", 16))}
    banks.update({("new_position", role): {key: value.to(cfg.device) for key, value in part.items()}
                  for role, part in probe.items()})
    metrics = {split: d.evaluate(model, banks, split, cfg) for split in ("seen", "new_position")}
    decisions = {split: m._checkpoint_validation_decision(value, cfg, stage="refiner")
                 for split, value in metrics.items()}
    groups = {split: group_decisions(value, cfg) for split, value in metrics.items()}
    expected = {f"{role}/{width}" for role in ("single_recording", "cross_event") for width in (10, 28)}
    if any(set(table) != expected or any(row["cases"] != 8 for row in table.values()) for table in groups.values()):
        raise RuntimeError("final evaluation must contain all four eight-case role/width groups")
    return {"metrics": metrics, "decisions": decisions, "group_decisions": groups,
            "failure_breakdown": {split: d.failure_breakdown(value) for split, value in metrics.items()},
            "diagnostic_gates_passed": all(v["scientific_acceptance"] for v in decisions.values())
                and all(row["passed"] for split in groups.values() for row in split.values())}


def run(args):
    source, output = Path(args.state_dir).resolve(), Path(args.out_dir).resolve()
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError("use a new diagnostic directory outside the frozen source")
    state, bank, cfg, metadata = a.load_frozen_source(
        source, a.LEGACY_COMMIT, legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength)
    rows = train_banks(bank, cfg)
    if not args.preflight_only and not (source / "probe_bank.pt").is_file():
        raise FileNotFoundError("final evaluation requires the original probe_bank.pt; never regenerate it")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent device fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    seed = int(cfg.seed if args.seed is None else args.seed)
    experiment = {"schema": SCHEMA, "seed": seed, "arms": ARMS, "steps_per_arm": STEPS,
                  "selection": "both_fixed_final_steps_no_probe_selection", "source": metadata,
                  "config": dataclasses.asdict(cfg), "transaction_schedule": bank["transaction_schedule"],
                  "frozen_physical_environment": state["fingerprint"]["mask_and_physical_environment"],
                  "stage_acceptance_policy": state["fingerprint"]["stage_acceptance_policy"],
                  "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                              "dtype": "float32", "device": str(device)},
                  "runtime_commit": m._training_code_revision(),
                  "implementation_sha256": {Path(p).name: a.file_sha256(p) for p in
                      (__file__, m.__file__, a.__file__, d.__file__, d.boundary_observables.__file__,
                       d.product_manifold.__file__, d.physical.__file__, d.physical_quality.__file__,
                       Path(__file__).with_name("bridge_feasibility.py"),
                       Path(__file__).with_name("refiner_optimizer.py"))}}
    experiment_hash = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    report = {"schema": SCHEMA, "experiment": experiment, "experiment_sha256": experiment_hash,
              "completed": False, "preflight_only": args.preflight_only, "arms": {},
              "probe_loaded": False, "source_weights_used_for_initialization": False,
              "scientific_acceptance": False, "publish_allowed": False, "pilot_allowed": False,
              "next_action": "review_paired_diagnostic_not_pilot"}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    m.save_json(experiment, output / "experiment.json")
    m.save_json(report, report_path)
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])

    def check_source():
        for name, digest in metadata["source_sha256"].items():
            if a.file_sha256(source / name) != digest:
                raise RuntimeError("frozen source changed during diagnostic")

    try:
        with torch.random.fork_rng(devices=cuda_devices), a.frozen_environment(state["fingerprint"], metadata["decoder_strengths"]):
            initial = paired_initial_states(cfg, seed)
            for arm in ARMS:
                destination = output / arm
                destination.mkdir()
                model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
                model.load_state_dict(initial[arm], strict=True)
                save_state(destination / "diagnostic_initial.pt", model, arm, 0, experiment_hash)
                print(json.dumps({"stage": "safe_start_arm_preflight", "arm": arm}), flush=True)
                report["arms"][arm] = {"initial_state_sha256": state_hash(initial[arm]),
                                       "initial_trunk_sha256": state_hash(initial[arm], trunk_only=True),
                                       "initialization_std": ARMS[arm], "completed_steps": 0,
                                       "preflight": initial_safety(model, rows, cfg)}
                m.save_json(report, report_path)
                del model
            preflight_passed = all(row["preflight"]["passed"] for row in report["arms"].values())
            report["preflight_passed"] = preflight_passed
            if not preflight_passed or args.preflight_only:
                check_source()
                report.update(completed=True, diagnostic_completed=False,
                              next_action="review_initial_safety_no_pilot")
                m.save_json(report, report_path)
                print(json.dumps({"stage": "safe_start_preflight_complete", "passed": preflight_passed,
                                  "report": str(report_path), "optimizer_steps": 0, "pilot_allowed": False}), flush=True)
                return 0 if preflight_passed else 2
            check_source()
            for arm in ARMS:
                model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
                model.load_state_dict(initial[arm], strict=True)
                report["arms"][arm].update(train_arm(model, initial[arm], bank, cfg, arm, output / arm, experiment_hash))
                m.save_json(report, report_path)
                del model
            # Probe data is structurally unavailable to both optimizer loops.
            check_source()
            probe, probe_hash = load_probe(source, state, bank, cfg)
            report.update(probe_loaded=True, probe_sha256=probe_hash)
            for arm in ARMS:
                path = output / arm / "diagnostic_latest.pt"
                if a.file_sha256(path) != report["arms"][arm]["final_checkpoint_sha256"]:
                    raise RuntimeError("final diagnostic checkpoint changed before evaluation")
                snapshot = m._trusted_torch_load(path, map_location="cpu")
                if (snapshot.get("experiment_sha256") != experiment_hash or snapshot.get("completed_steps") != STEPS
                        or snapshot.get("version") != MODEL_VERSION or snapshot.get("arm") != arm
                        or snapshot.get("formal_checkpoint") is not False or snapshot.get("publish_allowed") is not False
                        or snapshot.get("pilot_allowed") is not False
                        or state_hash(snapshot["model_state_dict"]) != report["arms"][arm]["final_state_sha256"]):
                    raise RuntimeError("not the prescribed final diagnostic state")
                model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
                model.load_state_dict(snapshot["model_state_dict"], strict=True)
                report["arms"][arm]["final"] = evaluate_final(model, bank, probe, cfg)
                m.save_json(report, report_path)
                del model
            if a.file_sha256(source / "probe_bank.pt") != probe_hash:
                raise RuntimeError("frozen probe changed during evaluation")
        check_source()
        report.update(completed=True, diagnostic_completed=True)
        m.save_json(report, report_path)
        for arm, entry in report["arms"].items():
            print(json.dumps({"stage": "safe_start_result", "arm": arm,
                              "diagnostic_gates_passed": entry["final"]["diagnostic_gates_passed"],
                              "failure_breakdown": entry["final"]["failure_breakdown"],
                              "optimizer_updates": entry["optimizer_updates"], "pilot_allowed": False}), flush=True)
        print(json.dumps({"stage": "safe_start_diagnostic_complete", "report": str(report_path),
                          "pilot_allowed": False, "publish_allowed": False}), flush=True)
        return 0  # Successful execution is not a successful scientific experiment.
    except BaseException as error:
        report.update(completed=False, error={"type": type(error).__name__, "message": str(error)})
        m.save_json(report, report_path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
