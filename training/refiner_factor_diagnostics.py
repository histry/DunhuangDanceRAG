"""Paired training-only corruption experiments. Never publishes formal models.

Use the existing fixed-fit cohort, not validation motion. Three fresh models
share initialization and training order. Unseen probes never select weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from training import motion_models as m
from training import refiner_diagnostics as fixed
from training import refiner_trace


SCHEMA = "refiner_factor_diagnostic_v1"
MODES = ("bridge_only", "tangent_only", "mixed")
SPLITS = ("fit_seen", "probe_unseen_noise", "probe_unseen_position")
VARIANTS = ("full_confidence", "no_smoothing", "no_cap")


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, value.dtype.str)).encode() + value.tobytes()).hexdigest()


def private_seed(seed, *parts):
    text = json.dumps([int(seed), *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def load_cohort(directory, db, cfg, db_path, val_path, expected_windows):
    directory = Path(directory)
    report = json.loads((directory / "diagnostic_report.json").read_text(encoding="utf-8"))
    if report.get("schema") != fixed.SCHEMA or report.get("role") != "training_fit_diagnostic_only":
        raise RuntimeError("cohort must come from a fixed training-only diagnosis")
    fingerprint = report["fingerprint"]
    for key, path in (("train_db_sha256", db_path), ("validation_db_sha256", val_path)):
        if fingerprint[key] != fixed.file_sha256(path):
            raise RuntimeError("cohort database changed")
    if fingerprint["config_sha256"] != m._training_config_sha256(cfg, stage="refiner"):
        raise RuntimeError("cohort configuration changed")
    archive = directory / "fixed_training_batch.npz"
    if fixed.file_sha256(archive) != report["fixed_batch_sha256"]:
        raise RuntimeError("fixed training batch checksum mismatch")
    windows = report["windows"]
    if len(windows) != expected_windows or not 1 <= expected_windows <= 16:
        raise RuntimeError("unexpected cohort size")
    indices = [int(row["event_index"]) for row in windows]
    if len(set(indices)) != len(indices):
        raise RuntimeError("duplicate cohort events")
    formats = np.asarray(db.get("source_formats", [])).astype(str)
    if len(formats) != len(db["paths"]) or set(formats) != {"chang_e_official_smpl"}:
        raise RuntimeError("official SMPL source provenance required")
    for row, index in zip(windows, indices):
        if not 0 <= index < len(db["paths"]):
            raise RuntimeError("cohort index not in training database")
        if Path(row["path"]).resolve() != Path(db["paths"][index]).resolve():
            raise RuntimeError("cohort event is not the declared training event")
        if row["source_uid"] != str(db["source_uids"][index]) or fixed.file_sha256(row["path"]) != row["sha256"]:
            raise RuntimeError("cohort source/content mismatch")
    with np.load(archive, allow_pickle=False) as saved:
        clean, cond = saved["clean"].copy(), saved["cond"].copy()
    if clean.shape != (len(windows), cfg.window_len, m.EDGE_DIM) or not np.all(np.isfinite(clean)):
        raise RuntimeError("invalid fixed clean motion")
    expected_cond = m._descriptor_values_in_training_coordinates(db, db)[indices]
    if cond.shape != expected_cond.shape or not np.allclose(cond, expected_cond, atol=1e-6, rtol=0):
        raise RuntimeError("cohort descriptor coordinates changed")
    return clean, cond, windows, {"report_sha256": fixed.file_sha256(directory / "diagnostic_report.json"),
                                "original_code_revision": fingerprint["code_revision"],
                                "fixed_batch_sha256": report["fixed_batch_sha256"]}


def allowed_seam_starts(cfg, width):
    """The formal generator's legal center/clamping rule, shared by probes."""
    frames = int(cfg.window_len)
    pairs = set()
    for center in range(max(2, frames // 5), max(3, 4 * frames // 5) + 1):
        a = max(1, center - width // 2)
        b = min(frames - 1, a + width)
        a = max(1, b - width)
        if b - a == width:
            pairs.add(a)
    if len(pairs) < 6:
        raise ValueError("too few legal training seam positions for paired probes")
    return np.asarray(sorted(pairs), dtype=int)


def make_recipes(cfg, count, seed, severity=.06):
    """Independent streams; changing mode cannot change geometry or random draws."""
    frames = int(cfg.window_len)
    low = max(4, int(round(cfg.transition_train_min_seconds * cfg.fps)))
    high = max(low, min(int(round(cfg.transition_train_max_seconds * cfg.fps)), frames // 3))
    widths = np.linspace(low, high, 4, dtype=int)
    result = []
    for window in range(count):
        seen = set()
        training = []
        for recipe_id, width in enumerate(widths):
            seed_value = private_seed(seed, window, recipe_id, "seam")
            positions = np.random.default_rng(seed_value).permutation(allowed_seam_starts(cfg, int(width)))
            a = next(int(p) for p in positions if (int(p), int(p) + int(width)) not in seen)
            b = a + int(width)
            seen.add((a, b))
            noise_seed = private_seed(seed, window, recipe_id, "fit_noise")
            tangent = m._refiner_tangent_noise_np(int(width), severity, cfg, rng=np.random.default_rng(noise_seed))
            row = {"window_index": window, "recipe_id": recipe_id, "a": a, "b": b,
                   "seam_seed": seed_value, "noise_seed": noise_seed,
                   "split": "fit_seen", "tangent": tangent}
            training.append(row)
            result.append(row)
            for probe in range(2):
                new_seed = private_seed(seed, window, recipe_id, "probe_noise", probe)
                new_tangent = m._refiner_tangent_noise_np(int(width), severity, cfg, rng=np.random.default_rng(new_seed))
                result.append({**row, "split": "probe_unseen_noise", "noise_seed": new_seed,
                               "probe_id": probe, "tangent": new_tangent})
        for recipe_id in (0, 3):
            row = training[recipe_id]
            seed_value = private_seed(seed, window, recipe_id, "probe_position")
            positions = np.random.default_rng(seed_value).permutation(allowed_seam_starts(cfg, row["b"] - row["a"]))
            a = next(int(p) for p in positions if (int(p), int(p) + row["b"] - row["a"]) not in seen)
            result.append({**row, "split": "probe_unseen_position", "a": a,
                           "b": a + row["b"] - row["a"], "seam_seed": seed_value})
    for index, row in enumerate(result):
        row["case_id"] = index
        row["tangent_sha256"] = array_hash(row["tangent"])
    return result


def prepare_bank(clean, cond, windows, rows, cfg, device, mode):
    """One bank or one online batch, with identical formal preprocessing."""
    bad, seams, provenance = [], [], []
    for row in rows:
        damaged, seam = m.degrade_for_refiner(
            clean[row["window_index"]], cfg=cfg, mode=mode, recipe=row,
            finalize_contract=not m._gpu_preprocessing_enabled(cfg, device)
        )
        bad.append(damaged)
        seams.append(seam)
        provenance.append({**windows[row["window_index"]],
                           **{k: v for k, v in row.items() if k != "tangent"},
                           "seam_core_frames": np.flatnonzero(seam[:, 0] >= .5).tolist()})
    indices = [row["window_index"] for row in rows]
    batch = m._prepare_refiner_batch(clean[indices], np.stack(bad), np.stack(seams), cond[indices], cfg, device)
    return batch, provenance


def build_banks(clean, cond, windows, recipes, cfg, device, *, modes=MODES, splits=SPLITS):
    banks = {}
    for mode in modes:
        for split in splits:
            if mode == "bridge_only" and split == "probe_unseen_noise":
                continue  # a deterministic duplicate is NOT an unseen probe
            rows = [row for row in recipes if row["split"] == split]
            banks[(mode, split)] = prepare_bank(clean, cond, windows, rows, cfg, device, mode)
            print(json.dumps({"stage": "factor_bank", "mode": mode, "split": split, "cases": len(rows)}), flush=True)
    return banks


def select_batch(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def informative_rates(details):
    """Near-zero targets are reported, never silently counted as repair wins."""
    result = {}
    for kind in ("geometry", "temporal"):
        eligible = []
        for row in details:
            gate = row[kind]
            if kind == "geometry":
                informative = gate["detail"]["degraded_product_log_l1_to_clean"] > 1e-6
            else:
                before = gate["detail"]["degraded"]
                informative = (np.mean([before[k] for k in ("seam_velocity_error", "seam_acceleration_error", "seam_jerk_error")]) > 1e-6
                               and before["endpoint_velocity_error"] > 1e-6)
            if informative:
                eligible.append(bool(gate["accepted"]))
        result[kind] = {"informative": len(eligible), "trivial": len(details) - len(eligible),
                        "passed": sum(eligible), "rate": float(np.mean(eligible)) if eligible else None}
    # Decompose the existing conjunctive temporal gate for diagnosis only.
    # Endpoint-near-zero cases must not inflate an apparent repair rate.
    for component in ("temporal", "endpoint"):
        eligible = []
        for row in details:
            detail = row["temporal"]["detail"]
            before = detail["degraded"]
            baseline = (before["endpoint_velocity_error"] if component == "endpoint"
                        else np.mean([before[k] for k in ("seam_velocity_error", "seam_acceleration_error", "seam_jerk_error")]))
            if baseline > 1e-6:
                gain = detail[f"{component}_repair_gain"]
                eligible.append(bool(np.isfinite(gain) and gain >= detail[f"minimum_{component}_repair_gain"]))
        result[f"{component}_gain_only"] = {
            "informative": len(eligible), "trivial": len(details) - len(eligible),
            "passed": sum(eligible), "rate": float(np.mean(eligible)) if eligible else None,
            "diagnostic_component_only": True,
        }
    return result


def evaluate_bank(model, bank, cfg, *, label, counterfactual_windows=0, batch_size=8):
    batch, provenance = bank
    accumulator = m._new_validation_physical_accumulator()
    details, errors, counterfactuals = [], [], []
    selected = []
    for i, row in enumerate(provenance):
        if row["window_index"] not in [provenance[j]["window_index"] for j in selected]:
            selected.append(i)
    selected = set(selected[:counterfactual_windows])
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(provenance), batch_size):
            stop = min(len(provenance), start + batch_size)
            part = select_batch(batch, slice(start, stop))
            count = stop - start
            trace = {}
            with m.torch.no_grad():
                prediction, identity = m._refiner_batch_outputs(model, part, cfg, trace=trace)
            plain = {k: v.detach().cpu().numpy() for k, v in part.items()}
            predicted, identities = prediction.detach().cpu().numpy(), identity.detach().cpu().numpy()
            decoded_trace = refiner_trace.detached_numpy(trace["repair"])
            clean_trace = refiner_trace.detached_numpy(trace["clean"])
            for local, global_index in enumerate(range(start, stop)):
                clean, bad, seam = plain["clean"][local], plain["bad"][local], plain["seam"][local]
                m._record_validation_physical_prediction(accumulator, predicted[local], clean, cfg, degraded=bad, seam_mask=seam)
                m._record_validation_clean_identity_prediction(accumulator, identities[local], clean, cfg)
                errors.append(float(np.abs(m.product_log_np(clean, predicted[local])).mean()))
                details.append({**provenance[global_index],
                                "geometry": accumulator["stage_repair_gates"][-1],
                                "temporal": accumulator["temporal_repair_gates"][-1],
                                "clean_identity": accumulator["clean_identity_gates"][-1],
                                "decoder": refiner_trace.summarize_window(decoded_trace, local, clean, bad, seam),
                                "clean_decoder": refiner_trace.summarize_window(clean_trace, local, clean, clean, seam)})
                if global_index in selected:
                    for variant in VARIANTS:
                        with m.torch.no_grad():
                            cf = m._decode_product_refiner_output(
                                part["bad"][local:local+1], trace["raw_output"][local:local+1],
                                part["joint"][local:local+1], part["root"][local:local+1],
                                part["contact"][local:local+1], cfg, diagnostic_variant=variant)
                            cf_clean = m._decode_product_refiner_output(
                                part["clean"][local:local+1], trace["raw_output"][count+local:count+local+1],
                                part["clean_joint"][local:local+1], part["clean_root"][local:local+1],
                                part["clean_contact"][local:local+1], cfg, diagnostic_variant=variant)
                        acc = m._new_validation_physical_accumulator()
                        m._record_validation_physical_prediction(acc, cf[0].cpu().numpy(), clean, cfg, degraded=bad, seam_mask=seam)
                        m._record_validation_clean_identity_prediction(acc, cf_clean[0].cpu().numpy(), clean, cfg)
                        counterfactuals.append({"case_id": provenance[global_index]["case_id"], "variant": variant,
                                               "diagnostic_only": True, "publish_allowed": False,
                                               "geometry": acc["stage_repair_gates"][0],
                                               "temporal": acc["temporal_repair_gates"][0],
                                               "clean_identity": acc["clean_identity_gates"][0],
                                               "physical_quality": m._summarize_validation_physical_metrics(acc)})
            print(json.dumps({"stage": "factor_evaluation_progress", "label": label, "completed_cases": stop,
                              "total_cases": len(provenance)}), flush=True)
    finally:
        model.train(was_training)
    return {"num_windows": len(provenance), "reconstruction_product_log_l1": float(np.mean(errors)),
            "physical_quality": m._summarize_validation_physical_metrics(accumulator),
            "informative_repair": informative_rates(details), "windows": details,
            "counterfactuals": counterfactuals, "used_for_checkpoint_selection": False}


def evaluate_splits(model, banks, mode, cfg, destination, step, counterfactual_windows):
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in SPLITS:
        if (mode, split) not in banks:
            summary[split] = {"applicable": False, "reason": "bridge_only_has_no_noise_factor"}
            continue
        metrics = evaluate_bank(model, banks[(mode, split)], cfg, label=f"{mode}/{step}/{split}",
                                counterfactual_windows=counterfactual_windows)
        path = destination / f"{step:06d}_{split}.json"
        m.save_json(metrics, path)
        summary[split] = {"applicable": True, "report": str(path),
                          "informative_repair": metrics["informative_repair"],
                          "observed": m._checkpoint_validation_decision(metrics, cfg, stage="refiner")["observed"]}
    return summary


def load_reference_snapshot(path, cfg, training_contract, validation_contract, device):
    """Read-only audit of a V6 training snapshot, NOT permission to resume it."""
    payload = m._trusted_torch_load(Path(path), map_location="cpu")
    if (payload.get("schema") != m.TRAINING_RESUME_SNAPSHOT_SCHEMA
            or payload.get("model_version") != m.REFINER_MODEL_VERSION
            or payload.get("stage") != "refiner" or payload.get("formal_checkpoint") is not False):
        raise RuntimeError("reference must be a V6 Refiner training snapshot")
    m.assert_motion_checkpoint_contract(payload, cfg, Path(path), "boundary_refiner")
    for current, key in ((training_contract, "training_event_db_contract"), (validation_contract, "validation_event_db_contract")):
        m.assert_same_event_db_contract(current["event_db_contract"], payload[key], context="factor reference snapshot")
    if payload["training_config_sha256"] != m._training_config_sha256(cfg, stage="refiner"):
        raise RuntimeError("reference snapshot configuration differs")
    model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {"path": str(Path(path).resolve()), "sha256": fixed.file_sha256(path),
                   "code_revision": payload["code_revision"], "completed_steps": payload["completed_steps"],
                   "read_only": True, "used_to_initialize_experiments": False}


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if not 1 <= args.steps <= 2000 or not 1 <= args.windows <= 16 or args.eval_every < 1:
        raise ValueError("factor diagnosis must be bounded: 1..2000 updates, 1..16 windows")
    if not 0 <= args.counterfactual_windows <= args.windows:
        raise ValueError("invalid counterfactual cohort size")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"output exists: {out}; use a new tag")
    db, val = m.load_db(args.db), m.load_db(args.val_db)
    train_contract = m._training_db_contract(db, cfg, "factor TRAIN database")
    val_contract = m._training_db_contract(val, cfg, "factor validation METADATA ONLY")
    separation = m._validate_source_disjoint(db, val)
    del val
    clean, cond, windows, cohort_provenance = load_cohort(args.fixed_fit_dir, db, cfg, args.db, args.val_db, args.windows)
    recipes = make_recipes(cfg, len(windows), cfg.seed)
    device = m.torch.device(cfg.device)
    reference, reference_info = None, None
    if args.reference_snapshot:
        reference, reference_info = load_reference_snapshot(args.reference_snapshot, cfg, train_contract, val_contract, device)
    banks = build_banks(clean, cond, windows, recipes, cfg, device)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "recipes.npz", **{f"tangent_{r['case_id']}": r["tangent"] for r in recipes})
    np.savez_compressed(out / "clean_cohort.npz", clean=clean, cond=cond)
    for (mode, split), (batch, _) in banks.items():
        np.savez_compressed(out / f"bank_{mode}_{split}.npz", **{k: v.cpu().numpy() for k, v in batch.items()})
    m.torch.manual_seed(cfg.seed)
    initial = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32)
    initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    del initial
    init_path = out / "diagnostic_initial_weights.pt"
    m._atomic_torch_save({"version": "refiner_factor_diagnostic_only_v1", "formal_checkpoint": False,
                          "publish_allowed": False, "model_state_dict": initial_state}, init_path)
    training_cases = len(banks[("mixed", "fit_seen")][1])
    batch_size = min(8, training_cases)
    order_rng = np.random.default_rng(private_seed(cfg.seed, "batch_order"))
    order = np.concatenate([order_rng.permutation(training_cases)
                            for _ in range((args.steps * batch_size + training_cases - 1) // training_cases)])
    order = order[:args.steps * batch_size].reshape(args.steps, batch_size)
    np.save(out / "training_order.npy", order, allow_pickle=False)
    fingerprint = fixed._fingerprint(args, cfg)
    fingerprint["factor_code_sha256"] = {str(Path(path).name): fixed.file_sha256(path)
                                         for path in (__file__, refiner_trace.__file__, Path(m.__file__).parents[1] / "motion_geometry/product_manifold.py")}
    report = {"schema": SCHEMA, "role": "training_corruption_factor_diagnostic_only", "completed": False,
              "published": False, "scientific_acceptance": False, "used_for_formal_checkpoint_selection": False,
              "formal_training_must_start_fresh": True, "fingerprint": fingerprint,
              "config": dataclasses.asdict(cfg), "target_steps": args.steps,
              "experimental_batch_size": batch_size, "corruption_severity": .06,
              "selection_policy": "fixed_final_step_not_best_unseen_score", "windows": windows,
              "cohort_provenance": cohort_provenance, "source_separation": separation,
              "reference_snapshot": reference_info, "reference_results": {},
              "recipes": [{k: v for k, v in row.items() if k != "tangent"} for row in recipes],
              "initial_weights_sha256": fixed.file_sha256(init_path), "training_order_sha256": array_hash(order),
              "bank_files_sha256": {p.name: fixed.file_sha256(p) for p in sorted(out.glob("*.npz"))},
              "modes": {}}
    report_path = out / "factor_report.json"
    m.save_json(report, report_path)
    if reference is not None:
        for mode in MODES:
            report["reference_results"][mode] = evaluate_splits(reference, banks, mode, cfg, out / "reference" / mode,
                                                               int(reference_info["completed_steps"]), args.counterfactual_windows)
            m.save_json(report, report_path)
        del reference
    for mode in MODES:
        m.torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32).to(device)
        model.load_state_dict(initial_state, strict=True)
        optimizer = m.torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        destination = out / mode
        destination.mkdir()
        entry = {"initial_weights_sha256": report["initial_weights_sha256"], "history": [],
                 "initialization": "fresh_shared_weights_not_reference_snapshot"}
        report["modes"][mode] = entry
        entry["baseline"] = evaluate_splits(model, banks, mode, cfg, destination, 0, 0)
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            batch = select_batch(banks[(mode, "fit_seen")][0], order[step - 1])
            repair, protection, terms, identity_terms = m._refiner_batch_objectives(model, batch, cfg)
            loss = repair + cfg.product_refiner_clean_identity_weight * protection
            logging = step == 1 or step % 25 == 0
            gradient = (m._refiner_gradient_diagnostics(model, repair, protection, cfg.product_refiner_clean_identity_weight)
                        if logging else None)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = m.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if logging:
                norm = float(gradient_norm)
                elapsed = time.perf_counter() - started
                item = {"stage": "factor_training", "mode": mode, "step": step, "target_steps": args.steps,
                        "repair_loss": float(repair.detach()), "clean_loss_unweighted": float(protection.detach()),
                        "repair_terms": {k: float(v.detach()) for k, v in terms.items()},
                        "clean_terms": {k: float(v.detach()) for k, v in identity_terms.items()},
                        "gradient": gradient, "global_clip_norm_before": norm,
                        "global_clip_scale": min(1.0, 1.0 / (norm + 1e-6)),
                        "elapsed_minutes": elapsed / 60, "eta_minutes": elapsed / step * (args.steps - step) / 60}
                with (destination / "gradients.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, allow_nan=False) + "\n")
                print(json.dumps(item, allow_nan=False), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                entry["history"].append({"step": step, "results": evaluate_splits(
                    model, banks, mode, cfg, destination, step,
                    args.counterfactual_windows if step == args.steps else 0)})
                entry["completed_steps"] = step
                m.save_json(report, report_path)
        # Fixed final step only; no optimizer state and no formal loader version.
        m._atomic_torch_save({"version": "refiner_factor_diagnostic_only_v1", "formal_checkpoint": False,
                              "publish_allowed": False, "completed_steps": args.steps,
                              "model_state_dict": model.state_dict(), "fingerprint": fingerprint},
                             destination / "diagnostic_final.pt")
        entry["completed"] = True
        del optimizer, model
        m.save_json(report, report_path)
    report.update(completed=True, next_action="review_factor_reports_not_full_training")
    m.save_json(report, report_path)
    print(json.dumps({"stage": "factor_diagnostic_complete", "report": str(report_path), "published": False,
                      "next_action": report["next_action"]}), flush=True)
    return 0  # execution complete != model acceptance


def main(argv=None):
    if m.REFINER_MODEL_VERSION != "product_manifold_boundary_refiner_v6":
        raise RuntimeError("V6 factor experiment retired; use training.refiner_bridge_diagnostics for V7")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--db", required=True)
    parser.add_argument("--val_db", required=True, help="Metadata only; no validation motion is read")
    parser.add_argument("--fixed_fit_dir", required=True)
    parser.add_argument("--reference_snapshot")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--counterfactual_windows", type=int, default=4)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
