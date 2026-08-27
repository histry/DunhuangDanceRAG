"""Paired fixed-versus-refreshed corruption diagnosis; never a formal trainer.

Only corruption reuse changes. Authentic TRAIN windows, seams, initialization,
sample order, objectives, decoder constraints and acceptance metrics stay fixed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np

from training import motion_models as m
from training import refiner_diagnostics as fixed
from training import refiner_factor_diagnostics as factors
from training import refiner_trace


SCHEMA = "refiner_noise_refresh_diagnostic_v1"
MODEL_VERSION = "refiner_noise_refresh_diagnostic_only_v1"
MODES = ("tangent_only", "mixed")
ARMS = ("fixed_noise", "refreshed_noise")
SPLITS = ("anchor_fixed_noise", "probe_unseen_noise", "probe_unseen_position")
SEVERITY = .06


def metadata(row):
    return {key: value for key, value in row.items() if key != "tangent"}


def load_factor_inputs(directory, args, cfg, clean, cond, windows):
    """Replay the previous recipes and *untrained* initial state, not its model."""
    directory = Path(directory)
    report = json.loads((directory / "factor_report.json").read_text(encoding="utf-8"))
    if (report.get("schema") != factors.SCHEMA or report.get("completed") is not True
            or report.get("published") is not False
            or report.get("role") != "training_corruption_factor_diagnostic_only"):
        raise RuntimeError("a completed, unpublished factor diagnosis is required")
    current = fixed._fingerprint(args, cfg)
    for key in ("config_sha256", "train_db_sha256", "validation_db_sha256",
                "stage_acceptance_policy", "mask_and_physical_environment"):
        if report["fingerprint"].get(key) != current[key]:
            raise RuntimeError(f"factor replay contract changed: {key}")
    if report["fingerprint"]["implementation_sha256"]["motion_models"] != current["implementation_sha256"]["motion_models"]:
        raise RuntimeError("formal model/objective implementation changed since the factor run")
    manifold = Path(m.__file__).parents[1] / "motion_geometry/product_manifold.py"
    if report["fingerprint"].get("factor_code_sha256", {}).get("product_manifold.py") != fixed.file_sha256(manifold):
        raise RuntimeError("product-manifold decoder implementation changed since the factor run")
    if report["windows"] != windows or report["corruption_severity"] != SEVERITY:
        raise RuntimeError("factor cohort or corruption intensity changed")
    for name in ("clean_cohort.npz", "recipes.npz"):
        if fixed.file_sha256(directory / name) != report["bank_files_sha256"][name]:
            raise RuntimeError(f"factor archive checksum mismatch: {name}")
    with np.load(directory / "clean_cohort.npz", allow_pickle=False) as saved:
        if not np.array_equal(clean, saved["clean"]) or not np.array_equal(cond, saved["cond"]):
            raise RuntimeError("factor clean cohort differs from fixed-fit archive")
    expected = factors.make_recipes(cfg, len(windows), cfg.seed, SEVERITY)
    if report["recipes"] != [metadata(row) for row in expected]:
        raise RuntimeError("factor recipe metadata cannot be replayed")
    with np.load(directory / "recipes.npz", allow_pickle=False) as saved:
        for row in expected:
            tangent = saved[f"tangent_{row['case_id']}"]
            if not np.array_equal(tangent, row["tangent"]):
                raise RuntimeError("factor tangent arrays cannot be replayed")
            row["tangent"] = tangent.copy()
    initial_path = directory / "diagnostic_initial_weights.pt"
    if fixed.file_sha256(initial_path) != report["initial_weights_sha256"]:
        raise RuntimeError("factor initial weight checksum mismatch")
    payload = m._trusted_torch_load(initial_path, map_location="cpu")
    if (payload.get("version") != "refiner_factor_diagnostic_only_v1"
            or payload.get("formal_checkpoint") is not False or payload.get("publish_allowed") is not False):
        raise RuntimeError("not a diagnostic initial state")
    # Do not trust a filename or version string to distinguish trained weights.
    m.torch.manual_seed(cfg.seed)
    initial = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32)
    actual = payload["model_state_dict"]
    if (set(actual) != set(initial.state_dict())
            or any(not m.torch.equal(value, actual[key]) for key, value in initial.state_dict().items())):
        raise RuntimeError("factor initial weights are not the declared fresh initialization")
    return expected, actual, {
        "directory": str(directory.resolve()), "report_sha256": fixed.file_sha256(directory / "factor_report.json"),
        "code_revision": report["fingerprint"]["code_revision"],
        "initial_weights_sha256": report["initial_weights_sha256"],
        "trained_reference_weights_used": False,
    }


class NoiseStream:
    """Private, collision-checked seeds AND actual tangent hashes across banks."""

    def __init__(self, cfg, old_recipes):
        self.cfg = cfg
        self.seeds = {int(row["noise_seed"]) for row in old_recipes}
        self.hashes = {row["tangent_sha256"] for row in old_recipes}

    def draw(self, span, *label):
        for attempt in range(10000):
            seed = factors.private_seed(self.cfg.seed, SCHEMA, *label, attempt)
            if seed in self.seeds:
                continue
            value = m._refiner_tangent_noise_np(span, SEVERITY, self.cfg, rng=np.random.default_rng(seed))
            digest = factors.array_hash(value)
            if digest not in self.hashes:
                self.seeds.add(seed)
                self.hashes.add(digest)
                return {"noise_seed": seed, "tangent": value, "tangent_sha256": digest}
        raise RuntimeError("unable to allocate disjoint diagnostic corruption")


def make_evaluation_recipes(old_recipes, cfg, stream):
    fit = [row for row in old_recipes if row["split"] == "fit_seen"]
    result = [{**row, "split": "anchor_fixed_noise", "case_id": f"anchor_{row['case_id']}"} for row in fit]
    occupied = {}
    for row in old_recipes:
        occupied.setdefault(row["window_index"], set()).add((row["a"], row["b"]))
    for row in fit:
        width = row["b"] - row["a"]
        for probe in range(2):
            noise = stream.draw(width, "held_out_noise", row["case_id"], probe)
            ref_id = f"noise_{row['case_id']}_{probe}"
            reference = {**row, **noise, "split": "probe_unseen_noise", "case_id": ref_id}
            result.append(reference)
            if probe == 0 and row["recipe_id"] in (0, 3):
                seed = factors.private_seed(cfg.seed, SCHEMA, "held_out_position", row["case_id"])
                starts = np.random.default_rng(seed).permutation(factors.allowed_seam_starts(cfg, width))
                # A different width at the same center is not a new position.
                # Exclude both prior starts and centers across ALL seam widths.
                a = next((int(p) for p in starts if all(
                    int(p) != left and 2 * int(p) + width != left + right
                    for left, right in occupied[row["window_index"]]
                )), None)
                if a is None:
                    raise RuntimeError("no new legal seam position available")
                occupied[row["window_index"]].add((a, a + width))
                result.append({**reference, "case_id": f"position_{row['case_id']}",
                               "split": "probe_unseen_position", "a": a, "b": a + width,
                               "seam_seed": seed, "paired_noise_case_id": ref_id})
    return result


def training_order(count, steps, seed):
    batch_size = min(8, count)
    rng = np.random.default_rng(factors.private_seed(seed, "batch_order"))
    draws = steps * batch_size
    return np.concatenate([rng.permutation(count) for _ in range((draws + count - 1) // count)])[:draws].reshape(steps, batch_size)


def make_refresh_plan(fit, order, stream):
    plan = []
    for step, indices in enumerate(order, 1):
        rows = []
        for position, index in enumerate(indices):
            row = fit[int(index)]
            noise = stream.draw(row["b"] - row["a"], "training_refresh", step, position, row["case_id"])
            rows.append({**row, **noise, "anchor_case_id": row["case_id"], "case_id": f"refresh_{step}_{position}",
                         "split": "refresh_training", "draw_step": step, "draw_position": position})
        plan.append(rows)
        if step == 1 or step % 100 == 0 or step == len(order):
            print(json.dumps({"stage": "refresh_plan", "completed_steps": step, "total_steps": len(order)}), flush=True)
    return plan


def save_noise_archive(path, rows):
    offsets = np.cumsum([0] + [len(row["tangent"]) for row in rows], dtype=np.int64)
    np.savez_compressed(path, tangent=np.concatenate([row["tangent"] for row in rows]), offsets=offsets,
                        noise_seeds=np.asarray([row["noise_seed"] for row in rows], dtype=np.uint32))


def mean_defined(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {"count": len(values), "mean": float(np.mean(values)) if values else None}


def compact_metrics(metrics, cfg):
    rows = metrics["windows"]
    result = {"num_cases": len(rows), "informative_repair": metrics["informative_repair"],
              "observed": m._checkpoint_validation_decision(metrics, cfg, stage="refiner")["observed"],
              "decoder": {}}
    for label in ("root_m", "rotation_rad"):
        result["decoder"][label] = {
            "target_cosine": mean_defined([row.get("decoder", {}).get(label, {}).get("applied_target_cosine") for row in rows]),
            "applied_to_target_ratio": mean_defined([row.get("decoder", {}).get(label, {}).get("applied_to_target_norm_ratio") for row in rows]),
            "cap_fraction": mean_defined([row.get("decoder", {}).get(label, {}).get("cap", {}).get("clipped_fraction") for row in rows]),
        }
    return result


def evaluate_no_edit(bank, cfg, label):
    """Exact no-op, including contacts. Zero logits would NOT be an exact no-op."""
    batch, provenance = bank
    acc = m._new_validation_physical_accumulator()
    details, errors = [], []
    plain = {key: value.detach().cpu().numpy() for key, value in batch.items()}
    for index, row in enumerate(provenance):
        clean, bad, seam = (plain[key][index] for key in ("clean", "bad", "seam"))
        m._record_validation_physical_prediction(acc, bad, clean, cfg, degraded=bad, seam_mask=seam)
        m._record_validation_clean_identity_prediction(acc, clean, clean, cfg)
        errors.append(float(np.abs(m.product_log_np(clean, bad)).mean()))
        details.append({**row, "geometry": acc["stage_repair_gates"][-1],
                        "temporal": acc["temporal_repair_gates"][-1],
                        "clean_identity": acc["clean_identity_gates"][-1],
                        "correction_is_exactly_zero": True, "direction_cosine": None})
        if (index + 1) % 8 == 0 or index + 1 == len(provenance):
            print(json.dumps({"stage": "no_edit_evaluation", "label": label,
                              "completed_cases": index + 1, "total_cases": len(provenance)}), flush=True)
    return {"num_windows": len(details), "reconstruction_product_log_l1": float(np.mean(errors)),
            "physical_quality": m._summarize_validation_physical_metrics(acc),
            "informative_repair": factors.informative_rates(details), "windows": details,
            "prediction_policy": "return_degraded_unchanged_and_clean_unchanged",
            "used_for_checkpoint_selection": False}


def position_pairs(reports):
    reference = {row["case_id"]: row for row in reports["probe_unseen_noise"]["windows"]}
    rows = []
    for row in reports["probe_unseen_position"]["windows"]:
        paired = reference[row["paired_noise_case_id"]]
        if row["tangent_sha256"] != paired["tangent_sha256"] or row["window_index"] != paired["window_index"]:
            raise RuntimeError("position-only contrast lost its paired noise/window")
        rows.append({"position_case_id": row["case_id"], "noise_case_id": paired["case_id"],
                     "window_index": row["window_index"],
                     "same_exact_noise": True,
                     "reference": {key: bool(paired[key]["accepted"]) for key in ("geometry", "temporal", "clean_identity")},
                     "new_position": {key: bool(row[key]["accepted"]) for key in ("geometry", "temporal", "clean_identity")}})
    return rows


def evaluate_all(model, banks, mode, cfg, destination, step):
    destination.mkdir(parents=True, exist_ok=True)
    summaries, reports = {}, {}
    for split in SPLITS:
        label = f"{destination.name}/{mode}/{step}/{split}"
        metrics = (evaluate_no_edit(banks[(mode, split)], cfg, label) if model is None else
                   factors.evaluate_bank(model, banks[(mode, split)], cfg, label=label))
        path = destination / f"{step:06d}_{split}.json"
        m.save_json(metrics, path)
        summaries[split] = {"report": str(path), **compact_metrics(metrics, cfg)}
        reports[split] = metrics
    pairs = position_pairs(reports)
    path = destination / f"{step:06d}_position_pairs.json"
    m.save_json({"paired_cases": pairs, "independent_validation": False}, path)
    summaries["position_pairs_report"] = str(path)
    return summaries


def compare_arms(report):
    comparison = {"schema": SCHEMA, "independent_validation": False, "scientific_acceptance": False,
                  "selection_policy": "fixed_final_step_not_best_probe_score", "modes": {}}
    for mode in MODES:
        comparison["modes"][mode] = {}
        for split in SPLITS:
            entry = {
                "no_edit": report["no_edit"][mode][split],
                **{arm: report["experiments"][mode][arm]["history"][-1]["results"][split] for arm in ARMS},
            }
            paired = [json.loads(Path(entry[arm]["report"]).read_text(encoding="utf-8"))["windows"] for arm in ARMS]
            if [row["case_id"] for row in paired[0]] != [row["case_id"] for row in paired[1]]:
                raise RuntimeError("arm comparisons must use identical cases in identical order")
            entry["paired_outcomes"] = {}
            for kind in ("geometry", "temporal", "clean_identity"):
                counts = dict.fromkeys(("both_pass", "fixed_only", "refreshed_only", "both_fail"), 0)
                excluded = 0
                for left, right in zip(*paired):
                    if kind != "clean_identity" and not factors.informative_rates([left])[kind]["informative"]:
                        excluded += 1
                        continue
                    a, b = bool(left[kind]["accepted"]), bool(right[kind]["accepted"])
                    counts["both_pass" if a and b else "fixed_only" if a else "refreshed_only" if b else "both_fail"] += 1
                entry["paired_outcomes"][kind] = {**counts, "trivial_excluded": excluded}
            comparison["modes"][mode][split] = entry
    return comparison


def print_comparison(comparison):
    print("mode / split / arm | geometry | temporal | clean_identity | joint_direction_cosine", flush=True)
    for mode, splits in comparison["modes"].items():
        for split, arms in splits.items():
            for arm in ("no_edit", *ARMS):
                entry = arms[arm]
                rates = entry["informative_repair"]
                cosine = entry["decoder"]["rotation_rad"]["target_cosine"]["mean"]
                print(f"{mode} / {split} / {arm} | "
                      f"{rates['geometry']['passed']}/{rates['geometry']['informative']} | "
                      f"{rates['temporal']['passed']}/{rates['temporal']['informative']} | "
                      f"{entry['observed']['clean_identity_rate']:.4f} | "
                      f"{cosine if cosine is not None else 'undefined_no_correction'}", flush=True)


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if not 1 <= args.steps <= 2000 or not 1 <= args.windows <= 16 or args.eval_every < 1:
        raise ValueError("diagnosis requires 1..2000 updates and 1..16 training windows")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"output exists: {out}; use a new tag")
    db, val = m.load_db(args.db), m.load_db(args.val_db)
    m._training_db_contract(db, cfg, "noise refresh TRAIN database")
    m._training_db_contract(val, cfg, "noise refresh validation METADATA ONLY")
    separation = m._validate_source_disjoint(db, val)
    del val
    clean, cond, windows, cohort = factors.load_cohort(args.fixed_fit_dir, db, cfg, args.db, args.val_db, args.windows)
    old, initial_state, parent = load_factor_inputs(args.factor_dir, args, cfg, clean, cond, windows)
    stream = NoiseStream(cfg, old)
    recipes = make_evaluation_recipes(old, cfg, stream)
    fit = [row for row in recipes if row["split"] == "anchor_fixed_noise"]
    order = training_order(len(fit), args.steps, cfg.seed)
    plan = make_refresh_plan(fit, order, stream)
    device = m.torch.device(cfg.device)
    banks = factors.build_banks(clean, cond, windows, recipes, cfg, device, modes=MODES, splits=SPLITS)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "clean_cohort.npz", clean=clean, cond=cond)
    save_noise_archive(out / "evaluation_noise.npz", recipes)
    save_noise_archive(out / "refresh_noise.npz", [row for batch in plan for row in batch])
    np.save(out / "training_order.npy", order, allow_pickle=False)
    m.save_json({"schema": SCHEMA, "evaluation": [metadata(row) for row in recipes],
                 "refresh_schedule": [[metadata(row) for row in batch] for batch in plan]}, out / "recipes.json")
    m._atomic_torch_save({"version": MODEL_VERSION, "formal_checkpoint": False, "publish_allowed": False,
                          "model_state_dict": initial_state}, out / "diagnostic_initial_weights.pt")
    fingerprint = fixed._fingerprint(args, cfg)
    fingerprint["diagnostic_code_sha256"] = {
        Path(path).name: fixed.file_sha256(path) for path in (__file__, factors.__file__, refiner_trace.__file__)
    }
    refresh_seeds = {row["noise_seed"] for batch in plan for row in batch}
    probe_seeds = {row["noise_seed"] for row in recipes if row["split"] != "anchor_fixed_noise"}
    report = {"schema": SCHEMA, "role": "training_noise_generalization_diagnostic_only", "completed": False,
              "published": False, "scientific_acceptance": False, "used_for_formal_checkpoint_selection": False,
              "formal_training_must_start_fresh": True, "fingerprint": fingerprint, "config": dataclasses.asdict(cfg),
              "parent_factor": parent, "cohort_provenance": cohort, "windows": windows, "source_separation": separation,
              "target_steps": args.steps, "batch_size": int(order.shape[1]), "corruption_severity": SEVERITY,
              "runtime": {"torch": str(m.torch.__version__), "numpy": str(np.__version__),
                          "device": str(device), "gpu_preprocessing": m._gpu_preprocessing_enabled(cfg, device)},
              "selection_policy": "fixed_final_step_not_best_probe_score",
              "controlled_variables": ["initial_weights", "optimizer", "window_order", "seam_geometry", "noise_intensity",
                                       "decoder_masks_smoothing_caps", "losses", "acceptance_criteria", "update_count"],
              "changed_variable": "reuse_fixed_noise_vs_fresh_noise_each_presentation",
              "anchor_role": {"fixed_noise": "training_cases", "refreshed_noise": "unseen_noise_at_training_positions"},
              "position_probe_role": "new_position_with_exact_noise_paired_to_unseen_noise_probe",
              "bridge_only": {"applicable": False, "reason": "deterministic_bridge_has_no_noise_to_refresh"},
              "isolation": {"validation_motion_read": False, "new_probe_noise_reuses_old_noise_seeds": False,
                            "refresh_unique_seeds": len(refresh_seeds), "held_out_noise_seeds": len(probe_seeds),
                            "refresh_probe_seed_overlap": len(refresh_seeds & probe_seeds),
                            "seeds_and_tangent_hashes_collision_checked": True},
              "artifacts_sha256": {p.name: fixed.file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
              "no_edit": {}, "experiments": {}}
    report_path = out / "noise_refresh_report.json"
    m.save_json(report, report_path)
    for mode in MODES:
        report["no_edit"][mode] = evaluate_all(None, banks, mode, cfg, out / "no_edit" / mode, 0)
        m.save_json(report, report_path)
    for mode in MODES:
        report["experiments"][mode] = {}
        for arm in ARMS:
            m.torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            random.seed(cfg.seed)
            model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32).to(device)
            model.load_state_dict(initial_state, strict=True)
            optimizer = m.torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
            destination = out / mode / arm
            entry = {"initial_weights_sha256": report["artifacts_sha256"]["diagnostic_initial_weights.pt"],
                     "training_order_sha256": factors.array_hash(order), "history": [], "completed": False}
            report["experiments"][mode][arm] = entry
            entry["baseline"] = evaluate_all(model, banks, mode, cfg, destination, 0)
            m.save_json(report, report_path)
            started, train_seconds = time.perf_counter(), 0.0
            for step, indices in enumerate(order, 1):
                train_start = time.perf_counter()
                batch = (factors.select_batch(banks[(mode, "anchor_fixed_noise")][0], indices) if arm == "fixed_noise" else
                         factors.prepare_bank(clean, cond, windows, plan[step - 1], cfg, device, mode)[0])
                repair, protection, terms, identity_terms = m._refiner_batch_objectives(model, batch, cfg)
                loss = repair + cfg.product_refiner_clean_identity_weight * protection
                logging = step == 1 or step % 25 == 0 or step == args.steps
                gradient = (m._refiner_gradient_diagnostics(model, repair, protection, cfg.product_refiner_clean_identity_weight)
                            if logging else None)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(m.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
                optimizer.step()
                train_seconds += time.perf_counter() - train_start
                if logging:
                    item = {"stage": "noise_refresh_training", "mode": mode, "arm": arm, "step": step, "target_steps": args.steps,
                            "repair_loss": float(repair.detach()), "clean_loss_unweighted": float(protection.detach()),
                            "repair_terms": {k: float(v.detach()) for k, v in terms.items()},
                            "clean_terms": {k: float(v.detach()) for k, v in identity_terms.items()},
                            "gradient": gradient, "global_clip_norm_before": norm,
                            "global_clip_scale": min(1.0, 1.0 / (norm + 1e-6)),
                            "elapsed_minutes": (time.perf_counter() - started) / 60,
                            "eta_training_only_minutes": train_seconds / step * (args.steps - step) / 60}
                    with (destination / "gradients.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(item, allow_nan=False) + "\n")
                    print(json.dumps(item, allow_nan=False), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    # Save the fixed-step candidate before costly diagnostics; never select a best probe score.
                    m._atomic_torch_save({"version": MODEL_VERSION, "formal_checkpoint": False, "publish_allowed": False,
                                          "completed_steps": step, "model_state_dict": model.state_dict(),
                                          "fingerprint": fingerprint, "mode": mode, "arm": arm},
                                         destination / "diagnostic_latest.pt")
                    entry["history"].append({"step": step, "results": evaluate_all(model, banks, mode, cfg, destination, step)})
                    entry["completed_steps"] = step
                    entry["training_presentations"] = step * int(order.shape[1])
                    m.save_json(report, report_path)
            entry["completed"] = True
            del model, optimizer
            m.save_json(report, report_path)
    comparison = compare_arms(report)
    m.save_json(comparison, out / "comparison.json")
    report.update(completed=True, next_action="review_new_noise_and_paired_position_results_not_full_training")
    m.save_json(report, report_path)
    print_comparison(comparison)
    print(json.dumps({"stage": "noise_refresh_diagnostic_complete", "report": str(report_path),
                      "comparison": str(out / "comparison.json"), "published": False, "scientific_acceptance": False}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--db", required=True)
    parser.add_argument("--val_db", required=True, help="Metadata only; no validation motion is read")
    parser.add_argument("--fixed_fit_dir", required=True)
    parser.add_argument("--factor_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval_every", type=int, default=100)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
