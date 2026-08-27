"""Fixed real-training-window learnability test; never publishes a model.

Run with ``python -m training.refiner_diagnostics``. Validation metadata is used
only to check source separation. No validation/test motion is fitted, selected,
or used to decide this diagnostic's outcome. Full training must start afresh.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from training import motion_models as m


SCHEMA = "refiner_fixed_train_fit_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_indices(source_ids, count):
    """Deterministic source-balanced selection, with no motion-quality filtering."""
    sources = np.asarray(source_ids).astype(str)
    if not 1 <= count <= 16 or count > len(sources):
        raise ValueError("diagnostic windows must be within [1,16] and available events")
    groups = [np.flatnonzero(sources == source) for source in sorted(set(sources))]
    quotas = [0] * len(groups)
    while sum(quotas) < count:
        for i, group in enumerate(groups):
            if quotas[i] < len(group) and sum(quotas) < count:
                quotas[i] += 1
    selected_groups = [group[np.linspace(0, len(group) - 1, n, dtype=int)].tolist()
                       for group, n in zip(groups, quotas)]
    return [int(group[rank]) for rank in range(max(quotas))
            for group in selected_groups if rank < len(group)]


def prepare_fixed_batch(db, cfg, count, seed, device):
    formats = np.asarray(db.get("source_formats", [])).astype(str)
    if len(formats) != len(db["paths"]) or set(formats) != {"chang_e_official_smpl"}:
        raise RuntimeError("fixed-fit diagnosis requires an official SMPL Event-DB")
    sources = np.asarray(db.get("source_uids", [])).astype(str)
    if len(sources) != len(db["paths"]) or not all(sources):
        raise RuntimeError("source-aligned training provenance is required")
    indices = fixed_indices(sources, count)
    cond = m._descriptor_values_in_training_coordinates(db, db)[indices]
    clean, bad, seams, provenance = [], [], [], []
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        for index in indices:
            path = Path(db["paths"][index])
            x = m.load_motion_window(path, cfg.window_len, cfg, random_crop=False)
            damaged, seam = m.degrade_for_refiner(
                x, cfg=cfg, finalize_contract=not m._gpu_preprocessing_enabled(cfg, device)
            )
            clean.append(x)
            bad.append(damaged)
            seams.append(seam)
            provenance.append({
                "event_index": index, "source_uid": str(sources[index]),
                "path": str(path.resolve()), "sha256": file_sha256(path),
                "seam_core_frames": np.flatnonzero(seam[:, 0] >= 0.5).tolist(),
            })
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    batch = m._prepare_refiner_batch(
        np.stack(clean), np.stack(bad), np.stack(seams), cond, cfg, device
    )
    return batch, provenance


def evaluate_fixed_batch(model, batch, provenance, cfg):
    was_training = model.training
    model.eval()
    accumulator = m._new_validation_physical_accumulator()
    details, rec = [], []
    try:
        with m.torch.no_grad():
            predictions, identities = m._refiner_batch_outputs(model, batch, cfg)
        clean, bad, seams, predictions, identities = (
            x.detach().cpu().numpy() for x in
            (batch["clean"], batch["bad"], batch["seam"], predictions, identities)
        )
        for index, item in enumerate(provenance):
            m._record_validation_physical_prediction(
                accumulator, predictions[index], clean[index], cfg,
                degraded=bad[index], seam_mask=seams[index],
            )
            m._record_validation_clean_identity_prediction(
                accumulator, identities[index], clean[index], cfg
            )
            rec.append(float(np.abs(m.product_log_np(clean[index], predictions[index])).mean()))
            details.append({
                **item, "geometry": accumulator["stage_repair_gates"][-1],
                "temporal": accumulator["temporal_repair_gates"][-1],
                "clean_identity": accumulator["clean_identity_gates"][-1],
            })
    finally:
        model.train(was_training)
    return {
        "num_windows": len(provenance), "reconstruction_product_log_l1": float(np.mean(rec)),
        "physical_quality": m._summarize_validation_physical_metrics(accumulator),
        "windows": details,
    }


def fit_decision(metrics, cfg):
    decision = m._checkpoint_validation_decision(metrics, cfg, stage="refiner")
    # Reuse criteria, never their publication semantics for training-only data.
    return {
        "fit_passed": bool(decision["scientific_acceptance"]),
        "reasons": decision["reasons"], "observed": decision["observed"],
        "thresholds": decision["thresholds"], "publish_allowed": False,
        "scientific_acceptance": False, "used_for_checkpoint_selection": False,
    }


def _fingerprint(args, cfg):
    return {
        "code_revision": m._training_code_revision(), "model_version": m.REFINER_MODEL_VERSION,
        "implementation_sha256": {
            "motion_models": file_sha256(Path(m.__file__)),
            "refiner_diagnostics": file_sha256(Path(__file__)),
        },
        "config_sha256": m._training_config_sha256(cfg, stage="refiner"),
        "train_db": str(Path(args.db).resolve()), "train_db_sha256": file_sha256(args.db),
        "validation_db": str(Path(args.val_db).resolve()),
        "validation_db_sha256": file_sha256(args.val_db),
        "stage_acceptance_policy": dataclasses.asdict(m.StageAcceptancePolicy.from_environment()),
        "mask_and_physical_environment": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith(("GROUNDING_", "PHYSICAL_", "CONTACT_"))
        },
    }


def check_report(args, cfg):
    report = json.loads(Path(args.check_report).read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("role") != "training_fit_diagnostic_only":
        raise RuntimeError("not a fixed training-fit diagnostic report")
    if report.get("fingerprint") != _fingerprint(args, cfg):
        raise RuntimeError("diagnostic revision/config/database/policy mismatch; run a fresh diagnosis")
    if report.get("published") is not False or report.get("used_for_formal_checkpoint_selection") is not False:
        raise RuntimeError("diagnostic publication semantics are invalid")
    if not report.get("completed") or not report.get("fit_gate", {}).get("fit_passed"):
        raise RuntimeError("fixed real-window fit has not passed; do not start full training")
    if len(report.get("windows", [])) != int(args.windows):
        raise RuntimeError("diagnostic window-count mismatch")
    if report["best"]["metrics"].get("num_windows") != len(report["windows"]):
        raise RuntimeError("diagnostic metrics are not window-aligned")
    # Re-evaluate actual stored metrics; do not trust a lone boolean.
    if not fit_decision(report["best"]["metrics"], cfg)["fit_passed"]:
        raise RuntimeError("stored diagnostic metrics fail current criteria")
    for item in report["windows"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("training event changed since fixed-fit diagnosis")
    print("FIXED_TRAIN_FIT_OK: fresh full training may start; held-out validation is still required", flush=True)
    return 0


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if args.check_report:
        return check_report(args, cfg)
    if not 1 <= args.steps <= 2000 or args.eval_every < 1 or args.gradient_every < 1:
        raise ValueError("diagnosis requires 1..2000 updates and positive logging intervals")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"diagnostic output already exists: {out}; use a new tag")
    db, val_db = m.load_db(args.db), m.load_db(args.val_db)
    train_contract = m._training_db_contract(db, cfg, "fixed diagnostic TRAIN database")
    val_contract = m._training_db_contract(val_db, cfg, "validation metadata only")
    separation = m._validate_source_disjoint(db, val_db)
    del val_db  # never read validation motion or use its metrics for this fit
    m.torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = m.torch.device(cfg.device)
    batch, provenance = prepare_fixed_batch(db, cfg, args.windows, cfg.seed + 47001, device)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "fixed_training_batch.npz",
        **{key: value.detach().cpu().numpy() for key, value in batch.items()},
    )
    model = m.ProductManifoldTemporalRefiner(m.EDGE_DIM, 32).to(device)
    optimizer = m.torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    baseline = evaluate_fixed_batch(model, batch, provenance, cfg)
    report = {
        "schema": SCHEMA, "role": "training_fit_diagnostic_only", "completed": False,
        "published": False, "used_for_formal_checkpoint_selection": False,
        "formal_training_must_start_fresh": True, "target_steps": args.steps,
        "fingerprint": _fingerprint(args, cfg), "config": dataclasses.asdict(cfg),
        "train_contract": train_contract, "validation_metadata_contract": val_contract,
        "source_separation": separation, "windows": provenance, "baseline": baseline,
        "fixed_batch_sha256": file_sha256(out / "fixed_training_batch.npz"),
        "history": [], "gradient_history": [], "best": None,
    }
    m.save_json(report, out / "diagnostic_report.json")
    started = time.perf_counter()
    best_score = None
    for step in range(1, args.steps + 1):
        repair, protection, terms, identity_terms = m._refiner_batch_objectives(model, batch, cfg)
        loss = repair + cfg.product_refiner_clean_identity_weight * protection
        if step == 1 or step % args.gradient_every == 0:
            gradient = {
                "stage": "fixed_fit_gradient_diagnostics", "step": step,
                **m._refiner_gradient_diagnostics(model, repair, protection, cfg.product_refiner_clean_identity_weight),
            }
            report["gradient_history"].append(gradient)
            with (out / "gradient_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(gradient, allow_nan=False) + "\n")
            print(json.dumps(gradient, allow_nan=False), flush=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        m.torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "stage": "fixed_train_fit", "step": step, "target_steps": args.steps,
                "loss": float(loss.detach()), "repair_loss": float(repair.detach()),
                "clean_loss_unweighted": float(protection.detach()),
                "geometry_margin": float(terms["repair_margin"].detach()),
                "clean_geometry_excess": float(identity_terms["geometry_excess"].detach()),
                "clean_jerk_excess": float(identity_terms["fk_temporal"].detach()),
                "clean_support_excess": float(identity_terms["support_excess"].detach()),
                "elapsed_minutes": round(elapsed / 60, 2),
                "eta_minutes": round(elapsed / step * (args.steps - step) / 60, 2),
            }), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_fixed_batch(model, batch, provenance, cfg)
            decision = fit_decision(metrics, cfg)
            score = m._refiner_validation_score(
                metrics, {"scientific_acceptance": decision["fit_passed"]}
            )
            item = {"step": step, "fit_gate": decision, "metrics": metrics}
            m.save_json(item, out / f"fit_step_{step:06d}.json")
            report["history"].append({"step": step, "fit_gate": decision})
            report["last"] = item
            if best_score is None or score > best_score:
                best_score = score
                report["best"] = item
                m._atomic_torch_save({
                    "version": "refiner_fixed_fit_diagnostic_only_v1", "formal_checkpoint": False,
                    "publish_allowed": False, "completed_steps": step,
                    "model_state_dict": model.state_dict(), "fingerprint": report["fingerprint"],
                }, out / "diagnostic_weights.pt")
            report.update(completed_steps=step, fit_gate=report["best"]["fit_gate"])
            print(json.dumps({"stage": "fixed_fit_evaluation", "step": step, **decision}), flush=True)
            m.save_json(report, out / "diagnostic_report.json")
    report.update(completed=True, elapsed_seconds=time.perf_counter() - started)
    m.save_json(report, out / "diagnostic_report.json")
    print(json.dumps({
        "stage": "fixed_fit_complete", "fit_passed": report["fit_gate"]["fit_passed"],
        "published": False, "report": str(out / "diagnostic_report.json"),
        "next_action": "review_report_then_fresh_training" if report["fit_gate"]["fit_passed"] else "do_not_start_formal_training",
    }), flush=True)
    return 0 if report["fit_gate"]["fit_passed"] else 2


def main(argv=None):
    if m.REFINER_MODEL_VERSION != "product_manifold_boundary_refiner_v6":
        raise RuntimeError("V6 fixed-noise diagnostic retired; use training.refiner_bridge_diagnostics for V7")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/motion_model.json")
    parser.add_argument("--db", required=True, help="Training Event-DB only")
    parser.add_argument("--val_db", required=True, help="Metadata for separation check only; never fitted")
    parser.add_argument("--out_dir")
    parser.add_argument("--check_report")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--gradient_every", type=int, default=25)
    args = parser.parse_args(argv)
    if not args.out_dir and not args.check_report:
        parser.error("--out_dir is required unless --check_report is supplied")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
