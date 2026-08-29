"""Training-only full-bridge learnability gate; never a formal checkpoint.

Eight source-balanced TRAIN windows; seen and held-out seam positions; both
single-recording occlusion and cross-event joins. Validation motion is never
loaded. The final fixed step, not the best probe result, decides readiness.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from training import motion_models as m
from training import refiner_diagnostics as common
from training.refiner_optimizer import record_update, validate_update_summary
from motion_geometry import boundary_observables
from motion_geometry import inbetween
from motion_geometry import product_manifold, physical
from contracts import physical_quality


SCHEMA = "refiner_observable_bridge_diagnostic_v10"
FIT_PROTOCOL = "complete_seen_bank_descent_v1"
PROBE_SCOPE = "unfitted_local_motion_context_within_train_windows"


def fingerprint(args, cfg):
    from training.bridge_feasibility import DIRECT_OPTIMIZER_PROTOCOL
    value = common._fingerprint(args, cfg)
    value["implementation_sha256"].update({
        "bridge_diagnostic": common.file_sha256(__file__),
        "boundary_observables": common.file_sha256(boundary_observables.__file__),
        "inbetween": common.file_sha256(inbetween.__file__),
        "bridge_feasibility": common.file_sha256(Path(__file__).with_name("bridge_feasibility.py")),
        "product_manifold": common.file_sha256(product_manifold.__file__),
        "physical_geometry": common.file_sha256(physical.__file__),
        "physical_quality": common.file_sha256(physical_quality.__file__),
        "refiner_optimizer": common.file_sha256(Path(__file__).with_name("refiner_optimizer.py")),
    })
    value["retraction_protocol"] = product_manifold.RETRACTION_PROTOCOL
    value["repair_safety_protocol"] = m.REFINER_REPAIR_SAFETY_PROTOCOL
    value["observable_objective_protocol"] = m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
    value["direct_optimizer_protocol"] = DIRECT_OPTIMIZER_PROTOCOL
    value["refiner_input_protocol"] = m.REFINER_INPUT_PROTOCOL
    value["refiner_update_protocol"] = m.REFINER_UPDATE_PROTOCOL
    value["fit_protocol"] = FIT_PROTOCOL
    value["probe_scope"] = PROBE_SCOPE
    return value


def fixed_fit_bank(banks):
    """Use EVERY seen TRAIN case; held-out positions never enter an update.

    Checking only a randomly selected 8/32 cases allowed an accepted step to
    undo gains on the other fixed cases. That is a legitimate SGD behavior,
    but confounds a tiny fixed-bank learnability diagnostic. This diagnostic
    uses the complete bank for BOTH gradients and post-update line search.
    Formal random-window training is deliberately unchanged.
    """
    roles = [banks[("seen", role)] for role in ("single_recording", "cross_event")]
    count = len(roles[0]["clean"])
    if count < 2 or count % 2 or len(roles[1]["clean"]) != count:
        raise ValueError("fixed fit bank requires equally sized paired role/width cases")
    train = {key: m.torch.cat([role[key] for role in roles]) for key in roles[0]}
    train["group"] = m.torch.as_tensor(
        [i % 2 for i in range(count)] + [2 + i % 2 for i in range(count)],
        device=train["clean"].device)
    return train


def fit_bank_contract(windows):
    return {"protocol": FIT_PROTOCOL, "cases_per_update": 4 * windows,
            "cases_per_role_width": windows, "gradient_scope": "complete_seen_bank",
            "line_search_scope": "complete_seen_bank", "probe_used_for_updates": False}


def fixed_bank_stalled(update):
    """No state change + no new inputs cannot justify repeating the same search.

    Only this deterministic fixed-bank control stops early. Formal training
    draws a new minibatch next time, so a single retained update must not stop it.
    An early stop never authorizes pilot, even if some metrics already pass.
    """
    return (not update["optimizer_update_accepted"]
            and update["reason"] in {"bounded_search_no_descent", "zero_gradient"})


def _cpu_tree(value):
    if isinstance(value, m.torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def save_fit_bank(destination, batch, report, cfg):
    """Portable exact TRAIN inputs for failure analysis, never a training asset.

    Do not reconstruct server cases from similarly named local Event-DB files:
    descriptor statistics, event cuts and normalized coordinates may differ.
    The only caller passes fixed_fit_bank(), never probe/validation tensors.
    """
    path = destination / "fit_bank.pt"
    m._atomic_torch_save({"schema":"refiner_train_fit_bank_v1", "train_only":True,
        "formal_checkpoint":False, "publish_allowed":False,
        "fingerprint":report["fingerprint"], "windows":report["windows"],
        "contract":report["fit_bank"], "config":dataclasses.asdict(cfg),
        "batch":_cpu_tree(batch)},path)
    return {"file":path.name,"sha256":common.file_sha256(path),
            "cases":len(batch["clean"]),"train_only":True}


def save_probe_bank(destination, banks, report, cfg):
    """Save exact held-out local contexts for replay, never for updates.

    V9 saved only the fitted bank, so its held-out failure could not be replayed
    away from the server Event-DB. The explicit probe-only contract makes the
    artifact auditable without turning validation inputs into training data.
    """
    path = destination / "probe_bank.pt"
    probe = {
        role: _cpu_tree(banks[("new_position", role)])
        for role in ("single_recording", "cross_event")
    }
    m._atomic_torch_save(
        {
            "schema": "refiner_local_context_probe_bank_v1",
            "probe_only": True,
            "updates_forbidden": True,
            "formal_checkpoint": False,
            "publish_allowed": False,
            "hidden_clean_single_recording_diagnostic_only": True,
            "fingerprint": report["fingerprint"],
            "windows": report["windows"],
            "config": dataclasses.asdict(cfg),
            "banks": probe,
        },
        path,
    )
    return {
        "file": path.name,
        "sha256": common.file_sha256(path),
        "cases": sum(len(row["clean"]) for row in probe.values()),
        "probe_only": True,
        "updates_forbidden": True,
    }


def save_diagnostic_state(destination, model, optimizer, report, step):
    """Exact retained state, explicitly incompatible with formal resume loaders."""
    m._atomic_torch_save({"schema":"refiner_diagnostic_state_v1",
        "formal_checkpoint":False,"publish_allowed":False,
        "completed_steps":step,"fingerprint":report["fingerprint"],
        "fit_bank_artifact":report["fit_bank_artifact"],
        "probe_bank_artifact":report.get("probe_bank_artifact"),
        "model_state_dict":_cpu_tree(model.state_dict()),
        "optimizer_state_dict":_cpu_tree(optimizer.state_dict()),
        "torch_rng":m.torch.get_rng_state()},destination / "diagnostic_state.pt")


def build_banks(clean, cond, sources, cfg, device, *, contact_ik=True):
    banks = {}
    recipes = {}
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            clean_rows, bad_rows, seams, conditions, identities, rows = [], [], [], [], [], []
            for index, original in enumerate(clean):
                partner = next((j for j in range(len(clean)) if sources[j] != sources[index]), None)
                if role == "cross_event" and partner is None:
                    raise RuntimeError("cross-event diagnosis needs multiple training sources")
                for recipe_id, width in enumerate((10, 28)):
                    # Moving the cut within original changes its local motion
                    # content too. This is NOT a pure translation-equivariance
                    # test, nor independent source-disjoint validation.
                    width = min(width, len(original) - 8)
                    a = max(3, (len(original) - width) // 2 + (-8 if recipe_id else 8))
                    if split == "new_position":
                        a += 7 if recipe_id else -7
                    b = a + width
                    bridge_info = {}
                    if role == "single_recording":
                        bad, seam = m.degrade_for_refiner(original, cfg=cfg, recipe={"a": a, "b": b},
                            contact_ik=contact_ik,bridge_report=bridge_info)
                        condition = np.tile(cond[index], (len(original), 1))
                    else:
                        bad, seam, condition = m.make_cross_event_boundary_np(
                            original, clean[partner], cond[index], cond[partner], cfg, start=a, width=width,
                            contact_ik=contact_ik,bridge_report=bridge_info)
                    clean_rows.append(original)
                    bad_rows.append(bad)
                    seams.append(seam)
                    conditions.append(condition)
                    identities.append(np.tile(cond[index], (len(original), 1)))
                    rows.append({"window": index, "source": sources[index], "role": role,
                                 "partner": partner if role == "cross_event" else None,
                                 "a": a, "b": b, "bridge":bridge_info,"hidden_clean_target": role == "single_recording"})
            batch = m._prepare_refiner_batch(np.stack(clean_rows), np.stack(bad_rows), np.stack(seams),
                                             np.stack(conditions), cfg, device)
            batch["clean_cond"] = m.torch.as_tensor(np.stack(identities), dtype=m.torch.float32, device=device)
            banks[(split, role)] = batch
            recipes[f"{split}/{role}"] = rows
    return banks, recipes


def evaluate(model, banks, split, cfg, *, predictions=None):
    physical = m._new_validation_physical_accumulator()
    errors, details, cross = [], [], []
    for role in ("single_recording", "cross_event"):
        bank = banks[(split, role)]
        for start in range(0, len(bank["clean"]), 8):
            batch = {k: v[start:start + 8] for k,v in bank.items()}
            decoder_rows = [None] * len(batch["clean"])
            with m.torch.no_grad():
                if predictions is not None:
                    pred, identity = predictions[(split,role)][start:start+8], batch["clean"]
                elif model is None:
                    pred, identity = batch["bad"], batch["clean"]
                else:
                    trace = {}
                    pred, identity = m._refiner_batch_outputs(model, batch, cfg, trace=trace)
                    from training.bridge_feasibility import decoder_summary
                    decoder_rows = decoder_summary(trace["repair"], batch["seam"])
            arrays = [x.detach().cpu().numpy() for x in (pred, identity, batch["bad"], batch["clean"], batch["seam"])]
            for case, (prediction, clean_prediction, reference, clean, seam) in enumerate(zip(*arrays)):
                if role == "single_recording":
                    m._record_validation_physical_prediction(physical, prediction, clean, cfg, degraded=reference, seam_mask=seam)
                    m._record_validation_clean_identity_prediction(physical, clean_prediction, clean, cfg)
                    errors.append(float(np.abs(m.product_log_np(clean, prediction)).mean()))
                    details.append({"case_index":start+case,"decoder":decoder_rows[case],
                                    "width":int(np.sum(seam >= .5)),"observable": physical["observable_boundary_gates"][-1],
                                    "clean_identity": physical["clean_identity_gates"][-1]})
                else:
                    gate = m._observable_boundary_audit(prediction, reference, seam, cfg)
                    safety = m._fixed_support_stage_gate(reference,prediction,cfg)
                    if not gate["reference_fidelity_accepted"]:
                        safety = {**safety,"accepted":False,"reasons":[*safety.get("reasons",[]),"cross_reference_geometry_budget_exceeded"]}
                    cross.append({"case_index":start+case,"decoder":decoder_rows[case],
                                  "width":int(np.sum(seam >= .5)),"observable": gate, "safety": safety, "hidden_clean_used": False})
            print(json.dumps({"stage": "bridge_probe", "split": split, "role": role,
                              "completed": min(start + 8, len(bank["clean"])), "total": len(bank["clean"])}), flush=True)
    gates = [row["observable"] for row in cross]
    return {"physical_quality": m._summarize_validation_physical_metrics(physical),
            "reconstruction_product_log_l1": float(np.mean(errors)), "windows": details,
            "cross_event": {"schema": m.BOUNDARY_PROTOCOL, "num_windows": len(cross),
                "endpoint": m._summarize_validation_gates(gates,accepted_key="endpoint_accepted"),
                "temporal": m._summarize_validation_gates(gates,accepted_key="temporal_accepted"),
                "physical_non_regression": m._summarize_validation_gates([r["safety"] for r in cross],accepted_key="accepted"),
                "endpoint_informative": sum(g["endpoint_informative"] for g in gates),
                "temporal_informative": sum(g["temporal_informative"] for g in gates), "windows": cross}}


def failure_breakdown(metrics):
    """Small console/report evidence; never a second, looser acceptance rule."""
    groups = {}
    for role, rows in (("single_recording", metrics["windows"]),
                       ("cross_event", metrics["cross_event"]["windows"])):
        for width in sorted({row["width"] for row in rows}):
            selected = [row for row in rows if row["width"] == width]
            gates = [row["observable"] for row in selected]
            reasons = Counter(reason for row in selected for reason in
                (row.get("safety") or row["observable"]["physical_non_regression"]).get("reasons", []))
            decoder = [row["decoder"] for row in selected if row.get("decoder") is not None]
            groups[f"{role}/{width}"] = {
                "cases": len(selected),
                "endpoint_pass": sum(bool(g["endpoint_accepted"]) for g in gates),
                "temporal_pass": sum(bool(g["temporal_accepted"]) for g in gates),
                "temporal_gain_pass": sum(bool(g["temporal_gain_only"]) for g in gates),
                "jerk_non_regression_pass": sum(bool(g["jerk_non_regression"]) for g in gates),
                "endpoint_gain_median": float(np.median([g["endpoint_gain"] for g in gates])),
                "temporal_gain_median": float(np.median([g["temporal_gain"] for g in gates])),
                "physical_failure_reasons": dict(sorted(reasons.items())),
                "decoder_means": ({key: float(np.mean([row[key] for row in decoder]))
                    for key in ("raw_tangent_rms", "applied_tangent_rms", "root_mask_mean",
                                "joint_mask_mean", "root_cap_fraction", "joint_cap_fraction")}
                    if decoder else None),
            }
    return groups


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if args.check_report:
        report = json.loads(Path(args.check_report).read_text(encoding="utf8"))
        if report.get("schema") != SCHEMA or report.get("fingerprint") != fingerprint(args,cfg):
            raise RuntimeError("bridge diagnostic protocol/config/code/database mismatch")
        if not report.get("completed") or report.get("published") is not False:
            raise RuntimeError("diagnostic not completed or incorrectly published")
        if report.get("stopped_early"):
            raise RuntimeError("fixed TRAIN-bank optimization stalled; review the diagnostic, do not train")
        if (report.get("target_steps") != 400 or report.get("completed_steps") != 400
                or len(report.get("windows",[])) != args.windows or args.windows != 8):
            raise RuntimeError("pilot requires the complete 8-window, 400-step protocol; smoke runs cannot authorize training")
        validate_update_summary(report.get("optimizer_updates", {}), 400)
        if report.get("fit_bank") != fit_bank_contract(args.windows):
            raise RuntimeError("diagnostic did not use the complete predefined TRAIN fit bank")
        from training.bridge_feasibility import check_foundation_report, group_decisions
        check_foundation_report(report["foundation_report"],fingerprint(args,cfg),cfg)
        for role in ("seen", "new_position"):
            if (report["final"][role]["physical_quality"].get("num_windows") != 2 * args.windows
                    or report["final"][role]["cross_event"].get("num_windows") != 2 * args.windows):
                raise RuntimeError("diagnostic case counts do not match the predefined protocol")
            decision = m._checkpoint_validation_decision(report["final"][role],cfg,stage="refiner")
            if not decision["scientific_acceptance"]:
                raise RuntimeError(f"{role} bridge diagnosis failed: {decision['reasons']}; do not train")
            if not all(row["passed"] for row in group_decisions(report["final"][role],cfg).values()):
                raise RuntimeError(f"{role} role/width subgroup failed; aggregate cannot authorize training")
        for row in report["windows"]:
            if common.file_sha256(row["path"]) != row["sha256"]:
                raise RuntimeError("training source window changed")
        print("BRIDGE_DIAGNOSTIC_READY: fresh pilot permitted; independent validation still required",flush=True)
        return 0
    if not args.out_dir or not 1 <= args.steps <= 2000 or args.eval_every < 1:
        raise ValueError("provide new out_dir, 1..2000 steps and positive eval_every")
    destination = Path(args.out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    from training.bridge_feasibility import run_foundation, check_foundation_report, group_decisions
    if not getattr(args,"baseline_only",False):
        if not getattr(args,"foundation_report",None):
            raise RuntimeError("run --baseline_only first and review the direct-optimization control; --foundation_report is required")
        check_foundation_report(args.foundation_report,fingerprint(args,cfg),cfg)
    db, val = m.load_db(args.db), m.load_db(args.val_db)
    m._training_db_contract(db,cfg,"bridge diagnostic TRAIN")
    formats = np.asarray(db.get("source_formats",[])).astype(str)
    if len(formats) != len(db["paths"]) or set(formats) != {"chang_e_official_smpl"}:
        raise RuntimeError("bridge diagnostic requires official SMPL training events")
    m._training_db_contract(val,cfg,"bridge diagnostic validation metadata only")
    separation = m._validate_source_disjoint(db,val)
    del val
    selected = common.fixed_indices(db["source_uids"],args.windows)
    clean = np.stack([m.load_motion_window(db["paths"][i],cfg.window_len,cfg,random_crop=False) for i in selected])
    cond = m._descriptor_values_in_training_coordinates(db,db)[selected]
    sources = [str(db["source_uids"][i]) for i in selected]
    device = m.torch.device(cfg.device)
    m.torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    banks, recipes = build_banks(clean,cond,sources,cfg,device)
    if getattr(args,"baseline_only",False):
        pure,_ = build_banks(clean,cond,sources,cfg,device,contact_ik=False)
        return run_foundation(args,cfg,banks,pure,recipes,fingerprint(args,cfg),
            [{"path":str(db["paths"][i]),"sha256":common.file_sha256(db["paths"][i])} for i in selected],separation)
    train = fixed_fit_bank(banks)
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
    optimizer = m.torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
    destination.mkdir(parents=True)
    report = {"schema":SCHEMA,"protocol":m.BOUNDARY_PROTOCOL,"fingerprint":fingerprint(args,cfg),
              "completed":False,"published":False,"independent_validation":False,
              "probe_scope":PROBE_SCOPE,
              "formal_training_must_start_fresh":True,"selection":"fixed_final_step",
              "foundation_report":str(Path(args.foundation_report).resolve()),
              "fit_bank":fit_bank_contract(args.windows),
              "source_separation":separation,"recipes":recipes,"target_steps":args.steps,
              "windows":[{"path":str(db["paths"][i]),"sha256":common.file_sha256(db["paths"][i])} for i in selected],
              "baseline":{},"history":[]}
    report["fit_bank_artifact"] = save_fit_bank(destination,train,report,cfg)
    report["probe_bank_artifact"] = save_probe_bank(
        destination, banks, report, cfg
    )
    save_diagnostic_state(destination,model,optimizer,report,0)
    for split in ("seen","new_position"):
        report["baseline"][split] = evaluate(None,banks,split,cfg)
    m.save_json(report,destination / "diagnostic_report.json")
    report["optimizer_updates"] = {}
    started = time.perf_counter()
    for step in range(1,args.steps + 1):
        batch = train
        repair,protection,terms,identity = m._refiner_batch_objectives(model,batch,cfg)
        loss = repair + cfg.product_refiner_clean_identity_weight * protection
        logging = step == 1 or step % 25 == 0 or step == args.steps
        gradient = m._refiner_gradient_diagnostics(model,repair,protection,cfg.product_refiner_clean_identity_weight) if logging else None
        components = m._refiner_component_gradients(model,terms,cfg) if logging else None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(m.torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        update = m.checked_refiner_step(
            optimizer, loss, lambda:m._refiner_total_batch_loss(model,batch,cfg),
            gradient_unscale=max(1.0,norm+1.0e-6))
        record_update(report["optimizer_updates"], update)
        with (destination / "optimizer_updates.jsonl").open("a",encoding="utf8") as handle:
            handle.write(json.dumps({"step":step,**update},allow_nan=False) + "\n")
        stopped_early = fixed_bank_stalled(update) and step < args.steps
        report["stopped_early"] = stopped_early
        report["termination_reason"] = update["reason"] if stopped_early else None
        if stopped_early and not logging:
            # backward() released the old graph. The rejected transaction has
            # restored the exact pre-update state; recompute on the SAME bank.
            r,p,t,_ = m._refiner_batch_objectives(model,batch,cfg)
            gradient = m._refiner_gradient_diagnostics(model,r,p,cfg.product_refiner_clean_identity_weight)
            components = m._refiner_component_gradients(model,t,cfg)
        if logging or stopped_early:
            save_diagnostic_state(destination,model,optimizer,report,step)
            row = {"stage":"observable_bridge_fit","step":step,"target_steps":args.steps,
                   "repair":float(repair.detach()),"clean":float(protection.detach()),
                   "terms":{k:float(v.detach()) for k,v in terms.items()},"gradient":gradient,
                   "component_gradients":components,"clip_norm_before":norm,
                   "optimizer_update":update,
                   "fit_bank":report["fit_bank"],
                   "elapsed_seconds":time.perf_counter()-started,
                   "optimizer_updates":dict(report["optimizer_updates"])}
            with (destination / "gradients.jsonl").open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row,allow_nan=False) + "\n")
            print(json.dumps(row,allow_nan=False),flush=True)
        if step % args.eval_every == 0 or step == args.steps or stopped_early:
            final = {split:evaluate(model,banks,split,cfg) for split in ("seen","new_position")}
            decisions = {split:m._checkpoint_validation_decision(metrics,cfg,stage="refiner") for split,metrics in final.items()}
            groups = {split:group_decisions(metrics,cfg) for split,metrics in final.items()}
            report.update(completed_steps=step,final=final,diagnostic_ready=(
                step == 400 and args.steps == 400 and args.windows == 8
                and all(d["scientific_acceptance"] for d in decisions.values())
                and all(g["passed"] for split in groups.values() for g in split.values())))
            report["group_decisions"] = groups
            breakdown = {split:failure_breakdown(metrics) for split,metrics in final.items()}
            report["failure_breakdown"] = breakdown
            # These decisions only judge train-window readiness, never publication.
            report["history"].append({"step":step,"readiness":{s:{"passed":d["scientific_acceptance"],"reasons":d["reasons"],"observed":d["observed"]} for s,d in decisions.items()}})
            m.save_json(report,destination / "diagnostic_report.json")
            m.save_json({"schema":SCHEMA,"fingerprint":report["fingerprint"],
                         "completed_steps":step,"diagnostic_ready":report["diagnostic_ready"],
                         "group_decisions":groups,"failure_breakdown":breakdown,
                         "optimizer_updates":report["optimizer_updates"],
                         "fit_bank":report["fit_bank"],
                         "fit_bank_artifact":report["fit_bank_artifact"],
                         "stopped_early":stopped_early,
                         "termination_reason":report["termination_reason"],
                         "scientific_acceptance":False,"publish_allowed":False},
                        destination / "summary.json")
            print(json.dumps({"stage":"bridge_readiness",**report["history"][-1]}),flush=True)
            for split, rows in breakdown.items():
                for group, row in rows.items():
                    print(json.dumps({"stage":"bridge_failure_breakdown","step":step,
                                      "split":split,"group":group,**row},allow_nan=False),flush=True)
        if stopped_early:
            print(json.dumps({"stage":"bridge_fixed_bank_stalled","completed_steps":step,
                              "target_steps":args.steps,"reason":report["termination_reason"],
                              "state_retained":True,"published":False}),flush=True)
            break
    report["completed"] = True
    m.save_json(report,destination / "diagnostic_report.json")
    m._atomic_torch_save({"version":"observable_bridge_diagnostic_only_v2","formal_checkpoint":False,
                         "publish_allowed":False,"model_state_dict":model.state_dict()},destination / "diagnostic_weights.pt")
    print(json.dumps({"stage":"bridge_diagnostic_complete","ready_for_fresh_pilot":report["diagnostic_ready"],
                      "completed_steps":report["completed_steps"],"stopped_early":report["stopped_early"],
                      "published":False,"report":str(destination / "diagnostic_report.json")}),flush=True)
    return 0 if report["diagnostic_ready"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/motion_model.json")
    parser.add_argument("--db",required=True)
    parser.add_argument("--val_db",required=True)
    parser.add_argument("--out_dir")
    parser.add_argument("--check_report")
    parser.add_argument("--windows",type=int,default=8)
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--eval_every",type=int,default=200)
    parser.add_argument("--baseline_only",action="store_true",help="pure bridge, contact IK and direct-output optimization only; no neural fitting")
    parser.add_argument("--foundation_report",help="reviewed, passing baseline/feasibility report from this exact revision")
    parser.add_argument("--direct_steps",type=int,default=200)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
