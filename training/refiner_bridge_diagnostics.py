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
from pathlib import Path

import numpy as np

from training import motion_models as m
from training import refiner_diagnostics as common
from motion_geometry import boundary_observables
from motion_geometry import inbetween
from motion_geometry import product_manifold, physical
from contracts import physical_quality


SCHEMA = "refiner_observable_bridge_diagnostic_v2"


def fingerprint(args, cfg):
    value = common._fingerprint(args, cfg)
    value["implementation_sha256"].update({
        "bridge_diagnostic": common.file_sha256(__file__),
        "boundary_observables": common.file_sha256(boundary_observables.__file__),
        "inbetween": common.file_sha256(inbetween.__file__),
        "bridge_feasibility": common.file_sha256(Path(__file__).with_name("bridge_feasibility.py")),
        "product_manifold": common.file_sha256(product_manifold.__file__),
        "physical_geometry": common.file_sha256(physical.__file__),
        "physical_quality": common.file_sha256(physical_quality.__file__),
    })
    value["retraction_protocol"] = product_manifold.RETRACTION_PROTOCOL
    return value


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
                    # Fixed positions differ across splits without accessing validation motion.
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
            with m.torch.no_grad():
                if predictions is not None:
                    pred, identity = predictions[(split,role)][start:start+8], batch["clean"]
                elif model is None:
                    pred, identity = batch["bad"], batch["clean"]
                else:
                    pred, identity = m._refiner_batch_outputs(model, batch, cfg)
            arrays = [x.detach().cpu().numpy() for x in (pred, identity, batch["bad"], batch["clean"], batch["seam"])]
            for prediction, clean_prediction, reference, clean, seam in zip(*arrays):
                if role == "single_recording":
                    m._record_validation_physical_prediction(physical, prediction, clean, cfg, degraded=reference, seam_mask=seam)
                    m._record_validation_clean_identity_prediction(physical, clean_prediction, clean, cfg)
                    errors.append(float(np.abs(m.product_log_np(clean, prediction)).mean()))
                    details.append({"width":int(np.sum(seam >= .5)),"observable": physical["observable_boundary_gates"][-1],
                                    "clean_identity": physical["clean_identity_gates"][-1]})
                else:
                    gate = m._observable_boundary_audit(prediction, reference, seam, cfg)
                    safety = m._fixed_support_stage_gate(reference,prediction,cfg)
                    if not gate["reference_fidelity_accepted"]:
                        safety = {**safety,"accepted":False,"reasons":[*safety.get("reasons",[]),"cross_reference_geometry_budget_exceeded"]}
                    cross.append({"width":int(np.sum(seam >= .5)),"observable": gate, "safety": safety, "hidden_clean_used": False})
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


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if args.check_report:
        report = json.loads(Path(args.check_report).read_text(encoding="utf8"))
        if report.get("schema") != SCHEMA or report.get("fingerprint") != fingerprint(args,cfg):
            raise RuntimeError("bridge diagnostic protocol/config/code/database mismatch")
        if not report.get("completed") or report.get("published") is not False:
            raise RuntimeError("diagnostic not completed or incorrectly published")
        if (report.get("target_steps") != 400 or report.get("completed_steps") != 400
                or len(report.get("windows",[])) != args.windows or args.windows != 8):
            raise RuntimeError("pilot requires the complete 8-window, 400-step protocol; smoke runs cannot authorize training")
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
    from training.bridge_feasibility import run_foundation, check_foundation_report, balanced_indices, group_decisions
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
    train = {key:m.torch.cat([banks[("seen",role)][key] for role in ("single_recording","cross_event")])
             for key in banks[("seen","single_recording")]}
    role_count = len(banks[("seen","single_recording")]["clean"])
    train["group"] = m.torch.as_tensor([i%2 for i in range(role_count)]+[2+i%2 for i in range(role_count)],device=device)
    model = m.ProductManifoldTemporalRefiner().to(device)
    optimizer = m.torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
    destination.mkdir(parents=True)
    report = {"schema":SCHEMA,"protocol":m.BOUNDARY_PROTOCOL,"fingerprint":fingerprint(args,cfg),
              "completed":False,"published":False,"independent_validation":False,
              "formal_training_must_start_fresh":True,"selection":"fixed_final_step",
              "foundation_report":str(Path(args.foundation_report).resolve()),
              "source_separation":separation,"recipes":recipes,"target_steps":args.steps,
              "windows":[{"path":str(db["paths"][i]),"sha256":common.file_sha256(db["paths"][i])} for i in selected],
              "baseline":{},"history":[]}
    for split in ("seen","new_position"):
        report["baseline"][split] = evaluate(None,banks,split,cfg)
    m.save_json(report,destination / "diagnostic_report.json")
    rng = np.random.default_rng(cfg.seed + 9001)
    for step in range(1,args.steps + 1):
        indices = balanced_indices(len(train["clean"])//2,rng)
        batch = {k:v[indices] for k,v in train.items()}
        repair,protection,terms,identity = m._refiner_batch_objectives(model,batch,cfg)
        loss = repair + cfg.product_refiner_clean_identity_weight * protection
        logging = step == 1 or step % 25 == 0 or step == args.steps
        gradient = m._refiner_gradient_diagnostics(model,repair,protection,cfg.product_refiner_clean_identity_weight) if logging else None
        components = m._refiner_component_gradients(model,terms,cfg) if logging else None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(m.torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        optimizer.step()
        if logging:
            row = {"stage":"observable_bridge_fit","step":step,"target_steps":args.steps,
                   "repair":float(repair.detach()),"clean":float(protection.detach()),
                   "terms":{k:float(v.detach()) for k,v in terms.items()},"gradient":gradient,
                   "component_gradients":components,"clip_norm_before":norm}
            with (destination / "gradients.jsonl").open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row,allow_nan=False) + "\n")
            print(json.dumps(row,allow_nan=False),flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            final = {split:evaluate(model,banks,split,cfg) for split in ("seen","new_position")}
            decisions = {split:m._checkpoint_validation_decision(metrics,cfg,stage="refiner") for split,metrics in final.items()}
            groups = {split:group_decisions(metrics,cfg) for split,metrics in final.items()}
            report.update(completed_steps=step,final=final,diagnostic_ready=(
                step == 400 and args.steps == 400 and args.windows == 8
                and all(d["scientific_acceptance"] for d in decisions.values())
                and all(g["passed"] for split in groups.values() for g in split.values())))
            report["group_decisions"] = groups
            # These decisions only judge train-window readiness, never publication.
            report["history"].append({"step":step,"readiness":{s:{"passed":d["scientific_acceptance"],"reasons":d["reasons"],"observed":d["observed"]} for s,d in decisions.items()}})
            m.save_json(report,destination / "diagnostic_report.json")
            print(json.dumps({"stage":"bridge_readiness",**report["history"][-1]}),flush=True)
    report["completed"] = True
    m.save_json(report,destination / "diagnostic_report.json")
    m._atomic_torch_save({"version":"observable_bridge_diagnostic_only_v2","formal_checkpoint":False,
                         "publish_allowed":False,"model_state_dict":model.state_dict()},destination / "diagnostic_weights.pt")
    print(json.dumps({"stage":"bridge_diagnostic_complete","ready_for_fresh_pilot":report["diagnostic_ready"],
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
