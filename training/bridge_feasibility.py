"""TRAIN-window interpolation/IK and direct-output feasibility controls.

Direct optimization fits each case separately (including the probe positions).
It is an optimistic reachability control, NOT generalization, a learned model,
or a proof of infeasibility when a finite optimizer fails. No formal weights are
written. Repair thresholds, masks, smoothing, caps and safety checks are shared.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from training import motion_models as m

SCHEMA = "bridge_foundation_feasibility_v1"


def balanced_indices(role_count, rng):
    """Equal count per role x alternating short/long recipe, no probe sampling."""
    if role_count < 2 or role_count % 2:
        raise ValueError("each role must contain paired short/long cases")
    groups = [np.arange(offset+parity,offset+role_count,2)
              for offset in (0,role_count) for parity in (0,1)]
    count = min(2,min(map(len,groups)))
    return np.concatenate([rng.choice(group,count,replace=False) for group in groups])


def group_decisions(metrics, cfg):
    result = {}
    for role,rows in (("single_recording",metrics["windows"]),("cross_event",metrics["cross_event"]["windows"])):
        for width in sorted({row["width"] for row in rows}):
            selected = [row for row in rows if row["width"] == width]
            threshold = math.ceil(cfg.checkpoint_validation_min_stage_repair_rate*len(selected))
            temporal_threshold = math.ceil(cfg.checkpoint_validation_min_temporal_repair_rate*len(selected))
            endpoint = sum(bool(row["observable"]["endpoint_accepted"]) for row in selected)
            temporal = sum(bool(row["observable"]["temporal_accepted"]) for row in selected)
            safety = sum(bool((row.get("safety") or row["observable"]["physical_non_regression"])["accepted"])
                         and bool(row["observable"]["reference_fidelity_accepted"]) for row in selected)
            result[f"{role}/{width}"] = {"cases":len(selected),"required":threshold,
                "endpoint":endpoint,"temporal":temporal,"physical_non_regression":safety,
                "temporal_required":temporal_threshold,
                "passed":len(selected)>0 and min(endpoint,safety)>=threshold and temporal>=temporal_threshold}
    return result


def decoder_summary(trace, seam):
    """Per-case actual edit/mask/cap evidence, without serializing giant tensors."""
    core = seam[...,0] >= .5
    rows = []
    for i in range(len(core)):
        active = core[i]
        raw,applied = trace["raw"][i,active],trace["after_cap"][i,active]
        before = trace["after_taper"][i,active]
        def scalar(x): return float(x.detach().cpu())
        rows.append({"raw_tangent_rms":scalar(raw.square().mean().sqrt()),
            "applied_tangent_rms":scalar(applied.square().mean().sqrt()),
            "root_mask_mean":scalar(trace["root_mask"][i,active].mean()),
            "joint_mask_mean":scalar(trace["joint_mask"][i,active].mean()),
            "root_cap_fraction":scalar((torch_norm(before[:,:3]-applied[:,:3])>1e-7).float().mean()),
            "joint_cap_fraction":scalar((torch_norm((before[:,3:]-applied[:,3:]).reshape(-1,24,3))>1e-7).float().mean())})
    return rows


def torch_norm(x):
    return m.torch.linalg.vector_norm(x,dim=-1)


def direct_optimize(bank,cfg,steps, *, label,log_path):
    """One free output tensor per case; no shared network, no hidden clean input."""
    torch = m.torch
    output = torch.nn.Parameter(bank["bad"].new_zeros((*bank["bad"].shape[:-1],m.PRODUCT_STATE_DIM)))
    optimizer = torch.optim.Adam([output],lr=0.003)
    masks = m._refiner_decode_masks(bank["joint"],bank["root"],bank["contact"],bank["seam"],cfg)
    started = time.perf_counter()
    for step in range(steps):
        prediction = m._decode_product_refiner_output(bank["bad"],output,*masks,cfg)
        losses,terms = m._observable_refiner_objective(prediction,bank["bad"],bank["seam"],cfg,reduction="none")
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Independent parameters; clipping is per case, never across cases.
        flat = output.grad.flatten(1)
        norm = torch.linalg.vector_norm(flat,dim=1)
        if not torch.isfinite(norm).all():
            raise RuntimeError("nonfinite direct-optimization gradient")
        output.grad.mul_((1/norm.clamp_min(1)).view(-1,1,1))
        old = output.detach().clone()
        optimizer.step()
        # A fixed Adam step can overshoot quiet windows by orders of magnitude.
        # Deterministic per-case backtracking keeps the feasibility control from
        # confusing optimizer divergence with an unattainable repair target.
        update = output.detach()-old
        selected = old.clone()
        pending = torch.ones(len(old),device=old.device,dtype=torch.bool)
        scale = torch.ones(len(old),device=old.device,dtype=old.dtype)
        with torch.no_grad():
            for _ in range(12):
                trial = old + update*scale[:,None,None]
                motion = m._decode_product_refiner_output(bank["bad"],trial,*masks,cfg)
                candidate,_ = m._observable_refiner_objective(motion,bank["bad"],bank["seam"],cfg,reduction="none")
                accepted = pending & torch.isfinite(candidate) & (candidate <= losses.detach()+1e-10)
                selected[accepted] = trial[accepted]
                pending &= ~accepted
                if not bool(pending.any()):
                    break
                scale[pending] *= .5
            output.copy_(selected)
            for state_key in ("exp_avg","exp_avg_sq"):
                optimizer.state[output][state_key][pending] = 0
        if step==0 or (step+1)%25==0 or step+1==steps:
            elapsed = time.perf_counter()-started
            row = {"stage":"direct_bridge_optimization","group":label,"step":step+1,"steps":steps,
                "loss":float(loss.detach()),"endpoint_loss":float(terms["endpoint_continuity"].detach().mean()),
                "temporal_loss":float(terms["temporal_supervision"].detach().mean()),"support_loss":float(terms["support_excess"].detach().mean()),
                "line_search_rejected_cases":int(pending.sum()),"line_search_min_scale":float(scale.min()),
                "elapsed_seconds":elapsed,"eta_seconds":elapsed/(step+1)*(steps-step-1)}
            with Path(log_path).open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps(row,allow_nan=False),flush=True)
    with torch.no_grad():
        trace = {}
        prediction = m._decode_product_refiner_output(bank["bad"],output,*masks,cfg,trace=trace)
    return prediction.detach(),decoder_summary(trace,bank["seam"])


def baseline_comparison(pure,banks,cfg):
    result = {}
    for key,bank in banks.items():
        rows = []
        for ref,after,seam in zip(pure[key]["bad"].cpu().numpy(),bank["bad"].cpu().numpy(),bank["seam"].cpu().numpy()):
            gate = m._observable_boundary_audit(after,ref,seam,cfg)
            safety = m._fixed_support_stage_gate(ref,after,cfg)
            # No-edit/IK baseline only needs preservation, not fictitious repair.
            before,proposed = gate["before"],gate["after"]
            preserved = (gate["reference_fidelity_accepted"] and safety["accepted"] and
                proposed["temporal_energy"] <= before["temporal_energy"]+1e-6 and
                proposed["endpoint_velocity_jump_mps"] <= before["endpoint_velocity_jump_mps"]+1e-6 and gate["jerk_non_regression"])
            rows.append({"width":int(np.sum(seam>=.5)),"observable":gate,"safety":safety,"preserved":bool(preserved)})
        result["/".join(key)] = {"windows":rows,"preserved":sum(r["preserved"] for r in rows),"cases":len(rows)}
    return result


def roundtrip_diagnostic(banks,cfg):
    """Measure no-edit decode/FK numerical error; NEVER enlarge tolerances here."""
    rows = []
    for key,bank in banks.items():
        with m.torch.no_grad():
            output = bank["bad"].new_zeros((*bank["bad"].shape[:-1],m.PRODUCT_STATE_DIM))
            decoded = m._decode_product_refiner_output(bank["bad"],output,
                *m._refiner_decode_masks(bank["joint"],bank["root"],bank["contact"],bank["seam"],cfg),cfg)
        for reference,candidate in zip(bank["bad"].cpu().numpy(),decoded.cpu().numpy()):
            gate = m._fixed_support_stage_gate(reference,candidate,cfg)
            rows.append({"group":"/".join(key),"gate":gate,
                "max_fk_roundtrip_m":float(np.abs(m.fk_24_np(reference)-m.fk_24_np(candidate)).max())})
    return {"policy":"measurement_only_no_tolerance_changes","windows":rows,
            "max_fk_roundtrip_m":max(r["max_fk_roundtrip_m"] for r in rows),
            "rejected_count":sum(not r["gate"]["accepted"] for r in rows)}


def foundation_decision(report,cfg):
    groups = {s:group_decisions(v,cfg) for s,v in report["direct"].items()}
    complete = report.get("completed") and report.get("direct_steps")==200 and len(report.get("windows",[]))==8
    expected = {f"{role}/{width}" for role in ("single_recording","cross_event") for width in (10,28)}
    adequate = set(groups)=={"seen","new_position"} and all(set(g)==expected for g in groups.values())
    direct_pass = adequate and all(row["cases"]==8 and row["passed"] for g in groups.values() for row in g.values())
    expected_baselines = {f"{s}/{r}" for s in ("seen","new_position") for r in ("single_recording","cross_event")}
    baseline_pass = set(report["interpolation_vs_ik"])==expected_baselines and all(
                        row["preserved"]>=math.ceil(cfg.checkpoint_validation_min_stage_repair_rate*row["cases"]) and row["cases"]==16
                        for row in report["interpolation_vs_ik"].values())
    reasons = []
    if not complete: reasons.append("incomplete_or_smoke_protocol")
    if not baseline_pass: reasons.append("contact_ik_baseline_regression")
    if report["roundtrip"]["rejected_count"]: reasons.append("no_edit_roundtrip_rejected")
    if not direct_pass: reasons.append("finite_direct_optimizer_did_not_demonstrate_repair_headroom")
    return {"ready_for_network_diagnostic":not reasons,"reasons":reasons,"groups":groups,
        "scientific_acceptance":False,"publish_allowed":False,
        "failure_is_not_a_proof_of_infeasibility":True}


def check_foundation_report(path,fingerprint,cfg):
    from training.refiner_diagnostics import file_sha256
    report = json.loads(Path(path).read_text(encoding="utf8"))
    if report.get("schema")!=SCHEMA or report.get("fingerprint")!=fingerprint or report.get("published") is not False:
        raise RuntimeError("foundation report protocol/config/code mismatch")
    for item in report["windows"]:
        if file_sha256(item["path"])!=item["sha256"]:
            raise RuntimeError("foundation training window changed")
    decision = foundation_decision(report,cfg)
    if not decision["ready_for_network_diagnostic"]:
        raise RuntimeError(f"foundation control failed: {decision['reasons']}; review per-case metrics before any fitting")
    return report


def run_foundation(args,cfg,banks,pure,recipes,fingerprint,windows,separation):
    from training.refiner_bridge_diagnostics import evaluate
    if not 1 <= args.direct_steps <= 2000:
        raise ValueError("direct_steps must be 1..2000; formal readiness requires exactly 200")
    destination = Path(args.out_dir)
    destination.mkdir(parents=True,exist_ok=False)
    report = {"schema":SCHEMA,"fingerprint":fingerprint,"completed":False,"published":False,
        "direct_steps":args.direct_steps,"windows":windows,"recipes":recipes,"source_separation":separation,
        "direct_control_fits_probe_cases_separately":True,"generalization_evidence":False,
        "clean_identity_in_controls":"no_edit_only_not_a_learned_protection_result",
        "pure_interpolation":{},"interpolation_ik":{},"direct":{},"decoder":{}}
    for split in ("seen","new_position"):
        report["pure_interpolation"][split] = evaluate(None,pure,split,cfg)
        report["interpolation_ik"][split] = evaluate(None,banks,split,cfg)
    report["interpolation_vs_ik"] = baseline_comparison(pure,banks,cfg)
    report["roundtrip"] = roundtrip_diagnostic(banks,cfg)
    m.save_json(report,destination/"foundation_report.json")
    predictions = {}
    for key,bank in banks.items():
        predictions[key],report["decoder"]["/".join(key)] = direct_optimize(bank,cfg,args.direct_steps,
            label="/".join(key),log_path=destination/"direct_gradients.jsonl")
        np.save(destination/("direct_"+"_".join(key)+".npy"),predictions[key].cpu().numpy())
        if key[1]=="cross_event":
            report["direct"][key[0]] = evaluate(None,banks,key[0],cfg,predictions=predictions)
            m.save_json(report,destination/"foundation_report.json")
    report["completed"] = True
    report["decision"] = foundation_decision(report,cfg)
    m.save_json(report,destination/"foundation_report.json")
    print(json.dumps({"stage":"bridge_foundation_complete","report":str(destination/"foundation_report.json"),
                      **report["decision"]}),flush=True)
    return 0 if report["decision"]["ready_for_network_diagnostic"] else 2
