"""TRAIN-window interpolation/IK and direct-output feasibility controls.

Direct optimization fits each case separately (including the probe positions).
It is an optimistic reachability control, NOT generalization, a learned model,
or a proof of infeasibility when a finite optimizer fails. No formal weights are
written. Repair thresholds, masks, smoothing, caps and safety checks are shared.
"""
from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path

import numpy as np

from training import motion_models as m

SCHEMA = "bridge_foundation_feasibility_v4"
DIRECT_SAFETY_PROTOCOL = "input_relative_safe_line_search_v1"
DIRECT_OPTIMIZER_PROTOCOL = "per_case_sum_descent_backtracking_v1"
DIRECT_BACKTRACK_STEPS = 24
DIRECT_STALL_PATIENCE = 3


def _gradient_direction(gradient):
    """Steepest descent with the same maximum coordinate step as Adam.

    Unlike an Adam proposal containing historical momentum, this direction is
    downhill for every nonzero, finite current per-case gradient.
    """
    maximum = gradient.abs().flatten(1).amax(1).clamp_min(1e-12)
    return -0.003 * gradient / maximum[:, None, None]


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


class _DirectSafetyChecker:
    """Audit only loss-improving trials; retain the ORIGINAL input as reference.

    Reference audits and bounded exact-content decisions are cached per case.
    This avoids repeating unchanged CPU audits in a CUDA backtracking loop;
    no approximation, batch-average safety or progressively relaxed budget is
    used. Final candidates are freshly audited, without cache reuse.
    """
    def __init__(self, reference, cfg):
        self.reference = reference.detach().cpu().numpy().copy()
        self.cfg = cfg
        self.before = [m._safe_validation_audit(x, cfg, role="direct_reference",
            support_policy="source_observation") for x in self.reference]
        self.cache = [{} for _ in self.reference]
        self.audit_calls = 0
        self.cache_hits = 0

    def check(self, index, candidate, *, fresh=False):
        key = hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).digest()
        cache = self.cache[index]
        if not fresh and key in cache:
            self.cache_hits += 1
            return cache[key]
        self.audit_calls += 1
        try:
            gate = m._fixed_support_stage_gate(self.reference[index], candidate, self.cfg,
                before_audit=self.before[index])
            reasons = list(gate["reasons"])
            accepted = bool(gate["accepted"])
            # A known physical rejection needs no additional SVD/log-map work.
            if accepted:
                _, accepted = m._observable_reference_fidelity(self.reference[index], candidate, self.cfg)
                if not accepted:
                    reasons.append("reference_geometry_budget_exceeded")
            result = (bool(accepted), reasons)
        except (ValueError, RuntimeError, FloatingPointError):
            result = (False, ["invalid_direct_candidate_audit"])
        if len(cache) >= 128:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result


def direct_optimize(bank,cfg,steps, *, label,log_path):
    """One free output tensor per case; no shared network, no hidden clean input."""
    torch = m.torch
    output = torch.nn.Parameter(bank["bad"].new_zeros((*bank["bad"].shape[:-1],m.PRODUCT_STATE_DIM)))
    optimizer = torch.optim.Adam([output],lr=0.003)
    masks = m._refiner_decode_masks(bank["joint"],bank["root"],bank["contact"],bank["seam"],cfg)
    started = time.perf_counter()
    safety = _DirectSafetyChecker(bank["bad"], cfg)
    safe_updates = np.zeros(len(output), dtype=int)
    unsafe_trials = np.zeros(len(output), dtype=int)
    rejection_reasons = [{} for _ in output]
    loss_rejections = torch.zeros(len(output), device=output.device, dtype=torch.long)
    resolution_rejections = torch.zeros_like(loss_rejections)
    fallback_attempts = torch.zeros_like(loss_rejections)
    fallback_updates = torch.zeros_like(loss_rejections)
    non_descent_steps = torch.zeros_like(loss_rejections)
    attempted_steps = torch.zeros_like(loss_rejections)
    target_satisfied = torch.zeros(len(output), device=output.device, dtype=torch.bool)
    search_stalled = torch.zeros_like(target_satisfied)
    stalled_steps = torch.zeros_like(loss_rejections)
    norm = output.new_zeros(len(output))
    last_gradient_norm = norm.clone()
    next_scale = torch.ones(len(output), device=output.device, dtype=output.dtype)
    for i, reference in enumerate(safety.reference):
        accepted, reasons = safety.check(i, reference)
        if not accepted:
            raise RuntimeError(f"direct no-edit reference {i} failed safety: {reasons}")
    print(json.dumps({"stage":"direct_safety_preflight", "group":label,
        "cases":len(output), "protocol":DIRECT_SAFETY_PROTOCOL,
        "optimizer_protocol":DIRECT_OPTIMIZER_PROTOCOL}), flush=True)
    for step in range(steps):
        prediction = m._decode_product_refiner_output(bank["bad"],output,*masks,cfg)
        losses,terms = m._observable_refiner_objective(prediction,bank["bad"],bank["seam"],cfg,reduction="none")
        loss = losses.mean()
        # A reachability control need not keep optimizing already achieved
        # TRAINING targets (10%, not the weaker 3% evaluation threshold).
        # Every current state is already safe. Freeze such cases, never use
        # probe pass/fail labels for stopping, and report actual attempted steps.
        if "endpoint_relative_gap" in terms and "temporal_relative_gap" in terms:
            achieved = torch.ones_like(target_satisfied)
            for key in ("endpoint_relative_gap", "temporal_relative_gap", "jerk",
                        "jerk_safety_excess", "support_excess", "observable_trust_excess"):
                achieved &= torch.isfinite(terms[key]) & (terms[key] == 0)
            target_satisfied |= achieved
        active = ~(target_satisfied | search_stalled)
        if not bool(active.any()):
            print(json.dumps({"stage":"direct_search_complete", "group":label,
                "budget":steps, "iterations":step, "cases":len(output),
                "target_satisfied_cases":int(target_satisfied.sum()),
                "search_stalled_cases":int(search_stalled.sum())}),flush=True)
            break
        attempted_steps += active.long()
        optimizer.zero_grad(set_to_none=True)
        # Independent free parameters: averaging before Adam made its epsilon
        # and per-case clipping depend on how many OTHER cases were present.
        # Report a mean, but differentiate the sum for this direct control.
        losses[active].sum().backward()
        # Independent parameters; clipping is per case, never across cases.
        flat = output.grad.flatten(1)
        norm = torch.linalg.vector_norm(flat,dim=1)
        if not torch.isfinite(norm).all():
            raise RuntimeError("nonfinite direct-optimization gradient")
        last_gradient_norm[active] = norm[active]
        output.grad.mul_((1/norm.clamp_min(1)).view(-1,1,1))
        old = output.detach().clone()
        optimizer.step()
        # A fixed Adam step can overshoot quiet windows by orders of magnitude.
        # A lower loss is necessary but insufficient: every accepted update
        # must also pass the exact physical/fidelity audit against bank['bad'].
        # A rejected trial leaves the last safe candidate intact.
        update = output.detach()-old
        descent = (output.grad * update).flatten(1).sum(1)
        steepest = _gradient_direction(output.grad)
        use_gradient = active & (~torch.isfinite(descent) | (descent >= 0))
        update[use_gradient] = steepest[use_gradient]
        non_descent_steps += use_gradient.long()
        fallback_attempts += use_gradient.long()
        selected = old.clone()
        pending = active.clone()
        scale = next_scale.clone()
        scale[use_gradient] = 1
        accepted_scale = torch.zeros_like(scale)
        used_fallback = use_gradient.clone()
        safety_rejections = 0
        gpu_jerk_rejections = 0
        with torch.no_grad():
            # Retry an exhausted Adam direction with CURRENT steepest descent.
            # More halving of an uphill momentum direction cannot fix it.
            for search in range(2):
                searching = pending.clone()
                if search:
                    searching &= ~used_fallback
                    update[searching] = steepest[searching]
                    scale[searching] = 1
                    used_fallback |= searching
                    fallback_attempts += searching.long()
                for _ in range(DIRECT_BACKTRACK_STEPS):
                    indices = searching.nonzero(as_tuple=False).flatten()
                    if not indices.numel():
                        break
                    # Only unfinished cases need more decoder/FK/quantile work.
                    trial = old[indices] + update[indices]*scale[indices,None,None]
                    motion = m._decode_product_refiner_output(bank["bad"][indices],trial,
                        *(mask[indices] for mask in masks),cfg)
                    candidate,candidate_terms = m._observable_refiner_objective(motion,
                        bank["bad"][indices],bank["seam"][indices],cfg,reduction="none")
                    changed = (motion != prediction.detach()[indices]).flatten(1).any(1)
                    # Exact no-ops and equal loss are NOT successful updates.
                    # Never accept a small increase hidden by a +1e-10 slack.
                    improves = torch.isfinite(candidate) & (candidate < losses.detach()[indices])
                    loss_rejections[indices] += (~improves).long()
                    resolution_rejections[indices] += (~changed).long()
                    accepted = improves & changed
                    tail_rejected = accepted & (candidate_terms["jerk_safety_excess"] > 0)
                    for index in indices[tail_rejected].cpu().tolist():
                        unsafe_trials[index] += 1
                        counts = rejection_reasons[index]
                        counts["gpu_jerk_budget_exceeded"] = counts.get("gpu_jerk_budget_exceeded", 0) + 1
                        gpu_jerk_rejections += 1
                    accepted &= ~tail_rejected
                    local_indices = accepted.nonzero(as_tuple=False).flatten()
                    candidates = motion[local_indices].cpu().numpy()
                    for local, index, proposed in zip(local_indices.cpu().tolist(), indices[local_indices].cpu().tolist(), candidates):
                        is_safe, reasons = safety.check(index, proposed)
                        if not is_safe:
                            accepted[local] = False
                            safety_rejections += 1
                            unsafe_trials[index] += 1
                            for reason in reasons:
                                counts = rejection_reasons[index]
                                counts[reason] = counts.get(reason, 0) + 1
                        else:
                            safe_updates[index] += 1
                    chosen = indices[accepted]
                    selected[chosen] = trial[accepted]
                    accepted_scale[chosen] = scale[chosen]
                    fallback_updates[chosen] += used_fallback[chosen].long()
                    pending[chosen] = False
                    searching[chosen] = False
                    # Below storage resolution, further halving the same
                    # direction cannot create a meaningful motion update.
                    searching[indices[~changed]] = False
                    scale[searching] *= .5
            output.copy_(selected)
            # At an unchanged state, retrying both exhausted directions with
            # reset moments hundreds of times is not additional repair evidence.
            # Stop only this finite search; NEVER turn a stalled case into a pass.
            stalled_steps = torch.where(pending, stalled_steps+1, torch.zeros_like(stalled_steps))
            search_stalled |= pending & (stalled_steps >= DIRECT_STALL_PATIENCE)
            next_scale = torch.where(pending, torch.ones_like(scale),
                (2 * accepted_scale).clamp(min=2.0**-(DIRECT_BACKTRACK_STEPS-1),max=1))
            for state_key in ("exp_avg","exp_avg_sq"):
                optimizer.state[output][state_key][pending | used_fallback] = 0
        if step==0 or (step+1)%25==0 or step+1==steps:
            elapsed = time.perf_counter()-started
            row = {"stage":"direct_bridge_optimization","group":label,"step":step+1,"steps":steps,
                "loss":float(loss.detach()),"endpoint_loss":float(terms["endpoint_continuity"].detach().mean()),
                "temporal_loss":float(terms["temporal_supervision"].detach().mean()),"support_loss":float(terms["support_excess"].detach().mean()),
                "line_search_rejected_cases":int(pending.sum()),"line_search_min_scale":float(scale.min()),
                "jerk_safety_loss":float(terms["jerk_safety_excess"].detach().mean()),
                "safety_rejected_trials":safety_rejections,
                "gpu_jerk_rejected_trials":gpu_jerk_rejections,
                "safety_audit_calls":safety.audit_calls,"safety_cache_hits":safety.cache_hits,
                "optimizer_protocol":DIRECT_OPTIMIZER_PROTOCOL,
                "adam_non_descent_cases":int(use_gradient.sum()),
                "gradient_fallback_cases":int(used_fallback.sum()),
                "target_satisfied_cases":int(target_satisfied.sum()),
                "search_stalled_cases":int(search_stalled.sum()),
                "active_cases":int(active.sum()),
                "loss_rejected_trials_total":int(loss_rejections.sum()),
                "resolution_limited_trials_total":int(resolution_rejections.sum()),
                "elapsed_seconds":elapsed,"eta_seconds":elapsed/(step+1)*(steps-step-1)}
            with Path(log_path).open("a",encoding="utf8") as handle:
                handle.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps(row,allow_nan=False),flush=True)
    with torch.no_grad():
        trace = {}
        prediction = m._decode_product_refiner_output(bank["bad"],output,*masks,cfg,trace=trace)
    summary = decoder_summary(trace,bank["seam"])
    for i, candidate in enumerate(prediction.cpu().numpy()):
        accepted, reasons = safety.check(i, candidate, fresh=True)
        if not accepted:
            raise RuntimeError(f"retained direct candidate {i} failed final safety: {reasons}")
        summary[i].update(safety_protocol=DIRECT_SAFETY_PROTOCOL, safety_accepted=True,
            optimizer_protocol=DIRECT_OPTIMIZER_PROTOCOL,
            safe_update_count=int(safe_updates[i]), unsafe_trial_count=int(unsafe_trials[i]),
            unsafe_trial_reasons=rejection_reasons[i],
            loss_rejected_trial_count=int(loss_rejections[i]),
            resolution_limited_trial_count=int(resolution_rejections[i]),
            non_descent_adam_steps=int(non_descent_steps[i]),
            gradient_fallback_attempts=int(fallback_attempts[i]),
            gradient_fallback_updates=int(fallback_updates[i]),
            last_pre_update_gradient_norm=float(last_gradient_norm[i]),
            attempted_optimizer_steps=int(attempted_steps[i]),
            target_satisfied=bool(target_satisfied[i]),
            search_stalled=bool(search_stalled[i]),
            retained_no_edit=bool(np.array_equal(candidate, safety.reference[i])))
    return prediction.detach(),summary


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
    """Require exact zero-edit values and audit them; NEVER enlarge tolerances."""
    rows = []
    for key,bank in banks.items():
        with m.torch.no_grad():
            output = bank["bad"].new_zeros((*bank["bad"].shape[:-1],m.PRODUCT_STATE_DIM))
            decoded = m._decode_product_refiner_output(bank["bad"],output,
                *m._refiner_decode_masks(bank["joint"],bank["root"],bank["contact"],bank["seam"],cfg),cfg)
        for case_index,(reference,candidate) in enumerate(zip(bank["bad"].cpu().numpy(),decoded.cpu().numpy())):
            gate = m._fixed_support_stage_gate(reference,candidate,cfg)
            error = float(np.abs(reference-candidate).max())
            rows.append({"group":"/".join(key),"case_index":case_index,"gate":gate,
                "exact_identity":bool(np.array_equal(reference,candidate)),
                "changed_values":int(np.count_nonzero(reference!=candidate)),
                "max_motion_abs_error":error if np.isfinite(error) else None,
                "max_fk_roundtrip_m":float(np.abs(m.fk_24_np(reference)-m.fk_24_np(candidate)).max())})
    return {"policy":"measurement_only_no_tolerance_changes","windows":rows,
            "cases":len(rows),"exact_identity_count":sum(r["exact_identity"] for r in rows),
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
    if report["roundtrip"].get("cases")!=64 or report["roundtrip"].get("exact_identity_count")!=64:
        reasons.append("no_edit_decode_not_identity")
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
        "direct_safety_protocol":DIRECT_SAFETY_PROTOCOL,
        "direct_optimizer_protocol":DIRECT_OPTIMIZER_PROTOCOL,
        "observable_objective_protocol":m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL,
        "direct_steps":args.direct_steps,"windows":windows,"recipes":recipes,"source_separation":separation,
        "direct_steps_semantics":"maximum_per_case_budget_stop_at_training_targets_or_stalled_search",
        "direct_control_fits_probe_cases_separately":True,"generalization_evidence":False,
        "clean_identity_in_controls":"no_edit_only_not_a_learned_protection_result",
        "pure_interpolation":{},"interpolation_ik":{},"direct":{},"decoder":{}}
    # Run the no-edit contract before spending the direct-optimization budget.
    # Keep every failure's metrics, not merely a final aggregate reason.
    report["roundtrip"] = roundtrip_diagnostic(banks,cfg)
    rt = report["roundtrip"]
    print(json.dumps({"stage":"bridge_zero_edit_preflight",**{
        k:rt[k] for k in ("cases","exact_identity_count","rejected_count","max_fk_roundtrip_m")}},allow_nan=False),flush=True)
    if rt["rejected_count"] or rt["exact_identity_count"]!=rt["cases"]:
        report["blocked_before_direct_optimization"] = True
        report["failure_reasons"] = ["no_edit_roundtrip_rejected"]
        m.save_json(report,destination/"foundation_report.json")
        print(json.dumps({"stage":"bridge_foundation_blocked","report":str(destination/"foundation_report.json"),
            "ready_for_network_diagnostic":False,"reasons":report["failure_reasons"]}),flush=True)
        return 2
    for split in ("seen","new_position"):
        report["pure_interpolation"][split] = evaluate(None,pure,split,cfg)
        report["interpolation_ik"][split] = evaluate(None,banks,split,cfg)
    report["interpolation_vs_ik"] = baseline_comparison(pure,banks,cfg)
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
