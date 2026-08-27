"""Detached, unit-separated decoder observations. No training decisions here."""
from __future__ import annotations

import numpy as np

from training import motion_models as m


STAGES = ("raw", "after_mask", "after_smoothing", "after_taper", "after_cap", "applied")


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.all(np.isfinite(values)):
        return {"count": int(values.size), "nonfinite": True}
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def ratio(numerator, denominator):
    return (float(numerator / denominator)
            if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 1e-12 else None)


def detached_numpy(trace):
    return {key: value.detach().cpu().numpy() if m.torch.is_tensor(value) else value
            for key, value in trace.items()}


def summarize_window(trace, index, clean, degraded, seam):
    """Statistics on the actual decoder chain, with denominators made explicit."""
    core = np.asarray(seam)[:, 0] >= .5
    halo = (np.asarray(seam)[:, 0] > 0) & ~core
    scopes = {"core": core, "halo": halo, "outside": ~(core | halo)}
    target = m.product_log_np(degraded, clean).astype(np.float64)
    root_mask = np.asarray(trace["root_mask"][index])[:, 0]
    joint_mask = np.asarray(trace["joint_mask"][index])
    masks = {"root_m": root_mask, "rotation_rad": joint_mask}
    blocks = {"root_m": lambda x: x[..., :3],
              "rotation_rad": lambda x: x[..., 3:].reshape(x.shape[:-1] + (24, 3))}
    result = {"schema": "refiner_decoder_trace_v1", "scopes": {k: int(v.sum()) for k, v in scopes.items()}}
    for label, split in blocks.items():
        target_vectors = split(target)
        target_norm = np.linalg.norm(target_vectors, axis=-1)
        stages = {name: split(np.asarray(trace[name][index], dtype=np.float64)) for name in STAGES}
        block = {"target": {k: stats(target_norm[v]) for k, v in scopes.items()}, "stages": {}}
        for name, vectors in stages.items():
            block["stages"][name] = {k: stats(np.linalg.norm(vectors, axis=-1)[v]) for k, v in scopes.items()}
        applied, desired = stages["applied"][core].ravel(), target_vectors[core].ravel()
        applied_size, desired_size = np.linalg.norm(applied), np.linalg.norm(desired)
        block["applied_to_target_norm_ratio"] = ratio(applied_size, desired_size)
        block["applied_target_cosine"] = ratio(float(applied @ desired), applied_size * desired_size)
        block["stage_norm_ratios_core"] = {
            f"{after}_over_{before}": ratio(np.linalg.norm(stages[after][core]), np.linalg.norm(stages[before][core]))
            for before, after in zip(STAGES[:-1], STAGES[1:])
        }
        weight = masks[label]
        block["mask_core"] = stats(weight[core])
        block["mask_by_scope"] = {k: stats(weight[v]) for k, v in scopes.items()}
        block["applied_where_mask_zero"] = stats(
            np.linalg.norm(stages["applied"], axis=-1)[weight == 0]
        )
        block["mask_positive_fraction_core"] = float((weight[core] > 0).mean()) if core.any() else None
        error = target_norm[core]
        block["target_error_uncovered_fraction"] = ratio(error[weight[core] == 0].sum(), error.sum())
        block["target_error_low_weight_fraction"] = ratio(error[weight[core] <= .2].sum(), error.sum())
        block["low_weight_threshold"] = .2
        pre_cap = np.linalg.norm(stages["after_taper"], axis=-1)
        active = np.broadcast_to(core[:, None], weight.shape) & (weight > 0) if weight.ndim == 2 else core & (weight > 0)
        cap = trace["root_cap_m" if label == "root_m" else "rotation_cap_rad"]
        count = int(active.sum())
        block["cap"] = {"limit": cap, "eligible_vectors": count,
                        "clipped_fraction": (float((pre_cap[active] > cap).mean())
                                             if cap is not None and cap > 0 and count else None)}
        result[label] = block
    result["joint_mask_body_groups_core"] = {
        "feet": stats(joint_mask[core][:, list(m.DEFAULT_FOOT_JOINTS)]),
        "lower_body": stats(joint_mask[core][:, list(m.LOWER_BODY_JOINTS)]),
        "non_lower_body": stats(joint_mask[core][:, [i for i in range(24) if i not in m.LOWER_BODY_JOINTS]]),
    }
    return result
