import os
import torch


_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}, "stripped_module_prefix"
    return state_dict, "unchanged"


def _add_module_prefix(state_dict):
    return {f"module.{key}": value for key, value in state_dict.items()}


def _align_prefix_to_reference(state_dict, reference_state_dict):
    if set(state_dict.keys()) == set(reference_state_dict.keys()):
        return state_dict, "unchanged"

    stripped, action = _strip_module_prefix(state_dict)
    if set(stripped.keys()) == set(reference_state_dict.keys()):
        return stripped, action

    prefixed = _add_module_prefix(state_dict)
    if set(prefixed.keys()) == set(reference_state_dict.keys()):
        return prefixed, "added_module_prefix"

    if reference_state_dict and all(key.startswith("module.") for key in reference_state_dict.keys()):
        prefixed = _add_module_prefix(stripped)
        return prefixed, "added_module_prefix"
    return stripped, action


def _spectral_norm_candidate(key):
    if key.endswith(".weight"):
        return f"{key[:-len('.weight')]}.parametrizations.weight.original"
    marker = ".parametrizations.weight.original"
    if key.endswith(marker):
        return f"{key[:-len(marker)]}.weight"
    return None


def _is_spectral_norm_buffer(key):
    return (
        ".parametrizations.weight.0._u" in key
        or ".parametrizations.weight.0._v" in key
    )


def _clone_reference_value(value):
    """Clone freshly initialized reference tensors without duplicating them on GPU.

    The old implementation used value.detach().clone(), which clones GPU tensors
    on GPU when the model has already been moved to cuda. During V10 inference
    this can duplicate a large part of the model state and trigger OOM before
    checkpoint loading finishes.

    CPU tensors are accepted by torch.nn.Module.load_state_dict; PyTorch copies
    them into the target parameter/buffer device during loading. Keeping these
    fallback initialized values on CPU dramatically lowers peak GPU memory.
    """
    if not torch.is_tensor(value):
        return value

    detach = value.detach()
    if _env_bool("EDGE_CHECKPOINT_COMPAT_CPU_MERGE", True):
        return detach.cpu().clone()
    return detach.clone()


def adapt_checkpoint_state_dict(checkpoint_state_dict, model, log_prefix="checkpoint"):
    """Return a strict-loadable state_dict, adapting old Linear weights to spectral_norm keys.

    Newer PyTorch parametrizations store spectral_norm weights under
    ``*.parametrizations.weight.original`` plus ``_u/_v`` buffers. Older checkpoints only
    have ``*.weight``. This helper maps the original weights and keeps any new buffers from
    the freshly initialized model so strict loading can still be used.

    V10 hotfix:
    - Fallback initialized reference tensors are cloned on CPU by default via
      EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1.
    - This avoids a second GPU copy of model.state_dict() during checkpoint
      adaptation, reducing peak GPU memory at inference startup.
    """
    if checkpoint_state_dict is None:
        raise ValueError("checkpoint_state_dict is None")

    reference = model.state_dict()
    source, prefix_action = _align_prefix_to_reference(dict(checkpoint_state_dict), reference)
    merged = {key: _clone_reference_value(value) for key, value in reference.items()}

    loaded = []
    remapped = []
    skipped_shape = []
    unexpected = []

    for key, value in source.items():
        target_key = key
        if target_key not in reference:
            candidate = _spectral_norm_candidate(key)
            if candidate in reference:
                target_key = candidate
                remapped.append((key, target_key))
            else:
                unexpected.append(key)
                continue

        if torch.is_tensor(value) and torch.is_tensor(reference[target_key]):
            if tuple(value.shape) != tuple(reference[target_key].shape):
                skipped_shape.append((key, tuple(value.shape), tuple(reference[target_key].shape)))
                continue

        # Keep checkpoint tensors as provided. They are usually loaded on CPU
        # already; load_state_dict will copy them to the model device.
        merged[target_key] = value
        loaded.append(target_key)

    loaded_set = set(loaded)
    kept_initialized = [key for key in reference if key not in loaded_set]
    kept_spectral_buffers = [key for key in kept_initialized if _is_spectral_norm_buffer(key)]
    kept_other = [key for key in kept_initialized if not _is_spectral_norm_buffer(key)]

    report = {
        "log_prefix": log_prefix,
        "prefix_action": prefix_action,
        "loaded_count": len(loaded),
        "remapped_count": len(remapped),
        "remapped": remapped,
        "skipped_shape": skipped_shape,
        "unexpected": unexpected,
        "kept_spectral_buffers": kept_spectral_buffers,
        "kept_other": kept_other,
        "cpu_merge": bool(_env_bool("EDGE_CHECKPOINT_COMPAT_CPU_MERGE", True)),
    }
    return merged, report


def summarize_adapt_report(report, max_items=8):
    lines = []
    prefix = report.get("log_prefix", "checkpoint")
    if report.get("prefix_action") != "unchanged":
        lines.append(f"{prefix}: prefix_action={report['prefix_action']}")
    if report.get("cpu_merge"):
        lines.append(f"{prefix}: checkpoint compatibility merge uses CPU fallback tensors")
    if report.get("remapped_count", 0):
        lines.append(f"{prefix}: remapped spectral_norm weights={report['remapped_count']}")
    if report.get("kept_spectral_buffers"):
        lines.append(f"{prefix}: initialized spectral_norm buffers={len(report['kept_spectral_buffers'])}")
    if report.get("kept_other"):
        sample = report["kept_other"][:max_items]
        lines.append(f"{prefix}: kept newly initialized keys={sample}")
    if report.get("skipped_shape"):
        sample = report["skipped_shape"][:max_items]
        lines.append(f"{prefix}: skipped shape-mismatch keys={sample}")
    if report.get("unexpected"):
        sample = report["unexpected"][:max_items]
        lines.append(f"{prefix}: ignored unexpected keys={sample}")
    return lines
