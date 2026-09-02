"""Read-only parameter-gradient attribution for the completed RCSP adapter.

The audit asks why the frozen final single-recording adapter has almost zero
temporal action alignment while the cross-event adapter is positively aligned.
It measures temporal-deficit gradients in each role head's own parameter space
on fixed TRAIN transaction 0 and the fixed seen/new-position final banks.  It
also compares the learned displacement from zero initialization with each
current negative gradient.  No optimizer, update, gradient surgery, width head,
checkpoint selection, production edit, or Pilot is permitted.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch

from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment


SCHEMA = "refiner_rcsp_single_direction_attribution_v1"
RCSP_SOURCE_SCHEMA = "refiner_role_conditioned_support_projection_experiment_v1"
RCSP_SOURCE_COMMIT = "5a344f2950183ceb4c8e938a3c26fa5d76a78c3f"
RCSP_REVIEW_SCHEMA = "refiner_role_conditioned_support_projection_result_review_v1"
TRANSACTION_INDEX = 0
GROUP_SPEC = {
    "single_short": ("single_recording", 10, 0),
    "single_long": ("single_recording", 28, 1),
    "cross_short": ("cross_event", 10, 2),
    "cross_long": ("cross_event", 28, 3),
}
SOURCE_ORDER = ("train_transaction_0", "seen", "new_position")


def _finite(value, label):
    result = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"nonfinite {label}")
    return result


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def _cosine(left, right):
    left = left.detach().cpu().double().reshape(-1)
    right = right.detach().cpu().double().reshape(-1)
    if left.shape != right.shape or not bool(torch.isfinite(left).all()) or not bool(
        torch.isfinite(right).all()
    ):
        raise ValueError("cosine vectors must be finite with identical shapes")
    denominator = _finite(left.norm() * right.norm(), "cosine denominator")
    if denominator == 0.0:
        return None
    return max(-1.0, min(1.0, _finite(torch.dot(left, right), "cosine dot") / denominator))


def _head_parameters(model, role):
    if role not in rcsp.ROLE_MAPPING:
        raise ValueError(f"unknown RCSP role: {role}")
    layer = getattr(model.adapter, "single_adapter" if role == "single_recording" else "cross_adapter")
    named = [(f"{role}.{name}", parameter) for name, parameter in layer.named_parameters()]
    if [name.rsplit(".", 1)[-1] for name, _ in named] != ["weight", "bias"]:
        raise RuntimeError("unexpected role-adapter parameter layout")
    return named


def _flat_parameter_gradient(objective, named, *, retain_graph):
    parameters = [parameter for _, parameter in named]
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = torch.cat(
        [
            (torch.zeros_like(parameter) if gradient is None else gradient)
            .detach()
            .cpu()
            .double()
            .reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )
    displacement = torch.cat(
        [parameter.detach().cpu().double().reshape(-1) for parameter in parameters]
    )
    if not bool(torch.isfinite(flat).all()) or not bool(torch.isfinite(displacement).all()):
        raise FloatingPointError("nonfinite RCSP parameter attribution vector")
    return flat, displacement


def _validate_group_batch(batch):
    if "group" not in batch or "role_id" not in batch:
        raise ValueError("attribution batch requires explicit group and role ids")
    count = int(batch["clean"].shape[0])
    if batch["group"].shape != (count,) or batch["role_id"].shape != (count,):
        raise ValueError("group/role ids do not match attribution batch")
    counts = {label: int((batch["group"] == group_id).sum()) for label, (_, _, group_id) in GROUP_SPEC.items()}
    if not counts or len(set(counts.values())) != 1 or next(iter(counts.values())) < 1:
        raise ValueError(f"attribution groups must be nonempty and balanced: {counts}")
    expected_role = rcsp.role_ids_from_train_groups(batch["group"])
    if not torch.equal(expected_role, batch["role_id"]):
        raise ValueError("explicit role ids disagree with the four-group contract")
    return counts


def parameter_gradient_geometry(model, batch, cfg, source):
    """True unclipped temporal gradients; autograd.grad never populates .grad."""
    counts = _validate_group_batch(batch)
    before_base = safe.state_hash(model.base.state_dict())
    before_adapter = safe.state_hash(model.adapter.state_dict())
    modes = {module: module.training for module in model.modules()}
    grouped = {}
    vectors = {}
    rows = []
    try:
        model.eval()
        with torch.enable_grad():
            role_id, joint_weight, root_weight = rcsp._route_values(batch, cfg)
            with model.route(role_id, joint_weight, root_weight, capture_details=False):
                m._refiner_batch_objectives(
                    model,
                    batch,
                    cfg,
                    group_objectives=grouped,
                )
            if set(grouped) != set(GROUP_SPEC):
                raise RuntimeError("incomplete RCSP group objective geometry")
            for index, label in enumerate(GROUP_SPEC):
                role, width, _group_id = GROUP_SPEC[label]
                objective = grouped[label]["temporal_deficit_mean"]
                named = _head_parameters(model, role)
                gradient, displacement = _flat_parameter_gradient(
                    objective,
                    named,
                    retain_graph=index + 1 < len(GROUP_SPEC),
                )
                key = f"{source}/{role}/{width}"
                vectors[key] = gradient
                rows.append(
                    {
                        "key": key,
                        "source": source,
                        "role": role,
                        "width": width,
                        "cases": counts[label],
                        "temporal_scientific_deficit_mean": _finite(
                            objective, "temporal scientific deficit"
                        ),
                        "parameter_gradient_norm": _finite(
                            gradient.norm(), "parameter gradient norm"
                        ),
                        "learned_parameter_displacement_norm": _finite(
                            displacement.norm(), "parameter displacement norm"
                        ),
                        "learned_displacement_vs_negative_gradient_cosine": _cosine(
                            displacement, -gradient
                        ),
                        "gradient_before_clipping": True,
                        "optimizer_update_performed": False,
                    }
                )
    finally:
        for module, mode in modes.items():
            module.training = mode
    if safe.state_hash(model.base.state_dict()) != before_base:
        raise RuntimeError("parameter attribution changed the frozen base")
    if safe.state_hash(model.adapter.state_dict()) != before_adapter:
        raise RuntimeError("parameter attribution changed the frozen adapter")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("read-only parameter attribution populated .grad")
    return rows, vectors


def final_split_batch(batch, metadata, split):
    indices = [index for index, row in enumerate(metadata) if row["split"] == split]
    if split not in ("seen", "new_position") or len(indices) != 32:
        raise ValueError(f"invalid fixed-final split: {split}")
    device = batch["clean"].device
    selected = torch.tensor(indices, dtype=torch.long, device=device)
    result = {key: value.index_select(0, selected) for key, value in batch.items()}
    selected_metadata = [metadata[index] for index in indices]
    group_ids = []
    for row in selected_metadata:
        match = [
            group_id
            for role, width, group_id in GROUP_SPEC.values()
            if row["role"] == role and row["width"] == width
        ]
        if len(match) != 1:
            raise ValueError("fixed-final metadata does not map to one role/width group")
        group_ids.append(match[0])
    result["group"] = torch.tensor(group_ids, dtype=torch.long, device=device)
    result["role_id"] = rcsp.role_ids_from_metadata(selected_metadata, device)
    _validate_group_batch(result)
    return result


def gradient_cosine_tables(rows, vectors):
    row_by_key = {row["key"]: row for row in rows}
    result = {}
    for role in rcsp.ROLE_MAPPING:
        order = [
            f"{source}/{role}/{width}"
            for source in SOURCE_ORDER
            for width in (10, 28)
        ]
        if any(key not in vectors or key not in row_by_key for key in order):
            raise ValueError(f"missing parameter gradient vector for {role}")
        matrix = [[_cosine(vectors[left], vectors[right]) for right in order] for left in order]
        result[role] = {
            "order": order,
            "gradient_norms": [row_by_key[key]["parameter_gradient_norm"] for key in order],
            "cosine": matrix,
            "same_source_width_10_vs_28": {
                source: matrix[2 * index][2 * index + 1]
                for index, source in enumerate(SOURCE_ORDER)
            },
            "train_to_final_same_width": {
                f"{split}/{width}": matrix[width_index][2 * split_index + width_index]
                for split_index, split in enumerate(("seen", "new_position"), start=1)
                for width_index, width in enumerate((10, 28))
            },
            "negative_cosine_pairs": [
                [order[left], order[right], matrix[left][right]]
                for left in range(len(order))
                for right in range(left + 1, len(order))
                if matrix[left][right] is not None and matrix[left][right] < 0.0
            ],
            "zero_gradient_keys": [
                key for key in order if row_by_key[key]["parameter_gradient_norm"] == 0.0
            ],
        }
    return result


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if float(denominator) != 0.0 else None


def existing_action_support_evidence(report):
    direction = report["direction_alignment"]["summary"]
    support = report["support_projection_stats"]["summary"]
    single = direction["role:single_recording"]
    cross = direction["role:cross_event"]
    single_cosine = single[
        "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
    ]
    cross_cosine = cross[
        "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
    ]
    if support["overall"]["projected_outside_support_max"] != 0.0:
        raise RuntimeError("completed RCSP report contains support escape")
    return {
        "direction_alignment_summary": direction,
        "support_projection_summary": support,
        "descriptive_ratios": {
            "single_to_cross_projected_direction_cosine_ratio": _safe_ratio(
                single_cosine, cross_cosine
            ),
            "cross_width_28_to_width_10_direction_cosine_ratio": _safe_ratio(
                direction["group:seen/cross_event/28"][
                    "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
                ]
                + direction["group:new_position/cross_event/28"][
                    "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
                ],
                direction["group:seen/cross_event/10"][
                    "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
                ]
                + direction["group:new_position/cross_event/10"][
                    "projected_adapter_delta_vs_negative_temporal_gradient_cosine_median"
                ],
            ),
            "single_to_cross_support_retention_ratio": _safe_ratio(
                support["role:single_recording"]["projection_retention_ratio_median"],
                support["role:cross_event"]["projection_retention_ratio_median"],
            ),
            "width_28_to_width_10_support_retention_ratio": _safe_ratio(
                support["width:28"]["projection_retention_ratio_median"],
                support["width:10"]["projection_retention_ratio_median"],
            ),
        },
        "projected_outside_support_max": 0.0,
        "selection_metric": False,
    }


def scientific_answers(rows, tables, existing):
    row_by_key = {row["key"]: row for row in rows}
    single_width_cosines = tables["single_recording"]["same_source_width_10_vs_28"]
    cross_width_cosines = tables["cross_event"]["same_source_width_10_vs_28"]
    single_conflicts = {
        source: value is not None and value < 0.0
        for source, value in single_width_cosines.items()
    }
    single_displacement_non_descent = {
        key: row_by_key[key]["learned_displacement_vs_negative_gradient_cosine"] is not None
        and row_by_key[key]["learned_displacement_vs_negative_gradient_cosine"] <= 0.0
        for key in row_by_key
        if "/single_recording/" in key
    }
    if any(single_conflicts.values()):
        classification = "SINGLE_HEAD_WITHIN_ROLE_WIDTH_GRADIENT_CONFLICT_OBSERVED"
    elif any(single_displacement_non_descent.values()):
        classification = "SINGLE_LEARNED_PARAMETER_DIRECTION_NON_DESCENT_OBSERVED"
    else:
        classification = "SINGLE_DIRECTION_WEAKNESS_NOT_EXPLAINED_BY_SIGN_CONFLICT"
    return {
        "single_direction_attribution": classification,
        "single_same_source_width_gradient_cosine": single_width_cosines,
        "single_same_source_width_gradient_conflict": single_conflicts,
        "cross_same_source_width_gradient_cosine": cross_width_cosines,
        "single_learned_displacement_non_descent": single_displacement_non_descent,
        "all_single_parameter_gradients_nonzero": all(
            row["parameter_gradient_norm"] > 0.0
            for row in rows
            if row["role"] == "single_recording"
        ),
        "single_to_cross_action_direction_cosine_ratio": existing["descriptive_ratios"][
            "single_to_cross_projected_direction_cosine_ratio"
        ],
        "width_28_support_retention_at_least_width_10": (
            existing["support_projection_summary"]["width:28"][
                "projection_retention_ratio_median"
            ]
            >= existing["support_projection_summary"]["width:10"][
                "projection_retention_ratio_median"
            ]
        ),
        "hard_support_escape_observed": False,
        "claim_boundary": (
            "Read-only local gradient attribution at one fixed adapter state. Negative cosine is "
            "a descriptive sign conflict; nonnegative cosine does not prove adequate capacity, "
            "generalization, finite-step effectiveness, or a root cause."
        ),
    }


def _load_rcsp_artifacts(directory, expected_source_commit):
    directory = Path(directory).resolve()
    paths = {
        name: directory / name
        for name in ("report.json", "adapter_final.pt", "reporting_logic_review_v1.json")
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("RCSP result/review artifact set is incomplete")
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    report = json.loads(paths["report.json"].read_text(encoding="utf-8"))
    review = json.loads(paths["reporting_logic_review_v1.json"].read_text(encoding="utf-8"))
    checkpoint = m._trusted_torch_load(paths["adapter_final.pt"], map_location="cpu")
    false_fields = (
        "checkpoint_selection_performed",
        "scale_selection_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    )
    if (
        report.get("schema") != RCSP_SOURCE_SCHEMA
        or report.get("completed") is not True
        or report.get("provenance", {}).get("runtime_commit") != expected_source_commit
        or any(report.get(field) is not False for field in false_fields)
    ):
        raise ValueError("not the completed diagnostic-only RCSP v1 report")
    descriptor = report.get("parameter_update_scope", {}).get("adapter_checkpoint", {})
    if descriptor.get("sha256") != hashes["adapter_final.pt"]:
        raise ValueError("RCSP adapter checkpoint hash mismatch")
    if (
        checkpoint.get("schema") != RCSP_SOURCE_SCHEMA
        or checkpoint.get("completed_steps") != rcsp.STEPS
        or checkpoint.get("formal_checkpoint") is not False
        or checkpoint.get("production_model_modified") is not False
        or checkpoint.get("checkpoint_selection_performed") is not False
        or checkpoint.get("publish_allowed") is not False
        or checkpoint.get("pilot_allowed") is not False
        or checkpoint.get("resume_allowed") is not False
    ):
        raise ValueError("invalid diagnostic-only RCSP adapter checkpoint")
    if (
        review.get("schema") != RCSP_REVIEW_SCHEMA
        or review.get("completed") is not True
        or review.get("source_report", {}).get("sha256") != hashes["report.json"]
        or review.get("measurement_recomputation_verified") is not True
        or review.get("formal_conclusion", {}).get("classification")
        != "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
        or review.get("formal_conclusion", {}).get("role_conditioning_alone_sufficient")
        is not False
        or review.get("production_model_modified") is not False
        or review.get("scientific_acceptance") is not False
        or review.get("pilot_allowed") is not False
    ):
        raise ValueError("RCSP reporting-logic review is missing or mismatched")
    return directory, paths, hashes, report, review, checkpoint


def run(args):
    source = Path(args.state_dir).resolve()
    trajectory, traj_paths, traj_hashes, traj_report, experiment, base_checkpoint = (
        failure._load_trajectory(args.trajectory_dir, args.expected_trajectory_commit)
    )
    rcsp_dir, rcsp_paths, rcsp_hashes, rcsp_report, rcsp_review, adapter_checkpoint = (
        _load_rcsp_artifacts(args.rcsp_dir, args.expected_rcsp_commit)
    )
    output = Path(args.output_dir).resolve()
    if (
        output.exists()
        or output.is_relative_to(source)
        or output.is_relative_to(trajectory)
        or output.is_relative_to(rcsp_dir)
    ):
        raise FileExistsError("attribution output must be a fresh directory outside immutable inputs")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source,
        group_audit.LEGACY_COMMIT,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength,
    )
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference the supplied frozen source")
    runtime_commit = m._training_code_revision()
    if runtime_commit != args.expected_main_commit:
        raise ValueError("runtime commit does not match --expected-main-commit")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    cfg = dataclasses.replace(cfg, device=str(device))
    source_paths = {
        name: source / name
        for name in ("diagnostic_report.json", "diagnostic_state.pt", "fit_bank.pt", "probe_bank.pt")
    }
    source_hashes = {name: _file_sha256(path) for name, path in source_paths.items()}
    output.mkdir(parents=True, exist_ok=False)
    failure_path = output / "failure.json"
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices), group_audit.frozen_environment(
            state["fingerprint"], source_metadata["decoder_strengths"]
        ):
            base = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
            base.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
            base.eval()
            base_hash = safe.state_hash(base.state_dict())
            if base_hash != traj_report["final_state_sha256"]:
                raise RuntimeError("loaded base does not match immutable trajectory final state")
            if adapter_checkpoint.get("base_state_sha256") != base_hash:
                raise ValueError("RCSP adapter does not reference the loaded frozen base")
            model = rcsp.FrozenBaseRCSPModel(base)
            model.adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
            rcsp.validate_parameter_scope(model)
            adapter_hash = safe.state_hash(model.adapter.state_dict())
            if adapter_hash != rcsp_report["parameter_update_scope"]["adapter_state_sha256"]:
                raise RuntimeError("loaded adapter state does not match completed RCSP report")
            initial_base_hash, initial_adapter_hash = base_hash, adapter_hash

            train = rcsp.attach_train_role_ids(
                rcsp._move_batch(
                    group_audit.materialize_transaction(bank, cfg, TRANSACTION_INDEX), device
                )
            )
            probe, probe_hash = safe.load_probe(source, state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(
                failure.final_banks(bank, probe, cfg)
            )
            final_batch = rcsp._move_batch(final_batch, device)

            rows, vectors = parameter_gradient_geometry(
                model, train, cfg, "train_transaction_0"
            )
            for split in ("seen", "new_position"):
                split_rows, split_vectors = parameter_gradient_geometry(
                    model,
                    final_split_batch(final_batch, final_metadata, split),
                    cfg,
                    split,
                )
                rows.extend(split_rows)
                vectors.update(split_vectors)
            tables = gradient_cosine_tables(rows, vectors)
            existing = existing_action_support_evidence(rcsp_report)
            answers = scientific_answers(rows, tables, existing)
            if safe.state_hash(model.base.state_dict()) != initial_base_hash:
                raise RuntimeError("final attribution changed frozen base state")
            if safe.state_hash(model.adapter.state_dict()) != initial_adapter_hash:
                raise RuntimeError("final attribution changed frozen adapter state")
            if _file_sha256(source_paths["probe_bank.pt"]) != probe_hash:
                raise RuntimeError("probe artifact changed during read-only attribution")

        for name, digest in source_hashes.items():
            if _file_sha256(source_paths[name]) != digest:
                raise RuntimeError("frozen source changed during attribution")
        for name, digest in traj_hashes.items():
            if _file_sha256(traj_paths[name]) != digest:
                raise RuntimeError("immutable trajectory changed during attribution")
        for name, digest in rcsp_hashes.items():
            if _file_sha256(rcsp_paths[name]) != digest:
                raise RuntimeError("completed RCSP artifact changed during attribution")

        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "source": source_metadata,
                "source_sha256_including_probe": source_hashes,
                "trajectory_commit": args.expected_trajectory_commit,
                "trajectory_directory": str(trajectory),
                "trajectory_sha256": traj_hashes,
                "rcsp_commit": args.expected_rcsp_commit,
                "rcsp_directory": str(rcsp_dir),
                "rcsp_sha256": rcsp_hashes,
                "rcsp_review_classification": rcsp_review["formal_conclusion"]["classification"],
            },
            "fixed_state": {
                "base_state_sha256": initial_base_hash,
                "adapter_state_sha256": initial_adapter_hash,
                "adapter_completed_steps": rcsp.STEPS,
                "adapter_initialization_reference": "exact_zero",
            },
            "audit_data": {
                "train_transaction_index": TRANSACTION_INDEX,
                "train_context_indices": list(bank["transaction_schedule"][TRANSACTION_INDEX]),
                "train_cases": 192,
                "seen_cases": 32,
                "new_position_cases": 32,
                "probe_used_for_read_only_attribution_only": True,
            },
            "parameter_gradient_rows": rows,
            "within_role_parameter_gradient_geometry": tables,
            "existing_action_and_support_evidence": existing,
            "scientific_answers": answers,
            "gradient_protocol": {
                "objective": "mean temporal_scientific_deficit per role-width group",
                "space": "corresponding_role_adapter_weight_and_bias",
                "before_clipping": True,
                "autograd_grad_only": True,
                "optimizer_constructed": False,
                "gradient_surgery_performed": False,
                "parameter_update_performed": False,
            },
            "optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "scale_selection_performed": False,
            "width_conditioning_added": False,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "next_action": "review_single_direction_attribution_no_pilot",
        }
        report_path = output / "report.json"
        _exclusive_json(report_path, report)
        print("RCSP SINGLE-DIRECTION ATTRIBUTION", flush=True)
        print(json.dumps(answers, ensure_ascii=False, allow_nan=False), flush=True)
        for role in rcsp.ROLE_MAPPING:
            print(
                json.dumps(
                    {
                        "stage": "within_role_parameter_gradient_geometry",
                        "role": role,
                        **tables[role],
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "stage": "refiner_rcsp_single_direction_attribution_complete",
                    "report": str(report_path),
                    "optimizer_steps": 0,
                    "parameter_update_performed": False,
                    "production_model_modified": False,
                    "scientific_acceptance": False,
                    "pilot_allowed": False,
                },
                allow_nan=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        if not failure_path.exists():
            _exclusive_json(
                failure_path,
                {
                    "schema": SCHEMA,
                    "completed": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "optimizer_steps": 0,
                    "parameter_update_performed": False,
                    "production_model_modified": False,
                    "scientific_acceptance": False,
                    "pilot_allowed": False,
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--rcsp-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument("--expected-rcsp-commit", default=RCSP_SOURCE_COMMIT)
    parser.add_argument(
        "--expected-trajectory-commit", default=failure.TRAJECTORY_COMMIT
    )
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
