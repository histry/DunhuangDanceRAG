"""Read-only anatomy-by-time decomposition of the frozen RCSP action direction.

All fixed-final 64 cases are evaluated with the completed step-400 A0 base and
RCSP adapter.  The authoritative temporal objective is differentiated with
respect to a detached copy of the raw 75D geometric action.  Parameters remain
frozen and parameter-space attribution is lineage/context only.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from motion_geometry.physical import EXTREMITY_JOINTS
from training import motion_models as m
from training import refiner_final_failure_audit as failure
from training import refiner_group_gradient_audit as group_audit
from training import refiner_rcsp_single_direction_attribution as parameter_audit
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_safe_start_diagnostics as safe
from training import refiner_temporal_action_alignment_audit as alignment


SCHEMA = "refiner_single_direction_decomposition_audit_v1"
REVIEWED_MAIN_BASELINE = "c2ceea1bfc449e51f697577ba2ec2dce9a70d699"
RCSP_SOURCE_COMMIT = "5a344f2950183ceb4c8e938a3c26fa5d76a78c3f"
PARAMETER_ATTRIBUTION_SCHEMA = "refiner_rcsp_single_direction_attribution_v1"
FINAL_CASES = 64
FINAL_CHUNK_SIZE = 8
ACTION_NAMES = ("base", "adapter", "total")
ANATOMY_NAMES = ("root", "body", "extremity")
TIME_NAMES = ("early", "center", "late")
PARTITION_ALGORITHM = "numpy.array_split_ordered_active_frame_indices_into_3"
PARITY_ATOL = 2.0e-6


def _anatomy_partition():
    extremity_joints = tuple(int(index) for index in EXTREMITY_JOINTS)
    body_joints = tuple(index for index in range(m.NUM_JOINTS) if index not in extremity_joints)
    root_coordinates = tuple(range(3))
    body_coordinates = tuple(
        coordinate
        for joint in body_joints
        for coordinate in range(3 + 3 * joint, 6 + 3 * joint)
    )
    extremity_coordinates = tuple(
        coordinate
        for joint in extremity_joints
        for coordinate in range(3 + 3 * joint, 6 + 3 * joint)
    )
    if set(body_joints) & set(extremity_joints):
        raise RuntimeError("body and extremity joints overlap")
    if set(body_joints) | set(extremity_joints) != set(range(m.NUM_JOINTS)):
        raise RuntimeError("body/extremity partition does not cover all joints")
    coordinates = {
        "root": root_coordinates,
        "body": body_coordinates,
        "extremity": extremity_coordinates,
    }
    flattened = [coordinate for name in ANATOMY_NAMES for coordinate in coordinates[name]]
    if len(flattened) != 75 or set(flattened) != set(range(75)):
        raise RuntimeError("anatomy partition does not cover 75D geometry exactly once")
    authoritative = {
        "root": tuple(alignment.GEOMETRY_BLOCKS["root_translation"]),
        "body": tuple(alignment.GEOMETRY_BLOCKS["body_joints"]),
        "extremity": tuple(alignment.GEOMETRY_BLOCKS["extremity_joints"]),
    }
    if coordinates != authoritative:
        raise RuntimeError("anatomy partition differs from authoritative alignment audit")
    return {
        "partition_source": (
            "motion_geometry.physical.EXTREMITY_JOINTS and "
            "training.refiner_temporal_action_alignment_audit.GEOMETRY_BLOCKS"
        ),
        "num_joints": m.NUM_JOINTS,
        "root_coordinate_indices": list(root_coordinates),
        "body_joint_indices": list(body_joints),
        "extremity_joint_indices": list(extremity_joints),
        "body_coordinate_indices": list(body_coordinates),
        "extremity_coordinate_indices": list(extremity_coordinates),
        "coordinates": coordinates,
        "complete_disjoint_75d": True,
    }


ANATOMY_PARTITION = _anatomy_partition()
ANATOMY_COORDINATES = ANATOMY_PARTITION["coordinates"]


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


def temporal_partition(active_frame_indices):
    active = [int(index) for index in active_frame_indices]
    if not active:
        raise ValueError("active_frame_count == 0")
    if active != sorted(active) or len(active) != len(set(active)):
        raise ValueError("active frame indices must be ordered and unique")
    parts = np.array_split(np.asarray(active, dtype=np.int64), 3)
    result = {name: [int(value) for value in part.tolist()] for name, part in zip(TIME_NAMES, parts)}
    flattened = [value for name in TIME_NAMES for value in result[name]]
    if flattened != active or any(
        set(result[left]) & set(result[right])
        for index, left in enumerate(TIME_NAMES)
        for right in TIME_NAMES[index + 1 :]
    ):
        raise RuntimeError("early/center/late partition is not complete and disjoint")
    return result


def contribution_stats(action, gradient):
    if action.shape != gradient.shape:
        raise ValueError("action and gradient block shapes differ")
    action = action.detach().cpu().double().reshape(-1)
    gradient = gradient.detach().cpu().double().reshape(-1)
    if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("nonfinite action-space decomposition block")
    action_norm = _finite(action.norm(), "action norm")
    gradient_norm = _finite(gradient.norm(), "gradient norm")
    contribution = action * (-gradient)
    positive = _finite(contribution.clamp_min(0).sum(), "positive contribution")
    negative = _finite(contribution.clamp_max(0).sum(), "negative contribution")
    signed = _finite(contribution.sum(), "signed contribution")
    absolute = _finite(contribution.abs().sum(), "absolute contribution")
    if not math.isclose(signed, positive + negative, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("signed contribution does not equal positive plus negative")
    if not math.isclose(absolute, positive - negative, rel_tol=1e-10, abs_tol=1e-12):
        raise RuntimeError("absolute contribution does not equal positive minus negative")
    cosine = None
    if action_norm != 0.0 and gradient_norm != 0.0:
        cosine = max(-1.0, min(1.0, signed / (action_norm * gradient_norm)))
    cancellation = None if absolute == 0.0 else max(0.0, min(1.0, 1.0 - abs(signed) / absolute))
    return {
        "gradient_norm": gradient_norm,
        "action_norm": action_norm,
        "cosine_to_negative_gradient": cosine,
        "signed_dot": signed,
        "positive_contribution_sum": positive,
        "negative_contribution_sum": negative,
        "absolute_contribution_sum": absolute,
        "cancellation_ratio": cancellation,
    }


def _select_block(value, frame_indices, coordinate_indices):
    frames = torch.as_tensor(frame_indices, dtype=torch.long, device=value.device)
    coordinates = torch.as_tensor(coordinate_indices, dtype=torch.long, device=value.device)
    return value.index_select(0, frames).index_select(1, coordinates)


def combined_block_stats(actions, gradient, frame_indices, coordinate_indices):
    selected_gradient = _select_block(gradient, frame_indices, coordinate_indices)
    individual = {
        name: contribution_stats(
            _select_block(actions[name], frame_indices, coordinate_indices), selected_gradient
        )
        for name in ACTION_NAMES
    }
    result = {"gradient_norm": individual["base"]["gradient_norm"]}
    for name in ACTION_NAMES:
        stats = individual[name]
        result.update(
            {
                f"{name}_action_norm": stats["action_norm"],
                f"{name}_cosine": stats["cosine_to_negative_gradient"],
                f"{name}_signed_dot": stats["signed_dot"],
                f"{name}_positive_contribution_sum": stats["positive_contribution_sum"],
                f"{name}_negative_contribution_sum": stats["negative_contribution_sum"],
                f"{name}_absolute_contribution_sum": stats["absolute_contribution_sum"],
                f"{name}_cancellation_ratio": stats["cancellation_ratio"],
            }
        )
    return result


def decompose_case(actions, gradient, binary_support, metadata):
    if any(value.shape != gradient.shape or value.shape[-1] != 75 for value in actions.values()):
        raise ValueError("BASE/ADAPTER/TOTAL actions and gradient must share [T,75]")
    if binary_support.shape != gradient.shape:
        raise ValueError("binary support does not match action geometry")
    active = torch.nonzero(binary_support.bool().any(dim=-1), as_tuple=False).flatten().tolist()
    partitions = temporal_partition(active)
    inactive = ~binary_support.bool()
    outside_max = _finite(gradient[inactive].abs().max(), "gradient outside support") if bool(inactive.any()) else 0.0
    if outside_max != 0.0:
        raise RuntimeError("authoritative temporal gradient escaped decoder support")
    all_frames = list(range(gradient.shape[0]))
    all_coordinates = list(range(75))
    whole = combined_block_stats(actions, gradient, all_frames, all_coordinates)
    active_whole = combined_block_stats(actions, gradient, active, all_coordinates)
    anatomy = {
        name: combined_block_stats(actions, gradient, active, ANATOMY_COORDINATES[name])
        for name in ANATOMY_NAMES
    }
    temporal = {
        name: combined_block_stats(actions, gradient, partitions[name], all_coordinates)
        for name in TIME_NAMES
    }
    anatomy_time = {
        f"{anatomy_name}/{time_name}": combined_block_stats(
            actions,
            gradient,
            partitions[time_name],
            ANATOMY_COORDINATES[anatomy_name],
        )
        for anatomy_name in ANATOMY_NAMES
        for time_name in TIME_NAMES
    }
    for action in ACTION_NAMES:
        block_sum = sum(row[f"{action}_signed_dot"] for row in anatomy_time.values())
        if not math.isclose(
            block_sum, whole[f"{action}_signed_dot"], rel_tol=1e-8, abs_tol=1e-10
        ):
            raise RuntimeError("anatomy-time signed contributions do not reconstruct whole action")
    return {
        **metadata,
        "active_frame_indices": active,
        "first_active_frame": active[0],
        "last_active_frame": active[-1],
        "active_frame_count": len(active),
        "temporal_frame_indices": partitions,
        "gradient_outside_binary_support_max": outside_max,
        "whole": whole,
        "active_whole": active_whole,
        "anatomy": anatomy,
        "temporal": temporal,
        "anatomy_time": anatomy_time,
    }


def _median(values):
    finite = [float(value) for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def summarize_blocks(blocks):
    if not blocks:
        raise ValueError("cannot summarize empty decomposition blocks")
    result = {"cases": len(blocks), "actions": {}}
    result["median_gradient_norm"] = _median(row["gradient_norm"] for row in blocks)
    for action in ACTION_NAMES:
        cosines = [row[f"{action}_cosine"] for row in blocks]
        signed = [row[f"{action}_signed_dot"] for row in blocks]
        result["actions"][action] = {
            "cases": len(blocks),
            "defined_cosine_cases": sum(value is not None for value in cosines),
            "median_cosine": _median(cosines),
            "median_signed_dot": _median(signed),
            "median_cancellation_ratio": _median(
                row[f"{action}_cancellation_ratio"] for row in blocks
            ),
            "median_absolute_contribution_sum": _median(
                row[f"{action}_absolute_contribution_sum"] for row in blocks
            ),
            "positive_signed_dot_cases": sum(value > 0.0 for value in signed),
            "negative_signed_dot_cases": sum(value < 0.0 for value in signed),
            "zero_or_undefined_cases": sum(
                value == 0.0 or cosine is None for value, cosine in zip(signed, cosines)
            ),
        }
    return result


def build_summaries(rows):
    if len(rows) != FINAL_CASES:
        raise ValueError("decomposition summary requires all 64 cases")
    scopes = {"overall": rows}
    for role in rcsp.ROLE_MAPPING:
        scopes[f"role:{role}"] = [row for row in rows if row["role"] == role]
    for width in (10, 28):
        scopes[f"width:{width}"] = [row for row in rows if row["width"] == width]
    for split, role in rcsp.FINAL_BLOCK_ORDER:
        for width in (10, 28):
            name = f"group:{split}/{role}/{width}"
            scopes[name] = [
                row
                for row in rows
                if row["split"] == split and row["role"] == role and row["width"] == width
            ]
    expected = {"overall": 64, "role:single_recording": 32, "role:cross_event": 32}
    if any(len(scopes[name]) != count for name, count in expected.items()) or any(
        len(values) != 8 for name, values in scopes.items() if name.startswith("group:")
    ):
        raise RuntimeError("fixed-final decomposition scope counts are invalid")
    summary = {
        name: {"cases": len(values), "whole": summarize_blocks([row["whole"] for row in values])}
        for name, values in scopes.items()
    }
    single = scopes["role:single_recording"]
    single_anatomy = {
        name: summarize_blocks([row["anatomy"][name] for row in single]) for name in ANATOMY_NAMES
    }
    single_temporal = {
        name: summarize_blocks([row["temporal"][name] for row in single]) for name in TIME_NAMES
    }
    single_anatomy_time = {
        name: summarize_blocks([row["anatomy_time"][name] for row in single])
        for name in (f"{anatomy}/{time}" for anatomy in ANATOMY_NAMES for time in TIME_NAMES)
    }
    return scopes, summary, single_anatomy, single_temporal, single_anatomy_time


def _sign(value):
    if value is None or value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def _comparison_row(left_rows, right_rows, block, left_label, right_label):
    left = [row["anatomy_time"][block] for row in left_rows]
    right = [row["anatomy_time"][block] for row in right_rows]
    if not left or not right:
        raise ValueError("direction comparison requires two nonempty unpaired groups")
    left_summary, right_summary = summarize_blocks(left), summarize_blocks(right)

    def compare_action(action):
        left_action = left_summary["actions"][action]
        right_action = right_summary["actions"][action]
        left_sign = _sign(left_action["median_signed_dot"])
        right_sign = _sign(right_action["median_signed_dot"])
        left_consistent = max(
            left_action["positive_signed_dot_cases"],
            left_action["negative_signed_dot_cases"],
        ) > len(left) / 2
        right_consistent = max(
            right_action["positive_signed_dot_cases"],
            right_action["negative_signed_dot_cases"],
        ) > len(right) / 2
        left_cosine = left_action["median_cosine"]
        right_cosine = right_action["median_cosine"]
        return {
            "left_median_cosine": left_cosine,
            "right_median_cosine": right_cosine,
            "median_cosine_difference_right_minus_left": (
                None
                if left_cosine is None or right_cosine is None
                else right_cosine - left_cosine
            ),
            "left_median_signed_dot": left_action["median_signed_dot"],
            "right_median_signed_dot": right_action["median_signed_dot"],
            "left_negative_cases": left_action["negative_signed_dot_cases"],
            "right_negative_cases": right_action["negative_signed_dot_cases"],
            "left_majority_sign_consistent": left_consistent,
            "right_majority_sign_consistent": right_consistent,
            "stable_sign_shift": (
                left_sign != 0
                and right_sign != 0
                and left_sign != right_sign
                and left_consistent
                and right_consistent
            ),
        }

    actions = {action: compare_action(action) for action in ("adapter", "total")}
    adapter = actions["adapter"]
    return {
        "block": block,
        "left": left_label,
        "right": right_label,
        "unpaired_group_comparison": True,
        "fake_case_pairing_performed": False,
        "left_cases": len(left),
        "right_cases": len(right),
        "actions": actions,
        "left_median_cosine": adapter["left_median_cosine"],
        "right_median_cosine": adapter["right_median_cosine"],
        "median_cosine_difference_right_minus_left": adapter[
            "median_cosine_difference_right_minus_left"
        ],
        "left_median_signed_dot": adapter["left_median_signed_dot"],
        "right_median_signed_dot": adapter["right_median_signed_dot"],
        "left_negative_cases": adapter["left_negative_cases"],
        "right_negative_cases": adapter["right_negative_cases"],
        "left_majority_sign_consistent": adapter["left_majority_sign_consistent"],
        "right_majority_sign_consistent": adapter["right_majority_sign_consistent"],
        "adapter_stable_sign_shift": actions["adapter"]["stable_sign_shift"],
        "total_stable_sign_shift": actions["total"]["stable_sign_shift"],
        "stable_sign_shift": any(row["stable_sign_shift"] for row in actions.values()),
    }


def source_conditioned_comparison(scopes):
    result = {}
    for width in (10, 28):
        left = scopes[f"group:seen/single_recording/{width}"]
        right = scopes[f"group:new_position/single_recording/{width}"]
        result[str(width)] = {
            block: _comparison_row(left, right, block, "seen", "new_position")
            for block in (f"{a}/{t}" for a in ANATOMY_NAMES for t in TIME_NAMES)
        }
    return {
        "comparison": result,
        "case_pairing_performed": False,
        "parameter_attribution_used_for_selection": False,
    }


def width_conditioned_comparison(scopes):
    result = {}
    for split in ("seen", "new_position"):
        left = scopes[f"group:{split}/single_recording/10"]
        right = scopes[f"group:{split}/single_recording/28"]
        result[split] = {
            block: _comparison_row(left, right, block, "width10", "width28")
            for block in (f"{a}/{t}" for a in ANATOMY_NAMES for t in TIME_NAMES)
        }
    return {
        "comparison": result,
        "direction_description_only": True,
        "normalization_or_threshold_investigated": False,
        "case_pairing_performed": False,
    }


def _dominant_block(summary, action="adapter", *, negative=False):
    candidates = []
    for name, row in summary.items():
        values = row["actions"][action]
        if negative:
            candidates.append(
                (
                    values["negative_signed_dot_cases"],
                    -(values["median_signed_dot"] or 0.0),
                    name,
                )
            )
        else:
            candidates.append(
                (values["median_absolute_contribution_sum"] or 0.0, name)
            )
    chosen = max(candidates) if candidates else None
    if chosen is None or chosen[0] == 0:
        return None
    return chosen[-1]


def parameter_to_action_bridge(parameter_report, scopes):
    parameter_rows = {
        row["key"]: row
        for row in parameter_report["parameter_gradient_rows"]
        if row["role"] == "single_recording"
    }
    expected = [
        f"{source}/single_recording/{width}"
        for source in parameter_audit.SOURCE_ORDER
        for width in (10, 28)
    ]
    if set(parameter_rows) != set(expected):
        raise ValueError("parameter attribution does not contain six fixed single conditions")
    rows = []
    for key in expected:
        parameter = parameter_rows[key]
        source, _role, width_text = key.split("/")
        row = {
            "condition": key,
            "parameter_gradient_norm": parameter["parameter_gradient_norm"],
            "learned_displacement_vs_negative_gradient_cosine": parameter[
                "learned_displacement_vs_negative_gradient_cosine"
            ],
            "parameter_evidence_only": source == "train_transaction_0",
        }
        if source != "train_transaction_0":
            group_rows = scopes[f"group:{source}/single_recording/{width_text}"]
            whole = summarize_blocks([item["whole"] for item in group_rows])
            blocks = {
                name: summarize_blocks([item["anatomy_time"][name] for item in group_rows])
                for name in (f"{a}/{t}" for a in ANATOMY_NAMES for t in TIME_NAMES)
            }
            row.update(
                {
                    "whole_adapter_cosine": whole["actions"]["adapter"]["median_cosine"],
                    "whole_total_cosine": whole["actions"]["total"]["median_cosine"],
                    "dominant_negative_anatomy_time_block": _dominant_block(
                        blocks, negative=True
                    ),
                }
            )
        rows.append(row)
    return {
        "rows": rows,
        "parameter_report_used_for_provenance_and_side_by_side_context_only": True,
        "train_action_decomposition_fabricated": False,
        "paired_correlation_computed": False,
        "case_selection_performed": False,
    }


def scientific_answers(
    summary,
    single_anatomy,
    single_temporal,
    single_anatomy_time,
    source_comparison,
    width_comparison,
    new_single_28,
    direction_parity_verified,
    parameter_report=None,
):
    whole_weak_by_action = {}
    for action in ("adapter", "total"):
        single_whole = summary["role:single_recording"]["whole"]["actions"][action]
        cross_whole = summary["role:cross_event"]["whole"]["actions"][action]
        whole_weak_by_action[action] = (
            single_whole["median_cosine"] is not None
            and cross_whole["median_cosine"] is not None
            and abs(single_whole["median_cosine"]) < abs(cross_whole["median_cosine"])
        )
    whole_weak = all(whole_weak_by_action.values())

    def mixed_sign_case(row, field, action):
        values = [block[f"{action}_signed_dot"] for block in row[field].values()]
        return any(value > 0.0 for value in values) and any(value < 0.0 for value in values)

    single_rows = [row for row in new_single_28["all_single_rows"]]
    anatomy_mixed_by_action = {
        action: sum(mixed_sign_case(row, "anatomy", action) for row in single_rows)
        for action in ("adapter", "total")
    }
    temporal_mixed_by_action = {
        action: sum(mixed_sign_case(row, "temporal", action) for row in single_rows)
        for action in ("adapter", "total")
    }
    anatomy_stable_signs_by_action = {
        action: {
            name: (
                1
                if row["actions"][action]["positive_signed_dot_cases"] > row["cases"] / 2
                else -1
                if row["actions"][action]["negative_signed_dot_cases"] > row["cases"] / 2
                else 0
            )
            for name, row in single_anatomy.items()
        }
        for action in ("adapter", "total")
    }
    temporal_stable_signs_by_action = {
        action: {
            name: (
                1
                if row["actions"][action]["positive_signed_dot_cases"] > row["cases"] / 2
                else -1
                if row["actions"][action]["negative_signed_dot_cases"] > row["cases"] / 2
                else 0
            )
            for name, row in single_temporal.items()
        }
        for action in ("adapter", "total")
    }
    anatomical_cancellation_by_action = {
        action: count > len(single_rows) / 2
        and 1 in anatomy_stable_signs_by_action[action].values()
        and -1 in anatomy_stable_signs_by_action[action].values()
        for action, count in anatomy_mixed_by_action.items()
    }
    temporal_cancellation_by_action = {
        action: count > len(single_rows) / 2
        and 1 in temporal_stable_signs_by_action[action].values()
        and -1 in temporal_stable_signs_by_action[action].values()
        for action, count in temporal_mixed_by_action.items()
    }
    anatomical_cancellation = any(anatomical_cancellation_by_action.values())
    temporal_cancellation = any(temporal_cancellation_by_action.values())
    localized_by_action = {
        action: {
            name: row["actions"][action]["negative_signed_dot_cases"] > row["cases"] / 2
            for name, row in single_anatomy_time.items()
        }
        for action in ("adapter", "total")
    }
    source_shifts = [
        row
        for groups in source_comparison["comparison"].values()
        for row in groups.values()
        if row["stable_sign_shift"]
    ]
    width_shifts = [
        row
        for groups in width_comparison["comparison"].values()
        for row in groups.values()
        if row["stable_sign_shift"]
    ]
    new_28_blocks = {
        name: summarize_blocks([row["anatomy_time"][name] for row in new_single_28["rows"]])
        for name in (f"{a}/{t}" for a in ANATOMY_NAMES for t in TIME_NAMES)
    }
    new_28_case_count = len(new_single_28["rows"])
    if new_28_case_count != 8 or new_single_28.get("cases", 8) != new_28_case_count:
        raise ValueError("new_position/single_recording/28 must retain exactly 8 cases")
    new_28_negative_by_action = {
        action: {
            name: row["actions"][action]["negative_signed_dot_cases"]
            > new_28_case_count / 2
            for name, row in new_28_blocks.items()
        }
        for action in ("adapter", "total")
    }
    localized = localized_by_action["adapter"]
    new_28_negative = new_28_negative_by_action["adapter"]
    any_localized = any(any(values.values()) for values in localized_by_action.values())
    any_new_28_negative = any(
        any(values.values()) for values in new_28_negative_by_action.values()
    )
    all_cross_rows = new_single_28.get("all_cross_rows", ())
    if all_cross_rows and len(all_cross_rows) != 32:
        raise ValueError("cross control must contain exactly 32 cases")
    cross_anatomy_time = (
        {
            name: summarize_blocks([row["anatomy_time"][name] for row in all_cross_rows])
            for name in (f"{a}/{t}" for a in ANATOMY_NAMES for t in TIME_NAMES)
        }
        if all_cross_rows
        else {}
    )
    all_local_blocks_relatively_weaker_by_action = {
        action: bool(cross_anatomy_time)
        and all(
            single_anatomy_time[name]["actions"][action]["median_cosine"] is not None
            and cross_anatomy_time[name]["actions"][action]["median_cosine"] is not None
            and abs(single_anatomy_time[name]["actions"][action]["median_cosine"])
            < abs(cross_anatomy_time[name]["actions"][action]["median_cosine"])
            for name in single_anatomy_time
        )
        for action in ("adapter", "total")
    }
    global_weakness = whole_weak and not any(
        (
            anatomical_cancellation,
            temporal_cancellation,
            any_localized,
            bool(source_shifts),
            bool(width_shifts),
        )
    ) and all(all_local_blocks_relatively_weaker_by_action.values())
    mechanisms = []
    if global_weakness:
        mechanisms.append("SINGLE_GLOBAL_DIRECTION_WEAKNESS")
    if anatomical_cancellation:
        mechanisms.append("SINGLE_ANATOMICAL_CANCELLATION")
    if temporal_cancellation:
        mechanisms.append("SINGLE_TEMPORAL_CANCELLATION")
    if any_localized:
        mechanisms.append("SINGLE_LOCALIZED_DIRECTION_MISMATCH")
    if source_shifts:
        mechanisms.append("SOURCE_CONDITIONED_SINGLE_DIRECTION_SHIFT")
    if width_shifts:
        mechanisms.append("WIDTH_CONDITIONED_SINGLE_DIRECTION_SHIFT")
    if any_new_28_negative:
        mechanisms.append("NEW_POSITION_SINGLE_28_LOCALIZED_ASCENT")
    classification = (
        "UNRESOLVED_BY_DECOMPOSITION"
        if not mechanisms
        else mechanisms[0]
        if len(mechanisms) == 1
        else "MULTIPLE_MECHANISMS_SUPPORTED"
    )
    parameter_answers = (parameter_report or {}).get("scientific_answers", {})
    parameter_action_ratio = parameter_answers.get(
        "single_to_cross_action_direction_cosine_ratio"
    )
    parameter_consistency_evidence = {
        "all_single_parameter_gradients_nonzero": parameter_answers.get(
            "all_single_parameter_gradients_nonzero"
        ),
        "single_to_cross_action_direction_cosine_ratio": parameter_action_ratio,
        "single_parameter_attribution_classification": parameter_answers.get(
            "single_direction_attribution"
        ),
        "used_for_case_or_block_selection": False,
    }
    parameter_consistent = (
        direction_parity_verified
        and whole_weak
        and parameter_answers.get("all_single_parameter_gradients_nonzero") is True
        and parameter_action_ratio is not None
        and abs(float(parameter_action_ratio)) < 1.0
    )
    return {
        "single_direction_decomposition": classification,
        "supported_descriptive_mechanisms": mechanisms,
        "whole_single_direction_weak": whole_weak,
        "whole_single_direction_weak_by_action": whole_weak_by_action,
        "all_local_blocks_relatively_weaker_by_action": (
            all_local_blocks_relatively_weaker_by_action
        ),
        "anatomical_cancellation_observed": anatomical_cancellation,
        "anatomical_cancellation_by_action": anatomical_cancellation_by_action,
        "anatomical_stable_signs_by_action": anatomy_stable_signs_by_action,
        "anatomical_mixed_sign_single_cases": anatomy_mixed_by_action["adapter"],
        "anatomical_mixed_sign_single_cases_by_action": anatomy_mixed_by_action,
        "temporal_cancellation_observed": temporal_cancellation,
        "temporal_cancellation_by_action": temporal_cancellation_by_action,
        "temporal_stable_signs_by_action": temporal_stable_signs_by_action,
        "temporal_mixed_sign_single_cases": temporal_mixed_by_action["adapter"],
        "temporal_mixed_sign_single_cases_by_action": temporal_mixed_by_action,
        "localized_negative_block_observed": any_localized,
        "localized_negative_blocks": localized,
        "localized_negative_blocks_by_action": localized_by_action,
        "source_conditioned_shift_observed": bool(source_shifts),
        "source_conditioned_shift_blocks": [row["block"] for row in source_shifts],
        "width_conditioned_shift_observed": bool(width_shifts),
        "width_conditioned_shift_blocks": [row["block"] for row in width_shifts],
        "new_position_single_28_localized_ascent": any_new_28_negative,
        "new_position_single_28_negative_blocks": new_28_negative,
        "new_position_single_28_negative_blocks_by_action": new_28_negative_by_action,
        "dominant_single_anatomy": _dominant_block(single_anatomy),
        "dominant_single_temporal_region": _dominant_block(single_temporal),
        "dominant_single_anatomy_time_block": _dominant_block(single_anatomy_time),
        "dominant_single_anatomy_by_action": {
            action: _dominant_block(single_anatomy, action)
            for action in ("adapter", "total")
        },
        "dominant_single_temporal_region_by_action": {
            action: _dominant_block(single_temporal, action)
            for action in ("adapter", "total")
        },
        "dominant_single_anatomy_time_block_by_action": {
            action: _dominant_block(single_anatomy_time, action)
            for action in ("adapter", "total")
        },
        "cross_control_preserved": direction_parity_verified,
        "parameter_attribution_consistent_with_action_decomposition": parameter_consistent,
        "parameter_attribution_consistency_evidence": parameter_consistency_evidence,
        "remaining_uncertainty": (
            "Block localization is descriptive at one frozen state. It does not distinguish "
            "representation, optimization history, decoder nonlinearity, or causal architecture."
        ),
        "claim_boundary": (
            "This is a read-only fixed-state action-space decomposition. It can localize where "
            "direction mismatch appears, but cannot prove a causal architectural root cause, "
            "select a new model, or authorize Pilot."
        ),
    }


def _validate_parameter_attribution(path, rcsp_report_sha256):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"parameter attribution report does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != PARAMETER_ATTRIBUTION_SCHEMA
        or report.get("completed") is not True
        or report.get("provenance", {}).get("runtime_commit") != REVIEWED_MAIN_BASELINE
        or report.get("provenance", {}).get("rcsp_commit") != RCSP_SOURCE_COMMIT
        or report.get("provenance", {}).get("rcsp_sha256", {}).get("report.json")
        != rcsp_report_sha256
        or report.get("optimizer_steps") != 0
        or report.get("gradient_protocol", {}).get("parameter_update_performed") is not False
        or report.get("production_model_modified") is not False
        or report.get("pilot_allowed") is not False
        or len(report.get("parameter_gradient_rows", ())) != 12
    ):
        raise ValueError("parameter attribution lineage or read-only contract mismatch")
    return path, _file_sha256(path), report


def _adapter_path_from_report(rcsp_dir, report):
    descriptor = report.get("parameter_update_scope", {}).get("adapter_checkpoint", {})
    value = descriptor.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError("RCSP report lacks adapter checkpoint path")
    path = Path(value).resolve()
    if path.parent != Path(rcsp_dir).resolve() or not path.is_file():
        raise ValueError("RCSP adapter path is missing or outside the reported result directory")
    if descriptor.get("sha256") != _file_sha256(path):
        raise ValueError("RCSP adapter checkpoint hash mismatch")
    return path


def _validate_adapter_checkpoint(checkpoint):
    if (
        checkpoint.get("schema") != parameter_audit.RCSP_SOURCE_SCHEMA
        or checkpoint.get("completed_steps") != rcsp.STEPS
        or checkpoint.get("formal_checkpoint") is not False
        or checkpoint.get("production_model_modified") is not False
        or checkpoint.get("checkpoint_selection_performed") is not False
        or checkpoint.get("publish_allowed") is not False
        or checkpoint.get("pilot_allowed") is not False
        or checkpoint.get("resume_allowed") is not False
    ):
        raise ValueError("invalid diagnostic-only RCSP adapter checkpoint")
    return checkpoint


def _validate_rcsp_report_and_review(rcsp_dir):
    rcsp_dir = Path(rcsp_dir).resolve()
    report_path = rcsp_dir / "report.json"
    review_path = rcsp_dir / "reporting_logic_review_v1.json"
    if not report_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("RCSP report/review artifact pair is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    report_hash = _file_sha256(report_path)
    report_false_fields = (
        "checkpoint_selection_performed",
        "scale_selection_performed",
        "production_model_modified",
        "production_inference_modified",
        "scientific_acceptance",
        "publish_allowed",
        "pilot_allowed",
    )
    if (
        report.get("schema") != parameter_audit.RCSP_SOURCE_SCHEMA
        or report.get("completed") is not True
        or report.get("base_model_frozen") is not True
        or report.get("adapter_only_training") is not True
        or report.get("optimizer_steps") != rcsp.STEPS
        or report.get("provenance", {}).get("runtime_commit") != RCSP_SOURCE_COMMIT
        or any(report.get(field) is not False for field in report_false_fields)
    ):
        raise ValueError("RCSP report lineage or frozen contract mismatch")
    if (
        review.get("schema") != parameter_audit.RCSP_REVIEW_SCHEMA
        or review.get("completed") is not True
        or review.get("source_report", {}).get("sha256") != report_hash
        or review.get("measurement_recomputation_verified") is not True
        or review.get("formal_conclusion", {}).get("classification")
        != "ROLE_CONDITIONING_USEFUL_BUT_WIDTH_DEPENDENT_MECHANISM_REMAINS"
        or review.get("formal_conclusion", {}).get("role_conditioning_alone_sufficient")
        is not False
        or review.get("production_model_modified") is not False
        or review.get("scientific_acceptance") is not False
        or review.get("pilot_allowed") is not False
    ):
        raise ValueError("RCSP review lineage or classification mismatch")
    adapter_path = _adapter_path_from_report(rcsp_dir, report)
    adapter_checkpoint = _validate_adapter_checkpoint(
        m._trusted_torch_load(adapter_path, map_location="cpu")
    )
    return {
        "directory": rcsp_dir,
        "report_path": report_path,
        "review_path": review_path,
        "adapter_path": adapter_path,
        "hashes": {
            "report.json": report_hash,
            "reporting_logic_review_v1.json": _file_sha256(review_path),
            "adapter_checkpoint": _file_sha256(adapter_path),
        },
        "report": report,
        "review": review,
        "adapter_checkpoint": adapter_checkpoint,
    }


def evaluate_decomposition(model, batch, metadata, cfg, rcsp_report):
    if len(metadata) != FINAL_CASES or int(batch["clean"].shape[0]) != FINAL_CASES:
        raise ValueError("single direction decomposition requires fixed final 64")
    source_direction = rcsp_report["direction_alignment"]["case_level"]
    if len(source_direction) != FINAL_CASES:
        raise ValueError("RCSP source direction rows are incomplete")
    rows = []
    maximum_parity_error = 0.0
    for start in range(0, FINAL_CASES, FINAL_CHUNK_SIZE):
        stop = start + FINAL_CHUNK_SIZE
        part = {key: value[start:stop] for key, value in batch.items()}
        part_meta = metadata[start:stop]
        masks = m._refiner_decode_masks(
            part["joint"], part["root"], part["contact"], part["seam"], cfg
        )
        role_id = rcsp.role_ids_from_metadata(part_meta, part["bad"].device)
        raw = model.forward_explicit(
            part["bad"],
            part["cond"],
            part["seam"],
            part["joint"],
            role_id,
            masks[0],
            masks[1],
            capture_details=True,
        )
        details = model.last_details
        if not torch.equal(raw[..., :4].detach(), details["raw_base"][..., :4]):
            raise RuntimeError("RCSP decomposition changed contact channels")
        action_variable = raw[..., 4:].detach().clone().requires_grad_(True)
        raw_for_decode = torch.cat((raw[..., :4].detach(), action_variable), dim=-1)
        prediction = m._decode_product_refiner_output(
            part["bad"], raw_for_decode, *masks, cfg
        )
        terms = alignment._scientific_terms(prediction, part["bad"], part["seam"], cfg)
        gradient = torch.autograd.grad(terms["temporal"].sum(), action_variable)[0]
        support = rcsp.binary_geometry_support(masks[0], masks[1])
        for local, meta in enumerate(part_meta):
            actions = {
                "base": details["raw_base"][local, ..., 4:],
                "adapter": details["adapter_projected"][local],
                "total": raw[local, ..., 4:].detach(),
            }
            row = decompose_case(
                actions,
                gradient[local],
                support[local],
                {
                    **meta,
                    "temporal_scientific_deficit": _finite(
                        terms["temporal"][local], "temporal deficit"
                    ),
                },
            )
            expected = source_direction[start + local]
            for identity_field in (
                "split",
                "role",
                "width",
                "case_index",
                "bank_case_index",
            ):
                if row.get(identity_field) != expected.get(identity_field):
                    raise RuntimeError(
                        f"RCSP direction row identity mismatch: {identity_field}"
                    )
            for field, action in (
                (
                    "projected_adapter_delta_vs_negative_temporal_gradient_cosine",
                    "adapter",
                ),
                ("adapted_total_action_vs_negative_temporal_gradient_cosine", "total"),
            ):
                actual, recorded = row["whole"][f"{action}_cosine"], expected[field]
                if actual is None or recorded is None:
                    if actual is not None or recorded is not None:
                        raise RuntimeError("RCSP direction defined/null parity mismatch")
                    error = 0.0
                else:
                    error = abs(actual - float(recorded))
                    if error > PARITY_ATOL:
                        raise RuntimeError("RCSP whole-action direction parity failed")
                maximum_parity_error = max(maximum_parity_error, error)
            rows.append(row)
        model.clear_last_details()
    return {
        "case_level": rows,
        "direction_parity": {
            "verified": True,
            "source": "completed RCSP direction_alignment.case_level",
            "max_abs_cosine_error": maximum_parity_error,
            "atol": PARITY_ATOL,
        },
    }


def run(args):
    source = Path(args.state_dir).resolve()
    trajectory, traj_paths, traj_hashes, traj_report, experiment, base_checkpoint = (
        failure._load_trajectory(args.trajectory_dir, args.expected_trajectory_commit)
    )
    rcsp_artifacts = _validate_rcsp_report_and_review(args.rcsp_dir)
    parameter_path, parameter_hash, parameter_report = _validate_parameter_attribution(
        args.parameter_attribution_report,
        rcsp_artifacts["hashes"]["report.json"],
    )
    output = Path(args.output_dir).resolve()
    immutable_roots = (source, trajectory, rcsp_artifacts["directory"], parameter_path.parent)
    if output.exists() or any(output.is_relative_to(path) for path in immutable_roots):
        raise FileExistsError("decomposition output must be fresh and outside immutable inputs")
    state, bank, cfg, source_metadata = group_audit.load_frozen_source(
        source,
        group_audit.LEGACY_COMMIT,
        legacy_core_strength=args.legacy_core_strength,
        legacy_transition_strength=args.legacy_transition_strength,
    )
    if experiment.get("source", {}).get("source_sha256") != source_metadata["source_sha256"]:
        raise ValueError("trajectory does not reference supplied frozen source")
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
                raise RuntimeError("loaded base differs from immutable trajectory final state")
            adapter_checkpoint = rcsp_artifacts["adapter_checkpoint"]
            if adapter_checkpoint.get("base_state_sha256") != base_hash:
                raise ValueError("RCSP adapter does not reference loaded base state")
            model = rcsp.FrozenBaseRCSPModel(base)
            model.adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            model.eval()
            adapter_hash = safe.state_hash(model.adapter.state_dict())
            if adapter_hash != rcsp_artifacts["report"]["parameter_update_scope"][
                "adapter_state_sha256"
            ]:
                raise RuntimeError("loaded RCSP adapter state hash mismatch")
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("base and adapter must be completely frozen")

            probe, probe_hash = safe.load_probe(source, state, bank, cfg)
            final_batch, final_metadata = alignment.combine_final_banks(
                failure.final_banks(bank, probe, cfg)
            )
            final_batch = rcsp._move_batch(final_batch, device)
            evaluated = evaluate_decomposition(
                model, final_batch, final_metadata, cfg, rcsp_artifacts["report"]
            )
            rows = evaluated["case_level"]
            scopes, summary, single_anatomy, single_temporal, single_anatomy_time = (
                build_summaries(rows)
            )
            source_comparison = source_conditioned_comparison(scopes)
            width_comparison = width_conditioned_comparison(scopes)
            all_single = scopes["role:single_recording"]
            new_single_28_rows = scopes["group:new_position/single_recording/28"]
            new_single_28 = {
                "cases": len(new_single_28_rows),
                "rows": new_single_28_rows,
                "all_single_rows": all_single,
                "all_cross_rows": scopes["role:cross_event"],
            }
            bridge = parameter_to_action_bridge(parameter_report, scopes)
            answers = scientific_answers(
                summary,
                single_anatomy,
                single_temporal,
                single_anatomy_time,
                source_comparison,
                width_comparison,
                new_single_28,
                evaluated["direction_parity"]["verified"],
                parameter_report,
            )
            if safe.state_hash(model.base.state_dict()) != base_hash:
                raise RuntimeError("decomposition changed frozen base state")
            if safe.state_hash(model.adapter.state_dict()) != adapter_hash:
                raise RuntimeError("decomposition changed frozen adapter state")
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("action-space autograd populated parameter .grad")
            if _file_sha256(source_paths["probe_bank.pt"]) != probe_hash:
                raise RuntimeError("probe changed during decomposition")

        for name, digest in source_hashes.items():
            if _file_sha256(source_paths[name]) != digest:
                raise RuntimeError("frozen source changed during decomposition")
        for name, digest in traj_hashes.items():
            if _file_sha256(traj_paths[name]) != digest:
                raise RuntimeError("trajectory changed during decomposition")
        for name, digest in rcsp_artifacts["hashes"].items():
            path = {
                "report.json": rcsp_artifacts["report_path"],
                "reporting_logic_review_v1.json": rcsp_artifacts["review_path"],
                "adapter_checkpoint": rcsp_artifacts["adapter_path"],
            }[name]
            if _file_sha256(path) != digest:
                raise RuntimeError("RCSP artifact changed during decomposition")
        if _file_sha256(parameter_path) != parameter_hash:
            raise RuntimeError("parameter attribution report changed during decomposition")

        report = {
            "schema": SCHEMA,
            "completed": True,
            "provenance": {
                "runtime_commit": runtime_commit,
                "reviewed_main_baseline": REVIEWED_MAIN_BASELINE,
                "source": source_metadata,
                "source_sha256_including_probe": source_hashes,
                "trajectory": {
                    "directory": str(trajectory),
                    "commit": args.expected_trajectory_commit,
                    "sha256": traj_hashes,
                },
                "rcsp": {
                    "directory": str(rcsp_artifacts["directory"]),
                    "runtime_commit": RCSP_SOURCE_COMMIT,
                    "sha256": rcsp_artifacts["hashes"],
                },
                "parameter_attribution": {
                    "path": str(parameter_path),
                    "schema": parameter_report["schema"],
                    "runtime_commit": parameter_report["provenance"]["runtime_commit"],
                    "sha256": parameter_hash,
                    "used_for_case_or_block_selection": False,
                },
            },
            "source": str(source),
            "trajectory": str(trajectory),
            "rcsp": str(rcsp_artifacts["directory"]),
            "parameter_attribution": str(parameter_path),
            "base_checkpoint": str(traj_paths["diagnostic_latest.pt"]),
            "adapter_checkpoint": str(rcsp_artifacts["adapter_path"]),
            "base_model_frozen": True,
            "adapter_frozen": True,
            "optimizer_steps": 0,
            "parameter_update_performed": False,
            "gradient_protocol": {
                "objective": "alignment._scientific_terms temporal scientific deficit",
                "target": "detached raw_geometric_action_75d_requires_grad_copy",
                "parameter_gradient_used_as_action_gradient": False,
                "torch_autograd_grad_only": True,
                "parameter_grad_remained_none": True,
                "gradient_surgery_performed": False,
            },
            "action_definitions": {
                "base": "raw_base[...,4:]",
                "adapter": "last_details['adapter_projected']",
                "total": "raw_adapted[...,4:]",
                "contact_channels_excluded": [0, 1, 2, 3],
                "geometry_coordinates": 75,
            },
            "anatomy_partition": {
                **{key: value for key, value in ANATOMY_PARTITION.items() if key != "coordinates"},
                "coordinates": {
                    key: list(value) for key, value in ANATOMY_COORDINATES.items()
                },
            },
            "temporal_partition": {
                "active_frame_definition": (
                    "frame where any coordinate in binary production root/joint effective support > 0"
                ),
                "partition_algorithm": PARTITION_ALGORITHM,
                "width_used_to_choose_frames": False,
                "complete_and_disjoint_per_case": True,
            },
            "case_counts": {
                "overall": len(rows),
                "single_recording": len(scopes["role:single_recording"]),
                "cross_event": len(scopes["role:cross_event"]),
                "new_position_single_28": len(new_single_28_rows),
                "groups": {
                    name.removeprefix("group:"): len(values)
                    for name, values in scopes.items()
                    if name.startswith("group:")
                },
            },
            "case_level": rows,
            "summary": summary,
            "single_summary": summary["role:single_recording"],
            "single_anatomy": single_anatomy,
            "single_temporal": single_temporal,
            "single_anatomy_time": single_anatomy_time,
            "source_conditioned_comparison": source_comparison,
            "width_conditioned_comparison": width_comparison,
            "new_position_single_28": {
                "cases": len(new_single_28_rows),
                "rows": new_single_28_rows,
            },
            "parameter_to_action_bridge": bridge,
            "scientific_answers": answers,
            "state_integrity": {
                "base_state_sha256_before_after": base_hash,
                "adapter_state_sha256_before_after": adapter_hash,
                "base": {
                    "before": base_hash,
                    "after": base_hash,
                    "unchanged": True,
                },
                "adapter": {
                    "before": adapter_hash,
                    "after": adapter_hash,
                    "unchanged": True,
                },
                "direction_parity": evaluated["direction_parity"],
                "all_parameter_grad_none": True,
                "source_unchanged": True,
                "trajectory_unchanged": True,
                "rcsp_unchanged": True,
                "parameter_attribution_unchanged": True,
            },
            "checkpoint_selection_performed": False,
            "scale_selection_performed": False,
            "architecture_selection_performed": False,
            "width_conditioning_added": False,
            "production_model_modified": False,
            "production_inference_modified": False,
            "scientific_acceptance": False,
            "publish_allowed": False,
            "pilot_allowed": False,
            "next_action": "review_single_direction_decomposition_then_stop",
        }
        report_path = output / "report.json"
        _exclusive_json(report_path, report)
        print("SCIENTIFIC ANSWERS", flush=True)
        print(json.dumps(answers, ensure_ascii=False, allow_nan=False), flush=True)
        print("SINGLE WHOLE SUMMARY", flush=True)
        print(json.dumps(report["single_summary"], allow_nan=False), flush=True)
        print("SINGLE ANATOMY SUMMARY", flush=True)
        print(json.dumps(single_anatomy, allow_nan=False), flush=True)
        print("SINGLE TEMPORAL SUMMARY", flush=True)
        print(json.dumps(single_temporal, allow_nan=False), flush=True)
        print("SINGLE ANATOMY X TIME SUMMARY", flush=True)
        print(json.dumps(single_anatomy_time, allow_nan=False), flush=True)
        print("SOURCE-CONDITIONED COMPARISON", flush=True)
        print(json.dumps(source_comparison, allow_nan=False), flush=True)
        print("WIDTH-CONDITIONED DIRECTION COMPARISON", flush=True)
        print(json.dumps(width_comparison, allow_nan=False), flush=True)
        print("NEW_POSITION/SINGLE/28 8 CASES", flush=True)
        print(json.dumps(report["new_position_single_28"], allow_nan=False), flush=True)
        print("PARAMETER-TO-ACTION BRIDGE", flush=True)
        print(json.dumps(bridge, allow_nan=False), flush=True)
        print(
            json.dumps(
                {
                    "stage": "refiner_single_direction_decomposition_audit_complete",
                    "report": str(report_path),
                    "cases": len(rows),
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
    parser.add_argument("--parameter-attribution-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-main-commit", required=True)
    parser.add_argument(
        "--expected-trajectory-commit", default=failure.TRAJECTORY_COMMIT
    )
    parser.add_argument("--legacy-core-strength", type=float, required=True)
    parser.add_argument("--legacy-transition-strength", type=float, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
