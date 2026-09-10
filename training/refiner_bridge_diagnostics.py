"""Training-only full-bridge learnability gate; never a formal checkpoint.

Eight source-balanced TRAIN windows; full-cycle TRAIN fitting and held-out seam
positions; both single-recording occlusion and cross-event joins. Validation
motion is never loaded. The final fixed step, not the best probe result,
decides readiness.
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


SCHEMA = "refiner_observable_bridge_diagnostic_v15_12f"
FIT_PROTOCOL = "exact_guard_constrained_subgroup_mgda_transaction_v7"

CONTEXT_RESERVOIR_PROTOCOL = (
    "all_probe_safe_farthest_order_rotating_c5_v1"
)
PROBE_SCOPE = "unfitted_local_motion_context_within_train_windows"
# Number of context banks inside ONE optimizer transaction.
# V15.4 does not increase this value.
FIT_CONTEXT_COUNT = 5
PROBE_START_GUARD_FRAMES = 6

OUTPUT_WARMUP_STEPS = 50
OUTPUT_WARMUP_LR_MULTIPLIER = 10.0
MGDA_MAX_ITERATIONS = 512
MGDA_DUALITY_GAP_TOLERANCE = 1.0e-10
MGDA_COMMON_DESCENT_RMS_EPSILON = 1.0e-8
EXACT_GUARD_ACTIVE_MARGIN_FRACTION = 1.0
EXACT_GUARD_PROJECTION_MAX_PASSES = 64
EXACT_GUARD_DERIVATIVE_EPSILON = 1.0e-10
EXACT_GUARD_MIN_EFFECTIVE_SCALE = 1.0e-7


def fingerprint(args, cfg):
    from training.bridge_feasibility import DIRECT_OPTIMIZER_PROTOCOL
    value = common._fingerprint(args, cfg)
    value["implementation_sha256"].update({
        "bridge_diagnostic": common.file_sha256(__file__),
        "motion_models": common.file_sha256(Path(m.__file__)),
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
    value["confidence_precondition_protocol"] = (
        m.REFINER_CONFIDENCE_PRECONDITION_PROTOCOL
    )
    value["confidence_precondition_max"] = float(
        m.REFINER_CONFIDENCE_PRECONDITION_MAX
    )
    value["refiner_batch_aggregation_protocol"] = m.REFINER_BATCH_AGGREGATION_PROTOCOL
    value["direct_optimizer_protocol"] = DIRECT_OPTIMIZER_PROTOCOL
    value["refiner_input_protocol"] = m.REFINER_INPUT_PROTOCOL
    value["condition_path_protocol"] = m.REFINER_CONDITION_PATH_PROTOCOL
    value["temporal_scientific_weight"] = float(
        m.REFINER_TEMPORAL_SCIENTIFIC_WEIGHT
    )
    value["scientific_group_aggregation"] = (
        m.REFINER_SCIENTIFIC_GROUP_AGGREGATION
    )
    value["component_guard_deadband"] = float(
        m.REFINER_COMPONENT_GUARD_DEADBAND
    )
    value["feasibility_guard_deadband"] = float(
        m.REFINER_FEASIBILITY_GUARD_DEADBAND
    )
    value["refiner_tangent_gradient_protocol"] = m.REFINER_TANGENT_GRADIENT_PROTOCOL
    value["refiner_update_protocol"] = m.REFINER_UPDATE_PROTOCOL
    value["fit_protocol"] = FIT_PROTOCOL
    value["context_reservoir_protocol"] = CONTEXT_RESERVOIR_PROTOCOL
    value["probe_scope"] = PROBE_SCOPE
    value["pareto_gradient_protocol"] = (
        "deterministic_exact_guard_constrained_subgroup_mgda_v1"
    )
    value["guard_envelope_protocol"] = "immutable_fixed_metric_anchor_v1"
    value["output_warmup_steps"] = OUTPUT_WARMUP_STEPS
    value["output_warmup_lr_multiplier"] = OUTPUT_WARMUP_LR_MULTIPLIER
    value["training_guard_threshold_sources"] = {
        "clean_geometry_max":
            "checkpoint_validation_max_clean_identity_product_log_l1",
        "clean_contact_max":
            "checkpoint_validation_max_clean_identity_contact_l1",
        "physical_excess":
            "zero_excess_plus_configured_numerical_tolerance",
    }
    return value


def fixed_fit_bank(banks, split="seen"):
    """Use EVERY seen TRAIN case; held-out positions never enter an update.

    Checking only a randomly selected 8/32 cases allowed an accepted step to
    undo gains on the other fixed cases. That is a legitimate SGD behavior,
    but confounds a tiny fixed-bank learnability diagnostic. This diagnostic
    uses the complete bank for BOTH gradients and post-update line search.
    Formal random-window training is deliberately unchanged.
    """
    if split == "new_position":
        raise ValueError("held-out new_position probe cannot be used for fitting")
    roles = [banks[(split, role)] for role in ("single_recording", "cross_event")]
    count = len(roles[0]["clean"])
    if count < 2 or count % 2 or len(roles[1]["clean"]) != count:
        raise ValueError("fixed fit bank requires equally sized paired role/width cases")
    train = {key: m.torch.cat([role[key] for role in roles]) for key in roles[0]}
    train["group"] = m.torch.as_tensor(
        [i % 2 for i in range(count)] + [2 + i % 2 for i in range(count)],
        device=train["clean"].device)
    return train


def _concat_fit_batches(anchor, context):
    if set(anchor) != set(context):
        raise ValueError("anchor/context fit batch layouts do not match")
    return {key: m.torch.cat([anchor[key], context[key]]) for key in anchor}


def _fit_context_indices(banks):
    """Return the contiguous set of materialized reservoir-bank indices."""
    by_index = {}

    for key in banks:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
        ):
            continue

        split, role = key

        if not isinstance(split, str):
            continue

        prefix = "fit_context_"

        if not split.startswith(prefix):
            continue

        index = int(
            split[len(prefix):]
        )

        by_index.setdefault(
            index,
            set(),
        ).add(role)

    if not by_index:
        raise RuntimeError(
            "no fit-context reservoir banks were materialized"
        )

    indices = sorted(
        by_index
    )

    if indices != list(
        range(len(indices))
    ):
        raise RuntimeError(
            "fit-context reservoir indices must be contiguous from zero"
        )

    expected_roles = {
        "single_recording",
        "cross_event",
    }

    for index in indices:
        if by_index[index] != expected_roles:
            raise RuntimeError(
                f"reservoir bank {index} does not contain both roles"
            )

    if len(indices) < FIT_CONTEXT_COUNT:
        raise RuntimeError(
            "reservoir has fewer than five context banks"
        )

    return tuple(indices)


def _reservoir_transaction_schedule(banks):
    """Deterministic rotating-C5 schedule over the safe-start reservoir.

    If the reservoir contains exactly C5 banks, there is only ONE unique
    full-C5 transaction. Returning five cyclic permutations would repeat the
    exact same cases and would not increase local-context coverage.

    For a larger reservoir R > C5, transaction t receives:
        t, t+1, ..., t+4  (mod R)

    Thus every optimizer step still contains exactly five context banks, while
    every reservoir bank is visited in a deterministic exposure-balanced cycle.
    """
    indices = _fit_context_indices(
        banks
    )

    count = len(indices)

    if count < FIT_CONTEXT_COUNT:
        raise RuntimeError(
            "reservoir contains fewer than C5 context banks"
        )

    # Degenerate legacy-compatible case:
    # all available context banks already fit in one transaction.
    if count == FIT_CONTEXT_COUNT:
        return (
            tuple(indices),
        )

    schedule = []

    for offset in range(count):
        row = tuple(
            indices[
                (offset + delta) % count
            ]
            for delta in range(
                FIT_CONTEXT_COUNT
            )
        )

        if len(row) != FIT_CONTEXT_COUNT:
            raise RuntimeError(
                "reservoir transaction does not contain C5 contexts"
            )

        if len(set(row)) != FIT_CONTEXT_COUNT:
            raise RuntimeError(
                "reservoir transaction duplicated a context bank"
            )

        schedule.append(row)

    appearances = {
        index: 0
        for index in indices
    }

    for row in schedule:
        for index in row:
            appearances[index] += 1

    # For R>C5, every bank must occur exactly C5 times during one
    # complete rotating schedule.
    if set(appearances.values()) != {
        FIT_CONTEXT_COUNT
    }:
        raise RuntimeError(
            "reservoir schedule is not exposure-balanced"
        )

    return tuple(schedule)


def _reservoir_transaction_batch(
    banks,
    selected_context_indices,
):
    """Materialize exactly ONE C5 optimizer transaction.

    The returned 192-case batch is fixed for the entire optimizer
    transaction: gradient, Armijo closure and group guard all consume
    this same object. No other reservoir transaction is materialized.
    """
    selected = tuple(
        int(index)
        for index in selected_context_indices
    )

    if len(selected) != FIT_CONTEXT_COUNT:
        raise ValueError(
            "lazy reservoir transaction must contain exactly C5 contexts"
        )

    if len(set(selected)) != FIT_CONTEXT_COUNT:
        raise ValueError(
            "lazy reservoir transaction contains duplicate contexts"
        )

    available = set(
        _fit_context_indices(banks)
    )

    if not set(selected).issubset(available):
        raise ValueError(
            "lazy transaction requested an unavailable context bank"
        )

    batch = fixed_fit_bank(
        banks,
        "seen",
    )

    anchor_cases = len(
        batch["clean"]
    )

    for index in selected:
        context = fixed_fit_bank(
            banks,
            f"fit_context_{index}",
        )

        if len(context["clean"]) != anchor_cases:
            raise RuntimeError(
                "context bank size differs from seen anchor"
            )

        batch = _concat_fit_batches(
            batch,
            context,
        )

    expected_cases = (
        anchor_cases
        * (
            1
            + FIT_CONTEXT_COUNT
        )
    )

    if len(batch["clean"]) != expected_cases:
        raise RuntimeError(
            "lazy V15.4.1 transaction changed cases/update"
        )

    return batch


def anchored_context_replay_banks(banks):
    """Build fixed per-step C5 transactions from a larger safe reservoir.

    The returned list is the deterministic transaction cycle.  Each returned
    batch is immutable for the duration of one optimizer transaction: gradient,
    Armijo closure and subgroup guard all consume that SAME batch.
    """
    anchor = fixed_fit_bank(
        banks,
        "seen",
    )

    context_indices = _fit_context_indices(
        banks
    )

    context_banks = {
        index: fixed_fit_bank(
            banks,
            f"fit_context_{index}",
        )
        for index in context_indices
    }

    schedule = _reservoir_transaction_schedule(
        banks
    )

    replay = []

    anchor_cases = len(
        anchor["clean"]
    )

    for selected in schedule:
        batch = anchor

        for index in selected:
            batch = _concat_fit_batches(
                batch,
                context_banks[index],
            )

        expected_cases = (
            anchor_cases
            * (
                1
                + FIT_CONTEXT_COUNT
            )
        )

        if len(batch["clean"]) != expected_cases:
            raise RuntimeError(
                "V15.4 transaction changed the fixed C5 batch size"
            )

        replay.append(batch)

    expected_transaction_count = (
        1
        if len(context_indices) == FIT_CONTEXT_COUNT
        else len(context_indices)
    )

    if len(replay) != expected_transaction_count:
        raise RuntimeError(
            "reservoir replay transaction count mismatch"
        )

    if len(replay) != len(schedule):
        raise RuntimeError(
            "reservoir replay/schedule length mismatch"
        )

    return replay

def fit_bank_contract(
    windows,
    cfg=None,
):
    """Auditable V15.4 reservoir transaction contract."""
    if cfg is None:
        cfg = m.MotionGenerationConfig()

    reservoir_cycle_length = (
        _context_reservoir_cycle_length(
            cfg.window_len
        )
    )

    return {
        "protocol": FIT_PROTOCOL,

        # Keep V15.3.1 transaction size exactly unchanged.
        "cases_per_update":
            4
            * windows
            * (
                1
                + FIT_CONTEXT_COUNT
            ),

        "cases_per_role_width":
            windows
            * (
                1
                + FIT_CONTEXT_COUNT
            ),

        "cases_per_role_width_per_bank":
            windows,

        "gradient_scope":
            "complete_seen_plus_rotating_c5_safe_context_reservoir",

        "line_search_scope":
            "same_complete_seen_plus_rotating_c5_transaction",

        "group_guard_scope":
            "fixed_complete_seen_train_anchor",

        "group_guard_reference":
            "immutable_fixed_metric_anchor",

        "group_guard_rolling_tolerance_accumulation":
            False,

        "historical_component_minimum_intersection":
            False,

        "endpoint_temporal_gradient_protocol":
            "deterministic_exact_guard_constrained_subgroup_mgda",

        "guard_gradient_scope":
            "same_fixed_complete_seen_train_anchor",

        "output_warmup_steps": OUTPUT_WARMUP_STEPS,

        "output_warmup_lr_multiplier":
            OUTPUT_WARMUP_LR_MULTIPLIER,

        "seen_anchor_cases_per_update":
            4 * windows,

        "context_cases_per_update":
            4
            * windows
            * FIT_CONTEXT_COUNT,

        "context_banks_per_update":
            FIT_CONTEXT_COUNT,

        "context_reservoir_cycle_length":
            reservoir_cycle_length,

        "context_reservoir_protocol":
            CONTEXT_RESERVOIR_PROTOCOL,

        "reservoir_every_legal_start_seen_per_cycle":
            True,

        "transaction_batch_fixed_within_step":
            True,

        "probe_start_guard_frames":
            PROBE_START_GUARD_FRAMES,

        "probe_used_for_updates":
            False,
    }

def fixed_bank_stalled(update):
    """A retained V12 update already represents the complete context cycle."""
    return (
        not update["optimizer_update_accepted"]
        and update["reason"] in {
            "bounded_search_no_descent",
            "zero_gradient",
            "pareto_stationary_or_no_common_descent",
            "no_exact_guard_constrained_common_descent",
            "resolution_limited_under_exact_guard",
        }
    )

def _cpu_tree(value):
    if isinstance(value, m.torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:_cpu_tree(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def save_fit_bank(
    destination,
    report,
    cfg,
    *,
    banks=None,
    schedule=None,
):
    """Save exact V15.4 TRAIN reservoir and deterministic transaction schedule.

    The artifact stores:
      * the seen anchor once;
      * every unique materialized context bank once;
      * the exact rotating-C5 transaction schedule.

    It deliberately does NOT duplicate the anchor and five context banks inside
    every serialized transaction.
    """
    if banks is None or schedule is None:
        raise ValueError(
            "V15.4 fit artifact requires banks and reservoir schedule"
        )

    context_indices = _fit_context_indices(
        banks
    )

    contract = report.get(
        "fit_bank",
        {},
    )

    contract_cycle = int(
        contract.get(
            "context_reservoir_cycle_length",
            -1,
        )
    )

    if contract_cycle != len(context_indices):
        raise RuntimeError(
            "serialized context reservoir length does not match "
            "the diagnostic fit-bank contract"
        )

    if int(
        contract.get(
            "context_banks_per_update",
            -1,
        )
    ) != FIT_CONTEXT_COUNT:
        raise RuntimeError(
            "serialized fit-bank contract changed C5 contexts/update"
        )

    expected_schedule = (
        _reservoir_transaction_schedule(
            banks
        )
    )

    schedule = tuple(
        tuple(int(i) for i in row)
        for row in schedule
    )

    if schedule != expected_schedule:
        raise RuntimeError(
            "serialized reservoir schedule does not match deterministic contract"
        )

    expected_transaction_count = (
        1
        if len(context_indices) == FIT_CONTEXT_COUNT
        else len(context_indices)
    )

    if len(schedule) != expected_transaction_count:
        raise RuntimeError(
            "serialized transaction schedule length violates "
            "the deterministic reservoir contract"
        )

    anchor = fixed_fit_bank(
        banks,
        "seen",
    )

    reservoir = {
        str(index): _cpu_tree(
            fixed_fit_bank(
                banks,
                f"fit_context_{index}",
            )
        )
        for index in context_indices
    }

    expected_cases = (
        len(anchor["clean"])
        * (
            1
            + FIT_CONTEXT_COUNT
        )
    )

    path = destination / "fit_bank.pt"

    m._atomic_torch_save(
        {
            "schema":
                "refiner_train_safe_start_context_reservoir_v4",

            "train_only":
                True,

            "formal_checkpoint":
                False,

            "publish_allowed":
                False,

            "fingerprint":
                report["fingerprint"],

            "windows":
                report["windows"],

            "contract":
                report["fit_bank"],

            "config":
                dataclasses.asdict(cfg),

            "anchor":
                _cpu_tree(anchor),

            "context_reservoir":
                reservoir,

            "transaction_schedule":
                schedule,
        },
        path,
    )

    return {
        "file":
            path.name,

        "sha256":
            common.file_sha256(path),

        "cases_per_update":
            expected_cases,

        "contexts_per_update":
            FIT_CONTEXT_COUNT,

        "reservoir_banks":
            len(context_indices),

        "transactions_per_cycle":
            len(schedule),

        "train_only":
            True,
    }

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


def _seen_and_probe_starts(frames, width, recipe_id):
    seen = max(3, (frames - width) // 2 + (-8 if recipe_id else 8))
    probe = seen + (7 if recipe_id else -7)
    return seen, probe


def _all_probe_safe_context_starts(
    frames,
    width,
    recipe_id,
):
    """Return every legal TRAIN context in deterministic spread-first order.

    V15.3.1 permanently fitted only the first five farthest-point contexts.
    V15.4 keeps those same safety rules but extends the deterministic ordering
    until EVERY legal non-probe start has been included.

    The exact probe start and its +/-6-frame guard remain forbidden.
    """
    seen, probe = _seen_and_probe_starts(
        frames,
        width,
        recipe_id,
    )

    eligible = [
        start
        for start in range(
            3,
            frames - width - 1,
        )
        if (
            start != seen
            and abs(start - probe)
            > PROBE_START_GUARD_FRAMES
        )
    ]

    if len(eligible) < FIT_CONTEXT_COUNT:
        raise ValueError(
            "motion window cannot support the required "
            "probe-safe C5 context transaction"
        )

    remaining = list(eligible)
    selected = []

    # Preserve the established V12-V15.3 farthest-point criterion.
    # The only V15.4 difference is that selection continues to exhaustion
    # instead of stopping after five.
    while remaining:
        anchors = [
            seen,
            probe,
            *selected,
        ]

        choice = max(
            remaining,
            key=lambda start: (
                min(
                    abs(start - anchor)
                    for anchor in anchors
                ),
                abs(start - probe),
                -start,
            ),
        )

        selected.append(choice)
        remaining.remove(choice)

    if len(selected) != len(set(selected)):
        raise RuntimeError(
            "safe-start reservoir contains duplicate starts"
        )

    if set(selected) != set(eligible):
        raise RuntimeError(
            "safe-start reservoir does not cover every legal TRAIN start"
        )

    if probe in selected:
        raise RuntimeError(
            "probe start leaked into TRAIN reservoir"
        )

    if any(
        abs(start - probe)
        <= PROBE_START_GUARD_FRAMES
        for start in selected
    ):
        raise RuntimeError(
            "TRAIN reservoir leaked into the probe guard"
        )

    return tuple(selected)


def _context_fit_starts(
    frames,
    width,
    recipe_id,
    count=FIT_CONTEXT_COUNT,
):
    """Compatibility view of the deterministic safe-start reservoir.

    Existing callers requesting C5 continue to receive exactly five
    spread-first TRAIN cuts.  V15.4 reservoir construction uses the complete
    ordering through ``_all_probe_safe_context_starts``.
    """
    starts = _all_probe_safe_context_starts(
        frames,
        width,
        recipe_id,
    )

    if count is None:
        return starts

    count = int(count)

    if count < 1:
        raise ValueError(
            "fit context count must be positive"
        )

    if len(starts) < count:
        raise ValueError(
            "motion window cannot support separated fit contexts"
        )

    return tuple(
        starts[:count]
    )


def _context_reservoir_cycle_length(frames):
    """Common deterministic reservoir cycle for all role/width groups.

    Width-specific reservoirs can have slightly different legal-start counts.
    A common cycle equal to their maximum length guarantees that every legal
    start of every group appears at least once.  Shorter reservoirs wrap
    deterministically; no probe context is introduced.
    """
    frames = int(frames)

    lengths = []

    for recipe_id, requested_width in enumerate(
        (10, 28)
    ):
        width = min(
            requested_width,
            frames - 8,
        )

        starts = _all_probe_safe_context_starts(
            frames,
            width,
            recipe_id,
        )

        if len(starts) < FIT_CONTEXT_COUNT:
            raise ValueError(
                "reservoir cannot provide five contexts per transaction"
            )

        lengths.append(
            len(starts)
        )

    cycle = max(lengths)

    if cycle < FIT_CONTEXT_COUNT:
        raise RuntimeError(
            "invalid safe-start reservoir cycle"
        )

    return cycle


def _split_start(
    split,
    frames,
    width,
    recipe_id,
):
    seen, probe = _seen_and_probe_starts(
        frames,
        width,
        recipe_id,
    )

    if split == "seen":
        return seen

    if split == "new_position":
        return probe

    prefix = "fit_context_"

    if not split.startswith(prefix):
        raise ValueError(
            f"unknown bridge diagnostic split: {split}"
        )

    context_index = int(
        split[len(prefix):]
    )

    if context_index < 0:
        raise ValueError(
            f"invalid fit context index: {context_index}"
        )

    starts = _all_probe_safe_context_starts(
        frames,
        width,
        recipe_id,
    )

    if not starts:
        raise RuntimeError(
            "empty TRAIN context reservoir"
        )

    start = starts[
        context_index % len(starts)
    ]

    if start == probe or (
        abs(start - probe)
        <= PROBE_START_GUARD_FRAMES
    ):
        raise RuntimeError(
            "reservoir replay selected a probe-guard start"
        )

    return start


def build_banks(
    clean,
    cond,
    sources,
    cfg,
    device,
    *,
    contact_ik=True,
    include_fit_contexts=False,
):
    banks = {}
    recipes = {}
    splits = ["seen", "new_position"]
    if include_fit_contexts:
        if len(clean) < 1:
            raise ValueError(
                "cannot build context reservoir from an empty TRAIN bank"
            )

        frames = int(
            len(clean[0])
        )

        if any(
            len(original) != frames
            for original in clean
        ):
            raise ValueError(
                "V15.4 reservoir requires equal-length TRAIN windows"
            )

        reservoir_count = (
            _context_reservoir_cycle_length(
                frames
            )
        )

        splits.extend(
            f"fit_context_{context_index}"
            for context_index in range(
                reservoir_count
            )
        )
    for split in splits:
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
                    a = _split_start(split, len(original), width, recipe_id)
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


def evaluate(model, banks, split, cfg, *, predictions=None, progress=True):
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
            if progress:
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


def _diagnostic_group_guard_values(terms, group_objectives):
    """Expose exact fixed-bank components instead of one feasibility sum.

    All values come from the same differentiable metric implementations used by
    the stage guard. Physical entries are signed residuals against the exact
    reference-relative allowed value and comparison epsilon, including the
    reversed sign for low-is-bad penetration. Endpoint and temporal remain
    separate representations of the unchanged observable 0.03 requirements.
    No component can be hidden by a lower aggregate loss.
    """
    values = {}
    for label in m.REFINER_GROUP_LABELS:
        objective = group_objectives[label]
        values[f"{label}.observable_endpoint_0p03"] = terms[
            f"group_{label}_endpoint_scientific_tail_risk"
        ]
        values[f"{label}.observable_temporal_0p03"] = terms[
            f"group_{label}_temporal_scientific_tail_risk"
        ]
        physical_terms = {
            "joint_jerk_p95": (
                "repair_joint_jerk_mps3_p95_signed_margin_max"
            ),
            "joint_jerk_max": (
                "repair_joint_jerk_mps3_max_signed_margin_max"
            ),
            "joint_jerk_window_p95": (
                "repair_joint_jerk_window_p95_max_mps3_signed_margin_max"
            ),
            "extremity_jerk_p95": (
                "repair_extremity_jerk_mps3_p95_signed_margin_max"
            ),
            "extremity_jerk_window_p95": (
                "repair_extremity_jerk_window_p95_max_mps3_signed_margin_max"
            ),
            "foot_skate_p95": (
                "repair_foot_skate_mps_p95_signed_margin_max"
            ),
            "foot_skate_max": (
                "repair_foot_skate_mps_max_signed_margin_max"
            ),
            "support_drift_p95": (
                "repair_foot_support_drift_m_p95_signed_margin_max"
            ),
            "support_drift_max": (
                "repair_foot_support_drift_m_max_signed_margin_max"
            ),
            "penetration": (
                "repair_foot_penetration_min_m_signed_margin_max"
            ),
            "boundary": "boundary_jerk_signed_margin_max",
        }
        for suffix, term_suffix in physical_terms.items():
            values[f"{label}.{suffix}"] = terms[
                f"group_{label}_{term_suffix}"
            ]
        support_parts = [
            values[f"{label}.{suffix}"]
            for suffix in (
                "foot_skate_p95",
                "foot_skate_max",
                "support_drift_p95",
                "support_drift_max",
                "penetration",
            )
        ]
        values[f"{label}.fixed_support"] = m.torch.stack(
            support_parts
        ).max()
        values[f"{label}.fidelity_geometry"] = objective[
            "clean_geometry_max"
        ]
        values[f"{label}.fidelity_contact"] = objective[
            "clean_contact_max"
        ]
        values[f"{label}.fidelity_temporal"] = objective[
            "clean_temporal_excess"
        ]
        values[f"{label}.fidelity_support"] = objective[
            "clean_support_excess"
        ]
    return values


def _diagnostic_guarded_loss(model, batch, cfg):
    groups = {}
    repair, protection, terms, _ = m._refiner_batch_objectives(
        model,
        batch,
        cfg,
        group_objectives=groups,
    )
    total = repair + cfg.product_refiner_clean_identity_weight * protection
    return total, _diagnostic_group_guard_values(terms, groups)


def _fixed_group_guard_metrics(model, batch, cfg):
    """Measure guard components on one immutable TRAIN anchor bank."""
    with m.torch.no_grad():
        _, values = _diagnostic_guarded_loss(model, batch, cfg)
    return {
        key: float(value.detach())
        for key, value in values.items()
    }


def _fixed_anchor_guarded_loss(model, train_batch, guard_batch, cfg):
    """Use the rotating C5 batch for Armijo and the fixed bank for guards."""
    train_loss = m._refiner_total_batch_loss(model, train_batch, cfg)
    _, fixed_groups = _diagnostic_guarded_loss(model, guard_batch, cfg)
    return train_loss, fixed_groups


def _mixed_group_guard_reference(anchor, best):
    """Return one immutable component anchor; ``best`` is diagnostic only."""
    if set(anchor) != set(best):
        raise ValueError("guard anchor/best keys differ")
    return dict(anchor)


def _group_guard_tolerances(anchor, cfg):
    """Return fixed, nonaccumulating allowances in audited metric units."""
    relative = {}
    absolute = {}
    base_relative = float(
        cfg.product_refiner_group_guard_relative_tolerance
    )
    base_absolute = float(
        cfg.product_refiner_group_guard_absolute_tolerance
    )
    for key in anchor:
        suffix = key.rsplit(".", 1)[-1] if "." in key else "total"
        if suffix == "fidelity_geometry":
            relative[key] = 0.0
            absolute[key] = max(
                0.0,
                float(
                    cfg.checkpoint_validation_max_clean_identity_product_log_l1
                ) - float(anchor[key]),
            )
        elif suffix == "fidelity_contact":
            relative[key] = 0.0
            absolute[key] = max(
                0.0,
                float(
                    cfg.checkpoint_validation_max_clean_identity_contact_l1
                ) - float(anchor[key]),
            )
        else:
            # Endpoint/temporal and physical excesses retain a fixed numerical
            # allowance around the immutable anchor. The clean no-op loss is a
            # soft objective and is deliberately absent from this hard guard.
            relative[key] = 0.0
            absolute[key] = base_absolute
    return relative, absolute


def _group_guard_metric_metadata(anchor, relative, absolute, cfg):
    """Describe every fixed Guard component and its unchanged source limit."""
    limits = physical_quality.PhysicalQualityLimits.from_environment()
    configured = {
        "joint_jerk_p95": limits.joint_jerk_mps3_p95,
        "joint_jerk_max": limits.joint_jerk_mps3_max,
        "joint_jerk_window_p95": limits.joint_jerk_window_p95_max_mps3,
        "extremity_jerk_p95": limits.extremity_jerk_mps3_p95,
        "extremity_jerk_window_p95": (
            limits.extremity_jerk_window_p95_max_mps3
        ),
        "foot_skate_p95": limits.foot_skate_mps_p95,
        "foot_skate_max": limits.foot_skate_mps_max,
        "support_drift_p95": limits.foot_support_drift_m_p95,
        "support_drift_max": limits.foot_support_drift_m_max,
        "penetration": limits.foot_penetration_min_m,
        "observable_endpoint_0p03": (
            cfg.checkpoint_validation_min_endpoint_repair_gain
        ),
        "observable_temporal_0p03": (
            cfg.checkpoint_validation_min_temporal_repair_gain
        ),
        "fidelity_geometry": (
            cfg.checkpoint_validation_max_clean_identity_product_log_l1
        ),
        "fidelity_contact": (
            cfg.checkpoint_validation_max_clean_identity_contact_l1
        ),
    }
    categories = {
        "joint_jerk_p95": "joint_jerk",
        "joint_jerk_max": "joint_jerk",
        "joint_jerk_window_p95": "joint_jerk",
        "extremity_jerk_p95": "extremity_jerk",
        "extremity_jerk_window_p95": "extremity_jerk",
        "foot_skate_p95": "foot_skate",
        "foot_skate_max": "foot_skate",
        "support_drift_p95": "support_drift",
        "support_drift_max": "support_drift",
        "penetration": "penetration",
        "fixed_support": "fixed_support",
        "boundary": "boundary",
        "fidelity_geometry": "fidelity",
        "fidelity_contact": "fidelity",
        "fidelity_temporal": "fidelity",
        "fidelity_support": "fidelity",
        "observable_endpoint_0p03": "observable_0p03",
        "observable_temporal_0p03": "observable_0p03",
    }
    metadata = {}
    for key, fixed_anchor in anchor.items():
        suffix = key.split(".", 1)[1]
        allowance = max(
            abs(float(fixed_anchor)) * float(relative[key]),
            float(absolute[key]),
        )
        metadata[key] = {
            "category": categories[suffix],
            "guard_value_domain": (
                "exact_stage_signed_residual_or_observable_fidelity_metric"
            ),
            "metric_direction": "high",
            "fixed_anchor": float(fixed_anchor),
            "guard_absolute_limit": float(fixed_anchor) + allowance,
            "absolute_upper_limit": float(fixed_anchor) + allowance,
            "configured_absolute_limit": (
                float(configured[suffix]) if suffix in configured else None
            ),
            "allowance": allowance,
            "rolling_tolerance_accumulation": False,
        }
    return metadata


def _tuple_dot(left, right):
    values = [
        (a.double() * b.double()).sum()
        for a, b in zip(left, right)
    ]
    return m.torch.stack(values).sum()


def _subgroup_scientific_objectives(terms, cfg):
    """Return the eight guarded role/width scientific objectives."""
    objectives = {}
    scientific_weight = float(
        cfg.product_refiner_repair_margin_weight
    )
    temporal_weight = float(
        m.REFINER_TEMPORAL_SCIENTIFIC_WEIGHT
    )
    for label in m.REFINER_GROUP_LABELS:
        endpoint_key = (
            f"group_{label}_endpoint_scientific_tail_risk"
        )
        temporal_key = (
            f"group_{label}_temporal_scientific_tail_risk"
        )
        if endpoint_key not in terms or temporal_key not in terms:
            raise RuntimeError(
                f"missing subgroup scientific objectives for {label}"
            )
        objectives[f"{label}.endpoint"] = (
            scientific_weight * terms[endpoint_key]
        )
        objectives[f"{label}.temporal"] = (
            scientific_weight
            * temporal_weight
            * terms[temporal_key]
        )
    return objectives


def _deterministic_mgda_weights(gram):
    """Minimize alpha.T @ gram @ alpha on the probability simplex.

    The problem has only eight variables.  Exact line-search Frank-Wolfe is
    deterministic, dependency-free and sufficient for deciding whether the
    normalized subgroup gradients contain a nonzero common descent vector.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("MGDA Gram matrix must be square")
    count = int(gram.shape[0])
    if count < 1:
        raise ValueError("MGDA requires at least one active gradient")
    alpha = gram.new_zeros((count,))
    alpha[int(m.torch.argmin(m.torch.diagonal(gram)).item())] = 1.0
    duality_gap = float("inf")
    iterations = 0
    for iterations in range(1, MGDA_MAX_ITERATIONS + 1):
        gradient = 2.0 * gram.mv(alpha)
        vertex_index = int(m.torch.argmin(gradient).item())
        vertex = gram.new_zeros((count,))
        vertex[vertex_index] = 1.0
        direction = vertex - alpha
        duality_gap = float(
            (alpha.dot(gradient) - gradient[vertex_index]).detach()
        )
        if duality_gap <= MGDA_DUALITY_GAP_TOLERANCE:
            break
        denominator = direction.dot(gram.mv(direction))
        if float(denominator.detach()) <= 0.0:
            break
        numerator = -alpha.dot(gram.mv(direction))
        step = (numerator / denominator).clamp(0.0, 1.0)
        alpha = alpha + step * direction
    alpha = alpha.clamp_min(0.0)
    alpha = alpha / alpha.sum().clamp_min(1.0e-24)
    return alpha, iterations, duality_gap


def _pareto_common_descent_backward(
    model,
    total_loss,
    terms,
    cfg,
    *,
    fixed_guard_values=None,
    guard_reference=None,
    guard_relative_tolerance=None,
    guard_absolute_tolerance=None,
    guard_metadata=None,
):
    """Backpropagate an exact-Guard-constrained eight-objective direction.

    Gradients are separated by role, width and observable, then normalized by
    their parameter RMS before the minimum-norm simplex problem is solved.  A
    near-zero convex-hull projection is reported as Pareto stationary and does
    not enter line search.  The non-scientific remainder is admitted only while
    every subgroup directional derivative remains non-positive for the actual
    update direction.  V15.12f additionally differentiates active constraints
    on the exact immutable Guard bank and projects the common direction into
    their linearized non-regression halfspaces before any real candidate is
    constructed. Exact closure and fixed-anchor guards remain decisive.
    """
    objectives = _subgroup_scientific_objectives(terms, cfg)
    parameters = [p for p in model.parameters() if p.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count < 1:
        raise RuntimeError("MGDA found no trainable Refiner parameters")
    task_names = list(objectives)
    task_gradients = []
    for objective in objectives.values():
        raw = m.torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        task_gradients.append([
            m.torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, raw)
        ])
    scientific_weight = float(
        cfg.product_refiner_repair_margin_weight
    )
    aggregate_scientific = scientific_weight * (
        terms["endpoint_training_objective"]
        + terms["temporal_training_objective"]
    )
    aggregate_raw = m.torch.autograd.grad(
        aggregate_scientific,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    aggregate_science_grad = [
        m.torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, aggregate_raw)
    ]
    constraint_records = {}
    constraint_gradients = []
    constraint_names = []
    guard_enabled = fixed_guard_values is not None
    if guard_enabled:
        mappings = (
            guard_reference,
            guard_relative_tolerance,
            guard_absolute_tolerance,
            guard_metadata,
        )
        if any(value is None for value in mappings):
            raise ValueError("exact Guard constraints require complete metadata")
        keys = set(fixed_guard_values)
        if any(set(value) != keys for value in mappings):
            raise ValueError("exact Guard constraint keys differ")
        for name, objective in fixed_guard_values.items():
            current = float(objective.detach())
            reference = float(guard_reference[name])
            allowance = max(
                abs(reference) * float(guard_relative_tolerance[name]),
                float(guard_absolute_tolerance[name]),
            )
            allowed = reference + allowance
            remaining = allowed - current
            numeric_tolerance = max(
                1.0e-12,
                abs(allowed) * 1.0e-9,
                allowance * 1.0e-6,
            )
            active_band = max(
                numeric_tolerance,
                allowance * EXACT_GUARD_ACTIVE_MARGIN_FRACTION,
            )
            record = {
                **dict(guard_metadata[name]),
                "current": current,
                "allowed": allowed,
                "remaining_margin": remaining,
                "numeric_tolerance": numeric_tolerance,
                "active_margin_band": active_band,
                "active": bool(
                    remaining <= active_band
                    or guard_metadata[name]["category"] == "observable_0p03"
                    or (
                        guard_metadata[name]["category"] != "fidelity"
                        and current > numeric_tolerance
                    )
                ),
            }
            constraint_records[name] = record
            if not record["active"]:
                continue
            raw = m.torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradient = [
                m.torch.zeros_like(parameter) if value is None else value
                for parameter, value in zip(parameters, raw)
            ]
            constraint_names.append(name)
            constraint_gradients.append(gradient)
    total_loss.backward()
    total_grad = [
        m.torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        for p in parameters
    ]
    epsilon = total_loss.new_tensor(1.0e-24, dtype=m.torch.float64)
    count_tensor = total_loss.new_tensor(
        float(parameter_count), dtype=m.torch.float64
    )
    norm_squares = m.torch.stack([
        _tuple_dot(gradient, gradient)
        for gradient in task_gradients
    ])
    rms_norms = (norm_squares / count_tensor).clamp_min(0.0).sqrt()
    active_indices = [
        index for index, value in enumerate(rms_norms)
        if float(value.detach()) > 1.0e-12
    ]
    normalized = []
    for index in active_indices:
        divisor = rms_norms[index].clamp_min(epsilon)
        normalized.append([
            value / divisor.to(value.dtype)
            for value in task_gradients[index]
        ])
    cosine_matrix = []
    for left_index, left in enumerate(task_gradients):
        row = []
        for right_index, right in enumerate(task_gradients):
            denominator = (
                norm_squares[left_index].clamp_min(epsilon).sqrt()
                * norm_squares[right_index].clamp_min(epsilon).sqrt()
            )
            value = _tuple_dot(left, right) / denominator
            if not (
                float(rms_norms[left_index].detach()) > 1.0e-12
                and float(rms_norms[right_index].detach()) > 1.0e-12
            ):
                value = value.new_zeros(())
            row.append(float(value.detach()))
        cosine_matrix.append(row)
    full_weights = total_loss.new_zeros(
        (len(task_names),), dtype=m.torch.float64
    )
    iterations = 0
    duality_gap = 0.0
    if normalized:
        gram = m.torch.stack([
            m.torch.stack([
                _tuple_dot(left, right) / count_tensor
                for right in normalized
            ])
            for left in normalized
        ])
        active_weights, iterations, duality_gap = (
            _deterministic_mgda_weights(gram)
        )
        for offset, index in enumerate(active_indices):
            full_weights[index] = active_weights[offset]
        normalized_common = [
            sum(
                active_weights[index].to(values[0].dtype) * values[parameter]
                for index, values in enumerate(normalized)
            )
            for parameter in range(len(parameters))
        ]
        mgda_min_norm = float(
            (_tuple_dot(normalized_common, normalized_common)
             / count_tensor).clamp_min(0.0).sqrt().detach()
        )
    else:
        normalized_common = [m.torch.zeros_like(value) for value in parameters]
        mgda_min_norm = 0.0
    unconstrained_common_exists = bool(
        mgda_min_norm > MGDA_COMMON_DESCENT_RMS_EPSILON
    )
    common_exists = unconstrained_common_exists
    nonzero_norms = [
        rms_norms[index] for index in active_indices
    ]
    target_rms = (
        m.torch.stack(nonzero_norms).median()
        if nonzero_norms
        else total_loss.new_zeros((), dtype=m.torch.float64)
    )
    common = [
        value * target_rms.to(value.dtype)
        for value in normalized_common
    ]
    projection_passes = 0
    active_constraint_gradients = []
    active_constraint_raw_gradients = []
    active_constraint_names = []
    for name, gradient in zip(constraint_names, constraint_gradients):
        norm_square = _tuple_dot(gradient, gradient)
        rms = (norm_square / count_tensor).clamp_min(0.0).sqrt()
        constraint_records[name]["gradient_rms"] = float(rms.detach())
        if float(rms.detach()) <= 1.0e-12:
            constraint_records[name]["active"] = False
            constraint_records[name]["inactive_reason"] = "zero_constraint_gradient"
            continue
        active_constraint_names.append(name)
        active_constraint_raw_gradients.append(gradient)
        active_constraint_gradients.append([
            value / rms.to(value.dtype)
            for value in gradient
        ])
    if common_exists and active_constraint_gradients:
        for projection_passes in range(1, EXACT_GUARD_PROJECTION_MAX_PASSES + 1):
            changed = False
            for gradient in active_constraint_gradients:
                dot = _tuple_dot(common, gradient)
                denominator = _tuple_dot(gradient, gradient).clamp_min(1.0e-24)
                tolerance = EXACT_GUARD_DERIVATIVE_EPSILON * max(
                    1.0,
                    abs(float(dot.detach())),
                )
                if float(dot.detach()) < -tolerance:
                    coefficient = dot / denominator
                    common = [
                        value - coefficient.to(value.dtype) * normal
                        for value, normal in zip(common, gradient)
                    ]
                    changed = True
            if not changed:
                break
    remainder = [
        total - science
        for total, science in zip(total_grad, aggregate_science_grad)
    ]
    remainder_scale = 1.0
    for task_gradient in task_gradients:
        common_dot = _tuple_dot(common, task_gradient)
        remainder_dot = _tuple_dot(remainder, task_gradient)
        common_value = float(common_dot.detach())
        remainder_value = float(remainder_dot.detach())
        if remainder_value < 0.0:
            remainder_scale = min(
                remainder_scale,
                max(0.0, 0.95 * common_value / (-remainder_value)),
            )
    for constraint_gradient in active_constraint_gradients:
        common_dot = _tuple_dot(common, constraint_gradient)
        remainder_dot = _tuple_dot(remainder, constraint_gradient)
        common_value = float(common_dot.detach())
        remainder_value = float(remainder_dot.detach())
        if remainder_value < 0.0:
            remainder_scale = min(
                remainder_scale,
                max(0.0, 0.95 * common_value / (-remainder_value)),
            )
    final_grad = (
        [
            science + remainder_scale * other
            for science, other in zip(common, remainder)
        ]
        if common_exists
        else [m.torch.zeros_like(value) for value in parameters]
    )
    directional_derivatives = {
        name: float(-_tuple_dot(final_grad, gradient).detach())
        for name, gradient in zip(task_names, task_gradients)
    }
    constraint_directional_derivatives = {
        name: float(-_tuple_dot(final_grad, gradient).detach())
        for name, gradient in zip(
            active_constraint_names,
            active_constraint_raw_gradients,
        )
    }
    derivative_scale = max(
        1.0,
        max(abs(value) for value in directional_derivatives.values()),
    )
    derivative_tolerance = 1.0e-10 * derivative_scale
    all_nonpositive = all(
        value <= derivative_tolerance
        for value in directional_derivatives.values()
    )
    one_strict = any(
        value < -derivative_tolerance
        for value in directional_derivatives.values()
    )
    constraint_scale = max(
        1.0,
        max(
            (abs(value) for value in constraint_directional_derivatives.values()),
            default=0.0,
        ),
    )
    constraint_tolerance = EXACT_GUARD_DERIVATIVE_EPSILON * constraint_scale
    constraints_nonpositive = all(
        value <= constraint_tolerance
        for value in constraint_directional_derivatives.values()
    )
    common_exists = bool(
        common_exists
        and all_nonpositive
        and one_strict
        and constraints_nonpositive
    )
    if not common_exists:
        final_grad = [m.torch.zeros_like(value) for value in parameters]
        directional_derivatives = {
            name: 0.0 for name in task_names
        }
        constraint_directional_derivatives = {
            name: 0.0 for name in active_constraint_names
        }
    for parameter, gradient in zip(parameters, final_grad):
        parameter.grad = gradient
    for name, derivative in constraint_directional_derivatives.items():
        constraint_records[name]["directional_derivative"] = derivative

    def guard_directional_derivatives(direction):
        return {
            name: float(_tuple_dot(gradient, direction).detach())
            for name, gradient in zip(constraint_names, constraint_gradients)
        }

    return {
        "protocol": "deterministic_exact_guard_constrained_subgroup_mgda_v1",
        "active": True,
        "task_names": task_names,
        "gradient_cosine_matrix": cosine_matrix,
        "gradient_norms_before_normalization": {
            name: float(rms.detach())
            for name, rms in zip(task_names, rms_norms)
        },
        "mgda_weights": {
            name: float(value.detach())
            for name, value in zip(task_names, full_weights)
        },
        "mgda_min_norm": mgda_min_norm,
        "mgda_iterations": int(iterations),
        "mgda_duality_gap": float(duality_gap),
        "common_direction_exists": common_exists,
        "unconstrained_common_descent_exists": (
            unconstrained_common_exists
        ),
        "common_descent_exists": common_exists,
        "constrained_common_descent_exists": common_exists,
        "reason": (
            "exact_guard_constrained_subgroup_common_descent"
            if common_exists
            else (
                "no_exact_guard_constrained_common_descent"
                if active_constraint_names
                else "pareto_stationary_or_no_common_descent"
            )
        ),
        "directional_derivatives": directional_derivatives,
        "all_directional_derivatives_nonpositive": all_nonpositive,
        "at_least_one_directional_derivative_strict": one_strict,
        "gradient_rms_rescale": float(target_rms.detach()),
        "non_scientific_remainder_scale": float(remainder_scale),
        "active_constraint_gradients": active_constraint_names,
        "active_constraint_count": len(active_constraint_names),
        "constraint_projection_passes": int(projection_passes),
        "constraint_directional_derivatives": (
            constraint_directional_derivatives
        ),
        "all_constraint_derivatives_nonpositive": constraints_nonpositive,
        "constraint_metrics": constraint_records,
        "_guard_directional_derivative_callback": (
            guard_directional_derivatives if guard_enabled else None
        ),
    }


def _diagnostic_optimizer(model, cfg):
    """Use a bounded output-head warmup without changing model architecture."""
    output_parameters = list(getattr(model, "out", model).parameters())
    output_ids = {id(parameter) for parameter in output_parameters}
    backbone = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in output_ids
    ]
    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "lr": cfg.lr,
            "diagnostic_role": "backbone",
        })
    groups.append({
        "params": output_parameters,
        "lr": cfg.lr * OUTPUT_WARMUP_LR_MULTIPLIER,
        "diagnostic_role": "output",
    })
    return m.torch.optim.AdamW(groups, weight_decay=1.0e-4)


def _set_diagnostic_learning_rates(optimizer, cfg, step):
    warmup = int(step) <= OUTPUT_WARMUP_STEPS
    for group in optimizer.param_groups:
        role = group.get("diagnostic_role", "backbone")
        group["lr"] = float(cfg.lr) * (
            OUTPUT_WARMUP_LR_MULTIPLIER
            if warmup and role == "output"
            else 1.0
        )
    return {
        "active": warmup,
        "steps": OUTPUT_WARMUP_STEPS,
        "output_lr_multiplier": (
            OUTPUT_WARMUP_LR_MULTIPLIER if warmup else 1.0
        ),
        "parameter_group_lrs": [
            float(group["lr"])
            for group in optimizer.param_groups
        ],
    }


def _decoder_amplitude_summary(trace, seam=None):
    repair = trace.get("repair", {})
    if not repair or seam is None:
        return {}
    active = seam[..., 0] >= 0.5
    result = {}
    for name in ("raw", "after_mask", "after_taper", "applied"):
        value = repair.get(name)
        if value is None:
            continue
        selected = value[active]
        result[f"{name}_tangent_rms"] = float(
            selected.double().square().mean().sqrt().detach()
        ) if selected.numel() else 0.0
        result[f"{name}_tangent_abs_max"] = float(
            selected.abs().max().detach()
        ) if selected.numel() else 0.0
    return result


def _decoder_amplitude_by_group(trace, seam, group):
    """Report output magnitude separately for all four scientific groups."""
    result = {}
    for index, label in enumerate(m.REFINER_GROUP_LABELS):
        selected = group == index
        if not bool(selected.any()):
            continue
        repair = trace.get("repair", {})
        sliced = {
            key: value[selected]
            for key, value in repair.items()
            if m.torch.is_tensor(value) and value.shape[0] == group.shape[0]
        }
        result[label] = _decoder_amplitude_summary(
            {"repair": sliced},
            seam[selected],
        )
    return result


def evaluate_fit_contexts(model, banks, cfg):
    """Evaluate every fitted temporal cut independently from held-out probes."""
    from training.bridge_feasibility import group_decisions

    contexts = {}
    passed = 0
    group_pass_counts = Counter()
    indices = _fit_context_indices(banks)

    for index in indices:
        split = f"fit_context_{index}"
        metrics = evaluate(
            model,
            banks,
            split,
            cfg,
            progress=False,
        )
        decision = m._checkpoint_validation_decision(
            metrics,
            cfg,
            stage="refiner",
        )
        groups = group_decisions(metrics, cfg)
        context_passed = bool(
            decision["scientific_acceptance"]
            and all(row["passed"] for row in groups.values())
        )
        passed += int(context_passed)
        for label, row in groups.items():
            group_pass_counts[label] += int(row["passed"])
        contexts[split] = {
            "passed": context_passed,
            "reasons": list(decision["reasons"]),
            "observed": dict(decision["observed"]),
            "group_decisions": groups,
            "failure_breakdown": failure_breakdown(metrics),
            "used_for_optimizer_updates": True,
            "held_out_probe": False,
        }
        completed = len(contexts)
        if completed % 8 == 0 or completed == len(indices):
            print(json.dumps({
                "stage": "bridge_fit_context_probe",
                "completed": completed,
                "total": len(indices),
                "contexts_passed": passed,
            }), flush=True)

    total = len(indices)
    return {
        "schema": "refiner_fit_context_independent_evaluation_v1",
        "contexts_evaluated": total,
        "context_indices": list(indices),
        "contexts_passed": passed,
        "contexts_failed": total - passed,
        "pass_rate": float(passed / total) if total else 0.0,
        "all_contexts_passed": bool(total and passed == total),
        "group_pass_counts": dict(sorted(group_pass_counts.items())),
        "probe_cases_used": False,
        "contexts": contexts,
    }


def learning_scope_diagnosis(fit_contexts, probe_decisions, probe_groups):
    """Separate inability to fit TRAIN cuts from held-out-cut failure."""
    fit_passed = bool(fit_contexts["all_contexts_passed"])
    probe_passed = bool(
        all(row["scientific_acceptance"] for row in probe_decisions.values())
        and all(
            group["passed"]
            for groups in probe_groups.values()
            for group in groups.values()
        )
    )
    if fit_passed and probe_passed:
        classification = "fit_context_and_probe_passed"
    elif fit_passed:
        classification = "fit_context_passed_probe_generalization_failed"
    else:
        classification = "fit_context_learning_failed"
    return {
        "schema": "refiner_learning_scope_diagnosis_v1",
        "classification": classification,
        "fit_context_passed": fit_passed,
        "held_out_probe_passed": probe_passed,
        "new_position_used_for_optimizer_updates": False,
    }


def run(args):
    cfg = m.MotionGenerationConfig.from_json(args.config).apply_env()
    if args.check_report:
        report = json.loads(Path(args.check_report).read_text(encoding="utf8"))
        if report.get("schema") != SCHEMA or report.get("fingerprint") != fingerprint(args,cfg):
            raise RuntimeError("bridge diagnostic protocol/config/code/database mismatch")
        if not report.get("completed") or report.get("published") is not False:
            raise RuntimeError("diagnostic not completed or incorrectly published")
        if report.get("stopped_early"):
            raise RuntimeError("TRAIN context-cycle optimization stalled; review the diagnostic, do not train")
        if (report.get("target_steps") != 400 or report.get("completed_steps") != 400
                or len(report.get("windows",[])) != args.windows or args.windows != 8):
            raise RuntimeError("pilot requires the complete 8-window, 400-step protocol; smoke runs cannot authorize training")
        validate_update_summary(report.get("optimizer_updates", {}), 400)
        if report.get("fit_bank") != fit_bank_contract(args.windows, cfg):
            raise RuntimeError("diagnostic did not use the complete predefined TRAIN context cycle")
        guard = report.get("group_guard_contract", {})
        if (
            guard.get("schema")
            != "refiner_exact_metric_anchor_guard_v4"
            or guard.get("bank") != "complete_seen_train_anchor"
            or guard.get("fixed_across_all_steps") is not True
            or guard.get("rolling_pre_step_reference_forbidden") is not True
        ):
            raise RuntimeError(
                "diagnostic did not use the fixed component-anchor guard"
            )
        fit_contexts = report.get("fit_context_evaluation", {})
        if (
            fit_contexts.get("schema")
            != "refiner_fit_context_independent_evaluation_v1"
            or fit_contexts.get("probe_cases_used") is not False
            or fit_contexts.get("all_contexts_passed") is not True
        ):
            raise RuntimeError(
                "independent fit-context evaluation failed; do not train"
            )
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
    banks, recipes = build_banks(
        clean,
        cond,
        sources,
        cfg,
        device,
        include_fit_contexts=not getattr(args,"baseline_only",False),
    )
    if getattr(args,"baseline_only",False):
        pure,_ = build_banks(clean,cond,sources,cfg,device,contact_ik=False)
        return run_foundation(args,cfg,banks,pure,recipes,fingerprint(args,cfg),
            [{"path":str(db["paths"][i]),"sha256":common.file_sha256(db["paths"][i])} for i in selected],separation)
    train_schedule = _reservoir_transaction_schedule(banks)
    train_cycle_length = len(train_schedule)

    if train_cycle_length < 1:
        raise RuntimeError(
            "empty V15.4.1 reservoir transaction schedule"
        )

    if args.steps == 400 and train_cycle_length > args.steps:
        raise RuntimeError(
            "400-step scientific diagnostic cannot cover one "
            "complete safe-start reservoir cycle"
        )
    model = m.ProductManifoldTemporalRefiner(fps=cfg.fps).to(device)
    optimizer = _diagnostic_optimizer(model, cfg)
    fixed_guard_batch = fixed_fit_bank(banks, "seen")
    fixed_guard_anchor = _fixed_group_guard_metrics(
        model,
        fixed_guard_batch,
        cfg,
    )
    fixed_guard_current = dict(fixed_guard_anchor)
    fixed_guard_best = dict(fixed_guard_anchor)
    (
        guard_relative_tolerance,
        guard_absolute_tolerance,
    ) = _group_guard_tolerances(fixed_guard_anchor, cfg)
    guard_metric_metadata = _group_guard_metric_metadata(
        fixed_guard_anchor,
        guard_relative_tolerance,
        guard_absolute_tolerance,
        cfg,
    )
    destination.mkdir(parents=True)
    report = {"schema":SCHEMA,"protocol":m.BOUNDARY_PROTOCOL,"fingerprint":fingerprint(args,cfg),
              "completed":False,"published":False,"independent_validation":False,
              "probe_scope":PROBE_SCOPE,
              "formal_training_must_start_fresh":True,"selection":"fixed_final_step",
              "foundation_report":str(Path(args.foundation_report).resolve()),
              "fit_bank":fit_bank_contract(args.windows, cfg),
              "source_separation":separation,"recipes":recipes,"target_steps":args.steps,
              "candidate_audit_artifact": str(
                  (destination / "optimizer_updates.jsonl").resolve()
              ),
              "gradient_audit_artifact": str(
                  (destination / "gradients.jsonl").resolve()
              ),
              "windows":[{"path":str(db["paths"][i]),"sha256":common.file_sha256(db["paths"][i])} for i in selected],
              "baseline":{},"history":[],
              "group_guard_contract": {
                  "schema": "refiner_exact_metric_anchor_guard_v4",
                  "bank": "complete_seen_train_anchor",
                  "fixed_across_all_steps": True,
                  "rolling_pre_step_reference_forbidden": True,
                  "historical_component_minimum_intersection": False,
                  "joint_reference": "not_used",
                  "component_reference": "immutable_initial_anchor",
                  "relative_tolerance": dict(guard_relative_tolerance),
                  "absolute_tolerance": dict(guard_absolute_tolerance),
                  "absolute_tolerance_domain":
                      "configured_gate_metrics_and_zero_excess",
                  "soft_clean_noop_used_as_hard_guard": False,
                  "clean_geometry_threshold": float(
                      cfg.checkpoint_validation_max_clean_identity_product_log_l1
                  ),
                  "clean_contact_threshold": float(
                      cfg.checkpoint_validation_max_clean_identity_contact_l1
                  ),
                  "production_gate_thresholds_changed": False,
                  "initial_anchor": dict(fixed_guard_anchor),
                  "best_so_far": dict(fixed_guard_best),
                  "current": dict(fixed_guard_current),
                  "metric_metadata": guard_metric_metadata,
                  "coarse_feasibility_sum_removed": True,
              },
              "output_amplitude_contract": {
                  "schema": "refiner_output_head_trust_warmup_v1",
                  "steps": OUTPUT_WARMUP_STEPS,
                  "lr_multiplier": OUTPUT_WARMUP_LR_MULTIPLIER,
                  "bounded_by_checked_step": True,
                  "decoder_caps_changed": False,
              },
              "multiobjective_contract": {
                  "schema": "refiner_exact_guard_constrained_subgroup_mgda_v2",
                  "protocol":
                      "deterministic_exact_guard_constrained_subgroup_mgda",
                  "objectives": [
                      f"{label}.{component}"
                      for label in m.REFINER_GROUP_LABELS
                      for component in ("endpoint", "temporal")
                  ],
                  "gradient_count": 8,
                  "solver": "deterministic_frank_wolfe_exact_line_search",
                  "max_iterations": MGDA_MAX_ITERATIONS,
                  "duality_gap_tolerance":
                      MGDA_DUALITY_GAP_TOLERANCE,
                  "common_descent_rms_epsilon":
                      MGDA_COMMON_DESCENT_RMS_EPSILON,
                  "near_zero_policy":
                      "pareto_stationary_or_no_common_descent",
                  "actual_loss_closure_required": True,
                  "physical_remainder_requires_scientific_nonregression": True,
                  "fixed_guard_gradient_scope": "complete_seen_train_anchor",
                  "active_margin_fraction": EXACT_GUARD_ACTIVE_MARGIN_FRACTION,
                  "constraint_projection_max_passes": (
                      EXACT_GUARD_PROJECTION_MAX_PASSES
                  ),
                  "minimum_effective_scale": EXACT_GUARD_MIN_EFFECTIVE_SCALE,
                  "production_gate_thresholds_changed": False,
              },
              "multiobjective_diagnostics": {
                  "steps_evaluated": 0,
                  "common_descent_steps": 0,
                  "pareto_stationary_steps": 0,
                  "constrained_common_descent_steps": 0,
                  "no_constrained_common_descent_steps": 0,
                  "active_constraint_counts": {},
                  "last": None,
              }}
    report["fit_bank_artifact"] = save_fit_bank(
        destination,
        report,
        cfg,
        banks=banks,
        schedule=train_schedule,
    )
    report["probe_bank_artifact"] = save_probe_bank(
        destination, banks, report, cfg
    )
    save_diagnostic_state(destination,model,optimizer,report,0)
    for split in ("seen","new_position"):
        report["baseline"][split] = evaluate(None,banks,split,cfg)
    m.save_json(report,destination / "diagnostic_report.json")
    report["optimizer_updates"] = {}
    started = time.perf_counter()
    consecutive_context_stalls = 0
    for step in range(1,args.steps + 1):
        fit_context_index = (
            (step - 1)
            % train_cycle_length
        )

        selected_context_indices = (
            train_schedule[
                fit_context_index
            ]
        )

        batch = _reservoir_transaction_batch(
            banks,
            selected_context_indices,
        )
        logging = step == 1 or step % 25 == 0 or step == args.steps
        amplitude_trace = {} if logging else None
        warmup = _set_diagnostic_learning_rates(optimizer, cfg, step)
        repair,protection,terms,identity = m._refiner_batch_objectives(
            model,
            batch,
            cfg,
            trace=amplitude_trace,
        )
        loss = repair + cfg.product_refiner_clean_identity_weight * protection
        _, fixed_guard_values = _diagnostic_guarded_loss(
            model,
            fixed_guard_batch,
            cfg,
        )
        group_guard_before = {
            key: float(value.detach())
            for key, value in fixed_guard_values.items()
        }
        if set(group_guard_before) != set(fixed_guard_current):
            raise RuntimeError("fixed Guard metric layout changed during fitting")
        gradient = m._refiner_gradient_diagnostics(model,repair,protection,cfg.product_refiner_clean_identity_weight) if logging else None
        components = m._refiner_component_gradients(model,terms,cfg) if logging else None
        optimizer.zero_grad(set_to_none=True)
        pareto_gradient = _pareto_common_descent_backward(
            model,
            loss,
            terms,
            cfg,
            fixed_guard_values=fixed_guard_values,
            guard_reference=fixed_guard_anchor,
            guard_relative_tolerance=guard_relative_tolerance,
            guard_absolute_tolerance=guard_absolute_tolerance,
            guard_metadata=guard_metric_metadata,
        )
        guard_derivative_callback = pareto_gradient.pop(
            "_guard_directional_derivative_callback"
        )
        norm = float(m.torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True))
        group_guard_best_before = dict(fixed_guard_best)
        group_guard_reference = _mixed_group_guard_reference(
            fixed_guard_anchor,
            group_guard_best_before,
        )
        update = m.checked_refiner_step(
            optimizer,
            loss,
            lambda: _fixed_anchor_guarded_loss(
                model,
                batch,
                fixed_guard_batch,
                cfg,
            ),
            gradient_unscale=max(1.0, norm + 1.0e-6),
            group_guard_before=group_guard_before,
            group_guard_reference=group_guard_reference,
            group_guard_relative_tolerance=guard_relative_tolerance,
            group_guard_absolute_tolerance=guard_absolute_tolerance,
            group_guard_metric_metadata=guard_metric_metadata,
            group_guard_directional_derivative=guard_derivative_callback,
            required_guard_improvement_keys=tuple(
                f"{label}.observable_{component}_0p03"
                for label in m.REFINER_GROUP_LABELS
                for component in ("endpoint", "temporal")
            ),
            minimum_effective_scale=EXACT_GUARD_MIN_EFFECTIVE_SCALE,
        )
        if not pareto_gradient["common_descent_exists"]:
            update["reason"] = pareto_gradient["reason"]
        update["subgroup_directional_derivatives"] = dict(
            pareto_gradient["directional_derivatives"]
        )
        update["common_descent_exists"] = bool(
            pareto_gradient["common_descent_exists"]
        )
        multiobjective = report["multiobjective_diagnostics"]
        multiobjective["steps_evaluated"] += 1
        if pareto_gradient["common_descent_exists"]:
            multiobjective["common_descent_steps"] += 1
            multiobjective["constrained_common_descent_steps"] += 1
        else:
            if not pareto_gradient[
                "unconstrained_common_descent_exists"
            ]:
                multiobjective["pareto_stationary_steps"] += 1
            multiobjective["no_constrained_common_descent_steps"] += 1
        active_counts = Counter(
            multiobjective.get("active_constraint_counts", {})
        )
        active_counts.update(pareto_gradient["active_constraint_gradients"])
        multiobjective["active_constraint_counts"] = dict(
            sorted(active_counts.items())
        )
        multiobjective["last"] = dict(pareto_gradient)
        if update["optimizer_update_accepted"]:
            fixed_guard_current = dict(update["group_guard_after"])
            fixed_guard_best = {
                key: min(
                    fixed_guard_best[key],
                    fixed_guard_current[key],
                )
                for key in fixed_guard_best
            }
        update["group_guard_anchor"] = dict(fixed_guard_anchor)
        update["group_guard_reference_policy"] = (
            "immutable_fixed_metric_anchor"
        )
        update["group_guard_best_before"] = group_guard_best_before
        update["group_guard_best_after"] = dict(fixed_guard_best)
        report["group_guard_contract"]["best_so_far"] = dict(
            fixed_guard_best
        )
        report["group_guard_contract"]["current"] = dict(
            fixed_guard_current
        )
        record_update(report["optimizer_updates"], update)
        with (destination / "optimizer_updates.jsonl").open("a",encoding="utf8") as handle:
            handle.write(json.dumps({"step":step,**update},allow_nan=False) + "\n")
        consecutive_context_stalls = (
            consecutive_context_stalls + 1
            if fixed_bank_stalled(update)
            else 0
        )
        stopped_early = (
            consecutive_context_stalls >= train_cycle_length
            and step < args.steps
        )
        report["stopped_early"] = stopped_early
        report["termination_reason"] = update["reason"] if stopped_early else None
        if stopped_early and not logging:
            # backward() released the old graph. The rejected transaction has
            # restored the exact pre-update state; recompute on the SAME bank.
            amplitude_trace = {}
            r,p,t,_ = m._refiner_batch_objectives(
                model, batch, cfg, trace=amplitude_trace
            )
            gradient = m._refiner_gradient_diagnostics(model,r,p,cfg.product_refiner_clean_identity_weight)
            components = m._refiner_component_gradients(model,t,cfg)
        if logging or stopped_early:
            amplitude_summary = _decoder_amplitude_summary(
                amplitude_trace or {}, batch.get("seam")
            )
            amplitude_by_group = _decoder_amplitude_by_group(
                amplitude_trace or {}, batch["seam"], batch["group"]
            )
            report["output_amplitude_diagnostics"] = {
                "step": step,
                "aggregate": amplitude_summary,
                "by_group": amplitude_by_group,
            }
            save_diagnostic_state(destination,model,optimizer,report,step)
            row = {"stage":"observable_bridge_fit","step":step,"target_steps":args.steps,
                   "repair":float(repair.detach()),"clean":float(protection.detach()),
                   "terms":{k:float(v.detach()) for k,v in terms.items()},"gradient":gradient,
                   "component_gradients":components,"clip_norm_before":norm,
                   "pareto_gradient":pareto_gradient,
                   "decoder_output_amplitude":amplitude_summary,
                   "decoder_output_amplitude_by_group":amplitude_by_group,
                   "output_warmup":warmup,
                   "optimizer_update":update,
                   "fit_context_index":fit_context_index,
                   "fit_reservoir_transaction_index":fit_context_index,
                   "fit_reservoir_context_indices":list(
                       selected_context_indices
                   ),
                   "fit_reservoir_cycle_length":train_cycle_length,
                   "fit_transaction_materialization":"lazy_current_step_only",
                   "full_cycle_transaction":True,
                   "consecutive_context_stalls":consecutive_context_stalls,
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
            fit_contexts = evaluate_fit_contexts(model, banks, cfg)
            scope_diagnosis = learning_scope_diagnosis(
                fit_contexts,
                decisions,
                groups,
            )
            report.update(completed_steps=step,final=final,diagnostic_ready=(
                step == 400 and args.steps == 400 and args.windows == 8
                and all(d["scientific_acceptance"] for d in decisions.values())
                and all(g["passed"] for split in groups.values() for g in split.values())
                and fit_contexts["all_contexts_passed"]))
            report["group_decisions"] = groups
            report["fit_context_evaluation"] = fit_contexts
            report["learning_scope_diagnosis"] = scope_diagnosis
            breakdown = {split:failure_breakdown(metrics) for split,metrics in final.items()}
            report["failure_breakdown"] = breakdown
            # These decisions only judge train-window readiness, never publication.
            fit_context_summary = {
                key: value
                for key, value in fit_contexts.items()
                if key != "contexts"
            }
            report["history"].append({"step":step,
                "readiness":{s:{"passed":d["scientific_acceptance"],"reasons":d["reasons"],"observed":d["observed"]} for s,d in decisions.items()},
                "fit_context_readiness":fit_context_summary,
                "learning_scope_diagnosis":scope_diagnosis})
            m.save_json(report,destination / "diagnostic_report.json")
            m.save_json({"schema":SCHEMA,"fingerprint":report["fingerprint"],
                         "completed_steps":step,"diagnostic_ready":report["diagnostic_ready"],
                         "group_decisions":groups,"failure_breakdown":breakdown,
                         "fit_context_evaluation":fit_context_summary,
                         "learning_scope_diagnosis":scope_diagnosis,
                         "group_guard_contract":report["group_guard_contract"],
                         "multiobjective_contract":report["multiobjective_contract"],
                          "multiobjective_diagnostics":report["multiobjective_diagnostics"],
                          "output_amplitude_diagnostics":report.get(
                              "output_amplitude_diagnostics"
                          ),
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
            print(json.dumps({"stage":"bridge_context_cycle_stalled","completed_steps":step,
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
