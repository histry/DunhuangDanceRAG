from collections import Counter
from types import SimpleNamespace

import torch

from training import motion_models as m
from training import bridge_feasibility as b
from training import refiner_bridge_diagnostics as d
from training.refiner_optimizer import REFINER_UPDATE_PROTOCOL


def test_v15_4_contract_preserves_scientific_objective():
    assert (
        d.SCHEMA
        == "refiner_observable_bridge_diagnostic_v15_4"
    )

    assert (
        d.FIT_PROTOCOL
        == "safe_start_context_reservoir_transaction_v2"
    )

    assert (
        d.CONTEXT_RESERVOIR_PROTOCOL
        == "all_probe_safe_farthest_order_rotating_c5_v1"
    )

    # Still C5 PER UPDATE.
    assert d.FIT_CONTEXT_COUNT == 5

    # V15.2/V15.3.1 scientific objective remains frozen.
    assert (
        m.REFINER_OBSERVABLE_OBJECTIVE_PROTOCOL
        == "scientific_feasibility_smooth_bottleneck_observable_v8"
    )

    assert (
        m.REFINER_BATCH_AGGREGATION_PROTOCOL
        == "group_balanced_scientific_mean_smooth_cvar_v2"
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_FRACTION
        == 0.25
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_MIX
        == 0.50
    )

    assert (
        m.REFINER_SCIENTIFIC_TAIL_TEMPERATURE
        == 1.0e-3
    )

    assert (
        b.DIRECT_OPTIMIZER_PROTOCOL
        == "per_case_scientific_feasibility_backtracking_v2"
    )

    assert (
        REFINER_UPDATE_PROTOCOL
        == "full_cycle_feasibility_guard_armijo_v7"
    )


def test_complete_reservoir_equals_exact_probe_safe_set():
    frames = 64

    for recipe_id, width in enumerate(
        (10, 28)
    ):
        width = min(
            width,
            frames - 8,
        )

        seen, probe = d._seen_and_probe_starts(
            frames,
            width,
            recipe_id,
        )

        expected = {
            start
            for start in range(
                3,
                frames - width - 1,
            )
            if (
                start != seen
                and abs(start - probe)
                > d.PROBE_START_GUARD_FRAMES
            )
        }

        reservoir = (
            d._all_probe_safe_context_starts(
                frames,
                width,
                recipe_id,
            )
        )

        assert len(reservoir) == len(
            set(reservoir)
        )

        assert set(reservoir) == expected

        assert seen not in reservoir
        assert probe not in reservoir

        assert all(
            abs(start - probe)
            > d.PROBE_START_GUARD_FRAMES
            for start in reservoir
        )


def test_legacy_c5_view_is_prefix_of_full_reservoir():
    frames = 64

    for recipe_id, width in enumerate(
        (10, 28)
    ):
        full = (
            d._all_probe_safe_context_starts(
                frames,
                width,
                recipe_id,
            )
        )

        c5 = d._context_fit_starts(
            frames,
            width,
            recipe_id,
        )

        assert len(c5) == 5
        assert c5 == full[:5]


def test_common_cycle_covers_every_group_start():
    frames = 64

    cycle = (
        d._context_reservoir_cycle_length(
            frames
        )
    )

    assert cycle >= 5

    for recipe_id, width in enumerate(
        (10, 28)
    ):
        width = min(
            width,
            frames - 8,
        )

        reservoir = (
            d._all_probe_safe_context_starts(
                frames,
                width,
                recipe_id,
            )
        )

        replayed = {
            d._split_start(
                f"fit_context_{index}",
                frames,
                width,
                recipe_id,
            )
            for index in range(cycle)
        }

        # Every legal TRAIN local context appears at least once.
        assert replayed == set(reservoir)


def _fake_bank(value):
    # fixed_fit_bank only requires equally sized role tensors
    # and builds the group tensor itself.
    return {
        "clean": torch.full(
            (2, 1),
            float(value),
        ),
    }


def _fake_reservoir_banks(count=7):
    banks = {}

    for role in (
        "single_recording",
        "cross_event",
    ):
        banks[
            (
                "seen",
                role,
            )
        ] = _fake_bank(-1)

    for index in range(count):
        for role in (
            "single_recording",
            "cross_event",
        ):
            banks[
                (
                    f"fit_context_{index}",
                    role,
                )
            ] = _fake_bank(index)

    return banks


def test_rotating_c5_schedule_is_balanced():
    banks = _fake_reservoir_banks(
        count=7
    )

    schedule = (
        d._reservoir_transaction_schedule(
            banks
        )
    )

    assert len(schedule) == 7

    for row in schedule:
        assert len(row) == 5
        assert len(set(row)) == 5

    counts = Counter(
        index
        for row in schedule
        for index in row
    )

    assert set(counts) == set(
        range(7)
    )

    # Every reservoir bank occurs exactly C5 times over one cycle.
    assert set(
        counts.values()
    ) == {5}


def test_transaction_keeps_c5_batch_size():
    banks = _fake_reservoir_banks(
        count=7
    )

    replay = (
        d.anchored_context_replay_banks(
            banks
        )
    )

    schedule = (
        d._reservoir_transaction_schedule(
            banks
        )
    )

    assert len(replay) == len(schedule) == 7

    # Fake anchor:
    #   2 single + 2 cross = 4
    #
    # C5 contexts:
    #   5 * 4 = 20
    #
    # total = 24.
    for batch in replay:
        assert len(batch["clean"]) == 24
        assert len(batch["group"]) == 24

        counts = torch.bincount(
            batch["group"].to(
                torch.long
            ),
            minlength=4,
        )

        # 6 cases/group:
        # 1 anchor + C5 context banks.
        assert counts.tolist() == [
            6,
            6,
            6,
            6,
        ]


def test_v15_4_formal_contract_remains_192_cases():
    cfg = SimpleNamespace(
        window_len=64,
    )

    contract = d.fit_bank_contract(
        8,
        cfg,
    )

    assert (
        contract["cases_per_update"]
        == 192
    )

    assert (
        contract["cases_per_role_width"]
        == 48
    )

    assert (
        contract["seen_anchor_cases_per_update"]
        == 32
    )

    assert (
        contract["context_cases_per_update"]
        == 160
    )

    assert (
        contract["context_banks_per_update"]
        == 5
    )

    assert (
        contract["probe_start_guard_frames"]
        == 6
    )

    assert (
        contract["probe_used_for_updates"]
        is False
    )

    assert (
        contract[
            "transaction_batch_fixed_within_step"
        ]
        is True
    )

    assert (
        contract[
            "reservoir_every_legal_start_seen_per_cycle"
        ]
        is True
    )


def test_probe_guard_never_leaks_across_full_reservoir_cycle():
    frames = 64

    cycle = (
        d._context_reservoir_cycle_length(
            frames
        )
    )

    for recipe_id, width in enumerate(
        (10, 28)
    ):
        width = min(
            width,
            frames - 8,
        )

        _, probe = (
            d._seen_and_probe_starts(
                frames,
                width,
                recipe_id,
            )
        )

        for context_index in range(
            cycle
        ):
            start = d._split_start(
                f"fit_context_{context_index}",
                frames,
                width,
                recipe_id,
            )

            assert start != probe

            assert (
                abs(start - probe)
                > d.PROBE_START_GUARD_FRAMES
            )



def test_exact_c5_reservoir_has_one_unique_transaction():
    banks = _fake_reservoir_banks(
        count=5
    )

    schedule = (
        d._reservoir_transaction_schedule(
            banks
        )
    )

    replay = (
        d.anchored_context_replay_banks(
            banks
        )
    )

    assert schedule == (
        (0, 1, 2, 3, 4),
    )

    assert len(replay) == 1

    # Fake anchor is 4 cases and each context bank is 4:
    # 4 + 5*4 = 24.
    assert len(
        replay[0]["clean"]
    ) == 24
