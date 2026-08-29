import pytest
import torch

from training import motion_models as m


def _group_terms():
    terms = {}

    for index, label in enumerate(m.REFINER_GROUP_LABELS):
        base = float(index + 1)

        terms[f"group_{label}_repair_total"] = torch.tensor(
            base,
            dtype=torch.float64,
        )
        terms[f"group_{label}_endpoint_continuity"] = torch.tensor(
            base + 0.1,
            dtype=torch.float64,
        )

        # Deliberately keep the gated temporal term at zero.
        # V13 must guard RAW temporal supervision instead.
        terms[f"group_{label}_temporal_supervision"] = torch.tensor(
            0.0,
            dtype=torch.float64,
        )
        terms[f"group_{label}_temporal_supervision_raw"] = torch.tensor(
            base + 0.2,
            dtype=torch.float64,
        )

    return terms


def test_v13_component_guard_has_twelve_keys():
    terms = _group_terms()

    guards = m._refiner_group_repair_losses(
        terms,
        require_all=True,
    )

    expected = set()

    for label in m.REFINER_GROUP_LABELS:
        expected.add(label)
        expected.add(f"{label}.endpoint")
        expected.add(f"{label}.temporal")

    assert set(guards) == expected
    assert len(guards) == 12


def test_v13_temporal_guard_uses_raw_not_priority_gated_term():
    terms = _group_terms()

    guards = m._refiner_group_repair_losses(
        terms,
        require_all=True,
    )

    for label in m.REFINER_GROUP_LABELS:
        assert float(
            guards[f"{label}.temporal"]
        ) == float(
            terms[f"group_{label}_temporal_supervision_raw"]
        )

        assert float(
            guards[f"{label}.temporal"]
        ) != float(
            terms[f"group_{label}_temporal_supervision"]
        )


def test_v13_component_guard_fails_closed_on_partial_group():
    terms = {
        "group_single_short_repair_total": torch.tensor(1.0),
        "group_single_short_endpoint_continuity": torch.tensor(0.5),
    }

    with pytest.raises(
        RuntimeError,
        match="incomplete Refiner subgroup objectives",
    ):
        m._refiner_group_repair_losses(
            terms,
            require_all=False,
        )
