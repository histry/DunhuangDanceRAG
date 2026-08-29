import torch

from training import bridge_feasibility as b
from training import motion_models as m


def test_v15_foundation_direct_protocol():
    assert (
        b.DIRECT_OPTIMIZER_PROTOCOL
        == "per_case_scientific_feasibility_backtracking_v2"
    )


def test_v15_scientific_gap_is_weaker_than_legacy_ten_percent_margin():
    baseline = torch.tensor(
        [1.0],
        dtype=torch.float64,
    )

    # 4% improvement:
    #   scientific 3% target -> PASS
    #   historical 10% margin -> still deficient
    proposed = torch.tensor(
        [0.96],
        dtype=torch.float64,
    )

    scientific_loss, scientific_gap = (
        m._smooth_observable_margin(
            proposed,
            baseline,
            0.03,
        )
    )

    legacy_loss, legacy_gap = (
        m._smooth_observable_margin(
            proposed,
            baseline,
            0.10,
        )
    )

    assert float(scientific_gap) == 0.0
    assert float(scientific_loss) == 0.0

    assert float(legacy_gap) > 0.0
    assert float(legacy_loss) > 0.0
