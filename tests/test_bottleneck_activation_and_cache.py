import numpy as np

from routing.bidirectional_reachability import BackwardReachabilityModel
from routing.hierarchical_constraint_model import ConstraintBudgetConfig


def _db():
    return {
        "paths": np.asarray(["a.npy", "b.npy", "c.npy", "d.npy"], dtype=object),
        "dance_keys": np.asarray(["d"] * 4, dtype=object),
        "source_uids": np.asarray(["s0", "s0", "s1", "s2"], dtype=object),
        "event_families": np.asarray(["f0", "f0", "f1", "f2"], dtype=object),
        "event_uids": np.asarray(["e0", "e1", "e2", "e3"], dtype=object),
        "anatomy_valid": np.asarray([True] * 4),
        "anatomy_quality": np.asarray([0.9] * 4, dtype=np.float32),
        "event_heading_valid": np.asarray([True] * 4),
    }


def test_predicted_bottleneck_activation_adds_future_candidate(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    config = ConstraintBudgetConfig.from_environment(total_slots=3)
    model = BackwardReachabilityModel.build([[0], [1], [2]], _db(), constraint_config=config)
    report = model.activate_candidates(slot=1, event_ids=[3], reason="unit_test")
    assert report["triggered"] is True
    assert 3 in model.layers[1]
    assert model.runtime_summary()["candidate_activations"] == 1


def test_compressed_cache_key_ignores_small_dual_float_noise(monkeypatch):
    monkeypatch.setenv("BR_HPR_ENABLE", "1")
    monkeypatch.setenv("BR_HPR_STATE_REACHABILITY_USAGE_QUANTIZATION", "0.10")
    config = ConstraintBudgetConfig.from_environment(total_slots=3)
    model = BackwardReachabilityModel.build([[0], [1], [2]], _db(), constraint_config=config)
    key_a = model._cache_key(
        slot=1,
        previous_event_id=0,
        selected_event_ids=(0,),
        usage=(0.01, 0.02, 0.03, 0.04, 0.0, 0.01),
        duals=(0.001,) * 6,
        recovery_budget_used=0.01,
        depth=0,
    )
    key_b = model._cache_key(
        slot=1,
        previous_event_id=0,
        selected_event_ids=(0,),
        usage=(0.02, 0.01, 0.04, 0.03, 0.0, 0.02),
        duals=(999.0,) * 6,
        recovery_budget_used=0.02,
        depth=0,
    )
    assert key_a == key_b
