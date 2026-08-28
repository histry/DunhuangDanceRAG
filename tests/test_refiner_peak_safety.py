"""Mean seam improvement must not hide a worse instantaneous joint peak."""
import json
from unittest import mock

import numpy as np
import pytest

from contracts.physical_quality import physical_metric_specs, _allowed_after_stage
from training import bridge_feasibility as f
from training import motion_models as m
from tests.test_bridge_feasibility import bank
from tests.test_duration_inbetween import motion

torch = m.torch


def device_or_skip(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")


def integrate_jerk(jerk, fps=30):
    zero = jerk.new_zeros((len(jerk), 3, 24, 3))
    return torch.cat([zero, jerk], 1).cumsum(1).cumsum(1).cumsum(1) / fps**3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_lower_mean_with_higher_peak_has_positive_safety_loss_and_gradient(device):
    device_or_skip(device)
    cfg = m.MotionGenerationConfig(device=device)
    jerk = torch.zeros((2, 117, 24, 3), dtype=torch.float64, device=device)
    jerk[..., 0] = torch.where(torch.arange(117, device=device) % 2 != 0, -20., 20.)[None, :, None]
    reference = integrate_jerk(jerk).requires_grad_(True)
    candidate_jerk = .5 * jerk
    candidate_jerk[0, 60, 23, 0] = 500.
    candidate_jerk[0, 61, 23, 0] = -500.
    # Another safe case cannot hide the first case's spike in a batch mean.
    candidate = integrate_jerk(candidate_jerk).detach().requires_grad_(True)
    seam = torch.zeros((2, 120, 1), device=device)
    seam[:, 48:76] = 1
    before = m.boundary_metrics_torch(reference, seam, cfg.fps)
    after = m.boundary_metrics_torch(candidate, seam, cfg.fps)
    assert torch.all(after["seam_jerk_mps3"] < before["seam_jerk_mps3"])
    assert torch.all(after["temporal_energy"] < before["temporal_energy"])
    loss, terms = m._repair_jerk_safety_loss_torch(candidate, reference, cfg)
    assert loss.shape == (2,) and loss[0] > 0 and loss[1] == 0
    assert terms["repair_jerk_max_excess"][0] > 0
    loss.sum().backward()
    assert reference.grad is None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad[0].abs().sum() > 0 and candidate.grad[1].abs().sum() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_safe_edits_and_identity_have_zero_safety_gradient(device):
    device_or_skip(device)
    cfg = m.MotionGenerationConfig(device=device)
    jerk = torch.full((2, 117, 24, 3), 20., device=device, dtype=torch.float64)
    reference = integrate_jerk(jerk)
    prediction = (reference * 1.001).detach().requires_grad_(True)
    loss, _ = m._repair_jerk_safety_loss_torch(prediction, reference, cfg)
    assert torch.count_nonzero(loss) == 0
    loss.sum().backward()
    assert torch.count_nonzero(prediction.grad) == 0
    # An already-over-limit input is allowed unchanged, but gets no new budget.
    hot = reference * 10000
    unchanged, _ = m._repair_jerk_safety_loss_torch(hot, hot, cfg)
    worse, _ = m._repair_jerk_safety_loss_torch(hot * 1.001, hot, cfg)
    assert torch.count_nonzero(unchanged) == 0 and torch.all(worse > 0)


@pytest.mark.parametrize("fraction", [0., .5, .999, 1.2])
def test_all_five_loss_budgets_match_stage_registry_including_absolute_ceiling(fraction):
    cfg = m.MotionGenerationConfig(device="cpu")
    specs = {s.key: s for s in physical_metric_specs(m.PhysicalQualityLimits.from_environment(),
                                                   m.StageAcceptancePolicy.from_environment())}
    labels = {"p95":"joint_jerk_mps3_p95", "max":"joint_jerk_mps3_max",
              "window_p95":"joint_jerk_window_p95_max_mps3",
              "extremity_p95":"extremity_jerk_mps3_p95",
              "extremity_window_p95":"extremity_jerk_window_p95_max_mps3"}
    before, after, expected = {}, {}, {}
    for label, key in labels.items():
        spec = specs[key]
        value = fraction * spec.absolute_limit
        allowed = _allowed_after_stage(value, spec.absolute_limit, spec.stage_ratio, spec.stage_margin)
        eps = max(1e-8, abs(value)*1e-6, abs(spec.absolute_limit)*1e-9)
        before[label] = torch.tensor([value, value], dtype=torch.float64)
        after[label] = torch.tensor([allowed, allowed + eps + .5], dtype=torch.float64)
        expected[label] = torch.tensor([0., .5 / max(1., allowed - value)], dtype=torch.float64)
    x = torch.zeros((2, 8, 24, 3), dtype=torch.float64)
    with mock.patch.object(m, "_clean_jerk_statistics_torch", side_effect=[after, before]):
        loss, terms = m._repair_jerk_safety_loss_torch(x, x, cfg)
    for label in labels:
        torch.testing.assert_close(terms[f"repair_jerk_{label}_excess"], expected[label], atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(loss, sum(expected.values()), atol=1e-12, rtol=1e-10)


def test_repair_branch_uses_tail_loss_not_only_clean_identity_branch():
    b, cfg = bank()
    x = b["bad"]
    base, _ = m._observable_refiner_objective(x, x, b["seam"], cfg, reduction="none")
    with mock.patch.object(m, "_repair_jerk_safety_loss_torch", return_value=(torch.tensor([2.]), {})):
        result, terms = m._observable_refiner_objective(x, x, b["seam"], cfg, reduction="none")
    torch.testing.assert_close(result - base, torch.tensor([2.], dtype=result.dtype))
    assert terms["jerk_safety_excess"].item() == 2.


def test_uploaded_three_peak_regressions_receive_positive_loss_without_changing_budgets():
    # Metric fixtures from the 82a7655 server report, not a claim to replay its
    # unavailable motion arrays. Other statistics are held unchanged here.
    reference = torch.tensor([804.3784790039062, 767.6563110351562, 1177.4442138671875], dtype=torch.float64)
    candidate = torch.tensor([886.147216796875, 1006.0115966796875, 1353.0810546875], dtype=torch.float64)
    allowed = reference*1.02+40
    cfg = m.MotionGenerationConfig(device="cpu")
    labels = ("p95", "max", "window_p95", "extremity_p95", "extremity_window_p95")
    before = {label:torch.zeros_like(reference) for label in labels}
    before["max"] = reference
    after = {**before, "max":candidate}
    x = torch.zeros((3, 8, 24, 3), dtype=torch.float64)
    with mock.patch.object(m.StageAcceptancePolicy, "from_environment", return_value=m.StageAcceptancePolicy()), \
         mock.patch.object(m.PhysicalQualityLimits, "from_environment", return_value=m.PhysicalQualityLimits()), \
         mock.patch.object(m, "_clean_jerk_statistics_torch", side_effect=[after, before]):
        loss, terms = m._repair_jerk_safety_loss_torch(x, x, cfg)
    expected = (candidate-allowed-reference*1e-6)/(allowed-reference)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(terms["repair_jerk_max_excess"], expected)
    assert torch.all(loss > 0)


def test_direct_optimizer_keeps_last_safe_candidate_instead_of_lower_unsafe_loss(tmp_path):
    b, cfg = bank()
    retained = []
    def objective(prediction, reference, seam, cfg, **kwargs):
        # Isolate the line-search rule from how hard the real repair task is.
        loss = (prediction[:, 20:24, 4] - reference[:, 20:24, 4] - .002).square().mean(1)
        return loss, {key:loss*0 for key in ("endpoint_continuity", "temporal_supervision",
                                            "support_excess", "jerk_safety_excess")}
    def gate(reference, candidate, *args, **kwargs):
        # Identity and the first safe improvement are admissible; subsequent
        # novel proposals represent lower-loss but physically unsafe edits.
        ok = np.array_equal(reference, candidate)
        if not ok and not retained:
            retained.append(candidate.copy())
        ok = ok or np.array_equal(candidate, retained[0])
        return {"accepted":ok, "reasons":[] if ok else ["joint_jerk_max_regressed"]}
    with mock.patch.object(m, "_fixed_support_stage_gate", side_effect=gate), \
         mock.patch.object(m, "_observable_refiner_objective", side_effect=objective):
        prediction, trace = f.direct_optimize(b, cfg, 3, label="safe-retention", log_path=tmp_path/"log.jsonl")
    assert retained
    np.testing.assert_array_equal(prediction.cpu().numpy()[0], retained[0])
    assert trace[0]["safety_accepted"] and trace[0]["unsafe_trial_count"] > 0
    assert trace[0]["unsafe_trial_reasons"]["joint_jerk_max_regressed"] > 0
    logs = [json.loads(line) for line in (tmp_path/"log.jsonl").read_text().splitlines()]
    assert any(r["safety_rejected_trials"] > 0 for r in logs)
    assert not list(tmp_path.glob("*.pt"))


def test_safety_cache_stays_input_relative_and_final_check_is_fresh():
    b, cfg = bank()
    original = b["bad"].numpy()[0]
    checker = f._DirectSafetyChecker(b["bad"], cfg)
    changed = original.copy(); changed[24, 4] += .001
    with mock.patch.object(m, "_fixed_support_stage_gate", return_value={"accepted":True, "reasons":[]}) as audit:
        for fresh in (False, False, True):
            assert checker.check(0, changed, fresh=fresh)[0]
        assert audit.call_count == 2 and checker.cache_hits == 1
        for call in audit.call_args_list:
            np.testing.assert_array_equal(call.args[0], original)


def test_rejected_physical_candidate_skips_expensive_fidelity_work():
    b, cfg = bank()
    checker = f._DirectSafetyChecker(b["bad"], cfg)
    with mock.patch.object(m, "_fixed_support_stage_gate", return_value={
            "accepted":False, "reasons":["joint_jerk_max_regressed"]}), \
         mock.patch.object(m, "_observable_reference_fidelity", side_effect=AssertionError("unnecessary SVD")):
        accepted, reasons = checker.check(0, b["bad"].numpy()[0])
    assert not accepted and reasons == ["joint_jerk_max_regressed"]


def test_gpu_tail_prefilter_does_not_send_known_violations_to_cpu_auditor(tmp_path):
    b, cfg = bank()
    def objective(prediction, reference, seam, cfg, **kwargs):
        loss = (prediction[:, 20:24, 4] - reference[:, 20:24, 4] - .002).square().mean(1)
        violation = (prediction != reference).flatten(1).any(1).to(loss.dtype)
        return loss, {"endpoint_continuity":loss*0, "temporal_supervision":loss*0,
                      "support_excess":loss*0, "jerk_safety_excess":violation}
    def audit(reference, candidate, *args, **kwargs):
        np.testing.assert_array_equal(candidate, reference)
        return {"accepted":True, "reasons":[]}
    with mock.patch.object(m, "_fixed_support_stage_gate", side_effect=audit), \
         mock.patch.object(m, "_observable_refiner_objective", side_effect=objective):
        prediction, trace = f.direct_optimize(b, cfg, 2, label="prefilter", log_path=tmp_path/"log.jsonl")
    torch.testing.assert_close(prediction, b["bad"], rtol=0, atol=0)
    assert trace[0]["retained_no_edit"]
    assert trace[0]["unsafe_trial_reasons"]["gpu_jerk_budget_exceeded"] > 0


def test_peak_location_matches_physical_metric_and_names_the_four_frame_stencil():
    cfg = m.MotionGenerationConfig(device="cpu")
    reference = motion(48)
    candidate = reference.copy(); candidate[24, 4] += .02
    seam = np.zeros((48, 1), np.float32); seam[16:32] = 1
    gate = m._observable_boundary_audit(candidate, reference, seam, cfg)
    peak = gate["jerk_peak_diagnostic"]["after"]
    physical = m._safe_validation_audit(candidate, cfg, role="test", support_policy="source_observation")
    assert peak["value_mps3"] == pytest.approx(physical["joint_jerk_mps3_max"], abs=1e-8)
    assert peak["stencil_start_frame"] <= 24 <= peak["stencil_end_frame"]
    assert peak["stencil_end_frame"] - peak["stencil_start_frame"] == 3
    assert peak["touches_seam_core"]
    assert peak["joint_name"] == m.JOINT_NAMES[peak["joint_index"]]


def test_old_foundation_schema_cannot_authorize_network_training(tmp_path):
    path = tmp_path/"old.json"
    path.write_text(json.dumps({"schema":"bridge_foundation_feasibility_v2", "published":False,
                                "fingerprint":{}}), encoding="utf8")
    with pytest.raises(RuntimeError, match="protocol/config/code mismatch"):
        f.check_foundation_report(path, {}, m.MotionGenerationConfig())
