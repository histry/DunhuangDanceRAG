from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training import motion_models as m
from training import refiner_role_conditioned_support_projection_experiment as rcsp
from training import refiner_role_phase_anatomy_low_rank_tangent_adaptation as rpa


def test_schema_and_parent_contract():
    assert rpa.SCHEMA == (
        "refiner_role_phase_anatomy_low_rank_tangent_adaptation_experiment_v1"
    )
    assert rpa.IMPLEMENTATION_PARENT_COMMIT == (
        "7a360736db0a184f137446f7da20f23b04fb97ad"
    )


def test_anatomy_partition_is_exact():
    partition = rpa.validate_anatomy_partition()
    assert tuple(partition["body_joints"]) == rpa.EXPECTED_BODY_JOINTS
    assert tuple(partition["extremity_joints"]) == rpa.EXPECTED_EXTREMITY_JOINTS
    assert partition["root_dimensions"] == 3
    assert partition["body_dimensions"] == 48
    assert partition["extremity_dimensions"] == 24
    assert partition["total_geometry_dimensions"] == 75


def test_fixed_ranks_and_parameter_count():
    adapter = rpa.RolePhaseAnatomyLowRankTangentAdapter(256)
    assert rpa.ROOT_RANK == 2
    assert rpa.BODY_RANK == 8
    assert rpa.EXTREMITY_RANK == 4
    assert adapter.parameter_count == 4692
    assert rpa.expected_parameter_count(256) == 4692


def test_zero_start_initialization_contract():
    adapter = rpa.RolePhaseAnatomyLowRankTangentAdapter(256)
    init = adapter.validate_initialization()
    assert init["root_up_zero"]
    assert init["body_up_zero"]
    assert init["extremity_up_zero"]
    assert init["conditioner_final_weight_zero"]
    assert init["conditioner_final_bias_one"]


def test_zero_start_forward_is_exact_zero_but_gradient_exists():
    adapter = rpa.RolePhaseAnatomyLowRankTangentAdapter(256)
    hidden = torch.linspace(-1.0, 1.0, 2 * 256 * 12).reshape(2, 256, 12)
    role_id = torch.tensor([0, 1], dtype=torch.long)
    phase = torch.linspace(0.05, 0.95, 12).view(1, 12, 1).repeat(2, 1, 1)
    duration = torch.tensor([10.0 / 30.0, 28.0 / 30.0]).view(2, 1, 1).expand(2, 12, 1)

    output, _ = adapter(hidden, role_id, phase, duration)
    assert torch.equal(output, torch.zeros_like(output))

    target = torch.linspace(-0.2, 0.3, output.numel()).reshape_as(output)
    loss = (output - target).square().mean()
    loss.backward()

    grads = [
        adapter.root_up.weight.grad,
        adapter.body_up.weight.grad,
        adapter.extremity_up.weight.grad,
    ]
    assert all(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert sum(float(grad.norm()) for grad in grads) > rpa.GRADIENT_NUMERICAL_TOL


def test_conditioner_uses_role_phase_duration_only():
    adapter = rpa.RolePhaseAnatomyLowRankTangentAdapter(256)
    assert adapter.conditioner[0].in_features == 4
    assert adapter.conditioner[0].out_features == 32
    assert adapter.conditioner[2].out_features == 14


@pytest.mark.parametrize("value, expected", [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])
def test_endpoint_envelope_exact_points(value, expected):
    phase = torch.tensor([[[value]]], dtype=torch.float64)
    envelope = rpa.endpoint_envelope(phase)
    assert float(envelope) == pytest.approx(expected, abs=1e-14)


def test_endpoint_envelope_is_bounded():
    phase = torch.linspace(0.0, 1.0, 1001).view(1, -1, 1)
    envelope = rpa.endpoint_envelope(phase)
    assert float(envelope.min()) >= 0.0
    assert float(envelope.max()) <= 1.0 + 1e-6


def test_no_width_argument_in_adapter_forward():
    names = rpa.RolePhaseAnatomyLowRankTangentAdapter.forward.__code__.co_varnames
    assert "width" not in names
    assert "width_id" not in names


def _identity6d():
    return torch.tensor(m.identity6d_np(), dtype=torch.float32).repeat(24)


def test_authoritative_phase_duration_matches_fk_duration_channel():
    cfg = m.MotionGenerationConfig(device="cpu")
    motion = torch.zeros(2, cfg.window_len, m.EDGE_DIM)
    motion[..., 7:] = _identity6d()
    seam = torch.zeros(2, cfg.window_len, 1)
    seam[0, 40:50] = 1.0
    seam[1, 40:68] = 1.0

    phase, duration, details = rpa.authoritative_phase_duration(
        motion, seam, cfg.fps
    )
    assert phase.shape == (2, cfg.window_len, 1)
    assert duration.shape == phase.shape
    assert details["duration_fk_channel_parity_max_abs_error"] == 0.0
    assert float(duration[0, 0, 0]) == pytest.approx(10.0 / cfg.fps)
    assert float(duration[1, 0, 0]) == pytest.approx(28.0 / cfg.fps)


def test_binary_support_projection_cannot_escape():
    batch, frames = 2, 8
    joint = torch.zeros(batch, frames, 24)
    root = torch.zeros(batch, frames, 1)
    joint[:, 2:6, :] = 0.18
    root[:, 2:6] = 0.18
    support = rcsp.binary_geometry_support(joint, root)

    raw = torch.ones(batch, frames, 75)
    projected = raw * support
    assert float((projected * (1.0 - support)).abs().max()) == 0.0


def test_scatter_partition_covers_75d_without_overlap():
    root = torch.ones(1, 2, 3)
    body = torch.full((1, 2, 48), 2.0)
    extremity = torch.full((1, 2, 24), 3.0)
    out = rpa.RolePhaseAnatomyLowRankTangentAdapter._scatter_joint_blocks(
        root, body, extremity
    )
    assert out.shape == (1, 2, 75)
    assert torch.equal(out[..., :3], root)

    joint = out[..., 3:].reshape(1, 2, 24, 3)
    assert torch.all(joint[..., list(rpa.BODY_JOINTS), :] == 2.0)
    assert torch.all(joint[..., list(rpa.EXTREMITY_JOINTS), :] == 3.0)


def test_decision_advance_requires_all_A_to_I():
    rows = []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                for index in range(8):
                    rcsp_pass = False
                    rpa_pass = role == "single_recording" and index == 0
                    rcsp_def = 1.0
                    rpa_def = 0.8 if (role == "cross_event" and width == 28) else 0.9
                    rows.append(
                        {
                            "split": split,
                            "role": role,
                            "width": width,
                            "BASE": {
                                "temporal_gate_pass": False,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": 1.1,
                                "endpoint_scientific_deficit": 1.0,
                            },
                            "RCSP": {
                                "temporal_gate_pass": rcsp_pass,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": rcsp_def,
                                "endpoint_scientific_deficit": 1.0,
                            },
                            "RPA_LRTA": {
                                "temporal_gate_pass": rpa_pass,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": rpa_def,
                                "endpoint_scientific_deficit": 0.9,
                            },
                            "temporal_newly_rescued_vs_rcsp": rpa_pass,
                            "temporal_regression_vs_rcsp": False,
                            "endpoint_newly_rescued_vs_rcsp": False,
                            "endpoint_regression_vs_rcsp": False,
                            "physical_regression_vs_rcsp": False,
                            "geometry_regression_vs_rcsp": False,
                            "clean_identity_regression_vs_rcsp": False,
                            "support_regression": False,
                            "contact_regression": False,
                            "rpa_residual_norm": 0.01,
                            "applied_action_norm_rcsp": 0.02,
                            "applied_action_norm_rpa": 0.02,
                            "temporal_alignment_rcsp": {
                                "cosine_to_negative_gradient": 0.1
                            },
                            "temporal_alignment_rpa": {
                                "cosine_to_negative_gradient": 0.2
                            },
                        }
                    )
    summaries = rpa.make_summaries(rows)
    decision = rpa.adjudicate(rows, summaries)
    assert all(decision["conditions"].values())
    assert decision["result"] == "RPA_LRTA_CANDIDATE_ADVANCE_REVIEW"
    assert decision["pilot_allowed"] is False


def test_safety_regression_blocks_advance():
    rows = []
    for split in ("seen", "new_position"):
        for role in ("single_recording", "cross_event"):
            for width in (10, 28):
                for index in range(8):
                    rows.append(
                        {
                            "split": split,
                            "role": role,
                            "width": width,
                            "BASE": {
                                "temporal_gate_pass": False,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": 1.2,
                                "endpoint_scientific_deficit": 1.0,
                            },
                            "RCSP": {
                                "temporal_gate_pass": False,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": 1.0,
                                "endpoint_scientific_deficit": 1.0,
                            },
                            "RPA_LRTA": {
                                "temporal_gate_pass": index == 0,
                                "endpoint_gate_pass": True,
                                "temporal_scientific_deficit": 0.8,
                                "endpoint_scientific_deficit": 0.9,
                            },
                            "temporal_newly_rescued_vs_rcsp": index == 0,
                            "temporal_regression_vs_rcsp": False,
                            "endpoint_newly_rescued_vs_rcsp": False,
                            "endpoint_regression_vs_rcsp": False,
                            "physical_regression_vs_rcsp": (
                                split == "seen"
                                and role == "cross_event"
                                and width == 28
                                and index == 0
                            ),
                            "geometry_regression_vs_rcsp": False,
                            "clean_identity_regression_vs_rcsp": False,
                            "support_regression": False,
                            "contact_regression": False,
                            "rpa_residual_norm": 0.01,
                            "applied_action_norm_rcsp": 0.02,
                            "applied_action_norm_rpa": 0.02,
                            "temporal_alignment_rcsp": {
                                "cosine_to_negative_gradient": 0.1
                            },
                            "temporal_alignment_rpa": {
                                "cosine_to_negative_gradient": 0.2
                            },
                        }
                    )
    summaries = rpa.make_summaries(rows)
    decision = rpa.adjudicate(rows, summaries)
    assert decision["conditions"]["G_no_safety_regression"] is False
    assert decision["result"] != "RPA_LRTA_CANDIDATE_ADVANCE_REVIEW"
