import unittest

import numpy as np

from motion_geometry.smpl24 import NUM_JOINTS, ROT6D_START
from training.motion_models import MotionGenerationConfig
from training.refiner_action_feasibility import (
    ACTION_DIM,
    STATUS_BUDGET_EXHAUSTED,
    ActionFeasibilityCase,
    FeasibilitySolverConfig,
    decode_geometry_action,
    evaluate_action_candidate,
    normalized_raw_action_norm,
    solve_action_feasibility,
)


def _identity_motion(frames=32):
    motion = np.zeros((frames, 151), dtype=np.float32)
    identity = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    motion[:, ROT6D_START:] = np.tile(identity, NUM_JOINTS)
    return motion


def _case():
    reference = _identity_motion()
    # Keep the reference observable but deliberately discontinuous at the
    # repair seam; no hidden clean target is supplied to the case.
    reference[8:16, 4] = 0.20
    seam = np.zeros((len(reference), 1), dtype=np.float32)
    seam[8:16] = 1.0
    cfg = MotionGenerationConfig(
        device="cpu",
        refiner_core_strength=1.0,
        refiner_transition_strength=1.0,
        product_refiner_residual_smoothing_passes=0,
        product_refiner_residual_taper_frames=0,
    )
    return ActionFeasibilityCase(
        case_id="synthetic-dev-0001",
        role="cross_event",
        width=8,
        position_stratum="seen",
        split="dev",
        reference=reference,
        seam=seam,
        joint_mask=np.broadcast_to(seam, (len(reference), NUM_JOINTS)).copy(),
        root_mask=seam.copy(),
        contact_mask=np.zeros((len(reference), 4), dtype=np.float32),
        condition=np.zeros((len(reference), 4), dtype=np.float32),
        cfg=cfg,
        source_uid="source-a",
        recording_uid="recording-a",
        left_source_uid="source-a",
        right_source_uid="source-b",
        left_recording_uid="recording-a",
        right_recording_uid="recording-b",
    )


class RefinerActionFeasibilityTests(unittest.TestCase):
    def test_zero_action_is_decoder_identity(self):
        case = _case()
        action = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
        decoded, detail = decode_geometry_action(case, action)
        np.testing.assert_allclose(decoded, case.reference, rtol=0.0, atol=1.0e-6)
        self.assertEqual(detail["contact_residual_max"], 0.0)

    def test_normalized_action_reports_raw_not_fk_metric(self):
        case = _case()
        action = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
        action[8, 0] = case.cfg.product_refiner_root_cap_m
        action[8, 3] = case.cfg.product_refiner_rotation_cap_rad
        self.assertGreater(normalized_raw_action_norm(action, case.cfg), 0.0)

    def test_support_outside_is_not_edited_by_decoder(self):
        case = _case()
        action = np.ones((case.frames, ACTION_DIM), dtype=np.float32)
        decoded, detail = decode_geometry_action(case, action)
        self.assertEqual(detail["support_outside_edit_max"], 0.0)
        np.testing.assert_allclose(decoded[:8], case.reference[:8], rtol=0.0, atol=1.0e-6)

    def test_root_and_rotation_caps_are_active(self):
        case = _case()
        action = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
        action[10, :3] = 10.0
        action[10, 3:] = 10.0
        _, detail = decode_geometry_action(case, action)
        self.assertGreater(detail["cap_saturation"]["root_fraction"], 0.0)
        self.assertGreater(detail["cap_saturation"]["rotation_fraction"], 0.0)

    def test_contact_residual_is_fixed_zero(self):
        case = _case()
        action = np.zeros((case.frames, ACTION_DIM), dtype=np.float32)
        action[10, :3] = 0.01
        decoded, _ = decode_geometry_action(case, action)
        np.testing.assert_allclose(decoded[:, :4], case.reference[:, :4], rtol=0.0, atol=1.0e-7)

    def test_evaluator_exposes_joint_conjunction_and_separate_audits(self):
        result = evaluate_action_candidate(
            _case(), np.zeros((_case().frames, ACTION_DIM), dtype=np.float32)
        )
        for key in (
            "endpoint_pass",
            "temporal_pass",
            "jerk_pass",
            "physical_pass",
            "fidelity_pass",
            "finite_pass",
            "joint_pass",
            "failure_reasons",
        ):
            self.assertIn(key, result)
        self.assertIn("physical_stage", result)
        self.assertIn("fixed_reference_support", result)
        self.assertFalse(result["hidden_clean_used"])

    def test_budget_exhaustion_rolls_back(self):
        case = _case()
        result = solve_action_feasibility(
            case,
            solver_config=FeasibilitySolverConfig(max_iterations=0),
        )
        self.assertIn(result.status, {STATUS_BUDGET_EXHAUSTED, "VERIFIED_FEASIBLE"})
        if result.status == STATUS_BUDGET_EXHAUSTED:
            self.assertTrue(result.rollback)
            np.testing.assert_array_equal(result.returned_motion, case.reference)


if __name__ == "__main__":
    unittest.main()
