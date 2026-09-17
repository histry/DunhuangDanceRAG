import unittest

from training import refiner_v15_15g_fixed_budget_correction as correction


class Paper2ExecutionPolicyTest(unittest.TestCase):
    def policy(
        self,
        *,
        intent=correction.PAPER2_EXECUTION_MECHANISM_PREREGISTERED,
        role="train_calibration",
        case_uid="txn:171",
        preregistered=("txn:171",),
        activated=False,
        calibration_forced=False,
    ):
        return correction._paper2_candidate_execution_policy(
            execution_intent=intent,
            evaluation_role=role,
            case_uid=case_uid,
            preregistered_case_uids=preregistered,
            runtime_activation_authorized=activated,
            calibration_forced_evaluation=calibration_forced,
        )

    def test_preregistered_mechanism_executes_without_selection_authority(self):
        policy = self.policy()
        self.assertTrue(policy["paper2_preregistered_evaluation"])
        self.assertTrue(policy["candidate_execution_required"])
        self.assertFalse(policy["runtime_selection_eligible"])
        self.assertFalse(policy["runtime_activation_overridden"])

    def test_runtime_activation_preserves_selection_authority(self):
        policy = self.policy(activated=True)
        self.assertTrue(policy["candidate_execution_required"])
        self.assertTrue(policy["runtime_selection_eligible"])

    def test_nonpreregistered_inactive_case_remains_skipped(self):
        policy = self.policy(case_uid="txn:other")
        self.assertFalse(policy["paper2_preregistered_evaluation"])
        self.assertFalse(policy["candidate_execution_required"])
        self.assertFalse(policy["runtime_selection_eligible"])

    def test_nonmechanism_roles_never_gain_preregistered_execution(self):
        for role in (
            "development_validation",
            "final_held_out",
        ):
            with self.subTest(role=role):
                policy = self.policy(role=role)
                self.assertFalse(policy["paper2_preregistered_evaluation"])
                self.assertFalse(policy["candidate_execution_required"])

    def test_diagnostic_candidate_cannot_reach_runtime_selector(self):
        policy = self.policy()
        with self.assertRaises(RuntimeError):
            correction._assert_paper2_runtime_selection_isolated(
                policy, "geodesic_joint_sqp_k5"
            )
        correction._assert_paper2_runtime_selection_isolated(
            policy, "identity"
        )


if __name__ == "__main__":
    unittest.main()
