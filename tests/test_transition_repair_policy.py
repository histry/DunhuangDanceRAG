#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch

import numpy as np

from routing.transition_repair_policy import TransitionRiskPolicy, transition_decision


class TransitionRepairPolicyTests(unittest.TestCase):
    def setUp(self):
        self.previous = np.zeros((8, 151), dtype=np.float32)
        self.following = np.zeros((8, 151), dtype=np.float32)
        self.policy = TransitionRiskPolicy(
            intrinsic_previous_weight=0.10,
            intrinsic_following_weight=0.10,
            pairwise_weight=0.80,
            low_threshold=0.35,
            high_threshold=0.70,
            residual_inpainting_threshold=0.55,
        )

    @patch("routing.transition_repair_policy.transition_multiscale_risk")
    def test_hard_physics_reject_reroutes(self, mocked):
        mocked.return_value = {"score": 0.1, "hard_reject": True}
        decision = transition_decision(
            self.previous,
            self.following,
            following_intrinsic_prior=0.2,
            policy=self.policy,
        )
        self.assertEqual(decision["action"], "reroute")
        self.assertTrue(decision["hard_reject"])

    @patch("routing.transition_repair_policy.transition_multiscale_risk")
    def test_low_pairwise_risk_joins_directly(self, mocked):
        mocked.return_value = {"score": 0.05, "hard_reject": False}
        decision = transition_decision(
            self.previous,
            self.following,
            following_intrinsic_prior=0.1,
            policy=self.policy,
        )
        self.assertEqual(decision["action"], "direct_join")

    @patch("routing.transition_repair_policy.transition_multiscale_risk")
    def test_high_residual_uses_masked_inpainting(self, mocked):
        mocked.return_value = {"score": 4.0, "hard_reject": False}
        decision = transition_decision(
            self.previous,
            self.following,
            previous_intrinsic_prior=0.8,
            following_intrinsic_prior=0.8,
            aligned_residual_risk=0.7,
            policy=self.policy,
        )
        self.assertEqual(decision["action"], "contact_guided_masked_inpainting")


if __name__ == "__main__":
    unittest.main()
