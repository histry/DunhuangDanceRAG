import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from routing.diversity import diversity_assessment, select_safe_diverse_proposal
from routing.dynamic_route import (
    DynamicBeamState,
    candidate_subset,
    clear_route_prior,
    prune_states,
    register_route_prior,
    route_prior_cost,
)


class RoutingDiversityTests(unittest.TestCase):
    def setUp(self):
        clear_route_prior()
        self.db = {
            "event_uids": np.asarray(["e0", "e1", "e2", "e3"], dtype=object),
            "source_uids": np.asarray(["s0", "s0", "s1", "s2"], dtype=object),
            "event_families": np.asarray(["f0", "f1", "f1", "f2"], dtype=object),
            "dance_keys": np.asarray(["d0", "d1", "d1", "d2"], dtype=object),
        }

    def test_exact_event_cooldown_is_hard(self):
        result = diversity_assessment(self.db, 0, [2, 0])
        self.assertFalse(result["hard_valid"])
        self.assertIn("event_uid_cooldown", result["hard_reasons"])

    def test_primary_is_a_soft_prior_not_an_absolute_commit(self):
        rows = [
            (SimpleNamespace(event_id=2, safe=True, risk_score=0.50, rank=0), {}),
            (SimpleNamespace(event_id=3, safe=True, risk_score=0.05, rank=1), {}),
        ]
        with patch.dict(os.environ, {"V46_54_PRIMARY_EVENT_BONUS": "0.18"}, clear=False):
            selected, _extra, decision = select_safe_diverse_proposal(
                rows, db=self.db, selected_event_ids=[], primary_event_id=2
            )
        self.assertEqual(selected.event_id, 3)
        self.assertEqual(decision, "reselected_heading_physics_diverse")

    def test_primary_soft_prior_breaks_a_near_tie(self):
        rows = [
            (SimpleNamespace(event_id=2, safe=True, risk_score=0.20, rank=0), {}),
            (SimpleNamespace(event_id=3, safe=True, risk_score=0.10, rank=1), {}),
        ]
        with patch.dict(
            os.environ,
            {
                "V46_54_PRIMARY_EVENT_BONUS": "0.18",
                "V46_54_CANDIDATE_RANK_WEIGHT": "0",
            },
            clear=False,
        ):
            selected, _extra, decision = select_safe_diverse_proposal(
                rows, db=self.db, selected_event_ids=[], primary_event_id=2
            )
        self.assertEqual(selected.event_id, 2)
        self.assertEqual(decision, "selected_primary_soft_prior")

    def test_source_run_reselects(self):
        rows = [
            (SimpleNamespace(event_id=1, safe=True, risk_score=0.01, rank=0), {}),
            (SimpleNamespace(event_id=2, safe=True, risk_score=0.2, rank=1), {}),
        ]
        with patch.dict(os.environ, {"V46_54_MAX_SOURCE_RUN": "1"}, clear=False):
            selected, _extra, decision = select_safe_diverse_proposal(
                rows, db=self.db, selected_event_ids=[0], primary_event_id=1
            )
        self.assertEqual(selected.event_id, 2)
        self.assertEqual(decision, "reselected_heading_physics_diverse")

    def test_global_source_and_family_shares_are_hard_after_warmup(self):
        with patch.dict(
            os.environ,
            {
                "V46_54_MIN_SHARE_HISTORY": "3",
                "V46_54_MAX_SOURCE_SHARE": "0.40",
                "V46_54_MAX_FAMILY_SHARE": "0.50",
                "V46_54_EVENT_COOLDOWN_SLOTS": "1",
                "V46_54_MAX_SOURCE_RUN": "3",
            },
            clear=False,
        ):
            source = diversity_assessment(self.db, 1, [0, 2, 3])
            family = diversity_assessment(self.db, 2, [1, 2, 3])
        self.assertFalse(source["hard_valid"])
        self.assertIn("source_share", source["hard_reasons"])
        self.assertFalse(family["hard_valid"])
        self.assertIn("family_share", family["hard_reasons"])

    def test_cooldown_is_not_silently_relaxed_when_pool_is_exhausted(self):
        rows = [
            (SimpleNamespace(event_id=0, safe=True, risk_score=0.01, rank=0), {}),
        ]
        with self.assertRaisesRegex(RuntimeError, "exhausted candidates"):
            select_safe_diverse_proposal(
                rows,
                db=self.db,
                selected_event_ids=[0],
                primary_event_id=0,
            )

    def test_physical_safety_is_never_relaxed(self):
        rows = [
            (SimpleNamespace(event_id=2, safe=False, risk_score=0.01, rank=0), {}),
            (SimpleNamespace(event_id=3, safe=False, risk_score=0.02, rank=1), {}),
        ]
        with self.assertRaisesRegex(RuntimeError, "physically_safe=0"):
            select_safe_diverse_proposal(
                rows,
                db=self.db,
                selected_event_ids=[],
                primary_event_id=2,
            )

    def test_source_covered_subset_survives_rank_collapse(self):
        candidates = [0, 1, 2, 3]
        subset = candidate_subset(
            candidates,
            self.db,
            limit=3,
            minimum_per_source=1,
            primary_event_id=0,
        )
        sources = {self.db["source_uids"][event_id] for event_id in subset}
        self.assertIn("s0", sources)
        self.assertIn("s1", sources)
        self.assertIn("s2", sources)

    def test_graph_sb_transition_posterior_is_exposed_as_soft_cost(self):
        register_route_prior(
            [[0, 2], [1, 3]],
            node_marginals=[np.asarray([0.8, 0.2]), np.asarray([0.3, 0.7])],
            transition_marginals=[np.asarray([[0.2, 0.8], [0.9, 0.1]])],
            chosen_path=[0, 3],
            source="fisher_rao_graph_sb",
        )
        preferred, _ = route_prior_cost(
            1,
            3,
            previous_event_id=0,
            fallback_rank=1,
            candidate_count=2,
        )
        weak, _ = route_prior_cost(
            1,
            1,
            previous_event_id=0,
            fallback_rank=0,
            candidate_count=2,
        )
        self.assertLess(preferred, weak)

    def test_beam_pruning_keeps_one_branch_per_source(self):
        states = [
            DynamicBeamState(
                motion=np.zeros((0, 151), dtype=np.float32),
                selected_event_ids=(0,),
                score=0.0,
            ),
            DynamicBeamState(
                motion=np.zeros((0, 151), dtype=np.float32),
                selected_event_ids=(1,),
                score=0.1,
            ),
            DynamicBeamState(
                motion=np.zeros((0, 151), dtype=np.float32),
                selected_event_ids=(2,),
                score=0.2,
            ),
        ]
        kept = prune_states(states, self.db, width=2)
        kept_sources = {
            self.db["source_uids"][state.selected_event_ids[-1]] for state in kept
        }
        self.assertEqual(len(kept_sources), 2)


if __name__ == "__main__":
    unittest.main()
