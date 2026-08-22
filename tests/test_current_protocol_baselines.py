from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from baselines.current_protocol import beam_route, greedy_route
from model.current_protocol_router_baseline import MeanPoolMusicMotionRouter


class CurrentProtocolBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = {
            "paths": np.asarray(["event-a.npy", "event-b.npy"], dtype=object),
            "event_uids": np.asarray(["event-a", "event-b"], dtype=object),
            "source_uids": np.asarray(["source-a", "source-b"], dtype=object),
            "event_geometry_combined_quality": np.asarray([0.5, 0.8], dtype=np.float32),
        }
        self.slots = [
            {
                "router_architecture": "ctsr_weak_temporal_v1",
                "formal_candidate_contract": "ctsr_weak_scheduler_siblings_v1",
                "formal_candidate_event_uids": ["event-a", "event-b"],
                "formal_candidate_router_probabilities": [0.25, 0.75],
            },
            {
                "router_architecture": "ctsr_weak_temporal_v1",
                "formal_candidate_contract": "ctsr_weak_scheduler_siblings_v1",
                "formal_candidate_event_uids": ["event-a", "event-b"],
                "formal_candidate_router_probabilities": [0.8, 0.2],
            },
        ]
        self.candidates = [[0, 1], [0, 1]]

    def test_router_baseline_removes_temporal_order(self) -> None:
        model = MeanPoolMusicMotionRouter()
        self.assertEqual("ctsr_mean_pool_mlp_baseline_v1", model.architecture)
        self.assertEqual((2, 96), tuple(model.encode_music(np_to_tensor(np.zeros((2, 64, 12), np.float32))).shape))
        self.assertFalse(hasattr(model, "temporal_order_head"))

    def test_greedy_uses_same_ctsr_candidates(self) -> None:
        report = greedy_route(self.slots, self.candidates, self.db)
        self.assertEqual([1, 0], report["chosen_event_path"])
        self.assertFalse(report["formal_fallback"])

    @patch("baselines.current_protocol.manifold_edge_cost", return_value={"total": 0.0})
    def test_beam_is_explicit_baseline_not_fallback(self, _edge) -> None:
        report = beam_route(self.slots, self.candidates, self.db, beam_size=4)
        self.assertEqual([1, 0], report["chosen_event_path"])
        self.assertEqual("ctsr_finite_beam_v1", report["baseline"])
        self.assertFalse(report["formal_fallback"])

    def test_rejects_non_ctsr_slots(self) -> None:
        invalid = [dict(self.slots[0]), dict(self.slots[1])]
        invalid[0]["router_architecture"] = "historical"
        with self.assertRaisesRegex(RuntimeError, "CTSR-Weak"):
            greedy_route(invalid, self.candidates, self.db)

    def test_formal_pipeline_executes_current_protocol_baselines(self) -> None:
        pipeline = (Path(__file__).resolve().parents[1] / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("training/current_protocol_router_baseline.py", pipeline)
        self.assertIn("scripts/evaluate_current_protocol_baselines.py", pipeline)
        self.assertIn("CURRENT_PROTOCOL_BASELINES_ENABLE", pipeline)


def np_to_tensor(value: np.ndarray):
    import torch

    return torch.from_numpy(value)


if __name__ == "__main__":
    unittest.main()
