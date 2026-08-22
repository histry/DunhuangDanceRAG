from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model.music_motion_router import load_router_checkpoint
from model.temporal_music_motion_router import TemporalMusicMotionRouter
from scheduling.hierarchical_graph_scheduler import (
    build_hierarchy_features,
    build_slot_query,
    hierarchical_node_scores,
)
from scheduling.retrieval import (
    LOCAL_ACTION_LABELS,
    aggregate_action_compatibility,
    precompute_music_routing,
)
from scheduling.temporal_router_contract import (
    FORMAL_PLANNER_CONTRACT,
    assert_formal_planner_scientific_contract,
    assert_formal_router_scientific_contract,
    resample_feature_sequence,
    scientific_supervision_contract,
)
from training.weak_semantic_ot import (
    DEFAULT_COST_WEIGHTS,
    sparse_sinkhorn_teacher,
    weighted_control_cost,
)


class CTSRWeakRouterTests(unittest.TestCase):
    def test_sequence_resampling_preserves_endpoints_and_time_axis(self) -> None:
        source = np.zeros((5, 12), dtype=np.float32)
        source[:, 0] = np.linspace(0.0, 1.0, 5)
        result = resample_feature_sequence(source, 17)
        self.assertEqual(result.shape, (17, 12))
        self.assertAlmostEqual(float(result[0, 0]), 0.0)
        self.assertAlmostEqual(float(result[-1, 0]), 1.0)
        self.assertGreater(float(np.diff(result[:, 0]).min()), 0.0)

    def test_teacher_does_not_infer_body_region_from_music(self) -> None:
        self.assertTrue(np.array_equal(DEFAULT_COST_WEIGHTS[1:4], np.zeros(3)))
        music = np.full((1, 12), 0.5, dtype=np.float32)
        motion_a = np.full((1, 12), 0.5, dtype=np.float32)
        motion_b = motion_a.copy()
        motion_b[:, 1:4] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        costs = weighted_control_cost(music, np.concatenate([motion_a, motion_b]))
        self.assertAlmostEqual(float(costs[0, 0]), float(costs[0, 1]), places=7)

    def test_sparse_ot_is_soft_normalized_and_declared_non_ground_truth(self) -> None:
        rng = np.random.default_rng(7)
        cost = rng.random((6, 24), dtype=np.float32)
        probability, report = sparse_sinkhorn_teacher(
            cost, [f"recording_{index % 4}" for index in range(24)], top_k=8
        )
        self.assertEqual(probability.shape, cost.shape)
        np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1.0e-5)
        self.assertGreater(np.count_nonzero(probability), len(probability))
        self.assertEqual(report["supervision_source"], "semantic_ot_teacher")
        self.assertFalse(report["is_ground_truth"])
        self.assertFalse(report["paired_audio_motion"])

    def test_sparse_ot_realistic_union_support_converges_past_legacy_cap(self) -> None:
        phrase_count, event_count = 10, 328
        rng = np.random.default_rng(263)
        base_cost = rng.random(event_count)
        cost = np.clip(
            base_cost[None, :]
            + rng.normal(0.0, 0.07, (phrase_count, event_count)),
            0.0,
            1.0,
        )
        groups = (
            ["flying_apsaras"] * 143
            + ["lotus_steps"] * 83
            + ["revelation_meditation"] * 75
            + ["pipa_behind_back"] * 27
        )

        _, legacy_report = sparse_sinkhorn_teacher(
            cost,
            groups,
            top_k=64,
            epsilon=0.12,
            max_iterations=200,
            tolerance=1.0e-5,
        )
        self.assertFalse(legacy_report["converged"])

        teacher, report = sparse_sinkhorn_teacher(
            cost,
            groups,
            top_k=64,
            epsilon=0.12,
            max_iterations=5000,
            tolerance=1.0e-5,
        )
        self.assertTrue(report["converged"])
        self.assertGreater(report["iterations"], 200)
        self.assertLessEqual(report["row_marginal_error"], report["tolerance"])
        self.assertLessEqual(report["column_marginal_error"], report["tolerance"])
        np.testing.assert_allclose(teacher.sum(axis=1), 1.0, atol=1.0e-5)

    def test_event_probabilities_remain_multi_label_at_action_level(self) -> None:
        events = np.asarray([[0.75, 0.25]], dtype=np.float32)
        items = [
            {"local_action_scores": {"pose_hold": 0.6, "floorwork": 0.4}},
            {"local_action_scores": {"locomotion": 0.5, "transition": 0.5}},
        ]
        action = aggregate_action_compatibility(events, items)
        self.assertEqual(action.shape, (1, len(LOCAL_ACTION_LABELS)))
        self.assertAlmostEqual(float(action.sum()), 1.0, places=6)
        self.assertGreater(float(action[0, LOCAL_ACTION_LABELS.index("pose_hold")]), 0.0)
        self.assertGreater(float(action[0, LOCAL_ACTION_LABELS.index("floorwork")]), 0.0)
        self.assertGreater(float(action[0, LOCAL_ACTION_LABELS.index("locomotion")]), 0.0)

    def test_formal_hierarchy_uses_nominal_action_probabilities(self) -> None:
        arrays = {
            "natural_duration": np.asarray([40.0, 45.0], dtype=np.float32),
            "style_score": np.asarray([0.8, 0.8], dtype=np.float32),
            "quality_score": np.asarray([0.9, 0.9], dtype=np.float32),
            "safety_score": np.asarray([0.9, 0.9], dtype=np.float32),
            "motion_desc": np.asarray(
                [[0.2, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
        }
        items = [
            {"local_action_scores": {"pose_hold": 0.8, "floorwork": 0.2}},
            {"local_action_scores": {"locomotion": 1.0}},
        ]
        hierarchy = build_hierarchy_features(arrays, items)
        self.assertEqual(
            str(hierarchy["hierarchy_semantic_contract"][0]),
            "weak_motion_local_action_multilabel_v1",
        )
        self.assertEqual(
            hierarchy["action_probs"].shape,
            (2, len(LOCAL_ACTION_LABELS)),
        )
        action_probability = np.zeros(len(LOCAL_ACTION_LABELS), dtype=np.float32)
        action_probability[LOCAL_ACTION_LABELS.index("pose_hold")] = 0.8
        action_probability[LOCAL_ACTION_LABELS.index("floorwork")] = 0.2
        phrase = SimpleNamespace(
            music_event="neutral_flow",
            energy=0.3,
            onset=0.2,
            beat_density=0.2,
            tension=0.1,
            calmness=0.7,
            boundary_accent_strength=0.1,
        )
        query = build_slot_query(
            phrase,
            target_natural=40.0,
            desired_activity=0.3,
            action_compatibility=action_probability,
        )
        score, components = hierarchical_node_scores(hierarchy, query)
        self.assertEqual(query["semantic_proxy"].shape, (15,))
        self.assertEqual(
            query["semantic_contract"],
            "semantic_ot_teacher_x_weak_motion_local_action",
        )
        self.assertGreater(
            float(components["hierarchy_coarse_score"][0]),
            float(components["hierarchy_coarse_score"][1]),
        )
        self.assertTrue(np.isfinite(score).all())

    def test_temporal_checkpoint_load_and_uncertainty_contract(self) -> None:
        model = TemporalMusicMotionRouter(
            hidden_dim=32,
            latent_dim=16,
            transformer_layers=1,
            transformer_heads=4,
        )
        config = {
            "architecture": "ctsr_weak_temporal_v1",
            "music_dim": 12,
            "motion_dim": 12,
            "hidden_dim": 32,
            "latent_dim": 16,
            "dropout": 0.1,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "sequence_frames": 16,
            "init_temperature": 0.12,
            "inference_temperature": 0.12,
            "feature_mean": [0.5] * 12,
            "feature_std": [0.25] * 12,
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "router.pt"
            torch.save(
                {
                    "architecture": "ctsr_weak_temporal_v1",
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "scientific_contract": scientific_supervision_contract(),
                },
                checkpoint_path,
            )
            loaded = load_router_checkpoint(checkpoint_path, device="cpu")
            queries = [np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)]
            sequences = np.stack(
                [np.zeros((16, 12), dtype=np.float32), np.ones((16, 12), dtype=np.float32)]
            )
            routing = precompute_music_routing(
                loaded,
                queries,
                np.random.default_rng(3).random((10, 12), dtype=np.float32),
                torch.device("cpu"),
                phrase_sequences=sequences,
            )
        self.assertEqual(routing["similarity"].shape, (2, 10))
        np.testing.assert_allclose(routing["probabilities"].sum(axis=1), 1.0, atol=1.0e-6)
        self.assertEqual(routing["architecture"], "ctsr_weak_temporal_v1")
        self.assertEqual(routing["supervision_source"], "semantic_ot_teacher")
        self.assertTrue(np.all(routing["entropy"] >= 0.0))
        self.assertTrue(np.all(routing["ood"] >= 0.0))

    def test_formal_contract_rejects_legacy_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "scientific_contract"):
            assert_formal_router_scientific_contract({"version": "legacy"})

    def test_formal_planner_contract_rejects_event_head_supervision(self) -> None:
        valid = {"formal_planner_contract": dict(FORMAL_PLANNER_CONTRACT)}
        self.assertFalse(
            assert_formal_planner_scientific_contract(valid)["categorical_event_head"]
        )
        invalid = {"formal_planner_contract": dict(FORMAL_PLANNER_CONTRACT)}
        invalid["formal_planner_contract"]["categorical_event_head"] = True
        with self.assertRaisesRegex(RuntimeError, "categorical_event_head"):
            assert_formal_planner_scientific_contract(invalid)


if __name__ == "__main__":
    unittest.main()
