import unittest

import numpy as np

from grounding import mixed_curvature as mixed
from grounding.manifold_ops import (
    bures_distance_sq_np,
    bures_distance_sq_torch,
)


@unittest.skipIf(mixed.torch is None, "PyTorch is not installed")
class MixedGroundingTorchTests(unittest.TestCase):
    def setUp(self):
        self.config = mixed.MixedGrounderConfig(
            clap_dim=16,
            temporal_dim=12,
            motion_geometry_dim=20,
            bodypart_count=5,
            bodypart_feature_dim=8,
            gaussian_dim=4,
            control_dim=4,
            num_sources=3,
            hidden_dim=32,
            lorentz_dim=6,
            sphere_dim=12,
            dropout=0.0,
        )

    def test_factor_shapes_constraints_and_backward(self):
        torch = mixed.torch
        model = mixed.MixedCurvatureGrounder(self.config)
        batch = 6
        audio = model.encode_audio(
            torch.randn(batch, 16), torch.randn(batch, 32, 12)
        )
        motion = model.encode_motion(
            torch.randn(batch, 20), torch.randn(batch, 5, 8)
        )
        self.assertEqual(tuple(audio["lorentz"].shape), (batch, 7))
        self.assertEqual(tuple(audio["sphere"].shape), (batch, 12))
        self.assertEqual(tuple(audio["gaussian_mean"].shape), (batch, 5, 4))
        self.assertEqual(
            tuple(audio["gaussian_covariance"].shape), (batch, 5, 4, 4)
        )
        sphere_norm = torch.linalg.vector_norm(audio["sphere"], dim=-1)
        self.assertTrue(torch.allclose(sphere_norm, torch.ones_like(sphere_norm)))
        eigenvalues = torch.linalg.eigvalsh(audio["gaussian_covariance"])
        self.assertTrue(bool((eigenvalues > 0.0).all()))
        logits, distance, variance = model.pairwise_logits(audio, motion)
        self.assertEqual(tuple(logits.shape), (batch, batch))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertTrue(bool((distance >= 0.0).all()))
        self.assertTrue(bool((variance > 0.0).all()))
        loss = logits.square().mean() + distance.mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_complete_batch_loss_is_finite(self):
        torch = mixed.torch
        model = mixed.MixedCurvatureGrounder(self.config)
        batch = 6
        identity = torch.eye(4).reshape(1, 1, 4, 4).expand(batch, 5, 4, 4)
        tensors = (
            torch.randn(batch, 16),
            torch.randn(batch, 32, 12),
            torch.randn(batch, 20),
            torch.randn(batch, 5, 8),
            torch.randn(batch, 5, 4),
            identity.clone(),
            torch.rand(batch, 4),
            torch.rand(batch).clamp_min(0.1),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            torch.tensor([0, 1, 2, 0, 1, 2]),
            torch.arange(batch),
        )
        loss, pieces = mixed._batch_loss(
            model,
            tensors,
            hierarchy_weight=0.2,
            gaussian_anchor_weight=0.25,
            control_weight=0.1,
            uncertainty_weight=0.05,
            source_weight=0.05,
            metric_balance_weight=0.01,
            hierarchy_margin=1.25,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("metric_balance", pieces)
        loss.backward()

    def test_invalid_hidden_group_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            mixed.MixedGrounderConfig(
                clap_dim=4,
                temporal_dim=12,
                motion_geometry_dim=4,
                bodypart_count=5,
                bodypart_feature_dim=8,
                gaussian_dim=3,
                control_dim=2,
                num_sources=2,
                hidden_dim=30,
            )

    def test_newton_schulz_bures_matches_numpy_reference(self):
        torch = mixed.torch
        left = np.asarray(
            [[[1.3, 0.2], [0.2, 0.9]], [[0.8, -0.1], [-0.1, 1.4]]],
            dtype=np.float32,
        )
        right = np.asarray(
            [[[0.7, 0.05], [0.05, 1.2]], [[1.1, 0.15], [0.15, 0.75]]],
            dtype=np.float32,
        )
        expected = bures_distance_sq_np(left, right)
        actual = bures_distance_sq_torch(
            torch.from_numpy(left), torch.from_numpy(right)
        ).detach().numpy()
        np.testing.assert_allclose(actual, expected, atol=2.0e-4, rtol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
