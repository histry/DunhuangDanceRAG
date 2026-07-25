#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class SemanticOTSoftDistillationTests(unittest.TestCase):
    def test_bidirectional_soft_loss_is_finite(self):
        from grounding.semantic_ot_grounder import _soft_transport_bidirectional_loss

        logits = torch.tensor(
            [[3.0, 1.0, 0.0, -1.0], [0.0, -1.0, 2.5, 1.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.65, 0.25, 0.05, 0.05], [0.05, 0.05, 0.65, 0.25]],
            dtype=torch.float32,
        )
        confidence = torch.tensor([0.8, 0.7], dtype=torch.float32)
        loss, forward, reverse = _soft_transport_bidirectional_loss(
            logits, target, confidence
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(forward))
        self.assertTrue(torch.isfinite(reverse))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cross_phrase_target_is_row_normalized(self):
        from grounding.semantic_ot_grounder import _semantic_cross_phrase_target

        music = torch.tensor(
            [[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]],
            dtype=torch.float32,
        )
        action = torch.tensor(
            [[0.65, 0.15, 0.1, 0.1], [0.1, 0.1, 0.65, 0.15], [0.2, 0.2, 0.2, 0.4]],
            dtype=torch.float32,
        )
        target = _semantic_cross_phrase_target(music, action, 0.25)
        self.assertEqual(tuple(target.shape), (2, 3))
        self.assertTrue(torch.allclose(target.sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertGreater(float(target[0, 0]), float(target[0, 1]))
        self.assertGreater(float(target[1, 1]), float(target[1, 0]))


if __name__ == "__main__":
    unittest.main()
