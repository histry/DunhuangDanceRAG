from __future__ import annotations

import unittest

import numpy as np

from training.motion_models import contact_ramp_weights_np


class IkTransactionTests(unittest.TestCase):
    def test_contact_ramp_softens_both_sides_of_support_island(self):
        contacts = np.zeros((12, 2), dtype=bool)
        contacts[2:10, 0] = True
        weights = contact_ramp_weights_np(
            contacts,
            fps=30.0,
            ramp_seconds=4.0 / 30.0,
        )
        self.assertEqual(weights.shape, contacts.shape)
        self.assertGreater(weights[5, 0], weights[2, 0])
        self.assertGreater(weights[6, 0], weights[9, 0])
        self.assertTrue(np.all(weights[~contacts] == 0.0))
        self.assertTrue(np.all((weights >= 0.0) & (weights <= 1.0)))


if __name__ == "__main__":
    unittest.main()
