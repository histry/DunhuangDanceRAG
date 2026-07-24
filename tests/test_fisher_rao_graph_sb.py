import itertools
import unittest

import numpy as np

from routing.fisher_rao import (
    categorical_kl,
    fisher_rao_distance,
    fisher_rao_midpoint,
    simplex_softmax,
)
from routing.graph_schrodinger import (
    multi_marginal_schrodinger,
    reference_markov_kernel,
    viterbi_path,
)


class FisherRaoSimplexTests(unittest.TestCase):
    def test_distance_has_expected_simplex_endpoints(self):
        same = fisher_rao_distance(
            np.asarray([0.25, 0.75]),
            np.asarray([0.25, 0.75]),
        )
        orthogonal = fisher_rao_distance(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        )
        self.assertAlmostEqual(float(same), 0.0, places=7)
        self.assertAlmostEqual(float(orthogonal), np.pi, places=7)

    def test_midpoint_stays_on_probability_simplex(self):
        midpoint = fisher_rao_midpoint(
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
        )
        np.testing.assert_allclose(midpoint, [0.5, 0.5], atol=1.0e-8)
        self.assertAlmostEqual(float(midpoint.sum()), 1.0)

    def test_masked_softmax_preserves_structural_zero(self):
        probability = simplex_softmax(
            np.asarray([1.0, 100.0, -1.0]),
            mask=np.asarray([True, False, True]),
        )
        self.assertEqual(float(probability[1]), 0.0)
        self.assertAlmostEqual(float(probability.sum()), 1.0)
        self.assertGreater(float(probability[0]), float(probability[2]))

    def test_kl_rejects_mass_outside_reference_support(self):
        with self.assertRaisesRegex(ValueError, "zero mass"):
            categorical_kl(
                np.asarray([0.5, 0.5]),
                np.asarray([1.0, 0.0]),
            )


class GraphSchrodingerTests(unittest.TestCase):
    def test_reference_kernel_keeps_hard_edges_zero(self):
        cost = np.asarray([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
        mask = np.asarray([[True, False], [True, True]])
        kernel = reference_markov_kernel(cost, feasible=mask, epsilon=0.5)
        self.assertEqual(float(kernel[0, 1]), 0.0)
        np.testing.assert_allclose(kernel.sum(axis=1), 1.0, atol=1.0e-10)

    def test_reference_kernel_rejects_dead_rows(self):
        with self.assertRaisesRegex(ValueError, "dead outgoing"):
            reference_markov_kernel(
                np.zeros((2, 2), dtype=np.float64),
                feasible=np.asarray([[True, False], [False, False]]),
            )

    def test_viterbi_matches_bruteforce_map_path(self):
        log_initial = np.log(np.asarray([0.6, 0.4]))
        transitions = (
            np.log(np.asarray([[0.9, 0.1], [0.2, 0.8]])),
            np.log(np.asarray([[0.4, 0.6], [0.7, 0.3]])),
        )
        potentials = (
            np.asarray([0.1, -0.1]),
            np.asarray([-0.2, 0.3]),
            np.asarray([0.0, 0.4]),
        )
        path, score = viterbi_path(log_initial, transitions, potentials)
        rows = []
        for candidate in itertools.product(range(2), repeat=3):
            value = (
                log_initial[candidate[0]]
                + potentials[0][candidate[0]]
                + transitions[0][candidate[0], candidate[1]]
                + potentials[1][candidate[1]]
                + transitions[1][candidate[1], candidate[2]]
                + potentials[2][candidate[2]]
            )
            rows.append((float(value), candidate))
        expected_score, expected_path = max(rows)
        self.assertEqual(path, expected_path)
        self.assertAlmostEqual(score, expected_score, places=10)

    def test_multi_marginal_ipf_matches_all_dense_targets(self):
        targets = (
            np.asarray([0.72, 0.28]),
            np.asarray([0.20, 0.50, 0.30]),
            np.asarray([0.15, 0.85]),
        )
        costs = (
            np.asarray([[0.0, 0.5, 1.1], [0.8, 0.2, 0.0]]),
            np.asarray([[0.1, 1.0], [0.3, 0.2], [1.2, 0.0]]),
        )
        result = multi_marginal_schrodinger(
            targets,
            costs,
            epsilon=0.4,
            maximum_iterations=400,
            tolerance=1.0e-9,
        )
        self.assertTrue(result.converged)
        self.assertLess(result.maximum_l1_residual, 1.0e-8)
        for fitted, target in zip(result.node_marginals, targets):
            np.testing.assert_allclose(fitted, target, atol=1.0e-8)
        self.assertEqual(len(result.map_path), len(targets))
        self.assertTrue(np.isfinite(result.map_log_probability))
        self.assertGreaterEqual(result.path_entropy, 0.0)

    def test_hard_transition_support_survives_ipf(self):
        target = np.asarray([0.35, 0.65])
        mask = np.eye(2, dtype=bool)
        result = multi_marginal_schrodinger(
            (target, target, target),
            (
                np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
            ),
            feasible_masks=(mask, mask),
            epsilon=0.5,
            maximum_iterations=100,
            tolerance=1.0e-9,
        )
        self.assertTrue(result.converged)
        for edge in result.edge_marginals:
            self.assertEqual(float(edge[0, 1]), 0.0)
            self.assertEqual(float(edge[1, 0]), 0.0)
        for posterior in result.posterior_transitions:
            np.testing.assert_allclose(posterior, np.eye(2), atol=1.0e-8)

    def test_single_slot_route_is_well_defined(self):
        target = np.asarray([0.1, 0.2, 0.7])
        result = multi_marginal_schrodinger(
            (target,),
            (),
            maximum_iterations=20,
            tolerance=1.0e-10,
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.node_marginals[0], target, atol=1.0e-9)
        self.assertEqual(result.map_path, (2,))
        self.assertEqual(result.edge_marginals, ())


if __name__ == "__main__":
    unittest.main()
