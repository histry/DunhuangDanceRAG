import unittest

from training.refiner_v15_15g1f4_parity import (
    _first_differences,
    _project_candidate_to_reference_schema,
)


class CanonicalParityProjectionTest(unittest.TestCase):
    def test_candidate_only_diagnostics_do_not_change_parity(self):
        reference = {
            "variants": {
                "k2": {
                    "accepted_steps": 1,
                    "history": [{"theta": 0.25, "accepted": True}],
                }
            }
        }
        candidate = {
            "variants": {
                "k2": {
                    "accepted_steps": 1,
                    "history": [{
                        "theta": 0.25,
                        "accepted": True,
                        "stage_timing_seconds": {"jet": 1.0},
                    }],
                    "function_call_counts": {"jet": 1},
                }
            },
            "new_top_level_diagnostic": True,
        }

        projected, additive_paths = _project_candidate_to_reference_schema(
            reference,
            candidate,
        )

        self.assertEqual(projected, reference)
        self.assertEqual(_first_differences(reference, projected), [])
        self.assertIn("$.new_top_level_diagnostic", additive_paths)
        self.assertIn(
            "$.variants.k2.history[0].stage_timing_seconds",
            additive_paths,
        )

    def test_changed_canonical_value_still_fails(self):
        reference = {"accepted_steps": 1, "history": [{"theta": 0.25}]}
        candidate = {"accepted_steps": 2, "history": [{"theta": 0.25}]}
        projected, _ = _project_candidate_to_reference_schema(
            reference,
            candidate,
        )
        differences = _first_differences(reference, projected)
        self.assertEqual(differences[0]["path"], "$.accepted_steps")

    def test_missing_canonical_field_still_fails(self):
        reference = {"accepted_steps": 1, "second_order_state": "ok"}
        candidate = {"accepted_steps": 1}
        projected, _ = _project_candidate_to_reference_schema(
            reference,
            candidate,
        )
        differences = _first_differences(reference, projected)
        self.assertEqual(differences[0]["path"], "$.second_order_state")

    def test_changed_list_structure_still_fails(self):
        reference = {"history": [{"theta": 0.25}]}
        candidate = {"history": [{"theta": 0.25}, {"theta": 0.125}]}
        projected, _ = _project_candidate_to_reference_schema(
            reference,
            candidate,
        )
        differences = _first_differences(reference, projected)
        self.assertEqual(differences[0]["path"], "$.history.length")


if __name__ == "__main__":
    unittest.main()
