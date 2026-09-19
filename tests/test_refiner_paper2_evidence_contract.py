import tempfile
import unittest
from pathlib import Path

import torch

from experiments.paper2 import run_paper2_eval
from experiments.paper2 import summarize_results
from training.refiner_paper2_matched_trial import JsonlMechanismRecorder


class Paper2EvidenceContractTest(unittest.TestCase):
    def test_candidate_identity_changes_with_rebuilt_candidate_state(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = JsonlMechanismRecorder(
                path=Path(directory) / "mechanism.jsonl",
                binding={"protocol_sha256": "protocol"},
                case_uids=("txn:1",),
                max_candidates_per_case_budget=2,
            )
            common = {
                "case_uid": "txn:1",
                "budget": 5,
                "iteration": 0,
                "theta_radians": 0.1,
                "backtrack": 0,
                "candidate_source": "g1f3_selected_physical_direction",
                "constraint_generation_depth": 0,
                "direction_sha256": "direction-a",
                "support_sha256": "support-a",
                "witness_sha256": "witness-a",
            }
            baseline = recorder.candidate_id(**common)
            for field, replacement in (
                ("constraint_generation_depth", 1),
                ("direction_sha256", "direction-b"),
                ("support_sha256", "support-b"),
                ("witness_sha256", "witness-b"),
            ):
                changed = {**common, field: replacement}
                self.assertNotEqual(baseline, recorder.candidate_id(**changed))

    def test_mechanism_rows_are_aggregated_once_per_trial(self):
        rows = []
        for normalized_e1, normalized_e2 in ((4.0, 1.0), (2.0, 3.0)):
            rows.append({
                "candidate_id": "candidate",
                "candidate_source": "source",
                "case_uid": "txn:1",
                "budget": 5,
                "theta_radians": 0.1,
                "constraint_generation_depth": 1,
                "direction_sha256": "direction",
                "support_sha256": "support",
                "witness_sha256": "witness",
                "radius_rms": 1.0e-4,
                "stable_witness": True,
                "normalized_E1": normalized_e1,
                "normalized_E2": normalized_e2,
            })
        aggregated = summarize_results._trial_mechanism_rows(rows)
        by_operator = {row["row_aggregation"]: row for row in aggregated}
        self.assertEqual(len(aggregated), 2)
        self.assertEqual(by_operator["median"]["normalized_E1"], 3.0)
        self.assertEqual(by_operator["median"]["normalized_E2"], 2.0)
        self.assertEqual(by_operator["worst"]["normalized_E1"], 4.0)
        self.assertEqual(by_operator["worst"]["normalized_E2"], 3.0)
        self.assertEqual(by_operator["median"]["metric_row_count"], 2)

    def test_manifest_must_match_bank_split_and_case_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "bank.pt"
            torch.save({
                "split": "validation",
                "split_manifest_content_sha256": "split-sha",
                "samples": [{
                    "case_uid": "txn:1",
                    "source_case_uid": "recording:1",
                    "transaction_id": "txn",
                }],
            }, bank_path)
            manifest = {
                "case_uids": ["txn:1"],
                "source_case_uids": ["recording:1"],
                "transaction_ids": ["txn"],
                "source_split_manifest_content_sha256": "split-sha",
            }
            run_paper2_eval._validate_manifest_against_bank(
                manifest, bank_path, "validation"
            )
            manifest["source_case_uids"] = ["recording:other"]
            with self.assertRaisesRegex(RuntimeError, "source-case mapping"):
                run_paper2_eval._validate_manifest_against_bank(
                    manifest, bank_path, "validation"
                )


if __name__ == "__main__":
    unittest.main()
