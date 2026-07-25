#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.calibrate_transition_risk import calibrate


class TransitionRiskCalibrationTests(unittest.TestCase):
    def test_quantile_thresholds_remain_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "events.npz"
            motions = np.zeros((8, 6, 151), dtype=np.float32)
            for index in range(len(motions)):
                motions[index, :, 4] = index * 0.01
            labels = np.asarray(
                ["pose_hold", "calm_meditative", "lyrical_flow", "footwork_flow",
                 "turning_climax", "aerial_curve", "percussive_accent", "instrument_phrase"],
                dtype=object,
            )
            np.savez_compressed(
                db,
                motions=motions,
                aesd_event_semantics=labels,
                source_uids=np.asarray([f"s{i % 3}" for i in range(8)], dtype=object),
            )

            def fake_risk(previous, bridge, following, fps=30.0):
                score = float(abs(previous[0, 4] - following[0, 4]))
                return {"score": score, "hard_reject": False}

            with patch(
                "evaluation.calibrate_transition_risk.transition_multiscale_risk",
                side_effect=fake_risk,
            ):
                report = calibrate(
                    db,
                    root / "calibration.json",
                    samples_per_category=3,
                    seed=9,
                )
            thresholds = report["global_thresholds"]
            self.assertLess(thresholds["low"], thresholds["high"])
            for category in report["categories"].values():
                self.assertLess(category["low_threshold"], category["high_threshold"])


if __name__ == "__main__":
    unittest.main()
