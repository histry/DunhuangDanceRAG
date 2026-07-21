import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from contracts.gravity import GravityThresholds, identity6d_np
from evaluation.audit_gravity import audit_one
from evaluation.audit_heading import _resolve_motion_ref_path


def moving_identity_motion(fps: float) -> np.ndarray:
    frames = int(fps) + 1
    time = np.arange(frames, dtype=np.float32) / float(fps)
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 4] = time
    motion[:, 5] = 0.95
    motion[:, 7:] = identity6d_np((frames, 24)).reshape(frames, -1)
    return motion


class AuditFpsAndPathTests(unittest.TestCase):
    def test_gravity_audit_uses_runtime_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.npy"
            np.save(path, moving_identity_motion(60.0))
            result = audit_one(path, GravityThresholds(), fps=60.0)
            self.assertEqual(result["fps"], 60.0)
            self.assertAlmostEqual(result["foot_speed_xz_p95_mps"], 1.0, places=4)

    def test_motion_ref_is_resolved_relative_to_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "reports"
            asset_dir = report_dir / "assets"
            asset_dir.mkdir(parents=True)
            report = report_dir / "report.json"
            report.write_text(json.dumps({}), encoding="utf-8")
            motion = root / "motion.npy"
            np.save(motion, np.zeros((1, 151), dtype=np.float32))
            reference = asset_dir / "reference.npy"
            np.save(reference, np.zeros((1, 151), dtype=np.float32))
            resolved = _resolve_motion_ref_path(
                report, motion, Path("assets") / "reference.npy"
            )
            self.assertEqual(resolved, reference.resolve())


if __name__ == "__main__":
    unittest.main()
