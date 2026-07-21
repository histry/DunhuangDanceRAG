import unittest

import numpy as np

from contracts.gravity import identity6d_np
from events.intrinsic_geometry import _database_fps, _geometry_descriptor


def motion_at_rate(fps: float) -> np.ndarray:
    frames = int(round(float(fps))) + 1
    time = np.arange(frames, dtype=np.float32) / float(fps)
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 5] = 0.93
    motion[:, 7:] = identity6d_np((frames, 24)).reshape(frames, -1)
    motion[:, :4] = 0.5 * time[:, None]
    return motion


class IntrinsicGeometryMultirateTests(unittest.TestCase):
    def test_database_rate_contract_rejects_mixed_rates(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one positive rate"):
            _database_fps(
                {"canonical_fps": np.asarray([30.0, 60.0], dtype=np.float32)}
            )

    def test_contact_change_descriptor_uses_per_second_units(self):
        rows = []
        for fps in (30.0, 60.0):
            item = _geometry_descriptor(
                motion_at_rate(fps),
                posture="standing",
                family="test",
                stage_role="middle",
                fps=fps,
                edge_frames=int(round(0.2 * fps)),
            )
            descriptor = np.asarray(item["descriptor"])
            # 12 root + 48 body-part + 14 trajectory + 4 contact means.
            rows.append(descriptor[78:82])
        np.testing.assert_allclose(rows[0], rows[1], atol=1.0e-5, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
