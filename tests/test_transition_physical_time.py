import unittest

import numpy as np

from support.transition_quality import transition_risk


def linear_motion(fps: float, seconds: float = 1.0) -> np.ndarray:
    frames = int(round(float(fps) * float(seconds))) + 1
    time = np.arange(frames, dtype=np.float32) / float(fps)
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 4] = 0.25 * time
    motion[:, 5] = 1.0
    identity6d = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:] = np.tile(identity6d, 24)
    motion[:, :4] = time[:, None] * 0.5
    return motion


class TransitionPhysicalTimeTests(unittest.TestCase):
    def test_contact_switch_rate_is_fps_invariant(self):
        values = []
        for fps in (30.0, 60.0):
            motion = linear_motion(fps)
            split = len(motion) // 3
            report = transition_risk(
                motion[:split],
                motion[split : 2 * split],
                motion[2 * split :],
                fps=fps,
            )
            values.append(report["contact_switch"])
        self.assertAlmostEqual(values[0], values[1], places=5)

    def test_invalid_fps_is_rejected_by_transition_sampler(self):
        from training.transition_diffusion import sample_transition_diffusion

        frame = linear_motion(30.0)[:1]
        with self.assertRaisesRegex(ValueError, "fps"):
            sample_transition_diffusion(
                None,
                frame[0],
                frame[0],
                3,
                np.zeros(12, dtype=np.float32),
                fps=0.0,
            )


if __name__ == "__main__":
    unittest.main()
