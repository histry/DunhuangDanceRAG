import unittest

import numpy as np

from motion_geometry.product_manifold import (
    PRODUCT_STATE_DIM,
    TANGENT_DIM,
    masked_retract_np,
    parallel_transport_np,
    product_distance_np,
    product_exp_np,
    product_log_np,
    riemannian_trust_region_refine_np,
    torch as product_torch,
)
from motion_geometry.rotations import matrix_to_rot6d_np, so3_exp_np
from motion_geometry.smpl24 import NUM_JOINTS, ROT6D_END, ROT6D_START


def identity_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frames, NUM_JOINTS, 3, 3)
    )
    motion[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(rotations).reshape(
        frames, -1
    )
    return motion


class ProductManifoldTests(unittest.TestCase):
    def test_v45_v46_state_dimensions(self) -> None:
        self.assertEqual(TANGENT_DIM, 75)
        self.assertEqual(PRODUCT_STATE_DIM, 79)

    def test_log_exp_round_trip(self) -> None:
        rng = np.random.default_rng(13)
        reference = identity_motion(7)
        reference[:, 4:7] = rng.normal(0.0, 0.2, size=(7, 3))
        tangent = rng.normal(0.0, 0.08, size=(7, TANGENT_DIM)).astype(np.float32)
        target = product_exp_np(reference, tangent)
        recovered = product_log_np(reference, target)
        np.testing.assert_allclose(recovered, tangent, atol=2.0e-5, rtol=2.0e-5)
        rebuilt = product_exp_np(reference, recovered)
        np.testing.assert_allclose(
            product_distance_np(target, rebuilt), 0.0, atol=2.0e-5
        )

    def test_masked_retract_preserves_unselected_factors(self) -> None:
        reference = identity_motion(5)
        tangent = np.zeros((5, TANGENT_DIM), dtype=np.float32)
        tangent[:, :3] = 0.25
        tangent[:, 3:] = 0.30
        joint_mask = np.zeros((5, NUM_JOINTS), dtype=np.float32)
        joint_mask[:, 4] = 1.0
        result = masked_retract_np(
            reference,
            tangent,
            joint_mask=joint_mask,
            root_mask=np.zeros((5,), dtype=np.float32),
        )
        np.testing.assert_array_equal(result[:, 4:7], reference[:, 4:7])
        delta = product_log_np(reference, result).reshape(5, -1)
        np.testing.assert_allclose(delta[:, :3], 0.0, atol=1.0e-7)
        joint_delta = delta[:, 3:].reshape(5, NUM_JOINTS, 3)
        np.testing.assert_allclose(joint_delta[:, :4], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(joint_delta[:, 5:], 0.0, atol=1.0e-6)
        self.assertGreater(float(np.linalg.norm(joint_delta[:, 4])), 0.0)

    def test_zero_mask_preserves_reference_exactly(self) -> None:
        rng = np.random.default_rng(23)
        reference = identity_motion(4)
        rotations = so3_exp_np(
            rng.normal(0.0, 0.3, size=(4, NUM_JOINTS, 3)).astype(np.float32)
        )
        reference[:, ROT6D_START:ROT6D_END] = matrix_to_rot6d_np(
            rotations
        ).reshape(4, -1)
        reference[:, 4:7] = rng.normal(0.0, 0.2, size=(4, 3))
        tangent = rng.normal(0.0, 0.5, size=(4, TANGENT_DIM)).astype(np.float32)
        result = masked_retract_np(
            reference,
            tangent,
            joint_mask=np.zeros((4, NUM_JOINTS), dtype=np.float32),
            root_mask=np.zeros((4,), dtype=np.float32),
        )
        np.testing.assert_array_equal(result, reference)

    def test_norm_caps_are_geometric(self) -> None:
        reference = identity_motion(3)
        tangent = np.ones((3, TANGENT_DIM), dtype=np.float32)
        result = masked_retract_np(
            reference,
            tangent,
            max_root_m=0.05,
            max_rotation_rad=0.10,
        )
        delta = product_log_np(reference, result)
        np.testing.assert_array_less(
            np.linalg.norm(delta[:, :3], axis=-1), 0.05001
        )
        joint_norm = np.linalg.norm(
            delta[:, 3:].reshape(3, NUM_JOINTS, 3), axis=-1
        )
        np.testing.assert_array_less(joint_norm, 0.10001)

    def test_product_distance_is_symmetric(self) -> None:
        reference = identity_motion(4)
        tangent = np.zeros((4, TANGENT_DIM), dtype=np.float32)
        tangent[:, 0] = np.linspace(0.0, 0.2, 4)
        tangent[:, 3:6] = np.asarray([0.1, -0.04, 0.02], dtype=np.float32)
        target = product_exp_np(reference, tangent)
        np.testing.assert_allclose(
            product_distance_np(reference, target),
            product_distance_np(target, reference),
            atol=2.0e-5,
        )

    def test_parallel_transport_preserves_each_factor_norm(self) -> None:
        rng = np.random.default_rng(31)
        reference = identity_motion(6)
        path = rng.normal(0.0, 0.12, size=(6, TANGENT_DIM)).astype(np.float32)
        target = product_exp_np(reference, path)
        tangent = rng.normal(0.0, 0.20, size=(6, TANGENT_DIM)).astype(np.float32)
        transported = parallel_transport_np(reference, target, tangent)
        np.testing.assert_allclose(
            np.linalg.norm(transported[:, :3], axis=-1),
            np.linalg.norm(tangent[:, :3], axis=-1),
            atol=2.0e-5,
        )
        before = np.linalg.norm(
            tangent[:, 3:].reshape(6, NUM_JOINTS, 3), axis=-1
        )
        after = np.linalg.norm(
            transported[:, 3:].reshape(6, NUM_JOINTS, 3), axis=-1
        )
        np.testing.assert_allclose(after, before, atol=2.0e-5)

    def test_contacts_are_not_modified_by_manifold_exp(self) -> None:
        reference = identity_motion(2)
        reference[:, :4] = np.asarray(
            [[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]],
            dtype=np.float32,
        )
        tangent = np.zeros((2, TANGENT_DIM), dtype=np.float32)
        target = product_exp_np(reference, tangent)
        np.testing.assert_array_equal(target[:, :4], reference[:, :4])

    def test_trust_region_is_nonincreasing_and_mask_safe(self) -> None:
        reference = identity_motion(12)
        tangent = np.zeros((12, TANGENT_DIM), dtype=np.float32)
        tangent[3:9, 0] = np.asarray(
            [0.0, 0.07, -0.05, 0.08, -0.04, 0.0], dtype=np.float32
        )
        tangent[3:9, 3:6] = np.asarray(
            [0.2, -0.1, 0.05], dtype=np.float32
        )
        proposal = product_exp_np(reference, tangent)
        joint_mask = np.zeros((12, NUM_JOINTS), dtype=np.float32)
        joint_mask[3:9, 0] = 1.0
        root_mask = np.zeros((12,), dtype=np.float32)
        root_mask[3:9] = 1.0
        refined, report = riemannian_trust_region_refine_np(
            reference,
            proposal,
            joint_mask=joint_mask,
            root_mask=root_mask,
            contact_mask=root_mask,
            steps=5,
        )
        self.assertTrue(report["objective_nonincreasing"])
        self.assertLessEqual(
            report["final_objective"], report["initial_objective"] + 1.0e-10
        )
        self.assertGreaterEqual(report["accepted_steps"], 1)
        np.testing.assert_allclose(
            product_log_np(reference, refined)[:3], 0.0, atol=1.0e-7
        )
        np.testing.assert_allclose(
            product_log_np(reference, refined)[9:], 0.0, atol=1.0e-7
        )


@unittest.skipIf(product_torch is None, "PyTorch is not installed")
class ProductManifoldTorchTests(unittest.TestCase):
    def test_torch_log_exp_round_trip_and_gradient(self) -> None:
        from motion_geometry.product_manifold import (
            product_exp_torch,
            product_log_torch,
        )

        reference_np = identity_motion(4)[None]
        reference = product_torch.from_numpy(reference_np)
        tangent = (
            product_torch.randn(1, 4, TANGENT_DIM) * 0.04
        ).requires_grad_(True)
        target = product_exp_torch(reference, tangent)
        recovered = product_log_torch(reference, target)
        self.assertEqual(tuple(recovered.shape), (1, 4, TANGENT_DIM))
        loss = (recovered**2).mean()
        loss.backward()
        self.assertIsNotNone(tangent.grad)
        self.assertTrue(bool(product_torch.isfinite(tangent.grad).all()))
        self.assertLess(
            float((recovered.detach() - tangent.detach()).abs().max()), 5.0e-4
        )


if __name__ == "__main__":
    unittest.main()
