import unittest

import numpy as np

from grounding.manifold_ops import lorentz_project_np
from motion_geometry.rotations import so3_exp_np
from routing.event_graph_geometry import (
    EventGraphGeometryConfig,
    event_node_feasibility,
    lorentz_hierarchy_distance,
    manifold_edge_cost,
    so3_product_endpoint_distance,
)


def database() -> dict:
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32), (3, 24, 3, 3)
    ).copy()
    rotation = so3_exp_np(
        np.asarray([0.0, np.pi / 2.0, 0.0], dtype=np.float32)
    )
    identity[2, 0] = rotation
    lorentz = lorentz_project_np(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.7, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    )
    zeros_joint = np.zeros((3, 24, 3), dtype=np.float32)
    zeros_root = np.zeros((3, 3), dtype=np.float32)
    return {
        "paths": np.asarray(["a", "b", "c"], dtype=object),
        "anatomy_hard_valid": np.asarray([True, True, False]),
        "event_heading_valid": np.asarray([True, True, True]),
        "v46_53_combined_quality": np.asarray([0.9, 0.8, 0.7]),
        "v46_55_entry_rotation_matrix": identity,
        "v46_55_exit_rotation_matrix": identity,
        "v46_53_mixed_lorentz": lorentz,
        "v46_53_mixed_curvature": np.asarray(1.0, dtype=np.float32),
        "v46_53_entry_omega": zeros_joint,
        "v46_53_exit_omega": zeros_joint,
        "v46_53_entry_alpha": zeros_joint,
        "v46_53_exit_alpha": zeros_joint,
        "v46_53_entry_root_velocity_mps": zeros_root,
        "v46_53_exit_root_velocity_mps": zeros_root,
        "posture_entry": np.asarray(["standing"] * 3, dtype=object),
        "posture_exit": np.asarray(["standing"] * 3, dtype=object),
        "pelvis_height_entry_norm": np.asarray([0.8] * 3),
        "pelvis_height_exit_norm": np.asarray([0.8] * 3),
        "entry_floor_offset_m": np.asarray([0.0, 0.0, 0.0]),
        "exit_floor_offset_m": np.asarray([0.0, 0.0, 0.0]),
        "contact_entry": np.asarray(
            [[0.0] * 4, [0.0] * 4, [1.0] * 4], dtype=np.float32
        ),
        "contact_exit": np.asarray(
            [[0.0] * 4, [0.0] * 4, [1.0] * 4], dtype=np.float32
        ),
    }


class EventGraphGeometryTests(unittest.TestCase):
    def test_node_gate_preserves_anatomy_and_heading(self):
        db = database()
        self.assertEqual(event_node_feasibility(db, 0), (True, ()))
        valid, reasons = event_node_feasibility(db, 2)
        self.assertFalse(valid)
        self.assertIn("anatomy_hard_valid", reasons)

    def test_so3_endpoint_distance_is_intrinsic(self):
        db = database()
        zero, available = so3_product_endpoint_distance(db, 0, 1)
        changed, changed_available = so3_product_endpoint_distance(db, 0, 2)
        self.assertTrue(available)
        self.assertTrue(changed_available)
        self.assertAlmostEqual(zero, 0.0, places=7)
        self.assertAlmostEqual(changed, (np.pi / 2.0) / np.sqrt(24.0), places=5)

    def test_lorentz_factor_changes_edge_distance(self):
        db = database()
        near, available = lorentz_hierarchy_distance(db, 0, 1)
        far, far_available = lorentz_hierarchy_distance(db, 0, 2)
        self.assertTrue(available)
        self.assertTrue(far_available)
        self.assertGreater(near, 0.0)
        self.assertGreater(far, near)

    def test_composite_cost_reports_manifold_availability(self):
        db = database()
        result = manifold_edge_cost(
            db,
            0,
            1,
            config=EventGraphGeometryConfig(
                so3_weight=1.0,
                lorentz_weight=1.0,
                posture_hard=0.0,
                floor_hard_m=0.0,
                contact_hard=0.0,
                root_velocity_hard_mps=0.0,
            ),
        )
        self.assertTrue(result["hard_feasible"])
        self.assertTrue(result["so3_available"])
        self.assertTrue(result["lorentz_available"])
        self.assertAlmostEqual(
            result["total"],
            result["physical"]
            + result["so3_product_distance_rad"]
            + result["lorentz_hierarchy_distance"],
            places=7,
        )

    def test_hard_contact_gap_removes_graph_edge(self):
        db = database()
        result = manifold_edge_cost(
            db,
            0,
            2,
            config=EventGraphGeometryConfig(
                posture_hard=0.0,
                floor_hard_m=0.0,
                contact_hard=0.75,
                root_velocity_hard_mps=0.0,
            ),
        )
        self.assertFalse(result["hard_feasible"])
        self.assertIn("contact", result["hard_reasons"])


if __name__ == "__main__":
    unittest.main()
