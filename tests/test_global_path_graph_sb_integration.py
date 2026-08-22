import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GlobalPathGraphSBIntegrationTests(unittest.TestCase):
    def test_graph_sb_route_and_fail_closed_nonconvergence(self):
        script = textwrap.dedent(
            """
            import json, os, sys, types, unittest.mock
            import numpy as np
            import torch

            package = types.ModuleType("pytorch3d")
            transforms = types.ModuleType("pytorch3d.transforms")
            transforms.__getattr__ = lambda name: (
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("pytorch3d placeholder invoked")
                )
            )
            package.transforms = transforms
            sys.modules["pytorch3d"] = package
            sys.modules["pytorch3d.transforms"] = transforms

            import routing.global_path as route

            count = 4
            rotation = np.broadcast_to(
                np.eye(3, dtype=np.float32), (count, 24, 3, 3)
            ).copy()
            joint = np.zeros((count, 24, 3), dtype=np.float32)
            root = np.zeros((count, 3), dtype=np.float32)
            uids = ["e0", "e1", "e2", "e3"]
            db = {
                "paths": np.asarray(["a", "b", "c", "d"], dtype=object),
                "event_uids": np.asarray(uids, dtype=object),
                "source_uids": np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
                "event_families": np.asarray(["f0", "f1", "f2", "f3"], dtype=object),
                "dance_keys": np.asarray(["d0", "d1", "d2", "d3"], dtype=object),
                "performer_groups": np.asarray(["female"] * count, dtype=object),
                "anatomy_hard_valid": np.ones(count, dtype=bool),
                "event_heading_valid": np.ones(count, dtype=bool),
                "event_geometry_combined_quality": np.full(count, 0.8, dtype=np.float32),
                "anatomy_quality": np.full(count, 0.9, dtype=np.float32),
                "graph_route_entry_rotation_matrix": rotation,
                "graph_route_exit_rotation_matrix": rotation,
                "event_geometry_entry_omega": joint,
                "event_geometry_exit_omega": joint,
                "event_geometry_entry_alpha": joint,
                "event_geometry_exit_alpha": joint,
                "event_geometry_entry_root_velocity_mps": root,
                "event_geometry_exit_root_velocity_mps": root,
                "posture_entry": np.asarray(["standing"] * count, dtype=object),
                "posture_exit": np.asarray(["standing"] * count, dtype=object),
                "contact_entry": np.zeros((count, 4), dtype=np.float32),
                "contact_exit": np.zeros((count, 4), dtype=np.float32),
            }
            slots = []
            for target in range(3):
                probability = [0.01] * count
                probability[target] = 0.97
                slots.append({
                    "router_architecture": "ctsr_weak_temporal_v1",
                    "formal_candidate_event_uids": list(uids),
                    "formal_candidate_router_probabilities": probability,
                })
            candidates = [list(range(count)) for _ in slots]
            common = {
                "PERFORMER_GROUP": "female",
                "PERFORMER_IDENTITY_MODE": "group",
                "PERFORMER_REQUIRE_SOLO_COMPATIBLE": "0",
                "GROUNDING_GLOBAL_ROUTE_ENABLE": "1",
                "GROUNDING_GLOBAL_ROUTE_TOPK": "4",
                "GRAPH_ROUTE_SOLVER": "fisher_rao_graph_sb",
            }
            with unittest.mock.patch.dict(os.environ, common, clear=False):
                direct = route._global_route_preorder(slots, candidates, db)
                direct_report = dict(route._GLOBAL_ROUTE_REPORT)

            failure = ""
            strict = {
                **common,
                "GRAPH_ROUTE_SB_MAX_ITER": "1",
                "GRAPH_ROUTE_SB_TOLERANCE": "1e-15",
            }
            with unittest.mock.patch.dict(os.environ, strict, clear=False):
                try:
                    route._global_route_preorder(slots, candidates, db)
                except RuntimeError as exc:
                    failure = str(exc)
            print(json.dumps({
                "direct_first": [row[0] for row in direct],
                "direct_path": direct_report["chosen_event_path"],
                "direct_solver": direct_report["solver"],
                "direct_fallback": direct_report["fallback_used"],
                "failure": failure,
            }))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            self.fail(
                "graph-SB integration subprocess failed:\n"
                + result.stdout
                + "\n"
                + result.stderr
            )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["direct_first"], [0, 1, 2])
        self.assertEqual(payload["direct_path"], [0, 1, 2])
        self.assertEqual(payload["direct_solver"], "fisher_rao_graph_sb")
        self.assertFalse(payload["direct_fallback"])
        self.assertIn("did not converge", payload["failure"])


if __name__ == "__main__":
    unittest.main()
