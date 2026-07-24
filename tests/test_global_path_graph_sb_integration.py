import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GlobalPathGraphSBIntegrationTests(unittest.TestCase):
    def test_graph_sb_route_and_auditable_fallback(self):
        # The production repository pins pytorch3d.  The lightweight unit-test
        # runtime does not ship that package, so this subprocess supplies import
        # placeholders only; no motion-generation function is invoked.
        script = textwrap.dedent(
            """
            import json, os, pathlib, sys, tempfile, types, unittest.mock
            import numpy as np
            import torch

            package = types.ModuleType("pytorch3d")
            transforms = types.ModuleType("pytorch3d.transforms")
            def placeholder(*args, **kwargs):
                raise RuntimeError("pytorch3d placeholder invoked")
            transforms.__getattr__ = lambda name: placeholder
            package.transforms = transforms
            sys.modules["pytorch3d"] = package
            sys.modules["pytorch3d.transforms"] = transforms

            import routing.global_path as route

            class Runtime:
                def score(self, slot, event_id):
                    return 0.95 if int(event_id) == int(slot["target"]) else 0.05

            route._runtime = lambda db: Runtime()
            count = 4
            rotation = np.broadcast_to(
                np.eye(3, dtype=np.float32), (count, 24, 3, 3)
            ).copy()
            joint = np.zeros((count, 24, 3), dtype=np.float32)
            root = np.zeros((count, 3), dtype=np.float32)
            db = {
                "paths": np.asarray(["a", "b", "c", "d"], dtype=object),
                "event_uids": np.asarray(["e0", "e1", "e2", "e3"], dtype=object),
                "source_uids": np.asarray(["s0", "s1", "s2", "s3"], dtype=object),
                "event_families": np.asarray(["f0", "f1", "f2", "f3"], dtype=object),
                "dance_keys": np.asarray(["d0", "d1", "d2", "d3"], dtype=object),
                "performer_groups": np.asarray(["female"] * count, dtype=object),
                "anatomy_hard_valid": np.ones(count, dtype=bool),
                "event_heading_valid": np.ones(count, dtype=bool),
                "v46_53_combined_quality": np.full(count, 0.8, dtype=np.float32),
                "anatomy_quality": np.full(count, 0.9, dtype=np.float32),
                "v46_55_entry_rotation_matrix": rotation,
                "v46_55_exit_rotation_matrix": rotation,
                "v46_53_entry_omega": joint,
                "v46_53_exit_omega": joint,
                "v46_53_entry_alpha": joint,
                "v46_53_exit_alpha": joint,
                "v46_53_entry_root_velocity_mps": root,
                "v46_53_exit_root_velocity_mps": root,
                "posture_entry": np.asarray(["standing"] * count, dtype=object),
                "posture_exit": np.asarray(["standing"] * count, dtype=object),
                "contact_entry": np.zeros((count, 4), dtype=np.float32),
                "contact_exit": np.zeros((count, 4), dtype=np.float32),
            }
            slots = [{"target": 0}, {"target": 1}, {"target": 2}]
            candidates = [list(range(count)) for _ in slots]
            common = {
                "PERFORMER_GROUP": "female",
                "V46_53_GLOBAL_ROUTE_TOPK": "4",
            }
            with unittest.mock.patch.dict(os.environ, common, clear=False):
                direct = route._graph_sb_global_route_preorder(
                    slots, candidates, db
                )
                direct_report = dict(route._GLOBAL_ROUTE_REPORT)
            fallback_env = {
                **common,
                "V46_55_ROUTE_SOLVER": "fisher_rao_graph_sb",
                "V46_55_SB_MAX_ITER": "1",
                "V46_55_SB_TOLERANCE": "1e-15",
                "V46_55_SB_ALLOW_LEGACY_FALLBACK": "1",
            }
            with unittest.mock.patch.dict(os.environ, fallback_env, clear=False):
                fallback = route._global_route_preorder(slots, candidates, db)
                fallback_report = dict(route._GLOBAL_ROUTE_REPORT)
            with tempfile.TemporaryDirectory() as tmp:
                report_path = pathlib.Path(tmp) / "report.json"
                report_path.write_text("{}", encoding="utf-8")
                route.v52._resolve_motion_path = lambda *args, **kwargs: None
                route.v52.save_json = lambda payload, path: path.write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                route._patch_report(report_path)
                patched_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            print(json.dumps({
                "direct_first": [row[0] for row in direct],
                "direct_path": direct_report["chosen_event_path"],
                "direct_converged": direct_report["schrodinger"]["converged"],
                "fallback_slots": len(fallback),
                "fallback_schema": fallback_report["schema"],
                "fallback_solver": fallback_report["solver"],
                "fallback_used": fallback_report["fallback_used"],
                "fallback_reason": fallback_report["fallback_reason"],
                "fallback_route_schema": fallback_report["fallback_route"]["schema"],
                "would_emit_v46_55_field": fallback_report["schema"].startswith("v46_55_"),
                "patched_has_compat": "v46_53_global_route" in patched_report,
                "patched_has_v46_55": "v46_55_graph_sb_route" in patched_report,
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
        self.assertTrue(payload["direct_converged"])
        self.assertEqual(payload["fallback_slots"], 3)
        self.assertEqual(
            payload["fallback_schema"],
            "v46_55_fisher_rao_graph_sb_fallback_v1",
        )
        self.assertEqual(payload["fallback_solver"], "legacy_beam")
        self.assertTrue(payload["fallback_used"])
        self.assertIn("did not converge", payload["fallback_reason"])
        self.assertEqual(
            payload["fallback_route_schema"],
            "v46_53_entropy_regularised_global_event_path",
        )
        self.assertTrue(payload["would_emit_v46_55_field"])
        self.assertTrue(payload["patched_has_compat"])
        self.assertTrue(payload["patched_has_v46_55"])


if __name__ == "__main__":
    unittest.main()
