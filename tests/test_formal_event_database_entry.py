import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from events import build_database_entry
from events.build_semantics import build_semantics
from events.filter_anatomy import filter_database
from events.intrinsic_geometry import augment_database
from motion_geometry.smpl24 import skeleton_contract_json
from scheduling.build_generation_index import build_generation_index


def identity_motion(frames: int = 30) -> np.ndarray:
    motion = np.zeros((frames, 151), dtype=np.float32)
    motion[:, 5] = 0.93
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 7:151] = np.tile(identity, 24)
    return motion


class FormalEventDatabaseEntryTests(unittest.TestCase):
    def test_entrypoint_runs_base_anatomy_then_intrinsic_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "train"
            out_dir.mkdir()
            (out_dir / "events.npz").write_bytes(b"fixture")
            calls = []

            with (
                mock.patch.object(
                    build_database_entry.base,
                    "main",
                    side_effect=lambda _args: calls.append("base") or 0,
                ),
                mock.patch.object(
                    build_database_entry,
                    "filter_database",
                    side_effect=lambda *_args, **_kwargs: (
                        calls.append("anatomy") or {"ok": True, "events_after": 1}
                    ),
                ),
                mock.patch.object(
                    build_database_entry,
                    "augment_database",
                    side_effect=lambda *_args, **_kwargs: (
                        calls.append("geometry") or {"ok": True, "num_events": 1}
                    ),
                ),
            ):
                result = build_database_entry.main(["--out_db", str(out_dir)])

            self.assertEqual(result, 0)
            self.assertEqual(calls, ["base", "anatomy", "geometry"])
            report = json.loads(
                (out_dir / "events.formal_build.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["ok"])
            self.assertFalse(report["archived_business_paths_restored"])

    def test_physical_enrichment_produces_scheduler_posture_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out_dir = root / "validation"
            out_dir.mkdir()
            motion_path = out_dir / "event_0000000.npy"
            np.save(motion_path, identity_motion())
            db_path = out_dir / "events.npz"
            np.savez_compressed(
                db_path,
                paths=np.asarray([str(motion_path)], dtype=object),
                source_uids=np.asarray(["source_a"], dtype=object),
                recording_uids=np.asarray(["recording_a"], dtype=object),
                source_files=np.asarray(["source_a.npz"], dtype=object),
                starts=np.asarray([0], dtype=np.int32),
                ends=np.asarray([30], dtype=np.int32),
                frames=np.asarray([30], dtype=np.int32),
                source_start_seconds=np.asarray([0.0], dtype=np.float64),
                source_end_seconds=np.asarray([1.0], dtype=np.float64),
                canonical_fps=np.asarray([30.0], dtype=np.float32),
                event_families=np.asarray(["pose_hold"], dtype=object),
                motion_stage_roles=np.asarray(["development"], dtype=object),
                event_semantics_schema_version=np.asarray(
                    "chang_e_five_layer_event_semantics_v2", dtype=object
                ),
                music_compatibility_scores_json=np.asarray(
                    ['{"pose_hold": 1.0}'], dtype=object
                ),
                event_quality_scores=np.asarray([0.9], dtype=np.float32),
                skeleton_contract_json=np.asarray(
                    skeleton_contract_json(), dtype=object
                ),
            )
            meta_path = out_dir / "events_meta.json"
            meta_path.write_text("[{}]", encoding="utf-8")

            environment = {
                "RETARGET_EVENT_DB_MIN_EVENTS_EVAL": "1",
                "RETARGET_EVENT_DB_MIN_KEEP_RATIO": "0",
                "RETARGET_EVENT_MIN_PER_SOURCE_EVAL": "1",
            }
            with mock.patch.dict(os.environ, environment):
                anatomy = filter_database(
                    db_path,
                    meta_path,
                    out_dir / "anatomy.json",
                )
                geometry = augment_database(
                    db_path,
                    out_dir / "geometry.json",
                    fps=None,
                )

            self.assertTrue(anatomy["ok"])
            self.assertTrue(geometry["ok"])
            with np.load(db_path, allow_pickle=True) as db:
                self.assertIn(
                    str(db["posture_entry"][0]),
                    {
                        "floor_pose",
                        "kneeling",
                        "deep_squat",
                        "half_squat",
                        "standing",
                        "aerial",
                    },
                )
                self.assertEqual(len(db["event_geometry_combined_quality"]), 1)

            aesd_path = out_dir / "events_aesd.npz"
            semantics = build_semantics(db_path, aesd_path)
            self.assertTrue(semantics["ok"])

            index_report = build_generation_index(
                aesd_path,
                out_dir / "index.json",
                out_dir / "index.npz",
            )
            self.assertEqual(index_report["num_events"], 1)


if __name__ == "__main__":
    unittest.main()
