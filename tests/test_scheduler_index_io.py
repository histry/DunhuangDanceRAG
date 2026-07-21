import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scheduling.index_io import (
    REQUIRED_EVENT_ARRAYS,
    load_shared_index,
    resolve_event_motion_path,
)


class SchedulerIndexPathTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_aligned_index(root: Path, layout=None):
        index = root / "event_index.json"
        metadata = {
            "items": [
                {
                    "pkl": "event.pkl",
                    "event_uid": "evt_test_contract",
                }
            ]
        }
        if layout is not None:
            metadata["rot6d_layout"] = layout
        index.write_text(json.dumps(metadata), encoding="utf-8")
        arrays = {name: np.zeros((1, 1), dtype=np.float32) for name in REQUIRED_EVENT_ARRAYS}
        npz = root / "duration_index.npz"
        np.savez(npz, **arrays)
        return index, npz

    def test_shared_index_requires_explicit_rotation_contract(self):
        with tempfile.TemporaryDirectory() as root_dir:
            index, npz = self._write_minimal_aligned_index(Path(root_dir))
            with self.assertRaisesRegex(RuntimeError, "no rot6d_layout"):
                load_shared_index(index, npz)

    def test_shared_index_rejects_legacy_row_runtime_assets(self):
        with tempfile.TemporaryDirectory() as root_dir:
            index, npz = self._write_minimal_aligned_index(
                Path(root_dir),
                layout="pytorch3d_row",
            )
            with self.assertRaisesRegex(RuntimeError, "requires canonical"):
                load_shared_index(index, npz)

    def test_shared_index_accepts_canonical_assets(self):
        with tempfile.TemporaryDirectory() as root_dir:
            index, npz = self._write_minimal_aligned_index(Path(root_dir), layout="column")
            metadata, arrays, items = load_shared_index(index, npz)
            try:
                self.assertEqual("column", metadata["rot6d_layout"])
                self.assertEqual(1, len(items))
            finally:
                arrays.close()

    def test_project_relative_motion_does_not_depend_on_cwd(self):
        with tempfile.TemporaryDirectory() as project_dir, tempfile.TemporaryDirectory() as other_dir:
            project = Path(project_dir)
            index = project / "assets" / "indexes" / "event_index.json"
            motion = project / "assets" / "events" / "example.pkl"
            index.parent.mkdir(parents=True)
            motion.parent.mkdir(parents=True)
            index.write_text('{"items": []}', encoding="utf-8")
            motion.write_bytes(b"event")

            previous = Path.cwd()
            try:
                os.chdir(other_dir)
                resolved = resolve_event_motion_path(
                    {"pkl": "assets/events/example.pkl"},
                    index,
                    project_root=project,
                )
            finally:
                os.chdir(previous)

            self.assertEqual(motion.resolve(), resolved)

    def test_declared_event_root_is_supported(self):
        with tempfile.TemporaryDirectory() as root_dir:
            root = Path(root_dir)
            index = root / "indexes" / "event_index.json"
            motion = root / "store" / "clip.pkl"
            index.parent.mkdir(parents=True)
            motion.parent.mkdir(parents=True)
            index.write_text('{"items": []}', encoding="utf-8")
            motion.write_bytes(b"event")

            resolved = resolve_event_motion_path(
                {"path": "clip.pkl"},
                index,
                metadata={"event_root": "../store"},
                project_root=root / "unrelated_project",
            )
            self.assertEqual(motion.resolve(), resolved)

    def test_empty_reference_fails_early(self):
        with tempfile.TemporaryDirectory() as root_dir:
            index = Path(root_dir) / "event_index.json"
            index.write_text('{"items": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "neither"):
                resolve_event_motion_path({}, index)


if __name__ == "__main__":
    unittest.main()
