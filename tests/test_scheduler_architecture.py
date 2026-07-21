import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SchedulerArchitectureTests(unittest.TestCase):
    def test_runtime_layout_is_project_owned(self):
        required = [
            "scheduling/whole_song_scheduler.py",
            "scheduling/music_slot_descriptor.py",
            "scheduling/index_io.py",
            "scheduling/retrieval.py",
            "scheduling/transition_builder.py",
            "scheduling/event_resampling.py",
            "scheduling/duration_features.py",
            "scheduling/duration_alignment.py",
            "scheduling/transition_diffusion.py",
            "motion_geometry/heading.py",
            "support/scheduler_common.py",
        ]
        legacy = [
            "vendor/edge_scheduler",
            "scheduling/schedule_whole_song.py",
            "scheduling/build_music_semantic_slot_descriptor.py",
            "scheduling/global_duration_alignment.py",
            "scheduling/duration_utils.py",
            "scheduling/extract_music_features.py",
            "scheduling/music_event_calibrated.py",
            "events/event_resampling.py",
            "support/turn_utils.py",
        ]
        self.assertEqual([], [path for path in required if not (ROOT / path).is_file()])
        self.assertEqual([], [path for path in legacy if (ROOT / path).exists()])

    def test_build_schedule_uses_project_modules(self):
        source = (ROOT / "scheduling" / "build_schedule.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"scheduling.whole_song_scheduler"', source)
        self.assertIn('"scheduling.music_slot_descriptor"', source)
        self.assertNotIn("DUNHUANG_SCHEDULER_RUNTIME_ROOT", source)
        self.assertNotIn("vendor/edge_scheduler", source)
        self.assertNotIn("vendor\\edge_scheduler", source)

    def test_scheduler_runtime_has_no_legacy_imports(self):
        roots = [
            ROOT / "scheduling",
            ROOT / "model",
            ROOT / "motion_geometry",
            ROOT / "support",
        ]
        failures = []
        for base in roots:
            for path in base.glob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module == "tools" or module.startswith("tools."):
                            failures.append(f"{path.relative_to(ROOT)}: {module}")
                        if module.startswith("model.v"):
                            failures.append(f"{path.relative_to(ROOT)}: {module}")
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "tools" or alias.name.startswith("tools."):
                                failures.append(
                                    f"{path.relative_to(ROOT)}: {alias.name}"
                                )
        self.assertEqual([], failures)

    def test_migration_manifest_is_valid_json(self):
        path = ROOT / "docs" / "development" / "path_migration.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            "scheduling/whole_song_scheduler.py",
            data["tools/schedule_v26_whole_song.py"],
        )
        self.assertEqual(
            "motion_geometry/heading.py",
            data["tools/v22_turn_utils.py"],
        )

    def test_pipeline_supplies_asset_bundle_fps_contract(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        marker = 'scheduling/build_asset_bundle.py'
        start = source.index(marker)
        block = source[start : start + 700]
        self.assertIn('--fps "$V46_51_FPS"', block)

    def test_pipeline_supplies_renderer_fps_contract(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        marker = 'rendering/render_motion.py'
        start = source.index(marker)
        block = source[start : start + 500]
        self.assertIn('--fps "$V46_51_FPS"', block)

    def test_runtime_launchers_do_not_embed_machine_specific_paths(self):
        paths = [
            ROOT / "scripts" / "pipeline.sh",
            ROOT / "scripts" / "run_experiment.sh",
            ROOT / "scripts" / "research_pipeline.sh",
            ROOT / "configs" / "paths.env",
            ROOT / "configs" / "scheduler.env",
            ROOT / "configs" / "experiment.env",
        ]
        failures = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            if "/home/disk" in source or "storage/EDGE" in source:
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual([], failures)

    def test_aist_dataset_exposes_rate_and_physical_contact_contracts(self):
        source = (ROOT / "dataset" / "dance_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("data_fps: int = 30", source)
        self.assertIn("raw_fps: int = 60", source)
        self.assertIn("contact_speed_threshold_mps", source)
        self.assertIn("* float(self.data_fps)", source)

    def test_scheduler_profiles_preserve_external_fps(self):
        for relative in ("configs/scheduler.env", "configs/experiment.env"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                'export V46_51_FPS="${V46_51_FPS:-30}"',
                source,
            )
            self.assertIn(
                'export V46_49_RETARGET_FPS="${V46_49_RETARGET_FPS:-$V46_51_FPS}"',
                source,
            )
            self.assertNotIn("export V46_51_FPS=30", source)
            self.assertNotIn("export V46_49_RETARGET_FPS=30", source)

    def test_scheduler_passes_physical_rate_to_deep_music_features(self):
        source = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        marker = "phrase_semantic_matrix("
        start = source.index(marker)
        block = source[start : start + 600]
        self.assertIn("fps=float(args.fps)", block)


if __name__ == "__main__":
    unittest.main()
