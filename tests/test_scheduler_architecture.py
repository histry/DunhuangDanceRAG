import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SchedulerArchitectureTests(unittest.TestCase):
    def test_scheduler_preserves_composed_root_trajectory(self):
        source = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("canonicalize_event_root_np", source)
        self.assertIn("compose_event_root_xz_np", source)
        self.assertIn("make_so3_transition", source)
        self.assertNotIn("content[:, ROOT_X] = 0.0", source)
        self.assertNotIn("motion[:, ROOT_X] = 0.0", source)

    def test_all_transition_backends_preserve_root_xz(self):
        paths = (
            "support/contact_inr.py",
            "training/boundary_dynamics.py",
        )
        forbidden = (
            "root[..., ROOT_X - ROOT.start] = 0.0",
            "root[..., ROOT_Z - ROOT.start] = 0.0",
            "result[:, ROOT_X] = 0.0",
            "result[:, ROOT_Z] = 0.0",
            "out[:, ROOT_X] = 0.0",
            "out[:, ROOT_Z] = 0.0",
        )
        failures = []
        for relative in paths:
            source = (ROOT / relative).read_text(encoding="utf-8")
            for statement in forbidden:
                if statement in source:
                    failures.append(f"{relative}: {statement}")
        self.assertEqual([], failures)

    def test_all_start_anchor_entrypoints_are_xz_translation_only(self):
        source = (ROOT / "support" / "motion_geometry.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def apply_start_anchor_so3(")
        end = source.index("\ndef temporal_so3_filter_np(", start)
        block = source[start:end]
        self.assertIn("stage_offset", block)
        self.assertNotIn("ROOT_Y] =", block)
        self.assertNotIn("CONTACT] =", block)
        self.assertNotIn("ROT] =", block)

    def test_scheduler_routes_with_physical_endpoint_state(self):
        source = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        for name in (
            "posture_root_height_gap_m",
            "posture_state_gap",
            "floor_gap_m",
            "contact_gap",
            "root_velocity_jump_mps",
            "physical_edge_hard_prune",
        ):
            self.assertIn(name, source)

    def test_ik_uses_contact_ramps_and_local_transactions(self):
        source = (ROOT / "training" / "motion_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("contact_ramp_weights_np", source)
        self.assertIn("local_ownership_window_transactions", source)
        self.assertIn("ik_local_transaction", source)
        self.assertIn(
            "trial[own_start:own_end] = blend_edge151_geodesic_np(",
            source,
        )

    def test_final_closed_loop_uses_same_geometry_contract_as_scheduler(self):
        source = (
            ROOT / "routing" / "anatomy_heading_closed_loop.py"
        ).read_text(encoding="utf-8")
        for name in (
            "canonicalize_event_root_np",
            "make_so3_transition",
            "project_transition_floor_np",
            "recompute_transition_contacts_np",
            "root_velocity_gap_mps",
            "floor_offset_gap_m",
        ):
            self.assertIn(name, source)
        contact_position = source.index("recompute_transition_contacts_np(")
        return_position = source.index(
            "return np.asarray(bridge, dtype=np.float32)",
            contact_position,
        )
        block = source[contact_position:return_position]
        self.assertNotIn("base.enforce_contract", block)
        heading = (
            ROOT / "routing" / "heading_closed_loop.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Closed-loop slot frame contract mismatch", heading)
        self.assertNotIn(
            "source_hint=f\"event_heading_slot_exact_len:",
            heading,
        )

    def test_generation_db_and_global_route_expose_physical_endpoints(self):
        geometry = (
            ROOT / "events" / "intrinsic_geometry.py"
        ).read_text(encoding="utf-8")
        route = (ROOT / "routing" / "global_path.py").read_text(
            encoding="utf-8"
        )
        for name in (
            "event_geometry_entry_root_velocity_mps",
            "event_geometry_exit_root_velocity_mps",
        ):
            self.assertIn(name, geometry)
            self.assertIn(name, route)
        self.assertIn("entry_floor_offset_m", route)
        self.assertIn("exit_floor_offset_m", route)

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
            "motion_geometry/heading.py",
            "support/scheduler_common.py",
            "support/scheduler_checkpoint_contracts.py",
            "training/music_corpus.py",
            "training/temporal_music_router.py",
            "training/duration_model.py",
            "training/whole_song_planner.py",
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

    def test_pipeline_supplies_asset_bundle_fps_contract(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        marker = 'scheduling/build_asset_bundle.py'
        start = source.index(marker)
        block = source[start : start + 700]
        self.assertIn('--fps "$GENERATION_FPS"', block)

    def test_formal_split_and_event_db_audit_share_one_manifest_contract(self):
        research = (ROOT / "configs" / "research.env").read_text(
            encoding="utf-8"
        )
        self.assertIn('RETARGET_CLEAN_TRAIN_RATIO:-0.50', research)
        self.assertIn('RETARGET_CLEAN_VAL_RATIO:-0.25', research)
        self.assertIn('RETARGET_CLEAN_TEST_RATIO:-0.25', research)

        pipeline = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'SPLIT_MANIFEST="$CACHE_SPLIT_ROOT/source_split_manifest.json"',
            pipeline,
        )
        self.assertIn('--split_manifest "$SPLIT_MANIFEST"', pipeline)
        self.assertIn('--split "$split"', pipeline)

    def test_formal_event_db_entry_keeps_physical_enrichment_reachable(self):
        entry = (ROOT / "events" / "build_database_entry.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("filter_database(", entry)
        self.assertIn("augment_database(", entry)
        self.assertNotIn("grounding", entry.lower())

        index = (ROOT / "scheduling" / "build_generation_index.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("FORMAL_ANATOMY_SCHEMA", index)
        self.assertIn("FORMAL_INTRINSIC_GEOMETRY_SCHEMA", index)

    def test_pipeline_trains_scheduler_models_before_motion_refinement(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        markers = [
            "training/temporal_music_router.py train",
            "training/duration_model.py train",
            "training/whole_song_planner.py train",
            "scheduling/build_asset_bundle.py",
            "scripts/run_no_training_regression.py",
            "train-refiner",
            "train-diffusion",
        ]
        positions = [source.index(marker) for marker in markers]
        self.assertEqual(sorted(positions), positions)
        self.assertNotIn("scheduling/resolve_assets.py", source)

    def test_formal_pipeline_does_not_retrieve_again_with_legacy_contrastive(self):
        pipeline = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train-contrastive", pipeline)
        generation = pipeline[pipeline.index("routing/boundary_closed_loop.py") :]
        self.assertNotIn("--contrastive", generation)

    def test_formal_shell_gate_is_complete_and_optional_assets_are_nounset_safe(self):
        pipeline = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        launcher = (ROOT / "scripts" / "research_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$PY" -m pytest -q', launcher)
        self.assertNotIn("unittest discover", launcher)
        self.assertIn('${GENERATION_RESOLVED_HIERARCHY_INDEX_NPZ:-}', pipeline)
        self.assertIn('${GENERATION_RESOLVED_START_POSE:-}', pipeline)
        validation = pipeline.index('scheduling/validate_schedule.py')
        baselines = pipeline.index('scripts/evaluate_current_protocol_baselines.py')
        self.assertLess(validation, baselines)
        scheduler = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        closed_loop = (ROOT / "routing" / "boundary_closed_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("formal_candidate_event_uids", scheduler)
        self.assertIn("formal_ctsr_scheduler_locked_candidates", closed_loop)
        self.assertIn("selected_event_motion_descriptor_v1", closed_loop)

    def test_planner_duration_clamp_is_physical_time_aware(self):
        source = (ROOT / "model" / "whole_song_planner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.duration_min_frames", source)
        self.assertIn("self.duration_max_frames", source)
        self.assertNotIn("clamp(8.0, 600.0)", source)

    def test_planner_weak_labels_include_boundary_compatibility(self):
        source = (ROOT / "training" / "whole_song_planner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("intrinsic_transition_cost_from_arrays", source)
        self.assertNotIn(
            "\n    transition_cost_from_arrays,",
            source,
        )
        self.assertIn("entry_angular_velocity_radps", source)
        self.assertIn("exit_angular_velocity_radps", source)
        self.assertIn("posture_state_distance", source)

    def test_generation_index_carries_discrete_posture_endpoints(self):
        source = (
            ROOT / "scheduling" / "build_generation_index.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "generation_aligned_scheduler_index_v5_product_state_endpoints",
            source,
        )
        self.assertIn('"posture_entry": str(posture_entry[index])', source)
        self.assertIn('"posture_exit": str(posture_exit[index])', source)

    def test_runtime_scheduler_validates_full_checkpoint_contract(self):
        source = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("assert_scheduler_checkpoint_contract", source)
        self.assertIn('metadata["event_db_contract"]', source)

    def test_duration_dynamics_use_intrinsic_physical_time(self):
        source = (ROOT / "model" / "duration_predictor.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def _duration_dynamics_features")
        block = source[start : start + 4300]
        self.assertIn("matrix_to_axis_angle(relative)", block)
        self.assertIn("* float(self.fps)", block)
        self.assertIn("/ float(self.fps)", block)

    def test_pipeline_supplies_renderer_fps_contract(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(encoding="utf-8")
        marker = 'rendering/render_motion.py'
        start = source.index(marker)
        block = source[start : start + 500]
        self.assertIn('--fps "$GENERATION_FPS"', block)

    def test_pipeline_supplies_all_audit_fps_contracts(self):
        pipeline = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        for marker in (
            "evaluation/audit_gravity.py",
            "evaluation/audit_heading.py",
        ):
            positions = []
            cursor = 0
            while True:
                index = pipeline.find(marker, cursor)
                if index < 0:
                    break
                positions.append(index)
                cursor = index + len(marker)
            self.assertTrue(positions, marker)
            for index in positions:
                self.assertIn(
                    '--fps "$GENERATION_FPS"', pipeline[index : index + 450]
                )

        research = (ROOT / "scripts" / "research_pipeline.sh").read_text(
            encoding="utf-8"
        )
        index = research.index("evaluation/audit_motion.py")
        self.assertIn(
            '--fps "${GENERATION_FPS:-30}"', research[index : index + 450]
        )

    def test_runtime_launchers_do_not_embed_machine_specific_paths(self):
        paths = [
            ROOT / "scripts" / "pipeline.sh",
            ROOT / "scripts" / "run_experiment.sh",
            ROOT / "scripts" / "research_pipeline.sh",
            ROOT / "configs" / "experiment.env",
        ]
        failures = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            if "/home/disk" in source or "storage/EDGE" in source:
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual([], failures)

    def test_event_asset_resolution_never_depends_on_process_cwd(self):
        for relative in (
            "scheduling/index_io.py",
            "scheduling/build_generation_index.py",
            "events/intrinsic_geometry.py",
            "training/motion_models.py",
            "evaluation/audit_heading.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("Path.cwd() / raw", source)
            self.assertNotIn("candidates = [raw]\n", source)

    def test_scheduler_profiles_preserve_external_fps(self):
        source = (ROOT / "configs" / "scheduler.env").read_text(encoding="utf-8")
        entry = (ROOT / "configs" / "experiment.env").read_text(encoding="utf-8")
        self.assertIn('source "$_EXPERIMENT_CONFIG_DIR/scheduler.env"', entry)
        self.assertIn('export GENERATION_FPS="${GENERATION_FPS:-30}"', source)
        self.assertIn(
            'export SOURCE_RETARGET_FPS="${SOURCE_RETARGET_FPS:-$GENERATION_FPS}"',
            source,
        )
        self.assertNotIn("export GENERATION_FPS=30", source)
        self.assertIn('export MOTION_FPS="$GENERATION_FPS"', source)
        self.assertNotIn("export SOURCE_RETARGET_FPS=30", source)

    def test_shell_launchers_use_only_authoritative_experiment_entry(self):
        failures = []
        for path in [ROOT / "run.sh", *(ROOT / "scripts").glob("*.sh")]:
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "source" not in line or "configs/" not in line or ".env" not in line:
                    continue
                if "configs/experiment.env" not in line:
                    failures.append(f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}")
        self.assertEqual([], failures)

    def test_official_launcher_python_fallback_is_one_valid_expansion(self):
        source = (
            ROOT / "scripts" / "run_official_smpl_full.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'PY="${PY:-${GENERATION_PYTHON:-${PYTHON_BIN:-python}}}"',
            source,
        )
        self.assertNotIn('PY="${\n', source)

    def test_config_variables_have_one_internal_owner(self):
        owners = {}
        pattern = re.compile(
            r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)="
        )
        for path in sorted((ROOT / "configs").glob("*.env")):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                match = pattern.match(line)
                if match is None or match.group(1).startswith("_"):
                    continue
                owners.setdefault(match.group(1), []).append(
                    f"{path.name}:{line_number}"
                )
        duplicates = {
            name: locations
            for name, locations in owners.items()
            if len(locations) > 1
        }
        self.assertEqual({}, duplicates)

    def test_motion_config_reads_canonical_pipeline_fps(self):
        source = (ROOT / "training" / "motion_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"MOTION_FPS": ("fps", float)', source)

    def test_pipeline_trains_motion_models_with_source_disjoint_validation(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        for command in ("train-refiner", "train-diffusion"):
            start = source.index(command)
            block = source[start : start + 350]
            self.assertIn('--db "$TRAIN_AESD"', block)
            self.assertIn('--val_db "$VAL_AESD"', block)

    def test_pipeline_uses_resume_only_motion_training_snapshots(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("REFINER_TRAINING_SNAPSHOT=", source)
        self.assertIn("DIFFUSION_TRAINING_SNAPSHOT=", source)
        self.assertIn('MOTION_TRAINING_AUTO_RESUME" == "1"', source)
        self.assertIn('--resume_snapshot "$REFINER_TRAINING_SNAPSHOT"', source)
        self.assertIn('--resume_snapshot "$DIFFUSION_TRAINING_SNAPSHOT"', source)
        self.assertIn(
            '--snapshot_every "$MOTION_TRAINING_SNAPSHOT_INTERVAL_STEPS"',
            source,
        )
        cleanup = source.index(
            "18. RETIRE TRAINING-ONLY RECOVERY SNAPSHOTS"
        )
        render = source.index("17. SCIENTIFIC FIXED-CAMERA RENDER")
        self.assertGreater(cleanup, render)
        cleanup_block = source[cleanup:]
        self.assertIn("retired_training_snapshots", cleanup_block)
        self.assertIn(
            'if [[ "$GENERATION_RETRAIN_REFINER" == "1" ]]',
            cleanup_block,
        )
        self.assertIn(
            'if [[ "$GENERATION_RETRAIN_DIFFUSION" == "1" ]]',
            cleanup_block,
        )

    def test_pipeline_uses_dependency_aware_scheduler_retraining(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'SCHEDULER_RETRAIN_SET="${GENERATION_RETRAIN_ROUTER}/'
            '${GENERATION_RETRAIN_DURATION}/${GENERATION_RETRAIN_PLANNER}"',
            source,
        )
        self.assertIn(
            'if [[ "$GENERATION_RETRAIN_PLANNER" != "1"',
            source,
        )
        self.assertIn(
            '"$GENERATION_RETRAIN_ROUTER" == "1"',
            source,
        )
        self.assertIn(
            '"$GENERATION_RETRAIN_DURATION" == "1"',
            source,
        )
        self.assertIn(
            "Retraining Router or Duration changes a Planner upstream checkpoint",
            source,
        )
        self.assertNotIn(
            "Router, Duration and Planner share one serialized Generation Index",
            source,
        )
        self.assertIn(
            'SCHEDULER_REBUILD_INDEX="$GENERATION_REBUILD_EVENT_DB"',
            source,
        )
        start = source.index(
            "6A. GENERATION-ALIGNED SCHEDULER INDEX LIFECYCLE"
        )
        end = source.index("export GENERATION_INDEX_JSON", start)
        block = source[start:end]
        self.assertIn('if [[ "$SCHEDULER_REBUILD_INDEX" == "1" ]]', block)
        self.assertNotIn("SCHEDULER_RETRAIN_ALL", source)
        self.assertIn("build_generation_index.py", block)
        self.assertIn(
            "Preserving the exact Scheduler Index bytes bound to existing checkpoints",
            block,
        )
        self.assertIn('require_file "$ALIGNED_INDEX_NPZ"', block)
        self.assertIn("INDEX_CANDIDATE_DIR=", block)
        self.assertIn("INDEX_ARCHIVE_DIR=", block)

    def test_pipeline_reuses_split_and_aesd_with_event_db(self):
        source = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'require_file "$CACHE_SPLIT_ROOT/source_split_manifest.json"',
            source,
        )
        self.assertIn(
            'require_file "$TRAIN_AESD" "existing train AESD Event-DB"',
            source,
        )
        self.assertIn(
            "Rebuilding Event-DB changes every learned-data contract",
            source,
        )

    def test_formal_music_semantics_use_only_librosa_and_project_router(self):
        profile = (ROOT / "configs" / "research.env").read_text(
            encoding="utf-8"
        )
        self.assertIn("export REQUIRE_LIBROSA_BACKEND=1", profile)
        self.assertNotIn("GENERATION_DEEP_MUSIC", profile)
        self.assertNotIn("GROUNDING_GROUNDER_ARCHITECTURE", profile)

        pipeline = (ROOT / "scripts" / "pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"${REQUIRE_LIBROSA_BACKEND:-0}" != "1"', pipeline)
        self.assertNotIn("GENERATION_DEEP_MUSIC", pipeline)
        self.assertNotIn("grounding.model", pipeline)

        self.assertIn('ROUTING_WEAK_OT_MAX_ITER:-5000', profile)
        self.assertIn('ROUTING_WEAK_OT_MAX_MARGINAL_ERROR:-0.0001', profile)
        self.assertIn('--teacher_max_marginal_error "$ROUTING_WEAK_OT_MAX_MARGINAL_ERROR"', pipeline)

        router = (ROOT / "training" / "temporal_music_router.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from training.weak_semantic_ot import",
            router,
        )
        self.assertIn("scientific_supervision_contract", router)
        contract = (ROOT / "scheduling" / "temporal_router_contract.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('TEMPORAL_ROUTER_SUPERVISION_SOURCE = "semantic_ot_teacher"', contract)
        self.assertNotIn("music_prior_ckpt", router)
        requirements = (ROOT / "requirements-core.txt").read_text(
            encoding="utf-8"
        ).lower()
        self.assertNotIn("laion-clap", requirements)

    def test_official_preflight_checks_training_and_final_render_dependencies(self):
        source = (ROOT / "evaluation" / "preflight_official_smpl.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("training/temporal_music_router.py", source)
        self.assertIn("ROUTING_FORMAL_ROUTER_ARCHITECTURE", source)
        self.assertIn('import librosa', source)
        self.assertIn('import pytorch3d', source)
        self.assertIn('shutil.which("ffmpeg")', source)
        self.assertIn('"formal_music_contract": music_contract', source)

    def test_optional_transition_model_receives_runtime_fps(self):
        source = (ROOT / "scheduling" / "whole_song_scheduler.py").read_text(
            encoding="utf-8"
        )
        transition_start = source.index("transition_bundle = load_optional_transition(")
        transition_block = source[transition_start : transition_start + 260]
        self.assertIn("fps=float(args.fps)", transition_block)
    def test_native_row_transition_models_have_explicit_layout_boundary(self):
        for relative in (
            "support/contact_inr.py",
            "training/boundary_dynamics.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("_project_native_motion", source)
            self.assertNotIn("project_motion_rotations_torch", source)

    def test_refiner_training_passes_multirate_config_to_corruption(self):
        source = (ROOT / "training" / "motion_models.py").read_text(
            encoding="utf-8"
        )
        start = source.index("def train_refiner(")
        end = source.index("class SinusoidalTimeEmbedding", start)
        block = source[start:end]
        self.assertIn("bad, seam = degrade_for_refiner(", block)
        self.assertIn("cfg=cfg,", block)
        self.assertIn(
            "finalize_contract=not gpu_preprocessing",
            block,
        )
        self.assertNotIn("degrade_for_refiner(clean)\n", block)

    def test_post_processing_reaudits_exact_returned_motion(self):
        anatomy = (ROOT / "routing" / "anatomy_heading_closed_loop.py").read_text(
            encoding="utf-8"
        )
        final_merge = (ROOT / "routing" / "global_path.py").read_text(
            encoding="utf-8"
        )
        for source in (anatomy, final_merge):
            self.assertIn('stage["final_audit"] = selected_audit', source)
            self.assertIn('stage["final_physical_gate"] = selected_gate', source)
            self.assertIn("ROUTING_SAFETY_FINAL_PHYSICAL_ROLLBACK", source)

    def test_final_generator_requires_complete_boundary_audit(self):
        source = (ROOT / "routing" / "boundary_closed_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_boundary_continuity(", source)
        self.assertIn('"BOUNDARY_REQUIRE_FINAL_BOUNDARY_GATE"', source)
        self.assertIn("expected_boundaries=max(", source)
        self.assertIn("if args.render_output and not required_failures", source)

    def test_formal_generator_emits_current_heading_and_stage_activity_contracts(self):
        generator = (ROOT / "routing" / "boundary_closed_loop.py").read_text(
            encoding="utf-8"
        )
        heading_audit = (ROOT / "evaluation" / "audit_heading.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"formal_boundary_aligned_heading_v1"', generator)
        self.assertIn("validate_formal_heading_contract(report)", heading_audit)
        self.assertNotIn("missing_event_heading_planner_report", heading_audit)
        for stage in ("retrieval", "refiner", "diffusion", "full_ik"):
            self.assertIn(f'"{stage}",', generator)


if __name__ == "__main__":
    unittest.main()
