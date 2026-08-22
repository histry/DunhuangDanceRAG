#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preflight for the formal Chang-E official-SMPL route."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline.chang_e_smpl_manifest import (
    load_manifest,
    manifest_sha256,
    validate_source,
)
from retargeting.official_smpl_source_preprocess import (
    discover_official_smpl_files,
    load_name_map,
)
from training.music_corpus import (
    assert_content_disjoint,
    discover_training_audio,
)


AUDIO_EXT = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}


def count_audio(path: Path) -> int:
    return sum(
        1
        for item in path.rglob("*")
        if (
            item.is_file()
            and item.suffix.lower()
            in AUDIO_EXT
        )
    )


def require_file(
    path: Path,
    label: str,
    errors: List[str],
) -> None:
    if (
        not path.is_file()
        or path.stat().st_size <= 0
    ):
        errors.append(
            f"missing {label}: {path}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=str(ROOT),
    )

    parser.add_argument(
        "--audio",
        required=True,
    )

    parser.add_argument(
        "--music_dir",
        required=True,
    )

    parser.add_argument(
        "--smpl_dir",
        required=True,
    )

    parser.add_argument(
        "--smpl_manifest",
        required=True,
    )

    parser.add_argument(
        "--name_map",
        default=None,
    )

    parser.add_argument(
        "--out",
        required=True,
    )
    parser.add_argument(
        "--router_provenance",
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    audio = Path(
        args.audio
    ).resolve()

    music_dir = Path(
        args.music_dir
    ).resolve()

    smpl_dir = Path(
        args.smpl_dir
    ).expanduser().resolve()

    manifest_path = Path(
        args.smpl_manifest
    ).expanduser().resolve()

    errors: List[str] = []
    warnings: List[str] = []

    require_file(
        audio,
        "input audio",
        errors,
    )

    if not music_dir.is_dir():
        errors.append(
            f"missing training music directory: "
            f"{music_dir}"
        )

    if (
        "test_music_bank"
        in str(music_dir)
        or "classical_eval"
        in str(music_dir)
    ):
        errors.append(
            "evaluation music must not enter training"
        )

    if not smpl_dir.is_dir():
        errors.append(
            f"missing official SMPL directory: "
            f"{smpl_dir}"
        )

    require_file(
        manifest_path,
        "official SMPL manifest",
        errors,
    )

    required_code = [
        "data_pipeline/chang_e_smpl_manifest.py",
        "retargeting/official_smpl_source_preprocess.py",
        "data_pipeline/split_sources.py",
        "events/build_database.py",
        "events/filter_anatomy.py",
        "training/motion_models.py",
        "training/temporal_music_router.py",
        "training/current_protocol_router_baseline.py",
        "training/weak_semantic_ot.py",
        "model/temporal_music_motion_router.py",
        "scheduling/temporal_router_contract.py",
        "training/duration_model.py",
        "training/whole_song_planner.py",
        "routing/boundary_closed_loop.py",
        "rendering/render_motion.py",
        "scripts/pipeline.sh",
        "scripts/run_official_smpl_loto_full.sh",
        "evaluation/validate_formal_route.py",
        "evaluation/audit_formal_single_person_db.py",
        "scripts/evaluate_current_protocol_baselines.py",
        "configs/experiment.env",
    ]

    for rel in required_code:
        require_file(
            root / rel,
            rel,
            errors,
        )

    music_contract = {
        "feature_model": "librosa_12d",
        "require_librosa_backend": os.environ.get(
            "REQUIRE_LIBROSA_BACKEND", "0"
        ),
        "router_architecture": os.environ.get(
            "ROUTING_FORMAL_ROUTER_ARCHITECTURE", ""
        ),
        "router_supervision_source": os.environ.get(
            "ROUTING_FORMAL_SUPERVISION_SOURCE", ""
        ),
        "freeze_music_encoder": os.environ.get(
            "ROUTING_SAFETY_FREEZE_MUSIC_ENCODER", "1"
        ),
    }
    expected_music_contract = {
        "feature_model": "librosa_12d",
        "require_librosa_backend": "1",
        "router_architecture": "ctsr_weak_temporal_v1",
        "router_supervision_source": "semantic_ot_teacher",
        "freeze_music_encoder": "0",
    }
    if music_contract != expected_music_contract:
        errors.append(
            "formal music contract must be Librosa 12D + scratch-trained "
            f"CTSR-Weak with no external pretrained model: {music_contract}"
        )
    graph_route_contract = {
        "solver": os.environ.get("GRAPH_ROUTE_SOLVER", ""),
    }
    if graph_route_contract != {"solver": "fisher_rao_graph_sb"}:
        errors.append(f"formal Graph-SB contract is not fail-closed: {graph_route_contract}")
    performer_contract = {
        "group": os.environ.get("PERFORMER_GROUP", "auto"),
        "identity_mode": os.environ.get("PERFORMER_IDENTITY_MODE", "group"),
        "require_solo_compatible": os.environ.get(
            "PERFORMER_REQUIRE_SOLO_COMPATIBLE", "0"
        ),
    }
    if performer_contract["require_solo_compatible"] != "1":
        errors.append(
            f"formal one-body routing does not exclude unreviewed pair tracks: {performer_contract}"
        )

    discovered = (
        discover_official_smpl_files(
            smpl_dir
        )
        if smpl_dir.is_dir()
        else []
    )

    name_map = load_name_map(
        (
            Path(args.name_map)
            .expanduser()
            .resolve()
        )
        if args.name_map
        else None
    )

    source_rows: List[
        Dict[str, Any]
    ] = []

    source_ids: set[str] = set()
    recording_uids: set[str] = set()

    manifest = None

    try:
        manifest = load_manifest(
            manifest_path,
            required=True,
        )
    except Exception as exc:
        errors.append(
            "official SMPL manifest invalid: "
            f"{exc}"
        )

    if manifest is not None:
        manifest_names = {
            Path(
                str(row["file"])
            ).name
            for row in manifest["sources"]
        }

        discovered_names = {
            path.name
            for path in discovered
        }

        missing = sorted(
            manifest_names
            - discovered_names
        )

        extra = sorted(
            discovered_names
            - manifest_names
        )

        if missing:
            errors.append(
                f"SMPL files missing: {missing}"
            )

        if extra:
            errors.append(
                f"unexpected SMPL files: {extra}"
            )

        for source in discovered:
            try:
                explicit = (
                    name_map.get(source.name)
                    or name_map.get(
                        str(
                            source.relative_to(
                                smpl_dir
                            )
                        )
                    )
                )

                contract = validate_source(
                    source,
                    manifest=manifest,
                    manifest_file=manifest_path,
                    explicit_source_id=explicit,
                    verify_hash=True,
                )

                source_id = str(
                    contract["source_id"]
                )

                if source_id in source_ids:
                    raise RuntimeError(
                        "duplicate source_id="
                        f"{source_id}"
                    )

                source_ids.add(source_id)

                recording_uids.add(
                    str(
                        contract[
                            "recording_uid"
                        ]
                    )
                )

                source_rows.append({
                    "source": str(source),
                    "source_id": source_id,
                    "recording_uid": (
                        contract[
                            "recording_uid"
                        ]
                    ),
                    "sequence_id": contract["sequence_id"],
                    "dancer_id": contract["dancer_id"],
                    "dancer_id_status": contract["dancer_id_status"],
                    "performer_group": (
                        contract[
                            "performer_group"
                        ]
                    ),
                    "recording_performer_count": contract[
                        "recording_performer_count"
                    ],
                    "solo_compatibility": contract["solo_compatibility"],
                    "solo_compatible": contract["solo_compatible"],
                    "solo_review_status": contract["solo_review_status"],
                    "dance_category": (
                        contract[
                            "dance_category"
                        ]
                    ),
                    "candidate_dance_category": contract.get(
                        "candidate_dance_category"
                    ),
                    "theme_label_status": contract["theme_label_status"],
                    "source_context": contract["source_context"],
                    "coordinate_system": contract["coordinate_system"],
                    "translation_units": contract["translation_units"],
                    "pose_layout": contract["pose_layout"],
                    "frames": contract[
                        "frames"
                    ],
                    "source_fps": contract[
                        "source_fps"
                    ],
                    "duration_seconds": (
                        contract[
                            "duration_seconds"
                        ]
                    ),
                    "embedded_fps": (
                        contract[
                            "embedded_fps"
                        ]
                    ),
                    "sha256": contract[
                        "sha256"
                    ],
                })

                if (
                    contract[
                        "embedded_fps_matches_manifest"
                    ]
                    is False
                ):
                    warnings.append(
                        f"{source_id}: embedded FPS "
                        f"{contract['embedded_fps']} "
                        "differs from authoritative "
                        "manifest source_fps "
                        f"{contract['source_fps']}"
                    )

            except Exception as exc:
                errors.append(
                    f"{source.name}: {exc}"
                )

    min_sources = int(
        float(
            os.environ.get(
                "RETARGET_MIN_OK_SOURCES",
                "8",
            )
        )
    )

    if len(source_ids) < min_sources:
        errors.append(
            f"official SMPL source count="
            f"{len(source_ids)} "
            f"< minimum={min_sources}"
        )

    if len(recording_uids) < 3:
        errors.append(
            f"recording groups="
            f"{len(recording_uids)} "
            "< minimum split requirement=3"
        )

    music_count = (
        count_audio(music_dir)
        if music_dir.is_dir()
        else 0
    )
    content_disjoint_report: Dict[str, Any] = {
        "schema": "music_content_disjoint_v1",
        "ok": False,
    }
    if music_dir.is_dir() and audio.is_file():
        try:
            training_audio = discover_training_audio([music_dir])
            if len(training_audio) != music_count:
                errors.append(
                    "training music contains byte-identical duplicates: "
                    f"files={music_count}, unique_content={len(training_audio)}"
                )
            content_disjoint_report = assert_content_disjoint(
                training_audio,
                [audio],
            )
        except Exception as exc:
            errors.append(f"training/target audio content-disjoint check failed: {exc}")

    expected_music = int(
        float(
            os.environ.get(
                "RETARGET_CLEAN_EXPECTED_TRAIN_MUSIC",
                "788",
            )
        )
    )

    if (
        expected_music > 0
        and music_count != expected_music
    ):
        errors.append(
            f"training music count="
            f"{music_count}; "
            f"expected={expected_music}"
        )

    runtime: Dict[str, Any] = {}

    try:
        import torch

        runtime["torch"] = (
            torch.__version__
        )

        runtime["cuda_available"] = bool(
            torch.cuda.is_available()
        )

        runtime["cuda_device"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        if not torch.cuda.is_available():
            errors.append(
                "CUDA is unavailable"
            )

    except Exception as exc:
        errors.append(
            f"PyTorch import failed: {exc}"
        )

    try:
        import librosa

        runtime["librosa"] = str(librosa.__version__)
    except Exception as exc:
        errors.append(f"Librosa import failed: {exc}")

    try:
        import pytorch3d

        runtime["pytorch3d"] = str(
            getattr(pytorch3d, "__version__", "installed")
        )
    except Exception as exc:
        errors.append(f"PyTorch3D import failed: {exc}")

    ffmpeg = shutil.which("ffmpeg")
    runtime["ffmpeg"] = ffmpeg
    if not ffmpeg:
        errors.append("ffmpeg executable is unavailable on PATH")

    try:
        from data_pipeline.split_sources import (
            exact_split_counts,
        )

        split_recording_uids = {
            str(row["recording_uid"])
            for row in source_rows
            if (
                performer_contract["require_solo_compatible"] != "1"
                or bool(row.get("solo_compatible", False))
            )
        }
        runtime["formal_split_recording_groups"] = len(split_recording_uids)

        runtime[
            "split_counts_at_recording_groups"
        ] = exact_split_counts(
            max(
                3,
                len(split_recording_uids),
            ),
            float(
                os.environ.get(
                    "GENERATION_TRAIN_RATIO",
                    "0.50",
                )
            ),
            float(
                os.environ.get(
                    "GENERATION_VAL_RATIO",
                    "0.25",
                )
            ),
            float(
                os.environ.get(
                    "GENERATION_TEST_RATIO",
                    "0.25",
                )
            ),
        )
        if (
            str(
                os.environ.get(
                    "GENERATION_SPLIT_PROTOCOL",
                    "category_covered_source_disjoint",
                )
            )
            == "category_covered_source_disjoint"
            and any(
                int(
                    runtime["split_counts_at_recording_groups"][split]
                )
                < 2
                for split in ("val", "test")
            )
        ):
            errors.append(
                "Formal ordinary source-disjoint evaluation requires at least "
                "two recording groups in both validation and test"
            )

    except Exception as exc:
        errors.append(
            "split contract self-test failed: "
            f"{exc}"
        )

    report = {
        "schema": (
            "chang_e_official_smpl_preflight_v3_formal_fail_closed"
        ),
        "formal_motion_source": (
            "chang_e_official_smpl"
        ),
        "root": str(root),
        "audio": str(audio),
        "music_dir": str(music_dir),
        "smpl_dir": str(smpl_dir),
        "smpl_manifest": str(
            manifest_path
        ),
        "smpl_manifest_sha256": (
            manifest_sha256(
                manifest_path
            )
            if manifest_path.is_file()
            else None
        ),
        "official_smpl_files": [
            str(path)
            for path in discovered
        ],
        "num_official_smpl_files": (
            len(discovered)
        ),
        "num_source_ids": (
            len(source_ids)
        ),
        "source_rows": source_rows,
        "num_recording_groups": (
            len(recording_uids)
        ),
        "coordinate_system": (
            manifest.get("coordinate_system") if manifest else None
        ),
        "translation_units": (
            manifest.get("translation_units") if manifest else None
        ),
        "pose_layout": manifest.get("pose_layout") if manifest else None,
        "unique_recording_duration_minutes": (
            manifest.get("unique_recording_duration_minutes") if manifest else None
        ),
        "recording_uids": sorted(
            recording_uids
        ),
        "training_music_count": (
            music_count
        ),
        "expected_training_music_count": (
            expected_music
        ),
        "music_router_initialization": "from_scratch_no_checkpoint_prior",
        "training_target_audio_content_disjoint": content_disjoint_report,
        "formal_music_contract": music_contract,
        "formal_graph_route_contract": graph_route_contract,
        "formal_performer_contract": performer_contract,
        "scientific_limitations": {
            "same_dancer_claim_supported": False,
            "dancer_disjoint_claim_supported": False,
            "reason": "global dancer_id is not verified in the released filenames/metadata",
            "pending_theme_sources": sorted(
                row["source_id"]
                for row in source_rows
                if row.get("theme_label_status") == "pending_official_confirmation"
            ),
            "excluded_unreviewed_multi_performer_sources": sorted(
                row["source_id"]
                for row in source_rows
                if not bool(row.get("solo_compatible", False))
            ),
            "themes_removed_from_formal_single_person_training": sorted(
                {
                    str(row["dance_category"])
                    for row in source_rows
                    if not bool(row.get("solo_compatible", False))
                }
            ),
            "leave_one_theme_out_launcher": str(
                root / "scripts/run_official_smpl_loto_full.sh"
            ),
            "ordinary_source_disjoint_and_loto_reported_separately": True,
        },
        "runtime": runtime,
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
