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
        default=None,
    )

    # Historical CLI compatibility only.
    parser.add_argument(
        "--source_manifest",
        dest="legacy_source_manifest",
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--name_map",
        default=None,
    )

    parser.add_argument(
        "--out",
        required=True,
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

    raw_manifest = (
        args.smpl_manifest
        or args.legacy_source_manifest
    )

    if not raw_manifest:
        parser.error(
            "--smpl_manifest is required"
        )

    manifest_path = Path(
        raw_manifest
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
        "training/music_router.py",
        "training/duration_model.py",
        "training/whole_song_planner.py",
        "routing/closed_loop.py",
        "rendering/render_motion.py",
        "scripts/pipeline.sh",
        "configs/experiment.env",
    ]

    for rel in required_code:
        require_file(
            root / rel,
            rel,
            errors,
        )

    router_prior = Path(
        os.environ.get("ROUTING_SAFETY_MUSIC_ENCODER_PRIOR_CKPT")
        or os.environ.get("MUSIC_ROUTER_WEIGHT")
        or root / "assets/weights/music/router.pt"
    ).expanduser().resolve()
    require_file(
        router_prior,
        "project-trained Librosa Router music-encoder prior",
        errors,
    )

    music_contract = {
        "deep_music_features": os.environ.get(
            "GENERATION_DEEP_MUSIC_FEATURES", "0"
        ),
        "require_deep_music": os.environ.get(
            "GENERATION_REQUIRE_DEEP_MUSIC", "0"
        ),
        "feature_model": os.environ.get(
            "GENERATION_DEEP_MUSIC_MODEL", "librosa_12d"
        ),
        "semantic_ot_enable": os.environ.get("SEMANTIC_OT_ENABLE", "0"),
        "grounder_architecture": os.environ.get(
            "GROUNDING_GROUNDER_ARCHITECTURE", "legacy"
        ),
    }
    expected_music_contract = {
        "deep_music_features": "0",
        "require_deep_music": "0",
        "feature_model": "librosa_12d",
        "semantic_ot_enable": "0",
        "grounder_architecture": "legacy",
    }
    if music_contract != expected_music_contract:
        errors.append(
            "formal music contract must be Librosa 12D + project-trained "
            f"Router with no external pretrained model: {music_contract}"
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

        runtime[
            "split_counts_at_recording_groups"
        ] = exact_split_counts(
            max(
                3,
                len(recording_uids),
            ),
            float(
                os.environ.get(
                    "GENERATION_TRAIN_RATIO",
                    "0.67",
                )
            ),
            float(
                os.environ.get(
                    "GENERATION_VAL_RATIO",
                    "0.165",
                )
            ),
            float(
                os.environ.get(
                    "GENERATION_TEST_RATIO",
                    "0.165",
                )
            ),
        )

    except Exception as exc:
        errors.append(
            "split contract self-test failed: "
            f"{exc}"
        )

    report = {
        "schema": (
            "chang_e_official_smpl_preflight_v2"
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
        "music_router_prior": str(router_prior),
        "formal_music_contract": music_contract,
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
