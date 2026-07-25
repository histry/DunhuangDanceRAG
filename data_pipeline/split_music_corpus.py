#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create deterministic song-disjoint music manifests for semantic OT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from training.music_corpus import audio_sha256, discover_training_audio


SCHEMA = "dunhuang_music_song_split_v1"


def split_corpus(
    music_dirs: Sequence[Path],
    out_dir: Path,
    *,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    seed: int = 20260724,
):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must lie in (0,1)")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must lie in (0,1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be below one")
    paths = [Path(value).resolve() for value in discover_training_audio([str(x) for x in music_dirs])]
    if len(paths) < 3:
        raise RuntimeError("At least three songs are required for train/val/test")
    unique_by_hash = {}
    duplicate_paths = []
    for path in paths:
        digest = audio_sha256(path)
        if digest in unique_by_hash:
            duplicate_paths.append(
                {"sha256": digest, "kept": str(unique_by_hash[digest]), "duplicate": str(path)}
            )
            continue
        unique_by_hash[digest] = path
    rows = [
        {
            "song_uid": "aud_" + digest[:24],
            "audio_path": str(path),
            "sha256": digest,
        }
        for digest, path in unique_by_hash.items()
    ]
    if len(rows) < 3:
        raise RuntimeError("At least three unique songs are required for train/val/test")
    rows.sort(key=lambda row: row["song_uid"])
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(rows), dtype=np.int64)
    rng.shuffle(order)
    train_count = max(1, int(round(len(rows) * float(train_ratio))))
    validation_count = max(1, int(round(len(rows) * float(validation_ratio))))
    if train_count + validation_count >= len(rows):
        validation_count = 1
        train_count = len(rows) - 2
    split_indices = {
        "train": order[:train_count],
        "validation": order[train_count : train_count + validation_count],
        "test": order[train_count + validation_count :],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    identities = {}
    for split, indices in split_indices.items():
        songs = [rows[int(index)] for index in indices]
        identities[split] = {row["song_uid"] for row in songs}
        payload = {
            "schema": SCHEMA,
            "split": split,
            "seed": int(seed),
            "songs": songs,
            "num_songs": len(songs),
        }
        target = out_dir / f"music_{split}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        reports[split] = {"manifest": str(target), "num_songs": len(songs)}
    for left in identities:
        for right in identities:
            if left >= right:
                continue
            overlap = identities[left].intersection(identities[right])
            if overlap:
                raise AssertionError(f"Song identities overlap: {left}/{right}: {overlap}")
    report = {
        "schema": SCHEMA,
        "num_songs": len(rows),
        "duplicate_audio_files_removed": int(len(duplicate_paths)),
        "duplicate_audio_files": duplicate_paths,
        "train_ratio": float(train_ratio),
        "validation_ratio": float(validation_ratio),
        "test_ratio": float(1.0 - train_ratio - validation_ratio),
        "splits": reports,
        "song_disjoint": True,
        "ok": True,
    }
    (out_dir / "music_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--validation_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args(argv)
    report = split_corpus(
        [Path(value) for value in args.music_dirs],
        Path(args.out_dir),
        train_ratio=float(args.train_ratio),
        validation_ratio=float(args.validation_ratio),
        seed=int(args.seed),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
