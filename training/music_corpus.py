"""Leakage-safe discovery and identity helpers for Scheduler music data."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
FORBIDDEN_TRAINING_PARTS = {"test", "test_music_bank", "classical_eval"}


def audio_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_training_audio(directories: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw in directories:
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Music training directory does not exist: {root}")
        if any(part.lower() in FORBIDDEN_TRAINING_PARTS for part in root.parts):
            raise RuntimeError(f"Evaluation/test music cannot enter training: {root}")
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
            and not any(
                part.lower() in FORBIDDEN_TRAINING_PARTS
                for part in path.relative_to(root).parts
            )
        )
    unique_paths = sorted(
        {path.resolve() for path in paths}, key=lambda path: path.as_posix()
    )
    if not unique_paths:
        raise RuntimeError("No Scheduler training audio was found")

    # A corpus assembled from several folders can contain byte-identical copies
    # under different names.  Path-only de-duplication would let the same song
    # enter both sides of a song-disjoint validation split.  Keep the first
    # deterministic path for each content identity.
    unique_by_content: dict[str, Path] = {}
    for path in unique_paths:
        unique_by_content.setdefault(audio_sha256(path), path)
    return list(unique_by_content.values())


def assert_content_disjoint(
    training_audio: Sequence[str | Path],
    heldout_audio: Sequence[str | Path],
) -> dict[str, Any]:
    """Fail when a held-out target is byte-identical to any training song."""
    training_hashes = {audio_sha256(path): str(Path(path).resolve()) for path in training_audio}
    heldout_rows = []
    collisions = []
    for raw in heldout_audio:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Held-out audio does not exist: {path}")
        digest = audio_sha256(path)
        row = {"path": str(path), "sha256": digest}
        heldout_rows.append(row)
        if digest in training_hashes:
            collisions.append(
                {
                    **row,
                    "training_path": training_hashes[digest],
                }
            )
    if collisions:
        raise RuntimeError(
            "Held-out target audio is content-identical to Scheduler training audio: "
            f"{collisions}"
        )
    return {
        "schema": "music_content_disjoint_v1",
        "ok": True,
        "num_training_hashes": len(training_hashes),
        "heldout": heldout_rows,
        "collisions": [],
    }
