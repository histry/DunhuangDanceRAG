#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.53 Event-DB builder.

Execution order:
1. run the public V46.52 anatomy-gated builder;
2. append intrinsic SO(3) flow, body-part dynamics and W2 barycentric fields;
3. train the dual-branch grounder on the train split, or embed val/test with the
   existing train checkpoint.

The file is deliberately a small wrapper so the preserved V46.51 slicer and all
legacy V44/V45/V46 NPZ keys remain unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from events import anatomy_pipeline as v52
from events.intrinsic_geometry import augment_database
from grounding.model import embed_database, train_grounder


def _env_bool(name: str, default: bool) -> bool:
    return str(os.environ.get(name, "1" if default else "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _value(args: Sequence[str], flag: str) -> Optional[str]:
    try:
        idx = list(args).index(flag)
        return list(args)[idx + 1]
    except Exception:
        return None


def _split_name(path: Path) -> str:
    tokens = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "train" in tokens or name == "train" or "train" in name:
        return "train"
    if "val" in tokens or "validation" in tokens or "val" in name:
        return "val"
    if "test" in tokens or "test" in name:
        return "test"
    return "unknown"


def _checkpoint_for(out_dir: Path) -> Path:
    explicit = str(os.environ.get("V46_53_GROUNDER_CKPT", "")).strip()
    if explicit:
        return Path(explicit)
    out_root = str(os.environ.get("OUT_ROOT", "")).strip()
    architecture = str(
        os.environ.get("V46_53_GROUNDER_ARCHITECTURE", "legacy")
    ).strip().lower()
    checkpoint_name = (
        "v46_53_mixed_curvature_grounder.pt"
        if architecture == "mixed"
        else "v46_53_dual_branch_grounder.pt"
    )
    if out_root:
        return Path(out_root) / checkpoint_name
    # split DB layout is usually .../event_db_split/train.  Put the shared model
    # two levels above so val/test wrappers can resolve the same checkpoint.
    parent = out_dir.parent.parent if out_dir.parent.name.lower() in {"train", "val", "test"} else out_dir.parent
    return parent / checkpoint_name


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rc = int(v52.main(args))
    if rc != 0:
        return rc

    out_raw = _value(args, "--out_db")
    if not out_raw:
        raise RuntimeError("V46.53 event builder requires --out_db")
    out_dir = Path(out_raw)
    db_path = out_dir / "events.npz"
    if not db_path.is_file():
        raise FileNotFoundError(str(db_path))

    geometry_report = augment_database(
        db_path,
        out_dir / "events.v46_53_geometry.audit.json",
        # Event-DB canonical_fps is authoritative.  Reading an unrelated shell
        # default here previously allowed 60 FPS databases to be augmented as
        # if they were 30 FPS when this wrapper was invoked standalone.
        fps=None,
    )

    split = _split_name(out_dir)
    architecture = str(
        os.environ.get("V46_53_GROUNDER_ARCHITECTURE", "legacy")
    ).strip().lower()
    if architecture not in {"legacy", "mixed"}:
        raise RuntimeError(
            "V46_53_GROUNDER_ARCHITECTURE must be 'legacy' or 'mixed', "
            f"got {architecture!r}"
        )
    ckpt = _checkpoint_for(out_dir)
    grounding_report = {
        "enabled": False,
        "reason": "disabled",
        "checkpoint": str(ckpt),
        "split": split,
    }
    if _env_bool("V46_53_GROUNDER_ENABLE", True):
        if split == "train" and _env_bool("V46_53_TRAIN_GROUNDER_ON_BUILD", True):
            if architecture == "mixed":
                paired_path = str(
                    os.environ.get("V46_53_GROUNDER_PAIRED_DATASET", "")
                ).strip()
                if not paired_path:
                    raise RuntimeError(
                        "Mixed grounding requires "
                        "V46_53_GROUNDER_PAIRED_DATASET; the builder will not "
                        "replace real audio with semantic noise"
                    )
                from grounding.mixed_curvature import (
                    embed_database_mixed,
                    train_mixed_grounder,
                )

                training_report = train_mixed_grounder(
                    paired_dataset_path=Path(paired_path),
                    out_path=ckpt,
                    epochs=_env_int("V46_53_MIXED_EPOCHS", 120),
                    batch_size=_env_int("V46_53_GROUNDER_BATCH", 96),
                    seed=_env_int("V46_53_SEED", 20260724),
                )
                embedding_report = embed_database_mixed(db_path, ckpt)
                grounding_report = {
                    "schema": "v46_53_mixed_train_and_embed_v1",
                    "ok": bool(
                        training_report.get("ok", False)
                        and embedding_report.get("ok", False)
                    ),
                    "checkpoint": str(ckpt),
                    "split": split,
                    "training": training_report,
                    "embedding": embedding_report,
                    "train_db_embedded": True,
                }
            else:
                grounding_report = train_grounder(
                    db_path=db_path,
                    out_path=ckpt,
                    steps=_env_int("V46_53_GROUNDER_STEPS", 1400),
                    batch_size=_env_int("V46_53_GROUNDER_BATCH", 128),
                    seed=_env_int("V46_53_SEED", 20260717),
                )
        elif ckpt.is_file():
            if architecture == "mixed":
                from grounding.mixed_curvature import embed_database_mixed

                grounding_report = embed_database_mixed(db_path, ckpt)
            else:
                grounding_report = embed_database(db_path, ckpt)
        else:
            grounding_report = {
                "enabled": False,
                "reason": "train checkpoint not available yet",
                "checkpoint": str(ckpt),
                "split": split,
            }

    report = {
        "schema": "v46_53_hierarchical_intrinsic_event_db",
        "split": split,
        "grounder_architecture": architecture,
        "out_dir": str(out_dir),
        "geometry": geometry_report,
        "grounding": grounding_report,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("V46_53_")},
        "ok": True,
    }
    (out_dir / "events.v46_53.build.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "schema": report["schema"],
        "split": split,
        "events": geometry_report.get("num_events"),
        "geometry_dim": geometry_report.get("geometry_dim"),
        "grounder_checkpoint": str(ckpt),
        "grounder_enabled": bool(grounding_report.get("ok", grounding_report.get("enabled", False))),
        "ok": True,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
