#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V46.52 wrapper: run the preserved V46.51 event builder, then anatomy-gate it."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Sequence

from events import build_database as base
from events.filter_anatomy import filter_database


def _value(args, flag):
    try:
        i = args.index(flag)
        return args[i + 1]
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rc = int(base.main(args))
    if rc != 0:
        return rc
    out_dir = _value(args, "--out_db")
    if not out_dir:
        raise RuntimeError("V46.52 event wrapper requires --out_db")
    root = Path(out_dir)
    filter_database(
        root / "events.npz",
        root / "events_meta.json",
        root / "events.v46_52_anatomy.audit.json",
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
