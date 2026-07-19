#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the current V46.53 Event-DB, then attach performer metadata."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from events.build_pipeline import main as build_latest
from events.augment_performer_metadata import augment_events_npz


def _arg_value(args: Sequence[str], flag: str) -> Optional[str]:
    try:
        index = list(args).index(flag)
        return str(args[index + 1])
    except Exception:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    rc = int(build_latest(args) or 0)
    if rc != 0:
        return rc
    out_db = _arg_value(args, "--out_db")
    if not out_db:
        raise RuntimeError("--out_db is required")
    events = Path(out_db) / "events.npz"
    augment_events_npz(
        str(events),
        require_known=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
