#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility alias for the unified formal Event-DB entrypoint."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from events.build_database_entry import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
