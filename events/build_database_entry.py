#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean-project wrapper for source-aware Event-DB construction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from events.research_build_database import main

if __name__ == "__main__":
    raise SystemExit(main())
