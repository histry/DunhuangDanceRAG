#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research entrypoint: current Geometry-Aware Routing stack + feasibility-aware contract."""
from __future__ import annotations

from typing import Optional, Sequence

import routing.global_path as latest
from routing.feasibility_contract import install
from routing.semantic_ot_integration import install as install_semantic_ot


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Install the repository's current SO(3)/anatomy/Grounder/masked-inpainting
    # stack first.  The feasibility patch then wraps the resulting functions.
    latest._install_global_path_patches()
    install_semantic_ot(latest)
    install(latest)
    return int(latest.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
