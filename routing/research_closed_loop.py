#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Research entrypoint: current V46.53 stack + feasibility/activity contracts."""
from __future__ import annotations

from typing import Optional, Sequence

import routing.global_path as latest
from routing.activity_guard import install as install_activity_guard
from routing.feasibility_contract import install
from routing.semantic_ot_integration import install as install_semantic_ot


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Install the repository's current SO(3)/anatomy/Grounder/masked-inpainting
    # stack first. Feasibility then wraps candidate construction, and the final
    # activity guard observes the fully composed pipeline without relaxing any
    # immutable physical, anatomy or severe-heading gate.
    latest._install_v53_patches()
    install_semantic_ot(latest)
    install(latest)
    install_activity_guard(latest)
    return int(latest.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
