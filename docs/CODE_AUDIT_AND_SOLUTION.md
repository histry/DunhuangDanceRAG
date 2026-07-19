# EDGE code audit and failure diagnosis

## Grounded failure

The previous formal run completed:

- 12/12 source retargeting;
- source gravity/anatomy gates;
- Event database construction and model training;
- 15,000 diffusion steps;
- a fresh 3,601-frame, 51-slot schedule.

Generation then stopped at slot 1 with:

`V46.50 heading contract exhausted candidates for slot 1`.

This is a feasibility-contract failure, not evidence that the diffusion model
needed more iterations.

## Code-level cause

The current stack performs semantic retrieval first and exposes only a limited
candidate preview.  Later layers independently apply stricter requirements:

- heading validity;
- Event anatomy validity and minimum anatomy quality;
- runtime core anatomy;
- a core-warp hard interval;
- multiscale tangent boundary rejection;
- observability rejection.

The scheduler/core-duration contract permits a wider warp range than the
runtime anatomy gate, and the first slot does not clamp its warp at all.
Consequently a semantically strong top-k set can contain no candidate that
survives the later physical transaction.

## Research solution

This patch keeps source and anatomy safety immutable and adds:

1. expanded candidate visibility;
2. current-DB performer filtering;
3. one canonical duration/warp feasibility function;
4. removal of stale scheduler Event identity;
5. exact frame-preserving slot splitting when duration coverage is empty;
6. three bounded feasibility tiers;
7. tangent softening only when base physical and anatomy checks pass;
8. explicit diagnostics in the final route report.

No globally unsafe candidate is accepted, and no complete Event core is
globally redrawn.
