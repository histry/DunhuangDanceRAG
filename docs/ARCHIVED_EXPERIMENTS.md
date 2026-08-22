# Archived experiments

The local annotated Git tag
`archive/legacy-motion-pipeline-e8217222` points to the last pre-cleanup
snapshot.

It contains historical 72/BVH business logic, copied EDGE weights and indexes,
external/mixed Grounder experiments, old migration scripts, and fallback
routes. Those artifacts are historical results only. They are not accepted by
the main-branch runtime and are not fair baselines for SMPL14 because the data,
semantics, split, Router supervision, and evaluation protocol differ.

Recovery is read-only:

```bash
git show archive/legacy-motion-pipeline-e8217222
git worktree add ../DunhuangDanceRAG-legacy archive/legacy-motion-pipeline-e8217222
```

Do not merge the archived worktree back into the formal branch.
