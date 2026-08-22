# Current formal architecture

The main branch contains one business path:

1. validate the authoritative 14-file Chang-E SMPL manifest;
2. convert explicit 165D pose layout to SMPL24/EDGE151 and build a clean source cache;
3. split recording groups before event slicing, then build Event-DBs with
   posture-aware Anatomy filtering and intrinsic physical endpoint geometry;
4. extract strict Librosa 12D phrase sequences from non-test music;
5. train CTSR-Weak Router, Duration model, and continuous Whole-Song Planner;
6. produce one final MSSD whose sibling candidates come from the same CTSR Router;
7. solve the global route with fail-closed Fisher-Rao Graph-SB;
8. perform boundary refinement, diffusion, lower-body IK, physical audits, and rendering.

Music controls phrase structure, timing, weak event compatibility, and route
probabilities. Theme names, props, and filenames never become local-action
truth. The continuous Planner has no categorical event head.

The final video contains one rendered skeleton. Formal single-person runs
accept only events with audited `solo_compatible=true`; gender is not treated
as dancer identity.

Historical BVH, external Grounder, copied EDGE checkpoint, and automatic
fallback paths are absent from main. Their last snapshot is recorded in
`docs/ARCHIVED_EXPERIMENTS.md`.
