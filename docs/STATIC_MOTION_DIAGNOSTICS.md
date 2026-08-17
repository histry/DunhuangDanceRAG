# Static-motion collapse diagnostics

The diagnostic branch adds an activity contract beside the existing immutable
physics, anatomy and severe-heading contracts. It does not relax those gates.

## Why a physically safe result can still collapse

The route objective historically optimized semantic association, event quality,
anatomy, transition risk, diversity and future reachability. None of those terms
requires a selected event to contain meaningful motion. In a low-resource safe
set, short `pose_motif` events can therefore dominate because they are easy to
stitch and rarely violate physical thresholds. The refiner and diffusion stages
edit only the seam mask, while anatomy/tangent guards may roll the edits back to
the retrieval reference. They cannot create whole-slot activity when the event
cores are already static.

## Added diagnostics

Every accepted motion now reports:

- mean, p95 and maximum joint angular velocity in rad/s;
- root travel, root speed and per-second travel;
- static-frame ratio and the longest static streak;
- per-slot activity and music-density targets;
- motion-density MAE/correlation;
- stage-wise activity for retrieval, refiner, diffusion and full IK;
- geodesic/root deltas between consecutive stages.

By default, four stage snapshots are written next to the final NPY:

```text
<stem>.activity_retrieval.npy
<stem>.activity_refiner.npy
<stem>.activity_diffusion.npy
<stem>.activity_full_ik.npy
```

The final sidecar is:

```text
<stem>.motion_activity.json
```

## Acceptance policy

For active music slots, candidate events receive an activity mismatch penalty.
A candidate is hard-rejected only when at least two independent collapse signs
are present (excessive static ratio, excessive static streak, insufficient
angular speed). Missing activity metadata never bypasses physical simulation;
the event file is measured directly with the canonical column-concatenated
Rot6D decoder.

The final whole-song gate adapts its limits to the scheduler's target activity.
A rejected run keeps the NPY and report for diagnosis, marks the report as
rejected, and exits non-zero rather than treating a physically safe static
motion as a valid result.

## Useful environment variables

```bash
MOTION_ACTIVITY_PREORDER_ENABLE=1
MOTION_ACTIVITY_PREORDER_SCAN_TOPK=96
MOTION_ACTIVITY_CANDIDATE_HARD_GATE=1
MOTION_ACTIVITY_FINAL_HARD_GATE=1
MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS=1

MOTION_ACTIVITY_STATIC_JOINT_RADPS=0.08
MOTION_ACTIVITY_STATIC_ROOT_MPS=0.015
MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_RATIO=0.84
MOTION_ACTIVITY_CANDIDATE_MAX_STATIC_SECONDS=1.75
MOTION_ACTIVITY_CANDIDATE_MIN_JOINT_RADPS=0.065
MOTION_ACTIVITY_FINAL_MAX_STATIC_RATIO=0.72
MOTION_ACTIVITY_FINAL_MAX_STATIC_SECONDS=4.0
MOTION_ACTIVITY_FINAL_MIN_JOINT_RADPS=0.045
```

## Standalone audit

```bash
python evaluation/motion_activity.py \
  --input outputs/.../br_hpr.motion.npy \
  --report outputs/.../br_hpr.report.json \
  --fps 30 \
  --out outputs/.../br_hpr.motion_activity.json \
  --fail-on-collapse
```

The command returns exit code `3` when the activity gate fails.
