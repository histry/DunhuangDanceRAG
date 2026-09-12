# Refiner V15.15g1: observable severity and Guard-aligned correction

V15.15g1 replaces the constant identity-objective activation test from
V15.15g. Before validation starts, it freezes a per-channel safe envelope from
the serialized observable conditions of train-split `identity_control`
samples. The envelope covers endpoint, temporal, root translation, root
velocity, yaw, support mismatch, seam coverage and relative phase-edge
density. The train/validation manifest hashes and source diagnostic must match.

At inference, every case is processed identically and only its observable
condition is read. A nonzero candidate is considered only when the Anchor lies
outside the frozen single-control envelope. `teacher_kind`, `single/cross` and
group labels are not read by activation or correction; they are attached after
selection for held-out evaluation.

The fixed-budget Euclidean and product-manifold corrections use the same
differentiable metric implementation as the existing stage audit. Their
objective includes endpoint and temporal scientific deficits plus
non-regression excess for these signed-margin fields:

- `repair_joint_jerk_mps3_p95_signed_margin`;
- `repair_joint_jerk_window_p95_max_mps3_signed_margin`;
- `repair_extremity_jerk_mps3_p95_signed_margin`;
- `repair_extremity_jerk_window_p95_max_mps3_signed_margin`;
- `boundary_jerk_signed_margin`.

Selection is fail-closed. A nonidentity candidate must have strict endpoint
and temporal scientific descent, no regression in any of the five Guard
proxies, exact `1e-4` owned-tangent RMS, zero tangent outside its ownership
scope and no numeric failure. Eligible candidates are ranked first by their
worst normalized Guard-proxy delta and then by the scientific correction
objective. When the Adapter itself is eligible, a corrected candidate may
replace it only by Pareto-dominating its five Guard-proxy deltas without
worsening the scientific objective. This retains an already safe Adapter
candidate when a lower scalar objective would worsen a Guard-aligned quantity.

The complete fixed Guard, Projector and `0.03` gate are unchanged and are not
used to choose a candidate. They remain the authoritative final acceptance
audit. Validation directions are never recycled as teachers.

Run the development probe on the server with:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15g1_observable_guard_aligned_server.sh
```

The command starts no Adapter training, pseudo-teacher recycling, promotion,
replay or video generation.
