# V15.14 weighted-DLS projected-candidate probe

V15.14 is a development-only offline feasibility probe. It starts from the
eight-subgroup unconstrained MGDA direction saved by a failed V15.13
diagnostic, constructs candidates at finite output tangent amplitudes, and
projects the candidate motion before applying the unchanged fixed exact Guard.

The projector fixes three implementation details:

- Ownership and the C2 seam taper enter the diagonal stiffness of the DLS
  normal equation. The solved update is never multiplied by a taper afterward.
- The linear system is solved with `torch.linalg.solve`; neither a
  pseudoinverse nor an explicit matrix inverse is used.
- Every candidate reports the residual before each IK iteration and
  `ik_residual_after_n_iters`, including a per-foot breakdown. The report also
  keeps exact-Guard blockers separate from IK convergence.

The probe reads only an unpublished diagnostic state and its TRAIN-only fixed
fit bank. It does not start training, use held-out `new_position` cases for an
update, promote a checkpoint, run a full replay, or generate a video. Physical,
fixed-support, fidelity, boundary, and observable 0.03 thresholds are inherited
unchanged from the source diagnostic's immutable Guard contract.

Run it on the server after checking out the supplied commit:

```bash
EXPECTED_COMMIT=<full-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_14_projected_candidate_probe_server.sh
```

Exit status 0 means at least one projected finite-amplitude candidate passed
the fixed exact Guard and produced a residual decrease above numeric audit
resolution. Exit status 2 is a completed scientific rejection. Any other exit
status is an execution failure.
