# Refiner V15.15f: bounded formal Adapter training baseline

V15.15f is the first split-safe bounded training entry point for the
observable-conditioned Adapter. It consumes the frozen V15.15e train and
validation banks as separate inputs. Their manifest content hash, composite
case identities and source-case identities must agree with a zero-overlap
split before the first update.

The baseline intentionally adds no optimizer geometry and creates no pseudo
teachers. It retains the V15.15d exact-radius normalization, case-isolated
fixed-Guard restoration, transaction/group-balanced sampling, ownership mask,
Retraction, Projector and unchanged 0.03 observability gate. Training is
limited to 200--500 steps and is partitioned into 10--20 step chunks. AdamW
state is resumed between chunks rather than silently restarted.

After each chunk, the train bank and validation bank receive independent full
closure audits. Validation uses the gate floor frozen into the current train
state; validation data never recalibrates the inference gate. Any failed
required case, identity control, scope audit, fixed Guard or Projector result
terminates the trial with scientific status 2. A checkpoint is written only
after every bounded audit passes. The checkpoint remains unpromoted and cannot
authorize replay, publication or video generation by itself.

Server execution uses:

```bash
EXPECTED_COMMIT=<full-main-sha> \
PY=/home/disk/lsm/conda_envs/edge/bin/python \
bash scripts/run_refiner_v15_15f_bounded_formal_adapter_server.sh
```

`TOTAL_STEPS` defaults to 300 and must remain in `[200, 500]`.
`AUDIT_EVERY` defaults to 20 and must remain in `[10, 20]`.

