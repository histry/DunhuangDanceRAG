# RPA-LRTA Direction Reporting Correction

## Scope

This is a read-only correction of the H/I direction-alignment reporting in the
frozen RPA-LRTA v2 formal experiment.

It does **not** reopen the method candidate. The frozen formal decision remains:

```text
RPA_LRTA_NOT_SUPPORTED
```

The correction does not train, tune, select, or modify any model, case, metric,
threshold, loss, decoder, gate, or production inference path.

## Why the correction is necessary

The original formal RPA evaluator recorded all direction-cosine medians as
`null`. Consequently H and I were mechanically `false`, but those booleans were
not scientifically interpretable as measured "no alignment gain".

The frozen A/B/F/G conditions already make the method-level rejection invariant
to H/I, so correcting H/I cannot advance RPA-LRTA to Pilot.

## Preserved scientific question

The original RPA evaluator compared its two raw geometric outputs directly with
one temporal gradient through `alignment_stats`. It did not support-mask the
primary H/I action.

Therefore the authoritative correction keeps:

```text
primary space = raw_all_geometry
```

Secondary descriptive views:

```text
raw_supported_geometry
soft_masked_supported_geometry
```

are reported but cannot redefine H/I.

## Correct gradient

The temporal gradient is computed at the exact frozen RCSP current point by
turning the already-computed raw RCSP output into a leaf:

```python
raw_leaf = raw_rcsp.detach().clone().requires_grad_(True)
```

The leaf then passes through the unchanged production decoder and unchanged
observable temporal objective:

```text
g_RCSP = d L_temporal / d raw_geometry
```

No model parameter participates in autograd.

Both actions use the same reference gradient:

```text
cos(RCSP raw geometry, -g_RCSP)
cos(RPA  raw geometry, -g_RCSP)
```

This preserves the original H/I question: whether RPA changes the action toward
a better local descent direction relative to the RCSP current point.

## Corrected conditions

H:

```text
single_recording:
defined RCSP and RPA cosine cases exist
AND median(RPA cosine) > median(RCSP cosine)
```

I:

```text
cross_event/28:
defined RCSP and RPA cosine cases exist
AND median(RPA cosine) > median(RCSP cosine)
```

No threshold, significance level, margin, or post-hoc tolerance is added.

## Frozen formal inputs

Formal RPA report SHA256:

```text
08fd36d5bd504a16cb5f18348358e8e236008e0758481e7c5372dddca0c6808e
```

Final RPA adapter SHA256:

```text
2b6a7ae7d08721bcff5b174403a7137ec7494c2f871a21ffc5a63bdc7be70110
```

Formal updates SHA256:

```text
aedcf96068976ead5988d055af248b067e641849214365ba9fea3fdee35f0a86
```

The supplied freeze manifest must reference these exact artifacts.

## Decision invariance

Even if corrected H and/or I become true:

```text
A = false
B = false
F = false
G = false
total temporal rescues = 1
```

Therefore:

- all A-I can never be true;
- `G=false` blocks the PARTIAL branch;
- one temporal rescue blocks the MECHANISM_ONLY branch.

The method-level result must remain:

```text
RPA_LRTA_NOT_SUPPORTED
```

## Final reporting classification

If both target direction statistics are defined and H/I remain false:

```text
RPA_DIRECTION_REPORTING_CORRECTED_NO_TARGET_ALIGNMENT_GAIN
```

If H or I becomes true:

```text
RPA_DIRECTION_MECHANISM_PRESENT_BUT_METHOD_REMAINS_UNSUPPORTED
```

If a target cohort remains direction-undefined:

```text
RPA_DIRECTION_REPORTING_REMAINS_UNRESOLVED
```

None of these classifications authorizes Pilot or additional architecture
search.
