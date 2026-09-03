# Final Refiner Evidence Synthesis

This is the terminal read-only scientific synthesis for the current Refiner
candidate-development protocol.

## Inputs

The stage consumes five explicit immutable inputs:

1. completed pre-RPA `refiner_joint_evidence_synthesis_v1`;
2. corrected RPA-LRTA v2 formal report;
3. RPA-LRTA v2 freeze manifest;
4. corrected RPA H/I direction report;
5. direction-correction freeze manifest.

It does not search for latest artifacts.

## No execution of the Refiner

The stage imports no PyTorch and performs no:

- model or checkpoint loading;
- forward pass;
- autograd;
- optimizer construction;
- metric recomputation;
- case recomputation;
- checkpoint selection;
- intervention search;
- architecture search.

Only standard-library JSON loading, SHA256 hashing, lineage checks and evidence
composition are allowed.

## Final scientific classification

```text
MULTIPLE_MANIPULABLE_MECHANISMS_WITHOUT_SUFFICIENT_SAFE_REFINER_CANDIDATE
```

This classification means:

- several mechanisms are empirically identifiable;
- role/direction effects are manipulable;
- corrected H confirms the single-recording direction mechanism;
- corrected I does not support a cross/28 direction gain;
- RPA-LRTA does not rescue single or cross/28 target gates;
- RPA-LRTA introduces endpoint and physical regressions;
- no current candidate qualifies for Pilot or production.

It does **not** claim one unique root cause, and does **not** claim that every
possible Refiner architecture is impossible.

## Final stop rule

```text
candidate_development_closed = true
new_architecture_search_authorized = false
new_metric_search_authorized = false
new_scale_search_authorized = false
new_width_search_authorized = false
new_direction_search_authorized = false
pilot_authorized = false
production_change_authorized = false
```

Next action:

```text
freeze_final_refiner_evidence_and_transition_to_manuscript_synthesis
```

## Outputs

A successful run creates exactly the scientific terminal artifacts:

```text
result/report.json
result/evidence_summary.md
result/freeze_manifest.json
```
