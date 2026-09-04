# Refiner Protocol V2 Preregistration

**Protocol name:** RC-FTP Refiner V2  
**Expanded name:** Role-Conditioned Feasible Tangent Projection Refiner  
**Protocol status:** PRE-REGISTERED / FROZEN BEFORE IMPLEMENTATION  
**Frozen scientific parent commit:** `65bd21c08af1d12b86030485f1735401bb30359e`  
**Parent evidence conclusion:** `MULTIPLE_MANIPULABLE_MECHANISMS_WITHOUT_SUFFICIENT_SAFE_REFINER_CANDIDATE`

---

## 1. Purpose and non-negotiable scope

Protocol V2 is a new Refiner research protocol created after Protocol V1 was scientifically closed.

Protocol V1 established that multiple failure mechanisms are real and some are causally manipulable, but no tested candidate was simultaneously effective and safe. In particular:

- role conditioning is a real but insufficient mechanism;
- width conditions intervention effectiveness but is not established as a unique root cause;
- single-recording action-direction alignment is causally supported but insufficient;
- RPA-LRTA produces a mechanistic signal but does not provide a sufficiently safe method candidate;
- method success must not be inferred from continuous deficit reduction alone.

Protocol V2 therefore does **not** continue RCSP, BCTR, SECDR, or RPA-LRTA search. It starts a new, independently preregistered method family whose central hypothesis is:

> Boundary refinement is better formulated as a role-conditioned constrained-action problem than as an unconstrained weighted-sum correction problem.

This preregistration is frozen **before V2 implementation and training**. Any change to the model, constraints, thresholds, data strata, seeds, training budget, blind-test set, or decision rules after implementation begins invalidates the present preregistration and requires a separately named protocol revision.

---

## 2. Frozen V1 components reused without scientific reinterpretation

The following V1 components are reused as fixed infrastructure and are not V2 contributions:

- EDGE-151D motion representation;
- product-manifold motion geometry;
- 79D Refiner output:
  - 4 contact residual logits,
  - 3 root-translation tangent residuals,
  - 24 × 3 local SO(3) tangent residuals;
- boundary observable protocol: `observable_duration_c2_bridge_v2`;
- existing boundary phase/core, external anchor-relative pose, and endpoint-velocity features;
- existing root-relative FK velocity / acceleration / jerk / duration features;
- existing joint-risk masks;
- existing soft-confidence decoder;
- existing supported-residual smoothing;
- existing inward taper;
- existing root and rotation caps;
- existing product-manifold retraction;
- existing true-chain-rule decoder gradient;
- existing physical-quality registry;
- existing stage-relative physical acceptance semantics;
- existing final whole-song physical fail-closed gate;
- existing transactional optimizer / Armijo rollback infrastructure.

The V2 implementation must not claim novelty for phase, duration, FK dynamics, product-manifold decoding, or transactional rollback because these already exist in the frozen parent implementation.

---

## 3. V2 primary hypothesis

Let the observed bridge motion be \(x\), the music/event condition be \(c\), the seam mask be \(S\), the joint-risk mask be \(M\), and the explicit boundary role be \(r\).

The proposal network predicts a raw tangent action

\[
u_0 = F_\theta(x,c,S,M,r).
\]

Instead of decoding \(u_0\) directly as the final Refiner action, V2 computes a constrained action

\[
u^* = \operatorname{FTP}(u_0; g_e,g_t,g_j,g_s,g_r),
\]

and then decodes

\[
x^* = D(x,u^*),
\]

where \(D\) is the unchanged true-chain-rule V1 decoder.

The central hypothesis is:

> Explicit projection of the neural proposal toward the joint feasible region of endpoint, temporal, jerk, support/contact, and root-vertical constraints will yield a safer and more effective Refiner than direct weighted-sum action prediction.

---

## 4. Model architecture

### 4.1 Shared V1 backbone

The V2 proposal network retains the V1 temporal backbone:

- hidden width: `256`;
- temporal convolution kernel size: `5`;
- dilations: `(1, 2, 5)`;
- receptive field: `33 frames`;
- normalization: existing framewise channel normalization;
- activation: SiLU;
- output dimension: `79`.

The existing observable input features remain unchanged.

### 4.2 New explicit role condition

V2 adds one explicit learned role embedding:

\[
r \in \{\texttt{single\_recording},\texttt{cross\_event}\}.
\]

The embedding dimension is fixed to:

```text
ROLE_EMBED_DIM = 4
```

The role embedding is repeated across the temporal axis and concatenated to the existing V1 input features before the input projection.

### 4.3 Forbidden architecture additions

The formal V2 method must **not** add:

- categorical width-10 / width-28 heads;
- anatomy-specific ROOT/BODY/EXTREMITY heads;
- RPA-LRTA adapters;
- SECDR direction-rotation modules;
- additional attention blocks;
- Transformer replacement backbones;
- learned constraint thresholds;
- learned publication gates.

Duration remains a continuous existing feature. Width is used for stratified sampling and reporting, not as a discrete output head.

---

## 5. Feasible Tangent Projection (FTP)

### 5.1 Proposal

\[
u_0 = F_\theta(x,c,S,M,r).
\]

### 5.2 True decoder path

Every constraint gradient must be computed through the complete unchanged decoder:

```text
raw tangent
→ soft confidence
→ supported residual smoothing
→ inward taper
→ root/rotation cap
→ product-manifold retraction
→ FK / physical observable
→ constraint
```

No binary-support surrogate gradient is permitted.

### 5.3 Local projection

At projection iterate \(u_k\), each active constraint is locally linearized:

\[
g_i(u_k + \Delta u)
\approx g_i(u_k)+\nabla_u g_i(u_k)^T\Delta u.
\]

The projection solves

\[
\min_{\Delta u}
\frac{1}{2}\|W\Delta u\|_2^2 + \rho(s_e+s_t),
\]

subject to the five preregistered constraints below.

Root and rotation tangent coordinates are normalized using the existing decoder caps:

\[
\tilde u_{\text{root}} = u_{\text{root}}/0.08,
\]

\[
\tilde u_{\text{rot}} = u_{\text{rot}}/0.35.
\]

No alternative tangent metric will be searched in Protocol V2.

### 5.4 Projection iterations

```text
FTP_ITERATIONS = 2
```

No adaptive iteration-count search is permitted.

If the action remains infeasible after two projection iterations, the case is marked infeasible for the V2 proposal. Existing runtime fallback / rollback semantics remain authoritative.

---

## 6. The five formal constraints

### Constraint 1 — Endpoint repair

Let \(E(x)\) denote the existing observable endpoint velocity jump. The frozen endpoint repair target is:

\[
\gamma_e=0.03.
\]

Define

\[
g_e(x^*)=
\frac{E(x^*)-(1-\gamma_e)E(x)}{\max(E(x),\epsilon)}.
\]

Feasibility requires

\[
g_e\le0.
\]

Training may use nonnegative endpoint slack \(s_e\), but final scientific evaluation requires zero slack.

### Constraint 2 — Temporal repair

Let the existing observable temporal energy be

\[
T(x)=\frac{A_{\rm seam}(x)}{10}+\frac{J_{\rm seam}(x)}{1000}.
\]

The frozen temporal repair target is:

\[
\gamma_t=0.03.
\]

Define

\[
g_t(x^*)=
\frac{T(x^*)-(1-\gamma_t)T(x)}{\max(T(x),\epsilon)}.
\]

Feasibility requires

\[
g_t\le0.
\]

Training may use nonnegative temporal slack \(s_t\), but final scientific evaluation requires zero slack.

### Constraint 3 — Jerk non-regression

The existing observable jerk rule is retained:

\[
J(x^*)\le1.02J(x)+\epsilon.
\]

Define

\[
g_j(x^*)=
\frac{J(x^*)-1.02J(x)-\epsilon}{\max(J(x),1)}.
\]

Feasibility requires

\[
g_j\le0.
\]

This is a hard constraint. No jerk slack is permitted.

### Constraint 4 — Support/contact safety

Support/contact feasibility is defined only by the frozen stage-relative physical registry.

The formal V2 implementation must reuse:

```text
PhysicalQualityLimits
StageAcceptancePolicy
physical_metric_specs()
```

for support/contact metrics including, as applicable:

- foot skate p95 / max;
- foot-support drift p95 / max;
- contact height;
- penetration.

The V2 method may implement differentiable surrogates, but the **authoritative decision** is the unchanged independent stage-relative audit.

No V2-specific support/contact thresholds may be introduced or tuned.

This is a hard constraint. No support/contact slack is permitted.

### Constraint 5 — Root-vertical safety

Root-vertical feasibility is defined by the existing frozen stage-relative physical registry, including:

- root-Y robust range;
- root vertical speed p95;
- root vertical speed max.

The V2 differentiable surrogate must use the frozen stage-relative budgets. The independent audit remains authoritative.

This is a hard constraint. No root-vertical slack is permitted.

---

## 7. Clean-identity protection

Clean identity is evaluated on a separate authentic-clean branch.

For clean input \(x_{\rm clean}\):

\[
D(x_{\rm clean},F_\theta(x_{\rm clean}))\approx x_{\rm clean}.
\]

The existing thresholds are frozen:

```text
clean identity product-log L1 <= 0.005
clean identity contact L1     <= 0.05
```

Formal clean-identity pass rate:

```text
>= 0.75
```

No threshold tuning is permitted.

---

## 8. Reference-fidelity protection

The existing reference-fidelity limits are frozen:

```text
FK p95 <= 0.15 m
FK max <= 1.00 m
product-log L1 <= 0.03
```

These values may not be relaxed after training begins.

---

## 9. Training data protocol

### 9.1 Source-disjoint split

All splits must be performed at `source_uid` level:

```text
TRAIN      70%
DEV        15%
BLIND TEST 15%
```

No recording, take, or source UID may appear in more than one split.

The exact source-UID lists and SHA256 hashes must be generated and frozen before the first V2 optimizer step.

### 9.2 Role-balanced training

```text
50% single_recording
50% cross_event
```

Cross-event cases are formal training data in V2 rather than validation-only cases.

### 9.3 Width-stratified batch

```text
BATCH_SIZE = 64
```

Each formal batch is:

```text
 8  single_recording width=10
 8  single_recording width=28
 8  cross_event      width=10
 8  cross_event      width=28
16  single_recording continuous width in [11,27]
16  cross_event      continuous width in [11,27]
-----------------------------------------------
64 total
```

Width 10 and 28 are fixed stress strata. Intermediate widths remain continuous and are not represented by categorical model heads.

### 9.4 Repair branch hidden-clean prohibition

The formal repair branch must not use hidden clean interior information.

Allowed repair supervision:

- observed bridge;
- external anchors;
- seam;
- role;
- current event/music condition;
- observable endpoint metrics;
- observable temporal metrics;
- stage-relative physical safety;
- minimum-edit objective.

Hidden-clean interior reconstruction may be used only as a clearly labelled diagnostic and must not affect formal V2 repair optimization or publication acceptance.

---

## 10. Formal training objective

\[
L_{\rm V2}
=
L_{\rm proposal}
+L_{\rm feasibility}
+\beta L_{\rm identity}
+\eta L_{\rm edit}.
\]

### 10.1 Proposal-to-projection distillation

\[
L_{\rm proposal}=\|u_0-\operatorname{sg}(u^*)\|_1.
\]

### 10.2 Residual feasibility loss

\[
L_{\rm feasibility}=\sum_i\operatorname{ReLU}(g_i(u^*)).
\]

The independent hard-constraint audit remains authoritative even if the surrogate loss is zero.

### 10.3 Minimum-edit objective

\[
L_{\rm edit}
=
\frac{1}{|\Omega|}
\sum_{t\in\Omega}
\|\log_{x_t}(x_t^*)\|_1.
\]

### 10.4 Clean identity

The clean branch uses the frozen clean-identity protection contract.

### 10.5 Loss-weight rule

The implementation must use one fixed set of \(\beta\), \(\eta\), and \(\rho\) across all three formal seeds.

These values must be written to the implementation manifest **before the first optimizer step** and may not be changed without declaring a new protocol revision.

No loss-weight sweep is authorized inside Protocol V2.

---

## 11. Optimizer and training budget

The existing transactional optimizer infrastructure is retained:

```text
full_cycle_feasibility_guard_armijo_v7
```

Formal training budget:

```text
8000 attempted optimizer steps per seed
```

This means attempted steps, not accepted steps.

No early stopping and no extension beyond 8000 attempted steps is permitted.

Validation may run at:

```text
1000, 2000, ..., 8000
```

Intermediate validation is diagnostic only.

Formal checkpoint:

```text
step 8000
```

There is no best-checkpoint selection.

---

## 12. Fixed random seeds

Exactly three formal seeds are preregistered:

```text
42
3407
8803
```

No additional seed may be added to rescue an unfavorable result and no seed may be dropped.

All three seeds use identical:

- source splits;
- blind-test case IDs;
- architecture;
- constraints;
- loss weights;
- optimizer configuration;
- training budget.

---

## 13. Blind test

### 13.1 Old Final64 exclusion

The Protocol V1 Final64 has already influenced diagnosis and V2 design.

Therefore:

```text
Protocol V1 Final64 MUST NOT be called the V2 blind test.
```

It may be reported only as historical development evidence.

### 13.2 New blind-test size

```text
128 cases
```

### 13.3 Eight fixed strata

| Stratum | Position status | Role | Width | N |
|---|---|---|---:|---:|
| S1 | seen_position | single_recording | 10 | 16 |
| S2 | seen_position | single_recording | 28 | 16 |
| S3 | seen_position | cross_event | 10 | 16 |
| S4 | seen_position | cross_event | 28 | 16 |
| S5 | new_position | single_recording | 10 | 16 |
| S6 | new_position | single_recording | 28 | 16 |
| S7 | new_position | cross_event | 10 | 16 |
| S8 | new_position | cross_event | 28 | 16 |

`seen_position` does **not** permit source leakage. It means only that the descriptor/position category is represented during training; all blind-test source UIDs remain source-disjoint.

`new_position` follows the existing V1 position-generalization semantics and also remains source-disjoint.

If the dataset cannot populate all eight strata with 16 valid source-disjoint cases, the blind test is considered **not constructible under this preregistration**. Cases may not be silently substituted across strata.

### 13.4 Blindness

Before the first V2 optimizer step:

- all 128 blind case IDs are fixed;
- case ordering is fixed;
- case strata are fixed;
- the case-list SHA256 is frozen.

No blind case may be inspected for method tuning.

---

## 14. Formal per-case acceptance

A case is a formal V2 rescue only if:

\[
G_{\rm joint}
=
G_{\rm endpoint}
\land
G_{\rm temporal}
\land
G_{\rm physical}
\land
G_{\rm fidelity}.
\]

Continuous deficit reduction without this conjunction is **not** a formal rescue.

---

## 15. Primary efficacy gates

For one seed to pass efficacy:

### 15.1 Overall joint gate

```text
>= 96 / 128
```

Equivalent to at least 75%.

### 15.2 Every-stratum joint gate

Each of the eight strata must satisfy:

```text
>= 12 / 16
```

### 15.3 Endpoint gate

Each of the eight strata must satisfy:

```text
>= 12 / 16
```

endpoint passes.

### 15.4 Temporal gate

Each of the eight strata must satisfy:

```text
>= 12 / 16
```

temporal passes.

No poor critical subgroup may be hidden by overall averaging.

---

## 16. Primary safety gates

Safety is fail-closed.

Across **all three seeds**:

- no blind case may introduce a new hard physical regression under the frozen stage-relative physical contract;
- no nonfinite motion is permitted;
- no invalid boundary case may be silently counted as passing;
- clean-identity pass rate must be at least 75%;
- reference-fidelity limits must remain satisfied;
- runtime rollback / fallback must remain available;
- no rejected validation checkpoint may be presented as a formal publishable checkpoint.

A method with improved temporal efficacy but a new hard physical regression is rejected.

---

## 17. Three-seed method-level rule

Method-level efficacy support requires:

```text
at least 2 of 3 seeds
```

to pass **all** efficacy gates in Section 15.

Method-level safety support requires:

```text
all 3 of 3 seeds
```

to pass all safety gates in Section 16.

All three seeds, including the worst seed, must be reported.

No best-seed result is permitted as the primary result.

---

## 18. Canonical checkpoint rule

Seed `42` is the preregistered canonical checkpoint for any later whole-song pilot.

It may be used for a formal pilot only if seed 42 itself passes:

- all efficacy gates;
- all safety gates;
- formal checkpoint publication validation.

If the method-level 2-of-3 efficacy rule passes but seed 42 does not, the method may be reported as scientifically supported, but **no formal pilot checkpoint is authorized under this preregistration**.

A separately preregistered pilot-checkpoint protocol would then be required.

---

## 19. Formal baselines and ablations

The blind test evaluates exactly these variants:

### B0 — Bridge only

```text
Bridge / transition output
No learned Refiner action
```

### B1 — Frozen V1 Refiner

Frozen V1 `ProductManifoldTemporalRefiner`.

### B2 — V2 without FTP

```text
V2 backbone
+ explicit role
- feasible tangent projection
```

### B3 — V2 without explicit role

```text
V2 backbone
+ FTP
- explicit role embedding
```

### FULL — RC-FTP V2

```text
V1 observable backbone
+ explicit role
+ feasible tangent projection
+ unchanged true-chain-rule decoder
```

No additional primary ablation is authorized.

---

## 20. Statistical analysis

### 20.1 Primary binary comparison

For paired blind-test joint-pass outcomes:

```text
FULL V2 vs frozen V1
```

use exact McNemar testing.

### 20.2 Mechanism ablations

For:

```text
FULL vs B2 (w/o FTP)
FULL vs B3 (w/o role)
```

use paired McNemar testing.

Holm correction is applied across these preregistered primary binary comparisons.

### 20.3 Continuous supporting metrics

For paired continuous differences such as:

- endpoint deficit;
- temporal deficit;
- action norm;

use paired Wilcoxon signed-rank testing and report:

- median paired difference;
- 95% bootstrap confidence interval.

Continuous improvements are supporting evidence only and cannot override a failed gate.

---

## 21. A/B/C/D stop rule

Exactly one terminal outcome must be assigned after the formal three-seed blind evaluation.

### A — `RC_FTP_V2_ADVANCE`

Assign A only if:

1. at least 2 of 3 seeds pass **all** efficacy gates;
2. all 3 seeds pass **all** safety gates;
3. no blind-test protocol violation occurred;
4. no threshold, case, seed, metric, or checkpoint was selected post hoc.

Consequence:

```text
Method-level V2 candidate is supported.
```

If seed 42 also individually passes all formal gates, seed 42 may proceed to a separately defined whole-song pilot stage.

### B — `RC_FTP_V2_PARTIAL_MECHANISTIC_SUCCESS`

Assign B if:

- safety remains clean;
- at least one preregistered efficacy measure improves over frozen V1;
- but one or more required overall or subgroup efficacy gates fail.

Examples:

```text
overall improves but single/28 < 12/16
continuous temporal deficit improves but joint gate fails
direction/alignment improves but temporal gate remains insufficient
```

Consequence:

```text
Mechanistic/partial result only.
No production or pilot promotion.
Protocol V2 stops.
No same-protocol architecture, loss, width, metric, or seed sweep.
```

### C — `RC_FTP_V2_SAFETY_REJECTED`

Assign C if any formal safety rule fails, including:

- any newly introduced hard physical regression;
- nonfinite output;
- systematic clean-identity failure;
- reference-fidelity violation;
- formal checkpoint publication rejection.

This classification applies even if temporal or endpoint efficacy improves.

Consequence:

```text
Candidate rejected for safety.
No pilot.
No production.
Protocol V2 stops.
```

### D — `RC_FTP_V2_NOT_SUPPORTED`

Assign D if:

- the method does not satisfy A;
- it does not qualify for B as a meaningful partial/mechanistic improvement;
- and no safety failure requiring C is the dominant terminal outcome.

Consequence:

```text
Protocol V2 method not supported.
No pilot.
No production.
Protocol V2 stops.
```

---

## 22. Explicit anti-post-hoc rules

After the first V2 optimizer step, the following are forbidden inside this protocol:

- changing any of the five constraints;
- changing the 3% endpoint target;
- changing the 3% temporal target;
- changing the 1.02 jerk non-regression factor;
- changing physical thresholds;
- changing clean-identity thresholds;
- changing reference-fidelity thresholds;
- changing FTP iteration count;
- adding or removing role categories;
- adding width heads;
- adding anatomy heads;
- changing the 8000-step budget;
- selecting an earlier best checkpoint;
- changing or dropping seeds;
- adding rescue seeds;
- replacing blind-test cases;
- moving cases between strata;
- using V1 Final64 as the V2 blind test;
- introducing a new primary metric after seeing results;
- relaxing subgroup gates;
- running architecture sweeps under the same protocol name.

Any such change requires a new preregistered protocol version.

---

## 23. Required provenance in every V2 report

Every formal V2 report must record:

```text
protocol_name
preregistration_path
preregistration_sha256
preregistration_git_commit
scientific_parent_commit
implementation_commit
training_split_sha256
dev_split_sha256
blind_test_sha256
seed
attempted_training_steps
accepted_training_steps
rollback_steps
model_version
boundary_protocol
constraint_protocol
projection_protocol
optimizer_protocol
checkpoint_path
checkpoint_sha256
scientific_acceptance
publish_allowed
pilot_allowed
```

If any required provenance field is missing, the result is not formal.

---

## 24. Required output artifacts

Each formal seed must produce at minimum:

```text
result/report.json
result/case_level.jsonl
result/training_updates.jsonl
result/checkpoint_decision.json
result/checkpoint.pt
```

The final three-seed synthesis must produce:

```text
result/report.json
result/evidence_summary.md
result/freeze_manifest.json
```

The final synthesis is read-only and may not retrain, reselect cases, select checkpoints, or recompute a different metric family.

---

## 25. Whole-song pilot boundary

Whole-song music-to-dance video evaluation is **not** part of the V2 method-development blind test.

It becomes formally authorized only when:

```text
terminal_outcome == RC_FTP_V2_ADVANCE
```

and the canonical seed-42 checkpoint independently satisfies all formal checkpoint requirements.

A whole-song pilot may then evaluate:

- boundary endpoint pass;
- boundary temporal pass;
- final physical gate;
- Peak Jerk;
- Exit Acceleration;
- foot skate;
- support drift;
- penetration;
- runtime transaction commit rate;
- rollback count;
- rendered MP4 quality.

Pilot results may not retroactively modify V2 method-development thresholds or blind-test decisions.

---

## 26. Frozen interpretation boundary

A positive V2 result would support:

> Explicit feasible tangent projection, combined with role conditioning, can convert some previously manipulable-but-insufficient Refiner mechanisms into a sufficiently safe constrained-action method under the preregistered evaluation contract.

A negative V2 result would **not** prove that all Refiner architectures are impossible.

It would support only:

> The preregistered RC-FTP V2 formulation did not provide a sufficient safe method under the frozen constraints, data split, training budget, seeds, and blind-test contract.

---

## 27. Freeze rule

This file must be frozen before V2 implementation begins.

The freeze procedure is:

1. save this file at `docs/refiner_protocol_v2_preregistration.md`;
2. compute SHA256;
3. commit the file alone;
4. record the resulting Git commit SHA;
5. tag or otherwise record the preregistration commit;
6. do not amend or rewrite that commit;
7. all V2 code must cite both the preregistration SHA256 and preregistration Git commit.

If the text changes after this point, the modified file is a new preregistration and must receive a new protocol name/version.

---

## 28. Final preregistered scientific question

> Can an explicit role-conditioned feasible tangent projection method produce a Refiner that simultaneously satisfies endpoint repair, temporal repair, jerk non-regression, support/contact safety, root-vertical safety, clean identity, and reference fidelity on a source-disjoint 128-case blind test, under a fixed 8000-step, three-seed training protocol?

The answer is determined only by the A/B/C/D stop rule in Section 21.
