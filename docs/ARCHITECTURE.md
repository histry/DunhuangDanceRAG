# Project Architecture

```text
Input whole-song audio
  -> music structure and semantic schedule
  -> source-disjoint anatomy-safe motion database
  -> semantic/intrinsic event grounding
  -> global event route planning
  -> boundary-risk local refinement
  -> gravity, anatomy, heading and duration audits
  -> fixed-camera rendering
```

## Functional code layout

- `contracts/`: anatomy, gravity, heading, duration and boundary contracts.
- `retargeting/`: Chang-E BVH to EDGE151 retargeting and cache construction.
- `data_pipeline/`: source-disjoint split and data preparation.
- `events/`: Event-DB, AESD semantics and intrinsic event geometry.
- `grounding/`: backward-compatible unpaired grounding plus the opt-in
  real-audio Lorentz × sphere × Gaussian-Bures-Wasserstein × Euclidean
  product-manifold grounder.
- `scheduling/`: unseen whole-song segmentation, Generation-aligned Scheduler
  indexes, formal Router/Duration/Planner assets and checkpoint validation.  A
  historical Router may contribute only its frozen music encoder; its motion
  branch and historical Duration/Planner weights are not reused after the
  Event-DB, FPS or rotation contract changes.
- `routing/`: global event route and heading-aware closed loop.
- `training/`: formal music-motion Router, Duration and whole-song Planner
  training, followed by semantic retriever, boundary refiner and local
  diffusion training.
- `evaluation/`: preflight and scientific audits.
- `rendering/`: motion-to-video rendering.

Historical schema strings remain inside the code only where checkpoints and
reports require backward compatibility. They are not part of the public path
layout.

## Paper-one mixed-curvature grounding

The research path is layered so the physical motion manifold used by V45/V46
does not leak into the latent retrieval metric:

```text
real audio interval
  -> unprojected normalized CLAP + 64-frame temporal features
  -> audio tower
motion event
  -> intrinsic SO(3) dynamics + disjoint body-part encoder
  -> motion tower
both towers
  -> shared-curvature Lorentz hierarchy factor
  -> spherical semantic factor
  -> five body-part Gaussian-BW factors
  -> Euclidean control factor
  -> heteroscedastic uncertainty
  -> fixed global positive-weight product distance
  -> multi-positive bidirectional contrastive objective
```

The mixed path is never enabled by silently reusing the legacy semantic-noise
view. It requires a schema-checked paired dataset; legacy checkpoints continue
to load with strict dispatch by checkpoint schema. See
`docs/MIXED_CURVATURE_GROUNDER.md` for the staged commands.

## Paper-two Fisher-Rao Graph-SB routing

The paper-two research route operates on a time-expanded categorical Event
graph. It does not reinterpret discrete Event identifiers as points in a
continuous pose manifold:

```text
slot grounding logits
  -> Fisher-Rao categorical target marginals
Event endpoint states
  -> 24-joint product-SO(3) edge distance
paper-one mixed embedding, when present
  -> Lorentz hierarchy edge distance
anatomy / heading / severe physics
  -> structural zero edges
all factors
  -> time-inhomogeneous Markov reference process
  -> multi-marginal discrete Schrödinger IPF
  -> Viterbi MAP or history-constrained posterior decoder
  -> existing heading / anatomy / physics simulator remains authoritative
```

Old deployments keep `legacy_beam` as the code-level default. The research and
experiment profiles explicitly select `fisher_rao_graph_sb`. Non-convergence,
dead graph rows and missing strict manifold fields either fail closed or fall
back to the legacy beam according to an explicit environment switch; every
fallback reason is stored in the final report. See
`docs/FISHER_RAO_GRAPH_SB_ROUTING.md`.
