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
- `grounding/`: unpaired semantic/intrinsic dual-branch grounding.
- `scheduling/`: unseen whole-song segmentation and fixed music priors.
- `routing/`: global event route and heading-aware closed loop.
- `training/`: semantic retriever, boundary refiner and local diffusion training.
- `evaluation/`: preflight and scientific audits.
- `rendering/`: motion-to-video rendering.

Historical schema strings remain inside the code only where checkpoints and
reports require backward compatibility. They are not part of the public path
layout.
