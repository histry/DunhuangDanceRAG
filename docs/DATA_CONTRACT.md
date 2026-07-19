# Data and Asset Contract

## Motion data

Place the 12 Chang-E/Dunhuang BVH sources in `assets/motion/bvh/`. Source split
is performed before event slicing. The canonical output representation is:

`151D = 4 foot contacts + root XYZ + 24 x Rot6D`.

## Music data

- `assets/music/train/`: 788 training songs from the 985-song **multi-genre**
  music-structure corpus. It must not be described as 985 classical songs.
- `assets/music/classical_eval/`: independent classical-music evaluation set.
- `assets/music/test/audio/dunhuangwu2.wav`: current whole-song demonstration input.

Evaluation and test music must never be passed to unpaired music training.

## Fixed music priors

- `assets/weights/music/router.pt`
- `assets/weights/music/planner.pt`
- `assets/weights/music/duration.pt`
- `assets/indexes/event_index.json`
- `assets/indexes/duration_index.npz`

Motion-side retriever/refiner/diffusion and the grounding model are trained into
each run's `checkpoints/` directory.
