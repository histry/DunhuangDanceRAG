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

## Bootstrap music prior and formal Scheduler assets

`assets/weights/music/router.pt` is a historical music-domain prior.  A formal
run may import only its `music_encoder` branch (frozen by default).  Its motion
encoder, the historical Planner and the historical Duration model are not
valid after the Generation Event-DB or FPS changes.

Every full run therefore performs this ordered chain:

1. rebuild the source-disjoint Event-DB and Generation-aligned Scheduler index;
2. train a new Router motion branch against that exact ordered index;
3. train a new Duration model with explicit `pytorch3d_row` native layout and
   canonical-column boundary conversion;
4. train a new whole-song Planner from song-disjoint weak labels produced by
   the new Router;
5. validate FPS, SMPL24, ordered `event_uid` fingerprint, Rot6D layout and file
   hashes before same-WAV regression;
6. only after that gate, train V44/V45/V46.

Motion-side retriever/refiner/diffusion and the grounding model are trained into
each run's `checkpoints/` directory.
