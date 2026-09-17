# DunhuangDanceRAG

Music-led single-person Dunhuang dance generation under the Chang-E official
SMPL14 protocol. The formal path trains from scratch with local Librosa 12D
features and project-trained models; it does not load an external pretrained
music model.

```bash
source configs/experiment.env
bash scripts/preflight.sh
bash run.sh assets/music/test/audio/dunhuangwu2.wav
```

See `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md`,
`docs/BASELINES.md`, and `docs/REPRODUCIBILITY.md`.

The opt-in Paper-1 candidate Outcome Bank, matched linear/MLP repairability
experiment, shadow ranking and sealed paired evaluation are documented in
`docs/REPAIRABILITY_PREDICTOR.md`.  Their default mode is `off` and they do not
change the formal Graph-SB unary or authoritative Guard.
