# DunhuangDanceRAG

A clean research release for low-resource whole-song Dunhuang dance generation.
The project uses anatomy-safe real motion events, unpaired music-semantic
grounding, global route planning and risk-masked local refinement.

## Run

```bash
conda env create -f environment.yml
conda activate dunhuang-dance-rag
bash scripts/preflight.sh
bash run.sh assets/music/test/audio/dunhuangwu2.wav
```

See `docs/ARCHITECTURE.md`, `docs/DATA_CONTRACT.md` and
`docs/REPRODUCIBILITY.md`.
