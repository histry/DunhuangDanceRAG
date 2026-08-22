# Installation and formal run

The server environment is expected to exist already.

```bash
REPO=/home/disk/lsm/storage/DunhuangDanceRAG
PY=/home/disk/lsm/conda_envs/edge/bin/python
cd "$REPO"

export GENERATION_PYTHON="$PY"
export EXPERIMENT_PROFILE=research
source configs/experiment.env

bash scripts/preflight.sh
bash scripts/run_official_smpl_full.sh \
  "$CHANG_E_OFFICIAL_SMPL_DIR" \
  assets/music/test/audio/dunhuangwu2.wav
```

The formal run requires a clean Git worktree, 14 manifest-authorized NPZ
sources, strict Librosa backend success for every training song, and
single-person-compatible Event-DB rows. It rebuilds all indexes and checkpoints
inside the run output directory.

After a successful trained run, generate another WAV without retraining:

```bash
bash scripts/generate_only.sh /path/to/audio.wav /path/to/trained_run auto
```
