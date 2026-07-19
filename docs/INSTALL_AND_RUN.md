# Installation and formal run

## 1. Install into EDGE

```bash
cd /path/to/EDGE_RESEARCH_FEASIBILITY_AND_CLEAN_SYNC_PATCH
bash install_edge.sh \
  /home/disk/lsm/storage/EDGE \
  /home/disk/lsm/conda_envs/edge/bin/python
```

## 2. Synchronize the cleaned project

```bash
bash install_dunhuang_dance_rag.sh \
  /home/disk/lsm/storage/DunhuangDanceRAG \
  /home/disk/lsm/conda_envs/edge/bin/python
```

EDGE should remain the development/training source of truth.  Synchronize the
clean project after every algorithmic contract change and only publish the
clean project after a full successful run.

## 3. Full rebuild and retraining

```bash
cd /home/disk/lsm/storage/EDGE

export V46_51_REBUILD_RETARGET_CACHE=1
export V46_51_REBUILD_EVENT_DB=1
export V46_51_RETRAIN_V44=1
export V46_51_RETRAIN_V45=1
export V46_51_RETRAIN_V46=1

export PERFORMER_GROUP=auto
export PERFORMER_ALLOW_CROSS_GROUP_RESCUE=0

bash scripts/run_v46_53_1_research.sh \
  "$PWD/test_music_bank/dunhuangwu2.wav"
```

## 4. Generation-only verification before a costly rebuild

The previous checkpoints may be used only as a diagnostic smoke test when the
old database paths are still present.  A formal result with new `change/` BVHs
requires a full cache/database/model rebuild.

## 5. Expected scientific invariants

- at least 8 source-safe BVHs;
- exact source-disjoint split;
- with 4F/8M and 8/2/2: train 2F+6M, val 1F+1M, test 1F+1M;
- Event DB contains `performer_groups` and `genders`;
- no source/anatomy unsafe rescue;
- generated frame count follows the current WAV;
- final report contains `research_feasibility` diagnostics.
