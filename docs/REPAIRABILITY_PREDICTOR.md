# Conditional repairability predictor

## Scientific boundary

The repairability model is a local ranking hint, not a safety classifier that
can authorize commit.  Fisher--Rao Graph-SB and its Router-probability unary are
unchanged.  The learned model runs after cheap bridge simulation, can rank only
candidates that already pass the existing pre-risk contract, and cannot bypass
full generation or the authoritative post-generation Guard.

Three runtime modes are available:

- `off`: byte-for-byte production decision logic; no checkpoint is loaded;
- `shadow`: score the complete local pool and record predictions, but preserve
  the original first-safe/minimum-risk choice;
- `rank`: rank only pre-safe candidates.  A checkpoint is rejected unless its
  held-out MLP-vs-linear promotion gate authorized ranking.  Low score spread or
  any non-finite prediction abstains to the original selector.

## 1. Build the Outcome Bank on the 4090

Use multiple full-song evaluation sequences from source-disjoint recording or
song groups.  For each normal pipeline invocation:

```bash
source configs/repairability.env
export REPAIRABILITY_MODE=off
export GAR_OUTCOME_BANK_ENABLE=1
export GAR_OUTCOME_BANK_PATH=outputs/paper1/repairability_outcomes_v2.jsonl
export GAR_OUTCOME_BANK_SEEDS=42,43,44,45
export BOUNDARY_RESELECT_TOPK=8
export GAR_OUTCOME_BANK_TOPK=8
bash run.sh
```

Capture freezes the final baseline event path, replaces one boundary candidate
at a time, resets the same random seed for every candidate, and executes the
complete Refiner/Diffusion/IK/audit path.  The JSONL writer is append-only and
skips completed `(case, candidate, seed)` keys, so interrupted jobs can resume.
At least two common seeds are mandatory.  Capture refuses to run with a learned
selector enabled, preventing circular labels.
`post_safe` is the conjunction of the full boundary-continuity, whole-motion
physical, activity and schedule hard gates; all component decisions and reason
codes remain in each version-2 record.  Version-1 banks must not be resumed or
mixed with this schema: use a fresh JSONL path after upgrading.
`GAR_OUTCOME_BANK_TOPK` may not truncate the production pool.  For a lower-cost
pilot, reduce `BOUNDARY_RESELECT_TOPK` in the frozen protocol as shown above so
every method and the bank still use exactly the same pool.

The default `group_id` is the full-song sequence ID, and candidate-level random
splits are never allowed.  `--split-isolation` can be frozen as `sequence`,
`recording`, `performer`, or `all`; the latter modes merge sequences connected by
shared source provenance before splitting.  Training fails if the selected policy
leaves fewer than three disjoint components.  The isolation policy is stored in
the training fingerprint and must be chosen before inspecting results.  The
server runner defaults to `all`; weaker isolation is an explicit protocol change.

## 2. Train matched baselines

After pushing the exact reviewed commit and completing all capture jobs:

```bash
export EXPECTED_COMMIT=$(git rev-parse HEAD)
export REPAIRABILITY_BANK=outputs/paper1/repairability_outcomes_v2.jsonl
export REPAIRABILITY_SEEDS=42,43,44,45
export REPAIRABILITY_SPLIT_ISOLATION=all
bash scripts/run_repairability_training_server.sh
```

The runner validates seed completeness, runs focused server tests, and trains a
linear model and MLP with identical features and splits.  Reports include AUPRC,
Brier score, NLL, ECE, risk MAE, Top-1 expected post-safe rate, Top-1 risk,
Router-probability retention, rank drift, and regret.  Router probability and
pre-risk are evaluated as non-learned baselines.

The MLP checkpoint remains shadow-only unless it improves internal holdout
Top-1 safety
over the linear model by the configured gate, does not regress expected risk,
and retains Router quality within tolerance.  This prevents architecture choice
from being justified merely by adding features.

## 3. Shadow and sealed ranking

First run full songs with:

```bash
export GAR_EVALUATION_TRACE_ENABLE=1
export GAR_EVALUATION_METHOD_VARIANT_ID=repairability_shadow
export REPAIRABILITY_MODE=shadow
export REPAIRABILITY_CHECKPOINT=/absolute/path/repairability_mlp.pt
bash run.sh
```

Only after inspecting paired shadow results should a gate-authorized checkpoint
be evaluated with `REPAIRABILITY_MODE=rank`.  Compare the same pools, generator,
Guard, seeds and song groups.  Primary decision metrics are initial post-safe
rate, reselection count and end-to-end runtime.  Final unsafe-boundary rate is a
non-inferiority constraint; Router quality, motion activity, diversity and
fidelity must not regress.  MLP forward time alone is not a runtime claim—the
complete-pool bridge feature cost is included in end-to-end time.
Set `GAR_EVALUATION_METHOD_VARIANT_ID=repairability_baseline` for the paired
`off` run and `repairability_rank` for the paired `rank` run; this field is
trace metadata and does not alter the decision path.

Paired baseline/rank GAR traces can be checked with:

```bash
python -m evaluation.repairability_paired_evaluation \
  --baseline /path/to/baseline_traces \
  --method /path/to/rank_traces \
  --training-bank outputs/paper1/repairability_outcomes_v2.jsonl \
  --require-sealed-isolation \
  --minimum-seeds 4 \
  --output outputs/paper1/repairability_paired_report.json
```

The evaluator fails closed when case/seed sets, candidate pools, generator
fingerprints or repair fingerprints differ, and reports paired bootstrap
intervals for initial safety, final safety and reselection count.  A sealed run
also fails if its sequence IDs or source recording IDs overlap the training bank.
Selector-only `REPAIRABILITY_*` settings are intentionally excluded from the
frozen generator fingerprint; pool, generator and repair equality are still
checked directly.
