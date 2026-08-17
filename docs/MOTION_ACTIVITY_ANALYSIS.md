# Motion Activity Analysis and Static-Collapse Acceptance

## Research objective

The existing whole-song pipeline can produce an EDGE151 sequence that is finite,
boundary-safe, anatomy-safe and physically safe while remaining nearly static.
This integration adds an independent motion-activity contract. It does not relax
physical, anatomy, performer or severe-heading hard gates.

## Direct integration points

The installer modifies the existing project files rather than introducing a
parallel V-numbered pipeline:

- `routing/heading_closed_loop.py`
  - measures activity on the exact resampled and heading-aligned event core;
  - adds an activity mismatch penalty to candidate selection;
  - rejects strongly static candidates only when the music slot requires motion
    and at least two independent collapse indicators agree;
  - saves retrieval, refiner, diffusion and full-IK stage outputs.
- `routing/feasibility_contract.py`
  - prevents bounded feasibility recovery from clearing an activity hard reject;
  - preserves all immutable safety gates.
- `routing/boundary_closed_loop.py`
  - evaluates whole-song and per-slot activity after the physical gate;
  - writes the NPY and diagnostic JSON first, then fails the run when collapse is
    detected so a static result cannot be silently accepted.
- `configs/experiment.env`
  - is the only public configuration entry and exposes stable
    `MOTION_ACTIVITY_*` thresholds.

The new analysis implementation is stored in
`evaluation/motion_activity_analysis.py`, following the project's scientific
module naming convention.

## Metrics

The report contains:

- mean, P95 and maximum joint angular velocity in rad/s;
- root 3D travel, horizontal travel and travel per second;
- static-frame ratio;
- longest static streak in frames and seconds;
- motion-density signal and mean density;
- 24-joint mean angular-speed profile;
- per-slot target activity, measured activity and density gap;
- high-activity slot failure fraction;
- motion-density alignment MAE and correlation.

Rot6D decoding follows the canonical column-concatenated EDGE convention:

```text
[R[:, 0], R[:, 1]]
```

## Stage-wise diagnosis

When `MOTION_ACTIVITY_SAVE_STAGE_OUTPUTS=1`, the normal generation command writes:

```text
<stem>.stage_retrieval.npy
<stem>.stage_refiner.npy
<stem>.stage_diffusion.npy
<stem>.stage_full_ik.npy
<stem>.motion_activity.json
```

Interpretation:

| First static stage | Likely collapse location |
|---|---|
| retrieval | retrieval, route selection or event resampling |
| refiner | boundary refiner |
| diffusion | local diffusion |
| full_ik | IK/contact correction or heading restoration |
| all NPY files dynamic but video static | rendering |

## Installation

From the downloaded integration package:

```bash
python scripts/integrate_motion_activity.py --repo /path/to/DunhuangDanceRAG --check
python scripts/integrate_motion_activity.py --repo /path/to/DunhuangDanceRAG
```

Then run:

```bash
cd /path/to/DunhuangDanceRAG
python -m unittest tests.test_motion_activity_analysis -v
```

Use the normal project launcher afterwards. No alternate V-numbered entrypoint is
required.

## Standalone analysis

```bash
python evaluation/motion_activity_analysis.py \
  --input outputs/run/motion.npy \
  --report outputs/run/report.json \
  --fps 30 \
  --fail-on-collapse
```

The command returns exit code `3` when a static collapse is detected.

Compare the four saved stages with:

```bash
python scripts/analyze_motion_activity_stages.py \
  --motion outputs/run/motion.npy \
  --fps 30
```

## Threshold policy

Candidate hard rejection requires all of the following:

1. the slot target activity is at least
   `MOTION_ACTIVITY_CANDIDATE_REQUIRED_TARGET`;
2. at least two independent static indicators fail;
3. `MOTION_ACTIVITY_CANDIDATE_HARD_GATE=1`.

The final gate rejects only when multiple global indicators agree, or when a
high-activity slot failure is accompanied by a global collapse indicator. This
avoids treating intentional short holds as failures while preventing a whole
song from degenerating into a static pose with jitter.

## Recommended regression sequence

1. Run the standalone analyzer on the known 30-second collapsed output.
2. Re-run generation and compare all four stage NPY files.
3. Run retrieval-only, retrieval+refiner, retrieval+refiner+diffusion and full
   pipeline ablations using the existing enable switches.
4. Run the 120-second song and inspect the predicted bottleneck together with
   per-slot activity. If no diverse hard-safe active events exist, retain the
   dead end rather than relaxing physical/anatomy/severe-heading gates.
