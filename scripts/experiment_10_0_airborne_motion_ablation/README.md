# Exp10.0 — Airborne-motion ablation with Exp7.3 A2

## Question

Exp10.0 measures how airborne/repositioning acceleration contributes to
classification when the classifier is held fixed to the Exp7.3 A2 training
contract.

The experiment compares three paired dataset variants:

- `original` (D0): `Encoder(a)`
- `postencode_mask` (D1): `m * Encoder(a)`
- `masked_accel_reencode` (D2): `m * Encoder(m * a)`

Action 0 and Action 1 are loaded together for every run.

The main scientific decomposition is:

```text
D0 - D1  -> evidence carried directly during airborne/reposition timesteps
D1 - D2  -> airborne acceleration influence through encoder temporal context
D0 - D2  -> total effect of removing airborne acceleration
```

## Training matrix

The model is exactly the Exp7.3 A2 contract:

```text
30 event channels
 -> L1 128 neurons, shifts (2,3,4)
 -> L2 128 neurons, shifts (2,3,4)
 -> bias-free Linear 128->12
 -> WCCE / valid temporal mean
```

L1, L2, and W are trained end-to-end with task-only loss. The maximum epoch,
minimum epoch, patience, Adam settings, threshold, and checkpoint rule are
reused from Exp7.3.

Cross-user evaluation reuses the Exp9.0 5-fold assignment logic. For each
rotation, one fold is test, the next fold is validation, and the remaining
three folds are training users.

```text
3 variants x 5 cross-user rotations x 3 seeds (11,23,37) = 45 runs
```

For a given `rotation x seed`, D0/D1/D2 share exactly the same:

- sample IDs and labels;
- valid lengths and padding geometry;
- train/validation/test users;
- model initialization seed;
- DataLoader seed/order;
- probe C-grid and probe random state.

Variant identity is deliberately excluded from those seeds.

## Per-checkpoint evaluations

Every selected checkpoint saves:

- native Linear/WCCE train/validation/test accuracy, balanced accuracy, macro-F1;
- same-W LIF deployment metrics with Exp7.3 beta/threshold/cap;
- Action-0-only and Action-1-only test metrics from the same combined-action model;
- epoch-wise train loss, validation loss, train BA, and validation BA.

## Representation probes

### Temporal-support requirement

Every representation probe in this experiment must be reported in two matched
temporal-support modes:

1. **valid-length masked** — the current behavior: timesteps after
   `valid_length` are excluded before whole-sequence or Fixed250 aggregation;
2. **whole-window unmasked** — aggregate over the complete padded inference
   window and do **not** zero/clear SNN synaptic current, membrane state, or
   spikes after `valid_length`.

For the whole-window version, the padded input may naturally be zero after the
valid sample, but the SNN trajectory is allowed to continue evolving from its
residual state. Fixed250 keeps the same absolute 250 ms bins in both modes.

Historical finalized artifacts that only contain the valid-length-masked
variant remain valid historical results. New or re-run probe evaluations must
publish both variants with explicit names/metadata.



The selected checkpoint is replayed once. The input, L1, and L2 features are
aggregated in memory and all probes are fitted inside the same Slurm task.

Input controls:

- event whole-count;
- ordered Fixed250 event count.

For both L1 and L2, the following internal states are probed:

- synaptic current;
- pre-reset membrane;
- binary spike;
- post-reset membrane.

Each hidden state receives:

- whole-sequence mean;
- ordered Fixed250 mean.

The binary spike state additionally receives:

- whole-count;
- ordered Fixed250 count.

This gives 22 probes per model. Every probe uses a train-only
`StandardScaler`, the Exp7.3.5 LogisticRegression C-grid, validation balanced
accuracy for C selection, and test evaluation only after selection.

## Multi-CPU execution

The workflow is:

```text
prepare split/data audit (1 CPU)
        |
        v
45-task Slurm array, one CPU per run, max concurrency 45
        |
        v
afterok finalizer / aggregator (1 CPU)
```

Each array task performs:

```text
load paired dataset
 -> train A2
 -> select best checkpoint
 -> native + same-W LIF evaluation
 -> replay selected checkpoint
 -> fit all 22 probes
 -> save run artifacts
```

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_0_cpu.bash
```

To force replacement of completed run artifacts:

```bash
EXP10_FORCE=1 bash scripts/bash_script/SNN_Bash/submit_exp_10_0_cpu.bash
```

List the 45 array-task identities:

```bash
python -m scripts.experiment_10_0_airborne_motion_ablation list-runs
```

## Dataset roots

D0 reads the original padded roots:

```text
outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
```

D1 reads:

```text
.../aligned-board-events_writing_motion_ablation/postencode_mask/segmentation_padded
```

for both actions.

D2 reads:

```text
.../aligned-board-events_writing_motion_ablation/masked_accel_reencode/segmentation_padded
```

for both actions.

The prepare stage hard-fails unless D0/D1/D2 have identical canonical sample
IDs, users, actions, labels, valid lengths, padded lengths, and source-trial
identity.

## Outputs

Final artifacts are written below:

```text
notebooks/artifacts/experiment_10_0_airborne_motion_ablation/a2_cross_user_5fold_v1/
```

Important outputs:

```text
audit.json
canonical_sample_manifest.csv
split/
checkpoints/
histories/
evaluations/
probe_evaluations/

run_metrics.csv
variant_run_summary.csv
variant_split_level.csv
variant_summary.csv
paired_variant_deltas.csv
paired_variant_split_deltas.csv
paired_variant_summary.csv

probe_runs.csv
probe_split_level.csv
probe_summary.csv
probe_paired_variant_deltas.csv
probe_paired_variant_split_deltas.csv
probe_paired_variant_summary.csv

manifest.json
```

The primary statistical unit is the cross-user rotation. The finalizer first
averages the three model seeds inside each rotation and then summarizes the
five rotation-level values. The 15 raw run values per variant are retained as
descriptive results, but are not treated as 15 independent user splits.
