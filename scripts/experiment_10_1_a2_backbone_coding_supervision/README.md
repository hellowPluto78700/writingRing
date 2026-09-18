# Exp10.1 — A2 backbone coding × supervision ablation

## Question

Exp10.1 tests whether the A2 backbone improves when three factors are varied
without changing the temporal readout:

1. input dataset: D0 original vs D1 post-encode airborne mask;
2. hidden communication: binary/binary (BB) vs MT3/MT3 (MM);
3. training objective: baseline L2 WCCE, cooperative L1+L2 joint, or baseline
   L2 WCCE plus 0.1 L1 timestep CE.

No phase-aware readout, recurrence, width sweep, or tau sweep is included.

## Dataset factor

- `original` (D0): `Encoder(a)`
- `postencode_mask` (D1): `m * Encoder(a)`

Both Action 0 and Action 1 are combined exactly as in Exp10.0. D0 and D1 must
have identical sample IDs, labels, users, actions, valid lengths, padded
geometry, and source-trial identity.

## Coding factor

### BB

```text
L1: 128 binary neurons, threshold 1.0x
L2: 128 binary neurons, threshold 1.0x
```

### MM

Both L1 and L2 use the Exp8.1.1 fixed heterogeneous-threshold MT3 population:

```text
threshold multipliers = 0.5x, 1.0x, 1.5x
width = 128 in each layer
shifts = (2,3,4) within every threshold bank
communication remains binary
```

Only BB and MM are repeated here because Exp8.1.1 already isolated MB/BM and
showed that the richer L1 MT representation is most meaningfully tested with
MT propagated through L2.

## Objective factor

### O0 — `l2_wcce`

```math
L = CE(mean_t(W2 z2_t), y)
```

This is the standard A2 time-shared readout.

### O1 — `l1_l2_joint`

```math
L = CE(mean_t(W1 z1_t + W2 z2_t), y)
```

This is cooperative joint evidence. It is deliberately **not** two independent
classification losses.

Inference uses both time-shared branches.

### O2 — `l2_wcce_plus_0p1_l1_tsce`

```math
L = CE(mean_t(W2 z2_t), y)
  + 0.1 * mean_{valid t} CE(W1 z1_t, y)
```

The L1 TSCE head is training-only. Native validation/test inference uses only:

```math
mean_t(W2 z2_t)
```

This makes any native gain attributable to changed backbone training rather
than extra inference capacity.

## Full run matrix

```text
2 datasets
x 2 coding conditions
x 3 objectives
x 1 locked cross-user split (rotation0)
x 3 seeds
= 36 independent runs
```

Seeds are `11,23,37`. Exp10.1 uses only Exp10.0/Exp9.0 cross-user
`rotation0`: test fold 0, validation fold 1, and train folds 2/3/4. For the
same seed, all 12 conditions share the same model-init stream, DataLoader
order, sample split, and probe random state. Dataset/coding/objective identity
is excluded from paired model initialization.

## Evaluation

Primary endpoint:

```text
native test balanced accuracy
```

Also saved:

- train/validation/test Accuracy, BA, Macro-F1;
- output-LIF transfer diagnostic;
- Action-0 and Action-1 test metrics;
- joint branch-removal diagnostics;
- per-epoch train/validation BA and loss;
- L1 auxiliary TSCE when applicable.

## Representation diagnostics

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



Every checkpoint receives the same 22-probe inventory as Exp10.0:

- input whole-count and Fixed250 count;
- L1/L2 synaptic current, pre-reset membrane, spike, and post-reset membrane;
- whole-mean and ordered Fixed250 mean for each state;
- whole-count and Fixed250 count for spikes.

The main information path is summarized as:

```text
L1 pre-reset
   -> L1 spike
   -> L2 pre-reset
   -> L2 spike
   -> native time-shared classifier
```

Derived diagnostics include:

```text
L1 quantization delta
L1 -> L2 transform delta
L2 quantization delta
```

MT activity is reported by threshold bank and synaptic shift so saturation of
the 0.5x threshold population is visible.

## Primary contrasts

The finalizer produces paired per-seed contrasts on the single locked split:

- D1 - D0;
- MM - BB;
- Joint - baseline;
- L1 TSCE - baseline;
- MT × Joint interaction;
- MT × L1-TSCE interaction.

This experiment does not estimate population-level split variance. The three
paired seeds are optimization replicates on one fixed cross-user split, so
mean/std across seeds are descriptive optimization statistics only.

## Multi-CPU strategy

```text
prepare/audit
    |
    v
36-task CPU Slurm array
one CPU per run
max 36 concurrent
    |
    v
afterok finalizer
    |
    v
aggregation-only notebook
```

Launch:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_1_cpu.bash
```

Force replacement of completed run artifacts:

```bash
EXP10_1_FORCE=1 bash scripts/bash_script/SNN_Bash/submit_exp_10_1_cpu.bash
```

List array identities:

```bash
python -m scripts.experiment_10_1_a2_backbone_coding_supervision list-runs
```

## Outputs

```text
notebooks/artifacts/
  experiment_10_1_a2_backbone_coding_supervision/
    d0_d1_bb_mm_supervision_single_split_v1/
```

Main aggregate outputs:

```text
audit.json
run_metrics.csv
condition_split_level.csv
condition_summary.csv
contrast_runs.csv
contrast_split_level.csv
contrast_summary.csv
probe_runs.csv
probe_split_level.csv
probe_summary.csv
information_path_runs.csv
information_path_split_level.csv
information_path_summary.csv
information_path_contrast_runs.csv
information_path_contrast_split_level.csv
information_path_contrast_summary.csv
activity_runs.csv
activity_summary.csv
manifest.json
```

The notebook reads only finalized artifacts and never trains models.
