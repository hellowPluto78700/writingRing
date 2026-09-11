# Experiment 0.2 — Endpoint Tail Regularization

## Question

Can the Exp0.1 binary direct-SNN backbones preserve valid-segment classification while suppressing pathological post-endpoint firing caused by long synaptic timescales?

## Fixed backbone/data contract

This experiment reuses the Exp0.1 direct binary SNN definitions and the same single user-disjoint split (`SPLIT_SEED=12345`). Three backbones are evaluated:

- `short_mid`: `(2,3) -> (2,3,4,5)`
- `mid_long`: `(2,3,4,5) -> (2,3,4,5,6,7)`
- `short_mid_long`: `(2,3) -> (2,3,4,5) -> (2,3,4,5,6,7)`

All hidden layers have width 128. Hidden and output spike caps are both 1. Forward dynamics are unchanged from Exp0.1; in particular, the actual synaptic recurrence remains `I_t = alpha I_{t-1} + W x_t`.

## Objectives and regularization profiles

Two task objectives are used:

- `whole_count_ce`
- `timestep_ce`

Five profiles are analyzed per objective:

- `none`: frozen Exp0.1 checkpoint, no retraining
- `a1`: valid-region hidden firing regularizer
- `a1_p2`: Exp0.1.5 all-fixes A1 + P2
- `tail`: endpoint-aligned post-end hidden spike-tail penalty
- `a1_tail`: valid-region A1 + endpoint-tail penalty

New runs use seeds `11,23,37`, exactly 50 epochs, and validation WholeCount balanced accuracy for checkpoint selection with validation WholeCount CE as tie-break.

## Gradient calibration

Every new regularized run performs one-time pre-training hidden-gradient calibration on five deterministic train-only batches. The target regularizer/task hidden-gradient norm ratio is 5%. Calibration is only over hidden linear weights; the output head is excluded. The resulting `kappa` is frozen for the run and multiplied by the existing 10-epoch linear warmup.

Base compositions:

- `a1`: `0.1 * A1`
- `a1_p2`: `0.1 * A1 + 0.01 * P2_all_fixes`
- `tail`: `0.1 * Tail`
- `a1_tail`: `0.1 * A1 + 0.1 * Tail`

## Endpoint-tail contract

Each sample is aligned to its own valid endpoint `L_i`.

- Classification uses only `t < L_i`.
- Evaluation/training tail rollout uses explicit zero input for `t >= L_i`.
- The model is rolled out for approximately 600 ms beyond every sample endpoint.
- Tail stages are approximately 0–200, 200–400, and 400–600 ms.
- Stage weights are `(1,2,4)` and the weighted sum is divided by 7.
- Hidden activity is averaged over neuron-steps within each layer, then averaged across hidden layers.
- The output layer is not tail-regularized.

Thus the tail objective is independent of unrelated padding length and never contributes spikes to the WholeCount classification readout.

## Diagnostics

Every new and frozen checkpoint is evaluated with the same endpoint rollout and produces:

- train/val/test classification metrics;
- valid hidden activity;
- three-stage hidden/output tail activity;
- final-hidden tail area;
- output tail area;
- final-hidden/output settling time proxy;
- final-hidden synaptic-state RMS in each tail stage;
- final-hidden synaptic-state tail RMS grouped by shift;
- one fixed test-sample raster figure containing final hidden and output spikes;
- compressed fixed-sample trace arrays for later state inspection.

The fixed raster sample is `test sample index 4`, matching the previous Exp0.1 firing-pattern diagnostic convention.

## Multi-CPU execution

There are 72 new runs:

```text
3 architectures x 2 objectives x 4 trainable profiles x 3 seeds = 72
```

They run as one Slurm array, one CPU core per task, capped at 50 concurrent tasks. The 18 frozen Exp0.1 controls are evaluated sequentially in a separate one-CPU job. A finalizer runs only after both jobs finish successfully and aggregates existing artifacts; it never trains or silently regenerates missing runs.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_2_endpoint_tail_cpu.bash
```

Artifacts are written to:

```text
notebooks/artifacts/experiment_0_2_endpoint_tail_regularization/endpoint_tail_regularization_v1/
```

Key finalized files:

- `summary.csv`
- `comparison_summary.csv`
- `paired_vs_frozen_exp01.csv`
- `calibration_summary.csv`
- `history_long.csv`
- `raster_index.csv`
- `manifest.json`
- `firing_patterns/**`

## Notebook policy

`notebooks/experiment_0_2_endpoint_tail_regularization.ipynb` is analysis-only. It reads finalized CSV/JSON/PNG artifacts and does not train, launch Slurm, or perform multiprocessing.
