# Exp0.1.5 — Gradient-Calibrated All-Fixes Regularization

## Scientific question

Exp0.1.4 showed that the original SAE-Dense P2+A1 formulation collapses the direct SNN because the regularizer gradient can dominate the classification gradient by orders of magnitude. Temporal EW-mass normalization is the strongest single repair, and combining temporal normalization, tau-gain normalization, and warmup largely removes the collapse. However, the 30-epoch `all_fixes` condition is weak and does not improve test BA over the no-regularization control.

Exp0.1.5 therefore freezes the **all-fixes mathematical formulation** and changes only its overall strength. The strength is expressed as a target hidden-parameter gradient ratio rather than a hand-picked lambda.

The primary question is:

```text
Can a non-destructive but non-trivial all-fixes regularizer improve classification when its
initial hidden-gradient strength is calibrated to 5%, 10%, or 20% of the task gradient?
```

## Fixed model / data contract

The experiment uses only the Exp0.1 binary direct-SNN family:

- Raw64 30-channel unsigned weighted event input.
- Hidden width 128 in every hidden layer.
- Binary hidden spikes (`hidden_cap=1`).
- 12 binary spiking output neurons (`output_cap=1`).
- Same data split, labels, optimizer, learning rate, weight decay, batch size, SNN dynamics, objective implementations, WholeCount deployment readout, and validation checkpoint-selection rule as Exp0.1.
- Five paired seeds: `11, 23, 37, 53, 71`.

Architectures:

- `short_mid`: `(2,3) -> (2,3,4,5)`
- `mid_long`: `(2,3,4,5) -> (2,3,4,5,6,7)`
- `short_mid_long`: `(2,3) -> (2,3,4,5) -> (2,3,4,5,6,7)`

Objectives:

- `whole_count_ce`
- `timestep_ce`

## All-fixes regularizer is frozen

Exp0.1.5 calls the Exp0.1.4 `all_fixes` P2/A1 implementation directly. No new P2 or A1 mathematical change is introduced.

For hidden pre-reset membrane `v_i(t)` and neuron-specific synaptic decay `alpha_i`, the P2 penalty path uses:

```text
v_penalty_i(t) = (1 - alpha_i) * v_i(t)
U_i(t) = 0.99 * U_i(t-1) + v_penalty_i(t)
M_i(t) = 0.99 * M_i(t-1) + 1
Ubar_i(T) = U_i(T) / M_i(T)
P2_l = mean_batch,neuron Ubar_i(T)^2
P2 = mean_layer P2_l
```

A1 remains the Exp0.1.4 valid-neuron-step hidden firing mean, averaged across hidden layers.

The base regularizer is:

```text
R0 = 0.01 * P2_all_fixes + 0.1 * A1
```

The `0.01` and `0.1` coefficients define the fixed relative P2:A1 composition. They are no longer interpreted as the final regularization strength.

## One-time hidden-gradient calibration

Each run performs a train-only calibration before any optimizer update.

Calibration uses:

- exactly 5 shuffled training batches, which covers the complete current training split at the default batch size of 128;
- a dedicated deterministic calibration-loader seed;
- no parameter update;
- hidden linear weights only when measuring gradients;
- the output classifier head is excluded from the gradient-ratio calculation.

For calibration batch `b`:

```text
r_b = || grad_hidden R0 ||_2 / || grad_hidden L_task ||_2
```

The robust calibration ratio is:

```text
r0 = median_b r_b
```

For target gradient ratio `rho`:

```text
kappa = rho / r0
```

Targets:

- `rho = 0.05`
- `rho = 0.10`
- `rho = 0.20`

`kappa` is computed once before training and is then frozen for the entire run. The experiment does **not** dynamically rebalance gradients during training.

All target strengths sharing the same `(architecture, objective, seed)` use the same model initialization and training-loader random stream. The calibration-loader stream is also target-independent, so the target-strength comparison is paired.

## Training loss and warmup

The training loss is:

```text
L(epoch) = L_task + warmup(epoch) * kappa * R0
warmup(epoch) = min(1, epoch / 10)
```

Therefore the full-strength calibrated target is reached at epoch 10. The calibration target is an initialization-time reference, not a constraint that is re-enforced later in training.

## Early stopping

Training is bounded by:

```text
minimum epochs = 50
maximum epochs = 100
patience = 30 epochs
```

Two validation progress signals are tracked independently:

1. validation WholeCount balanced accuracy;
2. validation native-objective loss (`whole_count_ce` or `timestep_ce`, matching the run).

A progress event occurs when either:

```text
validation WholeCount BA improves by > 1e-12
OR
validation objective loss improves by > 1e-4
```

After epoch 50, training stops when neither signal has improved for 30 consecutive epochs. The final checkpoint is **not** chosen by the early-stop signal. Checkpoint selection remains the Exp0.1 rule:

```text
highest validation WholeCount BA
then lower validation WholeCount CE as tie-break
```

## Run matrix

New training runs:

```text
3 target ratios x 3 architectures x 2 objectives x 5 seeds = 90 runs
```

The no-regularization baseline is **not retrained**. Finalization reuses the existing 30 frozen Exp0.1 binary direct-SNN runs. Those frozen baselines were trained with the Exp0.1 100-epoch budget and are therefore an established performance baseline rather than an equal-compute early-stopping control.

## Multi-CPU Slurm strategy

The array follows the repository default:

```text
#SBATCH --array=0-89%50
#SBATCH --cpus-per-task=1
```

Each array task performs one complete independent run:

```text
5-batch calibration
-> train 50-100 epochs
-> select best validation WholeCount checkpoint
-> evaluate train/val/test
-> save calibration, history, checkpoint, and evaluation artifacts
```

The finalizer runs only after all 90 tasks succeed and only aggregates existing artifacts. It does not retrain or silently regenerate missing runs.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_1_5_gradient_calibrated_all_fixes_cpu.bash
```

Slurm stdout/stderr are written directly in the submission directory:

```text
exp0_1_5_grcal_<ARRAY_JOB_ID>_<TASK_ID>.out
exp0_1_5_grcal_<ARRAY_JOB_ID>_<TASK_ID>.err
exp0_1_5_finalize_<JOB_ID>.out
exp0_1_5_finalize_<JOB_ID>.err
```

No task creates a repository-root `outputs/` directory.

## Per-run diagnostics

Each history records:

- target gradient ratio;
- calibration median ratio and frozen `kappa`;
- warmup scale and effective P2/A1 coefficients;
- task loss, P2, A1, base regularizer, weighted regularizer, and total loss;
- validation WholeCount BA;
- validation WholeCount CE;
- validation native-objective loss;
- last progress epoch and early-stop state;
- per-layer P2 terms.

At epochs `1, 5, 10, 20, 30, 50, 75, 100` when reached, the first training batch also records:

```text
||grad_hidden task||
||grad_hidden R0||
base R0 / task hidden-gradient ratio
effective regularizer / task hidden-gradient ratio
cosine(task hidden gradient, regularizer hidden gradient)
```

The cosine distinguishes compatible, orthogonal, and conflicting regularization directions.

## Finalized artifacts

All results live under:

```text
notebooks/artifacts/experiment_0_1_5_gradient_calibrated_all_fixes/gradient_calibrated_all_fixes_v1/
```

Per-run subdirectories:

- `calibrations/`
- `histories/`
- `checkpoints/`
- `evaluations/`

The finalizer writes:

- `runs.csv`
- `summary.csv`
- `history_long.csv`
- `calibrations.csv`
- `calibration_summary.csv`
- `gradient_trajectory_summary.csv`
- `stopping_summary.csv`
- `paired_vs_frozen_exp01.csv`
- `paired_vs_frozen_exp01_summary.csv`
- `frozen_exp01_binary_context.csv`
- `frozen_exp01_summary.csv`
- `comparison_summary.csv`
- `manifest.json`

## Notebook aggregation

`notebooks/experiment_0_1_5_gradient_calibrated_all_fixes.ipynb` is analysis-only. It reads finalized artifacts and reports:

- test BA by target / architecture / objective;
- paired test-BA change relative to frozen Exp0.1;
- calibration ratio and `kappa` stability across seeds;
- actual gradient-ratio drift across training;
- task/regularizer gradient cosine;
- firing-rate and dead-neuron behavior;
- early-stopping / best-epoch statistics;
- validation trajectories for the three target strengths.

The notebook contains no training, multiprocessing, or Slurm submission code.
