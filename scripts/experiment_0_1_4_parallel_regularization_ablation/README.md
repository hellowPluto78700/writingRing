# Exp0.1.4 — Parallel Regularization Repair Ablation

## Scientific question

Exp0.1.2 and Exp0.1.3 showed that directly adding the SAE-Dense P2+A1 regularizer can collapse the Exp0.1 direct SNN toward chance-level classification and strongly suppress firing. Exp0.1.4 asks which proposed repair prevents that failure, and whether combining all repairs can preserve useful classification dynamics.

This experiment is a short, paired 30-epoch diagnostic sweep. It is not intended to replace the 100-epoch Exp0.1 performance benchmark.

## Fixed SNN/training contract

Everything below is inherited from Exp0.1 unless explicitly changed by the condition:

- Raw64 30-channel unsigned weighted event input.
- Binary hidden spikes only (`hidden_cap=1`).
- Hidden width 128 in every hidden layer.
- 12 binary spiking class-output neurons (`output_cap=1`).
- Same optimizer, learning rate, weight decay, batch size, user split, labels, objective implementations, WholeCount deployment readout, and validation checkpoint-selection rule as Exp0.1.
- Same paired model-initialization and loader seeds for every condition with the same `(architecture, objective, seed)` tuple.
- Exactly 30 training epochs per run.

Architectures:

- `short_mid`: `(2,3) -> (2,3,4,5)`
- `mid_long`: `(2,3,4,5) -> (2,3,4,5,6,7)`
- `short_mid_long`: `(2,3) -> (2,3,4,5) -> (2,3,4,5,6,7)`

Objectives:

- `whole_count_ce`
- `timestep_ce`

Seeds: `11, 23, 37, 53, 71`.

## Conditions

Six conditions are trained independently from the same paired initialization/data order:

| Condition | P2 temporal definition | Tau-gain correction | Lambda warmup |
|---|---|---|---|
| `no_reg` | diagnostic Exp0.1.2 EW sum only | no | disabled; optimization weights are zero |
| `original_reg` | Exp0.1.2 EW sum | no | no |
| `temporal_norm` | EW mean | no | no |
| `tau_gain_norm` | Exp0.1.2 EW sum | yes | no |
| `warmup` | Exp0.1.2 EW sum | no | yes |
| `all_fixes` | EW mean | yes | yes |

The base coefficients remain `lambda_P2=0.01`, `lambda_A1=0.1`; warmup scales both together.

### Original P2

For hidden pre-reset membrane `v_i(t)`:

```text
U_i(t) = 0.99 U_i(t-1) + v_i(t)
P2_l = mean_{batch, neuron} U_i(T)^2
P2 = mean_l P2_l
```

Variable-length sequences update only on valid timesteps and freeze the endpoint after each sample ends.

### Temporal normalization

The cumulative endpoint is divided by its valid exponentially weighted mass:

```text
Ubar_i(T) = sum_t 0.99^(T-t) v_i(t) / sum_t 0.99^(T-t)
```

P2 is then computed from `Ubar` with the same batch/neuron/layer mean reduction. This isolates the effect of the cumulative temporal mass.

### Tau-gain normalization

For neuron-specific synaptic decay `alpha_i`, only the P2 penalty path uses:

```text
v_penalty_i(t) = (1 - alpha_i) * v_i(t)
```

The actual SNN forward state, spikes, membrane dynamics, and readout are unchanged. This condition tests whether long-timescale groups are disproportionately penalized simply because their synaptic state has higher persistence/gain.

### Warmup

For epochs `e=1..30`:

```text
scale(e) = min(1, e / 10)
lambda_P2(e) = 0.01 * scale(e)
lambda_A1(e) = 0.1 * scale(e)
```

This tests whether strong regularization during the earliest representation-formation phase pushes the SNN into a dying basin.

### All fixes

`all_fixes` combines temporal EW-mean normalization, `(1-alpha_i)` tau-gain normalization, and the 10-epoch linear warmup. A1 remains exactly the Exp0.1.2 hidden-firing definition so this experiment does not introduce an additional activity-penalty factor.

## Run matrix and multi-CPU execution

The full matrix is:

```text
6 conditions x 3 architectures x 2 objectives x 5 seeds = 180 runs
```

The Slurm array uses one independent run per task, one CPU core per task, and at most 50 concurrent tasks:

```text
#SBATCH --array=0-179%50
#SBATCH --cpus-per-task=1
```

Each task performs:

```text
train 30 epochs -> select best validation WholeCount checkpoint -> evaluate train/val/test -> save per-run artifacts
```

The finalizer is submitted with `afterok` and only aggregates existing run artifacts. It never trains or silently recreates a missing run.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_1_4_parallel_regularization_ablation_cpu.bash
```

The submit wrapper exports `REPO_ROOT`, matching the Exp0.1.2 Slurm pattern. Slurm stdout/stderr are written directly in the sbatch submission directory as:

```text
exp0_1_4_regab_<ARRAY_JOB_ID>_<TASK_ID>.out
exp0_1_4_regab_<ARRAY_JOB_ID>_<TASK_ID>.err
exp0_1_4_finalize_<JOB_ID>.out
exp0_1_4_finalize_<JOB_ID>.err
```

No compute task creates a repository-root `outputs/` directory.

## Gradient diagnostics

To measure optimizer competition without roughly doubling CPU cost, gradient norms are evaluated only on the first training batch of epochs:

```text
1, 2, 5, 10, 20, 30
```

The history records task, raw P2, raw A1, weighted P2, weighted A1, and combined regularizer gradient norms, including:

```text
weighted_P2_grad / task_grad
weighted_A1_grad / task_grad
combined_regularizer_grad / task_grad
```

For `no_reg`, original-scale P2/A1 values are still reported as diagnostics, but all weighted regularizer gradients are zero.

## Artifacts

Per-run artifacts are stored under:

```text
notebooks/artifacts/experiment_0_1_4_parallel_regularization_ablation/parallel_regularization_ablation_v1/
```

with `checkpoints/`, `evaluations/`, and `histories/` subdirectories.

After all 180 tasks succeed, the finalizer creates:

- `runs.csv` — one row per Exp0.1.4 run.
- `summary.csv` — five-seed metric summaries by condition / architecture / objective.
- `history_long.csv` — all 180 x 30 epoch histories in long format.
- `paired_condition_effects.csv` — every non-`no_reg` run paired against the same 30-epoch `no_reg` architecture/objective/seed.
- `paired_condition_summary.csv` — five-seed summary of paired BA/firing/dead-neuron changes.
- `gradient_diagnostics_summary.csv` — gradient competition summaries at the six diagnostic epochs.
- `frozen_exp01_binary_context.csv` and `frozen_exp01_summary.csv` — 100-epoch Exp0.1 binary direct-SNN results for historical context only.
- `comparison_summary.csv` — compact 30-epoch condition summaries plus the historical 100-epoch Exp0.1 context.
- `manifest.json` — exact experiment/condition contracts.

## Notebook

`notebooks/experiment_0_1_4_parallel_regularization_ablation.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts, builds paired tables, checks firing/dead-neuron behavior, inspects gradient competition, and plots validation trajectories. It contains no training, multiprocessing, or Slurm submission code.

## Primary interpretation

The key paired comparisons are:

- `original_reg -> temporal_norm`: cumulative temporal-mass effect.
- `original_reg -> tau_gain_norm`: heterogeneous synaptic-timescale bias.
- `original_reg -> warmup`: early dying-basin effect.
- `no_reg -> all_fixes`: whether all repairs together preserve or improve useful 30-epoch classification/firing dynamics.

The frozen 100-epoch Exp0.1 results should not be treated as an equal-budget paired control; the correct equal-budget control is the Exp0.1.4 `no_reg` condition.
