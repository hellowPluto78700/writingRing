# Experiment 5.3.2.3 — Pre-recurrent FF-SNN vs direct RSNN64

## Scientific question

Exp5.3.2.2 showed that increasing the one-layer recurrent WHEN width beyond roughly H=64-128 gives only marginal transferable gains while train capacity keeps increasing. Exp5.3.2.3 therefore asks a different question:

> Is the remaining supervised causal WHEN ceiling caused by insufficient nonlinear transformation **before** recurrence rather than insufficient recurrent width?

The recurrent state is fixed at H=64 for every newly trained condition. The frozen Local-SNN WHAT cache, user split, causal masking, objective, optimizer, checkpoint rule, probe grids, history ablations, and analysis protocol are unchanged.

## Conditions

The comparison matrix has four conditions and five master seeds `(11, 23, 37, 53, 71)`.

| condition | architecture | role |
| --- | --- | --- |
| `direct_rsnn64` | `WHAT128 -> RSNN64` | Primary direct recurrent reference, identity-validated reuse of Exp5.3.2.2 `rsnn_h64`; never retrained here |
| `linear64_rsnn64` | `WHAT128 -> Linear64 -> RSNN64` | Depth/parameterization control without an added spiking transform |
| `ffsnn64_rsnn64` | `WHAT128 -> short-tau FF-SNN64 spikes -> RSNN64` | Primary FF-SNN test |
| `ffsnn128_rsnn64` | `WHAT128 -> short-tau FF-SNN128 spikes -> RSNN64` | Checks whether FF64 itself is a front-end bottleneck |

`linear64_rsnn64` and `ffsnn64_rsnn64` have identical trainable weight shapes and use paired component initialization. The only architectural difference between them is the inserted short-tau LIF dynamics/spike nonlinearity.

The FF-SNN feeds **spikes** into the recurrent layer. There is no residual/skip connection, no temporal pooling, and no access to final duration.

## Fixed dynamics and objective

All SNN layers use the same short dynamics inherited from Exp5.3.2.2:

- `shift_syn = 1`
- `shift_mem = 1`
- same threshold, reset rule, and surrogate gradient
- recurrent layer width fixed at `64`
- RSNN recurrence remains `64 -> 64`

The supervised causal WHEN objective remains

\[
L_{WHEN}=L_{phase}+L_{progress}.
\]

The phase and progress targets are constructed from the valid gesture prefix. `T_i` is used **only to construct supervision targets and the valid mask**. It is not provided to the Local-SNN state, FF layer, RSNN, phase head, progress head, or deployment path.

Loss reduction remains sample-balanced: average over valid timesteps within each gesture, then average over gestures.

Checkpoint selection remains:

1. minimum validation joint objective loss;
2. tie -> lower validation sample-balanced Progress MAE;
3. tie -> higher validation Phase BA.

Test data are never used for checkpoint selection.

## Pairing and initialization

All trainable conditions reuse the unchanged Exp5.3.2 loader seeds, so minibatch order is paired within each master seed.

Initialization is deterministic. `linear64_rsnn64` and `ffsnn64_rsnn64` share the same component seed namespace, so their `front_projection`, `input_projection`, recurrent weights, and heads start identically. This makes their paired comparison specifically test the inserted LIF/spike transform rather than random initialization.

`ffsnn128_rsnn64` uses its own deterministic architecture namespace because its parameter shapes differ.

## Primary comparisons

### 1. Does an FF-SNN help at fixed recurrent capacity?

Primary comparison:

\[
WHAT_{128}\rightarrow FF\text{-}SNN_{64}\rightarrow RSNN_{64}
\]

versus

\[
WHAT_{128}\rightarrow RSNN_{64}.
\]

The main outcome is whether `ffsnn64_rsnn64` improves both membrane Phase BA and sample-balanced Progress MAE relative to the reused direct RSNN64.

### 2. Is any gain specifically due to the spiking transform?

Compare:

`ffsnn64_rsnn64` vs `linear64_rsnn64`.

Because the trainable weight shapes and initial values are paired, a consistent advantage for `ffsnn64_rsnn64` supports a benefit from the nonlinear short-memory spiking transform rather than merely an extra matrix factorization.

### 3. Is FF64 itself too narrow?

Compare:

`ffsnn128_rsnn64` vs `ffsnn64_rsnn64`.

This is a secondary capacity diagnostic. The recurrent state is still fixed at H=64.

## Evaluation

Every new trainable run performs `train -> best checkpoint -> evaluate -> save artifacts` inside one Slurm task.

The evaluation protocol is inherited from Exp5.3.2.2 and includes:

- membrane `U_t` phase probe;
- membrane sample-balanced progress MAE;
- per-gesture progress Spearman;
- monotonic violation rate;
- train/val/test probe generalization;
- state-reset history attribution `H_{reset}`;
- five temporal-shuffle replicates for `H_{shuffle}`;
- membrane, synaptic, instantaneous spike, trailing-250ms spike-count, and trailing-500ms spike-count probes;
- recurrent firing-rate utilization;
- effective membrane state dimension `D_{eff}`;
- PCA90 dimension `D_{90}`.

State-reset removes all temporal state in the tested architecture while preserving the current input. For FF-SNN conditions this resets **both** FF-SNN and recurrent SNN state each timestep.

## Run matrix and Slurm execution

Only the three new trainable conditions are scheduled. The direct RSNN64 source is reused.

- preparation: 5 tasks, one per seed;
- training/evaluation: **15 independent runs**;
- one CPU core per run;
- condition-major mapping:
  - tasks 0-4: `linear64_rsnn64`
  - tasks 5-9: `ffsnn64_rsnn64`
  - tasks 10-14: `ffsnn128_rsnn64`
- finalizer: aggregation only;
- dependencies use `afterok`.

Submit the complete chain with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_3_2_3_cpu.bash
```

The finalizer requires all **15/15** newly trained evaluations/histories, all five direct references, and all five baseline artifacts. It never retrains models or silently regenerates missing runs.

## Final artifacts

Artifacts are written under:

`notebooks/artifacts/experiment_5_3_2_3_ff_before_rsnn/ff_before_rsnn_v1/`

Expected finalized files:

- `runs.csv`
- `histories.csv`
- `probe_runs.csv`
- `ablation_runs.csv`
- `history_gain_runs.csv`
- `activity_runs.csv`
- `representation_runs.csv`
- `baseline_runs.csv`
- `local_reference.csv`
- `paired_deltas.csv`
- `manifest.json`

`paired_deltas.csv` stores per-seed differences against `direct_rsnn64` for Phase BA, Progress MAE, Spearman, and monotonic violation.

## Decision rule

Do not use a scalar composite score.

First ask whether a candidate is a strict mean Pareto improvement over `direct_rsnn64`:

- higher test membrane Phase BA;
- lower test sample-balanced Progress MAE.

Then use the following as diagnostics rather than hidden reweighting:

- consistency of paired per-seed deltas;
- `H_{reset}` and `H_{shuffle}`;
- train/validation gap;
- spike-domain accessibility;
- firing activity;
- parameter count;
- `D_{eff}` and `D_{90}`.

Interpretation:

- `FF-SNN64 > direct` and `FF-SNN64 > linear64`: the bottleneck is plausibly pre-recurrent nonlinear/dynamic transformation.
- `linear64 ~= FF-SNN64 > direct`: extra factorization/depth helps, but the spiking transform is not specifically responsible.
- `FF-SNN128 > FF-SNN64`: the front-end representation itself is capacity-limited.
- all transformed models ~= direct: the remaining WHEN ceiling is unlikely to be fixed by adding a feedforward SNN before recurrence; move to a different memory/dynamics hypothesis rather than increasing width again.

The notebook `notebooks/experiment_5_3_2_3_ff_before_rsnn.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts and does not train, refit probes, or submit jobs.
