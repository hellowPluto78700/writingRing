# Experiment 5.3.2.1 — RSNN WHEN objective upper-bound sweep

## Question

Experiment 5.3.2 established `rsnn_shortmem` as the strongest tested causal WHEN mechanism, but the learned WHEN state remained only moderately predictive of relative phase/progress. This experiment holds the **frozen WHAT representation and RSNN architecture fixed** and asks whether the training objective is limiting the quality of the WHEN state.

The deployment hypothesis remains:

```math
z_t^{WHAT} \rightarrow h_t^{WHEN}=RSNN(z_{1:t}) \rightarrow WHAT\times WHEN\ fusion
```

This experiment does **not** build the fusion layer. It only studies how much useful temporal context can be encoded in the RSNN state under different supervised WHEN losses.

## Fixed architecture and data

- Source WHAT: frozen Exp5.2 / Exp5.3.2 Local-SNN L2 trajectory, `z_t in R^128`.
- WHEN branch: the Exp5.3.2 `rsnn_shortmem` condition only.
- Hidden width: 128.
- `tau_mem ~= 22.5 ms`, `tau_syn ~= 22.5 ms`, recurrent `W_rec` enabled.
- Threshold/reset/surrogate/optimizer/batch size/epochs match Exp5.3.2.
- User-disjoint split and master seeds are unchanged: `(11, 23, 37, 53, 71)`.
- The final duration `T` is used only to construct supervision targets; it is never an inference input.

All objectives for the same seed use the same model initialization and the same training minibatch order.

## Targets

Continuous progress:

```math
p_t = t/(T-1) \in [0,1]
```

Ten-way relative phase:

```math
q_t = \min(9, \lfloor 10 p_t \rfloor)
```

The two component losses are the same sample-balanced valid-timestep losses used in Exp5.3.2:

```math
L_{phase} = CE(U_t \rightarrow q_t)
```

```math
L_{progress} = SmoothL1(U_t \rightarrow p_t)
```

Each gesture is averaged over its own valid timesteps first, then gestures are averaged across the batch. Padding never contributes to the loss.

## Five objectives

`lambda` is the weight on the phase term so that reducing `lambda` moves training toward continuous progress.

| objective | phase weight | progress weight | loss |
|---|---:|---:|---|
| `phase_only` | 1.0 | 0.0 | `L_phase` |
| `phase_heavy` | 5.0 | 1.0 | `5 L_phase + L_progress` |
| `joint` | 1.0 | 1.0 | `L_phase + L_progress` |
| `progress_heavy` | 0.2 | 1.0 | `0.2 L_phase + L_progress` |
| `progress_only` | 0.0 | 1.0 | `L_progress` |

This is intentionally not loss-normalized. The experiment also records shared-RSNN gradient norms so we can see whether a nominal coefficient actually changes which objective dominates representation learning.

## Checkpoint selection

Each objective is selected using validation data only:

1. minimize its own weighted validation objective loss;
2. tie-break by lower validation progress MAE;
3. then higher validation phase BA.

Native heads are retained for diagnostics, but they are **not the primary cross-objective comparison**. Under `phase_only` the progress head is not trained, and under `progress_only` the phase head is not trained.

## Primary common probes

After checkpoint selection the RSNN is frozen. Every objective is evaluated with the same post-hoc decoders on the same representations:

- shared LogisticRegression probe for 10-way phase;
- shared StandardScaler + Ridge probe for continuous progress.

The primary interface is post-reset membrane `U_t`. The same probe protocol is also applied to:

- synaptic state;
- instantaneous spikes;
- trailing 250 ms spike count;
- trailing 500 ms spike count.

This answers whether an objective produces useful WHEN internally and whether that information is exposed in spike-domain readouts.

## Progress quality beyond MAE

`ProgressMAE` alone does not tell us whether the decoded WHEN trajectory is temporally well behaved. For the common `U_t -> progress` Ridge probe, the experiment also reports per-gesture:

- sample-balanced progress MAE;
- Spearman correlation between predicted progress and timestep order;
- monotonic violation rate:

```math
M_{viol} = \frac{1}{T-1}\sum_t 1[\hat p_{t+1} < \hat p_t]
```

A useful WHEN state should have low MAE, high Spearman, and a low violation rate.

## Elapsed-time and WHAT baselines

Two baselines are recomputed with exactly the same phase/progress probe protocol:

1. `what_only`: current frozen `z_t` with no temporal state;
2. `elapsed_time_only`: scalar `t/fs` only, with no motion history and no final duration.

The elapsed-time baseline is critical. A better RSNN WHEN should not merely behave like a complicated clock. We therefore compare both absolute progress quality and the additional information that disappears when real history/order is removed.

## Causal history attribution

Every trained objective is evaluated in three modes:

- `ordered`: normal causal sequence;
- `state_reset`: clear synaptic, membrane, and recurrent spike history before each current `z_t`;
- `temporal_shuffle`: shuffle the valid WHAT sequence within each gesture while keeping the target timeline unchanged.

For phase accessibility:

```math
H_{reset}^{phase} = BA_{ordered} - BA_{reset}
```

```math
H_{shuffle}^{phase} = BA_{ordered} - BA_{shuffle}
```

For progress MAE, positive history gain means removing history made the decoded progress worse:

```math
H_{reset}^{progress} = MAE_{reset} - MAE_{ordered}
```

```math
H_{shuffle}^{progress} = MAE_{shuffle} - MAE_{ordered}
```

The final choice should not be based on MAE alone. A pure-progress objective that beats elapsed time but has near-zero `H_reset` / `H_shuffle` may have learned a clock-like state rather than task-relevant ordered context.

## Gradient dominance diagnostic

Once per epoch, on the same fixed training diagnostic batch, the runner computes L2 gradient norms on the shared RSNN parameters (`input_projection + recurrent weights`):

```math
G_{phase}=||\partial L_{phase}/\partial\theta_{RSNN}||_2
```

```math
G_{progress}=||\partial L_{progress}/\partial\theta_{RSNN}||_2
```

It also records objective-weighted contributions and fractions. This determines whether, for example, `0.2 L_phase + L_progress` is actually progress-dominated in gradient space.

## Run matrix

```text
5 objectives x 5 seeds = 25 independent runs
```

Repository-default multi-CPU workflow:

```text
5 frozen-WHAT / baseline preparation tasks
        -> afterok
25 independent train -> evaluate tasks (one CPU each)
        -> afterok
aggregation-only finalizer
        -> analysis-only notebook
```

Submit on Unity:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_3_2_1_cpu.bash
```

The Slurm array is capped at 25 concurrent tasks, below the repository default maximum of 50.

## Final artifacts

The finalizer writes under:

```text
notebooks/artifacts/experiment_5_3_2_1_when_objective_sweep/rsnn_when_objective_v1/
```

Expected aggregate files:

- `runs.csv`
- `histories.csv`
- `probe_runs.csv`
- `ablation_runs.csv`
- `history_gain_runs.csv`
- `gradient_diagnostics.csv`
- `baseline_runs.csv`
- `local_reference.csv`
- `manifest.json`

`notebooks/experiment_5_3_2_1_when_objective_sweep.ipynb` is analysis-only. It reads finalized artifacts, builds objective trade-off tables/plots, compares against elapsed time, visualizes history attribution and gradient dominance, and does not train models or submit jobs.

## Interpretation rule

The goal is not simply to minimize progress MAE. The strongest WHEN objective should ideally satisfy all of the following:

1. lower progress MAE than `elapsed_time_only`;
2. high per-gesture Spearman and low monotonic violation;
3. positive `H_reset` and `H_shuffle`, showing real history/order contributes;
4. retain useful phase accessibility;
5. expose a substantial fraction of WHEN information through spikes or short causal spike windows.

This experiment estimates the supervised objective ceiling of the fixed 128-neuron short-memory RSNN. It does not yet establish the ceiling of task-relevant semantic WHEN; that still requires the later WHAT x WHEN fusion experiment.
