# Experiment 5.5 — Non-SNN history-conditioned WHAT experts

## Scientific question

Does gesture history provide a better conditioning variable for mapping the frozen Local-SNN WHAT representation to class support than a shared mapping, absolute elapsed time, or an instantaneous nonlinear gating transform?

The frozen input is the existing Local-SNN L2 WHAT spike trajectory, `m_t in R^128`. Exp5.5 does not train or modify the Local SNN and does not use an SNN WHEN/Fusion implementation.

## Core readout

For expert conditions, each latent context state owns a full trainable matrix

`W_k in R^(12 x 128)`, `k=1..8`.

At each valid timestep,

`e_t = sum_k q_t,k W_k m_t`.

Gesture logits are the non-leaky valid-timestep evidence sum plus one final 12-D class bias. Expert matrices have no per-timestep bias.

The primary learned-history model uses a GRU64 and strict-history gating:

`q_t = softmax(G h_(t-1))`,

so `q_t` only depends on `m_1..m_(t-1)` and cannot see the current `m_t` before deciding how it should be interpreted.

The training objective is whole-gesture cross entropy only. There is no progress, phase, sticky, confidence, or balance auxiliary loss in this first mechanism-validation experiment.

## Conditions

Five conditions are predeclared and paired across seeds `(11, 23, 37, 53, 71)`:

1. `shared`: one global `12 x 128` WHAT matrix; no q or GRU.
2. `clock`: eight smooth RBF context probabilities over causal absolute elapsed time, clipped to the train-set maximum elapsed time.
3. `oracle_progress`: eight smooth RBF probabilities over `t/(T-1)`. This is a non-deployable diagnostic because it uses final valid length.
4. `reset_gru`: GRU64/state-head/expert architecture with state reset independently at every timestep. It can nonlinearly gate the current WHAT but contains no history.
5. `ordered_gru`: the same GRU64/state-head/expert parameterization, but with normal recurrent history and strict-history `h_(t-1)` gating.

`reset_gru` and `ordered_gru` use the same constructor seed per experiment seed. Their parameter shapes are identical; the intended causal contrast is history versus no history.

## Reference

Each seed first reuses the existing Exp5.4 frozen WHAT cache and creates the paired frozen-WHAT Fixed250 + full Linear reference using the Exp5.4.3 definition:

- valid WHAT spikes are summed in 250 ms absolute bins;
- features are flattened;
- `StandardScaler` is fit on train only;
- `LogisticRegression(lbfgs)` is fit on train;
- fitted weights are converted back to raw count space;
- offline and timestep-streaming logits must match within `1e-6` with identical predictions.

The reference is diagnostic only and never changes the predeclared Exp5.5 conditions.

## Model selection and test policy

Each run selects its own checkpoint only by validation balanced accuracy, tie-broken by validation loss. Early stopping patience is 25 epochs, with a maximum of 200 epochs.

The test split is never used to select epochs, hyperparameters, conditions, or architecture. All five conditions are fixed before the experiment starts.

## Required mechanism diagnostics

The experiment reports more than final balanced accuracy:

- paired `ordered_gru - reset_gru`: history-specific effect with matched recurrent/expert capacity;
- paired `ordered_gru - clock`: learned history versus absolute elapsed time;
- paired `ordered_gru - oracle_progress`: learned history versus perfect scalar relative progress;
- ordered-GRU shuffled-q evaluation: q values and occupancy are preserved, but within-gesture WHAT/q temporal alignment is destroyed;
- q-only linear probes from mean q and final q to diagnose class-code collapse;
- state occupancy, entropy, total variation, transition rate, and dwell duration;
- q-state profiles versus absolute elapsed time and relative progress;
- same-WHAT/different-history nearest-neighbor analysis on ordered-GRU test trajectories, reporting WHAT cosine similarity, q distance, and local evidence difference.

The strongest evidence for the semantic-context hypothesis is:

`ordered_gru > reset_gru`, `ordered_gru > clock`, `ordered_gru > shuffled_q`, together with highly similar WHAT vectors receiving different q/evidence under different histories.

## Multi-CPU execution

Exp5.5 follows `AGENTS.md` task-level parallelism exactly:

```text
5 independent reference-preparation tasks
        -> afterok
25 independent condition x seed tasks
        -> each task: train -> validation-select best checkpoint -> evaluate -> write per-run artifacts
        -> afterok
1 finalizer
        -> aggregate existing artifacts only
```

Run from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_cpu.bash
```

The 25 experiment tasks run with one CPU core each and at most 25 concurrent tasks, below the repository cap of 50. Compute jobs initialize Conda locally, prefer `writingring-gpu`, fall back to `writingring-viz`, and force BLAS/OpenMP thread counts to one.

## Artifact layout

Finalized artifacts live under:

```text
notebooks/artifacts/experiment_5_5_non_snn_history_experts/non_snn_history_experts_v1/
```

Important outputs:

- `references.csv`: per-seed Fixed250 reference metrics and streaming equivalence.
- `runs.csv`: one row per trained condition/seed.
- `summary.csv`: mean/SD/SEM summary by condition.
- `paired_deltas.csv`: paired mechanism contrasts.
- `state_diagnostics.csv`: occupancy/entropy/persistence metrics.
- `state_profiles.csv`: q probability versus elapsed time and relative progress.
- `q_only_probes.csv`: q-only class decodability.
- `shuffled_q.csv`: ordered-GRU shuffled-q replicates.
- `same_what_pairs.csv`: same-WHAT/different-history diagnostic pairs.
- `manifest.json`: locked protocol metadata.

Per-run checkpoints, histories, JSON evaluations, ordered-GRU trajectory NPZs, and same-WHAT CSVs are retained in subdirectories.

## Notebook aggregation policy

`notebooks/experiment_5_5_non_snn_history_experts.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts, performs tables/paired summaries, and makes Matplotlib figures. It never trains models, launches Slurm jobs, or regenerates missing run artifacts. Missing finalized artifacts are treated as an incomplete experiment rather than silently recomputed.
