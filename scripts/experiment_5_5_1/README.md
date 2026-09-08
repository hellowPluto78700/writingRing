# Experiment 5.5.1 — Regularized semantic WHEN

## Scientific question

Exp5.5 showed that state-conditioned WHAT-to-class experts can use a good temporal conditioning variable, but the unconstrained CE-only ordered GRU did not learn a useful history-aligned semantic WHEN state: ordered GRU did not beat matched reset GRU, clock, oracle progress, or shuffled-q.

Exp5.5.1 asks one narrower question:

> Can weak temporal/state regularization organize the existing GRU history representation into a useful semantic WHEN state without forcing the latent state itself to equal progress?

This experiment changes the WHEN objective only. It keeps the frozen Local-SNN WHAT trajectory, GRU64, K=8 full experts, strict-history routing, whole-gesture evidence sum, optimizer, checkpoint selection rule, and five Exp5 seeds fixed.

## Frozen input and strict-history routing

The input is the existing frozen Local-SNN L2 WHAT spike trajectory

`m_t in R^128`.

The ordered branch remains

`h_t = GRU(m_t, h_(t-1))`

and the current routing probability is

`q_t = softmax(G h_(t-1))`.

Therefore the current `m_t` cannot affect the routing decision used to interpret itself. The effective class mapping is

`W_eff(t) = sum_k q_t,k W_k`,

with eight independent full `W_k in R^(12 x 128)` expert matrices and no expert bias. Evidence is

`e_t = W_eff(t) m_t`,

and gesture logits are the valid-timestep evidence sum plus one final 12-D class bias.

## Progress auxiliary is not progress gating

A second head is attached to the current recurrent hidden state:

`r_hat_t = sigmoid(P h_t)`.

Its target during training is

`r_t = t / (T - 1)`

on valid timesteps only. This target uses final valid duration during training, but inference remains causal because neither `T` nor future WHAT is provided to the network.

The predicted progress scalar is **never** fed into `q_t`, the state head, RBFs, experts, or `W_eff(t)`. It exists only to constrain the information retained by `h_t`. Consequently two histories with the same relative progress are still free to produce different semantic q states.

## Auxiliary losses

Classification remains the main objective:

`L_cls = CE(sum_t e_t + b, y)`.

Three weak auxiliary losses are available:

1. Progress:
   `L_progress = SmoothL1(r_hat_t, r_t)` over valid timesteps, beta=0.1.
2. Sticky state:
   `L_sticky = mean ||q_t - q_(t-1)||_1` over valid adjacent pairs only.
3. Confidence:
   normalized entropy `H(q_t) / log(K)` over valid timesteps.

There is deliberately no state-balance/uniform-occupancy loss.

The locked primary weights are:

- `lambda_progress = 0.3`
- `lambda_sticky = 0.1`
- `lambda_confidence = 0.02`

These are tested as nested recipes rather than a large hyperparameter sweep.

## Validation-only screen

The CE-only Exp5.5 ordered-GRU run is reused as R0 and is not retrained.

Three new ordered-GRU recipes are screened across seeds `(11, 23, 37, 53, 71)`:

- R1 `progress`: `L_cls + 0.3 L_progress`
- R2 `progress_sticky`: `L_cls + 0.3 L_progress + 0.1 L_sticky`
- R3 `progress_sticky_confidence`: `L_cls + 0.3 L_progress + 0.1 L_sticky + 0.02 L_confidence`

Each run selects its own checkpoint by validation balanced accuracy, tie-broken by validation classification CE. The test split is not evaluated by screen tasks.

A recipe is eligible when its mean paired validation BA improvement over CE-only is positive and at least 4/5 seed deltas are nonnegative. Among eligible recipes, the selector uses a one-standard-error preference for the lowest-complexity recipe. If no recipe is eligible, the selector records a validation fallback and marks `regularization_rescue_supported=false`; test still does not influence the choice.

## Matched reset control

After `selection.json` is fixed, the selected recipe is trained with the capacity-matched reset-GRU branch for all five seeds. Reset mode applies the same GRU/state-head/progress-head/expert parameterization independently to each current timestep with zero recurrent history.

This yields the primary history-specific contrast:

`BA(reg_ordered) - BA(reg_reset)`.

The reset stage still records train/validation metrics only. Final test is opened only after both selection and matched reset artifacts exist.

## Final test diagnostics

The five final seed tasks load the already-selected ordered checkpoint and matched reset checkpoint, then evaluate train/val/test and write per-seed final artifacts. Required diagnostics include:

- progress MAE, RMSE, R2, Pearson, Spearman, and coarse 10-bin accuracy;
- q occupancy, entropy, total variation, transition rate, and dwell duration;
- q profiles versus relative progress and absolute elapsed time;
- q-only class probes from mean q and final q;
- a train-fit/test-evaluated Ridge probe from true relative progress to q, to diagnose whether semantic q has collapsed into a pure progress code;
- five deterministic within-gesture shuffled-q test replicates;
- same-WHAT/different-history nearest-neighbor pairs;
- stricter same-WHAT + matched-progress pairs requiring WHAT cosine similarity >=0.95 and relative-progress difference <=0.05.

The primary success pattern is:

`reg_ordered > ce_ordered`,

`reg_ordered > reg_reset`,

and

`reg_ordered > shuffled(reg_ordered)`.

Beating clock is stronger evidence; beating oracle relative progress together with matched-progress same-WHAT evidence would support semantic history beyond a scalar phase coordinate.

## Multi-CPU execution

Exp5.5.1 follows the repository task-level CPU policy:

```text
5 source-validation tasks
        -> afterok
15 ordered screen tasks (3 recipes x 5 seeds)
        -> afterok
1 validation-only screen selector
        -> afterok
5 matched reset tasks
        -> afterok
5 final test/diagnostic tasks
        -> afterok
1 artifact-only finalizer
```

Every compute task requests one CPU core, initializes Conda locally, prefers `writingring-gpu`, falls back to `writingring-viz`, disables CUDA, and sets OpenMP/BLAS thread counts to one.

Run from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_1_cpu.bash
```

## Artifact layout

Final artifacts are written under

```text
notebooks/artifacts/
experiment_5_5_1_regularized_semantic_when/
regularized_semantic_when_v1/
```

Important artifacts:

- `source_references/seed*.json`
- `screen_checkpoints/`, `screen_histories/`, `screen_evaluations/`
- `screen_runs.csv`
- `screen_summary.csv`
- `screen_paired_deltas.csv`
- `selection.json`
- `reset_checkpoints/`, `reset_histories/`, `reset_evaluations/`
- `final_evaluations/seed*.json`
- `final_runs.csv`
- `final_summary.csv`
- `final_paired_deltas.csv`
- `progress_metrics.csv`
- `state_diagnostics.csv`
- `state_profiles.csv`
- `q_only_probes.csv`
- `q_progress_probe.csv`
- `shuffled_q.csv`
- `same_what_pairs.csv`
- `matched_progress_same_what.csv`
- `manifest.json`

## Notebook aggregation policy

`notebooks/experiment_5_5_1_regularized_semantic_when.ipynb` is analysis-only. It requires finalized CSV/JSON artifacts, raises on missing required inputs, and never trains a model, chooses the winning recipe, invokes Slurm, or regenerates missing data. Selection is a durable artifact produced by the validation-only selector job; final tables and plots consume it rather than recomputing it in the notebook.
