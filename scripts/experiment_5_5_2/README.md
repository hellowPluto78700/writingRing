# Experiment 5.5.2 — Progress-Anchored History-Residual Routing

## Scientific question

Exp5.5 showed that fixed temporal anchors already route a bank of differentiated WHAT-to-class experts well: Clock is deployable and Oracle relative progress is a stronger diagnostic anchor. Exp5.5.1 then showed that giving a GRU freedom to learn the whole semantic WHEN state does not reliably improve cross-user classification.

Exp5.5.2 asks a narrower question:

> Once a strong temporal anchor is fixed, does past WHAT history provide an additional routing correction that cannot be explained by a more flexible coordinate map, current WHAT, or only the immediately previous WHAT state?

The frozen input remains the Local-SNN L2 WHAT trajectory `m_t in R^128`.

## Architecture

For each seed and anchor, Exp5.5's matching eight full expert matrices and class bias are frozen. The source RBF routing is represented by its original pre-softmax logits `a_t` and the new routing is

`q_t = softmax(a_t + delta_z_t)`

with

`delta_z_t = alpha * tanh(residual_t)`.

The final evidence remains

`e_t = (sum_k q_t,k W_k) m_t`

and valid-timestep evidence is summed once for the gesture-level CE loss.

This construction guarantees that zero residual projection gives the exact original anchor. Source validation checks q/logits/predictions to `1e-6` before the screen is allowed to run.

## Residual controls

Four matched residual modes are compared:

- **Coordinate** — anchor coordinate only through `1 -> 64 -> 8`; tests whether the fixed RBF map is simply too rigid.
- **Reset** — reset GRU64 on current `m_t`; tests instantaneous WHAT-dependent routing.
- **Lag1** — reset GRU64 on `m_(t-1)` only; tests whether one previous local state is sufficient.
- **Ordered** — GRU64 over history, with current correction from `h_(t-1)`; tests accumulated past context while enforcing that current `m_t` cannot route itself.

Lag1 and Ordered are exactly on the anchor at `t=0` because no past WHAT exists.

The expert bank and class bias are frozen for every mode. Only the residual branch trains. There are no progress, sticky, confidence, balance, diversity, or expert losses.

## Alpha screen and selection

`alpha in {0.25, 0.5, 1.0}` is screened on **Clock validation only** across seeds `(11, 23, 37, 53, 71)`.

Each residual mode selects its own alpha. Eligibility requires positive mean paired validation BA delta versus the paired Exp5.5 Clock baseline and at least 4/5 nonnegative seed deltas. Among eligible alphas, the smallest alpha within one SEM of the best mean validation BA is selected. If no alpha is eligible, the same fallback is recorded and that mode is marked `clock_residual_rescue_supported=false`.

Oracle and test never participate in alpha selection. After selection, the mode-specific Clock alpha is locked for the corresponding Oracle control.

## Final comparisons

The final table contains:

- Exp5.5 Clock
- Clock + Coordinate / Reset / Lag1 / Ordered
- Exp5.5 Oracle progress
- Oracle + Coordinate / Reset / Lag1 / Ordered
- Exp5.5.1 RegOrdered reference
- Fixed250 reference

Primary metric: balanced accuracy. Accuracy, macro-F1, and CE are retained as secondary metrics.

Mechanistic contrasts are reported as paired seed deltas. In particular:

- `Ordered - Anchor`: does a history residual help at all?
- `Ordered - Reset`: does past context add beyond current WHAT?
- `Ordered - Coordinate`: does past context add beyond flexible coordinate remapping?
- `Ordered - Lag1`: does accumulated history add beyond only `m_(t-1)`?

For Ordered, within-gesture circular shifts of the residual trajectory are the primary alignment control. Random permutation is a stronger secondary control. Both leave the anchor fixed.

## Multi-CPU execution

The experiment follows `AGENTS.md` task-level CPU execution:

```text
5 source validation tasks
        -> afterok
60 Clock screen tasks (4 modes x 3 alpha x 5 seeds, %50)
        -> afterok
1 validation-only selector
        -> afterok
20 Clock final evaluation tasks ---------\
                                         +-> afterok -> 1 artifact-only finalizer
20 Oracle train -> evaluate tasks -------/
```

Clock final jobs reuse the selected screen checkpoint; they do **not** retrain it. Oracle jobs train and evaluate their own selected-alpha checkpoint in one atomic task.

All compute jobs use one CPU core, initialize Conda on the compute node, prefer `writingring-gpu` and fall back to `writingring-viz`, disable CUDA, and set BLAS/OpenMP/NumExpr threads to one.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_2_cpu.bash
```

## Artifact layout

Artifacts are written under:

```text
notebooks/artifacts/
experiment_5_5_2_progress_anchored_history_residual/
progress_anchored_history_residual_v1/
```

Key files:

- `source_references/seed*.json`
- `screen_checkpoints/`, `screen_histories/`, `screen_evaluations/`
- `screen_runs.csv`
- `screen_summary.csv`
- `screen_paired_deltas.csv`
- `selection.json`
- `final_checkpoints/`, `final_histories/`, `final_evaluations/`
- `trajectories/`
- `same_what/`
- `final_runs.csv`
- `final_summary.csv`
- `final_paired_deltas.csv`
- `residual_diagnostics.csv`
- `residual_alignment.csv`
- `residual_only_probes.csv`
- `generalization_gap.csv`
- `clock_progress_alignment.csv`
- `same_what_same_coordinate.csv`
- `manifest.json`

## Notebook aggregation policy

`notebooks/experiment_5_5_2_progress_anchored_history_residual.ipynb` is strictly analysis-only. It requires the finalizer's CSV/JSON artifacts, raises when required artifacts are missing, and does not contain model training, alpha selection, Slurm submission, multiprocessing, or missing-run regeneration.

The notebook reports the locked selection first, then screen curves, final performance, paired mechanistic deltas, residual magnitude/alignment, validation-to-test generalization gaps, Clock progress-correction diagnostics, residual-only probes, and same-WHAT/same-coordinate examples.

The five seeds are paired stochastic seeds under the same user-disjoint split. They are not treated as five independent split replicates.
