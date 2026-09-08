# Experiment 5.4.3 — Causal elapsed-time readout capacity

## Question

Can a compact causal elapsed-time-conditioned readout recover the phase-specific evidence weighting achieved by frozen-WHAT Fixed250 + full Linear?

The experiment keeps the frozen Local-SNN WHAT representation, split, objective, and exact causal elapsed-time coordinate fixed. Only readout capacity changes.

## References

For each seed, Stage A first creates paired validation-only references from the same frozen WHAT cache:

- `what_wholecount`: frozen Exp5.4.1 direct WHAT base.
- `fixed250_full_linear`: valid WHAT spikes are summed in absolute 250 ms bins, flattened, standardized with train-only statistics, then classified by LogisticRegression.
- `streaming_fixed250`: the fitted scaler+Linear is converted to raw-count-space `W_b,b` and evaluated timestep-by-timestep. Offline and streaming logits must agree within `1e-6` and predictions must be identical.

The reference test split is not evaluated until a final `selection.json` is locked.

## Stage A — capacity map

Exact elapsed time is encoded using the existing 64-D causal elapsed context. The readout is

`W_eff(t) = W0 + alpha * sum_k g_k(t) DeltaW_k`,

where `W0` is the frozen WHAT whole-count readout, `g_k` are centered softmax gates, and factorized banks use `DeltaW_k = U_k V_k^T`.

Sweep:

- `K in {4, 8, 16}`
- `rank in {4, 8, 12}`; rank 12 is mathematically full rank for a 12 x 128 class/readout matrix.
- seeds `(11, 23, 37, 53, 71)`
- 45 independent CPU tasks.

Primary diagnostic:

`recovery = (BA_candidate - BA_what) / (BA_fixed250 - BA_what)`.

A strong validation recovery requires mean recovery >= 0.90 and mean gap to Fixed250 <= 0.01 BA, with positive mean paired improvement and at least 4/5 nonnegative seeds. If a strong configuration exists, the smallest supported parameter count is locked. Otherwise Stage A writes an extension mode.

## Stage B — diagnostic extension

Exactly 15 tasks, chosen before test is opened:

- `higher_k`: factorized full-rank `K in {20,24,32}, rank=12` if K=16 still improves over K=8 on at least 4/5 seeds.
- `direct_matrix`: direct full `DeltaW_k in R^(12x128)` for `K in {4,8,16}` otherwise, testing factorization/optimization versus temporal gating.

Stage B uses validation only and then writes the locked `selection.json`.

## Final test

Only after `selection.json` exists, 5 seed-level tasks evaluate:

- frozen WHAT whole-count;
- paired Fixed250 full Linear and its exact streaming equivalent;
- old K4/r4 elapsed anchor;
- selected Exp5.4.3 readout.

Diagnostics include oracle-gap recovery, gate utilization versus absolute elapsed time, effective-weight distance to Fixed250 matrices, residual-bank singular values/effective rank, rescue/harm, and relative/absolute prefix BA.

## Multi-CPU execution

Stage A:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_3_capacity_cpu.bash
```

This submits `5 reference tasks -> 45 capacity tasks -> one validation-only finalizer` with `afterok` dependencies.

Inspect:

```bash
cat notebooks/artifacts/experiment_5_4_3_elapsed_readout_capacity/elapsed_readout_capacity_v1/capacity_selection.json
```

Then Stage B/final:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_3_extension_cpu.bash
```

If Stage A already locked `selection.json`, the launcher skips extension and submits only the 5 final-test tasks plus aggregation finalizer. Otherwise it submits the 15-task diagnostic extension, validation selector, final-test array, and finalizer.

Every compute job initializes Conda locally, uses one CPU core, and sets BLAS/OpenMP thread counts to one. Finalizers aggregate existing artifacts only.

## Notebook policy

`notebooks/experiment_5_4_3_elapsed_readout_capacity.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts for tables and Matplotlib figures. It must never train, submit Slurm jobs, or regenerate missing run artifacts.
