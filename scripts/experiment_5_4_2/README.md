# Experiment 5.4.2 — Phase-conditioned readout mechanism search

Exp5.4.2 is a **mechanism search / oracle experiment**, not the next deployed SNN. It asks:

> With frozen WHAT and frozen causal WHEN already given, what low-capacity interaction can stably recover phase-dependent class evidence?

The common frozen path is the Exp5.4.1 direct WHAT classifier

\[
L_0=\sum_tW_0x_t+b,
\]

with canonical frozen 128D Local-SNN L2 spikes. The causal WHEN source is the same-seed Exp5.3.2.3 `ffsnn128_rsnn64` checkpoint, exposed as both 64D spikes and its 64D post-reset RSNN membrane trajectory.

## Stage A: fixed mechanism screen

Stage A fixes `K=4`, `rank=4` and runs exactly 10 conditions × 5 seeds = 50 validation-only tasks:

| condition | context | mechanism | role |
|---|---|---|---|
| `what_only` | none | frozen base | baseline |
| `additive_spike` | WHEN spike | additive | shortcut control |
| `additive_mem` | WHEN membrane | additive | shortcut control |
| `bilinear_spike` | WHEN spike | low-rank interaction | candidate |
| `bilinear_mem` | WHEN membrane | low-rank interaction | candidate |
| `bank_spike` | WHEN spike | phase-conditioned bank | candidate |
| `bank_mem` | WHEN membrane | phase-conditioned bank | candidate |
| `bank_mean_when` | time-averaged membrane | same bank | semantic-vs-temporal control |
| `bank_elapsed` | causal elapsed-time RBF | same bank | clock control |
| `bank_relphase_oracle` | true relative phase RBF | same bank | noncausal upper bound |

Only bilinear/bank spike/membrane conditions are eligible mechanism winners.

### Additive control

\[
\Delta e_t=A_hh_t,
\qquad
L=L_0+\sum_t\Delta e_t.
\]

It deliberately permits WHEN to act as class evidence without WHAT, so it is a negative scientific control.

### Low-rank bilinear candidate

\[
\Delta e_t=\alpha C[(A_xx_t)\odot(A_hh_t)].
\]

`alpha` starts exactly at zero, so epoch 0 is the frozen WHAT base. If residual WHAT or context is zero, the correction is structurally zero. At `r=4`, the factor matrices contain 816 weights; the implementation has 817 trainable parameters including `alpha`.

### Phase-conditioned readout bank

\[
p_t=\operatorname{softmax}(Gh_t),\qquad
g_t=p_t-\frac1K\mathbf1,
\]

\[
\Delta e_t=\alpha\sum_k g_{t,k}U_kV_k^Tx_t.
\]

`G` has no bias. Therefore zero context gives uniform `p_t`, centered gate `g_t=0`, and exactly zero correction. Residual-WHAT zero also gives zero correction. `alpha=0` makes epoch 0 a legal frozen-base checkpoint. At `K=4,r=4`, the implementation has 2497 trainable parameters including `alpha`.

## Source cache and membrane scaling

Five source/cache tasks reuse, but never rewrite, Exp5.4.1 / Exp5.4 / Exp5.3.2.3 artifacts. The new cache contains frozen WHAT, ordered/reset WHEN spikes, ordered/reset WHEN membranes, and provenance. Recomputed WHAT/WHEN spikes must exactly match the existing Exp5.4 cache.

Membrane scaling is strictly train-derived:

1. fit per-channel mean/std on **training valid timesteps only**;
2. exclude padding completely;
3. use the same scaler on train/val/test;
4. force scaled padding to zero.

Spikes are not scaled.

`bank_elapsed` uses causal absolute `t/fs` with 64 fixed RBFs and a train-derived elapsed range. `bank_relphase_oracle` uses noncausal `t/(T_i-1)` mapped to 64 RBFs and is never eligible for mechanism selection.

## Temporal/alignment controls

Candidate validation includes:

- `ordered`;
- `mean_when` using the same-source gesture mean;
- `when_zero`;
- five deterministic within-gesture `when_shuffle` replicates;
- `when_circular_shift` by about one-third of valid length;
- `reset_when`;
- `residual_what_zero`.

For bilinear/bank candidates, `when_zero` and `residual_what_zero` must exactly recover the frozen WHAT base. Reset-WHEN is secondary because resetting can collapse activity; final artifacts therefore report ordered/reset WHEN firing fractions.

## Validation-only mechanism selection

The screen finalizer only reads existing train/validation artifacts. It never trains and never evaluates test. A candidate is eligible only if:

1. mean paired validation BA delta vs WHAT > 0;
2. at least 4/5 seeds are non-negative;
3. ordered > same-source mean;
4. ordered > shuffle;
5. ordered > circular shift;
6. ordered > `bank_elapsed`.

Among eligible candidates, apply a one-standard-error rule and choose the lowest-parameter candidate in that band. The result is persisted as `screen_selection.json`. If none qualifies, refinement stops and test remains unopened.

## Stage B: winner-only capacity refinement

If a bank wins:

\[
K\in\{2,4,8\},\qquad r\in\{2,4,8\}
\]

=> 45 tasks across five seeds.

If bilinear wins:

\[
r\in\{2,4,8\}
\]

=> 15 tasks across five seeds.

Refinement is also validation-only and writes locked `selection.json` with `only_validation_selected=true` and `test_not_used_for_selection=true`. Only then may final test run.

## Final diagnostics

The locked final-test stage reports:

- candidate vs WHAT BA / accuracy / macro-F1 / loss;
- ordered/mean/zero/shuffle/circular/reset/residual-WHAT-zero ablations;
- residual ratio `||Delta L||/(||L0||+eps)` mean/median/P95/max;
- rescue/harm/retained-correct/retained-error;
- prefix BA at 10%, 20%, ..., 100%;
- ordered/reset WHEN firing fraction;
- for bank winners: gate entropy, bank utilization, temporal gate variation, and gate-vs-offline-relative-phase curves.

Relative phase is evaluation-only except the explicitly noncausal oracle control.

## Multi-CPU execution

This follows `AGENTS.md`: each independent run is one Slurm task with one CPU core; no more than 50 experiment tasks run concurrently; finalizers aggregate existing artifacts only.

Screen:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_2_screen_cpu.bash
```

After inspecting `screen_selection.json`, refine + locked final test:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_2_refine_cpu.bash
```

## Artifacts

Under:

```text
notebooks/artifacts/experiment_5_4_2_phase_conditioned_readout/phase_conditioned_readout_v1/
```

final outputs include:

- `screen_runs.csv`
- `screen_paired_deltas.csv`
- `refine_runs.csv`
- `final_runs.csv`
- `final_paired_deltas.csv`
- `ablation_runs.csv`
- `gate_activity.csv`
- `residual_metrics.csv`
- `rescue_harm.csv`
- `prefix_ba.csv`
- `selection.json`
- `manifest.json`

The notebook `notebooks/experiment_5_4_2_phase_conditioned_readout.ipynb` is analysis-only.
