# Exp7.3.8 — Frozen Linear W to LIF: current-gain sweep

## Question

Exp7.3.7 showed that the P5-style whole-count affine Linear head reaches about 58.6% test BA, but its raw `W` collapses to chance when copied directly into the standard output LIF (`beta=0.5`, `threshold=0.5`). Exp7.3.8 tests whether this is primarily an operating-range / current-scale mismatch.

The frozen representation is always the Exp7.3 A2/WCCE L2 cache. L1/L2 are never retrained.

## Stage A — reproduce Linear heads

For valid whole-count

\[
Z=\sum_{t=1}^{T}z_t,
\]

reproduce the Exp7.3.7 / Exp7.3.2 P4-P5 heads:

- `A0_linear_no_bias_W0`: \(W_0 Z\)
- `A1_linear_affine_W1_b1`: \(W_1 Z+b_1\)
- `A3_linear_affine_W1_bias_off`: \(W_1 Z\), same affine-trained `W1` with its bias removed at inference.

The requested cross-head control is:

- `A2_linear_hybrid_W0_b1`: \(W_0 Z+b_1\)

Here `W0` is copied exactly from the independently trained no-bias Linear head, while `b1` is copied exactly from the affine Linear head. Nothing is retrained or rescaled. This control asks whether the learned affine bias remains useful when attached to the no-bias-trained weight geometry.

## Stage B — scalar current-gain sweep

Two frozen Linear weight sources are tested independently:

1. `affine_w = W1`
2. `no_bias_w = W0`

The output LIF receives

\[
I_t=gWz_t
\]

with no input bias and no training of `W`. Sweep

\[
g\in\{0.125,0.25,0.5,1,2,4,8,16,32,64\}.
\]

For every gain, report train/validation/test accuracy, balanced accuracy, macro-F1, zero-output fraction, total output spikes, active output classes, count variance, and peak class spike rate.

Select `g` independently for each seed and weight source using validation balanced accuracy only. The first/smallest gain wins exact ties. Test metrics are not used for selection.

## Stage C — count-space bias only after gain matching

After selecting the LIF current gain, freeze everything and fit only the Exp7.3.7 count-space calibrator:

\[
s=h(C+q),\qquad h>0,
\]

where `h` is a CE-only positive logit gain and `q` is a 12D count offset. Deployment uses either

\[
\arg\max(C+q)
\]

or the integer virtual-count form

\[
\arg\max\left(C+\operatorname{round}(q-\min q)\right).
\]

This ordering is important: count bias is evaluated only after the frozen Linear `W` has first been placed into a non-degenerate LIF operating regime.

## Primary comparisons

The finalizer reports method-level summaries plus the following paired contrasts:

- affine Linear vs no-bias Linear;
- exact `b1` attached to `W0` vs `W0` alone;
- same-`W1` affine bias effect (`W1Z+b1` vs `W1Z`);
- selected-gain LIF gap to its corresponding Linear reference;
- recovery from continuous count bias after gain selection.

## Multi-CPU execution

There are three independent Slurm array tasks, one per seed. Each task performs both Linear-head reproductions, both complete gain sweeps, validation gain selection, count-bias fitting, and per-run evaluation. One CPU core is used per task. The `afterok` finalizer only aggregates completed artifacts.

The notebook is analysis-only. It reads finalized CSV/JSON outputs and shows method-level comparisons and gain-response curves; it does not retrain or dump every individual run as separate notebook output.

Run with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_8_cpu.bash
```
