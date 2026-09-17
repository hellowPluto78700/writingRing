# Exp8.0.5 — Frozen local backbone, phase-conditioned linear readout

## Question

Exp8.0.4 showed two facts at the same time:

1. restoring valid-length-mean CE recovers the local SNN backbone;
2. an end-to-end phase-aware head gives a small native BA gain, but the resulting frozen representation is not better than the time-shared/no-phase alternatives.

Exp8.0.5 removes backbone co-adaptation entirely.

The experiment asks:

\[
\boxed{
\text{Does a good frozen local SNN already contain information whose class meaning depends on temporal phase?}
}
\]

If yes, a downstream context/RSNN module has a clear job: infer context/phase from history and use it to reinterpret local evidence.

## Frozen source backbone

For each seed `11, 23, 37`, reuse the completed Exp8.0.4 checkpoint:

```text
method = l1_l2_timeshared_count
architecture = 234x234
```

The source model is loaded from:

```text
notebooks/artifacts/
  experiment_8_0_4_phase_aware_hierarchical_readout_mean/
    phase_aware_hierarchical_readout_mean_v1/
      checkpoints/
        l1_l2_timeshared_count__234x234__seed{seed}.pt
```

After loading:

- L1 is frozen;
- L2 is frozen;
- the old Exp8.0.4 classifier is not optimized or reused as the Exp8.0.5 head;
- only a new bias-free linear readout is trained.

Therefore any Exp8.0.5 difference is a readout/accessibility difference, not a change in the local representation.

## Frozen features

Let the frozen backbone produce binary spike sequences

\[
z^{(1)}_t,z^{(2)}_t\in\{0,1\}^{128}.
\]

For valid length \(T\):

\[
\mathrm{Whole}(L1)=\sum_{t<T}z^{(1)}_t,
\qquad
\mathrm{Whole}(L2)=\sum_{t<T}z^{(2)}_t.
\]

At 64 Hz, Fixed250 uses 16 timesteps per bin:

\[
\mathrm{Fixed250}_b(L1)
=
\sum_{t\in b,\;t<T}z^{(1)}_t.
\]

Every readout feature is divided by \(T\) before CE, so training uses the same valid-length normalization that fixed Exp8.0.3:

\[
\mathcal L=CE(s,y).
\]

## Methods

### A. `l2_whole`

\[
s=W_2\frac{\mathrm{Whole}(L2)}{T}.
\]

Purpose: frozen-L2 baseline.

### B. `l1_l2_timeshared`

\[
s=
W_1\frac{\mathrm{Whole}(L1)}{T}
+
W_2\frac{\mathrm{Whole}(L2)}{T}.
\]

The L1 feature-to-class mapping is time shared.

Primary interpretation of `B-A`:

\[
\boxed{
\text{Does frozen L1 add useful class evidence without any phase access?}
}
\]

### C. `l1_fixed250_l2_whole_true_phase`

\[
s=
\sum_b
W_{1,b}
\frac{\mathrm{Fixed250}_b(L1)}{T}
+
W_2\frac{\mathrm{Whole}(L2)}{T}.
\]

This exposes correct absolute 250-ms phase identity to the L1 readout.

### D. `l1_fixed250_l2_whole_destroyed_phase`

D has exactly the same effective feature dimension and linear-head parameter count as C.

For every sequence, draw a deterministic, sample-specific, non-zero cyclic phase offset \(r_i\):

\[
b'(t)=(b(t)+r_i)\bmod B.
\]

Then train/evaluate:

\[
s=
\sum_b
W_{1,b}
\frac{\mathrm{Shift}_{r_i}(\mathrm{Fixed250}(L1))_b}{T}
+
W_2\frac{\mathrm{Whole}(L2)}{T}.
\]

Properties:

- all \(B\times128\) L1 phase features remain active;
- C and D have the same effective function capacity and parameter count;
- C and D use identical head initialization at a fixed backbone seed;
- D destroys consistent absolute phase alignment across sequences.

Therefore the primary comparison is:

\[
\boxed{C-D}
\]

rather than the Exp8.0.4 parameter-count-only no-phase control.

## Readout training

Only the new linear head is optimized:

- bias: none;
- optimizer: same Adam LR and weight decay as the current Exp8 line;
- objective: valid-length-normalized CE;
- max epochs: 100;
- same minimum epoch, patience, and validation checkpoint rule used by Exp7.3/Exp8;
- backbone forward pass is always under `no_grad` and all backbone parameters have `requires_grad=False`.

## Branch-contribution diagnostics

For every trained readout, do not retrain. Evaluate the same head three ways:

\[
s_{\rm full}=s_{L1}+s_{L2},
\]

\[
s_{L1-only}=s_{L1},
\]

\[
s_{L2-only}=s_{L2}.
\]

Report:

```text
full_ba
l1_only_ba
l2_only_ba
l1_removal_drop = full_ba - l2_only_ba
l2_removal_drop = full_ba - l1_only_ba
```

The drops are removal diagnostics, not additive attribution scores; L1/L2 can interact.

The implementation also verifies numerically:

\[
s_{\rm full}=s_{L1}+s_{L2}.
\]

## Phase-alignment diagnostic

For the trained true-phase head C, keep weights frozen and circularly shift the L1 phase features at test time:

\[
b\rightarrow(b+\delta)\bmod B,
\qquad
\delta=0,\ldots,B-1.
\]

No retraining is allowed.

The strongest evidence for genuine phase use is:

\[
BA_{\delta=0}
>
BA_{\delta\neq0}
\]

for misaligned phase assignments.

This diagnostic separates "the head has many phase weights" from "the head needs the correct temporal alignment".

## Primary comparisons

### 1. `timeshared_vs_l2`

\[
B-A.
\]

Does frozen L1 add useful information at all?

### 2. `true_phase_vs_timeshared`

\[
C-B.
\]

Does explicit phase access add information beyond a time-shared L1 readout?

### 3. `true_phase_vs_destroyed_phase`

\[
\boxed{C-D}.
\]

Does correct absolute phase alignment matter when effective feature dimensionality and head capacity are matched?

## Interpretation

### Outcome supporting an RSNN/context layer

If:

\[
C>B
\]

and, more importantly,

\[
C>D,
\]

while phase-shifted C degrades away from \(\delta=0\), then:

\[
\boxed{
\text{the frozen local SNN already contains phase-conditionable evidence.}
}
\]

The later RSNN should learn:

\[
(z_t,h_{t-1})\rightarrow\tilde z_t
\]

so history/context replaces explicit absolute bin identity.

### Outcome not supporting phase-conditioned readout

If:

\[
C\approx B
\]

or

\[
C\approx D,
\]

then the Exp8.0.4 phase gain was primarily caused by end-to-end backbone/head co-adaptation. In that case, improve local representation training before adding a recurrent context module.

## Compute / Slurm

- 4 readouts x 3 source-backbone seeds = 12 independent CPU jobs;
- one CPU thread per task;
- Slurm array `0-11%12`;
- dependent `afterok` finalizer;
- each task independently loads its seed-matched frozen Exp8.0.4 checkpoint;
- notebook is aggregation-only and never trains/refits a model.

## Finalized outputs

```text
method_runs.csv
method_summary.csv
branch_ablation_runs.csv
branch_ablation_summary.csv
phase_shift_runs.csv
phase_shift_summary.csv
paired_deltas.csv
paired_delta_summary.csv
history_runs.csv
manifest.json
```

The aggregation notebook should emphasize method-level comparisons, paired seed deltas, branch removal diagnostics, and the true-phase circular-shift curve.
