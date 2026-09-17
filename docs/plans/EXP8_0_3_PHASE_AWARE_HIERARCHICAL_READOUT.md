# Exp8.0.3 — Phase-aware hierarchical L1/L2 readout

## Question

Exp8.0.2 showed that direct joint supervision substantially improves the frozen L1/L2 representations, and its strongest post-hoc probe is approximately

\[
\mathrm{Fixed250}(L1)+\mathrm{Whole}(L2).
\]

Those post-hoc features are **counts**, not valid-time means. Exp8.0.3 therefore asks whether that exact count-form information can be made native to the trained classifier while preserving a per-timestep 12-D evidence stream that can be transferred to the existing output LIF.

## Fixed backbone

All methods use the same local backbone:

- input: 30 unsigned event channels at 64 Hz;
- L1: 128 binary neurons, shifts `{2,3,4}`;
- L2: 128 binary neurons, shifts `{2,3,4}`;
- bias-free classification heads;
- end-to-end Adam training with the existing learning rate, weight decay, checkpoint rule, minimum epoch, patience, and 100-epoch maximum;
- seeds: `11, 23, 37`.

The independent variable is the count-form readout topology. Exp8.0.2 remains the external valid-mean reference; Exp8.0.3 deliberately uses count CE for all four internal methods so phase comparisons are normalization-matched.

## Methods

### A. `l2_only_count`

Count-only L2 control:

\[
e_t=W_2 z^{(2)}_t,
\qquad
s=\sum_{t<T}e_t=W_2\,\mathrm{Whole}(z^{(2)}).
\]

This is **not** the old A2 valid-mean objective; it is the normalization-matched count control for this experiment.

### B. `l1_l2_timeshared_count`

Time-shared L1+L2 count control:

\[
e_t=W_1 z^{(1)}_t+W_2z^{(2)}_t,
\]

\[
s=W_1\,\mathrm{Whole}(z^{(1)})+W_2\,\mathrm{Whole}(z^{(2)}).
\]

The same `W1` is used at every timestep.

### C. `l1_fixed250_l2_whole_count`

Primary phase-aware method. With 64-Hz data, `bin_steps=16` for 250 ms. Let `b(t)` denote the absolute 250-ms bin containing timestep `t`:

\[
e_t=W_{1,b(t)}z^{(1)}_t+W_2z^{(2)}_t.
\]

The native score is the unnormalized valid count accumulator:

\[
\boxed{s=\sum_{t<T}e_t}
\]

which is exactly

\[
\boxed{
 s=
 \sum_b W_{1,b}\,\mathrm{Fixed250}_b(z^{(1)})
 +W_2\,\mathrm{Whole}(z^{(2)})
}
\]

with no division by sequence length. This is the direct train-time version of the strongest Exp8.0.2 post-hoc feature family.

### D. `l1_capacity_no_phase_l2_whole_count`

Parameter-count control for C. It owns exactly the same `B x 12 x 128` L1 weight bank but receives no absolute bin identity. Its effective time-shared L1 weight is

\[
W_{1,\mathrm{eff}}
=
\frac{1}{\sqrt B}\sum_bW_{1,b},
\]

so

\[
e_t=W_{1,\mathrm{eff}}z^{(1)}_t+W_2z^{(2)}_t.
\]

Equivalently, the large linear head receives the same `Whole(L1)` count in every bin block, scaled by `1/sqrt(B)`. The scale is important: it keeps the effective initialization variance and effective gradient-step scale comparable to a conventional time-shared `128 -> 12` head while preserving the same raw parameter count as C.

Therefore C vs D is the primary test of **phase access**, not merely a larger head.

## Objective

Every Exp8.0.3 method uses exactly one task loss:

\[
\boxed{
\mathcal L=CE\left(\sum_{t<T}e_t,y\right)
}
\]

There is no `1/T` normalization, auxiliary CE, or additional regularizer.

This distinction is intentional. The Exp8.0.1/8.0.2 `Whole` and `Fixed250` probes are based on spike counts. A sample-dependent factor `1/T` preserves argmax for a frozen classifier but does **not** preserve CE optimization because it changes logit temperature across examples.

## Linear/readout equivalence check

For every run, Exp8.0.3 independently computes:

1. the valid-time sum of the per-timestep evidence stream; and
2. the explicit count-feature formula.

For C these are

\[
\sum_{t<T}
\left(W_{1,b(t)}z^{(1)}_t+W_2z^{(2)}_t\right)
\]

and

\[
W_1\mathrm{Fixed250}(z^{(1)})+W_2\mathrm{Whole}(z^{(2)}).
\]

The maximum absolute discrepancy is saved for every split. A synthetic CI test also asserts numerical equivalence. This is the key implementation contract for Exp8.0.3.

## Output-LIF transfer

The exact same per-timestep evidence `e_t` is injected into the existing output LIF:

- `alpha_out = 0`;
- `beta_out = 0.5`;
- threshold `0.5`;
- binary/cap-1 spike output;
- final prediction from valid-length output spike count.

No head is retrained for LIF evaluation.

## Diagnostics

Each run records:

1. native train/val/test BA and count-form CE;
2. same-evidence output-LIF BA and Linear-to-LIF penalty;
3. post-hoc frozen probes used in Exp8.0.1/8.0.2 for L1 whole, L2 whole, L1+L2 whole, L1 Fixed250, L2 Fixed250, L1+L2 Fixed250, and the two mixed readouts;
4. L1/L2 correctness overlap and oracle-union diagnostics;
5. trained-head norms;
6. for the large L1 weight bank: per-bin norm, adjacent-bin cosine, all-pair cosine, and between-bin weight dispersion;
7. native L1/L2 branch score RMS;
8. per-split accumulator-vs-explicit-feature equivalence error;
9. training history.

## Interpretation

The primary comparison is

\[
\boxed{C-D}.
\]

C and D have the same raw L1 head parameter count and the same count-form CE. A consistently positive paired C-D test-BA difference supports useful absolute 250-ms phase access.

Secondary comparisons:

- C vs B: explicit phase-aware L1 evidence versus a conventional time-shared L1 branch under the same count objective;
- B vs A: value of adding a time-shared L1 branch under count CE;
- Exp8.0.3 B vs the already-completed Exp8.0.2 joint result: descriptive reference for the effect of count versus valid-mean training, not an internal causal contrast;
- same-evidence LIF penalty: whether phase-aware count evidence is more or less compatible with the leaky output neuron.

If C is strong but the learned bin weights remain nearly identical (high cosine and tiny between-bin dispersion), the result should not be interpreted as phase use. Differentiated bin weights together with C>D are the stronger mechanism result.

## Compute / Slurm

- 4 methods x 3 seeds = 12 independent CPU jobs;
- one CPU thread per task;
- Slurm array `0-11%12`;
- a dependent `afterok` finalizer validates all 12 runs and writes aggregate CSV/JSON artifacts;
- notebook is aggregation-only and does not train or refit models.
