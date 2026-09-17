# Exp8.0.3 — Phase-aware hierarchical L1/L2 readout

## Question

Exp8.0.2 showed that direct joint supervision substantially improves the frozen L1/L2 representations, and that the strongest post-hoc probe is approximately

\[
\mathrm{Fixed250}(L1)+\mathrm{Whole}(L2).
\]

Exp8.0.3 asks whether that information can be made native to the trained classifier while preserving a per-timestep evidence stream that can be transferred to the existing output LIF.

## Fixed backbone

All methods use the same Exp8.0 / Exp7.3-A2 compatible local backbone:

- input: 30 unsigned event channels at 64 Hz;
- L1: 128 binary neurons, shifts `{2,3,4}`;
- L2: 128 binary neurons, shifts `{2,3,4}`;
- bias-free classification heads;
- end-to-end Adam training with the existing A2 learning rate, weight decay, checkpoint rule, minimum epoch, patience, and 100-epoch maximum;
- seeds: `11, 23, 37`.

The only independent variable is the readout topology.

## Methods

### A. `l2_only`

Existing A2 control:

\[
e_t=W_2 z^{(2)}_t.
\]

### B. `l1_l2_timeshared`

Exp8.0.2 joint-readout control:

\[
e_t=W_1 z^{(1)}_t+W_2z^{(2)}_t.
\]

The same `W1` is used at every timestep.

### C. `l1_fixed250_l2_whole`

Primary phase-aware method. With 64-Hz data, `bin_steps=16` for 250 ms. Let `b(t)` denote the absolute 250-ms bin containing timestep `t`:

\[
e_t=W_{1,b(t)}z^{(1)}_t+W_2z^{(2)}_t.
\]

The valid-length native score is

\[
s=\frac{1}{T}\sum_{t<T}e_t.
\]

Because all heads are bias-free, the class argmax is identical to the count-form score

\[
\sum_b W_{1,b}\,\mathrm{Fixed250}_b(z^{(1)})
+W_2\,\mathrm{Whole}(z^{(2)}).
\]

This provides the desired `L1 Fixed250 + L2 Whole` objective while still producing a 12-D evidence vector at every timestep.

### D. `l1_capacity_no_phase_l2_whole`

Parameter-count control for method C. It owns the same `B x 12 x 128` L1 weight bank, but receives no bin identity. The effective time-shared L1 weight is

\[
W_{1,\mathrm{eff}}=\sum_bW_{1,b},
\]

so

\[
e_t=W_{1,\mathrm{eff}}z^{(1)}_t+W_2z^{(2)}_t.
\]

Equivalently, this applies the large L1 linear head to `Whole(L1)` repeated into every Fixed250 slot. Therefore C vs D isolates phase access from the larger head parameter count.

## Objective

All four methods use one task loss only:

\[
\mathcal L=CE(s,y),\qquad
s=\frac{1}{T}\sum_{t<T}e_t.
\]

No auxiliary loss or regularizer is introduced.

## Linear/readout equivalence check

For every trained run, Exp8.0.3 explicitly recomputes the feature-form score from masked L1/L2 counts and checks it against the accumulated per-timestep evidence score. The maximum absolute discrepancy is saved. This guards the central claim that the phase-aware native head is exactly realizable as a time-varying per-timestep evidence stream.

## Output-LIF transfer

The same per-timestep evidence `e_t` is injected into the existing output LIF:

- `alpha_out = 0`;
- `beta_out = 0.5`;
- threshold `0.5`;
- binary/cap-1 spike output;
- final prediction from valid-length output spike count.

No head is retrained for the LIF evaluation.

## Diagnostics

Each run records:

1. native train/val/test BA and CE;
2. same-evidence output-LIF BA and Linear-to-LIF penalty;
3. post-hoc frozen probes used in Exp8.0.1/8.0.2 for L1 whole, L2 whole, L1+L2 whole, L1 Fixed250, L2 Fixed250, L1+L2 Fixed250, and the two mixed readouts;
4. L1/L2 correctness overlap and oracle-union diagnostics;
5. trained-head norms;
6. for the large L1 weight bank: per-bin norm, adjacent-bin cosine, all-pair cosine, and between-bin weight dispersion;
7. branch score RMS for the native L1 and L2 branches;
8. training history.

## Interpretation

The primary comparison is:

\[
\boxed{C-D}
\]

because C and D have the same L1 head parameter count. A consistent positive C-D test-BA difference supports useful absolute 250-ms phase access rather than head capacity alone.

Secondary comparisons:

- C vs B: does explicit phase-aware L1 evidence improve over the Exp8.0.2 time-shared joint head?
- B vs A: reproduction of the Exp8.0.2 joint-supervision effect in the new code path.
- same-W/evidence LIF penalty: does phase-aware evidence become more or less compatible with the leaky output neuron?

If C is strong but the learned bin weights remain nearly identical (high cosine, tiny between-bin dispersion), the improvement should not be interpreted as phase use. Conversely, differentiated bin weights plus C>D provide the stronger mechanism result.

## Compute / Slurm

- 4 methods x 3 seeds = 12 independent CPU jobs;
- one CPU thread per task;
- Slurm array `0-11%12`;
- a dependent `afterok` finalizer validates all 12 runs and writes aggregate CSV/JSON artifacts;
- notebook is aggregation-only and does not train or refit models.
