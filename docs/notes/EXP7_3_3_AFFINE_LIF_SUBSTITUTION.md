# Exp7.3.3 — Affine Linear to LIF Substitution

## Goal

Measure the output-neuron realization gap after Exp7.3.2 has already recovered the legacy L2 whole-count probe ceiling with an affine classifier.

The source classifier is always Exp7.3.2 P7:

\[
Z=\sum_t z_t,\qquad s_{\rm affine}=WZ+b.
\]

P7 is used because it exactly reproduces the legacy whole-count probe while making the sequence-level affine term explicit after folding the scaler into raw L2 coordinates.

## Parameter source

Exp7.3.2 did not persist sklearn `coef_` and `intercept_`. Exp7.3.3 therefore reconstructs P7 once per seed/backbone using:

- the same frozen Exp7.3 L2 cache;
- the already-selected Exp7.3.2 P7 `C`;
- the same `StandardScaler(with_mean=True, with_std=True)`;
- the same sklearn `LogisticRegression(lbfgs, max_iter=5000, fit_intercept=True)`;
- the same P7 classifier seed.

No hyperparameter is re-selected. The run fails if reconstructed P7 predictions/BA do not reproduce the saved Exp7.3.2 P7 result.

The scaler is folded into raw coordinates:

\[
W_{\rm raw}=W_{\rm scaled}D^{-1},
\]

\[
b_{\rm eff}=b_{\rm explicit}-W_{\rm raw}\mu,
\]

so the reference score is exactly:

\[
s=W_{\rm raw}Z+b_{\rm eff}.
\]

## Readouts

Every readout uses the same recovered `W_raw` and `b_eff`; there is no gain calibration, threshold sweep, beta sweep, or retraining.

1. `analog_affine`: \(WZ+b\).
2. `analog_w_only`: \(WZ\), to isolate removal of the sequence-level affine term before introducing spiking dynamics.
3. `lif_beta05_w_only`: same \(W\), fixed \(\beta=0.5\) LIF spike-count readout, no bias pulse.
4. `lif_beta05_bias_start`: inject \(b\) exactly once at the first valid timestep.
5. `lif_beta05_bias_end`: inject \(b\) exactly once at the last valid timestep.
6. `if_beta1_bias_start_count`: \(\beta=1\) IF spike count with start pulse.
7. `if_beta1_bias_start_charge`: \(\theta N+U_T\) with start pulse.
8. `if_beta1_bias_end_count`: \(\beta=1\) IF spike count with end pulse.
9. `if_beta1_bias_end_charge`: \(\theta N+U_T\) with end pulse.

For the charge-preserving controls, both start and end injection must satisfy:

\[
\theta N+U_T=WZ+b
\]

up to numerical tolerance. This confirms that any remaining difference in the \(\beta=0.5\) LIF cases is caused by the leaky threshold realization rather than missing affine information.

## Execution

There are 6 independent CPU tasks:

\[
3\ seeds\times2\ frozen\ backbones.
\]

Each task reconstructs its P7 classifier once and evaluates all readouts on train/val/test. The finalizer only aggregates completed JSON artifacts. The notebook is analysis-only.

Primary contrasts are:

- `analog_w_only - analog_affine`: cost of removing the sequence-level affine term;
- `lif_w_only - analog_affine`: combined affine-removal + LIF realization gap;
- `lif_start/end - lif_w_only`: benefit of one-shot affine injection;
- `lif_end - lif_start`: timing sensitivity under leakage;
- `IF beta=1 charge - analog_affine`: charge-preserving identity check.
