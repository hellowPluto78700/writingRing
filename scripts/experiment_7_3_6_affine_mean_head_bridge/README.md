# Exp7.3.6 — Affine mean-head bridge on the frozen WCCE backbone

## Scientific question

Exp7.3 showed that the frozen WCCE backbone is read well by a retrained bias-free Linear/WCCE head (B6), but its legacy/P7 affine probe remains stronger. Exp7.3.6 asks which part of that gap comes from the head formulation rather than the L2 representation itself.

All runs reuse the exact Exp7.3 WCCE L2 cache sourced from `A2_e2e_linear_wcce`; L1/L2 are never retrained.

The primary training input remains the valid temporal mean

\[
\bar z = \frac{1}{T}\sum_t z_t,
\]

because Exp7.2.6.1 already showed that Mean-CE avoids the sample-dependent logit scaling introduced by raw Sum-CE.

## 2 x 2 bridge

Four paired head formulations are trained for seeds `11,23,37`:

| Case | Train-only conditioning | Bias | Final raw form |
|---|---|---:|---|
| `H0_raw_no_bias` | none | no | `W z_bar` |
| `H1_raw_bias` | none | yes | `W z_bar + b` |
| `H2_scale_no_bias` | divide by train std only | no | folded `W_raw z_bar` |
| `H3_scale_bias` | divide by train std only | yes | folded `W_raw z_bar + b` |

No centering is used. For the scaled cases, training uses

\[
\tilde z = D^{-1}\bar z,
\]

then folds the scaler into the raw weight matrix:

\[
W_{raw}=W_{scaled}D^{-1}.
\]

Therefore deployment never requires a runtime scaler.

Within one seed, all four cases reuse the same Exp7.3 Stage-2 model-init seed and sample order. Scaled cases are initialized so that their epoch-0 raw function equals the unscaled initialization:

\[
W_{scaled,0}=W_0D,
\]

which guarantees

\[
W_{scaled,0}D^{-1}\bar z=W_0\bar z.
\]

## Primary contrasts

- `H1 - H0`: pure bias effect on raw mean features.
- `H2 - H0`: pure scale-conditioning effect without bias.
- `H3 - H2`: bias effect after conditioning.
- `H3 - H0`: full probe-inspired affine/conditioning bridge.

For `H1/H3`, the selected checkpoint is also evaluated with the same trained `W` but bias disabled. This separates direct affine-offset benefit from any indirect effect of allowing bias during optimization.

## References

The finalizer reads the committed Exp7.3.2 summary when available and records:

- P3 mean-affine probe test BA: the closest affine function-class reference;
- P7 whole-count affine probe test BA: the broader legacy accessibility reference.

P7 is not treated as an exact target formulation because with bias

\[
WZ+b \neq T(W\bar z+b).
\]

## LIF realization diagnostics

The primary scientific question is analog head formulation. LIF evaluation is secondary and never affects training or checkpoint selection.

For every selected raw `(W,b)`:

1. analog mean-affine reference: `W z_bar + b`;
2. beta=0.5 LIF with the affine bias injected once per valid timestep:
   \[
   I_t=Wz_t+b;
   \]
3. beta=1 IF spike-count control with the same repeated bias;
4. beta=1 charge-preserving readout
   \[
   Q=\theta N+U_T.
   \]

For the beta=1 control the implementation asserts

\[
Q=\sum_t(Wz_t+b)=WZ+Tb
\]

within numerical tolerance. Since

\[
WZ+Tb=T(W\bar z+b),
\]

the charge-preserving argmax must match the analog mean-affine classifier.

## Multi-CPU execution

The experiment follows the repository default task-level CPU pattern:

```text
3 seeds x 4 cases = 12 independent training tasks
        |
        +-> one CPU core per task
        +-> train -> select checkpoint -> evaluate -> write artifacts
        |
        +-> one afterok finalizer aggregates existing artifacts only
```

The worker array is capped at 12 concurrent tasks. Every compute job initializes Conda locally and sets BLAS/OpenMP threads to one.

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_6_cpu.bash
```

## Final artifacts

Finalized artifacts are written under

```text
notebooks/artifacts/experiment_7_3_6_affine_mean_head_bridge/affine_mean_head_bridge_v1/
```

Important outputs:

- `manifest.json`
- `method_runs.csv`
- `method_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`
- `bias_ablation.csv`
- `lif_realization_summary.csv`
- `source_reproduction_checks.json`
- `history_runs.csv`
- `history_summary.csv`
- per-run `evaluations/*.json`
- per-run `checkpoints/*.pt`
- per-run `histories/*.csv`

## Notebook contract

`notebooks/experiment_7_3_6_affine_mean_head_bridge.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts and shows method-level comparisons only. It does not import Torch, invoke Slurm, train models, or regenerate missing artifacts.
