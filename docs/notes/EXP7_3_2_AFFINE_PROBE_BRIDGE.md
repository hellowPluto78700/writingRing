# Experiment 7.3.2 — Affine Probe Bridge

## Question

Exp7.3.1 showed that longer Adam training and converged LBFGS do not fully close the gap between deployment-compatible Linear heads and the legacy Exp7.3 L2 whole-count probe. Exp7.3.2 isolates the remaining formulation differences in the legacy probe:

1. sequence aggregation: valid-time mean vs valid whole-count;
2. centering: `StandardScaler(with_mean=False)` vs `with_mean=True`;
3. affine freedom: `fit_intercept=False` vs `fit_intercept=True`.

All cases use per-feature standard-deviation scaling, the same sklearn multinomial LogisticRegression solver, the same Exp7.3 probe C grid, and validation balanced accuracy for model selection.

## Frozen representation

Reuse the Exp7.3 stage-2 frozen-L2 caches for both source backbones:

- `backbone_tsce`: A1 E2E Linear + TSCE backbone;
- `backbone_wcce`: A2 E2E Linear + WCCE backbone.

Seeds: `11, 23, 37`.

No SNN weights are retrained.

## 2 × 2 × 2 bridge matrix

For each frozen backbone and seed:

| Case | Aggregation | Scale by std | Center | Intercept |
|---|---|---:|---:|---:|
| P0 | mean | yes | no | no |
| P1 | mean | yes | no | yes |
| P2 | mean | yes | yes | no |
| P3 | mean | yes | yes | yes |
| P4 | whole-count | yes | no | no |
| P5 | whole-count | yes | no | yes |
| P6 | whole-count | yes | yes | no |
| P7 | whole-count | yes | yes | yes |

P7 is intentionally an exact reproduction of the legacy Exp7.3 L2 whole-count probe protocol:
valid L2 spike count -> `StandardScaler()` -> `LogisticRegression(solver="lbfgs", max_iter=5000, fit_intercept=True)` -> validation-selected C.

## Classifier protocol

For every case:

- fit scaler on train only;
- use `C in PROBE_C_GRID`;
- fit one sklearn LogisticRegression per C;
- choose the first C achieving the highest validation balanced accuracy, matching the legacy helper;
- test data is untouched until the selected classifier is reported.

The effective raw-space classifier is also recorded:

```math
s = W_{raw} x + b_{eff}.
```

For centered cases,

```math
b_{eff}=b_{explicit}-W_{scaled}(\mu/\sigma).
```

This separates an explicit learned intercept from the offset induced by centering.

## Primary contrasts

- Centering effect with bias off/on, separately for mean and whole-count.
- Bias effect with centering off/on, separately for mean and whole-count.
- Whole-count vs mean at matched centering/bias settings.
- Full bridge: `P7 - P0`.
- P7 reproduction error relative to the stored Exp7.3 legacy whole-count probe.

## Multi-CPU execution

There are `3 seeds × 2 backbones × 8 cases = 48` independent runs.

One Slurm array task handles one `(seed, backbone, case)` condition and performs its five-C sweep locally. Each task uses one CPU core and writes one JSON evaluation. Array concurrency is capped at 48. A single `afterok` finalizer only aggregates existing results; it never retrains or regenerates missing runs.

Run from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_2_cpu.bash
```

## Final artifacts

`notebooks/artifacts/experiment_7_3_2_affine_probe_bridge/affine_probe_bridge_v1/`

- `evaluations/*.json`
- `method_runs.csv`
- `method_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`
- `p7_legacy_reproduction.csv`
- `manifest.json`

The notebook `notebooks/experiment_7_3_2_affine_probe_bridge.ipynb` is analysis-only and reads these finalized artifacts.
