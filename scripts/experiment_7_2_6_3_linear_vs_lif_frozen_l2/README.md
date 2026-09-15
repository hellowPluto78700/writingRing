# Exp7.2.6.3 — Frozen-L2 Linear vs LIF decomposition

## Scientific question

Why do a Linear head and LIF output neurons produce substantially different classification BA when they receive the same L2 spike sequence?

This experiment intentionally freezes the Exp7.2.6.1 Mean-CE, bias-free `234x234` L2 representation and separates:

1. continuous-to-spike-count conversion loss;
2. membrane leakage loss;
3. Linear-trained vs LIF-trained output projection quality.

It does **not** retrain the SNN backbone end-to-end.

## Fixed contract

- backbone source: Exp7.2.6.1 Mean-CE, `bias=False`, `task_only`;
- architecture: `234x234 = (2,3,4) -> (2,3,4)`;
- seeds: `11, 23, 37`;
- valid-length masking only;
- output synaptic alpha: `0`;
- native LIF beta: `0.5`;
- threshold/reset/cap inherited from Exp7.2.6 (`cap=1`);
- Linear and LIF head training both use `CE(valid_mean(...), y)`;
- **CE logit gain is exactly `1.0` for both Linear and LIF**;
- primary LIF input gain is exactly `1.0`.

The validation-calibrated LIF input gain is reported only as a secondary amplitude/threshold diagnostic and is not the main result. It is distinct from CE logit gain.

## Part A — same L2, same W

Uses the already-trained Exp7.2.6.1 `W_linear` without retraining:

- `A0`: Linear analog accumulator;
- `A1`: beta=1 IF, charge-preserving `theta*N + U_T`;
- `A2`: same beta=1 trajectory, pure spike count;
- `A3`: beta=.5 LIF, pure spike count.

Primary decomposition:

- spike conversion penalty = `A1 - A2`;
- leakage penalty = `A2 - A3`;
- total Linear-to-LIF gap = `A0 - A3`.

A beta sweep `1,.95,.9,.8,.7,.6,.5` is performed with `W_linear`, cap=1, input gain=1 fixed.

Secondary diagnostics include validation-calibrated input gain, cap31, and bipolar output controls.

## Part B — same L2, separately trained W

Freeze the same L2 cache and train two paired `128 -> 12`, bias-free heads from the same initial W and with the same loader order:

- Linear: `CE(valid_mean(Wz), y)`;
- LIF: `CE(valid_mean(S), y)`;
- CE logit gain = `1.0` for both.

Then perform the 2x2 cross-check:

| projection | Analog readout | LIF beta=.5 readout |
| --- | --- | --- |
| `W_linear` | X1 | X2 |
| `W_lif` | X3 | X4 |

Interpretation:

- `X1-X3`: projection-learning gap;
- `X1-X2`: LIF dynamics loss for Linear-trained W;
- `X3-X4`: LIF dynamics loss for LIF-trained W.

## Multi-CPU execution

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_6_3_cpu.bash
```

Dependency graph:

```text
cache[3]
  |-- mechanism[3] --------------------|
  `-- paired head training[6] -> cross[3]
                                      |
                                 finalizer
```

The finalizer hard-fails if any required run is missing.

## Notebook contract

`notebooks/experiment_7_2_6_3_linear_vs_lif_frozen_l2.ipynb` is analysis-only. It reads only final aggregate CSV/JSON outputs and reports method-level comparisons; it does not train models, run inference, submit Slurm jobs, or display every run individually.
