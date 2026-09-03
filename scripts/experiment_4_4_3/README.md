# Experiment 4.4.3 — Heterogeneous Stage-2 membrane timescales

## Question

Can a Dense-RSNN with heterogeneous Stage-2 membrane time constants improve gesture-level endpoint memory beyond the best homogeneous single-tau architecture, without adding trainable parameters?

## Fixed protocol

- Input: Raw64, 30-channel scaled weighted event sequence.
- Local128: `shift34`, unchanged from Exp4.4.1.
- Stage-2 width: 128.
- Recurrence topology: dense.
- Event capacity: `multi_ho` (`hidden_cap=31`, `output_cap=31`).
- Threshold: 0.5.
- Output membrane time constant: 250 ms.
- Training objective: valid normalized Output WholeCount CE.
- Seeds: 11, 23, 37, 53, 71.
- Same split/scaling and paired initialization protocol as Exp4.4.1.

## New multi-tau profiles

### `tau250_500`

- 64 neurons at 250 ms.
- 64 neurons at 500 ms.

### `tau125_250_500`

The 128 Stage-2 neurons are divided deterministically across the three timescales:

- 43 neurons at 125 ms.
- 43 neurons at 250 ms.
- 42 neurons at 500 ms.

Only fixed per-neuron beta/tau assignment changes. The trainable parameter count is identical to the homogeneous Dense-RSNN.

## Reused controls

The finalizer reuses the exact paired Dense-RSNN rows from Exp4.4.1 for:

- `single250` — current best homogeneous endpoint-memory condition.
- `single500` — secondary homogeneous control.

Only the two heterogeneous profiles are newly trained: `2 profiles × 5 seeds = 10` new runs.

## Readout matrix

Every profile reports:

1. Output WholeCount BA.
2. Hidden WholeCount + train-only scaled Linear BA.
3. Stage-2 endpoint membrane `Uend` + train-only scaled Linear BA.
4. `Uend - HiddenCount` and `Uend - OutputCount` gaps.
5. Stage-2/output firing rates and tail-event fractions.

The primary paired comparison is each heterogeneous profile versus `single250`. Comparisons versus `single500` are also emitted for interpretation.

## Multi-CPU execution

Each new `(profile, seed)` run is one Slurm array task using one CPU core. The task performs train -> best checkpoint -> evaluation -> per-run artifact. The finalizer only aggregates the 10 new artifacts plus the finalized Exp4.4.1 controls.

From repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_4_3_cpu.bash
```

## Final artifacts

Written under:

```text
notebooks/artifacts/experiment_4_4_3_multitau_memory/stage2_multitau_dense_v1/
```

Files:

- `runs.csv`
- `summary.csv`
- `paired_effects.csv`
- `paired_effects_summary.csv`
- `manifest.json`

Use `notebooks/experiment_4_4_3_multitau_memory.ipynb` for aggregation and visualization only.
