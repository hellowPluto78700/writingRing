# Exp7.3.1 Frozen-L2 Linear Optimization

This experiment reuses the committed Exp7.3 frozen-L2 caches and trains only the final bias-free `128 -> 12` Linear weight matrix.

It fully crosses:

- backbone objective: `tsce`, `wcce`,
- head objective: `tsce`, `wcce`,
- seed: `11`, `23`, `37`,
- Linear optimization case: C0–C4 plus Ref.

Every selected Linear matrix is then evaluated **unchanged** through the standard `beta=0.5`, `threshold=0.5`, `cap=1` output LIF. No LIF retraining is performed.

See `docs/plans/EXP7_3_1_FROZEN_L2_LINEAR_OPTIMIZATION.md` for the frozen protocol.

## Launch

From the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_1_cpu.bash
```

Execution graph:

```text
12 Adam shared C0/C1 tasks -----------\
12 C2 LBFGS tasks --------------------+\
60 C3 regularization candidates ------+--> finalizer
60 C4 scaled candidates --------------+/
60 Ref converged candidates ----------/
```

All jobs are CPU-only, one core per task, and candidate arrays cap concurrency at 50.

## Final artifacts

Under:

`notebooks/artifacts/experiment_7_3_1_frozen_l2_linear_optimization/frozen_l2_linear_optimization_v1/`

The finalizer writes:

- `manifest.json`,
- `method_runs.csv` — 72 selected case/seed rows,
- `method_summary.csv`,
- `selected_regularization.csv`,
- `contrast_runs.csv`,
- `contrast_summary.csv`.

The notebook is analysis-only and reads these finalized files.
