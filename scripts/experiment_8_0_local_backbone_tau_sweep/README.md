# Exp8.0 — Local Backbone Temporal-Scale Sweep

Exp8.0 compares five two-layer SNN temporal-scale backbones under the exact Exp7.3 A2 Linear/WCCE training interface.

## Backbones

```text
234x234
123x234
123x123
123x345
234x345
```

Each backbone is trained for seeds `11, 23, 37`, for 15 total runs.

## What every run produces

Each CPU-array task performs the complete run:

1. end-to-end A2-style Linear/WCCE training;
2. native Linear train/val/test evaluation;
3. frozen same-W transfer to output LIF (`alpha=0`, `beta=0.5`), with no retraining;
4. L1 whole-count affine probe;
5. L1 ordered Fixed250 affine probe;
6. L2 whole-count affine probe;
7. L2 ordered Fixed250 affine probe;
8. per-shift firing/activity diagnostics;
9. a deterministic test-sample L1 raster;
10. the corresponding L2 raster;
11. the corresponding output-LIF raster;
12. checkpoint, history, evaluation JSON, and compressed raster trace.

All rasters use the same representative test sample definition across methods and seeds.

## Launch on Unity

From the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_cpu.bash
```

The submission script launches the 15-way CPU array and then an `afterok` finalizer.

To inspect the array mapping without training:

```bash
python -m scripts.experiment_8_0_local_backbone_tau_sweep list-runs
```

## Final artifacts

The finalizer writes under:

```text
notebooks/artifacts/experiment_8_0_local_backbone_tau_sweep/local_backbone_tau_sweep_v1/
```

Aggregate outputs:

```text
method_runs.csv
method_summary.csv
contrast_runs.csv
contrast_summary.csv
activity_runs.csv
activity_summary.csv
raster_index.csv
manifest.json
```

Per-run outputs include:

```text
checkpoints/
histories/
evaluations/
rasters/
```

## Notebook

Use:

```text
notebooks/experiment_8_0_local_backbone_tau_sweep.ipynb
```

The notebook is aggregation-only. The main section compares methods using finalized aggregate files; the raster appendix displays L1, L2, and output-LIF rasters for every one of the 15 trained runs.

See `docs/plans/EXP8_0_LOCAL_BACKBONE_TAU_SWEEP.md` for the scientific protocol and interpretation rules.
