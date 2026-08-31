# WithGyro experiments

## Experiment 0.1 — 60-event temporal representation probe

This experiment evaluates the derived Angular66 action-0 dataset using only event channels `0:60`:

- `0:30`: existing linear-acceleration polarity-split wavelet events;
- `30:60`: angular-acceleration polarity-split wavelet events;
- `60:66`: raw acceleration and gyroscope, excluded from the classifier input.

The experiment compares channel-wise event-count temporal representations:

- fixed duration: 50 ms and 150 ms requested bins (3 and 10 samples at 64 Hz; 46.875 ms and 156.25 ms actual);
- relative progress: 1, 2, 4, 6, 8, 10, 12, 16, and 20 bins.

It uses five user-disjoint split seeds `(11, 23, 37, 53, 71)`, train-only feature standardization, Logistic Regression and 5-NN, and reports Accuracy, Balanced Accuracy, and Macro-F1 on validation and test users.

### Multi-CPU execution

The sweep has 55 independent tasks:

```text
5 split seeds x (2 fixed-duration + 9 relative-progress conditions) = 55 tasks
```

Each Slurm array task uses one CPU core. Array concurrency is capped at 50. Each task builds one representation/split, fits both classifiers on the same standardized training features, evaluates validation/test, and writes one JSON artifact. An `afterok` finalizer aggregates existing run artifacts into CSV/JSON outputs; it does not retrain.

From the repository root on Unity:

```bash
bash scripts/bash_script/withGyro/submit_exp_0_1_pipeline.bash
```

Monitor with:

```bash
squeue -u "$USER"
```

The submit script also prints the array and finalizer job IDs. After completion, inspect:

```text
notebooks/artifacts/withGyro/experiment_0_1_temporal_representation_probe/linear_angular_accel_60event_v1/
```

Then open and run:

```text
notebooks/withGyro/experiment_0_1_temporal_representation_probe.ipynb
```

The notebook is analysis-only and requires the finalizer outputs.
