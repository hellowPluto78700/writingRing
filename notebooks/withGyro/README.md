# WithGyro experiments

## Experiment 0.1 — paired channel-ablation temporal representation probe

This experiment evaluates the derived Angular66 action-0 dataset with three paired event-channel conditions from the same samples and user splits:

- `accel30` = channels `0:30`: linear-acceleration polarity-split wavelet events;
- `angular30` = channels `30:60`: gyro-derived angular-acceleration polarity-split wavelet events;
- `combined60` = channels `0:60`: concatenated `accel30 + angular30`;
- channels `60:66`: raw acceleration and raw gyroscope, excluded from all classifier inputs.

The main gyro-contribution comparison is `combined60 - accel30`. Because all three channel sets use the same split seed, representation condition, samples, labels, and classifiers, gains are computed within each split first and then aggregated across the five splits.

The temporal representation sweep is:

- fixed duration: 50, 150, 250, 350, 450, 550, 650, 750, 850, 950, and 1050 ms requested bins;
- relative progress: 1, 2, 4, 6, 8, 10, 12, 16, and 20 bins.

At 64 Hz, fixed-duration requests quantize to 3, 10, 16, 22, 29, 35, 42, 48, 54, 61, and 67 samples per bin. Requested and actual durations are both saved.

The experiment uses five user-disjoint split seeds `(11, 23, 37, 53, 71)`, 12/4/4 train/validation/test users, Logistic Regression and 5-NN, and reports Accuracy, Balanced Accuracy, and Macro-F1. Each channel set is standardized independently using only its training-user features.

### Unity Conda environment

The launchers pin the supported environment to:

```text
/work/pi_jgummeso_umass_edu/$USER/.conda/envs/writingring-gpu
```

`submit_exp_0_1_pipeline.bash`, every Slurm worker, and the finalizer independently run:

```bash
module load conda/latest
eval "$(conda shell.bash hook)"
conda activate "$WRITINGRING_CONDA_PREFIX"
```

where `WRITINGRING_CONDA_PREFIX` defaults to the path above. They verify that `numpy`, `pandas`, and `sklearn` import before experiment execution.

### Multi-CPU execution

The sweep remains 100 independent Slurm tasks:

```text
5 split seeds x (11 fixed-duration + 9 relative-progress conditions) = 100 tasks
```

Each task uses one CPU core and runs all three channel sets and both classifiers internally:

```text
one task
├── accel30    -> Linear + 5NN
├── angular30  -> Linear + 5NN
└── combined60 -> Linear + 5NN
```

Thus each task performs six model fits while preserving exact paired comparisons. Array concurrency remains capped at 50 with `--array=0-99%50`. An `afterok` finalizer aggregates existing artifacts only; it does not retrain.

From the repository root on Unity:

```bash
bash scripts/bash_script/withGyro/submit_exp_0_1_pipeline.bash
```

Monitor with:

```bash
squeue -u "$USER"
```

Final artifacts are written under:

```text
notebooks/artifacts/withGyro/experiment_0_1_temporal_representation_probe/linear_angular_accel_channel_ablation_v3/
```

Key files:

```text
experiment_0_1_results.csv
experiment_0_1_summary.csv
experiment_0_1_paired_gains.csv
experiment_0_1_paired_gain_summary.csv
experiment_0_1_split_assignments.csv
provenance.json
```

Then open:

```text
notebooks/withGyro/experiment_0_1_temporal_representation_probe.ipynb
```

The notebook is analysis-only and reads the finalized v3 artifacts.
