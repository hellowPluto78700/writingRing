# TaskSpec — withGyro Experiment 0.2

## Goal

Add a no-SNN decoder probe over the finalized withGyro action-0 cohort to determine whether information from accel events, gyro-derived angular-acceleration events, and their concatenation is already linearly accessible or is recovered by structured nonlinear temporal interactions or a generic GRU.

## Required behavior

- Reuse Experiment 0.1 cohort, channel semantics, five user-disjoint split seeds `(11, 23, 37, 53, 71)`, and 12/4/4 train/validation/test user split construction.
- Evaluate channel sets `accel30`, `angular30`, and `combined60`; raw IMU channels `60:66` must remain excluded.
- Evaluate only `fixed250` and `relative10` temporal representations.
- Fit one frozen Logistic Regression baseline per channel-set/representation/split condition with the Experiment 0.1 train-only per-feature z-score semantics.
- Compare the frozen Linear baseline with three zero-initialized residual decoders: Local, Transition, and Local+Transition.
- Local and Transition definitions must match the Experiment 3.2 probe, except the input width must support both 30- and 60-channel representations.
- Compare a standalone one-layer unidirectional GRU over the same binned representation. Fixed250 padding must not alter the final valid hidden state.
- Neural inputs use train-only valid-bin per-channel RMS scaling.
- Select neural checkpoints by validation Balanced Accuracy, breaking ties with lower validation CE. Residual epoch 0 is a valid frozen-Linear checkpoint.
- Report Accuracy, Balanced Accuracy, and Macro-F1 and preserve paired split identities for all decoder and sensor comparisons.
- The finalizer must reject missing/mixed runs and must reproduce Experiment 0.1 Linear baselines within 0.01 absolute Balanced Accuracy.

## Preserved contracts

- Angular66 producer metadata and channel meanings are not modified.
- Experiment 0.1 implementation and finalized artifacts are read-only references.
- Fixed250 remains 16 samples/bin at 64 Hz; the final partial valid bin is retained and padded future bins are masked.
- Relative10 uses `np.array_split` on each sample's valid prefix and therefore has ten valid bins.
- No SNN modules are used in Experiment 0.2.

## Execution contract

- One single-core baseline job fits all 30 frozen Linear baselines.
- The baseline job must complete successfully before neural training starts.
- 120 independent neural runs are mapped one-per-Slurm-array-task:
  `3 channel sets x 2 representations x 4 neural decoders x 5 split seeds`.
- Every neural array task uses one CPU core and concurrency is capped at 50.
- A single `afterok` finalizer aggregates existing artifacts only and does not retrain or regenerate missing runs.
- Every Slurm job initializes Conda locally and uses the repository-supported pinned Unity `writingring-gpu` prefix.

## Outputs

Finalized artifacts live under:

```text
notebooks/artifacts/withGyro/
  experiment_0_2_nonlinear_temporal_decoder_probe/
    structured_residual_gru_v1/
```

Required finalized files include the full results, cross-split summary, paired decoder deltas, sensor deltas, nonlinear-synergy deltas, parameter counts, Experiment 0.1 Linear reproduction check, conclusion JSON, and provenance JSON.

## Validation

Mandatory fast checks:

```bash
python -m pytest -q tests/test_repository_source_syntax.py
python -m pytest -q tests/test_with_gyro_experiment_0_2_contract.py
```

The implementation is ready only when these checks pass in an environment with the repository dependencies installed.
