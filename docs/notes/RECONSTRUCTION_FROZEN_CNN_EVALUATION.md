# Experiment B: reconstructed acceleration with the frozen Experiment A CNN

`notebooks/experiment_B_reconstruction_frozen_cnn.ipynb` tests whether padded
event-derived reconstructed acceleration preserves the information used by the
raw-acceleration CNN trained in [Experiment A](ACCELERATION_CNN_REPRESENTATION_EVALUATION.md).
It is an evaluation-only experiment: it never trains or fine-tunes a CNN on
reconstructed acceleration.

## Required inputs

Set `ROOT` to the same padded dataset used by Experiment A, and set
`BASELINE_CHECKPOINT_PATH` to Experiment A's saved
`best_acceleration_cnn.pt`. The checkpoint supplies the architecture, class
mapping, user-disjoint train/validation/test split, and raw-train acceleration
normalization statistics. Run Experiment A with `SAVE_ARTIFACTS=True` first.

For every selected `*_paddedSpikeIMU.npy` package, the notebook requires its
labels, `valid_lengths`, `valid_mask`, padding summary and manifest, plus the
corresponding outputs of `scripts/reconstruct_padded_spike_accel.py`:

```text
<prefix>_padded_reconstructed_accel_m_s2.npy       # (S, T_pad, 3)
<prefix>_padded_reconstructed_accel_metadata.json
```

The reconstructed array must have the same segment and time axes as the raw
padded package, three acceleration channels, finite values, and exact zero
right padding. Its metadata must identify the expected padded,
segment-wise Custom Wavelet reconstruction and m/s² output. See [padded
acceleration reconstruction](SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md) for the
producer contract.

## Frozen evaluation protocol

Experiment B evaluates two matched held-out test inputs with the exact same
frozen CNN:

```text
raw paddedSpikeIMU[..., 15:18]       -> frozen CNN -> raw reference metrics
reconstructed acceleration (S,T,3)   -> frozen CNN -> Experiment B metrics
```

It loads, rather than recalculates, the raw-train mean and standard deviation
from the Experiment A checkpoint. The checkpoint's train, validation, and test
users are authoritative. The notebook validates reconstructed artifacts for
every discovered padded package during its paired-package preflight, although
it evaluates reconstructed inputs only for held-out test users. Sample order,
labels, valid lengths, and masks must match between raw and reconstructed test
inputs.

The downstream reference data remains raw throughout: retrieval and prototype
classification use raw-train embeddings; kNN `K` is selected with raw
validation queries against that gallery; and the linear probe is trained on
raw-train embeddings and selected on raw-validation embeddings. These controls
prevent adaptation to the reconstruction distribution.

## Outputs and interpretation

By default the notebook writes artifacts and figures to:

```text
notebooks/artifacts/experiment_B_reconstruction_frozen_cnn/
```

Saved artifacts include the direct classification, retrieval, geometry,
linear-probe, and unsupervised-comparison CSV files; paired embedding and
prediction-preservation tables; reconstructed test embeddings; the paired test
manifest; raw-validation kNN selection; and `experiment_B_provenance.json`.

Interpret frozen-CNN accuracy together with prediction agreement, paired
raw-to-reconstruction embedding cosine similarity, retrieval, and geometry
results. A drop in frozen-CNN accuracy alone can indicate raw-to-reconstruction
domain shift as well as lost task information; it is not evidence by itself
that the reconstruction destroyed information.
