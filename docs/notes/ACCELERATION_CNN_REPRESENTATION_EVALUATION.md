# Experiment A: raw-acceleration CNN representation evaluation

`notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb` is
an independent interactive experiment for raw-acceleration-only classification
and representation evaluation. It reads completed padded SpikeIMU packages,
trains a 1-D CNN on the producer's three acceleration channels, and evaluates
the learned 256-dimensional representation with held-out users. This is the
baseline checkpoint producer for Experiment B.

It is not part of the Action-0 SynNet CLI baseline and does not change that
training pipeline. It supports every complete `user_*/action_*` package found
under the selected padded dataset, rather than being limited to `action_0`.

## Input representation and scope

The notebook reads:

```text
<padded-root>/<user>/action_<id>/<user>_action_<id>_paddedSpikeIMU.npy
```

with the fixed producer shape `(S, T_pad, 21)`. Its model input is exactly:

```python
paddedSpikeIMU[..., 15:18]
```

That slice is the producer-copied x/y/z acceleration in m/s². The resulting
CNN tensor is `(B, 3, T_pad)`. Event channels `0:15`, the three gyroscope
channels `18:21`, and any Board alignment targets are not model inputs.

This is deliberately different from the optional reconstruction artifacts
made by `reconstruct_padded_spike_accel.py`:

```text
<prefix>_padded_reconstructed_accel_m_s2.npy  # (S, T_pad, 3)
```

The notebook does **not** read that file. Reconstruction is event-derived
acceleration, whereas this experiment uses the acceleration already present in
the final six SpikeIMU channels. Comparing those two representations requires
a separately configured experiment; neither is silently substituted for the
other.

Experiment B is that separate experiment: it loads this notebook's saved
checkpoint and evaluates reconstructed held-out test inputs without retraining
the CNN or re-estimating its normalization. Its complete protocol is documented
in [Experiment B: reconstructed acceleration with the frozen Experiment A
CNN](RECONSTRUCTION_FROZEN_CNN_EVALUATION.md).

## Padded-package contract

Set the notebook's `ROOT` configuration before running. It may identify the
`segmentation_padded/` directory itself, its combination root, or a higher
directory that contains exactly one padded dataset summary. The resolved root
must contain `padding_dataset_summary.json` and satisfy the normal padded
SpikeIMU contract:

- `input_kind` is `spike-imu`, feature schema is
  `signed_wavelet_events_plus_imu_v1`, and `channel_count` is 21;
- `target_length` and `sampling_rate_hz` are valid, and padding is on the
  right;
- every package supplies `paddedSpikeIMU`, labels, `valid_lengths`,
  `valid_mask`, its padding summary, and its padding manifest;
- values are finite; `paddedSpikeIMU` is `(S, T_pad, 21)`; labels and valid
  lengths are `(S,)`; and the boolean mask is `(S, T_pad)`;
- each mask is the contiguous valid prefix implied by its length, and package
  summaries agree on `T_pad`, right padding, and `overflow_policy=skip`;
- the manifest provides a contiguous `output_segment_index` that maps every
  retained padded row to its source segment.

Segments skipped by the padding producer are absent from axis 0 and therefore
absent from the experiment. The notebook reads the large array values through
memory maps, but validates the package metadata and row correspondence before
training.

See [Segment padding](SEGMENT_PADDING.md) for the producer contract and
[padded acceleration reconstruction](SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md)
for the separate event-derived output.

## Split, normalization, and model protocol

The split unit is the user: all actions, packages, and segments for one user
belong to exactly one of train, validation, or test. By default the notebook
uses a seeded 70% / 15% / remaining-user automatic split. Alternatively, set
all three explicit user lists; setting only some lists is rejected. The
configuration also requires every discovered user to be assigned and, by
default, every class to occur in all three splits.

Experiment A's checkpoint is the seed authority for the B/C/D runs. When a
reference A checkpoint is supplied, those runs inherit its `random_seed`, and
the top-level experiment seed must also match the evaluation seed. An explicit
different seed is rejected so that split assignment, loader shuffling, and
evaluation randomness cannot silently diverge across experiments. The linear
probe's configured `seed_offset` remains an intentional local offset from this
shared base seed.

Acceleration mean and standard deviation are calculated only from valid time
steps of train users. Each sample is normalized with those train statistics,
then its invalid right-padded positions are reset to zero. `valid_mask` is
propagated through the convolutional time dimension and used for masked global
average pooling, so right padding cannot contribute signal or pooling weight.

The CNN has four `Conv1d` stages (3→64→128→256→256), a 256-D masked pooled
embedding, and a linear classification head. The best checkpoint is selected
only by validation balanced accuracy.

After loading that checkpoint, the notebook reports classifier metrics and
representation diagnostics. Retrieval uses held-out test-user queries against
the train-user gallery; kNN `K` is selected only with validation users against
that gallery. Additional diagnostics include same-label-at-K, mAP, class
prototypes, a linear probe, K-means NMI/ARI/silhouette, and a PCA plot. The
notebook implements these analyses with NumPy and PyTorch, without a
scikit-learn dependency.

## Running and outputs

Install the optional notebook/training dependencies listed in `pyproject.toml`,
open the notebook from the repository checkout, set its experiment
configuration, then run it from the configuration and data-loading cells
onward. Its output switches default to `True`:

```text
SAVE_ARTIFACTS = True
SAVE_FIGURES = True
OUTPUT_DIR = notebooks/artifacts/acceleration_cnn_representation/
```

With artifact saving enabled, the output directory contains:

- `best_acceleration_cnn.pt`: model state, architecture, class mapping,
  normalization values, split users, selected epoch, and producer metadata;
- `embeddings_train.npz`, `embeddings_val.npz`, and `embeddings_test.npz`:
  penultimate embedding `h`, L2-normalized embedding `z`, labels, CNN
  predictions, logits, and stable sample IDs;
- source and embedding manifests, training and linear-probe histories, kNN
  selection/retrieval tables, and per-class diagnostic/result CSV files;
- `experiment_provenance.json`: resolved dataset, producer metadata, split,
  normalization, training, model, retrieval protocol, and result summary;
- enabled Matplotlib figures, including length/split distributions, learning
  curves, test confusion matrix, retrieval curve, intra-class distances, and
  test embedding PCA.

Rerunning overwrites the fixed-name files that it writes, but it does not clear
the output directory first. Preserve or redirect `OUTPUT_DIR` before running
experiments that need separate provenance, and remove obsolete optional files
yourself if a clean artifact directory is required.

## Relationship to Experiment B and Action-0 SynNet training

Experiment B consumes the `best_acceleration_cnn.pt` produced here, along with
the checkpoint's class mapping, user split, and raw-train normalization. Run
this notebook with `SAVE_ARTIFACTS=True` before Experiment B. Experiment B does
not replace this raw-input baseline or train a CNN on reconstructed data.

`python -m snn.train_action0` and `notebooks/action0_snn_training.ipynb` are
SynNet experiments over SpikeIMU event channels `0:15`; their producer and
split requirements are documented in [Action-0 SynNet training](ACTION0_SNN_TRAINING.md).
This CNN notebook shares the padded-package validation and user-disjoint
principle, but has a different input modality, model, class scope, and
evaluation goals.
