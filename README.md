# WritingRing Dataset and Processing Pipeline

This repository provides a Python 3.11 workflow for inspecting WritingRing
recordings and producing the derived artifacts used by the current
segmentation and Action-0 SynNet experiments.

The sample-data root is `data_sample/data`. Under a user/action directory,
the supported primary Ring input is `*_ring_0.bin`; `*_ring_1.bin` is ignored.
Each primary Ring file is a raw native-endian float64 binary stream reshaped
into rows of seven values: six IMU measurements followed by a timestamp. It
is not a NumPy `.npy` file. Action directory names are identifiers; this
repository does not establish human semantic labels for action IDs.

The raw acquisition-rate contract is not established here. Current processing
commands commonly use a nominal/default 200 Hz configuration; that setting is
not a claim about every source recording's acquisition rate.

## Known raw dataset corrections

The following corrections apply to the raw dataset:

- `data/user_4/0/0_timestamp.txt`: remove the line
  `1720481135986401 wrong`; there should be only one `wrong` marker before the
  following `f` marker.
- `data/user_9/0/0_timestamp.txt`: change the `l` marker timestamp to
  `1720740976674429`.
- `data/user_3/0/2_board_0.gz`: replace the first Board chunk with a valid
  serialized empty chunk; the original six-frame chunk had no press/lift event
  and introduced a timestamp backward jump before `board_1.gz`.

The first two entries are source-label corrections for
`user_4/action_0/dataset_0` and `user_9/action_0/dataset_0`; the Board entry
changes only the serialized contents of the specified Board chunk. None of
these changes alter the Ring binary format or the IMU samples.

`user_17` data is considered unreliable. It may be skipped during processing
or removed from the dataset before running experiments. For the acceleration-
CNN experiments, use the `excluded_users` configuration (exposed as
`EXCLUDED_USERS` in the Experiment A notebook) when keeping the files but
excluding this user from the cohort.

## Current workflow

```text
Ring recording (`*_ring_0.bin`)
  → discovery / inspection / Matplotlib visualization
  → 9-channel preprocessing and gravity handling
  → optional Custom Wavelet encoding
       ├── 15 signed/rectified event channels, or 30 polarity-split channels
       └── 21-channel signed/rectified or 36-channel polarity-split SpikeIMU artifact
  → optional Ring–Board alignment
  → label or aligned-Board-event variable-length segmentation
       ├── optional row-aligned acceleration reconstruction `(N, 3)`
       └── segment-length analysis and right-padding
             └── optional padded acceleration reconstruction `(S, T_pad, 3)`
             └── optional acceleration-CNN representation evaluation
                 ├── Experiment A: raw-acceleration baseline
                 │   (the trailing acceleration channels)
                 ├── Experiment B: reconstructed acceleration through the
                 │   frozen raw-trained CNN
                 ├── Experiment C: CNN trained and tested on reconstruction
                 └── Experiment D2: mixed raw/reconstruction training with
                     separate raw and reconstruction test evaluations
  → optional, separate Action-0 SynNet training (all published event channels)
       ├── CLI baseline with the full variant label mapping
       └── notebook common-label subset experiment
```

The Action-0 shell wrappers orchestrate preprocessing through padded artifacts.
`python -m snn.train_action0` is a separate optional training entry point that
consumes those padded packages.

The wrappers under `scripts/bash_script/action0_pipeline/` resolve their
relative data, configuration, and output paths from the repository root, so
the Action-0 commands can be launched using their repository path. The
low-pass aligned-Board wrapper currently uses `data` as its data root, action
`0`, `outputs/action0_rectified` as its output root, and
`PIPELINE_MODE=overwrite` (a full rebuild). Change the assignments in
`04_lowpass_aligned_board.sh` to use another data/output root or
`PIPELINE_MODE=continue` for resume behavior. The input layout must contain
primary recordings below `data/user_*/0/*_ring_0.bin`; aligned-Board mode also
requires matching Board chunks for each recording.

The reusable `snn/accel_reconstruction_eval/` modules provide the shared
configuration, padded-dataset validation and loading, user-disjoint splits,
normalization, CNN training, embedding extraction, representation metrics,
paired raw/reconstruction analysis, and checkpoint/result serialization used
by the acceleration experiments. Notebooks and scripts should compose these
modules and focus on experiment configuration, presentation, and
interpretation.

Experiment A defines the cohort for the acceleration-CNN experiments. Its
split configuration accepts `excluded_users` and `included_labels`: packages
are fully loaded and validated first, then users are excluded and labels are
selected before the user-disjoint split and training normalization are built.
`included_labels=None` retains every available label; when labels are named,
each must remain after user exclusion or the run fails explicitly. The
Experiment A notebook exposes these controls as `EXCLUDED_USERS` and
`INCLUDED_LABELS`.

Experiments B, C, and D inherit Experiment A's checkpointed train/validation/
test users and `class_to_idx` mapping. They do not accept separate user or
label cohort overrides; regenerate Experiment A before running them whenever
its cohort selection changes. A downstream run fails if its input cannot
satisfy the checkpointed cohort, rather than silently dropping users or
labels.

All four acceleration-CNN experiments accept either one dataset root or an
ordered list of Action 0 and Action 1 roots. The roots remain separate on disk;
the shared loader validates each root and combines compatible packages only in
memory. For example:

```python
DATASET_ROOTS = [
    Path("outputs/action0_rectified/low-pass/aligned-board-events"),
    Path("outputs/action1_rectified/low-pass/aligned-board-events"),
]
```

The selected roots must have compatible padded producer metadata, including
the same target length and sampling rate. Run A first for the chosen root set.
B, C, and D must use that same root/action/sample cohort and its A checkpoint;
they reject action or canonical sample-ID mismatches and never silently
intersect packages. Legacy A checkpoints remain usable for a single root only;
regenerate A before a two-root run.

Custom Wavelet keeps five bands, with frequencies configurable per pipeline
run. Use `ENCODER_FREQUENCIES_HZ="1 2 4 8 16"` in the bash pipeline or
`--encoder-frequencies-hz 1 2 4 8 16` on the CLI. Event channels are indexed by
axis and band; the published encoder spec/hash supplies their frequencies and
wavelet widths. Segmentation, padding, and multi-root consumers reject mixed
specifications (or old artifacts missing the identity), so regenerate those
artifacts before combining roots or reconstructing acceleration. At 200 Hz,
the 16 Hz band has encoder width 12.

The acceleration-CNN protocol supports three matched feature-extraction
probes: `cnn_s`, `cnn_m`, and `cnn_l`. A/B/C/D notebooks expose this as the
single `PROBE_VARIANT` configuration and pass it to the shared runners; model,
training, and metric implementations are not duplicated in notebooks. The
final comparison reads standardized per-probe summaries and reports balanced
accuracy as the primary metric, with macro-F1 as a secondary metric. Probe
complexity describes the convolutional feature-extraction hierarchy and should
not be interpreted as a pure parameter-count comparison.

## Setup

Use the `writingring-viz` Conda environment with Python 3.11:

```bash
conda activate writingring-viz
python -m pip install -e ".[test]"
```

The optional Action-0 training dependencies are available through the project
extras documented in `pyproject.toml`. For the acceleration-CNN helpers and
notebooks, install the relevant optional dependencies as needed:

```bash
python -m pip install -e ".[snn,notebook]"
```

## Entry points

Inspection and visualization:

```bash
python scripts/list_recordings.py --data-root data_sample/data

python scripts/inspect_recording.py \
  --data-root data_sample/data --user user_0 --action 0 --dataset-id 0

python scripts/plot_recording.py \
  --data-root data_sample/data --user user_0 --action 0 --dataset-id 0 \
  --output-dir outputs/dataset_0 --no-show
```

Derived-processing commands include:

```text
scripts/preprocess_ring_imu.py
scripts/encode_spikes.py
scripts/align_ring_board.py
scripts/segment_ring_imu.py
scripts/reconstruct_segmented_spike_accel.py
scripts/reconstruct_padded_spike_accel.py
scripts/analyze_segment_lengths.py
scripts/pad_segmented_imu.py
scripts/bash_script/action0_pipeline/*.sh
scripts/bash_script/preprocessing_pipeline/rebuild_two_action_frequency_rectify_variant.bash
scripts/plot_board_segment_trajectories.py
scripts/plot_board_trajectory_window.py
scripts/bash_script/Encoder_Evaluation_related/branch_board_trajectory_plot.bash
scripts/bash_script/Encoder_Evaluation_related/single_board_trajectory_plot.bash
scripts/bash_script/Encoder_Evaluation_related/reconstruct_spike_sequence.bash
python -m snn.train_action0
notebooks/action0_snn_training.ipynb
notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb
notebooks/experiment_B_reconstruction_frozen_cnn.ipynb
notebooks/experiment_1_1_reconstruction_temporal_shuffle.ipynb
notebooks/experiment_1_2_event_temporal_shuffle.ipynb
notebooks/experiment_1_3_event_temporal_binning_probe.ipynb
```

Use `--help` on an entry point for its required inputs and output controls.

For example, run the current low-pass aligned-Board pipeline from the
repository root with:

```bash
bash scripts/bash_script/action0_pipeline/04_lowpass_aligned_board.sh
```

To rebuild both Action 0 and Action 1 from completed pipeline roots while
testing a new five-band Custom Wavelet frequency sequence and/or a different
post-encode transform, use:

```bash
ACTION0_SOURCE_COMBINATION_ROOT="$PWD/outputs/action0_rectified/low-pass/aligned-board-events" \
ACTION1_SOURCE_COMBINATION_ROOT="$PWD/outputs/action1_rectified/low-pass/aligned-board-events" \
ENCODER_FREQUENCIES_HZ="1 2 3 4 5" \
POST_ENCODE_TRANSFORM="none" \
bash scripts/bash_script/preprocessing_pipeline/rebuild_two_action_frequency_rectify_variant.bash
```

`ENCODER_FREQUENCIES_HZ` must contain exactly five positive values; the script
sorts them numerically before encoding. `POST_ENCODE_TRANSFORM` accepts only
`none` or `AbsRectify` and may differ from the source variant. By default, the
destination roots are named under
`outputs/reencoded_wavelet_variants/` as
`action0_<transform>_wavelets_<frequencies>/` and
`action1_<transform>_wavelets_<frequencies>/`, followed by the pipeline stage
and boundary mode. Existing destination roots are replaced by default;
set `OVERWRITE_DEST=0` to fail instead.

The rebuild reuses the source `preprocessedIMU` files, requires identical
recording sets and timestamp provenance, and requires SpikeIMU channels
`15:21` to remain exactly equal. For `aligned-board-events`, alignment
provenance is rebound and segmentation is regenerated from the new SpikeIMU;
padding, when present and enabled, is rebuilt using each source dataset's
target length. A `wavelet_variant_validation.json` report is written to each
destination root.

Board trajectory visualization is available at two levels. To plot one PNG
for every exported Board-assisted segment, use the published segmentation
manifest (the script does not re-segment the recording):

```bash
python scripts/plot_board_segment_trajectories.py \
  --data-root data --user user_0 --action 0 --recording 0 --overwrite
```

Omit `--recording` to process every discovered recording for the selected
user/action. For `--action 0`, these images are written under
`outputs/plotting_verification/action0/<user>/recording_<id>/` and the script
reads the low-pass aligned-Board segmentation under
`outputs/action0_rectified/low-pass/aligned-board-events/segmentation`.
For `--action 1`, it uses the corresponding `action1` output directory and
`outputs/action1_rectified` segmentation root. An explicit `--output-root`
is treated as a base directory and also receives the selected action folder.

To inspect a selected Board trajectory window directly:

```bash
python scripts/plot_board_trajectory_window.py \
  --data-root data --user user_0 --recording 2 \
  --start-s 7 --end-s 9 \
  --output outputs/plotting_verification/user0_record2_7_9s.png
```

The window is `[start, end)` seconds from the Ring start by default. Use
`--time-origin board` to measure from the first loaded Board frame instead.

The reusable acceleration-CNN package is imported by notebooks or scripts, for
example:

```python
from snn.accel_reconstruction_eval.config import experiment_c_config

config = experiment_c_config(output_dir="outputs/experiments/C_reconstruction")
config.validate()
```

The helper package defines the following common protocols:

| Experiment | Train / reference | Validation | Test query | Normalization |
| --- | --- | --- | --- | --- |
| A | raw | raw | raw | raw train |
| B | frozen raw-trained model | raw reference | reconstruction | baseline checkpoint |
| C | reconstruction | reconstruction | reconstruction | reconstruction train |
| D2 | mixed raw + reconstruction | mixed | raw and reconstruction separately | mixed train |

For B, the raw-trained checkpoint, its user split, class mapping, and
normalization are authoritative. For C and D2, the checkpoint's user split and
class mapping are authoritative while each protocol retains its own training
and normalization behavior. The same metric and artifact conventions are used
across the protocols; D2 reuses one trained checkpoint for both test domains.

## Phase-A temporal-information experiments

Experiments 1.1–1.3 are matched temporal-information probes over the selected
Action 0 and Action 1 combination roots. They use five deterministic master
seeds `(11, 23, 37, 53, 71)` and the same user-disjoint train/validation/test
split within each seed. The notebooks expose `INCLUDED_LABELS`; the current
configuration retains the ordered labels `A, B, C, D, E, X, G, H, I, J, K,
L`, while `None` retains every available label.

All three experiments use each segment's published `valid_length`. Temporal
perturbations and feature construction operate only on the valid prefix;
right-padding is not shuffled or binned and is reset to zero before model
evaluation. Normalization, when used, is fitted from valid training samples
only. Balanced accuracy is the primary reported metric.

### Experiment 1.1 — Reconstruction temporal-scale ablation

[Experiment 1.1 notebook](notebooks/experiment_1_1_reconstruction_temporal_shuffle.ipynb)
uses the padded reconstructed-acceleration representation with three input
channels. A mask-aware CNN-L is trained from scratch for the original sequence
and for temporal block-shuffle conditions at 5, 50, 100, 200, and 500 ms.
Blocks move jointly across all channels, preserving order inside each block
while disrupting the order between blocks. The best CNN-L embedding is then
frozen and evaluated with a linear probe on held-out users. This experiment
asks how much temporal order remains readable from reconstructed acceleration.

### Experiment 1.2 — Event temporal-scale ablation

[Experiment 1.2 notebook](notebooks/experiment_1_2_event_temporal_shuffle.ipynb)
uses the first 15 channels of the padded SpikeIMU event representation, which
is the event input relevant to the later SNN design. It applies the same
original and 5/50/100/200/500 ms block conditions and retrains an Event-CNN-L
for each condition and seed. Each `[time, 15]` block is shuffled as one unit;
event channels are never permuted independently. A frozen-embedding linear
probe provides the held-out-user test metrics.

### Experiment 1.3 — Event temporal-binning probe

[Experiment 1.3 notebook](notebooks/experiment_1_3_event_temporal_binning_probe.ipynb)
uses the 30-channel polarity-split unsigned event representation but does not
train a CNN or shuffle blocks. For each segment, it sums weighted event values
within 1, 2, 4, 10, or 20 relative-time bins defined over that segment's valid
prefix, then fits a fresh linear probe for each seed and bin count. This is a
low-capacity baseline for testing whether coarse temporal position alone is
enough to explain the classification signal.

## Documentation

- [Source format and terminology](docs/notes/DATA_FORMAT.md)
- [Preprocessing and complete-recording handoff](docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md)
- [Spike encoding](docs/notes/SPIKE_ENCODING.md)
- [Ring–Board alignment](docs/notes/ALIGNMENT_OUTPUTS.md)
- [Label segmentation](docs/notes/IMU_SEGMENTATION.md) and [Board-event segmentation](docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md)
- [Variable-length and padded SpikeIMU acceleration reconstruction](docs/notes/SEGMENTED_SPIKE_ACCEL_RECONSTRUCTION.md)
- [Segment padding](docs/notes/SEGMENT_PADDING.md)
- [Action-0 training and notebook subset experiment](docs/notes/ACTION0_SNN_TRAINING.md) and [its shell wrappers](docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md)
- [Experiment A: raw-acceleration CNN representation evaluation](docs/notes/ACCELERATION_CNN_REPRESENTATION_EVALUATION.md)
- [Experiment B: reconstructed acceleration with the frozen Experiment A CNN](docs/notes/RECONSTRUCTION_FROZEN_CNN_EVALUATION.md)
- [Reusable acceleration-reconstruction evaluation helper modules](docs/notes/acceleration_reconstruction_helper_modules_guide.md)

`vendor/WritingRing/` contains historical/upstream acquisition and plotting
utilities. The repository's current implementation and entry points are the
modules and scripts described above.
