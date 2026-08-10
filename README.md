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

## Current workflow

```text
Ring recording (`*_ring_0.bin`)
  → discovery / inspection / Matplotlib visualization
  → 9-channel preprocessing and gravity handling
  → optional Custom Wavelet encoding
       ├── 15 event channels
       └── 21-channel SpikeIMU artifact
  → optional Ring–Board alignment
  → label or aligned-Board-event segmentation
  → variable-length segment analysis and right-padding
  → optional, separate Action-0 SynNet training (SpikeIMU channels 0:15)
       ├── CLI baseline with the full variant label mapping
       └── notebook common-label subset experiment
```

The Action-0 shell wrappers orchestrate preprocessing through padded artifacts.
`python -m snn.train_action0` is a separate optional training entry point that
consumes those padded packages.

## Setup

Use the `writingring-viz` Conda environment with Python 3.11:

```bash
conda activate writingring-viz
python -m pip install -e ".[test]"
```

The optional Action-0 training dependencies are available through the project
extras documented in `pyproject.toml`.

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
scripts/analyze_segment_lengths.py
scripts/pad_segmented_imu.py
scripts/action0_pipeline/*.sh
python -m snn.train_action0
notebooks/action0_snn_training.ipynb
```

Use `--help` on an entry point for its required inputs and output controls.

## Documentation

- [Source format and terminology](docs/notes/DATA_FORMAT.md)
- [Preprocessing and complete-recording handoff](docs/notes/GRAVITY_TO_SPIKE_PIPELINE.md)
- [Spike encoding](docs/notes/SPIKE_ENCODING.md)
- [Ring–Board alignment](docs/notes/ALIGNMENT_OUTPUTS.md)
- [Label segmentation](docs/notes/IMU_SEGMENTATION.md) and [Board-event segmentation](docs/notes/BOARD_EVENT_GUIDED_SEGMENTATION.md)
- [Segment padding](docs/notes/SEGMENT_PADDING.md)
- [Action-0 training and notebook subset experiment](docs/notes/ACTION0_SNN_TRAINING.md) and [its shell wrappers](docs/notes/SEGMENTATIONS_BASH_SCRIPTS.md)

`vendor/WritingRing/` contains historical/upstream acquisition and plotting
utilities. The repository's current implementation and entry points are the
modules and scripts described above.
