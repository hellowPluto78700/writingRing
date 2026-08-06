# Action-0 segmentation Bash pipelines

`docs/plans/Segmentations_Bash_Scripts_Plan.md` is implemented by the eight
entry points under `scripts/action0_pipeline/`; `_common.bash` is their shared
orchestration helper:

| Script | Gravity method | Boundary mode |
| --- | --- | --- |
| `01_raw_label.sh` | `raw` | `label` |
| `02_raw_aligned_board.sh` | `raw` | `aligned-board-events` |
| `03_lowpass_label.sh` | `low-pass` | `label` |
| `04_lowpass_aligned_board.sh` | `low-pass` | `aligned-board-events` |
| `05_madgwick_label.sh` | `madgwick` | `label` |
| `06_madgwick_aligned_board.sh` | `madgwick` | `aligned-board-events` |
| `07_xylo_label.sh` | `xylo-rotate-and-remove-gravity` | `label` |
| `08_xylo_aligned_board.sh` | `xylo-rotate-and-remove-gravity` | `aligned-board-events` |

Every entry point is independently runnable. The shared `_common.bash` contains
only Bash orchestration and contract checks; it does not parse Ring or Board
payloads. Each run discovers only
`DATA_ROOT/user_*/ACTION/*_ring_0.bin`, validates integer dataset IDs, and
passes one exact `user/action/dataset-id` selector to the existing Python
preprocessing CLI. Board mode checks that every recording has a matching Board
chunk but leaves numeric chunk ordering, deserialization, timestamp checks,
and identity validation to the repository loader.

The pipeline stages are:

```text
ring_0 discovery
  → complete nine-channel preprocessing per recording
  → one independent Custom Wavelet encoding per complete recording
  → SpikeIMU label segmentation, or per-recording SpikeIMU alignment followed
    by one aligned-Board segmentation per user
```

The default configuration is `DATA_ROOT=data`, `ACTION=0`, `SAMPLING_RATE=200`,
`ENCODER=custom-wavelet`,
`ENCODER_SETTINGS=configs/spike_encoding/custom_wavelet.json`,
`CONDA_ENV=writingring-viz`, and `OVERWRITE=0`. Relative paths are resolved
from the project root. `LOW_PASS_CUTOFF_HZ`, `MADGWICK_BETA`,
`MADGWICK_PROVISIONAL=1`, `OUTPUT_BASE`, and the other defaults can be
overridden with environment variables. Xylo entry points preflight the
Rockpool Xylo imports and fail with the repository installation hint when the
optional dependency is unavailable.

Each gravity/boundary combination is isolated under (the Xylo selector uses
the shorter plan directory name `xylo`):

```text
outputs/action0_pipeline/<gravity>/<boundary>/
├── preprocessedIMU/
├── spikeEncoding/custom-wavelet/
├── alignment/                 # Board mode only
│   ├── offsets/
│   ├── reports/
│   └── verification/
├── segmentation/
└── logs/
```

The scripts use strict fail-fast execution. A failed preprocessing, encoding,
alignment, or user segmentation command stops the run. `OVERWRITE=1` adds the
corresponding overwrite flags to the Python CLIs. Label mode adds only
`--overwrite` to segmentation because it does not create verification images;
Board mode additionally passes `--overwrite-verification`.

Before reporting success, the shared QA checks compare discovered `ring_0`,
preprocessing-summary, SpikeIMU, Board-offset (aligned mode), and per-user
segmentation-summary counts. It also verifies canonical timestamp sidecars,
metadata, matrices, manifests, Board targets/audits, and the `(N, 21)`
SpikeIMU contract. Logs contain discovery, per-recording preprocessing and
alignment, one encoding log, per-user segmentation, and QA output.

The scripts intentionally differ from the upstream plotting scripts in their
workflow policy: upstream `board_plot.py` globs every action-level gzip file
without dataset-prefix isolation or numeric ordering and selects a random
marker interval. The new wrappers select the repository recording identity,
invoke the tested loader/CLI contracts, and never manually iterate Board
chunks. They retain the upstream primary `ring_0` choice and the verified
complete-recording data semantics while keeping `data_sample/` and
`vendor/WritingRing/` read-only.
