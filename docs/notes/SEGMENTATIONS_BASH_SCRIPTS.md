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
  → global segment-length scan
  → right-padding to the configured coverage target
```

The default configuration is `DATA_ROOT=data`, `ACTION=0`, nominal
`SAMPLING_RATE=200`, `ENCODER=custom-wavelet`,
`ENCODER_SETTINGS=configs/spike_encoding/custom_wavelet.json`,
`POST_ENCODE_TRANSFORM=none`, `CONDA_ENV=writingring-viz`, and
`PIPELINE_MODE=continue`. Relative paths are
resolved from the project root. `LOW_PASS_CUTOFF_HZ`, `MADGWICK_BETA`,
`MADGWICK_PROVISIONAL=0`, `OUTPUT_BASE`, and the other defaults can be
overridden with environment variables. Padding is enabled for every entry
point after segmentation. Its defaults are `PADDING_COVERAGE=0.99`,
`PADDING_RECOMMENDATION=balanced`, `PADDING_ROUND_TO=1`, and
`PADDING_VALUE=0.0`. `PADDING_ANALYSIS_DIR` and `PADDING_OUTPUT_ROOT` can
override the analysis and padded-output locations. Set
`PADDING_RECOMMENDATION=p99` to use the exact empirical P99 target instead of
the smallest configured candidate that meets the requested coverage; use
`pure-padding` to retain every segment. `OVERWRITE=1` also replaces the
analysis and padded outputs. Xylo entry points preflight the Rockpool Xylo
imports and fail with the repository installation hint when the optional
dependency is unavailable.

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
│   └── padding_analysis/
├── segmentation_padded/
└── logs/
```

`segmentation/padding_analysis/` contains the global JSON/CSV/Matplotlib
length report. `segmentation_padded/` contains the fixed-length arrays,
valid-length masks, manifests, and the root padding summary. Padding is on the
right; segments longer than the selected target are retained in the manifest
with `exported=false` and skipped rather than truncated. Therefore the summary
always exposes the exact exported/skipped counts for the chosen coverage.

The scripts use strict fail-fast execution. `PIPELINE_MODE=continue` validates
each stage, retains complete valid stages, and resumes at a missing later
stage. Any partial/invalid stage, or downstream output without its prerequisite,
causes a full rebuild from preprocessing with overwrite flags. Set
`PIPELINE_MODE=overwrite` to rebuild from preprocessing without resume checks.
For encoded SpikeIMU artifacts, validity also requires the published
`settings.post_encode_transform` to equal the requested
`POST_ENCODE_TRANSFORM` (`none` is metadata `null`). Legacy metadata with no
transform key is treated as `none`; a transform mismatch makes encoding
invalid and uses this same existing preprocessing/full-rebuild path rather
than a transform-specific resume mode.
`OVERWRITE=1` is a legacy alias only when `PIPELINE_MODE` is unset; an explicit
`PIPELINE_MODE` takes precedence. A failed preprocessing, encoding, alignment,
or user segmentation command stops the run. Label mode adds only
`--overwrite` to segmentation because it does not create verification images;
Board mode additionally passes `--overwrite-verification`.

There is one aligned-Board downstream-staleness exception to the otherwise
structural full-rebuild rule. For a structurally readable segmentation summary,
`continue` reconstructs the current
`alignment_outcome_dependency` from public discovery, current feature/Board
provenance, validated outcomes, validated skip reasons, and report SHA-256
digests. A missing, legacy, malformed, or unequal dependency means that the
summary is stale even when its arrays are structurally valid. Alignment stays
reusable; the pipeline resumes at segmentation, explicitly replaces only the
segmentation and padding outputs, and regenerates padding. A missing,
unreadable, or non-object summary remains structural invalidity and retains the
full preprocessing rebuild policy. This is stage-level behavior, never
per-recording resume, and it does not alter standalone padding's schema.

For aligned-Board mode, alignment explicitly permits the defined initial
interval skip policy. The alignment stage validates one provenance-current
Python outcome for every discovered recording: `SUCCESS` proceeds to Board
segmentation and `SKIPPED` is a completed recording outcome that segmentation
omits. Missing, stale, malformed, conflicting, or `FAILED` outcomes invalidate
the whole stage and retain the existing full-rebuild `continue` behavior. Bash
does not parse alignment TXT or JSON artifacts; it only consumes the validated
terminal status returned by the Python outcome contract.

Before reporting success, the shared QA validates every aligned recording
through the Python outcome contract and reports total, success, skipped, and
invalid recording counts, and repeats the exact per-user
`alignment_outcome_dependency` reconciliation used by continue planning. It
then reconciles discovered `ring_0`,
preprocessing-summary, SpikeIMU, per-user segmentation-summary, and
padded-summary counts; completed recording skips are not expected to produce
segments. It also verifies canonical timestamp sidecars, metadata, matrices,
manifests, Board targets/audits for processed recordings, the `(N, 21)`
SpikeIMU contract, and that padded source/exported/segment-skipped counts
reconcile. Logs contain discovery, per-recording preprocessing and alignment,
one encoding log, per-user segmentation, padding analysis/publish output, and
QA output.

The scripts intentionally differ from the upstream plotting scripts in their
workflow policy: upstream `board_plot.py` globs every action-level gzip file
without dataset-prefix isolation or numeric ordering and selects a random
marker interval. The new wrappers select the repository recording identity,
invoke the tested loader/CLI contracts, and never manually iterate Board
chunks. They retain the upstream primary `ring_0` choice and the verified
complete-recording data semantics while keeping `data_sample/` and
`vendor/WritingRing/` read-only.
