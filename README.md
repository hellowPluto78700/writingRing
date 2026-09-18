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


## Probe temporal-support convention

For probe-based representation analysis, every probe definition should be
evaluated in **two temporal-support modes**. The feature/state being probed,
classifier protocol, train/validation/test split, regularization search, and
all other settings should remain identical; only the temporal support changes.

### 1. Valid-length masked probe

This is the existing/default probe behavior.

For a sample with valid length (T_{valid}), timesteps after the valid region
are masked out before aggregation:

[
m_t = mathbf{1}[t < T_{valid}].
]

Examples:

[
mathrm{WholeCount}_{valid}
=
sum_{t<T_{valid}} z_t,
]

and for an absolute 250 ms bin (b),

[
mathrm{Fixed250Count}_{valid,b}
=
sum_{tin b, t<T_{valid}} z_t.
]

For mean-valued hidden-state probes, the denominator is the number of valid
timesteps contributing to that whole-sequence or Fixed250 region.

### 2. Whole-window unmasked probe

The same probe must also be evaluated over the complete padded SNN window
(T_{window}), **without clearing or masking hidden states/spikes after
`valid_length`**.

Examples:

[
mathrm{WholeCount}_{window}
=
sum_{t<T_{window}} z_t,
]

and

[
mathrm{Fixed250Count}_{window,b}
=
sum_{tin b} z_t.
]

For mean-valued probes, use the full whole-window or full-bin timestep count as
the denominator. Do not replace post-`valid_length` SNN state, membrane,
synaptic current, or spike activity with zeros before aggregation.

This mode intentionally preserves any network dynamics that continue after the
last valid input sample, including residual synaptic current, membrane decay,
reset dynamics, and resulting spikes. The padded input itself may already be
zero after `valid_length`; the requirement is that probe extraction must not
apply an additional valid-length mask to the SNN trajectory.

The two modes answer different questions:

```text
valid-length masked
    -> information available only during the observed sample

whole-window unmasked
    -> information available from the complete fixed inference window,
       including post-valid residual SNN dynamics
```

For ordered Fixed250 probes, both modes use the same absolute bins anchored at
the start of the padded window. This is not relative-time binning.

Existing finalized probe artifacts created before this convention may contain
only the valid-length-masked version. Do not relabel those historical results;
new or re-run probe evaluations should publish both variants explicitly.

## Writing-only motion preprocessing: removing airborne/repositioning acceleration

For experiments that separate **stroke-related writing motion** from
**airborne/repositioning motion**, the repository provides a derived
preprocessing stage:

\`\`\`text
scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash
scripts/build_writing_motion_variants.py
\`\`\`

This stage starts from a **completed 64 Hz aligned-Board-event SpikeIMU
dataset**. It does not re-run Ring–Board alignment or segmentation. The
published source alignment, label ownership, segment sample geometry, and
Board event audit tables are treated as authoritative.

### Writing/contact mask definition

A timestep is treated as writing/contact only while the Board reports a
complete, valid, non-transient press/lift pair.

For a recording with valid touch pairs

\[
(p_1,l_1), (p_2,l_2), \ldots,
\]

the recording-level writing mask is

\[
m_t = 1
\iff
t \in [p_1,l_1] \cup [p_2,l_2] \cup \cdots .
\]

Press and lift samples are both included. For a multi-stroke letter, the
airborne gap between strokes therefore remains airborne. The implementation
does **not** replace multiple strokes with one broad
\`first_press -> last_lift\` interval.

For example:

\`\`\`text
segment:
|----------------------------------------------------|

Board:
       press1       lift1        press2       lift2
          |-----------|             |-----------|

writing mask:
000000111111111111000000000000001111111111110000000

airborne/reposition:
^^^^^^              ^^^^^^^^^^^^^^              ^^^^^
\`\`\`

The time axis and original segment duration are preserved. Writing strokes are
not concatenated after airborne samples are removed.

The source Board-event segmentation can contain boundary corner cases created
by carry-in handling, crossing-touch ownership, neighbor midpoint resolution,
or recording truncation. The derived mask follows an
**observable-intersection policy**:

- a valid touch partially outside the observable Ring recording is clipped to
  the available Ring sample range;
- a valid touch wholly outside the Ring recording contributes no Ring samples;
- an already assigned touch is intersected with the final published segment
  slice;
- the derived preprocessing never expands, shifts, or re-segments a source
  segment to recover clipped samples.

These cases are recorded in the generated interval CSVs and user reports
rather than being silently hidden.

### D0, D1, and D2 datasets

The preprocessing produces two derived variants while retaining the original
dataset as D0.

#### D0 — original representation

\[
D_0 = Encoder(a)
\]

This is the original aligned-Board-event dataset. It is referenced rather than
duplicated.

#### D1 — remove airborne motion after encoding

\[
D_1(t) = m_t \, Encoder(a)_t
\]

The original encoded representation is kept unchanged during writing/contact
samples and hard-zeroed outside the writing mask.

For the current 36-channel SpikeIMU schema, the builder zeros **all
input-visible motion channels** outside the writing mask:

\`\`\`text
30 polarity-split wavelet event channels
+ 3 physical acceleration channels
+ 3 physical gyroscope channels
= 36 channels
\`\`\`

This avoids leaking airborne/repositioning information through the trailing
physical IMU channels.

D1 answers approximately:

\[
\text{How much classification information is directly present during
airborne/repositioning periods?}
\]

A D0-vs-D1 performance difference measures the effect of removing those
airborne-period samples while keeping writing-period encoded values unchanged.

#### D2 — remove airborne acceleration before encoding, then re-encode

\[
a'_t = m_t a_t,
\]

followed by

\[
D_2(t) = m_t \, Encoder(a')_t.
\]

D2 is intentionally stricter than D1. Airborne acceleration is removed from
the **64 Hz acceleration that is directly supplied to the Custom Wavelet
encoder**, then the complete recording is encoded again using the exact source
encoder specification.

The implementation does **not** mask the original 200 Hz signal and then
resample it, because resampling/filtering could mix airborne motion back across
the press/lift boundaries.

It also does **not** independently encode each final segment. Custom Wavelet
encoding is run once on the complete masked recording so that encoder/filter
state evolves continuously across the same recording timeline as D0.

After re-encoding, D2 is masked once more:

\[
D_2(t) = 0 \quad \text{when } m_t=0,
\]

because a temporal filter can produce residual event activity immediately
outside a contact interval even when its input is already zero there.

D2 therefore measures the effect of airborne acceleration on the
**writing-period encoded representation through encoder temporal context**.

A useful interpretation is:

\[
D_0-D_1
\]

approximately isolates evidence carried directly by airborne/reposition
timesteps, while

\[
D_1-D_2
\]

captures how airborne acceleration changes the representation produced during
writing through the temporal encoder.

### Source contract

The current builder expects the source dataset to be:

\`\`\`text
64 Hz
36-channel SpikeIMU
30 event channels + 6 physical IMU channels
polarity_split_wavelet_events_plus_imu_v1
completed aligned-Board-event segmentation
\`\`\`

The D2 encoder is reconstructed from the source metadata. The source
\`spike_encoder_spec_sha256\` must match the reconstructed encoder identity;
the builder fails instead of silently using a different wavelet configuration.

### Generated dataset layout

If the source root is:

\`\`\`text
outputs/action0_wavelets_0e5_1_2_4_8_sr_64/
  low-pass/
    aligned-board-events/
\`\`\`

the default derived root is:

\`\`\`text
outputs/action0_wavelets_0e5_1_2_4_8_sr_64/
  low-pass/
    aligned-board-events_writing_motion_ablation/
\`\`\`

Its main structure is:

\`\`\`text
aligned-board-events_writing_motion_ablation/
├── writing_motion_ablation_manifest.json
├── user_reports/
├── logs/
│
├── original_reference/                 # D0 metadata/reference only
│   ├── dataset_reference.json
│   └── annotations/
│       └── user_*/
│           └── action_*/
│               ├── *_writing_mask.npy
│               ├── *_writing_intervals.csv
│               └── *_annotation_summary.json
│
├── postencode_mask/                    # D1 = m * Encoder(a)
│   ├── recordings/
│   ├── segmentation/
│   └── segmentation_padded/
│
├── masked_accel_reencode/              # D2 = m * Encoder(m * a)
│   ├── recordings/
│   ├── segmentation/
│   └── segmentation_padded/
│
└── writing_motion_verification/
    └── user_*/
        └── action_*/
            ├── <dataset_id>_writing_motion_verification.png
            └── <dataset_id>_writing_motion_verification.json
\`\`\`

Each D1/D2 segmented package retains the original source labels, segment
ordering, segment lengths, and padding geometry. Additional mask artifacts are
published alongside the derived data:

\`\`\`text
*_writing_mask.npy
*_writing_intervals.csv
*_padded_writing_mask.npy
\`\`\`

The repositioning mask can be obtained as

\[
\text{reposition\_mask}
=
\text{valid\_mask} \land \neg \text{writing\_mask}.
\]

### Verification plots

The writing-motion verification is branch-independent because D1 and D2 use
the same published writing mask and segment geometry.

The plot follows the original Board-event segmentation verification visual
contract:

\`\`\`text
blue line            Ring transient score
red dashed line      valid Board press
orange dashed line   valid Board lift
thin red/orange      transient press/lift
gray dashed line     timestamp label
red label line       skipped label
light green span     exported final segment
dark green span      retained writing/contact interval
green solid line     final segment start
green dotted line    final segment end
\`\`\`

Panels are 10 s wide, like the original segmentation verification figure.
Within a light-green exported segment, regions not covered by a darker writing
span correspond to airborne/repositioning samples removed by D1/D2.

The transient score is recomputed from the exact source-declared transient IMU
channels using the same robust first-difference transient-score implementation
used by the original segmentation pipeline. The verification code only
visualizes already-published geometry; it does not redefine segmentation or
the writing mask.

### Running the preprocessing

For Action 0, for example:

\`\`\`bash
MODE=submit \
SLURM_MAX_CONCURRENCY=20 \
WRITE_MASK_VERIFICATION=1 \
bash scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash \
  outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events
\`\`\`

For Action 1, pass the corresponding completed Action-1 source root. The
builder infers the action from the source segmentation package; there is no
separate \`ACTION=1\` flag:

\`\`\`bash
MODE=submit \
SLURM_MAX_CONCURRENCY=20 \
WRITE_MASK_VERIFICATION=1 \
bash scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash \
  outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events
\`\`\`

The launcher uses one CPU/Slurm array task per user and an \`afterok\`
finalizer. To replace an existing or partial derived root, add:

\`\`\`bash
OVERWRITE_DEST=1
\`\`\`

The final dataset should not be treated as complete until

\`\`\`text
writing_motion_ablation_manifest.json
\`\`\`

exists with

\`\`\`json
{
  "status": "PASS"
}
\`\`\`

and the expected user count.

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

**SNN execution note:** when independent random seeds can be scheduled separately, prefer multi-CPU execution with one CPU core per seed when possible; on Unity, Experiment 1.3.9 (`con500`, `lambda=0.1`, 40 epochs, seeds `11/23/101`) completed three seeds in parallel on `3 × 1` CPU cores in ~468 s (7.8 min), while one RTX 2080 Ti running the same three seeds sequentially took ~1045 s (17.4 min), making the multi-CPU strategy ~2.2× faster in time-to-results.

## Default multi-CPU experiment workflow

Unless an experiment has a concrete reason to use another execution model,
independent runs should be parallelized at the task level on Unity. Independent
run dimensions include random seed, objective, architecture, configuration,
and ablation condition.

The default execution pattern is:

```text
independent experiment conditions
        -> one Slurm array task per run
        -> one CPU core per task
        -> train
        -> select/save best checkpoint
        -> immediately evaluate that checkpoint when evaluation is run-local
        -> save per-run artifacts
all required jobs complete
        -> finalizer/aggregator
        -> analysis-only notebook for tables and plots
```

Repository defaults:

- Prefer Slurm arrays instead of sequentially looping over independent runs in
  one process.
- Use one CPU core per independent run unless profiling demonstrates a real
  benefit from multiple cores per run.
- Cap simultaneous experiment tasks at **50 CPUs** by default, for example
  `#SBATCH --array=0-N%50`. Do not exceed 50 concurrent experiment CPU tasks
  unless explicitly requested.
- Split sweeps over seed/objective/architecture/configuration into separate
  array tasks whenever those runs are independent.
- If evaluation depends only on the checkpoint produced by the current run,
  keep `train -> evaluate -> save artifacts` in the same Slurm task. This
  avoids waiting for the slowest training run before starting a second full
  evaluation array.
- Keep training and evaluation implemented as separate Python functions or
  entry points even when one Slurm task executes them consecutively, so probes
  and metrics can be rerun without retraining.
- Small reused/frozen baseline sets may be evaluated sequentially on one CPU
  while the main array runs.
- Use Slurm `afterok` dependencies when a finalizer depends on multiple job
  groups.
- Finalizers aggregate existing run artifacts only; missing runs should fail
  explicitly rather than being silently regenerated.
- Notebooks should be analysis-only whenever practical: read finalized
  CSV/JSON artifacts, aggregate, rank, and plot. Do not make notebooks the
  primary training or multiprocessing driver.
- Every Slurm compute node/job must initialize Conda locally before running
  experiment code. Batch scripts should run `module load conda/latest`, then
  `eval "$(conda shell.bash hook)"`, then activate `writingring-gpu`; fall back
  to `writingring-viz` only if `writingring-gpu` is unavailable.
- Do not rely on Conda activation inherited from the login/submit shell, and do
  not pass the submit shell's absolute Python executable path to compute nodes
  as the environment contract.
- Do not place comma-separated values such as label lists directly inside
  `sbatch --export=...`, because Slurm uses commas as variable separators.
  Export the complete value in the submit shell first, then submit with
  `--export=ALL` so the value is inherited intact.
- For one-core tasks, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` to avoid hidden CPU
  oversubscription.
- Size Slurm memory and walltime for the complete atomic run, including any
  evaluation performed after training.

Valid reasons to deviate include a demonstrably GPU-bound workload, materially
higher per-run memory requirements, unavoidable shared mutable state, a true
serial dependency between runs, unsafe concurrent artifact writes, or measured
benchmark evidence that another execution model is better. Document the reason
for a deviation in the experiment README or runner comments.

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