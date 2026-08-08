# Task: Dry-Run Ring Acceleration Frequency-Support Experiment

## Goal

Run a small exploratory experiment on one WritingRing recording to identify:

> Which frequencies are repeatedly high across many one-second IMU windows?

Analyze the three acceleration axes independently:

```text
acc_x
acc_y
acc_z

Generate one combined figure with:

3 rows × 1 column

The figure must use frequency-support percentage rather than mean PSD.

Input recording

Use:

data root:  data_sample/data
user:       user_0
action:     0
dataset ID: 0
Ring input: primary ring_0 only

Reuse the existing project workflow:

discover_recordings
→ select_recording
→ load_ring
→ compute_windowed_psd

Do not create another Ring binary parser.

Do not duplicate the existing FFT implementation.

Existing PSD configuration

Use the existing windowed PSD calculation with:

Nominal sampling rate: 200 Hz
Window duration:       1.0 s
Window size:           200 samples
Overlap:               50%
Hop size:              100 samples
Incomplete tail:       drop
Window function:       Hann
Detrending:            subtract window mean

For Dataset 0, approximately 101 complete windows are expected, but calculate
the actual count rather than hardcoding it.

Frequency-support calculation

Analyze only:

1–100 Hz

Exclude the 0 Hz bin.

For each acceleration axis:

1. Select PSD range

Take the existing per-window PSD values between 1 and 100 Hz.

2. Smooth neighboring frequency bins

Use a three-bin centered moving average independently within every window:

kernel = np.ones(3) / 3.0

Do not modify the original PSD arrays.

3. Normalize each window

Normalize every window by its total PSD in the selected frequency range:

relative_power = smoothed_psd / smoothed_psd.sum(
    axis=1,
    keepdims=True,
)

Each window should sum approximately to one.

4. Select high-power frequencies per window

For every window, calculate the 90th percentile of its relative-power values:

threshold = np.quantile(
    relative_power,
    0.90,
    axis=1,
    keepdims=True,
)

A frequency is high power in that window when:

high_power = relative_power >= threshold
5. Calculate support percentage

For every frequency:

support_count = np.count_nonzero(high_power, axis=0)

support_percent = (
    support_count
    / number_of_windows
    * 100.0
)

Interpretation:

support_percent[f] =
percentage of one-second windows in which frequency f belongs to the
window's highest-power frequency group
6. Calculate supporting strength

For each frequency, calculate the median relative power only among windows in
which that frequency was selected:

supporting_strength[f] = median(
    relative_power[high_power[:, f], f]
)

Use zero when no window supports the frequency.

Figure

Save one image:

outputs/dataset_0/ring_accel_frequency_support.png

Use exactly three vertically stacked subplots:

Acceleration X
Acceleration Y
Acceleration Z

For each subplot:

x-axis: frequency from 1 to 100 Hz;
y-axis: support percentage;
y-axis range: 0–100%;
y-axis must be linear;
show all frequencies as small neutral scatter points;
highlight frequencies with support of at least 20%;
use larger markers for higher supporting strength;
do not connect points with a line;
do not draw mean PSD;
do not draw median PSD;
do not draw individual PSD curves;
do not use logarithmic scaling.

Labels:

X-axis: Frequency (Hz)
Y-axis: Windows with high power (%)

Only the bottom subplot needs the x-axis label.

The title should include:

user_0 / action 0 / dataset 0
200 Hz nominal
1 s windows
50% overlap
90th-percentile high-power threshold

Label up to five top supported frequencies on each subplot.

Rank frequencies by:

support percentage descending;
supporting strength descending.
Console output

Print a compact summary:

Samples:
Windows:
Window size:
Hop size:
Frequency resolution:
Analyzed range:
High-power quantile:
Minimum highlighted support:
Output path:

Then print the top five frequencies for each axis:

Acceleration X:
  5 Hz: support=..., strength=...
  ...

Acceleration Y:
  ...

Acceleration Z:
  ...

Use calculated values.

Minimal implementation requirements

Prefer extending:

src/writingring/spectral.py
scripts/plot_ring_accel_spectrum.py

A small reusable helper or dataclass may be added, but this is an exploratory
experiment rather than a finalized package redesign.

Do not update:

README.md
other documentation
notebooks
Board modules
vendor/WritingRing/
data_sample/

Do not add:

marker segmentation
Board loading
Ring–Board synchronization
timestamp repair
resampling
interpolation
zero-padding
gyroscope analysis
spectrograms
ML feature export
Minimal validation

Add focused tests or direct assertions for:

every relative-power row sums approximately to one;
support percentage is between 0 and 100;
support count matches the high-power mask;
a frequency high in every synthetic window has 100% support;
an extreme frequency in only one window does not receive high support;
the generated figure has exactly three axes;
all three axes are linear and have 0–100% y-limits;
no mean or median PSD curve appears;
the output image is nonempty.

Do not build an extensive new test suite for this dry run.

Run the existing relevant spectral tests after the changes.

Experiment command

Run:

conda run --no-capture-output -n writingring-viz \
  python scripts/plot_ring_accel_spectrum.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --plot-mode support \
  --sampling-rate 200 \
  --window-seconds 1 \
  --overlap 0.5 \
  --frequency-min 1 \
  --frequency-max 100 \
  --high-power-quantile 0.90 \
  --minimum-support 20 \
  --frequency-smoothing-bins 3 \
  --top-k-labels 5 \
  --output outputs/dataset_0/ring_accel_frequency_support.png \
  --no-show \
  --overwrite

Visually inspect the generated image.

Deliverables

The experiment is complete when:

one frequency-support image is generated;
the image contains three vertically stacked acceleration panels;
the axes are linear and range from 0 to 100%;
no mean or median PSD curve is displayed;
frequencies from 1 to 100 Hz are analyzed;
top supported frequencies are printed for all three axes;
the result is checked with focused tests or assertions;
existing relevant spectral tests still pass;
PROGRESS.md records:
files changed;
command executed;
output path;
number of windows;
top frequencies for each axis;
limitations.

This is a dry-run result-generation task. Avoid unrelated refactoring or
documentation work.
