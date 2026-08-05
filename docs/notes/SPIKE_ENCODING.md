# Spike encoding

`scripts/encode_spikes.py` applies a registered encoder to an existing
nine-channel segmentation export. It is an independent, read-only
post-processing step: it never re-runs gravity removal, resamples data, changes
segment boundaries, or modifies the input IMU, labels, segmentation summary,
or padding artifacts.

## Custom Wavelet command

```bash
python scripts/encode_spikes.py \
  --input-imu <input-root>/<stem>_rawIMU.npy \
  --input-summary <input-root>/<stem>_segmentation_summary.json \
  --encoder custom-wavelet \
  --encoder-settings configs/spike_encoding/custom_wavelet.json \
  --sequence-mode offsets \
  --sequence-offsets <input-root>/<stem>_segment_offsets.npy
```

The required IMU input is a finite numeric `(N, 9)` NPY file loaded with
`allow_pickle=False`. Only its first three channels are read, in this order:
`acceleration_x_g`, `acceleration_y_g`, `acceleration_z_g`. They remain in g;
the encoding layer does not estimate orientation or remove gravity.

The optional input summary is read only. When present, its channel schema and
sampling rate are checked. Its gravity-removal method and acceleration
semantics are checked as a pair: both must be present or both omitted, the
method must be `raw`, `low-pass`, `madgwick`, or
`xylo-rotate-and-remove-gravity`, and the semantics must match that method.
Only validated provenance is copied into the published summary.

## Sequence state boundaries

Use `--sequence-mode offsets` with `--sequence-offsets` to reset the stateful
IIR and extrema-window history at every nonempty `[start, stop)` interval.
Offsets must be integer, strictly increasing, start at zero, and end at `N`.

With `--sequence-mode single-array`, or with neither sequence argument, the
whole input is processed as exactly one continuous sequence and `[0, N]` is
still published for auditability. The tool never guesses continuity between
rows.

## Sampling rate and settings

Custom Wavelet settings are JSON. The included
`configs/spike_encoding/custom_wavelet.json` uses acceleration wavelets,
frequencies `[0.5, 1, 2, 4, 8]` Hz, 200 Hz sampling, second-order Prony
numerator/denominator fits, and float32 output.

Sampling rate is used only to derive wavelet widths and extrema-window size;
it never resamples samples. An explicit `sampling_rate_hz` in settings takes
priority. If a compatible input summary also provides a rate, the two values
must match exactly. If settings omit the rate, the summary supplies it; if both
omit it, encoding fails.

`frequencies_hz` must be finite, positive, strictly increasing, below Nyquist,
and must not collapse to duplicate `int(sampling_rate_hz / frequency_hz)`
widths. The Custom Wavelet output is a signed local-extrema amplitude stream,
with axis-major, frequency-minor channel order; it is not a binary spike train.
That channel-order label is encoder-specific metadata. The generic framework
does not assume wavelets, frequencies, or any flattening order, so encoders
that do not declare a channel order omit it from their summaries.

## Published artifacts

Outputs always remain under the input IMU's parent:

```text
<input-root>/
└── custom-wavelet/
    └── <output-stem>/
        ├── <output-stem>_spikeEvents.npy
        ├── <output-stem>_spike_sequence_offsets.npy
        ├── <output-stem>_spike_sequences.csv
        └── <output-stem>_spike_encoding_summary.json
```

`<output-stem>` defaults to the input name without `_rawIMU.npy` and can be
set with `--output-stem`. `--output-root`, if supplied, must be the input
IMU's parent directory; this prevents an encoding run from silently publishing
outside the source root.

The event NPY has shape `(N, 3 * len(frequencies_hz))`, uses the requested
float32 or float64 dtype, contains finite values only, and preserves the input
sample count. The offsets NPY records the actual reset boundaries. The CSV has
one row per sequence with event counts and density. The JSON summary records
source provenance, effective settings and widths, channel names, reset policy,
dtype, representation, polarity, and aggregate statistics.

All sequences must encode and validate before publication. Files are staged,
reloaded for validation, and atomically published. Existing results are never
replaced unless `--overwrite` is supplied; even then, the command refuses to
remove unrelated files from the output directory.
