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
  --encoder-settings configs/spike_encoding/custom_wavelet.json
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

## Recording state boundaries and occurrence alignment

Custom Wavelet encodes the complete `(N, 9)` input as exactly one recording
and resets once at its start. It rejects `--sequence-mode`,
`--sequence-offsets`, `--recording-offsets`, label sidecars, and segment
sidecars. Segmentation belongs to the later timestamp-to-index stage, so no
label boundary can restart the IIR or create a synthetic edge.

For each recording, the encoder derives `H` from its actual odd extrema window:
`H = max_filter_time_samples // 2`. It reflect-pads `H` samples at both ends,
causally encodes the padded data, then selects detection rows
`[2H : 2H + N]`. This compensates only the fixed extrema-confirmation latency,
so row `i` again refers to the wavelet-extremum occurrence at input row `i`.
At 200 Hz the default 61-sample window gives `H = 30`, or 0.15 s per side.
No IIR phase/group-delay or warm-up compensation is performed; synthetic
recording-edge padding is an accepted assumption.

The generic boundary options remain available only to other registered
encoders. Custom Wavelet may optionally receive `--timestamps-path`; it uses
that NPY only to validate row count, strict monotonicity, and the sampling-rate
cadence, never for filtering or event detection.

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
        ├── <output-stem>_spikeIMU.npy
        ├── <output-stem>_spike_recording_offsets.npy
        ├── <output-stem>_spike_sequences.csv
        └── <output-stem>_spike_encoding_summary.json
```

`<output-stem>` defaults to the input name without `_rawIMU.npy` and can be
set with `--output-stem`. `--output-root`, if supplied, must be the input
IMU's parent directory; this prevents an encoding run from silently publishing
outside the source root.

`spikeEvents.npy` has shape `(N, 3 * len(frequencies_hz))`, uses the requested
float32 or float64 dtype, contains finite values only, and preserves the input
sample count. Its default shape is `(N, 15)` in axis-major, frequency-minor
order; values are signed wavelet-response extrema, not binary spikes. Plateau
ties emit the centre amplitude once, rather than adding it twice.

`spikeIMU.npy` is published only for Custom Wavelet's 15-channel output. It is
`column_stack((spikeEvents, rawIMU[:, 3:9]))` and has shape `(N, 21)`: 15 event
channels followed by the preprocessed `acceleration_x`, `acceleration_y`,
`acceleration_z` channels (m/s²) and the three gyroscope channels (rad/s).
These six channels exactly preserve the corresponding source `rawIMU` rows;
they are not necessarily raw sensor values. When the source preprocessing
method is `low-pass`, `madgwick`, or `xylo-rotate-and-remove-gravity`, the
acceleration triplet is already processed/gravity-removed. Only `raw`
preprocessing retains measured acceleration including gravity. The output
deliberately does not include the three `acceleration_*_g` input columns, the
original Ring acceleration columns, or the Ring timestamp. No source array is
modified.

The recording-offset NPY is `[0, N]` for Custom Wavelet's single recording.
Labels and segment offsets/lengths are not read by the encoder, but colocated
sidecars are validated and recorded as read-only provenance for downstream
segmentation. The summary explicitly states that they did not set reset
boundaries and that their indices were not shifted. The CSV has one row with
event counts and density. The JSON summary records source
provenance, effective settings and widths, padding, delay policy, channel
schemas, reset policy, dtype, representation, polarity, and aggregate
statistics.

All sequences must encode and validate before publication. Files are staged,
reloaded for validation, and atomically published. Existing results are never
replaced unless `--overwrite` is supplied; even then, the command refuses to
remove unrelated files from the output directory.

Other registered encoders retain the legacy `spike_sequence_offsets.npy`
artifact and `offset_semantics: "sequence"`. Publication recognises both
offset-artifact spellings during an overwrite, so an intermediate result can
be migrated without treating its owned offset file as unrelated content.
