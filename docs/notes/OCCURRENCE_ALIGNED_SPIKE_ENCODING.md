# Occurrence-aligned spike encoding

Custom Wavelet encoding treats one input file as one complete recording, not a
label segment. It resets once for the whole array, reflect-pads the recording
by half of its actual odd extrema window, and returns events to their
wavelet-extremum occurrence rows. The
default 200 Hz configuration has a 61-sample window, so each recording receives
30 samples (0.15 s) of synthetic context on each side.

For a recording of `N` samples and half-window `H`, the causal detector
produces padded detection-aligned rows and the exported events are:

```python
occurrence_events = detected_events[2 * H : 2 * H + N]
```

This compensates the extrema detector's fixed confirmation delay only. It does
not compensate IIR phase delay, frequency-dependent group delay, or IIR warmup.
The summary explicitly records those limits and the accepted reflect-boundary
assumption.

The encoder retains all fifteen default axis-major/frequency-minor event
channels. Events are signed wavelet-response local maxima and minima. A local
plateau that is both a maximum and a minimum emits the centre amplitude once.

Publication remains atomic and source arrays remain read-only. Each result now
contains both `(N, 15)` `spikeEvents.npy` and `(N, 21)` `spikeIMU.npy`. Custom
Wavelet rejects label and segment sidecar command-line arguments; an optional
timestamps NPY is validated for row count, monotonicity, and cadence only. The
latter concatenates those event channels with the unchanged source
`rawIMU[:, 3:9]` m/s² acceleration and gyroscope channels; it omits the three
g-domain acceleration columns. Existing labels, timestamps, segment offsets,
segment lengths, and segment manifests are referenced as source sidecars and
are neither shifted nor regenerated. The current segmenter still accepts only
six- or nine-channel input, so it is intentionally not re-run on `spikeIMU`.
