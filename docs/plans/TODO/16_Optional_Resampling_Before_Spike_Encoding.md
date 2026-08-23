# Plan: Optional Resampling Before Spike Encoding

## Workflow

**Class:** `HIGH_RISK / STANDARD`

This change affects pipeline contracts, timestamp semantics, persisted artifacts, encoder sampling-rate semantics, and alignment/segmentation provenance.

```text
PRIMARY
  → freeze TaskSpec
  → luna_worker
  → luna_verifier
```

No probe is required because the relevant repository contracts are already established. Replan only if implementation exposes a concrete contract ambiguity.

---

## Goal

Add an optional complete-recording resampling stage:

```text
Ring
→ gravity preprocessing
→ optional resampling
→ spike encoding
→ alignment
→ segmentation
→ padding
```

Pipeline configuration:

```bash
RESAMPLE_RATE_HZ=none   # default; preserve current behavior
RESAMPLE_RATE_HZ=64
```

Core rules:

- Gravity preprocessing continues to run at the existing source/nominal sampling rate.
- Resample the continuous 9-channel IMU, never already-generated spikes.
- Changing the resampling configuration invalidates encoding and all downstream artifacts.
- Existing gravity-preprocessed artifacts remain reusable.

---

## T001 — Resampling Stage and Artifact Contract

Add an independent complete-recording resampling stage between preprocessing and encoding.

Input:

```text
preprocessedIMU:        (N, 9)
canonical timestamps:   (N,)
preprocessing metadata
```

Output:

```text
resampledIMU:           (N', 9)
resampled timestamps:   (N',)
resampling metadata
```

Required behavior:

- `RESAMPLE_RATE_HZ=none` bypasses the stage and preserves current behavior.
- Downsampling must include anti-alias filtering.
- Source timestamps may contain duplicates; resampling must not require strictly increasing canonical timestamps.
- Build a strictly increasing working time axis for resampling.
- Target timestamps must form a uniform grid.
- Do not extrapolate beyond the source recording.
- Initial scope supports `none` and downsampling only; upsampling is out of scope.
- Resample acceleration in `m/s²` and gyro, then recompute:

```text
acceleration_g = acceleration_m_s2 / 9.80665
```

Preserve the existing preprocessing artifacts and their current semantics. Resampling must not overwrite or redefine them.

---

## T002 — Encoder Uses the Effective Sampling Rate

The encoder must use the sampling rate of its actual input representation.

```text
RESAMPLE_RATE_HZ=none
→ encoder input = existing preprocessed representation
→ effective Fs = preprocessing sampling rate

RESAMPLE_RATE_HZ=64
→ encoder input = resampled representation
→ effective Fs = 64 Hz
```

Wavelet frequencies remain physical frequencies in Hz.

For example:

```text
frequencies_hz = [1, 2, 4, 8, 16]
```

At 64 Hz, wavelet widths must be recomputed from the 64 Hz effective rate. Existing 200 Hz widths must not be reused.

Preserve the existing spike channel layout and ordering.

Spike metadata must identify:

```text
effective sampling rate
input artifact identity/hash
resampling identity, when applicable
```

Therefore 200 Hz and 64 Hz encodings must have different compatibility identities.

---

## T003 — Pipeline Orchestration and Stale/Reuse Rules

Update orchestration to:

```text
pipeline_preprocess
→ pipeline_resample_if_requested
→ pipeline_encode
→ pipeline_align
→ pipeline_segment
→ pipeline_pad
```

Required dependency behavior:

```text
preprocessing changed
→ resampling + encoding + alignment + segmentation + padding stale

resampling config/spec changed
→ encoding + alignment + segmentation + padding stale

encoding changed
→ alignment + segmentation + padding stale
```

For example, changing:

```text
64 Hz → 128 Hz
```

must:

```text
reuse preprocessing
rebuild resampling
rebuild encoding
rebuild alignment
rebuild segmentation
rebuild padding
```

It must not unnecessarily rerun gravity preprocessing.

With:

```bash
RESAMPLE_RATE_HZ=none
```

the encoder reads the existing preprocessing artifact directly; no dummy resampling artifact is required.

---

## T004 — Alignment, Segmentation, and Padding Closure

Do **not** change the alignment algorithm itself.

Resampling changes:

```text
sample count
timestamp grid
SpikeIMU content/hash
event timing/waveform
```

Therefore:

```text
resample
→ re-encode
→ re-align
→ re-segment
→ re-pad
```

A 200 Hz alignment result must never be silently reused with a 64 Hz SpikeIMU artifact.

Existing alignment and segmentation provenance checks should remain fail-closed.

Segmentation boundary semantics remain unchanged; segmentation simply operates on the new timestamp/sample grid.

Padding length must be recomputed from the new segment lengths. For approximately 3 seconds:

```text
200 Hz → ~600 samples
64 Hz  → ~192 samples
```

Old 200 Hz padded artifacts must not be reused.

---

## T005 — Integration, Regression, and Documentation

Run focused tests for each task, then one complete small-data E2E:

```text
preprocess
→ resample 64 Hz
→ encode
→ align
→ segment
→ pad
```

Also run a regression path with:

```bash
RESAMPLE_RATE_HZ=none
```

to confirm existing behavior remains compatible.

Update durable documentation so the pipeline is explicitly documented as:

```text
gravity preprocessing
→ optional resampling
→ spike encoding
→ alignment
→ segmentation
→ padding
```

Resampling must remain a separate pipeline stage, not an encoder-internal operation.

---

# Verifier Must Check

`luna_verifier` must independently validate the complete contract surface, not merely confirm that worker tests pass.

| Scenario | Expected Result |
| --- | --- |
| `RESAMPLE_RATE_HZ=none` | Existing pipeline behavior remains compatible |
| `RESAMPLE_RATE_HZ=64` | Produces a uniform 64 Hz complete-recording representation |
| Source timestamps contain duplicates | Resampling succeeds without requiring strict source timestamps |
| 64 Hz output timestamps | Strictly increasing, uniform, and no extrapolation |
| Resampled 9-channel IMU | Channel order and units unchanged; `m/s² = g × 9.80665` |
| 64 Hz input → wavelet encoder | Encoder effective sampling rate is actually 64 Hz |
| Physical wavelet frequencies at 64 Hz | Wavelet sample widths are recomputed for 64 Hz |
| 200 Hz alignment artifact used in 64 Hz run | Rejected/stale; never silently reused |
| `continue`: 64 Hz → 128 Hz | Reuse preprocessing; rebuild resampling and all downstream stages |
| Preprocessing source changes | Resampling and all downstream artifacts become stale |
| Resampling metadata/hash mismatch | Fail closed |
| 64 Hz segmentation | Uses the new timestamps/sample grid |
| 64 Hz padding | Target length is recomputed from new segment lengths |
| Existing preprocessing artifacts | Not modified and their semantics remain unchanged |
| `vendor/**`, `data_sample/**` | No modifications |
| Complete 64 Hz mini-E2E | `preprocess → resample → encode → align → segment → pad` succeeds |

Verifier outcome:

```text
PASS
```

only when the complete TaskSpec is satisfied.

Use:

```text
FAIL
```

for implementation defects, missing validation, or failing tests.

Use:

```text
REPLAN
```

only if the TaskSpec itself must materially change, such as requiring a change to timestamp-domain or alignment semantics.

---

## Non-Goals

Do not include:

```text
upsampling
downsampling generated spikes
changes to gravity-removal algorithms
changes to wavelet mathematical definitions
changes to alignment matching algorithms
changes to segmentation boundary semantics
SNN/CNN architecture changes
vendor/** changes
data_sample/** changes
```

## Final Contract

```text
preprocessed representation
        │
        ├─ RESAMPLE_RATE_HZ=none ───────────────┐
        │                                       ↓
        └─ RESAMPLE_RATE_HZ=<rate> → resample → encoder
                                                ↓
                                            alignment
                                                ↓
                                          segmentation
                                                ↓
                                             padding
```

**Resampling is an independent optional stage between preprocessing and spike encoding. Changing the resampled representation invalidates the downstream lineage from encoding onward, while the original gravity-preprocessed representation remains reusable.**
