# Writing-motion ablation preprocessing

`build_writing_motion_variants.bash` derives two writing-only representations
from a completed 64 Hz polarity-split `aligned-board-events` dataset while
keeping the original dataset as D0.

- D0: `Encoder(a)` — source dataset, referenced without copying feature arrays.
- D1: `m * Encoder(a)` — zero every motion channel outside assigned valid
  press/lift intervals after the existing encoder.
- D2: `m * Encoder(m * a)` — hard-mask the complete 64 Hz recording
  acceleration before running the exact source Custom Wavelet encoder, then
  mask the resulting segment again.

The source alignment and segmentation decisions are authoritative. The builder
does not recompute offsets or Board policies. Complete valid non-transient
physical touch pairs are retained in the recording-level D2 encoder state;
final segment exports keep only pairs already assigned to that published
segment. Press and lift mapped samples are inclusive.

Boundary corner cases follow an observable-intersection policy. A valid touch
that partially extends beyond the Ring recording is clipped to the available
recording rows; a touch wholly outside the recording is ignored because it has
no observable Ring sample. An already-assigned touch is intersected with the
final published segment slice, which may be narrower after carry-in/crossing or
neighbor-boundary collision resolution. The builder never expands or shifts the
source segment to recover clipped touch samples. These cases are counted in
the per-user reports and interval CSV audit fields.

## Run locally

```bash
MODE=local JOBS=16 \
WRITE_MASK_VERIFICATION=1 \
bash scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash \
  outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events
```

## Submit on Unity

```bash
MODE=submit SLURM_MAX_CONCURRENCY=20 \
bash scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash \
  outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events
```

Each user is one CPU worker. The finalizer publishes
`writing_motion_ablation_manifest.json` and root padding summaries only after
all user reports pass.

Each derived segmentation package additionally contains
`*_writing_mask.npy` and `*_writing_intervals.csv`; padded packages contain
`*_padded_writing_mask.npy`. D2 also writes recording-level masked-acceleration
verification PNG/JSON files under `masking/verification/<user>/action_<action>/`.

## Re-running after a partial Slurm array

If a previous array produced only a partial derived dataset, rerun the full
builder with `OVERWRITE_DEST=1` so successful users from the earlier attempt
are replaced transactionally along with previously failed users.

```bash
MODE=submit OVERWRITE_DEST=1 SLURM_MAX_CONCURRENCY=20 \
WRITE_MASK_VERIFICATION=1 \
bash scripts/bash_script/preprocessing_pipeline/build_writing_motion_variants.bash \
  outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events
```
