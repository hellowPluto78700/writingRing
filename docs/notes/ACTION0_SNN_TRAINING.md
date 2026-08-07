# Action0 label-segment SynNet training

`python -m snn.train_action0` is an independent, from-scratch segment
classifier. It always uses the local snnTorch `SynNet`, fixed label boundaries,
and channels `0:15` from padded SpikeIMU input. It does not reuse the legacy
HAR dataset/transforms, Board-event targets, Rockpool deployment model, or a
random segment split.

## Producer contract

The Action0 producer has two distinct representations:

```text
<variant>/label/segmentation/
    <user>/action_0/<user>_action_0_spikeIMU.npy       # (N, 21), variable length
    <user>/action_0/<user>_action_0_segment_offsets.npy
    <user>/action_0/<user>_action_0_segment_lengths.npy

<variant>/label/segmentation_padded/
    <user>/action_0/<user>_action_0_paddedSpikeIMU.npy # (S, T_pad, 21)
    <user>/action_0/<user>_action_0_labels.npy         # (S,)
    <user>/action_0/<user>_action_0_valid_lengths.npy  # (S,)
    <user>/action_0/<user>_action_0_valid_mask.npy     # (S, T_pad)
```

Training consumes `segmentation_padded`, not `segmentation`. This is an
intentional correction to the plan's illustrative path: the actual producer
in `src/writingring/segment_padding.py` creates the padded tensor and mask
there. Before padding, the producer uses `offsets[i]:offsets[i + 1]` to select
each segment. After padding, segment `i` is already directly addressed by
axis 0, so offsets are not copied into the padded package.

The dataset validates that every padded package is `(S, T_pad, 21)`, all
packages share `T_pad`, values are finite, masks are boolean contiguous
prefixes, and each valid length agrees with its mask. It returns:

```text
x          torch.float32  (T_pad, 15)
label      torch.long     ()
valid_mask torch.bool     (T_pad,)
```

The final six IMU channels are never passed to the model.

## Variant and split rules

The only accepted values for `--dataset_variant` are:

| CLI value | Producer directory |
| --- | --- |
| `lowpass` | `low-pass` |
| `raw` | `raw` |
| `madgwick` | `madgwick` |
| `xylo` | `xylo` |

The boundary is fixed internally to `label`; there is no boundary CLI option.
The train, validation, and test user lists are required and must be disjoint.
One global label-to-index mapping is discovered from the selected variant and
then shared by all three splits.

## Network and masking behavior

The default SynNet architecture is `[24, 24, 24]`, with `shift_syn=2` and
`shift_mem=1`. The original heterogeneous hidden-layer alphas remain intact:

```text
layer 1 shifts: 2-3
layer 2 shifts: 2-5
layer 3 shifts: 2-9
beta: 1 - 2**(-shift_mem) = 0.5
```

`sample_freq` is retained for the original tau diagnostics only; it never
remaps the shifts or changes alpha/beta. The model still advances all time
steps, including right padding. The mask affects only the cross-entropy mean,
final spike-count prediction, spike statistics, and optional spike
regularization. Thus padding cannot change loss, predicted class, or
`spkTotal`.

Rockpool is imported only if the legacy `SynNetRP` architecture is explicitly
selected. Action0 training always selects local `SynNet`, including when the
dataset variant is named `xylo`.

## Run a dry run

For the inspected low-pass producer output, the published sampling frequency
is 200 Hz. A minimal real-data smoke run is:

```bash
conda run --no-capture-output -n writingring-viz \
    python -m snn.train_action0 \
    --dataset_variant lowpass \
    --pipeline_root outputs/action0_pipeline \
    --sample_freq 200 \
    --train_users user_0 \
    --val_users user_1 \
    --test_users user_2 \
    --batch_size 1 \
    --dry_run
```

`--dry_run` resolves data, builds the datasets and DataLoaders, runs a forward
pass, masked loss, backward pass, and one optimizer step before exiting. Normal
training uses Adam and optionally saves the best validation-balanced-accuracy
checkpoint with the dataset variant, fixed boundary, mapping, channels, shifts,
sample frequency, and user splits.

## Upstream and producer differences

The upstream `vendor/WritingRing/ring_plot.py` reads only `*_ring_0.bin` as a
float64 `(N, 7)` matrix, while `board_plot.py` reads gzipped Board pickles for
visualization. This trainer intentionally does neither: it consumes the
repository's already-validated 21-channel SpikeIMU producer outputs and does
not infer any undocumented unit. It also never reads `*_ring_1.bin`.
