# Plan: Action0 SNN Training Process Correction and Contract Hardening

## 0. Plan Metadata

```text
Plan:
    02_SNN_Train_Process_Correction_Plan

Status:
    DONE

Repository:
    hellowPluto78700/writingRing

Validated baseline:
    29da4396b60d9357e0b8edffdb6f02ec4011a987

Primary durable contract:
    docs/notes/ACTION0_SNN_TRAINING.md

Related historical implementation plan:
    docs/plans/TODO/
    Check0_Ring_Action_Segment_SNN_Train_Code_Transform_Plan.md

Implementation scope:
    snn/action0_dataset.py
    snn/action0_losses.py
    snn/action0_engine.py
    snn/action0_parser.py
    snn/train_action0.py
    snn/utils_architectures.py
    tests/test_snn_action0_*.py

Producer source of truth:
    scripts/action0_pipeline/_common.bash
    src/writingring/segment_padding.py
    src/writingring/recording_features.py

Companion task file:
    docs/plans/Done/
    02_SNN_Train_Process_Correction_TASKS.md

Orchestration state:
    docs/plans/WORKBOARD.md
```

---

# 1. Background

The repository already contains an independent Action0 SynNet training path:

```text
python -m snn.train_action0
```

The current implementation was introduced as a label-segmented Action0 baseline and already implements most of the intended behavior:

```text
dataset variants:
    lowpass
    raw
    madgwick
    xylo

boundary:
    fixed internally to label

training representation:
    segmentation_padded

input:
    SpikeIMU channels 0:15

source SpikeIMU width:
    21 channels

network:
    local snnTorch SynNet

default hidden sizes:
    [24, 24, 24]

default shifts:
    shift_syn = 2
    shift_mem = 1

criterion:
    masked cross entropy

split:
    user-disjoint train / validation / test

training:
    Adam
    train from scratch baseline

padding behavior:
    padding is excluded from loss,
    final prediction,
    spike statistics,
    and spike regularization
```

The implementation is therefore not missing as a whole.

The purpose of this plan is to perform a second-stage correction and hardening pass so that the training path has:

1. an explicit producer/trainer data contract;
2. fail-fast validation of producer metadata;
3. consistent padded length and sampling-rate provenance;
4. deterministic dry-run semantics;
5. strict model-output validation;
6. well-defined checkpoint semantics;
7. correct epoch-level loss aggregation;
8. regression coverage for the above;
9. independent verification under the repository's current multi-agent workflow;
10. clear reconciliation with the historical Check0 acceptance criteria.

This plan must not redesign the SNN baseline.

---

# 2. Repository Knowledge and Contract Priority

For this work, use repository artifacts in the priority defined by `AGENTS.md`:

```text
1. docs/notes/**
2. README.md
3. current code and tests
4. docs/plans/**
5. FROZEN TaskSpec for the active implementation task
```

Therefore:

```text
docs/notes/ACTION0_SNN_TRAINING.md
```

is the durable current Action0 training contract.

The producer implementation in:

```text
src/writingring/segment_padding.py
scripts/action0_pipeline/_common.bash
```

is the source of truth for the actual artifact schema.

The historical Check0 plan remains useful for intended behavior and final acceptance, but illustrative assumptions in that plan must not override current verified producer behavior.

---

# 3. Important Plan Correction: `segmentation_padded`

The historical Check0 plan contains examples that point training at:

```text
<variant>/label/segmentation/
```

That path is no longer the correct fixed-length training representation.

The actual producer contract is:

```text
<variant>/label/segmentation/

    variable-length continuous SpikeIMU representation

    <user>/action_0/
        <user>_action_0_spikeIMU.npy
        <user>_action_0_labels.npy
        <user>_action_0_segment_offsets.npy
        <user>_action_0_segment_lengths.npy
```

followed by:

```text
<variant>/label/segmentation_padded/

    fixed-length training representation

    padding_dataset_summary.json
    padding_dataset_manifest.csv

    <user>/action_0/
        <user>_action_0_paddedSpikeIMU.npy
        <user>_action_0_labels.npy
        <user>_action_0_valid_lengths.npy
        <user>_action_0_valid_mask.npy
        ...
```

The trainer must continue consuming:

```text
segmentation_padded
```

and must not be changed back to:

```text
segmentation
```

The trainer must not:

```text
re-pad
re-segment
re-encode spikes
downsample
quantize
reconstruct segments using offsets
```

After padding, segment `i` is addressed directly on axis 0 of the padded tensor.

---

# 4. Current Implementation Assessment

## 4.1 Already implemented correctly

The following behavior is considered implemented and must be preserved.

### Dataset variants

```text
lowpass  -> low-pass
raw      -> raw
madgwick -> madgwick
xylo     -> xylo
```

### Fixed boundary

```text
BOUNDARY = "label"
```

There must be no Action0 training CLI option:

```text
--boundary
```

and no Action0 training branch for:

```text
aligned-board-events
```

### Dataset representation

Producer package:

```text
paddedSpikeIMU:
    (S, T_pad, 21)

labels:
    (S,)

valid_lengths:
    (S,)

valid_mask:
    (S, T_pad)
```

Dataset item:

```text
x:
    torch.float32
    (T_pad, 15)

label:
    torch.long
    ()

valid_mask:
    torch.bool
    (T_pad,)
```

### Channel selection

The network must receive only:

```text
SpikeIMU[..., 0:15]
```

The final six IMU channels:

```text
15:21
```

must never enter the Action0 SynNet.

### Label mapping

One deterministic global:

```text
class_to_idx
```

mapping must be discovered from the selected variant and reused for train, validation, and test.

### User split

```text
train_users
val_users
test_users
```

must be non-empty and pairwise disjoint.

A random segment-level split must not be introduced.

### Loss masking

Classification loss must ignore:

```text
valid_mask == False
```

positions.

### Prediction masking

Final class prediction must use only masked output spike counts.

### Spike statistics

Padding must not contribute to:

```text
spkTotal
mean total spikes
output spike count
spike regularization
```

### SynNet topology and dynamics

The existing local SynNet topology remains:

```text
fc1 / lif1
    ↓
fc2 / lif2
    ↓
fc3 / lif3
    ↓
fc4 / lif4
```

Hidden sizes remain configurable with the baseline default:

```text
[24, 24, 24]
```

The original shift definition must remain:

```python
shiftsSyn = list(range(shiftSyn, shiftSyn + 8))
```

and:

```python
beta = 1 - 2 ** (-shiftMem)
```

For:

```text
shift_syn = 2
shift_mem = 1
```

the expected behavior remains:

```text
lif1 synaptic shifts:
    2, 3

lif2:
    2, 3, 4, 5

lif3:
    2, 3, 4, 5, 6, 7, 8, 9

hidden membrane beta:
    0.5
```

### Sampling frequency semantics

`sample_freq` is retained for:

```text
experiment metadata
tau diagnostics
constructor compatibility
```

It must never automatically change:

```text
shift_syn
shift_mem
alpha
beta
```

### Mask/state semantics

Padding time steps must still advance the SynNet state.

Do not:

```text
skip masked time steps
freeze hidden state
reset at valid_length
shorten model input dynamically
```

The mask controls training/statistics semantics, not neuron dynamics.

### Rockpool

Action0 training always uses:

```text
network_type = SynNet
```

including:

```text
dataset_variant = xylo
```

Rockpool must be loaded only when the legacy:

```text
SynNetRP
```

path is explicitly requested outside the Action0 baseline.

---

# 5. Problems to Correct

## P0-1 — Producer metadata is not consumed by the trainer

The padded producer root already publishes:

```text
padding_dataset_summary.json
```

including metadata such as:

```text
target_length
sampling_rate_hz
channel_count
input_kind
feature_schema
padding_side
overflow_policy
segment counts
```

The current trainer validates individual `.npy` arrays but does not use this root metadata as a training contract.

Consequences include:

```text
wrong sample_freq can be supplied manually;

trainer can report a sampling rate different from
the producer's published sampling rate;

checkpoint metadata can therefore contain incorrect provenance;

producer target_length is not explicitly tied to
the train/val/test datasets.
```

---

## P0-2 — Common `T_pad` is not enforced across all splits

Each individual `Action0SegmentDataset` validates that its own packages share one padded length.

However, train, validation, and test are constructed independently.

Therefore this invalid state is not guaranteed to fail:

```text
train:
    T_pad = 512

validation:
    T_pad = 640

test:
    T_pad = 512
```

All three splits must be validated against one producer-defined:

```text
target_length
```

---

## P0-3 — Dry-run inspection and optimization may use different batches

The trainer currently obtains a first shuffled train batch for inspection and preview.

It then creates another iterator when `run_epoch(..., max_batches=1)` is called.

Because the train DataLoader uses:

```text
shuffle=True
```

the batch used for:

```text
inspection
preview forward
```

is not guaranteed to be the batch used for:

```text
loss
backward
optimizer.step
```

`--dry_run` must represent one coherent batch from input inspection through optimizer step.

---

## P0-4 — Checkpoint compatibility validation is incomplete

The checkpoint stores substantial experiment metadata, but loading currently validates only a subset.

A checkpoint should not silently load when configuration differs in fields that affect model/data semantics.

At minimum, compatibility checks are needed for:

```text
dataset_variant
boundary
class_to_idx
input_channels
num_inputs
num_outputs
hidden_sizes
shift_syn
shift_mem
sample_freq
train_users
val_users
test_users
```

Checkpoint execution semantics must also be explicitly defined.

The project must not claim "exact resume" unless epoch state, best-state semantics, random state, and DataLoader shuffle state are all handled consistently.

---

## P1-1 — Engine does not assert exact class dimension

The engine currently verifies:

```text
output.ndim == 3
output[:2] matches input[:2]
number of classes > 0
```

but it does not verify:

```text
output.shape[2] == expected_num_classes
```

This should become a hard invariant.

---

## P1-2 — Epoch loss aggregation is not the global valid-timestep mean

The criterion correctly computes a masked valid-timestep mean within each batch.

The engine then aggregates batch losses using approximately:

```text
batch loss × batch size
```

This weights batches by number of segments instead of number of valid time steps.

For uneven segment lengths, reported epoch loss therefore differs from the actual global valid-timestep masked mean.

Gradient calculation is not affected.

Metric reporting is.

---

## P1-3 — Historical absolute paths remain in legacy `/snn`

Legacy files still contain machine-specific paths such as:

```text
/home/...
/work/...
```

The historical Check0 acceptance criterion states:

```text
no old-machine absolute path
```

The scope of that statement is ambiguous:

```text
Action0 path only
```

versus:

```text
entire /snn tree
```

This must be resolved by Probe and frozen TaskSpec before editing legacy code.

Legacy cleanup must not be allowed to accidentally redesign the HAR trainer.

---

## P1-4 — Verification evidence is incomplete

Action0 regression tests exist, but several new hardening behaviors are not covered.

Additionally, final verification must ensure SNN tests actually execute rather than being skipped because optional dependencies are unavailable.

The repository workflow requires:

```text
Python 3.11
writingring-viz
pytest after code changes
```

The current workboard has previously recorded a Python 3.10.x environment discrepancy.

Python version must therefore become an explicit verification gate.

---

# 6. Goals

This plan is complete when the Action0 baseline has the following properties.

### G1 — Producer contract is explicit

Training validates the exact padded producer package it consumes.

### G2 — Data provenance cannot silently disagree

Trainer sampling metadata must agree with published producer sampling metadata.

### G3 — Padded geometry is globally consistent

All train/validation/test packages must use the producer's single target length.

### G4 — Existing SNN dynamics remain unchanged

Contract hardening must not modify the original SynNet topology or shift behavior.

### G5 — Padding remains semantically excluded

Padding cannot influence:

```text
classification loss
prediction
spike statistics
spike regularization
```

### G6 — Dry-run is a real one-batch training step

The same batch must flow through:

```text
inspect
→ forward
→ loss
→ backward
→ optimizer.step
```

### G7 — Checkpoint mismatch fails early

A checkpoint incompatible with the selected dataset/model configuration must be rejected before loading model state.

### G8 — Metrics have precise semantics

Epoch loss must correspond to the masked valid-timestep objective being reported.

### G9 — Tests provide direct evidence

All important contracts must have targeted regression tests.

### G10 — Repository workflow is satisfied

Every implementation task must pass:

```text
PROBE
→ FROZEN TaskSpec
→ Worker
→ fresh Verifier
→ documentation
→ DONE
```

---

# 7. Non-Goals

This plan must not introduce:

```text
new SNN architectures

SynNet topology changes

learn_alpha

learn_beta

sample-rate-to-shift remapping

new surrogate gradients

FirstWin loss

random segment splitting

all21 feature mode

imu6 feature mode

Board-event targets in Action0 training

aligned-board-events Action0 training

automatic Rockpool selection

transfer learning as baseline behavior

new padding algorithms

new spike encoding

new downsampling

new quantization

new Action0 preprocessing behavior

performance tuning

hyperparameter search

final model-accuracy optimization
```

This plan is about correctness, contract enforcement, reproducibility, and verification.

---

# 8. Producer Contract to Freeze

The expected producer root is:

```text
<pipeline_root>/
    <variant_dir>/
        label/
            segmentation_padded/
```

Supported variant mapping:

```text
lowpass  -> low-pass
raw      -> raw
madgwick -> madgwick
xylo     -> xylo
```

The producer root must contain:

```text
padding_dataset_summary.json
```

The frozen trainer contract should validate at least:

```text
input_kind == "spike-imu"

feature_schema ==
    canonical SpikeIMU feature schema

channel_count == 21

target_length:
    integer
    > 0

sampling_rate_hz:
    finite
    > 0

padding_side == "right"
```

Where supported by the producer metadata, also validate consistency of:

```text
processed_user_action_count
source_segment_count
segment_count
skipped_segment_count
label-only / board-assisted package counts
```

Only fields confirmed by the Probe as stable producer contract should become mandatory trainer checks.

Do not invent producer fields.

---

# 9. Training Dataset Contract

For every selected Action0 package:

```text
paddedSpikeIMU:
    numeric
    finite
    shape = (S, T_pad, 21)

labels:
    shape = (S,)

valid_lengths:
    integer
    shape = (S,)
    1 <= valid_length <= T_pad

valid_mask:
    bool
    shape = (S, T_pad)
```

For every segment:

```text
valid_mask[i, t] ==
    (t < valid_lengths[i])
```

so the mask is always a contiguous valid prefix followed by right padding.

Dataset output remains:

```text
features:
    float32
    (T_pad, 15)

label:
    long scalar

valid_mask:
    bool
    (T_pad,)
```

---

# 10. Model Contract

Action0 must construct:

```text
local snnTorch SynNet
```

with:

```text
inputSize = 15

outputSize =
    len(class_to_idx)

hiddenSizes =
    args.neurons_network

sampleFreq =
    args.sample_freq

shiftSyn =
    args.shift_syn

shiftMem =
    args.shift_mem
```

Do not calculate model alpha/beta in the trainer.

The trainer may calculate tau values only for human-readable diagnostics.

---

# 11. Mask Contract

The model still processes:

```text
all T_pad time steps
```

including right padding.

The mask is applied only to:

```text
masked CE

final output spike count

prediction

spkTotal

spike statistics

optional spike regularization
```

No masked timestep may be skipped inside the SynNet recurrent temporal evolution.

---

# 12. Task DAG

Initial dependency graph:

```text
PLAN INITIALIZATION
        │
        ▼
T001 Producer / Trainer Contract Hardening
        │
        ▼
T002 Engine + Dry-Run Correctness
        │
        ▼
T003 Checkpoint Contract
        │
        ├────────────────────┐
        │                    │
        ▼                    ▼
T005 Verification       T004 Legacy Scope Probe
        ▲                    │
        │                    ├─ scope = Action0 only
        │                    │      → no legacy code change
        │                    │
        │                    └─ scope = whole /snn
        │                           ↓
        │                       T004B Legacy Portability
        │                           │
        └───────────────────────────┘
                    │
                    ▼
          T006 Documentation / Closure
```

`T004` may be probed in parallel with earlier read-only work.

Implementation should remain sequential unless the PRIMARY thread confirms write scopes do not overlap.

---

# 13. Plan Initialization

Before implementation:

1. Set this plan as the active plan in:

```text
docs/plans/WORKBOARD.md
```

1. Create:

```text
docs/plans/TODO/
02_SNN_Train_Process_Correction_TASKS.md
```

1. Record baseline:

```text
29da4396b60d9357e0b8edffdb6f02ec4011a987
```

1. Set all implementation tasks initially to:

```text
DRAFT
```

1. Do not fully freeze downstream TaskSpecs in advance.

Use:

```text
DAG early.
TaskSpec late.
```

Dependency-ready tasks are probed and frozen one at a time.

---

# 14. T001 — Producer / Trainer Contract Hardening

## Status

```text
DONE
```

## Dependencies

```text
none
```

## Goal

Make `segmentation_padded` producer metadata an explicit runtime training contract.

## Probe scope

`luna_probe` should inspect only the relevant artifacts:

```text
docs/notes/ACTION0_SNN_TRAINING.md

docs/notes/SEGMENT_PADDING.md

scripts/action0_pipeline/_common.bash

src/writingring/segment_padding.py

src/writingring/recording_features.py

snn/action0_dataset.py

snn/train_action0.py

tests/test_snn_action0_dataset.py

tests/test_snn_action0_smoke.py
```

## Probe questions

Confirm:

```text
exact root summary filename

stable root summary keys

canonical SpikeIMU schema constant

channel count

target_length semantics

sampling_rate_hz semantics

padding_side semantics

whether label/Board package-count fields
are mandatory or diagnostic

whether importing canonical schema constants
from writingring.recording_features is safe
from the SNN package
```

## Freeze conditions

TaskSpec may become FROZEN only after the producer contract is confirmed.

## Expected allowed writes

```text
snn/action0_dataset.py

snn/train_action0.py

tests/test_snn_action0_dataset.py

tests/test_snn_action0_smoke.py
```

If required by confirmed import architecture:

```text
tests supporting the producer metadata contract
```

## Forbidden writes

```text
src/writingring/**

scripts/action0_pipeline/**

snn/utils_architectures.py

vendor/**

data_sample/**
```

Producer behavior is not being redesigned by this task.

## Expected implementation

Introduce a validated metadata representation, for example:

```text
PaddingDatasetMetadata
```

containing confirmed stable fields such as:

```text
target_length
sampling_rate_hz
channel_count
input_kind
feature_schema
padding_side
```

Add a loader similar to:

```text
load_padding_dataset_metadata(...)
```

which reads:

```text
segmentation_padded/
padding_dataset_summary.json
```

and fails with explicit contract errors for malformed metadata.

Training startup should become:

```text
resolve variant
→ resolve label root
→ resolve segmentation_padded
→ load producer metadata
→ validate producer contract
→ discover global class mapping
→ build train dataset
→ build validation dataset
→ build test dataset
→ validate split datasets against producer metadata
→ build DataLoaders
```

## Sampling-rate validation

Require:

```text
CLI sample_freq
==
producer sampling_rate_hz
```

using an explicitly defined floating-point comparison policy.

This validation is metadata/provenance validation only.

It must not change:

```text
shift_syn
shift_mem
alpha
beta
```

## Global padded-length validation

Require:

```text
train_dataset.padded_length
==
validation_dataset.padded_length
==
test_dataset.padded_length
==
producer.target_length
```

## Logging

Startup diagnostics should expose at least:

```text
Dataset variant

Boundary

Dataset root

Segmentation root

Producer:
    input_kind
    feature_schema
    channel_count
    target_length
    sampling_rate_hz

Trainer:
    input channel slice 0:15
```

## Acceptance criteria

T001 PASS requires tests showing:

```text
valid metadata loads

missing root summary fails

malformed summary fails

wrong input_kind fails

wrong feature_schema fails

wrong channel_count fails

invalid target_length fails

wrong padding_side fails

sample_freq mismatch fails

package T_pad != producer target_length fails

cross-split T_pad mismatch fails

valid existing producer package still trains
```

## Replan triggers

Return to PRIMARY if Probe discovers:

```text
root summary is not a stable producer contract;

sampling_rate_hz has semantics different from
the trainer's sample_freq;

canonical schema constants cannot be imported
without creating an undesirable package dependency;

different variants legitimately publish different
metadata structures that require architecture-level
policy.
```

---

# 15. T002 — Engine and Dry-Run Correctness

## Status

```text
DONE
```

## Dependencies

```text
T001 DONE
```

## Goal

Make output shape checks, epoch loss reporting, and dry-run behavior mathematically and operationally precise.

## Probe scope

```text
snn/action0_engine.py

snn/action0_losses.py

snn/train_action0.py

tests/test_snn_action0_loss.py

tests/test_snn_action0_smoke.py
```

## Required contracts to preserve

```text
masked CE formula

masked final spike counts

zero-spike score stability

valid-neuron-time spike regularization

one optimizer step per train batch

existing SynNet dynamics
```

## Expected allowed writes

```text
snn/action0_engine.py

snn/train_action0.py

tests/test_snn_action0_loss.py

tests/test_snn_action0_smoke.py
```

## Forbidden writes

```text
snn/utils_architectures.py
unless Probe finds an implementation defect
that cannot be corrected within the frozen scope

producer code

pipeline code
```

## Change A — Exact output class dimension

`run_epoch()` should receive or derive:

```text
expected_num_classes
```

and require:

```text
output.shape ==
    (
        batch_size,
        padded_time,
        expected_num_classes,
    )
```

Do not accept merely:

```text
class_count > 0
```

## Change B — Epoch loss aggregation

Batch objective remains:

```text
masked valid-timestep mean
```

Epoch loss should be accumulated using valid timestep weight.

Conceptually:

```text
batch_valid_steps =
    valid_mask.sum()

loss_numerator +=
    batch_loss * batch_valid_steps

valid_step_total +=
    batch_valid_steps

epoch_loss =
    loss_numerator /
    valid_step_total
```

The Worker must verify that this remains correct when spike regularization is enabled.

## Change C — Dry-run same-batch behavior

The dry run must obtain one batch:

```text
first_batch
```

and use that exact batch for:

```text
inspection
preview output
masked loss
backward
optimizer.step
report
```

It must not create a second shuffled train iterator for the optimizer step.

Reuse shared training code where possible.

Do not duplicate an independent manual training implementation unless the frozen TaskSpec requires it.

## Acceptance criteria

Tests must demonstrate:

```text
wrong output class count fails

epoch loss equals the global masked
valid-timestep objective

padding-only output modifications do not
change classification loss

padding-only spike modifications do not
change predictions

dry-run preview batch equals optimization batch

dry-run performs exactly one optimizer.step()

dry-run exits after that step
```

## Replan triggers

Replan if:

```text
loss object cannot expose enough information
to aggregate epoch loss correctly;

spike regularization causes batch-weighting semantics
to differ from the frozen objective;

dry-run reuse requires changing the public engine API
beyond the frozen scope.
```

---

# 16. T003 — Checkpoint Contract and Restore Semantics

## Status

```text
DONE
```

## Dependencies

```text
T002 DONE
```

## Goal

Prevent incompatible checkpoints from silently changing the experiment and define exactly what `--model_checkpoint` means.

## Important design constraint

Do not call checkpoint loading:

```text
exact deterministic resume
```

unless all required state for exact trajectory continuation is restored.

That may include:

```text
model state
optimizer state
epoch
best metric
best model state semantics
Python RNG state
NumPy RNG state
torch CPU RNG state
CUDA RNG state
DataLoader/shuffle generator state
```

The current task must therefore Probe and freeze one explicit checkpoint model.

## Preferred low-risk decision

Unless exact trajectory resume is a current product requirement, prefer:

```text
configuration-compatible checkpoint restore
```

with precise documentation rather than claiming exact deterministic resume.

If exact resume is required, expand the TaskSpec accordingly before implementation.

## Probe questions

Determine:

```text
Is --model_checkpoint intended for:

A. warm/model restore

B. optimizer continuation

C. true epoch resume

D. deterministic exact resume
```

Also determine:

```text
meaning of num_epochs after loading;

whether learning_rate may differ;

whether spike_regularization may differ;

whether user splits may differ;

whether optimizer state must load;

whether old checkpoints without schema version
must remain supported.
```

## Current metadata requiring compatibility validation

At minimum:

```text
dataset_variant

boundary

class_to_idx

input_channels

num_inputs

num_outputs

hidden_sizes

shift_syn

shift_mem

sample_freq

train_users

val_users

test_users
```

Depending on the frozen semantics, also include:

```text
learning_rate

spike_regularization

random_seed

checkpoint_schema_version
```

## Recommended checkpoint schema

Introduce:

```text
checkpoint_schema_version
```

for future compatibility handling.

Do not silently infer incompatible historical checkpoint formats.

## Validation ordering

Checkpoint compatibility must be checked:

```text
before model state is applied
```

and before optimizer state is applied.

Mismatch messages should name:

```text
field
checkpoint value
requested/current value
```

## Acceptance criteria

Regression tests must reject at least:

```text
wrong dataset variant

wrong boundary

wrong class mapping

wrong input channels

wrong num inputs

wrong num outputs

wrong hidden sizes

wrong shift_syn

wrong shift_mem

wrong sample_freq

wrong user split
```

If exact resume is selected, additionally test:

```text
epoch continuation

best metric continuation

optimizer continuation

required RNG/shuffle state behavior

resumed trajectory contract
```

## Replan triggers

Mandatory REPLAN if deciding checkpoint semantics requires choosing between:

```text
warm start
versus
true resume
```

and that decision cannot be derived from current repository contracts.

Do not allow the Worker to decide this implicitly.

---

# 17. T004 — Legacy `/snn` Absolute-Path Scope Reconciliation

## Status

```text
DONE — Scope B; T004B completed
```

## Dependencies

Probe may run independently.

Implementation, if needed, should complete before final verification.

## Goal

Resolve historical Check0 acceptance condition:

```text
[22]
no old-machine absolute path
```

without unnecessarily destabilizing the legacy HAR trainer.

## Probe scope

```text
docs/plans/TODO/
Check0_Ring_Action_Segment_SNN_Train_Code_Transform_Plan.md

docs/notes/ACTION0_SNN_TRAINING.md

snn/har_snn.py

snn/utils_run.py

snn/utils_parser.py

relevant legacy tests
```

## Required Probe decision

Classify `[22]` as one of:

```text
Scope A:
    Action0 training path only

Scope B:
    entire /snn tree
```

## Scope A outcome

If `[22]` applies only to the new Action0 path:

```text
no legacy implementation changes are required
```

Record the clarified scope in:

```text
02 plan/task documentation
```

and reconcile the historical Check0 wording during final documentation.

## Scope B outcome

Create a separate frozen implementation TaskSpec:

```text
T004B Legacy SNN Portability
```

### Expected write scope

```text
snn/har_snn.py

snn/utils_run.py

possibly snn/utils_parser.py

relevant tests
```

### Required behavior

Remove:

```text
machine-specific sys.path insertion

hard-coded /home/... model paths

hard-coded /work/... dataset paths
```

Replace them with:

```text
pathlib.Path

package-relative imports where appropriate

explicit CLI/configurable roots
```

### Contracts to preserve

Do not change:

```text
legacy HAR data transformation semantics

legacy network selection

legacy loss behavior

legacy metrics

legacy model topology

legacy training schedule
```

This task is portability cleanup only.

## Replan triggers

Replan if legacy path cleanup exposes:

```text
undocumented deployment assumptions

external filesystem dependencies

expected compatibility with old launch scripts

architecture-level import packaging issues
```

---

# 18. T005 — Regression Matrix and Runtime Verification

## Status

```text
DONE
```

## Dependencies

```text
T001 DONE
T002 DONE
T003 DONE

and, if Scope B applies:
T004B DONE
```

## Goal

Provide independent evidence that the corrected SNN path satisfies both the current durable contract and applicable Check0 acceptance criteria.

No new behavior should normally be introduced in T005.

---

## 18.1 Environment Gate

Before interpreting pytest results:

```bash
conda run --no-capture-output \
  -n writingring-viz \
  python --version
```

Required:

```text
Python 3.11.x
```

If the environment still reports Python 3.10.x:

```text
verification is BLOCKED
```

for repository-policy compliance until the environment discrepancy is resolved or the repository policy is explicitly changed by the appropriate owner.

Do not silently ignore it.

---

## 18.2 Targeted SNN Test Suite

Run:

```bash
conda run --no-capture-output \
  -n writingring-viz \
  pytest -q -rs \
  tests/test_snn_action0_dataset.py \
  tests/test_snn_action0_loss.py \
  tests/test_snn_action0_model.py \
  tests/test_snn_action0_smoke.py
```

Verifier must record:

```text
passed count

failed count

skipped count

whether torch tests executed

whether snntorch tests executed
```

A suite in which all core Action0 tests are skipped does not satisfy acceptance.

---

## 18.3 Required Regression Coverage

### Dataset contract

```text
all four variant path resolutions

boundary fixed to label

segmentation_padded path

global class mapping

only first 15 channels returned

finite data validation

bool contiguous-prefix mask validation

valid_lengths consistency

producer summary load

producer schema validation

producer channel count validation

producer target length validation

cross-split target length validation

sample frequency validation
```

### Split behavior

```text
train/val overlap fails

train/test overlap fails

val/test overlap fails
```

### Model behavior

```text
beta = 0.5 for shift_mem=1

layer 1 alpha shift mapping = 2,3

layer 2 = 2,3,4,5

layer 3 = 2..9

sample_freq 64 / 100 / 200
does not alter alpha/beta

mask does not change temporal advancement

masked spkTotal excludes padding
```

### Rockpool behavior

Test that:

```text
constructing and training local SynNet
does not require importing Rockpool
```

and:

```text
dataset_variant=xylo
still selects local SynNet
```

### Loss behavior

```text
padding cannot change CE

padding cannot change final prediction

all-zero spike outputs do not produce NaN

spike regularization uses valid neuron-time

epoch loss uses correct valid-timestep weighting
```

### Engine behavior

```text
exact output class dimension

input / label / mask geometry

finite loss

masked predictions

masked statistics
```

### Dry-run behavior

```text
same first batch used throughout

forward occurs

masked loss occurs

backward occurs

exactly one optimizer step occurs

report occurs

process exits after dry-run
```

### Checkpoint behavior

Cover all compatibility fields frozen in T003.

---

## 18.4 Full Repository Test Suite

After targeted tests:

```bash
conda run --no-capture-output \
  -n writingring-viz \
  pytest -q
```

Any regression caused by SNN changes must be repaired before final verification.

---

## 18.5 Real Producer Dry Run

Run at least one real:

```text
lowpass
label
segmentation_padded
```

dataset smoke test.

Representative command:

```bash
conda run --no-capture-output \
  -n writingring-viz \
  python -m snn.train_action0 \
  --dataset_variant lowpass \
  --pipeline_root outputs/action0_pipeline \
  --sample_freq <PUBLISHED_HZ> \
  --train_users <TRAIN_USER> \
  --val_users <VAL_USER> \
  --test_users <TEST_USER> \
  --batch_size 1 \
  --dry_run
```

The verifier must confirm output demonstrates:

```text
dataset variant

boundary = label

segmentation_padded root

producer input kind

producer feature schema

producer channels = 21

model input channels = 15

producer target T_pad

batch T_pad matches target

producer sampling frequency

CLI sampling frequency matches producer

shift_syn

shift_mem

beta

output shape = (B, T_pad, K)

finite loss

valid spike statistics

one optimizer step
```

---

## 18.6 Variant Resolution Verification

Synthetic/path regression tests must cover:

```text
lowpass:
    low-pass/label/segmentation_padded

raw:
    raw/label/segmentation_padded

madgwick:
    madgwick/label/segmentation_padded

xylo:
    xylo/label/segmentation_padded
```

No Action0 trainer path may resolve:

```text
aligned-board-events
```

---

## 18.7 Tiny Overfit Diagnostic

After smoke verification, optionally run a tiny controlled overfit diagnostic when suitable producer data is available.

Purpose:

```text
prove gradients affect parameters
prove loss can decrease
detect permanently dead output
detect NaN/Inf instability
```

This is not a model-quality benchmark.

Do not alter the formal user-disjoint evaluation protocol to achieve an overfit result.

---

# 19. T006 — Documentation, Reconciliation, and Plan Closure

## Status

```text
DONE
```

## Dependencies

```text
T005 verifier PASS
```

## Goal

Update durable documentation only after verified implementation behavior is known.

## PRIMARY-owned documentation scope

Potentially:

```text
docs/notes/ACTION0_SNN_TRAINING.md

docs/plans/TODO/
02_SNN_Train_Process_Correction_Plan.md

docs/plans/TODO/
02_SNN_Train_Process_Correction_TASKS.md

docs/plans/TODO/
Check0_Ring_Action_Segment_SNN_Train_Code_Transform_Plan.md

docs/plans/WORKBOARD.md

README.md
only if user-facing invocation or repository-level
behavior actually changed
```

## Required note updates

Document verified behavior such as:

```text
producer summary validation

sample_freq provenance validation

global T_pad validation

dry-run same-batch semantics

checkpoint compatibility semantics

epoch loss definition
```

Do not document unverified implementation intent as current behavior.

## Check0 reconciliation

Resolve the historical contradiction:

```text
Check0:
    segmentation/

Current producer:
    segmentation_padded/
```

Mark the historical path as illustrative/stale and point to the verified producer contract.

Also resolve `[22]` absolute-path scope according to T004.

## Plan state

Once every task is DONE:

```text
02 plan = DONE
```

If repository conventions call for completed plans to move under:

```text
docs/plans/Done/
```

perform that documentation-state transition and update `WORKBOARD.md` accordingly.

Do not leave stale active-plan pointers behind.

## T006 execution record

T005 was independently verified in the user-authorized isolated
`writingring-test` environment (Python 3.11.15), without modifying the
existing `writingring-viz` environment, which still reports Python 3.10.20.
The mirrored direct runtime pins passed imports and `pip check`; targeted
Action0 tests passed 26/0 skipped, checkpoint plus Scope-B legacy portability
tests passed 29/0 skipped, and the full suite passed 502 with one known skip
for an unavailable real `user_0/action_0/dataset_0` SpikeIMU artifact. The
real lowpass producer dry run passed with the verified padded root, 21→15
input slice, target length 1024, rate 200 Hz, shifts 2/1, beta 0.5, 52-class
output, finite loss, valid spike statistics, and the tested one-step path.

The optional tiny-overfit diagnostic was not run because no separately frozen
fixture/harness exists. The validation result does not establish real producer
coverage for raw, madgwick, or xylo; their four path mappings are covered by
synthetic regression tests.

## T006 verification and closure

The initial documentation verifier identified only stale lifecycle markers;
PRIMARY synchronized the plan, task record, and WORKBOARD and a fresh verifier
then passed the documentation contract. This plan is complete, and its plan
and companion task record have moved to `docs/plans/Done/`. README remained
unchanged because its repository-level invocation stayed accurate.

---

# 20. Required Task Lifecycle

Every implementation task follows:

```text
DRAFT
  ↓
PROBING
  ↓
FROZEN
  ↓
IMPLEMENTING
  ↓
VERIFYING
  ↓
DOCUMENTING
  ↓
DONE
```

## Probe

Spawn:

```text
luna_probe
```

for one dependency-ready DRAFT task.

Expected result:

```text
CONTRACT_PROBE_PACKET
```

Result handling:

```text
CONFIRMED
    → PRIMARY may freeze TaskSpec

REVISE
    → revise TaskSpec and re-probe if needed

BLOCKED
    → do not implement
```

## Freeze

Only PRIMARY marks a TaskSpec:

```text
FROZEN
```

Every frozen TaskSpec must contain:

```text
task ID

goal

source plan

dependencies

validated_against_commit

required behavior

contracts to preserve

contracts intentionally changed

allowed write paths

forbidden write paths

acceptance criteria

validation commands

replan triggers
```

## Implement

Spawn exactly one:

```text
luna_worker
```

for one FROZEN TaskSpec.

The Worker may only modify authorized paths.

If the TaskSpec becomes invalid:

```text
NEEDS_REPLAN
```

must be returned.

The Worker must not silently redesign the task.

## Verify

After Worker DONE, spawn a fresh:

```text
luna_verifier
```

Verifier independently checks:

```text
TaskSpec

actual diff

relevant contracts

tests

runtime evidence

acceptance criteria
```

Possible result:

```text
PASS
FAIL
REPLAN
```

Worker DONE alone never completes a task.

## Document

After Verifier PASS, PRIMARY updates only affected durable documentation.

Then and only then mark task:

```text
DONE
```

---

# 21. Continuous Orchestration Rule

Once the user asks PRIMARY to execute this full plan, do not stop after an ordinary lifecycle transition.

The following are not stopping points:

```text
Probe completed

TaskSpec frozen

Worker completed

Verifier passed

one task became DONE

a downstream task became runnable

WORKBOARD was updated
```

After every state transition:

```text
inspect DAG
→ choose next runnable action
→ continue automatically
```

Return control to the user only when:

```text
the whole requested plan is DONE;

a blocker requires information/access only
the user can provide;

a user-level product/architecture choice
cannot be resolved from repository evidence;

an unrecoverable tool/environment failure occurs;

the user explicitly asks to stop.
```

---

# 22. Final Check0 Acceptance Reconciliation

The corrected implementation must satisfy the applicable historical conditions.

```text
[1]
CLI supports:
lowpass / raw / madgwick / xylo

[2]
lowpass resolves:
low-pass/label

[3]
boundary is always label

[4]
no Action0 aligned-board-events branch

[5]
model input only channels 0:15

[6]
batch:
(B, T_pad, 15)

[7]
labels:
(B,)

[8]
valid_mask:
(B, T_pad)

[9]
padding excluded from classification loss

[10]
padding excluded from final spike-count prediction

[11]
padding excluded from spkTotal/statistics

[12]
SynNet topology unchanged

[13]
shift_syn parameter and passing style preserved

[14]
shift_mem parameter and passing style preserved

[15]
defaults:
shift_syn=2
shift_mem=1

[16]
beta remains:
1 - 2**(-shiftMem)

[17]
heterogeneous shifts remain:
range(shiftSyn, shiftSyn+8)

[18]
sample_freq does not automatically change alpha/beta

[19]
Action0 network_type = SynNet

[20]
dataset_variant=xylo
does not imply SynNetRP

[21]
train / val / test users are disjoint

[22]
old-machine path condition resolved
according to T004 frozen scope

[23]
W&B is optional

[24]
Rockpool is not required for local SynNet

[25]
dry_run performs:
data
→ forward
→ loss
→ backward
→ optimizer step

[26]
all required new pytest tests execute and pass
```

---

# 23. New Hardening Acceptance Criteria

In addition to Check0:

```text
[27]
Trainer reads and validates
padding_dataset_summary.json

[28]
Producer input_kind must be spike-imu

[29]
Producer feature schema must match
canonical SpikeIMU schema

[30]
Producer channel_count must be 21

[31]
Trainer input channel count remains 15

[32]
Producer target_length is positive and valid

[33]
train / validation / test T_pad all equal
producer target_length

[34]
CLI sample_freq agrees with
producer sampling_rate_hz

[35]
sample_freq validation does not alter
shift_syn / shift_mem / alpha / beta

[36]
padding_side contract is right-padding

[37]
engine asserts exact output class count

[38]
epoch loss has explicitly verified
valid-timestep weighting semantics

[39]
dry-run inspection and optimizer step
operate on the same batch

[40]
dry-run performs exactly one optimizer step

[41]
checkpoint configuration mismatches
fail before state is loaded

[42]
checkpoint semantics are explicitly documented

[43]
local SynNet can import/run without Rockpool

[44]
core Action0 tests are not silently skipped

[45]
verification runs under Python 3.11

[46]
targeted SNN pytest passes

[47]
full repository pytest passes

[48]
at least one real producer-backed
Action0 dry run passes
```

---

# 24. Definition of Done

The overall plan is DONE only when all of the following are true:

```text
T001 DONE

T002 DONE

T003 DONE

T004 scope resolved

T004B DONE if required

T005 independent verifier PASS

T006 documentation complete

all P0 issues resolved

all in-scope P1 issues resolved

no unresolved blocker remains

Python 3.11 verification gate satisfied

core Action0 tests actually executed

targeted pytest passed

full pytest passed

real producer dry-run passed

ACTION0_SNN_TRAINING.md matches
verified implementation behavior

WORKBOARD records final state
```

The following are insufficient to declare completion:

```text
code was written

Worker returned DONE

synthetic smoke test passed

documentation was edited

tests exist but were not executed

tests were skipped

GitHub contains no failing CI status

one individual task passed
```

---

# 25. Recommended Implementation Order

Use this operational order:

```text
1.
Initialize 02 plan + task DAG + WORKBOARD

2.
Probe T001

3.
Freeze T001

4.
Implement producer/trainer contract

5.
Verify T001

6.
Document T001 and mark DONE

7.
Probe T002

8.
Freeze T002

9.
Implement engine/dry-run correction

10.
Verify T002

11.
Document T002 and mark DONE

12.
Probe T003 checkpoint semantics

13.
Resolve restore/resume definition

14.
Freeze T003

15.
Implement checkpoint hardening

16.
Verify T003

17.
Document T003 and mark DONE

18.
Complete T004 legacy scope probe

19.
If needed:
    freeze + implement + verify T004B

20.
Run environment gate

21.
Run targeted Action0 test matrix

22.
Run full pytest

23.
Run real lowpass dry-run

24.
Run optional tiny-overfit diagnostic

25.
Fresh final verifier

26.
Reconcile ACTION0 note + Check0 plan

27.
Update WORKBOARD

28.
Close / move completed plan according to
repository documentation convention
```

---

# 26. Recommended Commit Boundaries

If implementation is later published through Git, prefer small reviewable commits.

### Commit 1

```text
fix(snn): validate Action0 padded producer contract
```

Contains:

```text
producer metadata loader

schema/channel/target checks

sample frequency provenance check

cross-split T_pad validation

related tests
```

### Commit 2

```text
fix(snn): harden masked engine and dry run
```

Contains:

```text
exact output class validation

valid-timestep epoch loss aggregation

same-batch dry run

dry-run optimizer-step regression
```

### Commit 3

```text
fix(snn): validate Action0 checkpoint contract
```

Contains:

```text
checkpoint schema/metadata validation

frozen restore semantics

checkpoint regression tests
```

### Commit 4, only if required

```text
fix(snn): remove legacy machine-specific paths
```

Contains only legacy portability changes authorized by T004B.

### Documentation commit

```text
docs(snn): reconcile Action0 training contracts
```

Contains only verified durable documentation updates.

---

# 27. Risk Register

## R1 — Accidentally changing SynNet dynamics

Mitigation:

```text
freeze existing alpha/beta regression tests

forbid unnecessary utils_architectures changes

verify sample_freq invariance
```

## R2 — Treating historical plan path as source of truth

Mitigation:

```text
explicitly freeze segmentation_padded contract

reference producer implementation

document historical correction
```

## R3 — Over-scoping checkpoint work

Mitigation:

```text
Probe restore/resume semantics first

do not claim exact resume without RNG/shuffle state
```

## R4 — Legacy cleanup destabilizes current Action0 work

Mitigation:

```text
make T004 conditional and separately frozen
```

## R5 — Tests pass by skipping SNN dependencies

Mitigation:

```text
pytest -rs

record skip count

require torch/snntorch tests to execute
```

## R6 — Environment violates project policy

Mitigation:

```text
Python 3.11 is a pre-verification gate
```

## R7 — Documentation gets ahead of implementation

Mitigation:

```text
durable docs update only after verifier PASS
```

---

# 28. Expected End State

After this plan is complete, the Action0 path should have one clear chain of trust:

```text
Action0 producer
    │
    │ publishes validated metadata +
    │ padded SpikeIMU packages
    ▼
segmentation_padded
    │
    │ contract validation
    ▼
Action0SegmentDataset
    │
    │ channels 0:15 +
    │ labels +
    │ valid mask
    ▼
DataLoader
    │
    ▼
local SynNet
    │
    │ original shift dynamics
    │ all padded time steps advance state
    ▼
masked CE
    │
    ├── padding excluded
    ├── prediction masked
    ├── statistics masked
    └── regularization masked
    │
    ▼
train / val / test
    │
    │ user-disjoint
    ▼
validated checkpoint / metrics
```

with verification evidence showing:

```text
producer metadata agrees with trainer

data geometry agrees across splits

sampling provenance is correct

SynNet dynamics did not change

padding cannot contaminate objective/prediction

dry-run is a real one-batch optimization path

checkpoint incompatibilities cannot silently load

tests execute in the required environment

real producer data reaches a successful training step
```

The desired final result is not a new SNN architecture.

The desired final result is a **contract-complete, reproducible, independently verified version of the existing Action0 SynNet baseline**.
