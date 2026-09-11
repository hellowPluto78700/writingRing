# Experiment 5.2.2 — Frozen Local SNN to Multi-`tau_syn` Phase-Aware Decoder

## Question

Given the exact validated local representation used by Experiment 5.1, can one additional feed-forward SNN layer with heterogeneous synaptic time constants reorganize temporally ordered local information into a representation that is accessible by a simple whole-sequence count readout?

The main path is:

```text
Raw64 weighted events
  -> frozen Experiment 5.1 local SNN
       L1: 128, shifts (2,3,4)
       L2: 128, shifts (2,3,4)
  -> frozen binary L2 trajectory at 64 Hz
  -> trainable L3: 128 FF Synaptic-LIF neurons
  -> whole-sequence readout
```

L1/L2 are frozen. Only `W3` and `Wo` are trainable.

## Frozen local source

```text
source experiment: experiment_5_1_boundary_free_temporal_decoder
source protocol:   frozen_local_leaky_interface_rsnn_v1
source seeds:      11, 23, 101
```

Exp5.2.2 reuses the exact Experiment 5.1 local checkpoints/caches.

## L3 dynamics

For frozen local spike vector `z_t`:

```text
I_t = alpha_i * I_(t-1) + W3 z_t
U_t^- = beta * U_(t-1) + I_t
S_t = threshold(U_t^-)
U_t = U_t^- - S_t * theta
```

L3 is feed-forward; there is no recurrent matrix.

Fixed dynamics:

```text
tau_mem = 22.54 ms
threshold = 0.5
reset = subtract
hidden spike cap = 1
```

Only `tau_syn` changes across the temporal profiles. At 64 Hz, shifts 2..7 correspond approximately to 54, 117, 242, 492, 992, and 1992 ms.

## Temporal profiles

Single-timescale controls:

```text
s2, s3, s4, s5, s6, s7
```

Multi-timescale banks:

```text
s234567
s4567
s567
s67
```

Neuron allocation is fixed at 128 neurons total:

| Profile | Shifts | Neurons per shift |
|---|---|---|
| `s2` ... `s7` | one shift | 128 |
| `s234567` | 2,3,4,5,6,7 | 22,21,21,21,21,22 |
| `s4567` | 4,5,6,7 | 32,32,32,32 |
| `s567` | 5,6,7 | 43,42,43 |
| `s67` | 6,7 | 64,64 |

## Readouts

### `hidden_count_linear`

```text
C_L3 = valid sum_t S_t^L3
logits = Wo C_L3
loss = CE(logits, y)
```

`Wo` is `Linear(128, 12, bias=False)`, so:

```text
Wo sum_t S_t = sum_t Wo S_t
```

No explicit phase/bin identity is available to this head.

### `output_lif`

```text
q_t = Wo S_t^L3
O_t = OutputLIF(q_t)
C_out = valid sum_t O_t
loss = CE(C_out, y)
```

The output layer uses:

```text
tau_mem,out = 22.54 ms
output synaptic state = none
output spike cap = 1
```

The output-LIF state is never reset by the L3 reset interventions.

## Trainable parameters

```text
W3: 128 x 128 = 16,384
Wo: 128 x 12  =  1,536
Total          = 17,920
```

No L1/L2 weights, time constants, threshold, recurrence, regularizer, auxiliary loss, or event cap are trainable/swept here.

## Training matrix

```text
10 temporal profiles
x 2 readouts
x 3 frozen-local seeds
= 60 independent training runs
```

Every training run uses continuous L3 state. Reset interventions are inference-only.

Checkpoint selection is validation-only:

1. maximum native validation balanced accuracy;
2. tie-break by lower native validation CE.

Test results, probes, and reset results never select a checkpoint.

## Inference-time reset interventions

Every selected checkpoint is evaluated under:

```text
normal
resetall250
reset234
reset2345
reset23456
reset67
reset567
```

At 64 Hz, 250 ms is 16 timesteps. Immediately before timesteps 16, 32, 48, ... selected L3 groups are cleared:

```text
I_L3 = 0
U_L3 = 0
```

Requested resets are intersected with the active profile. For example, `reset234` on `s4567` resets only shift 4.

The intervention measures reliance on continuous cross-250-ms L3 state. It does not estimate the performance of a separately retrained architecture.

## Main evaluation

For every selected checkpoint/intervention, report native train/validation/test:

```text
accuracy
balanced accuracy
macro-F1
cross-entropy
```

Fresh post-hoc Linear probes are fitted independently for each intervention:

```text
l3_whole_count
l3_fixed250_ordered
l3_relative10_ordered
```

The matching frozen-L2 references are:

```text
local_whole_count
local_fixed250_ordered
local_relative10_ordered
```

For `output_lif`, also evaluate the trained pre-output-LIF evidence:

```text
Q = valid sum_t Wo S_t^L3
```

This separates:

```text
fresh Linear on L3 count
  -> trained Wo before output LIF
  -> native output-LIF spike count
```

## Fixed-threshold operating-point diagnostics

`theta = 0.5` is fixed and is not a sweep variable.

Per shift group, record:

```text
FR_s                                  = firing_rate_hz
P(S_t = 1)                            = spike_probability
E|I_t|                                = mean_abs_input_current
E|U_t^-|                              = mean_abs_pre_reset_membrane
P(U_t^- > theta)                      = pre_reset_above_threshold_probability
```

The especially important long groups are `s6` and `s7`.

These diagnostics are written to:

```text
threshold_activity.csv
```

Other activity outputs include:

```text
event_rate_hz
nonzero_fraction
active_neuron_fraction
events_per_neuron_gesture
mean_abs_syn
rms_syn
mean_abs_mem
rms_mem
```

## Dataset-level analysis notebook

`notebooks/experiment_5_2_2_frozen_local_multitau_syn.ipynb` is analysis-only. It reads finalized artifacts and performs profile comparison, reset effects, representation probes, output bottleneck analysis, and per-shift operating-point analysis. It never trains models or launches Slurm jobs.

## Single-segment dynamics notebook

The mechanism-level visualization is now an interactive notebook:

```text
notebooks/experiment_5_2_2_single_segment_dynamics.ipynb
```

This notebook replaces the earlier standalone single-segment diagnostic script.

It is checkpoint-only:

- loads one existing Exp5.2.2 checkpoint;
- loads the matching frozen Experiment 5.1 L2 cache;
- never calls backward/optimizer;
- runs one selected segment under `normal` and one reset intervention;
- defaults to `s234567 + hidden_count_linear + seed11`, `normal` vs `reset567`;
- prefers a `normal_wrong_reset_correct` test segment when available.

### Full padded-window rule

The notebook traces and plots the **full padded window**, not only the valid gesture interval.

The original `valid_length` is always preserved separately. Every time-axis figure:

1. draws a red dashed **Valid end** line at `valid_length`;
2. shades the region after `valid_length` as padding/post-valid dynamics;
3. keeps the native prediction/readout restricted to the original valid interval.

Therefore post-valid activity is visualization/diagnostic evidence only. It is not retroactively added to the training loss or the main Exp5.2.2 BA.

### Figures

The notebook produces inline:

```text
01 L3 spike raster
02 L3 signed synaptic-state I heatmap
03 L3 pre-reset membrane U^- heatmap
04 L3 post-reset membrane U heatmap
05 L3 shift-group firing fraction
06 readout activity
```

For `hidden_count_linear`, Figure 06 is the cumulative shared-Linear class evidence:

```text
Wo * cumulative_sum_t(S_t^L3)
```

The curve is shown over the full padded window, but the Valid end line marks where the real classifier stops accumulating evidence.

For `output_lif`, Figure 06 instead shows:

```text
12-neuron output spike raster
output post-reset membrane traces
cumulative output spike counts
```

again with the same Valid end marker.

### Compact mechanism table

The notebook also summarizes valid-vs-padding behavior per shift group:

```text
valid_firing_rate_hz
padding_firing_rate_hz
padding_spike_fraction_of_full_window
max_run_length_valid
max_run_length_full
post_reset_U_above_threshold_given_valid_spike
endpoint_mean_abs_I
endpoint_mean_abs_U
```

These values help distinguish sparse firing from sustained firing/latching and show whether substantial state/activity remains after the valid endpoint.

To inspect a different checkpoint or sample, edit the configuration cell at the top of the notebook:

```python
PROFILE = "s234567"
READOUT = "hidden_count_linear"
SEED = 11
SPLIT = "test"
INTERVENTION = "reset567"
SELECTION = "normal_wrong_reset_correct"
SAMPLE_INDEX = None
```

Set `SAMPLE_INDEX` to an integer to force one specific segment.

## Multi-CPU execution

Exp5.2.2 follows the repository multi-CPU contract:

```text
3 frozen-local preparation tasks
  -> Slurm array 0-2%3

prepare-local afterok
  -> 60 decoder tasks
  -> Slurm array 0-59%50
  -> one CPU core per task
  -> train -> select -> evaluate

all decoder tasks succeed
  -> afterok finalizer
  -> aggregate existing artifacts only
```

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_2_2_cpu.bash
```

## Finalized artifacts

```text
notebooks/artifacts/experiment_5_2_2_frozen_local_multitau_syn/
  frozen_exp51_l2_multitau_syn_wholecount_v1/
    checkpoints/
    evaluations/
    threshold_diagnostics/
    histories/
    local_references/
    runs.csv
    ablation_runs.csv
    probes.csv
    activity.csv
    threshold_activity.csv
    histories.csv
    local_reference.csv
    manifest.json
```

The finalizer aggregates existing artifacts only; it does not retrain or rerun missing experiments.

## Required checks

```bash
python -m pytest -q tests/test_experiment_5_2_2_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
