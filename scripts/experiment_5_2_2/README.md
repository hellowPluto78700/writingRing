# Experiment 5.2.2 — Frozen Local SNN to Multi-`tau_syn` Phase-Aware Decoder

## Question

Given the exact validated local representation used by Experiment 5.1, can one additional feed-forward SNN layer with heterogeneous synaptic time constants reorganize temporally ordered local information into a representation that is accessible by a simple whole-sequence count readout?

The experiment fixes the local extractor and changes only:

1. the temporal basis of the third SNN layer; and
2. whether the final 128-to-12 mapping is read directly from the L3 count vector or is followed by 12 short-memory output LIF neurons.

The intended computation is:

```text
Raw64 weighted events
  -> frozen Exp5.1 local SNN
       L1: 128, shifts (2,3,4)
       L2: 128, shifts (2,3,4)
  -> frozen binary L2 trajectory at 64 Hz
  -> trainable L3: 128 FF LIF neurons with explicit tau_syn state
  -> whole-sequence readout
```

L1/L2 are never updated in an Exp5.2.2 decoder run. Only `W3` and `Wo` are trainable.

## Frozen local source

Exp5.2.2 reuses/reproduces the exact Experiment 5.1 frozen local checkpoints and caches:

```text
source experiment: experiment_5_1_boundary_free_temporal_decoder
source protocol:   frozen_local_leaky_interface_rsnn_v1
source seeds:      11, 23, 101
```

Three one-CPU preparation tasks call the Experiment 5.1 local-cache path, validate/reproduce the same Exp3-exact local checkpoint, and compute Exp5.2.2 local-reference probes. The 60 decoder tasks only load those frozen L2 trajectories.

## L3 dynamics

For frozen local spike vector `z_t`:

```text
I_t = alpha_i * I_(t-1) + W3 z_t
U_t^- = beta * U_(t-1) + I_t
S_t = threshold(U_t^-)
U_t = U_t^- - S_t * theta
```

L3 is feed-forward: there is no recurrent matrix.

Fixed L3 membrane dynamics:

```text
tau_mem = 22.54 ms
threshold = 0.5
reset = subtract
hidden spike cap = 1
```

The threshold is fixed at `theta = 0.5` and is not a sweep variable. Only `alpha_i`, hence `tau_syn`, changes across profiles. At 64 Hz, shifts 2..7 correspond approximately to 54, 117, 242, 492, 992, and 1992 ms.

## Temporal profiles

Single-`tau_syn` controls:

```text
s2, s3, s4, s5, s6, s7
```

Heterogeneous banks:

```text
s234567
s4567
s567
s67
```

Neuron allocation is deterministic and always totals 128:

| Profile | Shifts | Neurons per shift |
|---|---|---|
| `s2` ... `s7` | one shift | 128 |
| `s234567` | 2,3,4,5,6,7 | 22,21,21,21,21,22 |
| `s4567` | 4,5,6,7 | 32,32,32,32 |
| `s567` | 5,6,7 | 43,42,43 |
| `s67` | 6,7 | 64,64 |

This keeps width and trainable parameter count fixed while changing only the temporal basis.

## Two native readouts

### `hidden_count_linear`

L3 spikes are summed over the valid gesture:

```text
C_L3 = sum_t S_t^L3
logits = Wo C_L3
loss = CE(logits, y)
```

`Wo` is `Linear(128, 12, bias=False)`. Therefore the same class mapping is used at every timestep:

```text
Wo sum_t S_t = sum_t Wo S_t
```

No explicit bin/phase index is available to the readout. Strong performance therefore requires L3 dynamics to make temporal context accessible through L3 neuron identity/counts.

### `output_lif`

The same `Wo` first produces a 12-D class drive at every timestep:

```text
q_t = Wo S_t^L3
O_t = OutputLIF(q_t)
C_out = sum_t O_t
loss = CE(C_out, y)
```

The output LIF uses:

```text
tau_mem,out = 22.54 ms
output synaptic state = none
output spike cap = 1
```

The output layer therefore does not add long-term memory. This branch isolates the additional threshold/reset/binary-spike bottleneck after the L3 representation.

## Trainable parameters

Both readouts train exactly:

```text
W3: 128 x 128 = 16,384
Wo: 128 x 12  =  1,536
Total          = 17,920
```

There are no trainable L1/L2, time-constant, threshold, recurrence, regularizer, auxiliary-loss, or multi-H parameters.

`W3` and `Wo` initialization streams and minibatch-order streams are paired across temporal profiles and readouts for each source seed.

## Training matrix

```text
10 temporal profiles
x 2 readouts
x 3 frozen-local seeds
= 60 independent training runs
```

Every run uses normal continuous L3 state during training. Reset interventions are never used for optimization.

Checkpoint selection is strictly per-run validation-only:

1. maximum native validation balanced accuracy;
2. exact tie-break by lower native validation CE.

Test results, probes, intervention results, or threshold diagnostics never select a checkpoint or temporal profile.

## Mandatory inference-time reset interventions

Every selected checkpoint is evaluated under seven logical conditions:

```text
normal
resetall250
reset234
reset2345
reset23456
reset67
reset567
```

At 64 Hz, 250 ms is 16 raw timesteps. For every reset condition, immediately before timesteps 16, 32, 48, ... the selected L3 neuron groups have both states cleared:

```text
I_L3 = 0
U_L3 = 0
```

The output-LIF state is never reset.

Requested resets are intersected with the profile. For example, `reset234` on `s4567` effectively resets shift 4 only. No-op and equivalent effective masks are recorded explicitly; equivalent masks share one actual forward evaluation and are expanded back into all requested logical rows by the artifact writer.

These interventions are causal diagnostics on the trained solution. They measure reliance on continuous cross-250-ms state; they do not estimate the retrained capacity of an architecture with those states removed.

## Mandatory evaluation

For every selected checkpoint and every logical intervention, report native train/validation/test accuracy, balanced accuracy, macro-F1, and CE.

Three fresh post-hoc Linear probes are fitted independently for every intervention using only the corresponding training representation; `C` is selected on validation BA and test is reported only afterward:

```text
l3_whole_count
l3_fixed250_ordered
l3_relative10_ordered
```

The first probe is the primary representation metric. Fixed250 and Relative10 reveal how much information remains available when an external temporal ordering is reintroduced.

The preparation stage also produces frozen L2 references:

```text
local_whole_count
local_fixed250_ordered
local_relative10_ordered
```

For `output_lif`, the trained pre-output-LIF evidence is also evaluated without retraining:

```text
Q = sum_t Wo S_t^L3
```

This gives the diagnostic chain:

```text
fresh Linear on L3 count
  -> trained Wo before output LIF
  -> native output-LIF spike count
```

## Fixed-threshold operating-point diagnostics

`theta = 0.5` remains fixed and is not swept. After each selected checkpoint is evaluated, the same task computes the L3 operating point overall and separately for every `tau_syn`/shift group under every reset condition.

For each group, record:

```text
FR_s                                  = firing_rate_hz
P(S_t = 1)                            = spike_probability
E|I_t|                                = mean_abs_input_current
E|U_t^-|                              = mean_abs_pre_reset_membrane
P(U_t^- > theta)                      = pre_reset_above_threshold_probability
```

Here `U_t^-` is the exact membrane value before thresholding and subtractive reset:

```text
U_t^- = beta * U_(t-1) + I_t
```

`MacroMultiSpikeLIF` returns this quantity before reset, and the diagnostic also has the exact identity `U_t^- = U_t + S_t * theta` for the binary subtractive-reset L3 used here.

Because hidden spike cap is one, `P(S_t = 1)` numerically matches the existing binary `nonzero_fraction`, while `FR_s = fs * P(S_t = 1)`. Both names are retained deliberately because they answer different analysis questions.

The key groups are `s6` and `s7` (approximately 1 s and 2 s synaptic traces). Their firing, accumulated input-current magnitude, pre-reset membrane magnitude, and threshold-crossing probability are inspected under `normal`, `resetall250`, `reset67`, and `reset567` to determine whether the long paths are actually operating near a useful firing regime rather than being dead or saturated.

These diagnostics are written to `threshold_activity.csv`; they are evaluation-only and do not affect training or checkpoint selection.

## Activity diagnostics

For normal and every reset intervention, activity is reported overall and separately for every shift group present in the profile:

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

Long groups `s6` and `s7` are especially important for determining whether the nominal 1-2 s synaptic paths are actually active and whether 250-ms resets disrupt their accumulated synaptic state.

## Key derived notebook diagnostics

The analysis notebook computes paired reset effects:

```text
Delta(metric, reset) = metric(normal) - metric(reset)
```

and the temporal-organization gap:

```text
organization_gap = Delta BA(l3_whole_count) - Delta BA(l3_fixed250_ordered)
```

A positive organization gap means the intervention damages count-accessible temporal organization more strongly than it damages information that can still be recovered with explicit Fixed250 ordering.

The notebook also compares cumulative fast/mid resets (`234`, `2345`, `23456`) with long-state resets (`67`, `567`), separates L3 representation quality, trained-`Wo` utilization, and output-LIF compression, and reports the fixed-threshold operating-point metrics with special attention to shifts 6 and 7.

## Single-segment dynamics diagnostic

A checkpoint-only single-segment dynamics diagnostic is provided by:

```text
scripts/experiment_5_2_2_single_segment_dynamics.py
```

It never retrains the model, never updates `W3`/`Wo`, and never extends the segment beyond its original valid endpoint. It loads one already-selected Exp5.2.2 checkpoint, runs the same test segment under `normal` and one reset intervention, and records the exact timestep-level L3 dynamics.

The default diagnostic targets the most informative current comparison:

```text
profile:       s234567
readout:       hidden_count_linear
seed:          11
split:         test
intervention:  reset567
selection:     auto
```

`--selection auto` searches the requested split with this priority:

```text
normal wrong -> reset correct
prediction changed
both wrong
both correct
first sample
```

An exact segment can instead be selected with `--sample-index N`.

Run from the repository root after the Exp5.2.2 checkpoints and frozen Exp5.1 L2 cache exist:

```bash
python -u -m scripts.experiment_5_2_2_single_segment_dynamics \
  --profile s234567 \
  --readout hidden_count_linear \
  --seed 11 \
  --split test \
  --intervention reset567 \
  --selection auto \
  --device cpu \
  --threads 1
```

For a known sample:

```bash
python -u -m scripts.experiment_5_2_2_single_segment_dynamics \
  --profile s234567 \
  --readout hidden_count_linear \
  --seed 11 \
  --split test \
  --intervention reset567 \
  --sample-index 0
```

For `output_lif`, change only `--readout output_lif`; the readout figure then switches from cumulative shared-Linear class evidence to native output-neuron spike raster, output membrane traces, and cumulative output spike counts.

Artifacts are written under:

```text
notebooks/artifacts/experiment_5_2_2_frozen_local_multitau_syn/
  frozen_exp51_l2_multitau_syn_wholecount_v1/
    single_segment_dynamics/
      <profile>__<readout>__seed<seed>/
        <split>_sampleXXXX__normal_vs_<intervention>/
```

Each diagnostic directory contains:

```text
01_l3_spike_raster.png
02_l3_synaptic_state_heatmap.png
03_l3_pre_reset_membrane_heatmap.png
04_l3_post_reset_membrane_heatmap.png
05_l3_group_firing.png
06_readout_activity.png
trace_normal.npz
trace_<intervention>.npz
neuron_metrics.csv
group_metrics.csv
summary.json
```

The L3 raster uses one row per neuron with black points for emitted spikes. Neurons are ordered by the profile's fixed shift-group allocation, so the `s234567` plot is visually partitioned into `s2|s3|s4|s5|s6|s7`.

The state heatmaps retain the signed synaptic current and membrane values; they do not replace the states with absolute magnitudes. This is required to distinguish large positive backlog from large negative state.

`neuron_metrics.csv` and `group_metrics.csv` summarize firing-run structure and backlog-related diagnostics, including:

```text
first_spike_timestep
mean_run_length
max_run_length
fraction_spikes_in_runs_ge4
fraction_spikes_in_runs_ge8
post_reset_above_threshold_given_spike
endpoint_synaptic
endpoint_post_reset_membrane
```

These are descriptive mechanism diagnostics. BA remains the main dataset-level Exp5.2.2 evaluation and is not recomputed as the primary output of this single-segment visualization.

## Multi-CPU execution

Exp5.2.2 follows `AGENTS.md` task-level multi-CPU rules:

```text
3 frozen-local preparation tasks
  -> Slurm array 0-2%3
  -> one CPU core per task

prepare-local afterok
  -> 60 independent decoder tasks
  -> Slurm array 0-59%50
  -> one CPU core per task
  -> train -> select checkpoint -> evaluate all deduplicated reset masks
  -> fit intervention-specific probes
  -> evaluate fixed-threshold operating point per tau_syn group
  -> write per-run artifacts

all 60 decoder tasks succeed
  -> afterok finalizer
  -> concatenate existing artifacts only
  -> merge threshold diagnostics with activity rows
  -> analysis-only notebook
```

The reset/probe/threshold-diagnostic functions are separate from training, so evaluation can be rerun without retraining. A rerun of an existing run skips the checkpoint unless `--force` is supplied.

Every Slurm compute task initializes Conda locally and uses one CPU thread. Array concurrency never exceeds 50.

Submit from the repository root:

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

The finalizer fails on missing run, history, local-reference, or threshold-diagnostic artifacts. It never retrains, reruns interventions, selects a profile, or computes the scientific conclusion.

## Notebook aggregation policy

`notebooks/experiment_5_2_2_frozen_local_multitau_syn.ipynb` is analysis-only. It must:

1. read only finalized CSV/JSON outputs;
2. compute mean +/- SD over the three paired source seeds;
3. select one temporal profile per readout using mean native validation BA only;
4. report selected test BA only after validation selection;
5. plot native BA across all single/multi-`tau_syn` profiles;
6. compare frozen L2 and selected L3 WholeCount/Fixed250/Relative10 accessibility;
7. compute paired normal-minus-reset effects for native and probe metrics;
8. compute the WholeCount-vs-Fixed250 organization-gap diagnostic;
9. compare fresh L3-count probe, trained pre-LIF `Wo`, and native output-LIF performance;
10. inspect per-shift firing/state statistics, especially shifts 6/7;
11. report `FR_s`, `P(S_t=1)`, `E|I_t|`, `E|U_t^-|`, and `P(U_t^- > theta)` by shift, emphasizing s6/s7;
12. plot validation learning curves for validation-selected configurations;
13. never train, call Slurm, regenerate artifacts, or select on test BA.

## Required checks

```bash
python -m pytest -q tests/test_experiment_5_2_2_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
