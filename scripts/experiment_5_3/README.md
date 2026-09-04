# Experiment 5.3 - Synaptic history contextualization of frozen local evidence

## Scientific question

Experiment 5.3 asks whether temporal SNN dynamics can transform the validated frozen Exp3 local spike trajectory into **history-conditioned local evidence** without exposing Fixed250 or Relative10 position at inference.

```text
Raw64
  -> frozen Exp3-exact local SNN
  -> raw 128-D L2 spike trajectory at 64 Hz
  -> temporal contextualizer
  -> signed class evidence e_t
  -> non-leaky analog accumulator A_T = sum_valid(e_t)
  -> final accumulated CE
```

The experiment separates three questions:

1. Does a longer single synaptic time constant make history accessible after orderless pooling?
2. Does fixed multi-tau synaptic heterogeneity outperform the best single time constant at the same trainable parameter count?
3. Does preserving the local WHAT path and using an RSNN only as a context gate outperform rewriting the complete local representation?

## Frozen local source

All conditions consume the exact Exp5.2 frozen-local cache:

```text
Raw64 30-channel weighted events
  -> L1: 128 Synaptic neurons, shifts (2, 3, 4)
  -> L2: 128 Synaptic neurons, shifts (2, 3, 4)
  -> cached binary L2 trajectory, no pooling
```

The cache identity remains:

```text
experiment_5_2_frozen_local_tauR_sweep
frozen_exp3_l2_endpoint_tauR_v1
```

Seeds are `(11, 23, 37, 53, 71)`. Within a seed, every condition receives the same sample identities, labels, valid lengths, frozen local cache, data-loader order, evidence-head initialization, and input-projection initialization whenever that module exists.

## Conditions

There are 11 conditions.

| Condition | Context mechanism | Fixed dynamics |
|---|---|---|
| `direct` | no contextualizer; local spike goes directly to evidence head | none |
| `lif_beta100` | one short LIF layer | beta = 1.00 |
| `lif_beta050` | one short LIF layer | beta = 0.50 |
| `syn_single_s4` | one Synaptic-LIF layer | shift_syn = 4 |
| `syn_single_s5` | one Synaptic-LIF layer | shift_syn = 5 |
| `syn_single_s6` | one Synaptic-LIF layer | shift_syn = 6 |
| `syn_multi_s23456` | one heterogeneous Synaptic-LIF layer | shifts = (2, 3, 4, 5, 6) |
| `syn_multi_s456` | one heterogeneous Synaptic-LIF layer | shifts = (4, 5, 6) |
| `syn_multi_s56` | one heterogeneous Synaptic-LIF layer | shifts = (5, 6) |
| `rsnn_beta050` | short membrane plus dense previous-spike recurrence | beta = 0.50 |
| `factorized_rsnn_beta050` | direct WHAT path gated by a separate recurrent context state | beta = 0.50 |

At 64 Hz, repository-style synaptic shifts correspond approximately to:

| shift | decay | tau_syn |
|---:|---:|---:|
| 2 | 0.750000 | 54 ms |
| 3 | 0.875000 | 117 ms |
| 4 | 0.937500 | 242 ms |
| 5 | 0.968750 | 492 ms |
| 6 | 0.984375 | 992 ms |

The single-tau sweep is therefore `(4, 5, 6)`. The multi-tau comparison is `(2,3,4,5,6)` versus `(4,5,6)` versus `(5,6)`.

The two pure-LIF controls differ only in beta:

```text
lif_beta100: tau_mem = inf (no passive membrane leak)
lif_beta050: tau_mem about 22.5 ms at 64 Hz
```

All Synaptic-LIF conditions fix beta at `0.50`, so long history is isolated to the synaptic state. Tau, beta, threshold, and reset are fixed. Threshold is `0.5`, reset is subtractive, and width is 128.

## Contextualization equations

### Direct control

```math
e_t = W_o z_t,
\qquad
A_T = W_o \sum_t z_t.
```

This is the end-to-end counterpart of Local WholeCount plus Linear.

### LIF control

```math
u_t = \beta u_{t-1} + W_{in}z_t,
\qquad
s_t = H(u_t-\theta),
\qquad
e_t = W_os_t.
```

### Single- and multi-tau Synaptic-LIF

```math
i_{j,t} = \alpha_j i_{j,t-1} + [W_{in}z_t]_j,
```

```math
u_{j,t} = \beta u_{j,t-1} + i_{j,t},
\qquad
s_{j,t} = H(u_{j,t}-\theta),
\qquad
e_t = W_os_t.
```

Single-tau conditions assign one alpha to all 128 neurons. Multi-tau conditions assign requested shifts deterministically and as evenly as possible across 128 neurons. Single- and multi-tau models therefore have identical trainable parameter counts.

### RSNN rewriting control

```math
u_t = \beta u_{t-1} + W_{in}z_t + W_{rec}s_{t-1},
\qquad
e_t = W_os_t.
```

### Factorized RSNN

The recurrent branch estimates causal context:

```math
h_t = RSNN(W_{in}z_t,h_{t-1}).
```

The local spike path is retained directly:

```math
g_t = 2\sigma(W_gh_t),
\qquad
\tilde z_t = g_t\odot z_t,
\qquad
e_t = W_o\tilde z_t.
```

The factor of two initializes the gate around one rather than one half. This condition tests whether temporal context should modulate WHAT instead of forcing WHAT through a rewritten recurrent representation.

## Evidence accumulator and objective

Every condition uses the same signed, non-leaky, non-spiking class-evidence accumulator:

```math
A_i = \sum_{t<T_i} e_{i,t}.
```

The evidence head has `bias=False`, so duration cannot enter through a repeated per-timestep class bias.

Training uses only final accumulated supervision:

```math
\bar A_i = 5\frac{A_i}{T_i},
\qquad
L=CE(\bar A_i,y_i).
```

There is no timestep CE, Fixed250 CE, Relative10 CE, or auxiliary objective. Valid-length normalization stabilizes the CE scale across variable-duration gestures and does not change the per-sample argmax.

Checkpoint selection is maximum native validation accumulator balanced accuracy, tie-broken by validation accumulated CE. Test metrics never select a checkpoint or tau configuration.

## Primary and mechanistic measurements

### Native deployment metric

The primary result is native test balanced accuracy from the signed accumulator.

### Existing Linear probes

The Exp5.2 train-only `StandardScaler -> balanced LogisticRegression` protocol is reused for:

- `hidden_whole_count`
- `hidden_fixed250_ordered`
- `hidden_relative10_ordered`
- `hidden_uend`

For rewriting models, `hidden_whole_count` is the contextual spike count. For `direct`, it is the local spike count. For the factorized model, it is the orderless sum of gated local features.

The central representation diagnostic is:

```math
G_{context}=BA_{HiddenWholeCount}-BA_{LocalWholeCount}.
```

A positive gain means class-relevant history became accessible after orderless aggregation. Relative10 remains a trajectory-information reference; it does not by itself prove internal phase encoding.

The notebook also computes:

```math
G^{local}_{timing}=BA_{LocalRelative10}-BA_{LocalWholeCount},
```

```math
G^{hidden}_{timing}=BA_{HiddenRelative10}-BA_{HiddenWholeCount},
```

and the BA-based temporal-accessibility recovery ratio:

```math
R_{internal}=\frac{BA_{HiddenWholeCount}-BA_{LocalWholeCount}}
{BA_{LocalRelative10}-BA_{LocalWholeCount}}.
```

This ratio is not interpreted as mutual information.

### Phase probes

Every gesture contributes exactly ten examples by averaging each internal trajectory inside ten offline relative-progress bins. Linear probes classify bin index from:

- frozen local spikes;
- contextualized representation;
- synaptic current;
- membrane state;
- context spikes;
- factorized gate.

Relative phase labels are post-hoc diagnostics only; they are never SNN inputs or training targets.

### Causal ablations

Each selected checkpoint is evaluated under:

1. `state_reset`: synaptic, membrane, and recurrent spike state are reset before every timestep;
2. `temporal_shuffle`: only the valid local trajectory is permuted before the contextualizer, using five deterministic replicates.

For each ablation, Exp5.3 records native accumulator performance and a newly fitted Hidden WholeCount Linear probe. A genuine history-writing effect should improve ordered Hidden WholeCount and lose that advantage under state reset and temporal shuffle.

### Activity and tail diagnostics

Runs record valid context events, two-second zero-input tail activity, accumulator norm, and mean absolute class evidence per valid step. This is especially important for shift 6, whose synaptic time constant is close to one second.

## Planned comparisons

The analysis notebook reports paired effects by master seed for:

```text
lif_beta100 - lif_beta050
single_s5 - single_s4
single_s6 - single_s5
multi_s56 - single_s5
multi_s56 - single_s6
multi_s456 - multi_s56
multi_s23456 - multi_s456
best_multi - best_single
best_multi - rsnn_beta050
factorized_rsnn_beta050 - best_rewriting_condition
```

Best single-tau and multi-tau configurations are selected using mean validation native accumulator BA only. Their test results are exposed after selection.

## Multi-CPU execution

The run matrix is:

```text
11 conditions x 5 seeds = 55 independent runs
```

The repository task-level CPU strategy is used:

```text
5 frozen-local verification/phase-probe tasks
  -> Slurm array 0-4%5

local preparation afterok
  -> 55 independent train/evaluate tasks
  -> Slurm array 0-54%50
  -> one CPU core per task
  -> train -> validation-select -> evaluate -> probes -> ablations -> artifacts

all 55 tasks succeed
  -> one afterok finalizer
  -> concatenate completed artifacts only
  -> analysis-only notebook
```

Submit from repository root:

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_5_3_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_5_3_cpu.bash
```

## Finalized artifacts

```text
notebooks/artifacts/
  experiment_5_3_synaptic_contextual_evidence/
    frozen_local_synaptic_history_context_v1/
      runs.csv
      histories.csv
      probe_runs.csv
      phase_probe_runs.csv
      ablation_runs.csv
      local_reference.csv
      manifest.json
```

`notebooks/experiment_5_3_synaptic_contextual_evidence.ipynb` is analysis-only. It validates the manifest, computes validation-selected summaries and paired effects, and creates the requested figures. It never trains, launches Slurm, regenerates missing runs, or selects a condition using test data.
