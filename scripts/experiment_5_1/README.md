# Experiment 5.1 — Boundary-free temporal decoding of frozen local SNN features

## Question

Experiment 5.0.1 established that the two-layer multi-`tau_syn` SNN produces a strong, linearly accessible local representation. The remaining question is whether a recurrent SNN can decode longer temporal context from that representation without relying on explicit Fixed250/Relative10 preprocessing.

Exp5.1 therefore freezes the validated local extractor and tests whether a **boundary-free leaky interface** makes the recurrent decoding problem easier than feeding raw L2 spikes directly.

## Frozen local extractor

The source is the exact Exp5.0.1 / historical Exp3 timestep-CE local model:

```text
Raw64 30-channel weighted events
  -> L1: 128 Synaptic neurons, shifts (2,3,4)
  -> L2: 128 Synaptic neurons, shifts (2,3,4)
```

The local model is frozen. Only seeds `(11, 23, 101)` are used because those exact local checkpoints were validated in Exp5.0.1.

On a fresh clone, three one-CPU preparation tasks reproduce/reuse the Exp5.0.1 exact checkpoints and cache L2 binary spike trajectories once per seed. All decoder conditions for that seed then consume exactly the same cached local representation.

## Interface ablation

Two interfaces are compared.

### `raw_l2`

```text
S_t^L2 -> temporal decoder
```

No additional temporal aggregation is applied.

### `leaky242`

Each of the 128 L2 neurons has its own independent, non-spiking leaky accumulator:

```text
A_t = alpha * A_(t-1) + S_t^L2
alpha = 1 - 2^-4 = 15/16
```

There is:

- no cross-neuron mixing;
- no trainable weight;
- no threshold;
- no reset;
- no temporal downsampling.

At 64 Hz this alpha corresponds to approximately 242 ms exponential time constant. Its DC mass is

```text
1 / (1 - alpha) = 16 samples
```

which directly matches the 16 raw samples in a 250 ms Fixed250 bin. It is therefore a causal, boundary-free analogue of channel-wise/motif-wise Fixed250 accumulation rather than another learned feature layer.

## Temporal decoder

The decoder receives either raw L2 spikes or the leaky interface at the original 64 Hz rate.

```text
128 -> Temporal128 -> Linear(128,12,bias=True)
```

Architectures:

- `ff`: no learned recurrent matrix; temporal state comes only from LIF membrane decay;
- `rsnn`: adds `128 x 128` hidden-to-hidden recurrence from the previous hidden spike.

Temporal membrane constants:

```text
125, 250, 500, 1000 ms
```

The temporal layer remains binary-spiking. There is deliberately **no output LIF** in Exp5.1.

## Training objective and readout

At every raw timestep the temporal hidden spike vector produces analog class evidence:

```text
e_t = W s_t^R + b
```

The classifier does **not** use timestep CE. Instead, valid class evidence is averaged over the whole gesture:

```text
E = sum_t m_t e_t / sum_t m_t
loss = CE(E, y)
```

where `m_t` is the valid-length mask.

This objective permits early timesteps to remain ambiguous and lets temporal state change the evidence emitted later. It also avoids reintroducing the 12-neuron spiking-output bottleneck identified in Exp5.0.1.

Checkpoint selection is:

1. maximum validation native sequence balanced accuracy;
2. tie-break minimum validation CE.

## Run matrix

```text
2 interfaces
x 2 architectures
x 4 temporal tau_mem values
x 3 frozen-local seeds
= 48 independent decoder runs
```

Initialization and minibatch order are paired across all decoder conditions for a given seed. `input_hidden` and analog-head parameters use condition-independent random streams; the recurrent matrix has its own paired stream only when present.

## Linear probes

All probes are post-hoc diagnostics and never participate in decoder training or checkpoint selection. They use the repository Exp3-style train-only scaler / LogisticRegression C-grid protocol.

### Interface probes

Computed once per `(interface, seed)` before any temporal decoder interpretation:

- `interface_valid_sum`
- `interface_fixed250_ordered`
- `interface_relative10_ordered`
- `interface_endpoint`

These answer whether leaky242 preserves the strong phase-aware information already present in L2 and whether it creates a more informative instantaneous state.

### Temporal hidden probes

Computed after each selected FF/RSNN checkpoint:

- `hidden_whole_count`
- `hidden_fixed250_ordered`
- `hidden_relative10_ordered`
- `hidden_uend`

The main long-memory diagnostic is `hidden_uend`. If `hidden_uend` approaches the phase-aware trajectory probes, recurrent state has consolidated information that previously required explicit temporal access.

## Multi-CPU execution

The experiment follows `AGENTS.md`.

```text
3 local-cache tasks
  -> Slurm array 0-2%3

local-cache afterok
  -> 48 decoder tasks, array 0-47%48
  -> 6 interface-probe tasks, array 0-5%6

48 decoders + 6 probes succeed
  -> afterok finalizer
```

Every task uses one CPU core and initializes Conda locally. Decoder tasks perform:

```text
load frozen L2 -> train temporal decoder -> select checkpoint -> evaluate -> temporal probes -> per-run artifacts
```

The finalizer only concatenates existing artifacts. It does **not** compute experiment conclusions, choose the best tau, or regenerate missing tasks.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_1_cpu.bash
```

## Notebook aggregation and visualization policy

`notebooks/experiment_5_1_boundary_free_temporal_decoder.ipynb` is analysis-only.

It must:

1. compute mean ± SD from finalized per-seed rows;
2. select temporal `tau_mem` by **mean validation native BA only** before reporting selected test results;
3. show native test BA versus temporal tau for raw-L2 and leaky242, separately for FF and RSNN;
4. compute paired `leaky242 - raw_l2` effects by seed for every architecture/tau;
5. visualize interface probes to test whether leaky242 preserves Fixed250/Relative10 accessibility;
6. compare L2 reference, interface representation, and temporal hidden probes for validation-selected configurations;
7. compare `hidden_uend` with hidden WholeCount and Relative10 to diagnose state consolidation;
8. plot validation learning curves for the validation-selected best configuration.

The notebook must never select a configuration on test BA.

## Finalized artifacts

```text
notebooks/artifacts/experiment_5_1_boundary_free_temporal_decoder/
  frozen_local_leaky_interface_rsnn_v1/
    runs.csv
    interface_probes.csv
    histories.csv
    local_reference.csv
    manifest.json
```

## Required checks

```bash
python -m pytest -q tests/test_repository_source_syntax.py tests/test_experiment_5_1_contract.py
```
