# Experiment 7.0 — Hierarchical temporal SNN with context-conditioned evidence

## Scientific questions

Exp7.0 separates two questions:

1. Does normalized/unit-DC synaptic current improve long-timescale SNN dynamics relative to the legacy update?
2. Should long context contribute additive class evidence, or modulate the interpretation of middle-scale WHAT evidence?

All trainable SNN conditions use the same Raw64 user-disjoint split and the same short-tau output LIF dynamics used by Exp6.0.

## Data contract

- Raw64 30-channel event spike train
- 64 Hz
- globally padded length 256
- 12 classes
- fixed split seed `12345`
- training seeds `11, 23, 37`
- no pre-SNN pooling or scaling for hierarchical SNN runs

## References

### B0 — Raw + Fixed250 + Linear

```text
Raw64 -> ordered 250-ms channel counts -> flatten
      -> train-only StandardScaler
      -> validation-selected LogisticRegression
```

At 64 Hz, 250 ms is 16 timesteps, so a 256-step sequence yields 16 ordered bins and 480 raw features.

### B1 — Local SNN + Fixed250 + Linear

Reuse the Exp0.1 `local_234x2` binary architecture:

```text
Raw30
-> L1: 128 neurons, shifts {2,3,4}
-> L2: 128 neurons, shifts {2,3,4}
-> ordered Fixed250 L2 spike counts
-> train-only StandardScaler
-> validation-selected LogisticRegression
```

Each local layer uses 43/43/42 neurons for shifts 2/3/4. The SNN is trained with its original temporary timestep classification head; the final deployment probe is fit only after reloading the validation-selected SNN checkpoint.

## Hierarchical SNN backbone

```text
Raw30 -> Short(s4) -> Middle(s5) -> Long(s6)
```

Each hidden layer has 128 binary-spike neurons, hidden cap 1, threshold 0.5, and short membrane dynamics (`tau_mem = 22.54 ms`). The synaptic shifts correspond at 64 Hz to approximately:

```text
s4 = 242 ms
s5 = 492 ms
s6 = 992 ms
```

`what_only` stops after the middle layer; the other methods include the long layer.

## Paired neuron-dynamics ablation

Every hierarchical method is trained in two paired modes.

Legacy:

```text
I_t = alpha * I_{t-1} + W x_t
```

Normalized/unit-DC:

```text
I_t = alpha * I_{t-1} + (1 - alpha) * W x_t
```

For a matched `(method, seed)` pair, split, initialization stream, loader order, optimizer, hidden/output widths, thresholds, membrane dynamics, epoch budget, and checkpoint rule are shared. The synaptic injection equation is the intended difference.

## Methods

### M0 — WHAT only

```text
J_t = W_what z_t^M
```

This measures how much classification performance is available from the short+middle hierarchy without a long context layer.

### A — All-scale skip additive

```text
J_t = W_S z_t^S + W_M z_t^M + W_L z_t^L
```

This is the generic multiscale additive baseline.

### C — Additive WHAT + CONTEXT

```text
e_t^what    = W_what z_t^M
e_t^context = W_ctx  z_t^L
J_t         = e_t^what + e_t^context
```

This tests context as additional class evidence.

### D — Class-specific context gain

```text
e_t^what = W_what z_t^M
g_t       = 1 + tanh(W_g z_t^L)
J_t       = g_t * e_t^what
```

`W_g` is initialized to zero, so `g_t = 1` at initialization. The gain remains in `(0,2)` and can suppress or amplify each class evidence channel without flipping its sign.

## Output and objective

All hierarchical methods use the Exp6.0 short-tau binary output LIF:

- `tau_mem,out = 22.54 ms`
- threshold 0.5
- output cap 1
- no long output synaptic state

Deployment and training objective:

```text
c_k = sum_{t < valid_length} S_out[k,t]
L = CE(c, y)
```

Padding never contributes to WholeCount.

## Training / checkpoint contract

- max epochs: 100
- minimum epoch gate: 20
- patience: 30 epochs without a new best validation BA
- validation BA is primary checkpoint metric
- lower validation loss breaks BA ties
- test is evaluated only after reloading the best checkpoint

Every SNN worker writes:

- checkpoint
- history CSV
- train/validation loss vs epoch
- train/validation BA vs epoch
- best/stopped epoch
- final evaluation JSON

## Final-model activity diagnostics

All runs use the same deterministic validation sample: the first sample whose valid length is closest to the validation median. Input is explicitly zeroed after the valid endpoint.

Each final best checkpoint writes per-layer neuron-vs-time rasters and compressed arrays.

Hierarchical runs save:

```text
short spikes/current/membrane
middle spikes/current/membrane
long spikes/current/membrane   # except WHAT-only
output spikes
class evidence
WHAT evidence                  # where defined
context gain                   # D only
```

Local-SNN baseline runs save L1 and L2 spike rasters. Raw+Fixed250+Linear has no neuron raster.

Per-layer summaries include valid firing fraction, zero-tail firing fraction, and valid spikes/neuron/s.

## Run matrix

```text
4 hierarchical methods
x 2 synapse modes
x 3 seeds
= 24 hierarchical SNN runs

3 local-SNN baseline runs
1 raw Fixed250+Linear reference job
```

Total gradient-training runs: 27.

## Multi-CPU / Slurm strategy

Workers never write shared run artifacts.

```text
1 one-core raw baseline job
3 one-core local-SNN array tasks
24 one-core hierarchical array tasks
1 one-core artifact-only finalizer
```

The baseline, local array, and hierarchical array run independently. The finalizer has an `afterok` dependency on all three jobs.

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_0_cpu.bash
```

Inspect run identities without training:

```bash
python -m scripts.experiment_7_0_hierarchical_context_snn list-runs
python -m scripts.experiment_7_0_hierarchical_context_snn list-local-runs
```

## Aggregation policy

The finalizer is the only process that writes shared comparison files:

```text
runs.csv
summary.csv
paired_neuron_deltas.csv
method_deltas.csv
firing_runs.csv
firing_summary.csv
manifest.json
summary_plots/
```

`runs.csv` and per-run directories remain available for audit/debugging, but the analysis notebook intentionally does not display individual training runs or individual seeds.

## Notebook policy

`notebooks/experiment_7_0_hierarchical_context_snn.ipynb` is analysis-only. It reads finalizer outputs and presents only method-level comparisons:

- mean +/- SD test BA by method and neuron dynamics
- paired normalized-minus-legacy BA effect by method
- architecture contrasts such as context-gain minus additive-context
- aggregated layer firing / zero-tail statistics

The notebook does not train models, refit probes, invoke Slurm, or render per-run learning curves/rasters.
