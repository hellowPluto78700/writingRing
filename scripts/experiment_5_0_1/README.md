# Experiment 5.0.1 — Analog-Head Control

## Question

Experiment 3 learned strong local representations with a two-layer multi-`tau_syn` backbone and `timestep_ce`, while Experiment 5.0 showed a large degradation when timestep supervision was routed through a 12-neuron spiking output layer.

A single Exp3 rerun only proves reproducibility. A two-condition comparison can still be confounded by using different random/data streams. Exp5.0.1 therefore uses **three** analog-head controls so the three possible causes are separated cleanly.

## Conditions

### A. `exp3_exact_analog`

Historical reproduction gate:

```text
Exp3 snnTorch.Synaptic hidden dynamics
+ Exp3 initialization/data-loader streams
+ Linear(128,12,bias=True)
+ timestep CE
```

This should reproduce historical Exp3.0.3-B / Exp3.0.5.

### B. `exp3_synaptic_exp5stream_analog`

Paired hidden-dynamics control:

```text
Exp3 snnTorch.Synaptic hidden dynamics
+ Exp5.0 paired initialization/data-loader streams
+ Linear(128,12,bias=True)
+ timestep CE
```

### C. `exp5_macro_exp5stream_analog`

Paired Exp5-hidden control:

```text
Exp5.0 binary MacroMultiSpike hidden dynamics through L2
+ Exp5.0 paired initialization/data-loader streams
+ Linear(128,12,bias=True)
+ timestep CE
```

The experiment intentionally has **no output LIF** in any new condition.

This gives three interpretation gates:

```text
A vs historical Exp3
    -> reproducibility / protocol-drift check

B vs C
    -> hidden implementation / surrogate-dynamics effect
       (same initialization, same data order, same analog head)

C vs existing Exp5 timestep_ce + binary
    -> spiking-output-path effect
       (same Exp5 hidden dynamics and paired Exp5 random/data stream)
```

## Shared architecture

All new conditions use:

```text
Raw64 30-channel unsigned weighted events
  -> L1: 128 neurons, shifts (2,3,4)
  -> L2: 128 neurons, shifts (2,3,4)
  -> Linear(128,12,bias=True)
```

Shared constants:

- `tau_mem = 22.54 ms`
- threshold `0.5`
- subtract reset
- surrogate slope `25`
- Adam, `lr=1e-3`
- 100 epochs
- user split seed `12345`
- training seeds `(11,23,101)`

## Exact pairing contracts

For condition A, the code reuses `experiment_3_0_4_l2_width_representation_capacity.L2WidthNet(128, timestep_ce)` and the historical Exp3 seed streams:

```text
shared_backbone_init
objective-specific timestep_ce head_init
Exp3 train / val / test loader seeds
```

The contract test verifies its initial `state_dict` is elementwise identical to historical `L3AblationNet('B')`.

Conditions B and C both use the same Exp5 stream:

```text
exp5_0_paired / model_init
exp5_0_paired / train_loader
exp5_0_paired / val_loader
exp5_0_paired / test_loader
```

Their random parameterized modules are constructed in the same order (`f1`, `f2`, `head`), so B and C start with identical feedforward/head parameters. Only the hidden neuron dynamics differ.

Condition C is additionally checked against existing Exp5.0 binary: with the same seed and input, its L1/L2 spike trajectories must be identical to `LocalEvidenceSNN(... hidden_cap=1, output_cap=1)` through L2.

## Loss and checkpoint selection

For valid timestep `t`:

```text
s_t^L2 -> Linear(128,12) -> logits_t
L = mean_t CE(logits_t, y)
```

The native segment logits are the mean valid timestep logits.

Every new run selects its checkpoint by:

1. maximum validation native analog-head balanced accuracy;
2. tie-break minimum validation CE.

This matches historical Exp3. Exp5.0 used validation Output WholeCount BA; that remains part of the existing Exp5 reference rather than being silently mixed into the new controls.

## Frozen probes

After checkpoint selection, the SNN is frozen and the exact Experiment 3.0.5 probe protocol is reused:

- `full_count`
- `fixed250_ordered`
- `fixed250_shuffled`
- `fixed250_pca128`
- `relative10_ordered`
- `relative10_shuffled`
- `relative10_pca128`
- `duration_only`

The probe implementation reuses the Exp3 train-only `StandardScaler`, LogisticRegression `C` grid, validation selection, and PCA controls.

Primary representation comparisons:

```text
L2 FullCount + Linear
L2 Fixed250 ordered + Linear
L2 Relative10 ordered + Linear
```

## Reference comparisons

The finalizer reads committed source artifacts from:

- Exp3.0.5 `timestep_ce` — historical analog-head reference;
- Exp5.0 `timestep_ce + binary` — binary spiking-output reference;
- Exp5.0 `timestep_ce + multi_ho` — higher output event-capacity reference.

`comparison_runs.csv` places all sources into one schema. The notebook performs mean/SD and paired-delta aggregation.

Pairing rules:

- A vs historical Exp3: seeds `11,23,101`;
- B vs C: seeds `11,23,101`;
- C vs existing Exp5: only common seeds `11,23` are treated as paired.

## Multi-CPU execution

Three conditions x three seeds = nine independent training runs:

```text
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=1
```

Each array task performs:

```text
train -> select best checkpoint -> frozen probe evaluation -> per-run JSON
```

The `afterok` finalizer requires all nine artifacts and writes:

```text
runs.csv
comparison_runs.csv
manifest.json
```

It does not retrain or regenerate missing runs. The notebook is analysis-only and performs the requested aggregation from these finalized tables.

## Run

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_5_0_1_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_5_0_1_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_5_0_1_exp3_analog_head_control/analog_head_three_way_control_v3/
```
