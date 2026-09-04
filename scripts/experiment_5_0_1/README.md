# Experiment 5.0.1 — Analog-Head Control

## Question

Experiment 3 learned strong local representations with a two-layer multi-`tau_syn` backbone and `timestep_ce`, while Experiment 5.0 showed a large degradation when timestep supervision was routed through a 12-neuron spiking output layer.

A single Exp3 rerun would only prove reproducibility. It would not isolate whether the Exp3/Exp5 gap comes from the hidden-neuron implementation or from the spiking output bottleneck. Exp5.0.1 therefore trains two paired analog-head controls:

1. `exp3_synaptic_analog`: exact historical Exp3 `snnTorch.Synaptic` hidden dynamics.
2. `exp5_macro_binary_analog`: exact Exp5.0 binary MacroMultiSpike hidden dynamics through L2, but with the output LIF removed.

Both use the same task head:

```text
L2 spike -> Linear(128,12,bias=True) -> timestep CE
```

This gives three mechanistic comparisons:

```text
new Exp3 analog vs historical Exp3
    -> reproducibility / protocol-drift check

new Exp3 analog vs new Exp5-Macro analog
    -> hidden implementation / surrogate-dynamics effect

new Exp5-Macro analog vs existing Exp5 binary spiking-output
    -> effect of routing timestep supervision through the spiking output path
```

## Shared architecture

Both new conditions use:

```text
Raw64 30-channel unsigned weighted events
  -> L1: 128 neurons, shifts (2,3,4)
  -> L2: 128 neurons, shifts (2,3,4)
  -> Linear(128,12,bias=True)
```

The experiment intentionally has **no output LIF** in either new condition.

Shared constants:

- `tau_mem = 22.54 ms`
- threshold `0.5`
- subtract reset
- surrogate slope `25`
- Adam, `lr=1e-3`
- 100 epochs
- user split seed `12345`
- training seeds `(11,23,101)`

## Condition A — exact Exp3 Synaptic hidden dynamics

This condition directly reuses `experiment_3_0_4_l2_width_representation_capacity.L2WidthNet(128, timestep_ce)` and therefore matches the historical `No-L3` Exp3.0.3-B model through L2.

Initialization and loader streams exactly match Exp3:

```text
shared_backbone_init
objective-specific timestep_ce head_init
Exp3 train / val / test loader seeds
```

The contract test additionally verifies that, for the same seed, the initial `state_dict` is elementwise identical to historical `L3AblationNet('B')`.

## Condition B — Exp5 Macro binary hidden dynamics + analog head

This condition reproduces Exp5.0 binary hidden dynamics through L2:

```text
syn_t = alpha * syn_(t-1) + W x_t
membrane/spike = MacroMultiSpikeLIF(..., cap=1)
```

It uses the Exp5.0 `exp5_0_paired/model_init` random stream and the Exp5.0 paired loader streams. `f1` and `f2` are constructed in the same order as Exp5.0, so their initial weights are directly paired with the existing Exp5 binary model.

The only task-head change is:

```text
Exp5.0: 128 -> Linear(bias=False) -> output LIF -> output spikes
Exp5.0.1: 128 -> Linear(bias=True) -> analog logits
```

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

This matches the historical Exp3 selection criterion. Exp5.0 used validation Output WholeCount BA, which remains part of the historical reference rather than being silently mixed into the new controls.

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

Pairing rules are explicit:

- new Exp3 vs historical Exp3: seeds `11,23,101`;
- new Exp3 vs new Exp5-Macro analog: seeds `11,23,101`;
- new Exp5-Macro analog vs existing Exp5: only common seeds `11,23` are treated as paired.

## Multi-CPU execution

Two conditions x three seeds = six independent training runs:

```text
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
```

Each array task performs:

```text
train -> select best checkpoint -> frozen probe evaluation -> per-run JSON
```

The `afterok` finalizer requires all six artifacts and writes:

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
notebooks/artifacts/experiment_5_0_1_exp3_analog_head_control/analog_head_hidden_dynamics_control_v2/
```
