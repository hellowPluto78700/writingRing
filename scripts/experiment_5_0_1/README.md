# Experiment 5.0.1 — Exp3 Analog-Head Control

## Question

Experiment 3 learned strong local representations with a two-layer multi-`tau_syn` backbone and `timestep_ce`, while Experiment 5.0 showed a large degradation when timestep supervision was routed through a 12-neuron spiking output layer.

This control asks:

> If we restore the exact Experiment 3 training path — `L2 spike -> analog Linear(128,12,bias=True) -> timestep CE` — while keeping the same `128_(2,3,4) -> 128_(2,3,4)` local backbone, do we recover the Experiment 3 representation quality?

A recovery would support the interpretation that the main Exp5.0 timestep degradation came from the spiking output bottleneck, not from failure of the local multi-`tau_syn` backbone.

## Exact control contract

The model is instantiated from `experiment_3_0_4_l2_width_representation_capacity.L2WidthNet` with:

```text
Raw64 30-channel unsigned weighted events
  -> L1: 128 Synaptic neurons, shifts (2,3,4)
  -> L2: 128 Synaptic neurons, shifts (2,3,4)
  -> Linear(128,12,bias=True)
```

The experiment intentionally has **no output LIF**.

Shared Exp3 constants are retained:

- `tau_mem = 22.54 ms`
- threshold `0.5`
- subtract reset
- fast-sigmoid surrogate slope `25`
- Adam, `lr=1e-3`
- 100 epochs
- user split seed `12345`
- training seeds `(11,23,101)`

Initialization and loader seeds match Experiment 3.0.4/3.0.3-B:

```text
shared_backbone_init
objective-specific head_init
train / val / test loader streams
```

## Loss and checkpoint selection

For valid timestep `t`:

```text
s_t^L2 -> Linear(128,12) -> logits_t
```

and

```text
L = mean_t CE(logits_t, y)
```

The segment-level native logits are the mean valid timestep logits, matching Experiment 3.

Every run selects its checkpoint by:

1. maximum validation native analog-head balanced accuracy;
2. tie-break minimum validation CE.

This is deliberately different from Experiment 5.0, where all objectives selected checkpoints by validation Output WholeCount BA.

## Frozen probes

After checkpoint selection, the SNN is frozen and the Experiment 3.0.5 probe protocol is reused exactly:

- `full_count`
- `fixed250_ordered`
- `fixed250_shuffled`
- `fixed250_pca128`
- `relative10_ordered`
- `relative10_shuffled`
- `relative10_pca128`
- `duration_only`

The probe implementation reuses the Exp3 train-only `StandardScaler`, LogisticRegression `C` grid, validation selection, and PCA controls.

Primary comparisons are:

```text
L2 full count + Linear
L2 Fixed250 ordered + Linear
L2 Relative10 ordered + Linear
```

## Reference comparisons

The finalizer also reads the committed source artifacts from:

- Exp3.0.5 `timestep_ce` — exact historical analog-head reference;
- Exp5.0 `timestep_ce + binary` — spiking-output bottleneck;
- Exp5.0 `timestep_ce + multi_ho` — higher output event-capacity reference.

`comparison_runs.csv` puts these sources into a common schema. The notebook performs the final mean/SD and paired-delta aggregation.

## Multi-CPU execution

There are only three independent training seeds, so the Slurm array is:

```text
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=1
```

Each array task performs:

```text
train -> select best checkpoint -> frozen probe evaluation -> per-run JSON
```

The `afterok` finalizer checks that all three artifacts exist and writes `runs.csv`, `comparison_runs.csv`, and `manifest.json`. It does not retrain or regenerate missing runs.

The notebook is analysis-only and performs the requested result aggregation from these finalized tables.

## Run

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_5_0_1_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_5_0_1_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_5_0_1_exp3_analog_head_control/exp3_exact_analog_timestep_head_v1/
```
