# Exp11.0 — D0/D1 context × fusion factorial with frozen-L1 control

## Goal

Exp11.0 asks whether the temporal information exposed by external Fixed250 probes can be internalized by:

1. recurrent context;
2. a Fusion SNN that remaps current local evidence using context;
3. their interaction.

The new frozen-L1 control separates **using the original pretrained local representation** from **end-to-end reshaping of L1 by WCCE**.

## Dataset factor

- D0 `original`: `Encoder(a)`
- D1 `postencode_mask`: `m * Encoder(a)`

D0 and D1 remain paired by sample identity, user split, seed, common random initialization, and DataLoader ordering.

## A/B/C/D architecture factor

| Case | Context | Fusion | Architecture |
| --- | --- | --- | --- |
| A | FF / recurrence off | off | `L1 -> FF context -> Linear` |
| B | diagonal or dense RSNN | off | `L1 -> RSNN -> Linear` |
| C | FF / recurrence off | on | `L1 -> FF context -> Fusion(L1, context) -> Linear` |
| D | diagonal or dense RSNN | on | `L1 -> RSNN context -> Fusion(L1, context) -> Linear` |

B and D each have diagonal and dense variants, so each dataset/L1-mode/seed has six concrete architectures.

## L1 modes

Three L1 conditions are now used:

### dynamics_only

- L1 dynamics follow Exp10.2.
- `input -> L1` weight is paired-random.
- L1 weight is trainable.

### pretrained_trainable

- Load the seed-matched source `input -> L1` weight.
- Keep it trainable under Exp11.0 WCCE.
- Measures the best end-to-end result when L1 is allowed to adapt.

### pretrained_frozen

- Load the exact same seed-matched source `input -> L1` weight.
- Set `requires_grad=False`.
- Exclude the L1 weight from the Adam parameter list.
- Downstream context/Fusion/readout remain trainable.

This branch directly tests whether the downstream SNN can exploit the temporal representation that already existed before Exp11.0 training.

D1 source is the existing Exp10.2.1 binary/L1-mem2/L2-mem1 checkpoint. D0 first trains the same three matched source checkpoints as before. Only `input -> L1` is inherited.

## Dynamics

L1:

- width 128;
- `tau_mem ~= 54 ms`;
- synaptic shifts `(2,3,4)`;
- binary communication.

Context:

- width 128;
- `tau_syn ~= tau_mem ~= 54 ms`;
- topology `ff / diagonal / dense`.

Fusion, when enabled:

- width 128;
- `tau_syn ~= tau_mem ~= 54 ms`;
- receives both L1 spikes and context spikes;
- no recurrence.

## Objective

All main runs use only valid-length mean WCCE:

```math
e_t = W_o h_t^{readout}
```

```math
score = \frac{1}{T_{valid}}\sum_{t<T_{valid}} e_t
```

```math
\mathcal{L}=CE(score,y)
```

No TSCE or auxiliary loss is used. Gradient norm is clipped at 1.0.

## Main factorial

```text
2 datasets
x 3 L1 modes
x 3 context topologies
x 2 Fusion states
x 3 seeds
= 108 main runs
```

The run list keeps D0/D1 adjacent for every matched condition. Slurm caps concurrency at 50.

## Frozen-L1 mechanism questions

### Can recurrence use the original representation?

Under `pretrained_frozen`:

```text
B - A
```

tests recurrence without allowing WCCE to change L1.

### Does context-conditioned remapping help?

Under `pretrained_frozen`:

```text
D - B
```

tests whether Fusion adds value on top of a fixed recurrent context.

### How much does end-to-end L1 adaptation matter?

```text
pretrained_trainable - pretrained_frozen
```

is reported for every dataset/topology/Fusion/seed combination.

## L1 representation preservation

Each pretrained run stores:

```text
relative Frobenius drift =
||W_L1(final) - W_L1(source)||_F / ||W_L1(source)||_F
```

and cosine similarity to the source L1 weight.

Expected frozen behavior:

```text
relative drift = 0
cosine similarity = 1
```

The finalizer also compares both L1 analog and binary probes:

- L1 pre-reset whole;
- L1 pre-reset Fixed250;
- L1 binary whole;
- L1 binary Fixed250.

This matters because the previously observed ~70% Fixed250 signal is associated more closely with the L1 pre-reset analog state than with the binary spike output.

## Readout probes

The final hidden representation is called `readout`:

- Fusion off: `readout = context`;
- Fusion on: `readout = Fusion(L1, context)`.

For L1, context/RSNN, and readout, both valid-length and whole-window probe support are evaluated.

Primary temporal diagnostic:

```math
G_{temporal}=BA_{Fixed250}-BA_{whole}.
```

## Main contrasts

Finalization reports:

- diagonal/dense recurrence minus FF;
- Fusion on minus off;
- recurrence × Fusion `(D-C)-(B-A)`;
- paired D1-D0;
- pretrained_trainable minus dynamics_only;
- pretrained_frozen minus dynamics_only;
- pretrained_trainable minus pretrained_frozen.

## Multi-CPU execution

```text
prepare
  |
  v
3-way D0 matched-source array
  |
  v
108-way main array, max 50 concurrent
  |
  v
afterok finalizer
  |
  v
aggregation-only notebook
```

Main array:

```text
#SBATCH --array=0-107%50
```

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

Output root:

```text
notebooks/artifacts/
  experiment_11_0_rsnn_history_internalization/
    d0_d1_l1mem2_context_fusion_frozen_l1_v4/
```
