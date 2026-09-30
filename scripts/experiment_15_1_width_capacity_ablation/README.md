# Experiment 15.1 — Hidden-Width Capacity Ablation

## Purpose

Exp15 showed that the 128-neuron two-layer WCCE backbone can fit the training users extremely well while validation/test balanced accuracy remains much lower, and that the added context-dependent scalar gate changes its internal dynamics without becoming strongly necessary for OOD classification.

Exp15.1 asks a narrower question before changing the loss or gate mechanism:

```text
Is the current 128-neuron-per-layer SNN over-capacity for the available training set,
and does reducing hidden width make the unchanged Exp15 memory gate more useful?
```

This is a pure capacity ablation. The only intended experimental axis is hidden width.

## Widths

Both hidden layers use the same width:

```text
H in {64, 96, 128}
L1 = H
L2 = H
```

The 128-neuron condition is the completed Exp15 reference and is never retrained in Exp15.1.

The 64- and 96-neuron conditions each receive their own independently trained width-matched B0 baseline before any continuation/gating case is run.

## Locked contract

Everything below is inherited from Exp15/CoreBenchmark and must remain unchanged:

- production dataset and frozen train/validation/test user split;
- labels and 30 event input channels;
- seeds 11, 23, 37;
- two hidden SNN layers;
- shifts `(2,3,4)/(2,3,4)`;
- tau_mem = 22.54 ms;
- threshold = 0.5;
- unnormalized synaptic current update;
- WCCE task loss;
- Adam learning rate / weight decay / epoch budget / patience;
- native bias-free accumulator readout;
- checkpoint selection by validation native BA, then validation mean-logit CE, with epoch 0 eligible;
- Exp15 scalar gate equation;
- q=0.9 function-preserving gate initialization;
- Exp15 gate diagnostics;
- CoreBenchmark L2 spike probes;
- Exp15 Phase 1.5 interventions A0/A1/A2/A3/A4/A4b.

Exp15.1 intentionally does **not** add TSCE, Prefix-WCCE, cross-user loss, regularization, suppress-only gating, group/neuron-wise gates, adaptive tau, or any other mechanism.

## Why 64 and 96 require new B0 checkpoints

A smaller model cannot inherit the 128-neuron checkpoint by truncation because that would mix width effects with an arbitrary projection/inheritance rule.

For each new width:

```text
B0_H: train from the same paired initialization rule and the same CoreBenchmark data/order
  |
  +--> C0_H: continue the ungated B0_H
  +--> GF_H: insert the Exp15 gate, freeze the B0_H backbone/readout
  +--> GJ_H: insert the Exp15 gate, jointly continue gate + backbone/readout
```

The matched comparisons therefore remain:

```text
GF_H - B0_H
GJ_H - C0_H
```

within each width.

## Parameter counts

Ignoring non-trainable state and buffers, the ungated two-layer model has:

```text
P(H) = 30H + H^2 + 12H = H^2 + 42H
```

So:

```text
H=64:   6,784 baseline parameters
H=96:  13,248 baseline parameters
H=128: 21,760 baseline parameters
```

The scalar Exp15 gate adds:

```text
2H + 1
```

parameters.

Thus H=64 uses about 31% of the H=128 baseline parameter count.

## Phase 1 evaluation

For every B0/C0/GF/GJ checkpoint, report:

- train/validation/test BA, accuracy, macro-F1;
- train-test BA gap;
- per-OOD-user BA;
- L2 no-bias and affine probes:
  - whole_count;
  - fixed250_ordered;
  - fixed250_shuffled;
  - relative10_ordered;
  - relative10_shuffled;
- probe contrasts:
  - G_resolved;
  - G_order;
  - G_relative_order;
  - Gap_rel10_whole;
- GF/GJ gate diagnostics from Exp15.

The finalizer writes width-level summary tables so the primary analysis can directly compare:

```text
width -> train BA / val BA / test BA / generalization gap
width -> WholeCount / Fixed250 / Relative10 accessibility
width -> gate engagement and Phase1.5 intervention dependency
```

## Phase 1.5

GF/GJ at H=64 and H=96 run the exact Exp15 causal interventions:

- A0 learned gate;
- A1 force m[t]=1;
- A2 preserve per-sequence mean gate but remove within-sequence variation;
- A3 shuffle the learned gate trajectory within valid timesteps;
- A4 remove the history term from the gate;
- A4b preserve only the sequence-average history contribution.

H=128 Phase 1.5 results are reused from Exp15.

A particularly informative capacity result would be:

```text
H decreases
  -> train BA decreases or saturates less
  -> test BA stays stable or increases
  -> train-test gap decreases
  -> A0-A1/A2/A3 becomes larger
```

because that would support the hypothesis that memory selection becomes more useful when representational capacity is actually constrained.

## Multi-CPU execution

Exp15.1 uses task-level CPU arrays:

```text
prepare
  -> B0 training: 6 tasks
       H64/H96 x 3 seeds
  -> Phase 1 continuation: 18 tasks
       H64/H96 x C0/GF/GJ x 3 seeds
  -> Phase 1.5: 12 tasks
       H64/H96 x GF/GJ x 3 seeds
  -> finalizer
       merge new H64/H96 results with immutable H128 Exp15 artifacts
```

Each task requests one CPU core and sets OMP/MKL/OpenBLAS/NumExpr thread counts to one.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_15_1_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/
```

## Validation

Focused checks:

```bash
python -m pytest -q tests/test_repository_source_syntax.py
python -m pytest -q tests/test_experiment_15_1_width_capacity_ablation_contract.py
```
