# Exp7.4 — Latent Softmax Evidence Decomposition

## Goal

Exp7.4 extends Exp7.3 A2 (`A2_e2e_linear_wcce`) while changing only the classification head. The purpose is to isolate whether a per-timestep latent softmax bottleneck helps or harms sequence classification, and whether any loss is specifically caused by removal of evidence magnitude.

All methods use the Exp7.3 A2 backbone/training contract:

- architecture: `234x234`;
- hidden width: 128;
- L1 shifts: `(2,3,4)`;
- L2 shifts: `(2,3,4)`;
- binary hidden spikes;
- all L1/L2/head parameters train end-to-end;
- no bias;
- objective: WCCE over valid-timestep mean evidence;
- seeds: `11, 23, 37`;
- max epochs: 100;
- minimum epochs: 20;
- patience: 30;
- checkpoint selection: validation balanced accuracy, validation WCCE loss as tiebreak;
- paired model initialization and data order reuse the Exp7.3 A2 seeds.

## Head cases

Let `z_t in R^128` denote the L2 output at timestep `t`.

### A — A2 direct Linear baseline

```text
z_t -> W -> 12-class evidence
```

Formula:

```math
e_t = W z_t.
```

This case is intended as a strict Exp7.3 A2 replication.

### B — Two-Linear factorization control

```text
z_t -> W1(128x128) -> h_t -> W2(128x12) -> 12-class evidence
```

Formula:

```math
h_t = W_1 z_t,
```

```math
e_t = W_2 h_t.
```

Because there is no nonlinearity, this remains a linear mapping in `z_t`. `B-A` therefore controls for optimization / overparameterization effects caused by factorizing the head.

### C — Latent Softmax bottleneck

```text
z_t -> W1 -> h_t -> softmax(tau=1) -> q_t -> W2 -> evidence
```

Formula:

```math
q_t = softmax(h_t),
```

```math
e_t = W_2 q_t.
```

`C-B` isolates the effect of the normalized 128-way latent competition.

### D — Latent Softmax + deterministic RMS magnitude

```text
z_t -> W1 -> h_t
              |-> softmax -> q_t
              |-> RMS ----> m_t
q_t * m_t -> W2 -> evidence
```

Formula:

```math
m_t = \sqrt{\frac{1}{128}\sum_i h_{t,i}^2},
```

```math
e_t = W_2(m_t q_t).
```

`D-C` tests whether restoring a deterministic measure of latent activation magnitude recovers performance lost by softmax normalization. No learned gain branch is used in this experiment so that the comparison remains mechanistic.

## Sequence objective

For all cases, valid timestep evidence is averaged exactly as in Exp7.3 A2:

```math
s = \frac{1}{T}\sum_{t=1}^{T} e_t,
```

and training uses:

```math
L = CE(s, y).
```

No TSCE or auxiliary loss is added.

## Diagnostics

In addition to train/val/test balanced accuracy and WCCE loss, Exp7.4 records:

- L2 whole-count affine probe BA;
- L2 fixed-250ms affine probe BA;
- latent RMS;
- softmax entropy;
- normalized softmax entropy;
- maximum latent probability;
- effective number of latent states `exp(entropy)`;
- evidence RMS;
- magnitude RMS for Case D.

The finalizer produces paired contrasts:

- `B - A`: factorization effect;
- `C - B`: softmax normalization effect;
- `D - C`: RMS-magnitude restoration effect;
- `D - B`: residual difference between normalized+magnitude routing and the factorized linear control.

## Multi-CPU execution

There are 12 independent training tasks:

```text
4 methods x 3 seeds = 12 jobs
```

Each Slurm array task uses one CPU and one Python thread. The array concurrency is capped at 12.

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_4_cpu.bash
```

This submits:

1. `run_exp_7_4_cpu_array.bash` (`0-11%12`);
2. `finalize_exp_7_4_cpu.bash` with an `afterok` dependency on the full array.

## Outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_7_4_latent_softmax_evidence/latent_softmax_evidence_v1/
```

Key aggregate files:

```text
method_runs.csv
method_summary.csv
contrast_runs.csv
contrast_summary.csv
manifest.json
```

Per-run checkpoints, histories, and evaluation JSON files are also retained.
