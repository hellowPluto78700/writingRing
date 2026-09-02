# Experiment 4.0.3 — Hierarchical Fixed250 Multi-Spike SNN

## Question

Given **exactly the same Fixed250 information** used by Exp4.0, Exp4.0.1, the Fixed250+Linear baseline, and Exp4.2, does separating local spike-code formation from recurrent temporal integration let an SNN approach continuous-RNN decoding performance?

The experiment deliberately does **not** recover raw within-bin timing. Every decoder sees only the existing zero-preserving scaled Fixed250 vectors

\[
z_b[c] = \sum_{t \in W_b} x_t[c], \qquad z_b \in \mathbb{R}^{30}.
\]

No 125 ms / 62.5 ms re-binning, Relative10, causal position feature, or other added temporal information is allowed in this experiment.

## Fixed protocol

- Same Fixed250 representation, train/val/test user split, train-only channel scaling, and seeds `(11, 23, 37, 53, 71)` as Exp4.0/4.0.1.
- Every hidden/output neuron uses the Exp4.0.1 cap-31 `MacroMultiSpikeLIF` rule.
- Hidden/event communication remains raw `0..31`; the final output event evidence is divided by 31 before WholeCount CE, matching Exp4.0.1 Multi-HO.
- All padded macro timesteps execute. Input after the causal endpoint is zero, but SNN state is never frozen by the valid mask.
- Primary objective/checkpoint readout is causal valid normalized WholeCount CE.
- The post-hoc Linear probe never participates in training or checkpoint selection.

## Conditions

### Reused baseline — `rsnn128_multispike_reused`

Reuses the Exp4.0.1 `RSNN H128, tau_mem=250 ms, Multi-HO (31,31)` checkpoints and fully-spiking metrics. This baseline is **not retrained**.

Topology:

\[
30 \rightarrow 128_R \rightarrow 12.
\]

Parameter count: 21,760.

A separate one-CPU job loads the five frozen checkpoints and trains a diagnostic Linear probe on the final hidden membrane at the causal endpoint.

### Capacity control — `rsnn176_capacity`

\[
30 \rightarrow 176_R \rightarrow 12.
\]

Parameter count:

\[
30(176) + 176^2 + 176(12) = 38,368.
\]

This nearly parameter-matches the primary two-layer hierarchy (38,144 parameters). It tests whether any gain is simply caused by additional capacity.

### Primary hierarchy — `local128_rsnn128`

\[
30 \rightarrow 128_{Local} \rightarrow 128_R \rightarrow 12.
\]

Layer 1 has no learned recurrence and `beta=0`, so the previous Fixed250 bin cannot influence the next Layer-1 state. Its role is only

\[
z_b \rightarrow \text{local population event code}.
\]

Layer 2 has `tau_mem=250 ms` and learned recurrence, and therefore owns the sequence-level temporal state.

Parameter count:

\[
30(128) + 128(128) + 128^2 + 128(12) = 38,144.
\]

### Generic depth control — `rsnn128_rsnn128`

\[
30 \rightarrow 128_R \rightarrow 128_R \rightarrow 12.
\]

Both layers have `tau_mem=250 ms` and learned recurrence. This distinguishes the proposed functional separation from generic recurrent depth.

Parameter count: 54,528.

### No-learned-recurrence control — `local128_ff128`

\[
30 \rightarrow 128_{Local} \rightarrow 128_{FF} \rightarrow 12.
\]

Layer 1 remains stateless (`beta=0`). Layer 2 has passive LIF membrane decay (`tau_mem=250 ms`) but **no learned recurrent matrix**. This measures how much two-layer nonlinear/spiking transformation achieves without learned temporal transition memory.

Parameter count: 21,760.

## Endpoint-state Linear probe

After selecting the best fully-spiking checkpoint, freeze the SNN and extract only

\[
h_i = U^{state}_{B_i},
\]

the membrane of the **final temporal hidden layer at the causal endpoint**.

A balanced logistic-regression probe then predicts the class from this single vector:

\[
\hat y = \operatorname{Linear}(h_i).
\]

The probe has no access to individual Fixed250 phases, the original `z_1, ..., z_B`, a flattened temporal representation, or any history buffer. Its purpose is diagnostic:

- high endpoint-state probe BA + low fully-spiking BA -> long-term state is informative, output spiking readout is lossy;
- low endpoint-state probe BA -> temporal SNN state itself remains insufficient.

The production/primary metric remains the fully-spiking normalized WholeCount result.

## Run matrix and multi-CPU execution

Four new conditions × five seeds = **20 independent training runs**.

Per `AGENTS.md`:

```text
20 independent runs
  -> Slurm array 0-19%20
  -> one CPU core per task
  -> train -> select best checkpoint -> evaluate -> endpoint-state probe -> per-run JSON

5 frozen Exp4.0.1 baseline checkpoints
  -> one small one-CPU probe job

array + baseline probe both succeed
  -> afterok finalizer
  -> aggregate only; never retrain missing runs
  -> analysis-only notebook
```

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_3_cpu.bash
```

## Primary interpretation

The main comparison is

\[
\text{1L RSNN128 baseline}
\quad vs \quad
\text{RSNN176 capacity control}
\quad vs \quad
\text{Local128 -> RSNN128}
\quad vs \quad
\text{RSNN128 -> RSNN128}.
\]

The strongest support for the hierarchy hypothesis would be a consistent gain of `local128_rsnn128` over both the reused one-layer baseline and the nearly parameter-matched `rsnn176_capacity` control, with endpoint-state probe performance moving toward the continuous RNN reference.
