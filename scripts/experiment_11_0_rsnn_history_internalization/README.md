# Exp11.0 — RSNN history internalization

## Question

Exp10.2.2 showed that ordered Fixed250 probes can decode substantially more temporal information than a time-collapsed shared readout. Exp11.0 tests whether recurrent SNN dynamics can internalize that temporal ordering so that a single time-shared 128x12 output matrix can use history-aware representations directly.

The mechanism hypothesis is:

```text
external Fixed250 temporal decoder
        ->
recurrent internal context
        ->
history-conditioned fusion representation
        ->
single shared Linear readout
```

## Source contract

Exp11.0 inherits the locked Exp10.2.2/Exp10.2.1 setting:

- dataset: D1 `postencode_mask`;
- cross-user rotation: 0;
- seeds: `11, 23, 37`;
- L1 width: 128;
- L1 membrane shift: 2, approximately 54 ms;
- L1 synaptic shifts remain the original multi-tau `(2,3,4)`;
- binary hidden communication;
- native objective: valid-length time-shared WCCE;
- checkpoint selection: validation BA, validation objective loss as tiebreak.

The seed-matched pretrained source is specifically the Exp10.2.1 condition:

```text
binary__l1mem2__l2mem1__seed{seed}
```

Exp10.2.2 is evaluation-only, so its source checkpoints are physically stored under Exp10.2.1.

## Architecture

```text
30-channel D1 events
    |
    v
L1 local SNN, 128
    | z_t (binary)
    +-----------------------------+
    |                             |
    v                             |
RSNN context, 128                 |
    | r_t (binary)                |
    +-------------+---------------+
                  |
                  v
        feed-forward Fusion SNN, 128
          input = Wz z_t + Wr r_t
                  |
                  v q_t (binary)
           shared Linear 128->12
                  |
                  v
       valid-length mean WCCE
```

Fusion is deliberately non-recurrent. Memory is assigned to the RSNN; Fusion is the history-conditioned representation transform.

## L1 initialization factor

Two conditions:

1. `dynamics_only`
   - inherits the Exp10.2 L1 time constants and binary dynamics;
   - `input->L1` weight starts from the paired random initialization.

2. `pretrained_input`
   - uses the same L1 dynamics;
   - copies only `hidden_linears.0.weight` from the seed-matched Exp10.2.1 best checkpoint;
   - the copied weight remains trainable during Exp11.0 end-to-end training.

No Exp10.2 L2 or output weights are inherited.

## RSNN recurrence factor

Three topologies:

### ff

```math
R_t = 0
```

No recurrent contribution. This is the depth/parameter-path control.

### diagonal

```math
R_t = d \odot s^R_{t-1}
```

Each recurrent neuron feeds back only to itself. There is no cross-neuron mixing.

### dense

```math
R_t = W_{rec}s^R_{t-1},
\qquad
W_{rec}\in\mathbb{R}^{128\times128}.
```

This allows distributed recurrent state.

The paired dense matrix and diagonal vector share the same initialization stream; the diagonal condition is initialized from the diagonal of the paired dense matrix.

Recurrent parameter counts:

- ff: 0
- diagonal: 128
- dense: 16,384

Dense-vs-diagonal therefore remains an architecture comparison, not a parameter-matched comparison.

## RSNN and Fusion dynamics

Both use:

```text
tau_syn ~= 54 ms
tau_mem ~= 54 ms
binary communication
subtractive reset
```

At 64 Hz the decay is approximately 0.75.

RSNN:

```math
I^R_t
=
\alpha_R I^R_{t-1}
+
W_{in}z_t
+
R_t
```

```math
U^{R,-}_t
=
\beta_R U^R_{t-1}
+
I^R_t.
```

Fusion:

```math
I^F_t
=
\alpha_F I^F_{t-1}
+
W_z z_t
+
W_r r_t
```

```math
U^{F,-}_t
=
\beta_F U^F_{t-1}
+
I^F_t.
```

## Objective

No auxiliary loss is used in the primary experiment.

```math
e_t=W_o q_t
```

```math
score
=
\frac{1}{T_{valid}}
\sum_{t<T_{valid}} e_t
```

```math
\mathcal{L}=CE(score,y).
```

The output matrix is bias-free and time-shared.

Gradient norm is clipped at 1.0 because dense recurrence can otherwise create unstable recurrent amplification.

## Factorial

```text
2 L1 init modes
x 3 RSNN topologies
x 3 seeds
= 18 independent runs
```

For a fixed seed, all conditions share paired initial streams for every common randomly initialized layer and the same DataLoader ordering.

## Representation diagnostics

Exp11.0 keeps the Exp10.2.2 dual temporal-support policy:

1. `valid`: aggregate only `t < valid_length`;
2. `window`: aggregate over the full padded 256-step inference window and allow residual SNN state to evolve naturally.

For L1, RSNN and Fusion the experiment evaluates:

- pre-reset membrane:
  - whole mean;
  - ordered Fixed250 mean.
- binary communication:
  - whole mean;
  - ordered Fixed250 mean;
  - whole count;
  - Fixed250 count.

The main mechanism metric is:

```math
G_{temporal}
=
BA_{Fixed250}
-
BA_{whole}.
```

The desired mechanism signature is:

```text
L1:     Fixed250 >> whole
Fusion: Fixed250 ~= whole
```

while Fusion whole/native BA rises. This would indicate that temporal ordering formerly exposed only by an external Fixed250 decoder has become accessible after the recurrent history-conditioned transform.

Probe classifiers are train-only StandardScaler + LogisticRegression. C is selected by validation BA. Test labels remain untouched until C selection.

## Additional RSNN diagnostics

Each run records:

- L1, RSNN and Fusion firing activity;
- post-valid residual firing fraction;
- mean absolute RSNN external input;
- mean absolute recurrent input;
- recurrent/external input magnitude ratio;
- recurrent weight norm.

These diagnostics are required because dense recurrence can create sustained recurrent excitation even with moderate 54 ms time constants.

## Multi-CPU execution

The experiment follows `AGENTS.md`:

```text
prepare / audit
      |
      v
18 independent CPU runs
one Slurm array task per run
one CPU core per task
      |
      v
afterok finalizer
      |
      v
aggregation-only notebook
```

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

Force rerun completed per-run artifacts:

```bash
EXP11_0_FORCE=1 bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

List run identities:

```bash
python -m scripts.experiment_11_0_rsnn_history_internalization list-runs
```

Outputs:

```text
notebooks/artifacts/
  experiment_11_0_rsnn_history_internalization/
    d1_l1mem2_rsnn_fusion_internalization_v1/
```

The finalizer aggregates existing artifacts only and never retrains missing runs. The notebook is analysis-only.
