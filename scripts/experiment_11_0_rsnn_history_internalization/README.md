# Exp11.0 — D0/D1 context × fusion factorial

## Question

Exp10.2.2 showed that ordered Fixed250 probes expose temporal information that the native time-shared readout cannot fully use. Exp11.0 tests two separate mechanisms for internalizing that information:

1. recurrence can convert local L1 evidence into a history-aware context representation;
2. a Fusion SNN can use that context to remap the current local representation before the shared Linear readout.

The experiment now implements the complete A/B/C/D decomposition rather than keeping Fusion always enabled.

## Dataset factor

- D0 `original`: `Encoder(a)`
- D1 `postencode_mask`: `m * Encoder(a)`

D0 and D1 use paired sample geometry, user split, seed, common random initialization streams, and DataLoader ordering.

## A/B/C/D architecture decomposition

| Case | Context topology | Fusion | Architecture | Main question |
| --- | --- | --- | --- | --- |
| A | FF / recurrence off | off | `L1 -> FF context -> Linear` | depth / context control |
| B | diagonal or dense RSNN | off | `L1 -> RSNN -> Linear` | recurrence itself |
| C | FF / recurrence off | on | `L1 -> FF context -> Fusion(L1, context) -> Linear` | Fusion architecture itself |
| D | diagonal or dense RSNN | on | `L1 -> RSNN context -> Fusion(L1, context) -> Linear` | target history-conditioned remapping |

B and D each have two recurrence variants:

- `diagonal`: 128 trainable self-recurrent weights;
- `dense`: full `128 x 128` recurrent matrix.

This gives six concrete architecture configurations per dataset/L1-init/seed:

```text
A         = ff       + fusion off
B-diag    = diagonal + fusion off
B-dense   = dense    + fusion off
C         = ff       + fusion on
D-diag    = diagonal + fusion on
D-dense   = dense    + fusion on
```

## Dynamics

L1:

- width 128;
- `tau_mem ~= 54 ms` from membrane shift2;
- original multi-tau synaptic shifts `(2,3,4)`;
- binary communication.

Context layer:

- width 128;
- `tau_syn ~= 54 ms`;
- `tau_mem ~= 54 ms`;
- binary communication;
- topology `ff / diagonal / dense`.

`ff` means **no recurrent feedback**, but the context neurons still have their intrinsic 54 ms synaptic and membrane dynamics. Therefore A/B isolates recurrent feedback on top of matched local neuron dynamics.

Fusion layer, when enabled:

```math
I_t^F
=
\alpha_F I_{t-1}^F
+
W_z z_t
+
W_r r_t
```

with width 128, `tau_syn ~= tau_mem ~= 54 ms`, binary communication, and no recurrence.

When Fusion is disabled, no Fusion weight matrices are instantiated. The shared Linear reads the context output directly.

## Readout representation

To make probes comparable across all four cases, the final hidden representation is named `readout`:

- Fusion off: `readout_t = context_t`;
- Fusion on: `readout_t = Fusion(L1_t, context_t)`.

Thus the same `readout whole` and `readout Fixed250` probes can be compared across A/B/C/D without pretending a Fusion layer exists in A/B.

## L1 initialization factor

Two modes:

### dynamics_only

Use the Exp10.2 L1 dynamics but paired-random `input -> L1` weights.

### pretrained_input

Copy only the seed-matched source `input -> L1` matrix and keep it trainable.

D1 source:

```text
Exp10.2.1 binary__l1mem2__l2mem1__seed{seed}
```

D0 has no pre-existing matched source, so Exp11.0 first trains three D0 binary/WCCE `L1 mem2 / L2 mem1` source checkpoints. D0 never inherits D1-trained input weights.

## Objective

Every main run uses only:

```math
e_t = W_o h_t^{readout}
```

```math
score =
\frac{1}{T_{valid}}
\sum_{t<T_{valid}} e_t
```

```math
\mathcal{L}=CE(score,y)
```

No TSCE or auxiliary loss is used. The output matrix is bias-free and time-shared. Gradient norm is clipped at 1.0.

## Main factorial

```text
2 datasets
x 2 L1 init modes
x 3 context topologies
x 2 fusion states
x 3 seeds
= 72 main runs
```

Before the main factorial, three matched D0 source runs are trained.

## Core contrasts

The finalizer computes paired contrasts for identical seed/split conditions.

### Recurrence itself

```text
B-diag - A
B-dense - A
```

and, with Fusion enabled:

```text
D-diag - C
D-dense - C
```

### Fusion itself

```text
C - A
D-diag - B-diag
D-dense - B-dense
```

implemented as `fusion_on_minus_off`.

### Recurrence × Fusion interaction

The key mechanism interaction is:

```text
(D - C) - (B - A)
```

reported separately for diagonal and dense recurrence.

A positive interaction means recurrence becomes more useful when the history state is allowed to remap current L1 evidence through Fusion, rather than only being decoded directly.

### Dataset and pretraining effects

The finalizer also reports:

- paired `D1 - D0`;
- `pretrained_input - dynamics_only`;
- D1 × recurrence;
- D1 × Fusion;
- pretraining × recurrence.

## Fixed250 internalization diagnostics

For L1, context/RSNN, and final `readout`, under both valid-length and whole-window support:

- pre-reset whole mean;
- pre-reset ordered Fixed250 mean;
- binary whole mean;
- binary ordered Fixed250 mean;
- binary whole count;
- binary Fixed250 count.

Main diagnostic:

```math
G_{temporal}
=
BA_{Fixed250}
-
BA_{whole}
```

The desired target-model signature is that the final `readout` gap becomes smaller while whole/native BA increases.

## Recurrence health diagnostics

Each run records:

- firing activity for L1/context/readout;
- dead-neuron fraction;
- post-valid residual firing;
- mean absolute external context input;
- mean absolute recurrent input;
- recurrent/external input magnitude ratio;
- recurrent weight norm.

These are used to detect sustained recurrent excitation.

## Multi-CPU execution

Execution graph:

```text
prepare
  |
  v
3-way D0 matched-source array
  |
  v
72-way main array, max 50 concurrent
  |
  v
afterok finalizer
  |
  v
aggregation-only notebook
```

The main array is:

```text
#SBATCH --array=0-71%50
```

which follows the repository default maximum of 50 concurrent CPU experiment tasks.

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

Inspect run mapping:

```bash
python -m scripts.experiment_11_0_rsnn_history_internalization list-source-runs
python -m scripts.experiment_11_0_rsnn_history_internalization list-runs
```

Output root:

```text
notebooks/artifacts/
  experiment_11_0_rsnn_history_internalization/
    d0_d1_l1mem2_context_fusion_factorial_v3/
```

The finalizer only aggregates existing artifacts. The notebook is analysis-only.
