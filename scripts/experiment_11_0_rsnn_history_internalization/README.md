# Exp11.0 — D0/D1 RSNN history internalization

## Question

Exp10.2.2 showed that ordered Fixed250 probes expose temporal information that the time-shared native readout cannot fully use. Exp11.0 asks whether an RSNN can internalize that ordering into a history-aware representation, and now tests the mechanism **in parallel on both D0 and D1**.

Dataset factor:

- `original` = D0 = `Encoder(a)`
- `postencode_mask` = D1 = `m * Encoder(a)`

The main scientific comparison is therefore not only recurrence vs no recurrence, but also whether the D1 preprocessing advantage survives after recurrent history modeling.

## Architecture

```text
30-channel D0/D1 events
    |
    v
L1 local SNN, 128
    | z_t binary
    +----------------------------+
    |                            |
    v                            |
RSNN context, 128                |
    | r_t binary                 |
    +------------+---------------+
                 |
                 v
       Fusion SNN, 128
        Wz z_t + Wr r_t
                 |
                 v q_t binary
        shared Linear 128 -> 12
                 |
                 v
       valid-length mean WCCE
```

Fusion is feed-forward. Long-range state is assigned to the RSNN branch.

## Locked dynamics

L1:
- width 128
- `tau_mem ~= 54 ms` from membrane shift2
- original multi-tau synaptic shifts `(2,3,4)`
- binary communication

RSNN and Fusion:
- width 128
- `tau_syn ~= 54 ms`
- `tau_mem ~= 54 ms`
- binary communication

Fusion has no recurrent connection. At 64 Hz the 54 ms decay factor is approximately 0.75.

## L1 initialization factor

Two conditions are tested on both datasets.

### dynamics_only

Only the Exp10.2 L1 dynamics are inherited. The input->L1 matrix uses the paired random initialization.

### pretrained_input

The input->L1 matrix is initialized from a seed-matched source and remains trainable.

For D1, the source is the existing Exp10.2.1 condition:

```text
binary__l1mem2__l2mem1__seed{seed}
```

For D0, the repository did not contain an equivalent `L1 mem2 / L2 mem1` checkpoint. Exp11.0 therefore trains three **matched D0 source checkpoints** first using the same binary/WCCE dynamics and checkpoint-selection rule. This avoids copying D1-trained weights into D0.

Only `input->L1` is copied. L2/output weights from source training are never inherited.

## Recurrence factor

Three topologies:

- `ff`: no recurrent contribution, `Wrec=0`
- `diagonal`: 128 self-recurrent weights, no cross-neuron mixing
- `dense`: full `128 x 128` recurrent matrix

Recurrent parameter counts:

- ff: 0
- diagonal: 128
- dense: 16,384

Dense vs diagonal is an architecture comparison, not a parameter-matched comparison.

## Objective

No auxiliary objective is used.

```math
e_t = W_o q_t
```

```math
score = \frac{1}{T_{valid}}\sum_{t<T_{valid}} e_t
```

```math
\mathcal{L}=CE(score,y)
```

The output matrix is bias-free and time-shared. Gradient norm is clipped at 1.0 to limit recurrent amplification.

## Main factorial

```text
2 datasets: D0, D1
x 2 L1 init modes
x 3 recurrence topologies
x 3 seeds
= 36 main runs
```

Before these runs, three D0 matched-source runs are trained.

For a fixed seed, D0 and D1 share identical split/sample geometry, shared random initialization streams, DataLoader ordering, topology, dynamics, and objective. Therefore `D1-D0` is a paired contrast.

## Fixed250 internalization diagnostics

The Exp10.2.2 valid/window support policy is preserved.

For L1, RSNN, and Fusion:

- pre-reset membrane: whole mean and ordered Fixed250 mean
- binary communication: whole mean, ordered Fixed250 mean, whole count, Fixed250 count

Main mechanism metric:

```math
G_{temporal} = BA_{Fixed250} - BA_{whole}
```

Desired internalization pattern:

```text
L1:      Fixed250 >> whole
RSNN:    gap begins to shrink
Fusion:  Fixed250 ~= whole
```

while Fusion whole/native BA increases.

The finalizer reports recurrence effects within D0/D1, pretrained-input effects, paired `D1-D0` effects, D1 x recurrence interactions, pretraining x recurrence interactions, and L1 -> RSNN -> Fusion temporal-gap movement.

## Recurrence health diagnostics

Each run records L1/RSNN/Fusion firing activity, dead-neuron fraction, post-valid residual firing, mean absolute external RSNN input, mean absolute recurrent input, recurrent/external input magnitude ratio, and recurrent weight norm.

## Multi-CPU execution

Execution graph:

```text
prepare
  |
  v
3-way D0 matched-source array
  |
  v
36-way D0+D1 main array
  |
  v
afterok finalizer
  |
  v
aggregation-only notebook
```

The 36 main runs execute concurrently, so D0 and D1 are not run sequentially.

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

Force source/main reruns:

```bash
EXP11_0_FORCE_SOURCE=1 bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
EXP11_0_FORCE=1 bash scripts/bash_script/SNN_Bash/submit_exp_11_0_cpu.bash
```

Inspect identities:

```bash
python -m scripts.experiment_11_0_rsnn_history_internalization list-source-runs
python -m scripts.experiment_11_0_rsnn_history_internalization list-runs
```

Output root:

```text
notebooks/artifacts/
  experiment_11_0_rsnn_history_internalization/
    d0_d1_l1mem2_rsnn_fusion_internalization_v2/
```

The finalizer aggregates existing artifacts only. The notebook performs analysis only.
