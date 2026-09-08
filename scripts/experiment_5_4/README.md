# Experiment 5.4 — SNN-native causal WHAT–WHEN fusion

## Question

Experiments 5.3.2.1–5.3.2.3 established a supervised causal WHEN representation but did not test whether that representation improves the **final letter-classification task**. Exp5.4 asks:

> Can frozen history-dependent WHEN spikes contextualize frozen local WHAT spikes so that a small fixed-synapse Fusion SNN produces better whole-gesture class evidence than WHAT alone, a causal elapsed-time clock, reset-WHEN, or a parameter-matched linear fusion control?

The target deployment path is:

```text
raw spikes
   -> frozen Local WHAT SNN -> s_t^WHAT (128 spikes)
                         |\
                         | -> frozen FF-SNN128 -> RSNN64 -> s_t^WHEN (64 spikes)
                         |                         |
                         +-----------+-------------+
                                     v
                              Fusion SNN128
                   W_what s_t^WHAT + W_when s_t^WHEN
                                     |
                                    LIF
                                     |
                               s_t^Fusion
                                     |
                         fixed bias-free W_out
                                     |
                            12D class evidence e_t
                                     |
                          non-leaky accumulator
                                     |
                               A_T = sum_t e_t
                                     |
                        one final class bias + CE
```

There is **no Fixed250/Relative10 pooling, no flattening, no attention, and no recurrent classifier after fusion**. The only whole-segment reduction is the final evidence accumulation.

## Frozen sources

### WHAT

The WHAT input is the same frozen Exp5.2 / Exp5.3 Local-SNN L2 trajectory:

```math
s_t^{WHAT} \in \{0,1\}^{128}.
```

### WHEN

The primary WHEN source is the Exp5.3.2.3 `ffsnn128_rsnn64` checkpoint for the same seed:

```math
s_{1:t}^{WHAT}
\rightarrow \text{FF-SNN}_{128}
\rightarrow \text{RSNN}_{64}
\rightarrow s_t^{WHEN}\in\{0,1\}^{64}.
```

The Exp5.3.2.3 joint objective is already fixed by the earlier objective study. Exp5.4 does **not** retrain WHAT or WHEN.

For each seed, a preparation task identity-validates the source artifacts and writes a frozen fusion-input cache containing:

- `what_{train,val,test}`;
- `when_ordered_{train,val,test}`;
- `when_reset_{train,val,test}`.

`when_reset` clears both FF-SNN and RSNN state before every timestep, retaining only the current WHAT input mapping.

## Primary Fusion SNN

The fusion layer has 128 neurons and no recurrent weight matrix.

```math
I_t^F = \alpha_F I_{t-1}^F
        + W_{WHAT}s_t^{WHAT}
        + W_{WHEN}s_t^{WHEN}
```

```math
U_t^F = \beta_F U_{t-1}^F + I_t^F
```

```math
s_t^F = H(U_t^F-\theta_F).
```

The layer uses the inherited short shift:

```text
shift_syn = 1
shift_mem = 1
```

so the Fusion SNN is a **local nonlinear combiner**, not a second long-term temporal decoder. There is no Fusion recurrence.

The class evidence is:

```math
e_t = W_{out}s_t^F,
```

where `W_out` is bias-free. In other words, **W_out is bias-free** and cannot inject a per-timestep duration cue. The non-leaky accumulator is:

```math
A_T=\sum_{t<T_i}e_t.
```

A single learned class bias is added **once after accumulation**:

```math
\ell_i=A_{T_i}+b_{class}.
```

This prevents an unintended duration cue `T_i b` that would occur if a per-timestep output bias were accumulated.

Training uses **final whole-sequence CE only**:

```math
L=CE(\ell_i,y_i).
```

## Why additive fixed synapses still implement WHAT × WHEN

Before thresholding, the two streams are additive:

```math
W_{WHAT}s_t^{WHAT}+W_{WHEN}s_t^{WHEN}.
```

The LIF threshold makes the mapping nonlinear. For the same WHAT pattern `X`, two WHEN states `A` and `B` can drive different fusion neurons across threshold:

```math
s_t^F(X\mid A) \neq s_t^F(X\mid B).
```

Therefore the fixed output projection receives different context-conditioned spike populations without any dynamic weight update. All synaptic matrices remain fixed after training; only synaptic/membrane state and spikes vary online.

## Six paired conditions

All six conditions use the same five seeds `(11, 23, 37, 53, 71)`, the same minibatch order within seed, and the same initial `W_what`, `W_when`, `W_out`, and class bias within seed.

| condition | WHAT input | context input | fusion | purpose |
|---|---|---|---|---|
| `what_only_lif` | frozen WHAT | zero | LIF | WHAT + same Fusion capacity control |
| `when_only_lif` | zero | ordered WHEN | LIF | how much letter identity remains in WHEN alone |
| `what_elapsed_lif` | frozen WHAT | causal elapsed-time basis | LIF | strong clock baseline |
| `what_resetwhen_lif` | frozen WHAT | reset-WHEN | LIF | instantaneous nonlinear-transform control |
| `what_when_linear` | frozen WHAT | ordered WHEN | linear | parameter-matched additive control |
| `what_when_fusion_lif` | frozen WHAT | ordered WHEN | LIF | **primary SNN-native fusion** |

This gives:

```text
6 conditions x 5 seeds = 30 independent train/evaluate runs
```

### Parameter matching

Every condition instantiates the same trainable shapes:

- `W_what`: `128 -> 128`, bias-free;
- `W_when`: `64 -> 128`, bias-free;
- `W_out`: `128 -> 12`, bias-free;
- one 12D class bias added after accumulation.

The linear control removes only Fusion LIF state/thresholding. It does not receive extra parameters.

## Causal elapsed-time baseline

The elapsed baseline receives **absolute causal time only**:

```math
e_t=t/f_s.
```

It never receives final gesture duration. To make the clock control strong, `e_t` is converted to a fixed 64D RBF basis. Centers span `0` to the **training-set maximum elapsed time**; val/test elapsed time beyond that range is clipped to the train-derived maximum.

Thus:

```math
context_t^{elapsed}=\phi(t/f_s)\in\mathbb R^{64},
```

but never:

```math
t/T_i.
```

## Variable length and causality

For sample `i`, valid mask:

```math
m_{i,t}=1[t<T_i].
```

Padding never updates the Fusion synaptic state, membrane state, spikes, evidence, or accumulator. The valid length is used only to stop padded updates and identify the endpoint. `T_i` is not provided as WHAT, WHEN, elapsed context, or Fusion input.

## Test-time attribution of the trained primary model

After training `what_when_fusion_lif`, the same checkpoint is evaluated without retraining under:

1. `when_zero` — replace aligned WHEN spikes by zero;
2. `when_shuffle` — shuffle valid WHEN timesteps within each gesture; five deterministic replicates;
3. `when_circular_shift` — circularly offset the valid WHEN trajectory by roughly one third of its length.

The circular-shift test is especially important because it preserves the gesture-specific WHEN trajectory and spike statistics while breaking alignment with the current WHAT evidence.

## Primary scientific comparisons

The main result is final letter balanced accuracy.

### Does WHEN add final-task value?

```math
\Delta_{WHEN}=BA(WHAT\times WHEN)-BA(WHAT\ only).
```

### Does learned WHEN beat a clock?

```math
\Delta_{clock}=BA(WHAT\times WHEN)-BA(WHAT+elapsed).
```

### Does ordered history matter?

```math
\Delta_{history}=BA(WHAT\times WHEN)-BA(WHAT+resetWHEN).
```

### Does the LIF interaction matter?

```math
\Delta_{interaction}=BA(WHAT\times WHEN\;FusionLIF)-BA(WHAT+WHEN\;linear).
```

### Does correct temporal alignment matter?

The trained primary model should degrade when WHEN is zeroed, shuffled, or circularly shifted.

A strong positive result is not merely “the primary condition has the highest mean.” The intended evidence chain is:

- better than WHAT-only;
- better than the strong causal elapsed-time baseline;
- better than reset-WHEN;
- better than parameter-matched linear fusion;
- aligned WHEN performs better than zeroed/shuffled/shifted WHEN.

## Checkpoint selection

Each fusion model is selected using validation data only:

1. maximize validation balanced accuracy;
2. tie-break by lower validation cross entropy.

No test metric participates in checkpoint selection.

## Multi-CPU execution

Exp5.4 follows `AGENTS.md` task-level multi-CPU execution:

```text
5 frozen fusion-input preparation tasks
        -> afterok
30 independent train -> best checkpoint -> evaluate tasks
        -> afterok
1 aggregation-only finalizer
        -> analysis-only notebook
```

Each compute task uses one CPU core. The 30-run array stays below the repository default 50-task concurrency cap.

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_cpu.bash
```

## Final artifacts

Finalizer outputs under:

```text
notebooks/artifacts/experiment_5_4_snn_native_fusion/snn_native_what_when_fusion_v1/
```

- `runs.csv` — train/val/test final-task metrics for 30 runs;
- `histories.csv` — per-epoch train/validation curves;
- `ablation_runs.csv` — primary-model zero/shuffle/circular-shift tests;
- `activity_runs.csv` — Fusion firing fraction for LIF conditions;
- `paired_deltas.csv` — per-seed primary-minus-control deltas;
- `manifest.json` — durable protocol identity and interpretation contract.

The notebook `notebooks/experiment_5_4_snn_native_fusion.ipynb` is analysis-only. It reads finalized artifacts; it does not train models, regenerate frozen features, or submit Slurm jobs.
