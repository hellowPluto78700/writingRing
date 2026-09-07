# Experiment 5.3.1 - Factorized gate causal attribution

## Scientific question

Experiment 5.3.1 isolates the only positive representation result from Experiment 5.3: the factorized RSNN increased phase-blind Hidden WholeCount accessibility while largely preserving the ordered Relative10 trajectory. The key unresolved question is whether that gain came from **instantaneous content-conditioned nonlinear gating** or from **causal temporal history**.

The experiment therefore keeps the validated frozen Local SNN and the WHAT bypass fixed, and changes only the mechanism that generates the multiplicative gate.

```text
Raw64
  -> frozen Exp3-exact Local SNN
  -> 128-D binary L2 spike trajectory z_t
  -> WHAT bypass -------------------------------> (*) -> contextualized feature z~_t
                    \                            ^
                     -> gate generator u_t -> g_t

z~_t = g_t * z_t
  -> bias-free Linear(128, 12) signed evidence e_t
  -> non-leaky analog accumulator A_T = sum_valid(e_t)
  -> final accumulated CE
```

The primary hypothesis is deliberately narrow:

```text
H0: the Exp5.3 factorized gain is mainly instantaneous nonlinear content gating.
H1: stateful temporal context adds class-relevant accessibility beyond the trained FF gate.
```

## Frozen local source

All conditions reuse the exact Exp5.2 frozen-local cache:

```text
experiment_5_2_frozen_local_tauR_sweep
frozen_exp3_l2_endpoint_tauR_v1
```

The cached representation is the binary 128-D L2 spike trajectory at 64 Hz. No Fixed250, Relative10, or future valid length is supplied to the gate generator. Valid length is used only for masked final accumulation and post-hoc probes.

Seeds remain paired:

```text
(11, 23, 37, 53, 71)
```

Within each seed, all conditions share the same frozen local samples, labels, valid lengths, loader seed, `input_projection`, `gate_projection`, and `evidence_head` initialization. The RSNN condition alone adds a separately seeded recurrent matrix.

## Four pre-specified conditions

There is no tau sweep and no validation-based condition selection. All four conditions are reported.

| Condition | Gate generator | beta | History source |
|---|---|---:|---|
| `factorized_ff_gate` | one-step LIF-like operator, no carried state | 0.0 | none |
| `factorized_lif22` | passive LIF state | 0.50 | short membrane history, tau_mem about 22.5 ms |
| `factorized_lif242` | passive LIF state | 0.9375 | long membrane history, tau_mem about 242 ms |
| `factorized_rsnn22` | LIF state plus previous-spike recurrence | 0.50 | short passive state plus learned recurrence |

The FF gate is not merely a post-hoc reset of an RSNN checkpoint. It is trained from scratch with no carried temporal state. Because beta is zero and there is no recurrent matrix, its gate is a deterministic function of the current local feature only.

## Shared factorized equations

For every condition, WHAT is preserved through a bypass. The gate branch computes a state from the same current local feature:

```math
d_t = W_{in}z_t + \mathbb{1}_{rec}W_{rec}s_{t-1},
```

```math
u_t^- = \beta u_{t-1} + d_t,
\qquad
s_t = H(u_t^- - \theta),
```

followed by repository-style reset. The multiplicative gate is

```math
g_t = 2\sigma(W_gu_t+b_g),
```

and the preserved local feature is modulated rather than replaced:

```math
\tilde z_t = g_t\odot z_t.
```

Signed class evidence is

```math
e_t=W_o\tilde z_t,
```

where the evidence head has `bias=False`.

The four conditions differ only in whether `u_{t-1}` is carried, how slowly it leaks, and whether `s_{t-1}` enters through a trainable recurrent matrix.

## Training objective

The training objective is unchanged from Exp5.3 so that attribution is not confounded with a different loss:

```math
A_i=\sum_{t<T_i}e_{i,t},
```

```math
L=CE\left(5\frac{A_i}{T_i},y_i\right).
```

There is no timestep CE, Fixed250 CE, Relative10 CE, phase supervision, or auxiliary gate objective.

Each checkpoint is selected by maximum native validation accumulator balanced accuracy, tie-broken by validation CE. Test metrics and all linear probes are post-selection only.

## New evaluation matrix

Experiment 5.3.1 explicitly separates four questions.

### 1. Preservation ceiling

`hidden_relative10_ordered` is an ordered-trajectory ceiling only:

```math
P_m = BA(\mathrm{HiddenRelative10}_m).
```

It answers whether the factorized transform preserved the discrimination structure that can be decoded when temporal order is supplied externally. It is **not** evidence that WHEN has been internalized.

### 2. Phase-blind internalization

The primary representation metric is

```math
I_m = BA(\mathrm{HiddenWholeCount}_m)-BA(\mathrm{LocalWholeCount}).
```

WholeCount removes the explicit temporal index before the Linear probe. A positive value means the transformed feature identity is more linearly class-accessible after orderless aggregation.

The main paired causal contrasts are

```math
H_{passive22}=WC_{LIF22}-WC_{FF},
```

```math
H_{passive242}=WC_{LIF242}-WC_{FF},
```

```math
H_{rec}=WC_{RSNN22}-WC_{LIF22}.
```

These contrasts distinguish current-content nonlinear remapping from passive temporal state and learned recurrence.

### 3. Causal attribution within a trained model

Every checkpoint is evaluated under:

- `ordered`: normal causal trajectory;
- `state_reset`: membrane and recurrent spike state are reset before every timestep;
- `temporal_shuffle`: the valid local trajectory is permuted before the gate generator, using five deterministic replicates.

For every intervention the experiment records:

- native accumulator BA;
- a newly fitted Hidden WholeCount Linear probe (`refit` accessibility);
- an **ordered-trained transfer probe** evaluated without refitting.

The transfer probe is fitted once on ordered train data, chooses regularization from ordered validation data, and then keeps both scaler and classifier fixed when evaluating state-reset or shuffled representations. This distinguishes a representation that remains linearly decodable after refitting from one whose coordinate semantics remain compatible with the ordered representation.

### 4. Direct history-sensitivity diagnostics

For the exact same current `z_t`, the model is run normally and with history reset. The following quantities are recorded on train/val/test:

```text
gate_history_mae
active_feature_gate_history_mae
evidence_history_mae
accumulator_history_l1_mean
```

`active_feature_gate_history_mae` weights the gate difference only where the frozen Local SNN is firing, because gate changes on zero local features cannot alter `g_t * z_t`.

These diagnostics answer whether the history branch materially changes the actual local evidence, not merely whether a membrane trajectory exists.

## Secondary probes

The normal ordered checkpoint also records:

- `hidden_fixed250_ordered`
- `hidden_relative10_ordered`
- `phase_contextual`
- `phase_membrane`
- `phase_gate`

Phase labels are post-hoc relative-progress deciles. They are not training targets and are not primary success criteria.

## Success criterion

Evidence for genuine history internalization requires a convergent pattern:

```text
HiddenWholeCount(stateful) > HiddenWholeCount(FF)
AND
HiddenWholeCount(ordered) > HiddenWholeCount(state_reset)
AND
HiddenRelative10 remains high
AND
native accumulator BA also improves
```

A stateful model that has a large internal state but does not beat the trained FF gate on phase-blind WholeCount is not counted as successful WHEN internalization.

## Multi-CPU execution

The experiment follows repository task-level Slurm defaults.

### Frozen-local preparation

```text
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=1
```

Each task validates/prepares one Exp5.2 frozen-local seed cache.

### Independent train/evaluate tasks

Four conditions x five seeds produce exactly 20 independent runs:

```text
#SBATCH --array=0-19%20
#SBATCH --cpus-per-task=1
```

Each run task performs atomically:

```text
load frozen cache
-> train one condition/seed
-> select best checkpoint on validation native BA
-> evaluate ordered checkpoint
-> fit ordered classification probes
-> evaluate state-reset intervention
-> evaluate five temporal-shuffle replicates
-> evaluate ordered-trained transfer probe without refitting
-> compute gate/evidence history sensitivity
-> write checkpoint/history/evaluation artifacts
```

All compute jobs initialize Conda locally and set OMP/MKL/OPENBLAS/NUMEXPR threads to one.

### Finalizer

The submission chain is

```text
prepare array
  -> afterok
20-run train/evaluate array
  -> afterok
finalizer
```

The finalizer never retrains and never regenerates a missing run. Missing or identity-mismatched artifacts raise an error.

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_3_1_cpu.bash
```

## Finalized artifacts

The finalizer writes only aggregated durable results:

```text
notebooks/artifacts/experiment_5_3_1_factorized_gate_attribution/
  factorized_gate_causal_attribution_v1/
    runs.csv
    histories.csv
    probe_runs.csv
    phase_probe_runs.csv
    ablation_runs.csv
    history_sensitivity_runs.csv
    local_reference.csv
    manifest.json
```

Per-run checkpoints, histories, and evaluation JSON files remain in their own subdirectories.

## Analysis-only notebook

`notebooks/experiment_5_3_1_factorized_gate_attribution.ipynb` reads finalized artifacts only. It does not train models, run multiprocessing, or submit Slurm jobs.

The notebook is organized around the new matrix:

1. **Preservation and internalization** - Local WholeCount / Local Relative10 references against Hidden WholeCount / Hidden Relative10.
2. **Paired contribution decomposition** - FF -> LIF22, FF -> LIF242, and LIF22 -> RSNN22 for Hidden WholeCount and native BA.
3. **Causal intervention** - ordered, state-reset, and temporal-shuffle for native BA, refit WholeCount, and ordered-trained transfer WholeCount.
4. **History sensitivity** - gate, active-gate, evidence, and accumulated-evidence changes caused by history.
5. **Training curves and secondary phase probes** - diagnostics only.

No condition is selected by test performance, and Relative10 is never interpreted as direct proof of WHEN internalization.
