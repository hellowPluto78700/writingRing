# Experiment 4.0.6 — Raw64 local membrane-memory ablation

## Question

Does the poor Raw64 result in Exp4.0.5 mainly come from the stateless local spike encoder (`beta_local=0`), and can short-term membrane integration recover useful local evidence without explicit Fixed250 pooling?

The experiment keeps the same Raw64 weighted event input, the same Local128 -> RSNN128 -> 12 topology, the same long-term recurrent/output dynamics, the same spike-cap conditions, and the same user split. Only the Local128 membrane time-constant distribution changes.

## Fixed architecture

```text
Raw64, 30 weighted event channels
  -> Local128: stateful LIF membrane, no learned recurrence
  -> RSNN128: tau_mem = 250 ms physical time
  -> 12 output neurons: tau_mem = 250 ms physical time
```

There is **no explicit 250-ms pooling or accumulator layer** in Exp4.0.6. Local spikes are emitted at 64 Hz and immediately drive the RSNN at 64 Hz. The experiment therefore tests whether local membrane state alone can perform useful short-term accumulation while preserving the original event timing.

## Local-memory conditions

The physical timescales are derived from the repository's existing shift convention at 64 Hz:

```text
alpha(shift) = 1 - 2^(-shift)
tau_ms(shift) = -(1000 / 64) / log(alpha)
```

The exact values are computed in code rather than copied as rounded constants.

| Condition | Local shift groups | Neuron allocation | Meaning |
|---|---|---:|---|
| `beta0` | none | 128 | frozen Exp4.0.5 Raw64 baseline; no local temporal memory |
| `shift4` | {4} | 128 | single slow local integration scale, approximately 250 ms |
| `shift34` | {3,4} | 64 / 64 | medium + slow local integration |
| `shift234` | {2,3,4} | 43 / 43 / 42 | fast + medium + slow local integration |

These are **tau_mem** conditions. They borrow the physical timescales corresponding to the repository's shift 2/3/4 convention; they are not synaptic-state (`tau_syn`) layers.

Local128 has no learned recurrent matrix in any condition. This isolates the effect of membrane memory from learned local recurrence.

## Spike-cap conditions

The same three event-cap conditions as the Raw64 portion of Exp4.0.5 are retained:

- `binary`: hidden cap 1, output cap 1
- `multi_h`: hidden cap 31, output cap 1
- `multi_ho`: hidden cap 31, output cap 31

The hidden cap applies to both Local128 and RSNN128, matching Exp4.0.5.

## Input and memory controls

- Input is exactly the Exp4.0.5 `raw64` representation.
- The channel scale is still fit on training-user valid Fixed250 bins and then applied to Raw64; this preserves the Exp4.0.5 input contract.
- Raw timestep is 15.625 ms.
- RSNN and output `tau_mem` remain 250 ms physical time, so their beta values are unchanged relative to Exp4.0.5 Raw64.
- New runs reuse the exact Exp4.0.5 `exp4_0_5_paired` random stream for model initialization and loader ordering.

## Training and readouts

Every new SNN is trained once with valid normalized output WholeCount CE. The selected best checkpoint is then frozen and evaluated with three readouts:

1. **Output WholeCount** — primary fully-spiking metric.
2. **Hidden WholeCount + Linear** — sum valid RSNN hidden spikes to one 128-D vector, then fit a post-hoc linear probe.
3. **Uend + Linear** — causal valid-endpoint RSNN membrane state, one 128-D vector, then fit a post-hoc linear probe.

The two post-hoc probes now use:

```text
train-only StandardScaler -> balanced LogisticRegression(lbfgs)
```

The scaler and classifier are fit only on training-user frozen representations. Validation/test are transform/evaluation only. Standardization does not add nonlinear or temporal access; the overall classifier remains affine in the original feature vector.

## Baseline reuse

`beta0` is **not retrained**. Exp4.0.6 loads the 15 Raw64 Exp4.0.5 checkpoints (3 spike-cap variants x 5 seeds), recomputes all metrics, and refits the two probes with the corrected train-only standardization. This makes the beta0 probe comparison numerically consistent with the new conditions while preserving the original SNN training.

## Run matrix

New training:

```text
3 Local-memory conditions x 3 spike-cap variants x 5 seeds = 45 runs
```

Frozen baseline probe/evaluation:

```text
1 beta0 condition x 3 spike-cap variants x 5 seeds = 15 tasks
```

Task mapping for the 45-run training array is deterministic:

```text
0-14   : shift4   (binary, multi_h, multi_ho; each with seeds 11,23,37,53,71)
15-29  : shift34
30-44  : shift234
```

The 15 baseline tasks are:

```text
0-4   : beta0 binary
5-9   : beta0 multi_h
10-14 : beta0 multi_ho
```

## Main paired comparisons

For every spike-cap mode and readout:

- `shift4 - beta0`: does approximately-250-ms local membrane accumulation rescue Raw64?
- `shift34 - beta0`: does medium+slow heterogeneous memory help further?
- `shift234 - beta0`: does fast+medium+slow heterogeneous memory help further?
- `shift34 - shift4`: heterogeneous medium scale vs single slow scale.
- `shift234 - shift34`: value of adding the fast scale.

Within every Local-memory condition, also report:

- `multi_h - binary`
- `multi_ho - binary`

## Multi-CPU execution

The repository default one-core-per-independent-run rule is used.

Training array:

```bash
#SBATCH --array=0-44%45
#SBATCH --cpus-per-task=1
```

Frozen baseline array:

```bash
#SBATCH --array=0-14%15
#SBATCH --cpus-per-task=1
```

Every compute task initializes Conda locally and pins common BLAS/OpenMP thread counts to one. Each training task performs train -> best checkpoint -> evaluation -> scaled probes -> per-run artifact. The finalizer is submitted with an `afterok` dependency on both arrays and only aggregates existing artifacts; it never fills in missing runs.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_6_cpu.bash
```

Finalized outputs are written under:

```text
notebooks/artifacts/experiment_4_0_6_local_mem_raw64/raw64_local_mem_shift_ablation_v1/
```

Primary files are `runs.csv`, `summary.csv`, `paired_effects.csv`, and `paired_effects_summary.csv`.
