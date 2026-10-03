# Exp18.2 — History Carrier × Loss Geometry

## Scientific question

Exp18.2 tests whether the high-rate solution observed under WholeCount cross-entropy is primarily driven by loss geometry or by the hidden history carrier.

The conceptual 2 x 3 design is:

| History carrier | WCCE | Normalized-WCCE | Margin-WCCE |
| --- | --- | --- | --- |
| I / I | CoreBenchmark O0 reference | trained | trained |
| U / U | Exp18 U_NORMAL reference | trained | trained |

Only the four non-WCCE cells are newly trained. With seeds 11/23/37 this is 12 formal runs.

The experiment inherits the Exp18/CoreBenchmark contract: locked dataset and user split, 30 event channels, two 128-neuron hidden layers, multi-tau shifts (2,3,4)/(2,3,4), threshold 0.5, surrogate slope 25, one spike per neuron per timestep, bias-free accumulator head, Adam lr 0.001, no weight decay, batch 128, max 100 epochs, min 20, patience 30, and checkpoint selection by native validation BA then native validation mean-logit CE then earliest epoch.

## Carriers

I-carrier layers are identical to CoreBenchmark O0:

```text
I[t]    = alpha_slow I[t-1] + W x[t]
Upre[t] = beta_fast U[t-1] + I[t]
U[t]    = Upre[t] - theta s[t]
```

U-carrier layers are identical to Exp18 U_NORMAL:

```text
I[t]    = beta_fast I[t-1] + W x[t]
Upre[t] = alpha_slow U[t-1] + I[t]
U[t]    = Upre[t] - theta s[t]
```

No detached reset is used in Exp18.2.

## Losses

### WCCE reference

```text
C = sum_t s[t]
z = R C / T_valid
L = CE(z, y)
```

No WCCE reference is retrained.

### Normalized-WCCE

For L2 valid-window spike counts:

```text
C = sum_t s[t]
C_hat = C / ||C||_1
z = gamma R C_hat
L = CE(z, y)
```

For samples with zero total count, `C_hat=0`.

The denominator is differentiable and is not detached. Therefore uniform radial scaling of the count vector does not directly lower the classification loss.

`gamma` is fixed once per carrier x seed before training. It is calibrated on the epoch-0 training split only, without labels, by matching RMS logit scale:

```text
gamma = RMS(raw WCCE logits) / RMS(R C_hat)
```

The calibration is numerical only: it does not inspect labels, validation metrics, or test data.

### Margin-WCCE

Native logits remain:

```text
z = R C / T_valid
```

with multiclass margin:

```text
m = z_y - max_{k != y} z_k
L = max(0, 1 - m)
```

The target margin is fixed to 1.0 and is not swept. Once the margin exceeds 1, further count amplification receives no direct loss reward.

## References

- `I_WCCE_REF`: finalized CoreBenchmark O0 artifacts, reused exactly.
- `U_WCCE_REF`: finalized Exp18 `U_NORMAL` artifacts, reused exactly.

`prepare` locks checkpoint/native/probe/trace hashes for both references and fails if any required artifact is missing. The finalizer rechecks those hashes.

## Required per-run outputs

Each of the 12 new formal runs performs in one Slurm array task:

```text
train
 -> checkpoint selection
 -> native train/val/test evaluation
 -> canonical CoreBenchmark probes
 -> per-layer/per-tau firing diagnostics
 -> count/norm diagnostics
 -> complete.json
```

Required run artifacts:

- `loss_config.json`
- `initial.pt`
- `checkpoint.pt`
- `history.json`
- `native.json`
- `activity.json`
- `traces.npz`
- `probes.json`
- `probe_search.json`
- `probe_decoders.npz`
- `probe_predictions.npz`
- `complete.json`

The run is not complete merely because Slurm reports COMPLETED.

## Diagnostics

Primary outcome groups are:

1. native train/val/test BA;
2. L2 firing-rate distribution: mean, P90, P95, fraction >=10 Hz, fraction >=20 Hz;
3. L2 count magnitude: mean total count and mean count L2 norm;
4. L2 spike/no-bias WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled;
5. Relative10-native and Fixed250-native gaps;
6. Fixed250 and Relative10 order-shuffle gaps;
7. Frobenius norms of L1, L2, and accumulator-head weights.

These distinguish a reduction in firing pressure from a simple transfer of scale pressure into weight norms.

## Primary contrasts

For each carrier and seed:

```text
NWCCE - WCCE
MWCCE - WCCE
```

For each new loss:

```text
(U loss effect) - (I loss effect)
```

The latter is the carrier-by-loss interaction.

Interpretation is descriptive and paired across the three locked optimization seeds.

## Execution contract

Recommended concurrency: 12 CPU tasks.

Exact Slurm DAG:

```text
compute-node smoke
  -> prepare
      -> 12-task formal train/evaluate/probe/activity array
          -> finalize
```

No separate reference jobs are needed because both WCCE references already contain the required native/probe/trace artifacts.

All CPU jobs use:

`scripts/bash_script/SNN_Bash/slurm_cpu_env.bash`

Formal submit command:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_18_2_cpu.bash
```

The smoke job loads the real locked Core/Exp18 artifacts, calibrates Normalized-WCCE for both carriers, and runs forward/loss/backward/optimizer/checkpoint/probe-path checks for all four formal cases.

## Resume and provenance

`prepare` locks the Exp18.2 source hash, dependency source hashes, CoreBenchmark identity/hashes, Exp18 protocol hash, reference hashes, formal case mapping, carrier mapping, loss mapping, and fixed margin target.

Resume rules:

- valid `complete.json` + matching provenance -> skip;
- valid checkpoint + matching provenance + matching `loss_config.json` -> skip training and continue evaluation;
- changed source/dependency/core identity/case/seed/carrier/loss -> fail rather than silently reuse.

Any implementation change after `prepare` requires a fresh results directory or deliberate protocol-lock regeneration.

## Non-goals

Exp18.2 does not add firing regularization, weight normalization, tau/threshold/width sweeps, Prefix/TSCE, detached reset, hybrid UI/IU carriers, or per-neuron saturating readouts. Those would confound the carrier-by-loss test.
