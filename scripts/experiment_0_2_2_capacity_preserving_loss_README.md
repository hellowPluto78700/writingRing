# Exp0.2.2 — Capacity-Preserving Anti-Persistent-Firing Loss

This experiment tests whether the final hidden `s6/s7` populations can stop behaving like persistent-firing units without solving the regularization objective by globally silencing the long-timescale neurons.

## Fixed protocol

- Backbones: `mid_long`, `short_mid_long`
- Objective: `whole_count_ce`
- Seeds: `11, 23, 37`
- Binary hidden/output spikes, same Exp0.2 dynamics and readout
- Total training budget: 50 epochs
- Shared WC-only warmup: epochs 1–5
- Condition-specific training: epochs 6–50
- Long population: final-hidden shifts `s6` (~0.99 s) and `s7` (~1.99 s)

## Conditions

| condition | regularizer |
|---|---|
| `wc_only` | none |
| `sat` | sustained-firing occupancy penalty |
| `relative_tail` | endpoint tail / valid firing-ratio penalty |
| `sat_relative_tail` | saturation + relative tail |
| `sat_relative_tail_capacity` | saturation + relative tail + warmup-relative capacity floor |

The regularizers never touch `s2-s5` directly and never change the SNN recurrence equations.

## Healthy reference

A single preprocessing job loads the frozen strong `short_mid + whole_count_ce + none` reference checkpoints for seeds 11/23/37 and evaluates **train split only**.

It derives:

- `rho_max`: P95 of 250-ms causal occupancy over final-hidden `s4/s5` neurons.
- `gamma[0:3]`: P90 of healthy tail/valid firing-rate ratio for endpoint-relative 0–200, 200–400 and 400–600 ms stages.

The resulting file is:

```text
notebooks/artifacts/experiment_0_2_2_capacity_preserving_loss/
  capacity_preserving_loss_v1/reference/healthy_reference.json
```

No validation or test samples are used to fit these targets.

## Losses

For a long neuron `j`, define 250-ms causal occupancy

```text
q_j(t) = mean spike occupancy over the previous 250 ms.
```

Saturation loss is a hinge penalty only above the healthy reference threshold:

```text
L_sat = mean relu(q_j(t) - rho_max)^2
```

so ordinary sparse/bursty firing is not penalized.

For shift `s in {6,7}`, relative-tail loss uses

```text
R_s,q = FR_tail(s,q) / (FR_valid(s) + eps)
L_relative_tail = weighted mean relu(R_s,q - gamma_q)^2
```

with stage weights `(1,2,4)`. Scaling both valid and tail firing down together therefore does not directly solve the ratio penalty.

The capacity condition additionally uses the matching shared epoch-5 model as a per-training-sample reference:

```text
L_capacity = mean [relu(0.7 * FR_warmup - FR_current) / FR_warmup]^2
```

for each of `s6` and `s7`. Samples whose warmup reference is effectively zero are excluded from this floor.

## Gradient calibration and schedule

For every non-baseline condition, after loading the exact shared epoch-5 checkpoint, regularizer strength is calibrated on five deterministic train batches using only the final-hidden incoming-weight rows assigned to `s6/s7`:

```text
median(||grad L_reg|| / ||grad L_task||) -> target 0.05
```

The resulting `kappa` is fixed for the rest of training. An effectively zero regularizer gradient is a hard error rather than an unbounded rescaling.

Regularization schedule:

```text
epoch 1-5:  scale = 0
epoch 6-15: linear ramp 0.1 -> 1.0
epoch 16-50: scale = 1
```

## Per-epoch metrics

After every epoch, the model is evaluated deterministically with no shuffle on train and validation. History contains:

- train / validation balanced accuracy
- train / validation WholeCount CE
- training optimization task loss
- raw regularizer
- weighted regularizer contribution
- total optimization loss
- saturation / relative-tail / capacity components and weighted components
- train / validation long valid FR, tail FR and tail/valid ratio
- output outgoing-weight norm for final-hidden shifts `s2...s7`
- fraction of output-weight energy assigned to `s6+s7`

The test split is evaluated only once after the validation-selected best checkpoint has been fixed.

## Training-curve figure

Every main run saves one 2x3 figure:

1. Train BA + Val BA
2. Train WholeCount CE + Val WholeCount CE
3. Train optimization task loss + weighted regularizer + total loss **on the same axes**
4. Train/Val long valid FR + long tail FR
5. Train/Val long tail/valid FR ratio
6. `s6`, `s7` outgoing-weight utilization + long output-weight energy fraction

Vertical markers indicate epoch 5, epoch 15 and the selected best epoch.

## Multi-CPU execution

The Slurm pipeline follows repository task-level parallelism:

```text
reference job (1 CPU) -----------+
                                 +--> main array (30 one-core tasks, %30) --> finalizer
warmup array (6 one-core tasks) -+
```

The main array consists of

```text
2 backbones x 5 conditions x 3 seeds = 30 runs
```

and each task performs condition-specific training, train/val diagnostics, validation checkpoint selection, one-time test evaluation and per-run artifact generation.

Run from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_2_2_capacity_preserving_cpu.bash
```

## Final artifacts

The aggregation-only finalizer writes:

```text
summary.csv
comparison_summary.csv
calibration_summary.csv
history_long.csv
shift_history_long.csv
manifest.json
```

The notebook `notebooks/experiment_0_2_2_capacity_preserving_loss.ipynb` is analysis-only.
