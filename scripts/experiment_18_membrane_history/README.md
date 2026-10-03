# Exp18 — consumable multi-tau membrane history under WCCE

## Scientific question

Exp18 asks whether the CoreBenchmark O0 temporal poles work better for native WholeCount/WCCE classification when the slow history pole is moved from an unreset synaptic-current state into the resettable membrane state.

The experiment is deliberately minimal. Dataset, locked cross-user split, seeds, width, connectivity, threshold, surrogate, WCCE reduction, optimizer, checkpoint selection, bias-free accumulator and probe protocol are inherited from CoreBenchmark v1.1 O0.

## Locked reference and pole swap

The reference is the already-finalized CoreBenchmark O0 checkpoint for seeds 11/23/37:

```text
I_REF:
I[t]     = alpha_slow I[t-1] + W x[t]
Upre[t]  = beta_fast U[t-1] + I[t]
U[t]     = Upre[t] - theta s[t]
```

with multi-tau shifts (2,3,4) in both hidden layers and the CoreBenchmark 22.54 ms fast membrane pole.

Exp18 swaps the two linear poles:

```text
U_NORMAL / U_DETACH:
I[t]     = beta_fast I[t-1] + W x[t]
Upre[t]  = alpha_slow U[t-1] + I[t]
```

Before any spike/reset, the two cascaded linear filters have the same transfer function because the two poles are only reordered. The intervention therefore targets which state is reset/consumed by spiking rather than introducing a new temporal pole.

## Cases

- `I_REF`: no retraining; reuse finalized CoreBenchmark O0 artifacts.
- `U_NORMAL`: forward reset `U = Upre - theta*s`; backward differentiates through the surrogate spike in the reset branch.
- `U_DETACH`: identical forward reset, but `U = Upre - theta*stopgrad(s)`. The spike/evidence branch remains differentiable; only the reset branch is detached.

The formal training matrix is 2 cases x 3 seeds = 6 new runs. All cases use L1/L2 width 128, shifts (2,3,4)/(2,3,4), bias-free accumulator WCCE, seeds 11/23/37, and the CoreBenchmark locked user split.

## Checkpoint and evaluation contract

Epoch 0 is a candidate. Selection is validation native accumulator BA, then validation valid-mean-logit CE, then earliest epoch. Test metrics and probes never select a checkpoint.

Every U-history checkpoint runs the canonical CoreBenchmark probes at L1/L2 for spike and pre-reset state: WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled, and no-bias/affine decoders with the canonical validation-only C search and shuffle seeds.

The reference metrics/probes are read from O0 rather than recomputed or retrained.

## Mechanism diagnostics

For I_REF, U_NORMAL and U_DETACH, diagnostics run on the validation split only and never update weights.

Two losses are differentiated:

1. ordinary valid-mean WCCE;
2. a diagnostic-only last-16-valid-step (250 ms) suffix CE.

The suffix loss removes direct early-spike credit, so early-state gradients arise through future computation. Gradients are aggregated by layer, tau group and the number of future spikes/reset events. This directly tests whether normal reset differentiation introduces reset-count-dependent attenuation and whether detaching the reset branch restores the state-gradient path.

Activity is also reported per tau group (43/43/42 neurons): firing Hz, mean absolute pre-reset state and pre-reset/threshold ratio conditioned on firing. Bursting is descriptive, not a success criterion.

## Interpretation contract

- U_NORMAL > I_REF with preserved probe information supports consumable membrane history as a better WCCE organization mechanism.
- U_DETACH > U_NORMAL together with a longer suffix-credit horizon supports a backward reset-gradient bottleneck.
- U_NORMAL approximately U_DETACH > I_REF points to a forward consumption benefit rather than a reset-gradient benefit.
- Both U cases poor despite healthy gradient diagnostics motivates a later shallow-U/deep-I hybrid rather than another loss tweak.
- Strong temporal probes but weak native BA indicate organization/readout geometry failure rather than simple information loss.

No tau sweep, threshold sweep, width sweep, Prefix/TSCE objective, firing regularizer, hybrid architecture or partial reset-gradient sweep belongs to Exp18.0.

## Artifacts

Default root:

`notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1/`

Each U run writes initial/checkpoint/history/native/traces and the full CoreBenchmark probe artifacts. Diagnostics are under `diagnostics/<case>__seed<seed>/gradient_diagnostics.json`. Finalization writes `aggregate.json` and fails if a required run or I_REF diagnostic is missing.

## Slurm

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_18_cpu.bash
```

The DAG is:

```text
prepare
  -> 6-task train/evaluate/diagnose array
  -> 3-task I_REF diagnostic array
  -> finalize
```

Each independent run is one CPU task. The training task immediately evaluates its own selected checkpoint. The I_REF diagnostic tasks never retrain O0.
