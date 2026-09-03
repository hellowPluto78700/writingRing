# Experiment 4.3 — Stage-2 long-term memory and recurrence validation

## Question

Before optimizing Stage-2 memory parameters, establish two mechanistic facts for the current best Raw64 architecture:

1. Does the Stage-2 state retain discriminative information beyond the approximately 250-ms Local128 integration scale?
2. Is that information materially supported by learned recurrent feedback rather than passive membrane integration alone?

This experiment is deliberately a **validation experiment**, not a tau/topology optimization sweep.

## Fixed architecture

```text
Raw64, 30 scaled weighted-event channels
  -> Local128 shift34
       64 neurons: tau_mem ~= 117 ms
       64 neurons: tau_mem ~= 242 ms
       no local recurrence
  -> Stage2-128, tau_mem = 250 ms
  -> 12 output neurons, tau_mem = 250 ms
```

The event-cap condition is fixed to the current best software diagnostic:

```text
multi_ho: hidden cap = 31, output cap = 31
```

Threshold, input scaling, split, optimizer, training objective, and seeds are inherited from Exp4.0.6.

Local128 is fixed **architecturally** throughout Exp4.3. Its weights are still trained end-to-end in the newly trained FF control, matching the training contract of the current RSNN. Stage-2 interventions never reset or ablate Local128 state.

## Training/reuse policy

Exp4.3 does not retrain the current RSNN. It reuses the five existing Exp4.0.6 checkpoints:

```text
condition = shift34
variant   = multi_ho
seeds     = 11, 23, 37, 53, 71
```

Only five new FF controls are trained. The FF model keeps the same Local128, Stage2 width, passive Stage2 membrane dynamics, output layer, initialization stream, and training protocol. The recurrent contribution

```text
W_rec * S_state[t-1]
```

is functionally disabled.

Therefore the new training cost is:

```text
1 architecture x 5 paired seeds = 5 training runs
```

## V1 — Independently trained FF vs RSNN

Compare, seed-by-seed:

```text
FF:   U[t] = beta U[t-1] + W_in S_local[t]
RSNN: U[t] = beta U[t-1] + W_in S_local[t] + W_rec S_state[t-1]
```

Report for train/val/test:

- Output valid WholeCount BA/Macro-F1
- Hidden valid WholeCount + frozen Linear probe
- Stage-2 valid endpoint membrane `Uend` + frozen Linear probe
- normal firing/tail diagnostics inherited from the Exp4.0.5/4.0.6 evaluator

The paired `RSNN - FF` effect tests whether adding recurrent temporal feedback improves the same Stage-2 task beyond passive membrane persistence.

## V2 — Test-time recurrence ablation

For each frozen trained RSNN checkpoint, run the same samples twice:

```text
normal:        + W_rec S[t-1]
recurrent_off: omit W_rec S[t-1]
```

No weights are refit. The post-hoc probes are also **not refit**. A large drop from `normal` to `recurrent_off`, together with V1, is evidence that the trained model actively depends on recurrent history.

This intervention creates a test-time distribution shift, so it is interpreted jointly with the independently trained FF control rather than by itself.

## V3 — Periodic Stage-2 state reset

At inference, reset only:

```text
Stage2 membrane U_state = 0
previous Stage2 spikes S_state[t-1] = 0
```

at fixed horizons:

```text
250 ms
500 ms
1000 ms
NoReset
```

Local128 is never reset. Output membrane is also not reset, because the intervention is intended to destroy only Stage-2 memory.

The **primary metric is `Uend + frozen Linear`**. Output WholeCount is reported only as a secondary system metric because WholeCount itself is an external accumulator and can retain evidence emitted before a reset.

The same probe fitted on the normal train-user endpoint representation is reused for every reset condition. No ablation-specific probe is trained.

A useful long-memory pattern is expected to look qualitatively like:

```text
Reset250 < Reset500 < Reset1000 <= NoReset
```

Exact monotonicity is not required for every seed, but the paired effect should weaken as the permitted Stage-2 horizon grows.

## V4 — Silent-delay retention

After each sample's true valid endpoint, force input to zero and let the network evolve naturally for:

```text
0, 250, 500, 1000, 2000 ms
```

Then decode Stage-2 membrane state with the exact same `Uend` probe fitted at delay 0.

This is done for both FF and RSNN. It separates passive membrane retention from additional recurrent retention:

```text
BA_FF(delay)
BA_RSNN(delay)
```

The implementation explicitly zeros padded input after each sample's own valid endpoint before adding the requested silent delay, so shorter samples do not accidentally receive nonzero padded content.

## Probe contract

Two probes are fit once per model/seed from **normal training-user frozen representations**:

```text
train-only StandardScaler
  -> balanced LogisticRegression(lbfgs)
```

- Hidden WholeCount probe
- `Uend` probe

Both probes are frozen for V2/V3. The `Uend` probe is also frozen across all V4 delays. Validation and test never participate in fitting or scaling.

## Multi-CPU execution

Exp4.3 follows the repository default: one independent run/evaluation per Slurm array task, one CPU core per task, thread counts pinned to one.

### FF training array

```text
5 tasks: seeds 11,23,37,53,71
one task = train FF -> select best checkpoint -> V1/V3/V4 evaluation -> artifact
```

### Frozen RSNN evaluation array

```text
5 tasks: seeds 11,23,37,53,71
one task = load Exp4.0.6 checkpoint -> V1/V2/V3/V4 evaluation -> artifact
```

The finalizer has an `afterok` dependency on both arrays and only aggregates completed per-seed artifacts.

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_3_cpu.bash
```

## Outputs

Per-run artifacts:

```text
notebooks/artifacts/experiment_4_3_long_term_memory_validation/
  stage2_recurrence_memory_validation_v1/
    checkpoints/       # new FF only
    histories/         # new FF only
    evaluations/       # FF and frozen RSNN validation bundles
```

Finalized tables:

- `classification_runs.csv`
- `classification_summary.csv`
- `retention_runs.csv`
- `retention_summary.csv`
- `paired_effects.csv`
- `paired_effects_summary.csv`
- `manifest.json`

Key paired effects include:

- `rsnn_minus_ff`
- `rsnn_normal_minus_recurrent_off`
- `normal_minus_periodic_reset`
- `silent_delay_drop_from_0ms`

## Interpretation boundary

`Uend > OutputWholeCount` is **not** evidence that recurrence caused the memory. It only indicates that the internal endpoint state contains information not fully exposed by the spiking readout.

The recurrence claim should be based on the combined V1 + V2 evidence. The long-horizon claim should be based primarily on V3 and V4, with `Uend` as the main diagnostic and WholeCount treated as secondary where its accumulation can confound memory attribution.
