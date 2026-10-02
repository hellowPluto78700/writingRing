# CoreBenchmark extension: output residual-leakage sweep

This directory is an **extension** of CoreBenchmark v1. It does not add cases to `core_benchmark_v1.protocol.runs()`, change the Core protocol version, or alter existing CoreBenchmark semantics.

## Scientific question

Does leakage of evidence that has not yet been emitted through a cap-1 output spike hurt classification, and does that downstream leakage pressure change how L1/L2 organize and communicate class information?

The hidden backbone is locked to `30 -> 128 (2,3,4) -> 128 (2,3,4) -> 12`, bias-free throughout. For output evidence `e[t] = W_out z_L2[t]`:

`Upre[t] = beta * U[t-1] + e[t]`

`s[t] = 1[Upre[t] >= theta]`

`U[t] = Upre[t] - theta * s[t]`

The membrane is signed and unclamped, reset is immediate/subtractive, spike cap is one, theta is 0.5, and there is no output synaptic filter. Intermediate classifier trajectory values are only `theta*s[t]`. For each sample, only its own final valid timestep receives `+U[T]`.

Thus `A[T] = theta * sum_t s[t] + U[T]`. At `beta=1`, this telescopes exactly to `sum_t e[t]`, so WCCE beta=1 is the exact accumulator anchor. For beta<1, emitted spike evidence is retained while residual evidence waiting in U is subject to leakage.

## Formal matrix

- beta: `0.0, 0.1, ..., 1.0` (11 values)
- objective: `wcce`, `tsce`
- seeds: `11, 23, 37`
- total: **66 paired training runs**

All runs reuse the Core production dataset contract, locked user split, Core `Protocol`, Core `BenchmarkNet`, and Core paired initialization/data-loader streams. No extension setting is inserted into Core's protocol fingerprint or run matrix.

### WCCE

`CE((theta*sum(s) + U[T]) / T, y)`. At beta=1 this is algebraically identical to Core O0 WCCE.

### TSCE

Per-sample valid-timestep mean CE over the output trajectory. Intermediate valid timesteps contain `theta*s[t]`; only the final valid timestep contains `theta*s[T] + U[T]`. This is intentionally not Core O1, whose TSCE acts on analog instantaneous evidence.
## Checkpoint selection

Training minimizes only the training objective. Checkpoints maximize validation native BA, break BA ties with lower extension-objective validation CE, then retain the earliest exact tie. Test data and probes never select checkpoints.

## Primary evaluation

Each selected checkpoint reports native spike+final-residual BA, spike-only BA, analog counterfactual BA from `sum_t Wz[t]` without retraining, prediction disagreements, endpoint residual magnitude, signed-leakage drift magnitude, and output backlog fractions above 1/2/4 thresholds.

The implementation retains the identity `analog_score - native_score = leakage_drift`. Signed leakage drift is not interpreted as a nonnegative information loss.

## Representation probes

The complete Core probe suite is reused: L1/L2; spike and pre-reset states; WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled; true no-bias and affine decoders; train-only scale normalization; the Core C grid; validation-only C selection; and the Core shuffle semantics.

Primary representation summaries are L2 spike/no-bias WholeCount, Fixed250 ordered, Fixed250 order-minus-shuffle, Relative10 ordered, and corresponding L1-to-L2 changes.

## Communication diagnostics

For L1/L2 the extension saves firing-rate distributions, fractions below 1/5 Hz and above 10/20/30 Hz, and sustained-run mean/median/P90/max plus fractions of spikes in runs >=2/4/8/16 timesteps.

Train-rate-defined high30 and low30 groups receive no-bias WholeCount, Fixed250 ordered, and Relative10 ordered probes at L1 and L2.

Pruning removes top/bottom 10/20/30% L2 neurons by training-user firing rate and evaluates both fixed-W damage and frozen-backbone output-W retraining. In Slurm execution it shares the same postprocess task as probes/diagnostics rather than using a second array.

A selected-checkpoint gradient diagnostic bins `||dL/de[t]||` into ten normalized sequence-time bins.

## Execution

Default production root:

`core_benchmark_v1/results/output_residual_leakage_v1/`

Commands:

```bash
python -m core_benchmark_v1.extensions.output_residual_leakage plan
python -m core_benchmark_v1.extensions.output_residual_leakage prepare
python -m core_benchmark_v1.extensions.output_residual_leakage run --task-id 0
python -m core_benchmark_v1.extensions.output_residual_leakage postprocess --task-id 0
python -m core_benchmark_v1.extensions.output_residual_leakage finalize
```

The Slurm pipeline uses one preparation job, a 66-task training array, one dependent 66-task combined postprocess array (probes/diagnostics followed by pruning/W-retraining), and a finalizer that only consumes artifacts and fails closed on missing runs. Postprocess is stage-resumable: existing `analysis_complete.json` or `pruning_complete.json` markers skip the corresponding completed stage.