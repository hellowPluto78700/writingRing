# CoreBenchmark extension: fully-drained positive-spike output

This is a sibling extension of CoreBenchmark v1. It does not modify the canonical Core run matrix or the existing `output_residual_leakage` extension.

## Question

How much of end-to-end spiking-readout performance is limited by cap-1 output serialization within the valid sequence, rather than by the absence of discriminative evidence in the endpoint output state?

During each sample's valid window, output dynamics exactly follow Core `R_LIF_E2E`:

[
e_t=W_{out}z_t,quad 	ilde U_t=eta U_{t-1}+e_t,quad
s_t=1[	ilde U_tge	heta],quad U_t=	ilde U_t-	heta s_t.
]

Spikes are positive-only and cap-1. After the sample's own final valid timestep, the backbone stops completely and no new evidence is admitted. The output state is then serialized with `beta_drain=1` and zero input until no positive unit remains supra-threshold. Negative residual never emits a spike.

The classifier is spike-only:

[
C=C_{valid}+C_{drain},qquad
L=CE(C/T,y),
]

using the exact same valid-length denominator and logit scaling as Core `R_LIF_E2E`. No endpoint residual is added to the logits.

## Formal matrix

- beta: `0.0, 0.1, ..., 1.0`
- seeds: `11, 23, 37`
- objective: WCCE only
- 33 total drain-trained runs
- max drain safety cap: 1024 serialization iterations
- strict reference: existing Core `R_LIF_E2E beta=0.5`; it is not retrained here

## Evaluation

Every selected checkpoint reports both:

- drained spike-only BA: `(C_valid + C_drain) / T`
- valid-only ablation BA: `C_valid / T`

and diagnostics including drain gain, valid/drain spike counts, drain spike fraction, required serialization depth (mean/median/P90/P99/max), positive and negative endpoint residual magnitudes, and L1/L2 firing-rate statistics.

The full standard Core representation probe suite is run on L1/L2 traces.

## Selection

Checkpoints are selected by validation drained BA, then lower validation drained mean-logit CE, then earliest epoch. Valid-only BA, test metrics, probes, and drain diagnostics never select checkpoints.

## Execution

Default root:

`core_benchmark_v1/results/output_spike_drain_v1/`

```bash
python -m core_benchmark_v1.extensions.output_spike_drain plan
python -m core_benchmark_v1.extensions.output_spike_drain prepare
python -m core_benchmark_v1.extensions.output_spike_drain run --task-id 0
python -m core_benchmark_v1.extensions.output_spike_drain postprocess --task-id 0
python -m core_benchmark_v1.extensions.output_spike_drain finalize
```

Slurm is `prepare -> 33 train array -> 33 postprocess array -> finalize`, with one-hour walltimes.
