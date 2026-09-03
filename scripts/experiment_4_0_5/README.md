# Experiment 4.0.5 — Temporal resolution vs event capacity

## Question

For the same hierarchical SNN architecture and the same physical 250 ms recurrent/output membrane time constant, how much of the decoding gap is caused by coarse temporal discretization, and how much is recovered by genuine within-Fixed250 timing?

## Architecture

All runs use:

```text
30 event channels
  -> Local128, beta=0, no learned recurrence
  -> RSNN128, tau_mem=250 ms in physical time
  -> 12 output neurons, tau_mem=250 ms in physical time
```

No width/depth/tau sweep is part of this experiment.

## Input representations

### `fixed250`

Current Exp4 representation. The 30 weighted event channels are summed inside each 250 ms bin. One Fixed250 vector is one SNN timestep. Within-bin timing is discarded.

### `repeat4`

The exact same scaled Fixed250 vector is deterministically split into four equal 62.5 ms microsteps:

```text
z_b -> [z_b/4, z_b/4, z_b/4, z_b/4]
```

The four microsteps sum exactly to the original Fixed250 vector. This does not restore within-bin timing and does not multiply total input event mass. It only increases internal temporal/firing opportunities.

### `raw64`

The original 64 Hz, 30-channel weighted event sequence is used directly. This restores genuine within-Fixed250 event timing and therefore contains more temporal information than the other two representations.

## Shared scaling

One zero-preserving per-channel scale is fit from **training users + valid Fixed250 bins only**. No mean subtraction is used. The same scale is applied to `fixed250`, `repeat4`, and `raw64`.

Thus raw event mass aggregated after scaling matches the scaled Fixed250 event mass, apart from numerical tolerance.

## Physical-memory matching

The recurrent and output layers always use physical `tau_mem=250 ms`. Their per-step beta is recomputed from each representation's timestep:

```text
fixed250: dt = 250 ms
repeat4:  dt = 62.5 ms
raw64:    dt = 15.625 ms
beta = exp(-dt / 250 ms)
```

This keeps passive membrane decay aligned in physical time. Learned recurrent state transitions occur more often at higher temporal resolution; that is an intentional part of the temporal-resolution comparison.

The Local layer remains `beta=0` for all representations so it only converts the current input vector into a local population event code.

## Event-cap variants

Only three variants are trained:

| Variant | Hidden cap | Output cap |
|---|---:|---:|
| Binary | 1 | 1 |
| Multi-H | 31 | 1 |
| Multi-HO | 31 | 31 |

All use the same MacroMultiSpikeLIF formulation and subtractive reset.

## Run matrix

```text
3 input representations
x 3 event-cap variants
x 5 seeds (11, 23, 37, 53, 71)
= 45 independent SNN training runs
```

Array ordering is representation-major, then variant, then seed:

```text
0-4    fixed250 / binary
5-9    fixed250 / multi_h
10-14  fixed250 / multi_ho
15-19  repeat4 / binary
20-24  repeat4 / multi_h
25-29  repeat4 / multi_ho
30-34  raw64 / binary
35-39  raw64 / multi_h
40-44  raw64 / multi_ho
```

## Training objective

Every SNN is trained once using valid normalized output WholeCount CE:

```text
CE(sum_valid(S_out / output_cap), y)
```

Checkpoint selection uses validation valid-count balanced accuracy, with validation loss as tie-breaker.

## Readouts from the same frozen checkpoint

Three readouts are evaluated for every run:

1. **Output WholeCount** — production-style fully-spiking result.
2. **Hidden WholeCount + Linear** — sum RSNN hidden spikes over valid time, then train a balanced LogisticRegression on the resulting 128-D vector. It has no phase-slot access.
3. **Uend + Linear** — take the RSNN post-reset membrane at the causal valid endpoint, then train the same balanced LogisticRegression. It has no phase-slot access.

The two Linear probes are post-hoc diagnostics only and never participate in SNN training/checkpoint selection.

## Main paired effects

- `repeat4 - fixed250`: effect of more SNN temporal opportunities **without adding information**.
- `raw64 - repeat4`: additional effect of genuine within-bin timing.
- `multi_h - binary` and `multi_ho - binary` within each representation: dependence on per-step event capacity.
- `Uend+Linear - Output WholeCount`: state-to-spike/readout loss.

## Diagnostics

Per-run evaluation records:

- events/neuron/timestep;
- events/neuron/second;
- fraction zero / one / >1 / >=4;
- cap-hit fraction;
- pre-reset membrane mean/max;
- valid/tail event fractions;
- mean valid timesteps and physical duration;
- theoretical WholeCount evidence levels.

Use rates per physical second, not only per timestep, when comparing the three temporal resolutions.

## Multi-CPU execution

Each independent SNN run is one Slurm array task, one CPU core per task:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_5_cpu.bash
```

The array is `0-44%45`. Every task performs train -> best checkpoint -> all three readouts -> per-run artifact. The finalizer runs with `afterok` and only aggregates existing artifacts; it never retrains or fills missing runs.

Artifacts are written under:

```text
notebooks/artifacts/experiment_4_0_5_temporal_resolution_event_capacity/
  hierarchical_temporal_resolution_event_cap_v1/
```
