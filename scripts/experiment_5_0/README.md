# Experiment 5.0 — Local-evidence supervision scale × event capacity

## Scientific question

With the strongest Exp3-style local SNN backbone fixed, which temporal supervision scale best learns local class evidence that can be consumed by a final hardware-friendly WholeCount readout, and how much does hidden/output multi-event capacity matter?

The experiment intentionally excludes recurrence and long membrane memory. It studies a local-evidence architecture:

```text
Raw64 30-channel weighted events
  -> L1: 128 Synaptic-LIF, shifts (2,3,4)
  -> L2: 128 Synaptic-LIF, shifts (2,3,4)
  -> 12 spiking class-support neurons
  -> valid WholeCount + argmax
```

L1/L2 neurons are divided as evenly as possible across shifts `(2,3,4)`. At 64 Hz those shifts correspond to approximately 54/117/242 ms synaptic time constants. The membrane time constant stays at the Exp3 value, `22.54 ms`; threshold is `0.5`; reset is subtractive.

There is no Fixed250/Fixed500/Relative10 pooling in the inference path. Those partitions are used only by selected training objectives and frozen diagnostic probes.

## Training objectives

All objectives supervise the same 12 output-spike trajectory and every checkpoint is selected using the same deployment metric: validation Output WholeCount balanced accuracy, tie-broken by validation normalized WholeCount CE.

1. `whole_count_ce`
   - normalized valid output evidence over the complete gesture;
   - one CE per gesture.
2. `timestep_ce`
   - one CE for every valid raw 64-Hz timestep;
   - strongest local supervision.
3. `fixed250_bin_ce`
   - valid sequence partitioned into fixed 250-ms windows (16 samples at 64 Hz);
   - one shared class interpretation for every bin;
   - partial terminal bins are duration-weighted.
4. `fixed500_bin_ce`
   - identical construction with fixed 500-ms windows (32 samples at 64 Hz).
5. `relative10_bin_ce`
   - each valid gesture is divided into ten relative-progress bins;
   - valid endpoint is used only to construct the training loss, never as SNN input.

For bin-level objectives, the mean normalized output event evidence inside each bin is used before CE. Bin losses are weighted by the number of valid samples in that bin, so a short partial terminal bin cannot receive the same weight as a complete physical-time bin.

A fixed training-only logit gain is shared by every condition so binary and multi-event outputs have comparable CE scale. Final inference remains raw valid WholeCount + argmax, for which positive scalar normalization does not change the prediction.

## Event-cap variants

The cap definitions match the established Exp4 convention:

| Variant | Hidden cap | Output cap | Interpretation |
|---|---:|---:|---|
| `binary` | 1 | 1 | pure binary communication |
| `multi_h` | 31 | 1 | multi-event hidden communication, binary output |
| `multi_ho` | 31 | 31 | multi-event hidden and output communication |

All variants share the same explicit synaptic and membrane state equations. Only the number of communicable threshold crossings per raw timestep changes.

## Run matrix

```text
5 objectives × 3 event-cap variants × 5 seeds = 75 independent runs
```

Seeds:

```text
11, 23, 37, 53, 71
```

For a given seed, all 15 objective/variant conditions use paired model initialization and train-loader random streams.

## Frozen readout matrix

Every selected checkpoint is frozen and evaluated with:

1. `Output WholeCount` — **primary deployment metric**, no Linear.
2. `L1 WholeCount + Linear` — 128D orderless L1 spike-count probe.
3. `L2 WholeCount + Linear` — 128D orderless L2 spike-count probe; primary hidden local-evidence diagnostic.
4. `L2 Fixed250 ordered + Linear` — ordered 250-ms L2 spike-count bins flattened; absolute coarse temporal position is visible.
5. `L2 Relative10 ordered + Linear` — ordered relative-progress L2 spike-count bins flattened; normalized phase is visible.
6. `L2 Uend + Linear` — only the 128D L2 membrane state at the valid endpoint timestep; no trajectory is exposed directly.

All probes are train-user `StandardScaler -> balanced LogisticRegression(lbfgs)`. Validation/test are transform/evaluation only.

The matrix separates three bottlenecks:

```text
L2 phase-aware Linear
  -> remove explicit temporal position
L2 WholeCount + Linear
  -> compress 128 hidden feature channels to 12 class-support outputs
Output WholeCount
```

## Diagnostics

For L1, L2, and output populations, every split records:

- valid events per neuron per second;
- tail-event fraction after the valid endpoint.

These metrics quantify event cost and whether a purportedly local network develops excessive persistence.

## Multi-CPU execution

This experiment follows `AGENTS.md` exactly:

```text
75 independent conditions
  -> Slurm array 0-74%50
  -> one CPU core per task
  -> train -> select best checkpoint -> frozen readouts/diagnostics -> per-run JSON
all 75 tasks succeed
  -> one afterok finalizer
  -> runs/summary/paired-effect CSVs + manifest
  -> analysis-only notebook
```

Every compute task initializes Conda locally, prefers `writingring-gpu`, falls back to `writingring-viz`, disables CUDA, and pins BLAS/OpenMP thread counts to one.

## Submit

From repository root:

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_5_0_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_5_0_cpu.bash
```

Monitor with:

```bash
squeue -u $USER
```

## Artifacts

Finalized outputs are written to:

```text
notebooks/artifacts/
  experiment_5_0_local_evidence_objectives/
    local_objective_x_event_capacity_v1/
```

Primary files:

- `runs.csv`
- `summary.csv`
- `paired_loss_effects.csv`
- `paired_loss_effects_summary.csv`
- `paired_variant_effects.csv`
- `paired_variant_effects_summary.csv`
- `manifest.json`

The notebook `notebooks/experiment_5_0_local_evidence_objectives.ipynb` is analysis-only and never trains or launches Slurm jobs.
