# Exp0.2 Shift-Resolved Firing-Rate Analysis — TaskSpec

## Goal

Add a checkpoint-only post-evaluation for Exp0.2 that directly measures whether regularization preferentially suppresses long-\(\tau_{syn}\) neurons rather than merely reducing firing uniformly across all synaptic timescales.

## Scientific outputs

For every Exp0.2 evaluated checkpoint (72 new + 18 frozen Exp0.1 baselines), measure the final hidden layer grouped by configured synaptic shift:

- valid-region firing rate, Hz/neuron
- tail stage 1 firing rate: endpoint to ~200 ms
- tail stage 2 firing rate: ~200–400 ms
- tail stage 3 firing rate: ~400–600 ms
- full 600-ms tail firing rate, Hz/neuron
- corresponding spike counts per neuron
- relative change versus the matched frozen Exp0.1 `none` checkpoint for the same architecture/objective/seed/shift
- tail-to-valid firing-rate ratio for each shift

The analysis must use exactly the same data split, endpoint-relative zero-input rollout, checkpoint selection, shift grouping and 64-Hz timing conventions as Exp0.2.

## Execution contract

- Checkpoint-only: no training and no optimizer state changes.
- 90 independent evaluation tasks: 3 architectures x 2 objectives x 5 profiles x 3 seeds.
- One Slurm array task per checkpoint, one CPU core per task, concurrency capped at 50.
- Each task writes one per-checkpoint CSV under the Exp0.2 artifact tree.
- A dependent single-CPU finalizer only aggregates existing per-checkpoint CSVs; it never loads models or evaluates data.
- The notebook is analysis-only and reads finalized CSVs.

## Aggregates

Finalizer must emit:

- `shift_fire_rate_summary.csv`: long form, one row per architecture/objective/profile/seed/shift
- `shift_fire_rate_comparison.csv`: mean/std across seeds per architecture/objective/profile/shift
- `shift_fire_rate_vs_none.csv`: paired deltas and percentage changes versus frozen Exp0.1 `none`
- `shift_fire_rate_long_tau_summary.csv`: pooled long-timescale summary, using the shifts actually present in the final hidden layer and explicitly identifying the long group
- `shift_fire_rate_manifest.json`

## Long-timescale grouping

Do not invent shifts absent from a backbone. Report every configured shift independently. For pooled summaries, define the long group from the final hidden layer as shifts `>=5` where available; also retain per-shift results so conclusions never depend on the pooled definition.

## Acceptance criteria

- Exactly 90 per-checkpoint task specs are generated and are unique.
- Final-hidden neuron slices exactly match Exp0.2's existing shift grouping helper.
- Firing-rate denominators are neuron-seconds, not raw timestep counts; 64-Hz discrete stage lengths are recorded in the manifest.
- Valid firing excludes padding and all post-end rollout samples.
- Tail firing excludes valid samples and is endpoint-relative for every sample.
- `none` rows are reused as the paired baseline and are never retrained.
- Finalizer fails loudly on missing/duplicate task artifacts.
- Slurm runner requests one CPU and sets BLAS/OpenMP thread counts to one.
- Notebook does not train, spawn multiprocessing, or evaluate checkpoints.
