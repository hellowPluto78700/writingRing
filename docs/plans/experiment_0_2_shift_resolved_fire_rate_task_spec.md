# Exp0.2 Shift-Resolved Firing-Rate Analysis — TaskSpec

## Goal

Add a checkpoint-only post-evaluation for Exp0.2 that directly measures whether regularization preferentially suppresses long-\(\tau_{syn}\) neurons rather than merely reducing firing uniformly across all synaptic timescales.

## Required behavior

For every Exp0.2 evaluated checkpoint (72 new + 18 frozen Exp0.1 baselines), measure the final hidden layer grouped by the exact configured synaptic-shift slices already used by Exp0.2:

- shift, alpha, and corresponding `tau_syn_ms`
- valid-region firing rate in Hz/neuron
- tail stage 1 firing rate: endpoint to ~200 ms
- tail stage 2 firing rate: ~200–400 ms
- tail stage 3 firing rate: ~400–600 ms
- full 600-ms tail firing rate in Hz/neuron
- corresponding spike counts per neuron/sample
- tail-to-valid firing-rate ratio
- paired delta and percentage change versus the matching frozen Exp0.1 `none` checkpoint
- `tail_specific_suppression_pp = pct_change(valid FR) - pct_change(tail FR)`, where positive values mean preferential post-end suppression

The analysis must use exactly the same data split, endpoint-relative zero-input rollout, checkpoint identity checks, shift grouping and 64-Hz timing conventions as Exp0.2.

## Preserved contracts

- No checkpoint is retrained or modified.
- Frozen Exp0.1 `none` checkpoints remain read-only.
- Valid firing excludes padding and every post-end rollout timestep.
- Tail firing excludes valid samples and is endpoint-relative per sample.
- Firing-rate denominators are neuron-seconds, not raw timestep counts.
- The final-hidden shift slices must call the existing Exp0.2 shift-grouping helper rather than independently redefining neuron ownership.

## Execution contract

- 90 independent evaluation tasks: 3 architectures x 2 objectives x 5 profiles x 3 seeds.
- One Slurm array task per checkpoint, one CPU core per task, concurrency capped at 50.
- Each task writes one per-checkpoint CSV under the Exp0.2 artifact tree.
- A dependent single-CPU finalizer only aggregates existing per-checkpoint CSVs; it never loads models or evaluates data.
- The notebook is analysis-only and reads finalized CSVs.

## Aggregates

Finalizer must emit:

- `shift_fire_rate_summary.csv`: long form, one row per architecture/objective/profile/seed/shift
- `shift_fire_rate_comparison.csv`: mean/std across seeds per architecture/objective/profile/shift
- `shift_fire_rate_vs_none.csv`: paired per-shift deltas, percentage changes, and tail-specific suppression
- `shift_fire_rate_long_tau_summary.csv`: neuron-count-weighted pooled long-timescale rows per seed
- `shift_fire_rate_long_tau_comparison.csv`: pooled long-timescale mean/std across seeds
- `shift_fire_rate_long_tau_vs_none.csv`: pooled paired comparison against `none`
- `shift_fire_rate_manifest.json`

## Long-timescale grouping

Do not invent shifts absent from a backbone. Report every configured shift independently. For pooled summaries, define the long group from the final hidden layer as shifts `>=5` where available; retain per-shift results as the primary scientific evidence.

## Acceptance criteria

- Exactly 90 per-checkpoint task specs are generated and unique.
- Final-hidden neuron slices exactly match Exp0.2's existing shift grouping helper.
- Alpha/tau mapping is monotonic with shift and uses the actual evaluation sampling rate.
- Firing-rate denominators are neuron-seconds; discrete stage lengths are recorded in the manifest.
- `none` rows are reused as the paired baseline and never retrained.
- Positive `tail_specific_suppression_pp` has the documented interpretation that tail firing fell more strongly than valid firing.
- Pooled long-τ rates are weighted by final-hidden neuron counts.
- Finalizer fails loudly on missing/duplicate task artifacts.
- Slurm runner requests one CPU and sets BLAS/OpenMP thread counts to one.
- Notebook does not train, spawn multiprocessing, or evaluate checkpoints.

## Validation

- `python -m pytest -q tests/test_experiment_0_2_shift_resolved_fire_rate_contract.py`
- `python -m pytest -q tests/test_repository_source_syntax.py`

## Replan triggers

Replan only if Exp0.2 checkpoint identity, data split, endpoint rollout, or final-hidden shift allocation changed incompatibly after the result commit used as this analysis baseline.
