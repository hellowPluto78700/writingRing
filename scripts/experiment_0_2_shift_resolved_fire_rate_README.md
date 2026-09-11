# Exp0.2 shift-resolved firing-rate post-evaluation

This analysis directly tests whether Exp0.2 regularizers preferentially suppress long-timescale final-hidden neurons.

It is checkpoint-only: no model is retrained. Every existing Exp0.2 checkpoint and every frozen Exp0.1 `none` checkpoint is evaluated on the Exp0.2 test split with the same endpoint-relative 600-ms zero-input rollout.

For each configured final-hidden synaptic shift, the analysis records:

- `alpha` and the corresponding `tau_syn_ms`
- valid-region firing rate in Hz/neuron
- endpoint-tail stage firing rates for approximately 0–200, 200–400 and 400–600 ms
- full 600-ms tail firing rate in Hz/neuron
- spike counts per neuron/sample
- tail/valid firing-rate ratio
- paired percentage change versus the matching frozen Exp0.1 `none` checkpoint
- `tail_specific_suppression_pp`, defined as `%change(valid FR) - %change(tail FR)`; positive values mean the tail was suppressed more strongly than valid firing

Firing rates are normalized by neuron-seconds, so shift groups with slightly different neuron counts remain comparable.

The pooled `long` diagnostic is defined as configured final-hidden shifts `>=5` where available. This corresponds to the longer final-hidden time constants, while the per-shift CSV remains the primary evidence and should be used for scientific claims.

## Unity / Slurm

Submit the checkpoint-only evaluation with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_2_shift_fire_rate_cpu.bash
```

This launches 90 independent single-core array tasks (`0-89%50`):

```text
3 architectures
x 2 objectives
x 5 profiles (none, a1, a1_p2, tail, a1_tail)
x 3 seeds
= 90 checkpoint evaluations
```

Each task loads exactly one checkpoint, evaluates the test split, and writes one per-checkpoint CSV. A dependent single-core finalizer runs only after the complete array succeeds and only aggregates existing CSVs; it never loads a model or reruns evaluation.

## Outputs

Outputs are written under:

```text
notebooks/artifacts/experiment_0_2_endpoint_tail_regularization/
  endpoint_tail_regularization_v1/
    experiment_0_2_shift_resolved_fire_rate/
      shift_fire_rate_v1/
```

Finalized files:

- `shift_fire_rate_summary.csv` — per architecture/objective/profile/seed/shift
- `shift_fire_rate_comparison.csv` — mean/std across seeds for each shift
- `shift_fire_rate_vs_none.csv` — paired per-shift delta, percentage change, and endpoint-selectivity metrics
- `shift_fire_rate_long_tau_summary.csv` — neuron-count-weighted pooled long-τ rows per seed
- `shift_fire_rate_long_tau_comparison.csv` — long-τ mean/std across seeds
- `shift_fire_rate_long_tau_vs_none.csv` — direct pooled long-τ paired comparison versus frozen `none`
- `shift_fire_rate_manifest.json`

The notebook `notebooks/experiment_0_2_shift_resolved_fire_rate.ipynb` is analysis-only and reads these finalized artifacts.

## Main interpretation

Within the same architecture/objective/seed/shift, compare:

```text
pct_change_valid_firing_rate_hz_vs_none
pct_change_tail_firing_rate_hz_vs_none
tail_specific_suppression_pp
```

Interpretation:

```text
tail_specific_suppression_pp >> 0
    -> tail firing is preferentially suppressed relative to valid firing

tail_specific_suppression_pp ~= 0
    -> regularizer mainly scales valid and post-end firing down together

tail_specific_suppression_pp < 0
    -> valid firing is suppressed more strongly than the tail
```

Then inspect the metric as a function of `shift` / `tau_syn_ms`. If s6/s7 show substantially stronger positive endpoint selectivity than s2/s3, that supports a long-τ-specific improvement. If all shifts move together, the regularizer is acting more like global activity suppression than a fix for long-timescale persistent firing.
