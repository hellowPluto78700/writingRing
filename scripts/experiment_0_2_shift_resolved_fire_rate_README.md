# Exp0.2 shift-resolved firing-rate post-evaluation

This analysis directly tests whether Exp0.2 regularizers preferentially suppress long-timescale final-hidden neurons.

It is checkpoint-only: no model is retrained. Every existing Exp0.2 checkpoint and every frozen Exp0.1 `none` checkpoint is evaluated on the Exp0.2 test split with the same endpoint-relative 600-ms zero-input rollout.

For each configured final-hidden synaptic shift, the analysis records valid-region firing rate and the firing rates in the three endpoint-tail stages (approximately 0–200, 200–400 and 400–600 ms), plus total-tail firing rate, spikes per neuron and tail/valid ratios. Firing rates are normalized by neuron-seconds.

The pooled `long` diagnostic is defined as configured final-hidden shifts `>=5` where available, but the per-shift CSV is the primary result and should be used for scientific claims.

## Unity / Slurm

Submit the whole checkpoint-only evaluation with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_2_shift_fire_rate_cpu.bash
```

This launches 90 single-core array tasks with at most 50 concurrent tasks. A dependent single-core finalizer runs only if the entire array succeeds.

## Outputs

Outputs are written under:

```text
notebooks/artifacts/experiment_0_2_endpoint_tail_regularization/
  endpoint_tail_regularization_v1/
    experiment_0_2_shift_resolved_fire_rate/
      shift_fire_rate_v1/
```

Finalized files:

- `shift_fire_rate_summary.csv`
- `shift_fire_rate_comparison.csv`
- `shift_fire_rate_vs_none.csv`
- `shift_fire_rate_long_tau_summary.csv`
- `shift_fire_rate_manifest.json`

The notebook `notebooks/experiment_0_2_shift_resolved_fire_rate.ipynb` is analysis-only and reads these finalized artifacts.

## Main interpretation

Within the same architecture/objective/seed/shift, compare:

- `% change valid firing rate vs none`
- `% change tail firing rate vs none`
- `% change tail stage 1/2/3 firing rate vs none`

If long shifts (for example s6/s7) show much larger negative tail changes than valid-region changes, the regularizer is selectively reducing post-end long-timescale firing. If valid and tail changes are similar across shifts, the regularizer is mainly suppressing overall firing rather than solving endpoint-specific persistence.
