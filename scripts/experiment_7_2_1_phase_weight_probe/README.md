# Experiment 7.2.1 — Phase-specific vs shared L2 readout

This is a frozen-checkpoint extension of Exp7.2. It asks a narrow question:

> Does the trained L2 representation already contain enough phase context for one shared classifier, or does classification still require phase-specific downstream weights?

No SNN is retrained. The extension reuses all 84 Exp7.2 checkpoints.

## Three L2 readouts

For each checkpoint, extract the frozen L2 binary spikes and compare:

1. `l2_wholecount_shared`
   - valid-length L2 spike counts are summed over all time / all Fixed250 bins
   - feature dimension: 128
   - classifier: train-only StandardScaler + validation-selected LogisticRegression
   - mathematically equivalent to one shared temporal weight matrix:
     `W * sum_b z_b = sum_b W z_b`

2. `l2_fixed250_ordered`
   - reuse the exact Exp7.2 L2 Fixed250 probe
   - 16 ordered 250-ms bins x 128 neurons = 2048 features
   - each absolute time bin can receive its own effective weight matrix `W_b`

3. `l2_fixed250_phase_shuffled`
   - use the same 16x128 Fixed250 counts and the same 2048-D linear-probe capacity
   - independently permute the 16 bin positions for every sample
   - repeat this deterministic phase destruction 3 times and average the metrics
   - this preserves every sample's bin contents while destroying consistent absolute-phase identity

The ordered Fixed250 bins are absolute 250-ms positions from sequence start, not relative-length phase bins.

## Primary interpretation

Two paired test-set contrasts are primary:

- `ordered_minus_wholecount`
  - positive values mean phase-specific weights outperform one shared time-invariant weight matrix.

- `ordered_minus_phase_shuffled`
  - positive values mean the gain depends on consistent phase alignment, not merely the larger 2048-D feature space / parameter count.

A useful secondary control is `phase_shuffled_minus_wholecount`.

Interpretation guide:

- ordered >> wholecount and ordered >> phase-shuffled: strong evidence that temporal position remains external to L2 and downstream phase-specific weighting is useful.
- ordered >> wholecount but ordered ~= phase-shuffled: the gain is more consistent with generic high-dimensional capacity than phase identity.
- ordered ~= wholecount: the L2 representation is already largely sufficient for a shared classifier.

## Compute protocol

The extension evaluates the same 84 Exp7.2 run specs:

`7 architectures x 2 training families x 2 regularization conditions x 3 seeds = 84 frozen checkpoint evaluations`

There are **0 SNN training runs**.

The Slurm worker array uses one CPU per task and at most 50 concurrent workers. Each task loads one Exp7.2 checkpoint, extracts frozen L2 activity, fits the new linear probes, and writes one run JSON. A dependency finalizer creates aggregate CSVs for the notebook.

## Aggregate artifacts

The finalizer writes:

```text
notebooks/artifacts/experiment_7_2_1_phase_weight_probe/phase_weight_probe_v1/
  architecture_table.csv
  probe_summary.csv
  paired_deltas.csv
  manifest.json
```

Raw per-run JSON / CSV files are retained for debugging, but the notebook only reads the aggregate artifacts above.

## Run

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_1_phase_probe_cpu.bash
```

To inspect the 84 frozen-checkpoint task mapping:

```bash
python -m scripts.experiment_7_2_1_phase_weight_probe list-runs
```
