# Exp7.2.6.1 — Sum-CE vs Mean-CE

## Scientific question

Exp7.2.4 A2 and Exp7.2.6 C0 use the same `234x234` two-layer SNN backbone and a shared analog linear output, but they do not train the sequence classifier with the same logit scaling:

- Exp7.2.6 C0: `CE(sum_valid_t W z_t, y)`
- Exp7.2.4 A2: `CE(mean_valid_t (W z_t + b), y)`

Exp7.2.6.1 isolates the first difference only:

```text
Does dividing the valid sequence logit by valid length before CE explain the large BA gap?
```

This experiment does **not** introduce a new loss family, temporal regularizer, output neuron, scaler, or bias term.

## Locked protocol

- architecture: `234x234`
  - L1 shifts `(2,3,4)`
  - L2 shifts `(2,3,4)`
- hidden width: 128
- regularization: `task_only` only
- seeds: `11, 23, 37`
- output projection: `Linear(128 -> 12, bias=False)`
- optimizer/LR/weight decay: identical to Exp7.2.6 C0
- max epochs: 100
- minimum epochs: 20
- patience: 30
- checkpoint rule: highest validation BA; validation objective loss only breaks BA ties
- valid-length masking everywhere

## Conditions

### S — Sum-CE baseline (reused, no retraining)

Exact existing Exp7.2.6 C0 checkpoints:

```math
s = \sum_{t<T} W z_t,
\qquad
\mathcal L_S = CE(s,y).
```

All three seed runs are reused from Exp7.2.6.1's parent experiment. Missing Exp7.2.6 C0 checkpoints/evaluations are fatal.

### M — Mean-CE (3 new training runs)

Exact paired rerun with one change:

```math
m = \frac{1}{T}\sum_{t<T} W z_t,
\qquad
\mathcal L_M = CE(m,y).
```

Model initialization and dataloader seed streams are the exact Exp7.2.6 C0 streams via `_reference_seed(...)`.

## Why readout is a sanity check, not another condition

For a fixed bias-free model:

```math
s = Tm,
\qquad T>0,
```

therefore

```math
\arg\max_k s_k = \arg\max_k m_k.
```

Exp7.2.6.1 checks this on every split for both the reused Sum-CE model and the newly trained Mean-CE model. The experiment hard-fails if predictions differ.

Thus any native BA difference between the two trained conditions comes from **training-time CE scaling**, not from choosing sum versus mean at inference.

## Scaling diagnostics

For the paired initialized model and trained models, the experiment records:

- CE under sum logits and mean logits
- mean/std logit L2 norm
- correlation between valid length and logit norm
- analytical per-sample output-weight CE gradient Frobenius norm
- correlation between valid length and that gradient norm

For bias-free linear readout with aggregated L2 feature `h`:

```math
\nabla_W CE(Wh,y) = (p-y_{onehot}) h^T,
```

so the per-sample Frobenius norm is computed exactly as

```math
\|p-y_{onehot}\|_2 \cdot \|h\|_2.
```

The Mean-CE training history records these diagnostics on validation data every epoch without changing optimization.

## L2 representation probes

After each Mean-CE run, L2 is frozen and evaluated with the same three Exp7.2.6 probes:

1. `l2_wholecount_matched_biasfree`
2. `l2_wholecount_linear` — train-only StandardScaler + validation-selected LogisticRegression
3. `l2_fixed250_linear` — ordered 16x128 Fixed250 features + the same standardized probe protocol

The finalizer reads the corresponding existing Exp7.2.6 C0 probe results as the paired Sum-CE baseline.

## Multi-CPU execution

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_6_1_cpu.bash
```

Pipeline:

1. `0-2%3` CPU array: three Mean-CE E2E training runs, one seed per CPU task.
2. `afterok` `0-2%3` CPU array: freeze each Mean-CE L2 and fit the three probes.
3. `afterok` one-CPU finalizer: merge the three new Mean-CE runs with the three reused Exp7.2.6 C0 Sum-CE runs.

There are only **3 new E2E trainings**. Sum-CE is never redundantly retrained.

## Final artifacts

The finalizer writes:

- `native_performance_runs.csv`
- `native_performance_summary.csv`
- `l2_probe_runs.csv`
- `l2_probe_summary.csv`
- `paired_delta_runs.csv`
- `paired_delta_summary.csv`
- `readout_equivalence_runs.csv`
- `readout_equivalence_summary.csv`
- `scaling_diagnostics_runs.csv`
- `scaling_diagnostics_summary.csv`
- `training_history_runs.csv`
- `training_history_summary.csv`
- `manifest.json`

The notebook `notebooks/experiment_7_2_6_1_sum_vs_mean_ce.ipynb` is analysis-only and reads summary CSVs only.

## Primary interpretation

Primary test:

```math
\Delta BA = BA_{MeanCE} - BA_{SumCE}.
```

If Mean-CE restores a large fraction of the Exp7.2.4-vs-Exp7.2.6 gap while fixed-model sum/mean predictions remain identical, the gap is attributable to training-time logit/gradient scaling from sequence summation rather than the analog linear readout itself.

If Mean-CE does not recover BA, the next isolated control is `mean + bias=True`; bias is intentionally not changed in Exp7.2.6.1.
