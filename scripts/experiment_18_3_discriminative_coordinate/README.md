# Exp18.3 — High-Rate Discriminative-Coordinate Decomposition

## Scientific question

Exp18.3 asks why high-rate L2 neurons become the preferred coordinates for the native bias-free accumulator.

The experiment is **artifact-only**. It never trains, resumes, or modifies an SNN. It compares finalized checkpoints from the Exp18/Exp18.2 chain:

| Case | Carrier | Loss | Source |
| --- | --- | --- | --- |
| `I_WCCE` | I/I | WCCE | CoreBenchmark O0 |
| `I_MWCCE` | I/I | Margin-WCCE | Exp18.2 |
| `U_WCCE` | U/U | WCCE | Exp18 U_NORMAL |
| `U_MWCCE` | U/U | Margin-WCCE | Exp18.2 |

Seeds are 11/23/37. This produces 12 independent diagnostic tasks.

NWCCE is intentionally excluded from the primary decomposition because Exp18.2 showed communication collapse, and the U-based NWCCE runs also contain an epoch-0 scale-calibration under-scaling confound. Exp18.3 instead contrasts high-rate WCCE with the lower-rate MWCCE regime that still retains substantial temporal probe information.

## Core hypothesis

The experiment does **not** define high firing as information by itself. It decomposes neuron utility into:

```text
rate magnitude
x class selectivity
x readout alignment
x transfer stability
-> native discriminative utility
```

The main question is whether high-rate neurons are useful because they are intrinsically more class-selective, because the native head aligns more strongly to them, because their class profile transfers better across users, or because those properties combine.

## Per-neuron quantities

All primary ranking variables are computed on the **training split only**.

For L2 neuron `j`:

- `train_rate_hz`: mean valid-window spike rate;
- `train_class_eta2`: one-way class eta-squared of normalized count/occupancy;
- `head_weight_norm`: native head column norm;
- `head_contrast_norm`: class-centered native head column norm;
- `train_readout_alignment_cos`: cosine between the neuron class-rate profile and its class-head profile after centering across classes;
- `train_aligned_signal`: centered class-rate profile dot centered head profile;
- `class_profile_train_test_cos`: transfer of the class-rate profile from train users to test users;
- `train/test_margin_contribution`: contribution to the true-vs-strongest-competitor native margin;
- single-neuron mean-replacement ablation ΔCE / ΔBA;
- single-neuron zero ablation ΔCE / ΔBA.

Mean replacement isolates class-varying information around the training mean. Zero ablation measures the total deployed contribution including mean activity.

## Why occupancy is the native feature

For the current bias-free WholeCount/WCCE readout:

```text
z = R (sum_t s_t / T_valid)
```

so valid-window occupancy is exactly the native feature presented to the head. No post-hoc probe is required for the native utility calculation.

## Group ablations

Train-only rankings define matched top-30% groups:

- `high_rate30`;
- `low_rate30`;
- `high_selectivity30`;
- `high_alignment30` using aligned-signal ranking;
- `high_margin30`;
- 20 deterministic `random30` controls.

For every group, the frozen native head is evaluated under:

1. `mean_replacement`;
2. `zero`.

No group is selected using validation or test labels.

## Correlation and multivariate decomposition

Each run reports Spearman relationships between training rate and:

- train/test class eta-squared;
- head contrast norm;
- train/test readout alignment;
- class-profile train→test transfer;
- train/test margin contribution;
- test mean-replacement ΔCE / ΔBA.

A descriptive standardized OLS predicts test mean-replacement ΔCE from:

- log(1 + train rate);
- train class eta-squared;
- train readout alignment;
- head contrast norm;
- train→test profile transfer.

This regression is diagnostic, not a causal estimator. Its role is to ask whether rate retains unique association with held-out-user native utility after the measured selectivity/alignment terms are included.

## Primary comparisons

Within each carrier, compare MWCCE against WCCE for:

- rate distribution;
- class selectivity;
- readout alignment;
- transfer stability;
- native margin contribution;
- single-neuron ablation utility;
- top-rate-group ablation relative to matched random groups.

Key interpretations:

- If WCCE high-rate neurons have higher class selectivity **and** stronger readout alignment, high rate is part of a jointly learned discriminative coordinate.
- If rate correlates with utility only through selectivity/alignment, rate is mainly a marker of those properties rather than an independent source of information.
- If MWCCE preserves selectivity but loses alignment/native utility, the main loss-induced effect is representation formatting for the accumulator.
- If top-rate ablation is no more harmful than random after matching dimensionality, high rate is redundant rather than uniquely causal.
- If high-rate groups are more harmful under WCCE but not MWCCE, WCCE specifically concentrates native reliance into high-rate coordinates.

## Anti-leakage contract

- neuron rankings use training data only;
- group membership uses training data only;
- deterministic random masks depend only on case, seed, and replicate;
- test labels are used only for diagnostic evaluation after rankings are fixed;
- no SNN or head parameter is retrained;
- source checkpoints/traces are hash-locked by `prepare`;
- finalization fails if any source artifact changes.

## Required per-run artifacts

Each of the 12 diagnostic tasks must produce:

- `neurons.csv`;
- `group_ablation.csv`;
- `summary.json`;
- `complete.json`.

A Slurm task reaching COMPLETED is not sufficient without a valid PASS `complete.json`.

## Final aggregate artifacts

`aggregate/` contains:

- `neurons.csv`;
- `group_ablation.csv`;
- `run_summary.csv`;
- `correlations.csv`;
- `utility_ols.csv`;
- `case_summary.csv`;
- `paired_loss_contrasts.csv`;
- `group_summary.csv`.

The root `aggregate.json` hashes all aggregate outputs and reports PASS only after all 12 source-locked diagnostic runs complete.

## Slurm execution contract

Recommended concurrency: 12 CPU tasks.

Exact DAG:

```text
compute-node smoke
  -> prepare
      -> 12-task artifact-only diagnostic array
          -> finalize
```

No training stage exists.

All jobs use:

`scripts/bash_script/SNN_Bash/slurm_cpu_env.bash`

Formal submission:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_18_3_cpu.bash
```

Resume behavior:

- valid `complete.json` + matching provenance -> skip;
- missing or mismatched provenance -> recompute or fail;
- changed source checkpoint/trace hash -> fail rather than silently reuse.

The compute-node smoke loads one seed for every formal case and exercises class profiles, readout alignment, native margin contribution, and single-neuron ablation through the real finalized artifact paths.
