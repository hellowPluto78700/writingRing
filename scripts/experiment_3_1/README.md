# Experiment 3.1 — Raw vs SNN representation value

## Question

Does the frozen multi-tau SNN add discriminative value beyond the original 30-channel event representation when both are evaluated on the exact same samples, user split, temporal aggregation, and linear-probe protocol?

This experiment is intentionally diagnostic. It does **not** train a new SNN.

## Frozen source backbone

Experiment 3.1 reads `selected_backbone.json` finalized by Experiment 3.0.5 and uses the selected Experiment 3.0.3-B / W128 checkpoint for each seed.

Current source contract:

- input: 30 unsigned event channels at 64 Hz;
- L1: 128 neurons, shifts `(2,3,4)`;
- L2: 128 neurons, shifts `(2,3,4)`;
- no L3;
- frozen SNN seeds: `11, 23, 101`;
- user split: the Phase-C split with `SPLIT_SEED=12345`;
- SNN representation: L2 spike sequence.

The selected training objective is read from Experiment 3.0.5 rather than hard-coded, so Experiment 3.1 evaluates the exact backbone selected by validation-only frozen probes.

## Matched representations

For every sample and every frozen SNN seed, the experiment constructs three representations from the same event sequence:

1. `raw`: original 30-channel event sequence;
2. `snn_l2`: frozen 128-channel L2 spike sequence;
3. `raw_plus_snn`: concatenation of the corresponding raw and L2 temporal features.

The primary goal is to report:

```text
Delta_SNN    = BA(SNN-L2) - BA(Raw)
Delta_Fusion = BA(Raw+SNN) - BA(Raw)
```

A positive `Delta_SNN` means the SNN representation itself is more linearly discriminative than the matched raw representation. A positive `Delta_Fusion` with a weak/negative `Delta_SNN` instead supports a complementary-feature interpretation.

## Temporal probes

Each representation is evaluated with the same probe family:

- `full_count`;
- `fixed250_ordered`;
- `fixed250_shuffled`;
- `fixed250_pca128`;
- `relative10_ordered`;
- `relative10_shuffled`;
- `relative10_pca128`.

`duration_only` is evaluated once per repeated seed as a leakage control and must be numerically identical across repeated tasks.

At 64 Hz, 250 ms is 16 samples. Relative10 divides each gesture's valid duration into ten progress bins using the same `relative_counts` implementation as the Phase-C SNN objectives.

### Fair controls

- Raw and SNN use identical train/validation/test sample identities and labels.
- Raw and SNN temporal shuffle controls use the **same sample-wise permutation** for a given split, so temporal-order destruction is paired.
- Invalid fixed-bin padding is left in place.
- PCA is fit on training data only and reduces Raw, SNN, and Fusion to the same 128D dimensionality.
- Logistic-regression `C` selection uses validation BA only; test data is never used for probe selection.
- The probe random seed depends on representation/probe type, not SNN seed. Therefore the repeated deterministic Raw baseline must be bitwise/numerically identical across the three array tasks; the finalizer rejects mismatches.

## Primary outputs

The finalizer writes:

- `experiment_3_1_probe_results.csv`: all repeated per-seed probe results;
- `experiment_3_1_probe_summary.csv`: Raw/SNN/Fusion summaries;
- `experiment_3_1_paired_deltas.csv`: per-seed matched BA deltas;
- `experiment_3_1_delta_summary.csv`: mean/sd matched deltas;
- `experiment_3_1_primary_comparison.csv`: Fixed250 and Relative10 ordered/PCA128 comparison table;
- `experiment_3_1_conclusion.json`: compact machine-readable primary result.

Artifacts are written under:

```text
notebooks/artifacts/
  experiment_3_1_raw_vs_snn_representation_value/
    matched_repr_v1/
```

## Multi-CPU execution

There are three independent frozen-evaluation tasks:

```text
seed 11
seed 23
seed 101
```

Each Slurm array task uses one CPU core and evaluates Raw, the matching frozen SNN, and Raw+SNN together. This keeps all representation comparisons paired within a task while still parallelizing the expensive frozen SNN forward pass across seeds.

The finalizer is an `afterok` dependency and only aggregates already-written JSON artifacts.

Run from repository root:

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_3_1_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_3_1_pipeline.bash
```

The experiment requires finalized Experiment 3.0.5 selection metadata and the corresponding Experiment 3.0.3-B checkpoints to already exist on the machine running the Slurm jobs.
