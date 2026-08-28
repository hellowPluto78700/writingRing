# Experiment 3.0.5 — Frozen representation accessibility

## Question

Does the frozen W128 multi-tau SNN already contain the temporal information needed for gesture classification, and how much of the observed performance difference is due to the readout rather than the backbone representation?

## Reused backbone

No SNN is trained in this experiment.

The experiment reuses Experiment 3.0.3 architecture B / Experiment 3.0.4 W128 checkpoints:

- input: 30 unsigned event channels;
- L1: 128 neurons, shifts `(2,3,4)`;
- L2: 128 neurons, shifts `(2,3,4)`;
- no L3;
- objectives: `timestep_ce`, `relative10_sequence_ce`, `fixed250_sequence_ce`;
- seeds: 11, 23, 101.

The representation is the frozen L2 spike sequence, not the native class logits.

## Frozen probes

Each objective/seed pair is evaluated with:

1. `full_count`: 128D orderless L2 spike count;
2. `fixed250_ordered`: 16 x 128 absolute-time bins;
3. `fixed250_shuffled`: same bins/dimension/counts with sample-wise valid-bin permutation;
4. `fixed250_pca128`: train-only scaler -> PCA128 -> scaler -> linear probe;
5. `relative10_ordered`: 10 x 128 relative-progress bins;
6. `relative10_shuffled`: sample-wise relative-bin permutation;
7. `relative10_pca128`: dimension-matched PCA128 control;
8. `duration_only`: valid gesture duration only.

Scaler/PCA are fit on train only. Logistic-regression `C` is selected by validation BA. Test is evaluated only after the probe is fixed.

The shuffled controls preserve feature dimension and per-bin contents while destroying temporal position. PCA128 is a dimension-sensitivity control, not the primary temporal-order control.

## Derived diagnostics

- absolute-time position gain = ordered fixed250 BA - shuffled fixed250 BA;
- relative-phase position gain = ordered relative10 BA - shuffled relative10 BA;
- post-hoc linear accessibility gap = best frozen-probe BA - native BA;
- duration-only BA.

The finalizer selects one source objective for Experiment 3.0.6 using mean **validation** BA across `full_count`, `fixed250_pca128`, and `relative10_pca128` over the three seeds.

## Multi-CPU execution

There are 9 independent frozen-evaluation tasks:

```text
3 objectives x 3 seeds = 9 tasks
```

Each Slurm array task uses one CPU core and writes one independent JSON artifact. The finalizer only aggregates completed artifacts and writes `selected_backbone.json`.

## Run

From the repository root:

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_3_0_5_to_3_0_7_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_3_0_5_to_3_0_7_pipeline.bash
```

Artifacts are written under:

```text
notebooks/artifacts/
  experiment_3_0_5_frozen_representation_accessibility/
    frozen_access_v1/
```
