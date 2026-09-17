# Exp9.0 — Rotating within-user vs cross-user generalization

## Question

Exp9.0 separates two sources of error:

1. **within-user unseen-segment generalization** — the model has seen the user, but never the held-out segment;
2. **cross-user generalization** — validation/test users are completely absent from training.

The model family is held fixed. Only the evaluation split changes.

## Methods

Only two methods are run:

1. `raw250_linear` — **Raw250 + Linear**
   - Raw64 30-channel unsigned events.
   - Ordered 250 ms counts.
   - Train-only StandardScaler.
   - LogisticRegression with C selected by validation balanced accuracy.

2. `a2_234x234`
   - Exp7.3 A2-compatible `30 -> 128 -> 128 -> 12`.
   - L1 shifts `(2,3,4)`, L2 shifts `(2,3,4)`.
   - Binary hidden LIF neurons.
   - Shared bias-free Linear head.
   - Whole-sequence CE/WCCE-compatible valid-time mean evidence.
   - Validation BA checkpoint selection with validation loss tie-break.

No MM, RSNN, tau sweep, new loss, or augmentation is included.

## Five-fold rotating protocol

There are five folds. For rotation k:

- fold k = test;
- fold (k+1) mod 5 = validation;
- the other three folds = training.

Thus every sample is test exactly once and validation exactly once, while the train/val/test fold ratio is 3/1/1 = 60/20/20.

### A — within-user CV

Split unit: **segment**.

A StratifiedKFold is built independently inside every user. Therefore every user contributes segments to every fold. Sparse `(user,class)` pairs are retained; a class with fewer than five samples cannot appear in all five folds, which is reported rather than treated as an error.

The dataset has only 1–2 source trials per user, so source-trial grouping would make 5-fold within-user evaluation impossible. Sharing a recording session across folds is intentional here: this benchmark measures new segments from an already-seen user/session domain.

### B — cross-user CV

Split unit: **user**.

Twenty users are assigned to five user folds. Each rotation uses about:

- 12 users for training;
- 4 users for validation;
- 4 completely unseen users for test.

A user never appears in more than one fold, so the test-user condition is strict.

## Out-of-fold metrics

Fold mean/std is reported, but the primary result is **pooled out-of-fold test performance**.

Across five rotations every segment is test exactly once for each CV mode and method. The finalizer concatenates those predictions and reports:

- Accuracy;
- Balanced Accuracy — primary;
- Macro-F1;
- per-user OOF metrics;
- per-class OOF metrics;
- pooled confusion matrices.

The primary domain-shift diagnostic is:

[
G_{user}=BA_{within,OOF}-BA_{cross,OOF}.
]

## Multi-CPU strategy

`2 CV modes x 2 methods x 5 rotations = 20 independent jobs`.

Submission graph:

```text
prepare fold assignments
        |
        v
20-way CPU array
        |
        v
finalizer
```

Each array task requests one CPU and forces BLAS/OpenMP thread counts to one.

Run:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_9_0_cpu.bash
```

List array task identities:

```bash
python -m scripts.experiment_9_0_within_user_generalization list-runs
```

## Main outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_9_0_within_user_generalization/
  rotating_grouped_cv_v2/
```

Key files:

- `audit.json`
- `fold_summary.csv`
- `rotation_summary.csv`
- `within_user_fold_counts.csv`
- `cross_user_fold_users.csv`
- `metric_runs.csv`
- `metric_summary.csv`
- `prediction_runs.csv`
- `oof_test_predictions.csv`
- `oof_test_metrics.csv`
- `oof_per_user_test.csv`
- `oof_per_class_test.csv`
- `confusion_summary.csv`
- `within_vs_cross_user.csv`

The notebook is aggregation-only and never trains or refits a model.
