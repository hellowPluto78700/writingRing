# Exp9.0 — Within-user unseen-segment generalization

## Question

Exp9.0 isolates ordinary segment generalization from unseen-user domain shift.
It asks: **if a user is represented in training, how well do the same models classify new, never-trained segments from that user?**

The experiment deliberately does not change the classifier family, SNN architecture, tau values, or training objective.

## Methods

Only two methods are run:

1. `raw250_linear` — **Raw250 + Linear**
   - Raw64 30-channel unsigned events.
   - Ordered 250 ms counts (`16` samples/bin at 64 Hz).
   - Flatten -> train-only `StandardScaler` -> LogisticRegression.
   - `C` selected on validation balanced accuracy.

2. `a2_234x234`
   - Exp7.3 A2-compatible network: `30 -> 128 -> 128 -> 12`.
   - L1 shifts `(2,3,4)`, L2 shifts `(2,3,4)`.
   - Binary hidden LIF neurons.
   - Shared bias-free `128 -> 12` linear head.
   - Valid-time mean evidence + whole-sequence CE (WCCE).
   - Checkpoint selection by validation BA, with validation CE tie-break.

No MM, RSNN, new regularizer, augmentation, or tau sweep is included.

## Split protocol

The target split is `60/20/20` train/val/test within each user while preserving all three splits for every active class.

A source trial is defined as:

```text
<user>/action_<action>/<package.stem>
```

All segments from one source trial must remain in one split. The assignment optimizer targets the per-`(user,class)` 60/20/20 counts while enforcing at least one train, val, and test sample for every active `(user,class)` pair.

Five repeated split seeds are used:

```text
11 23 37 53 71
```

### Insufficient `(user,class)` pairs

Pairs with fewer than 3 samples cannot satisfy strict train/val/test coverage. The prepare stage always writes:

- `manifests/initial_pair_summary.csv`
- `manifests/insufficient_user_class_pairs.csv`
- `manifests/audit.json`

The default policy is `error`: the audit job stops before training, leaving the summary for inspection.
Rerun with one explicit policy if needed:

```bash
export EXP9_INSUFFICIENT_POLICY=exclude_pair
# or: exclude_user
# or: exclude_class
bash scripts/bash_script/SNN_Bash/submit_exp_9_0_cpu.bash
```

The selected policy and excluded coverage are recorded in all artifacts.

## Multi-CPU Slurm strategy

Submission is a three-stage dependency chain:

```text
manifest/audit job
      |
      v
10-way CPU array = 2 methods x 5 split seeds
      |
      v
finalizer
```

Each array task uses one CPU and forces BLAS/OpenMP thread counts to 1, so tasks scale across CPU nodes without nested oversubscription.

Run:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_9_0_cpu.bash
```

List task identities locally:

```bash
python -m scripts.experiment_9_0_within_user_generalization list-runs
```

## Primary outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_9_0_within_user_generalization/
  within_user_segment_generalization_v1/
```

Important files:

- `metric_runs.csv`
- `metric_summary.csv`
- `per_user_runs.csv`, `per_user_summary.csv`
- `per_class_runs.csv`, `per_class_summary.csv`
- `confusion_summary.csv`
- `within_vs_cross_user.csv`
- `manifest.json`
- `manifests/*.csv`

Primary metric: balanced accuracy. Secondary metrics: accuracy and macro-F1.

The final notebook is aggregation-only; it never retrains or refits a model.
