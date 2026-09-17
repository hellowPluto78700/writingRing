# Exp9.0 — Within-user Segment Generalization

## Goal

Quantify how much of the current generalization gap comes from unseen segments of known users versus unseen-user domain shift.

The experiment changes **only the split protocol**. It intentionally keeps the two reference methods fixed:

- **A — Raw250 + Linear**: raw 30-channel event stream, ordered 250 ms count features, train-only scaling, validation-selected LogisticRegression.
- **B — A2 `(234)(234)`**: Exp7.3 A2-compatible two-hidden-layer binary SNN with shared linear WCCE readout.

No MM, RSNN, tau sweep, new regularizer, new augmentation, or auxiliary loss is included.

## Within-user split

For every active user and class, target counts follow a `60/20/20` train/val/test split. Integer allocation uses largest remainder and requires at least one sample in train, val, and test whenever the pair has at least three samples.

The physical split unit is the **source trial**, not a derived segment. All segments from:

```text
<user>/action_<action>/<package.stem>
```

remain in exactly one split. Trial assignment is optimized per user to approach the `(user,class)` 60/20/20 targets while enforcing class presence in all three splits.

Five repeated split seeds are used: `11, 23, 37, 53, 71`.

## Insufficient-data audit

The prepare stage always writes the complete `(user,class)` table. A pair with fewer than three segments is marked `insufficient` because strict three-way coverage is impossible.

Default behavior is **fail closed**: write the audit, then stop before training. One of these policies must be selected explicitly if the audit is not clean:

- `exclude_pair`
- `exclude_user`
- `exclude_class`

The Slurm entrypoint reads the policy from `EXP9_INSUFFICIENT_POLICY`; if unset, it uses `error`.

## Experiment matrix

`2 methods x 5 split seeds = 10 independent CPU jobs`.

Each task reports train/val/test:

- accuracy
- balanced accuracy (primary)
- macro-F1

It also writes per-user metrics, per-class precision/recall/F1, and confusion matrices.

## Multi-CPU execution

The dependency chain is:

```text
prepare/audit -> 10-task CPU array -> finalizer
```

Every task requests one CPU and sets OpenMP/MKL/OpenBLAS/NumExpr to one thread. The array therefore parallelizes across independent model/split jobs without nested CPU oversubscription.

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_9_0_cpu.bash
```

If the audit reports insufficient pairs, inspect:

```text
notebooks/artifacts/experiment_9_0_within_user_generalization/
  within_user_segment_generalization_v1/manifests/
```

then resubmit with an explicit policy, for example:

```bash
EXP9_INSUFFICIENT_POLICY=exclude_pair \
  bash scripts/bash_script/SNN_Bash/submit_exp_9_0_cpu.bash
```

## Interpretation

The primary diagnostic quantity is:

\[
G_{user}=BA_{within-user,test}-BA_{cross-user,test}.
\]

A large positive gap means the models classify unseen segments from known users substantially better than segments from unseen users, supporting cross-user domain shift as the dominant bottleneck.

The finalizer opportunistically reads the existing Exp0.1 Raw250 and Exp8.0 A2 cross-user artifacts when available and writes `within_vs_cross_user.csv`. Missing historical artifacts do not invalidate the Exp9.0 within-user benchmark.
