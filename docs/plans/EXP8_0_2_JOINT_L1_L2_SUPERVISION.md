# Exp8.0.2 — Joint L1/L2 Supervision

## Goal

Exp8.0.1 showed that a post-hoc frozen `L1 + L2` linear probe does not consistently outperform the best single layer, even though L1-only-correct and L2-only-correct samples both exist. Exp8.0.2 tests whether this is a consequence of the original training topology, where only L2 is directly supervised.

The experiment keeps the local SNN backbone fixed to the Exp8.0 winner/baseline:

```text
30 -> L1(128, shifts 2/3/4) -> L2(128, shifts 2/3/4)
```

and changes only the supervision/readout topology.

## Methods

All methods use Exp7.3 A2-compatible end-to-end training, the same data split, optimizer, early stopping, paired initialization/data-order seed streams, hidden neuron dynamics, task-only classification objective family, and seeds `11, 23, 37`.

### A. `l2_only`

Current A2 topology and strict replication control:

```math
s = W_2 \bar z_2,
\qquad
\mathcal L = CE(s,y).
```

Only L2 directly drives classification.

### B. `l1_l2_joint`

Both levels directly drive the same classifier score:

```math
s = W_1 \bar z_1 + W_2 \bar z_2,
\qquad
\mathcal L = CE(s,y).
```

This tests whether direct supervision of L1 causes the hierarchy to preserve useful multi-level evidence that was previously absorbed into L2.

### C. `l2_main_l1_aux`

The deployed/native classifier remains L2-only, while L1 receives weak auxiliary supervision:

```math
s_2 = W_2 \bar z_2,
\qquad
s_1 = W_1 \bar z_1,
```

```math
\mathcal L = CE(s_2,y) + \lambda CE(s_1,y),
\qquad \lambda=0.1.
```

Checkpoint balanced accuracy and native test accuracy are computed from the L2 main head. The total objective loss is used only as the loss-side checkpoint tie breaker. This separates “directly use L1 at inference” from “use L1 supervision only to shape the backbone.”

## Training controls

- backbone: `234x234` only;
- hidden width: 128/128;
- binary hidden spikes;
- bias-free classification heads;
- Exp7.3 A2 optimizer, LR, weight decay, epoch budget, early stopping, and checkpoint improvement rule;
- seeds: `11, 23, 37`;
- no additional regularizer;
- no tau sweep;
- no backbone pretraining/freeze;
- every method is trained end-to-end from the paired initialization stream.

There are exactly:

```text
3 methods x 3 seeds = 9 independent CPU training jobs
```

## Native readout evaluation

For each best checkpoint report train/val/test balanced accuracy for the readout used by that method during training:

- `l2_only`: `W2 z2`;
- `l1_l2_joint`: `W1 z1 + W2 z2`;
- `l2_main_l1_aux`: `W2 z2`.

Also evaluate the same trained evidence with the Exp8.0 constrained output LIF realization:

```math
I_t=e_t,\quad \alpha_{out}=0,\quad \beta_{out}=0.5,
```

with the standard threshold/cap and valid-region output spike-count classification. No readout weight is retrained for this transfer.

## Frozen representation probes

After training, freeze the checkpoint and extract L1/L2 binary spike trajectories. Reuse the Exp8.0.1 probe protocol exactly:

1. `l1_whole`
2. `l2_whole`
3. `l1_l2_whole`
4. `l1_fixed250`
5. `l2_fixed250`
6. `l1_l2_fixed250`
7. `l1whole_l2fixed250`
8. `l1fixed250_l2whole`

Each diagnostic probe uses train-only `StandardScaler`, validation-selected logistic-regression `C`, and test evaluation only after model selection.

The central representation metrics are:

```math
G_{whole}=BA(L1+L2) - \max(BA(L1),BA(L2))
```

and

```math
G_{F250}=BA(L1+L2) - \max(BA(L1),BA(L2)).
```

The experiment is successful mechanistically if joint/auxiliary supervision changes these fusion gains, not merely if the native head has more parameters.

## Complementarity diagnostics

For L1-only and L2-only probes, preserve Exp8.0.1 correctness overlap statistics:

- both correct;
- L1-only correct;
- L2-only correct;
- both wrong;
- prediction disagreement;
- oracle union accuracy.

For all fused probes record coefficient block RMS/Frobenius norms and L1/L2 norm fractions.

For the trained heads also record `W1`/`W2` Frobenius and RMS norms. This distinguishes a nominal joint topology from a solution that effectively ignores one level.

## CPU execution

Use one Slurm array task per `(method, seed)` pair:

```text
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=1
```

Each task performs:

```text
train -> select best checkpoint -> native evaluation -> same-W LIF transfer
      -> frozen representation probes -> per-run artifacts
```

Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.

A finalizer is submitted with `afterok` on the training array. It must fail on missing runs and only aggregate existing artifacts.

## Finalized artifacts

The finalizer writes at least:

```text
method_runs.csv
method_summary.csv
probe_runs.csv
probe_summary.csv
fusion_gain_runs.csv
fusion_gain_summary.csv
correctness_overlap_runs.csv
correctness_overlap_summary.csv
coef_block_runs.csv
coef_block_summary.csv
trained_head_runs.csv
trained_head_summary.csv
history_runs.csv
manifest.json
```

## Notebook strategy

`notebooks/experiment_8_0_2_joint_l1_l2_supervision.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts and shows method-level comparisons, not nine separate run narratives.

Main figures/tables:

1. native Linear and same-W output-LIF BA by method;
2. L1/L2/L1+L2 whole-count probe BA;
3. L1/L2/L1+L2 Fixed250 probe BA;
4. fusion gain over the best single layer;
5. correctness overlap / oracle-union diagnostics;
6. trained-head and post-hoc probe block usage;
7. mean train/validation BA and loss curves across seeds.

## Interpretation

- If `l1_l2_joint` increases native BA but post-hoc fusion gain remains near zero, the gain is primarily a different classifier/training path, not evidence that L1/L2 became complementary.
- If post-hoc `L1+L2` fusion gain becomes consistently positive after joint supervision, the original L2-only topology was suppressing useful multi-level complementarity.
- If `l2_main_l1_aux` improves L2-only native/probe performance without requiring L1 at inference, weak deep supervision is the cleaner deployment path.
- If neither method improves representation probes, the limited fusion in Exp8.0.1 is more likely an intrinsic property of this hierarchical local backbone than a readout-training artifact.
