# Experiment 16.1 — z-only gate × Prefix interaction on one node

## Motivation

Exp16 showed three things: min-val-CE checkpoint selection was not a stable improvement; ungated Prefix calibration selected lambda=0; and the WCCE-trained suppressive gate converged to an almost-always-open policy.

Exp16.1 fixes the experimental design rather than adding capacity.

- Execution control: all paired runs execute inside one physical Slurm node.
- Mechanism simplification: compare a representation-conditioned z-only gate against the previous z+history gate.

## Questions

Q1: Can Prefix become useful specifically when a write gate exists?

Delta_g(lambda) = [GP_g(lambda)-G_g] - [P(lambda)-C0].

Q2: Is explicit previous-L2 history necessary for the gate?

z-only: g_t = sigmoid(G_z z_t + b_g).

z+history: g_t = sigmoid(G_z z_t + G_h h_{t-1} + b_g).

The z-only gate is still indirectly history-aware because z_t is produced by the stateful SNN backbone. Only the explicit recurrent controller input is removed.

Q3: Does Prefix recruit a genuinely selective write policy rather than another nearly-open multiplier?

## Frozen training contract

- actions 0+1, common 12 labels;
- frozen train/val/test user split;
- seeds 11, 23, 37;
- two SNN layers, width 128;
- shifts (2,3,4) in both layers;
- 64 Hz, 30 event channels;
- tau_mem = 22.54 ms;
- threshold = 0.5;
- unnormalized synaptic update;
- bias-free analog readout;
- CoreBenchmark Adam/LR/weight decay/batch size;
- max 100 epochs, min 20, patience 30;
- checkpoint selection = maximum native validation BA, then minimum validation CE, then earliest epoch;
- gate initialization g0 = 0.90;
- true suppressive multiplier g in (0,1), with no g/0.9 amplification.

No test metric is used for training, checkpoint selection, lambda selection, or case selection.

## Prefix objective

Prefix is exactly the Exp14.1/Exp16 accumulated-evidence objective:

L_P = 0.5 CE(A_0.50,y) + 0.5 CE(A_0.75,y).

A_p = mean evidence over the first ceil(pT) valid timesteps.

The full-sequence loss remains WCCE.

Exp16.1 does not use Phase0 to suppress Prefix. It directly tests lambda in {0.01, 0.03, 0.05}.

## Cases

For every seed:

| family | gate | lambda |
|---|---|---:|
| C0 | none | 0 |
| P | none | 0.01 / 0.03 / 0.05 |
| GZ0 | z-only | 0 |
| GPZ | z-only | 0.01 / 0.03 / 0.05 |
| GZH0 | z+history | 0 |
| GPZH | z+history | 0.01 / 0.03 / 0.05 |

This is 12 runs per seed × 3 seeds = 36 training runs.

Primary interactions:

Delta_z(lambda) = [GPZ_lambda-GZ0] - [P_lambda-C0].

Delta_zh(lambda) = [GPZH_lambda-GZH0] - [P_lambda-C0].

## Outputs

Native metrics: train/val/test BA, accuracy, macro-F1, mean-logit CE, train-test gap, selected epoch and stopped epoch.

L1/L2 probes: WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled, no-bias and affine decoders.

Primary representation quantity:

G_collapse = BA_Relative10 - BA_WholeCount.

The desired pattern is WholeCount up, Relative10 approximately stable, and collapse gap down. A smaller gap caused mainly by Relative10 falling is not a success.

## Gate diagnostics

Every gated run records mean/std of g, P(g<0.1), P(g>0.9), mean g(1-g), and a normalized 10-phase gate trajectory.

The key comparison to Exp16 is whether Prefix prevents the gate from converging toward g≈1.

## Gradient diagnostics

For fixed diagnostic epochs, Exp16.1 records Prefix-vs-WCCE gradient ratio and cosine for R, W_L1, W_L2, G_z, G_h when present, and gate bias.

r_q = lambda ||grad_q L_P|| / (||grad_q L_W|| + eps).

First-batch optimizer relative-update norms are also logged.

## Functional gate ablations

Every gated checkpoint is evaluated without retraining under:

1. learned gate;
2. g=1;
3. per-sample time-mean gate;
4. within-sample time-shuffled gate with three deterministic shuffle seeds.

A functional selective gate should show meaningful degradation when learned timing/content alignment is destroyed.

## Slurm execution contract

Exp16.1 now uses the same per-run scheduling pattern as the earlier experiments. Each independent training or ablation task requests one CPU and may be scheduled on any available Unity node.

Training array:
- 36 tasks: array 0-35;
- maximum concurrency 30;
- 1 CPU per task;
- 12 GB memory per task;
- 2 hour limit per task.

Functional-ablation array:
- 24 tasks: array 0-23;
- maximum concurrency 24;
- 1 CPU per task;
- 12 GB memory per task;
- 2 hour limit per task.

All numerical libraries are fixed to one thread per process: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1.

Every checkpoint records hostname. Finalization reports the observed node set so node-family variation can be inspected, but it no longer requires all cases to share one physical node.

## Submission

From the repository root on Unity:

git pull origin main
bash scripts/bash_script/SNN_Bash/submit_exp_16_1_cpu.bash

The wrapper submits:
prepare -> 36-task training array -> 24-task functional-ablation array -> finalize

with afterok dependencies between stages.

Individual array tasks write their normal Slurm stdout/stderr according to the cluster output configuration.

## Main interpretation files

- aggregate/interaction_summary.csv: validation/test Gate×Prefix interactions by gate mode and lambda;
- aggregate/gate_summary.csv: whether the gate actually becomes selective;
- aggregate/collapse_gap.csv: WholeCount/Relative10 internalization behavior;
- aggregate/functional_ablation_native_deltas.csv: functional dependence on learned gate behavior;
- aggregate/gradient_geometry.csv: whether Prefix credit actually reaches the gate and whether it conflicts with WCCE.

A native BA gain alone is insufficient to claim selective memory. Strong evidence requires positive interaction together with nontrivial gate selectivity and functional dependence on the learned gate policy.
