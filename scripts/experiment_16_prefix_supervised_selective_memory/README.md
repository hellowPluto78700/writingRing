# Experiment 16 — Prefix-supervised selective memory

## Question

Exp16 asks whether intermediate **accumulated-state** supervision becomes useful when the SNN has an explicit context-dependent remember/write gate.

The primary hypothesis is not simply that Prefix improves classification. It is:

\[
\text{Prefix credit} + \text{trainable write policy}
\rightarrow
\text{memory-oriented co-specialization}
\]

between candidate representation and write policy.

Exp16 deliberately separates protocol calibration from mechanism testing.

## Frozen CoreBenchmark contract

Unless explicitly unlocked below, Exp16 keeps the current CoreBenchmark production contract:

- dataset/cache and user split;
- seeds 11, 23, 37;
- two SNN layers, width 128;
- shifts (2,3,4) in both layers;
- 64 Hz, 30 event channels;
- tau_mem = 22.54 ms;
- bias-free analog readout;
- optimizer, LR, weight decay, batch size and epoch budget;
- task-level CPU Slurm execution.

The gate geometry is inherited from Exp15: one scalar gate per sample/timestep, conditioned on current L1 spikes and previous L2 spikes. Exp16 changes only the multiplier semantics from Exp15's g/0.9 to the true suppressive multiplier g:

\[
g_t=\sigma(G_z z_t + G_h h_{t-1}+b_g),\qquad 0<g_t<1.
\]

The initial gate value is 0.9. Common non-gate parameters retain paired initialization across same-seed cases.

## Prefix loss

Prefix is the Exp14.1 accumulated-evidence objective, not TSCE.

\[
A_p = \frac{1}{\lceil pT\rceil}\sum_{t=1}^{\lceil pT\rceil} e_t,
\qquad p\in\{0.50,0.75\}.
\]

\[
L_P =
0.5\,CE(A_{0.50},y)
+
0.5\,CE(A_{0.75},y).
\]

The full-sequence objective remains:

\[
L_W = CE\left(\frac1T\sum_{t=1}^T e_t,y\right).
\]

There is no 25% Prefix, no 100% Prefix duplicate, and no TSCE in Exp16.

# Phase 0A — checkpoint-selection control

For every seed, Exp16 trains **one ordinary ungated WCCE trajectory**. The trajectory and early stopping remain governed by the existing CoreBenchmark rule: highest validation BA, then lower validation mean-logit CE.

From the exact same visited trajectory, Exp16 saves two selected checkpoints:

- C0-BA: repository-standard BA-first checkpoint;
- C0-CE: minimum validation mean-logit CE checkpoint, with validation BA only as tie-break.

Therefore the two cases have identical initialization, batches, optimizer steps and visited epochs. They differ only in which visited epoch is selected.

The rule for later phases is chosen with validation only. C0-CE is adopted if its validation BA is within 0.5 pp of C0-BA on at least 2/3 seeds; otherwise BA-first remains frozen. Test metrics are reported but never enter this decision.

# Phase 0B — Prefix coefficient calibration

Candidates:

\[
\lambda_P\in\{0,\;0.01,\;0.03,\;0.05,\;0.10\}.
\]

At preserved C0 trajectory epochs 10, 20, 30, 50 when those epochs were actually reached, Exp16 clones model and optimizer state and performs a one-epoch lookahead.

A nonzero lambda is eligible only when:

\[
CE_{val}(0)-CE_{val}(\lambda)\ge0.005
\]

and:

\[
BA_{val}(\lambda)-BA_{val}(0)\ge -0.5\text{ pp}.
\]

A lambda must have at least one eligible point on at least 2/3 seeds. Among supported lambdas, choose the **smallest** lambda. If none qualifies, lambda_P*=0 is a valid negative result. Test metrics are not used.

# Phase 1 — Gate × Prefix factorial

| Case | Gate | Training objective |
|---|---|---|
| C0 | no | WCCE |
| P0 | no | WCCE + fixed Prefix |
| G0 | suppressive context gate | WCCE |
| GP_J | suppressive context gate | WCCE + fixed Prefix |

All cases train from scratch with paired common-weight initialization and use the Phase0-frozen checkpoint rule. P0 and GP_J share one fixed lambda_P*.

Primary validation interaction:

\[
\Delta_{int}
=
(BA_{GP_J}-BA_{G0})
-
(BA_{P0}-BA_{C0}).
\]

Phase1.5 is enabled only if both GP_J-G0 > 0 and Delta_int > 0 on mean validation native BA. Test data do not enter this gate.

## Required evaluation

Every Phase1 case reports native train/val/test BA, accuracy, macro-F1 and CE; train-test gap; and L1/L2 CoreBenchmark temporal probes including whole-count, fixed250 ordered/shuffled, and relative10 ordered/shuffled.

The important accumulation quantity remains relative10 minus WholeCount. The desired pattern is WholeCount improvement with Relative10 preserved, not merely larger temporally resolved decodability.

Existing Exp14 retrieval/history diagnostics are also attempted from saved traces.

## Gate diagnostics

For gated cases report mean/std of g, P(g<0.1), P(g>0.9), mean g(1-g), and the normalized 10-phase gate trajectory.

Exp16 intentionally preserves Exp15's scalar gate geometry. It does not add a vector gate because that would confound Prefix supervision with gate capacity.

## Gradient diagnostics

At preregistered epochs report Prefix-vs-WCCE weighted gradient ratio and cosine for R, W_L1, W_L2, G_z, G_h, and gate bias. Also log first-batch optimizer relative update norm.

# Phase 1.5 — conditional pathway attribution

Phase1.5 runs only after a positive validation Gate × Prefix interaction.

- GP_stopR: Prefix updates all upstream parameters except head.weight.
- GP_gate: Prefix directly updates only gate parameters. The forward graph is not detached; autograd.grad computes the full recurrent total derivative.
- GP_noGate: Prefix directly updates non-gate parameters but contributes no direct gate gradient.

WCCE always updates all trainable parameters normally.

Interpretation is limited to direct auxiliary-gradient routing. Multi-epoch indirect co-adaptation remains possible.

# Functional mechanism ablations

Inference-only:

1. learned gate;
2. gate_one: g_t=1;
3. gate_time_mean: replace each sample's time-varying gate by its valid-time mean;
4. gate_time_shuffle: shuffle each sample's learned valid gate trajectory, with three deterministic shuffle seeds.

These test whether gating itself, dynamic gating, and content/history-aligned gate timing are functionally necessary.

# Decision flow

~~~text
Phase0A
shared C0 trajectory
  -> BA-first checkpoint
  -> min-val-CE checkpoint
  -> freeze checkpoint rule

Phase0B
Prefix lambda lookahead on validation only
  -> freeze lambda_P*

Phase1
C0 / P0 / G0 / GP_J
  -> positive validation Gate x Prefix interaction?
       no  -> report negative mechanism result
       yes -> Phase1.5

Phase1.5
GP_stopR / GP_gate / GP_noGate

Functional attribution
learned / g=1 / time-mean / time-shuffle
~~~

A negative result at any decision point is intended and must not be converted into test-guided selection.

# CPU execution

One independent run per Slurm array task, one CPU core per task.

- Phase0A: 3 tasks.
- Phase0B: 60 lookahead slots; unreached base epochs return SKIPPED.
- Phase1: 12 tasks.
- Phase1.5: 9 conditional tasks.
- Functional ablation: 15 gated-model slots; unavailable conditional checkpoints return SKIPPED.

Finalizers aggregate existing artifacts and make validation-only decisions; they do not retrain models.
