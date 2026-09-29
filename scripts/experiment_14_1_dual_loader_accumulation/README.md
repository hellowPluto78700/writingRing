# Experiment 14.1 — Dual-Loader Transferable Accumulation

## Question

Exp14 showed that phase-conditioned cross-user supervision improves L2 cross-user geometry, retrieval, fixed250 and relative10 probes, but the improvement does not transfer proportionally to whole-count/native WCCE decoding.

Exp14.1 separates two targets:

1. **Transferability** — preserve Exp14 C2 cross-user phase alignment while preserving the original CoreBenchmark random **task-loss sampling distribution**. The structured Phase-CU auxiliary gradient remains an intentional inductive bias, so the overall backbone optimization distribution is not claimed to be unchanged.
2. **Shared-horizon decodability** — test whether the native shared readout can extract class-consistent accumulated evidence at multiple temporal horizons without forcing per-timestep TSCE. Stronger claims about intrinsic temporal organization require the direct evidence diagnostics below.

The primary question is:

> Can a single SNN learn cross-user transferable contextual states and make those states accumulate in a shared class-evidence space without erasing useful phase-dependent information?

## Frozen contract

The experiment reuses the finalized CoreBenchmark production cache and split:

- action0 + action1, 12 benchmark labels, 30 event input channels, 64 Hz;
- train users: 0,1,2,5,7,8,11,12,13,14,15,18,19,20;
- validation users: 4,9,16;
- test/OOD users: 3,6,10;
- seeds: 11,23,37;
- two hidden layers, width 128;
- L1 shifts (2,3,4), L2 shifts (2,3,4);
- tau_mem = 22.54 ms, threshold = 0.5;
- unnormalized synaptic update;
- bias-free analog output head;
- valid-mean-logit WCCE.

No SNN dynamics, architecture, tau, split, or native output geometry are changed.

## Dual-loader training

For Phase-CU cases, each optimization step uses a Core task batch plus one independent structured auxiliary batch. C0 bypasses auxiliary sampling entirely.

### Task batch

The task batch is the exact CoreBenchmark shuffled DataLoader. It is used for:

- native WCCE;
- Prefix-WCCE.

The structured cross-user sampler never contributes examples to these two losses.

### Auxiliary batch

The auxiliary batch is redesigned for the actual class-user sample distribution and is used only for phase-conditioned cross-user SupCon:

- 8 classes/batch;
- 8 distinct eligible users/class;
- exactly 1 unique segment from each selected (class,user) cell;
- 64 unique samples/batch.

A user is eligible for a class whenever that class-user cell contains at least one training segment. Cells with zero samples are skipped for that class only. Cells with one sample are valid; cells with two or three samples contribute one uniformly sampled segment each time that user is selected. This makes the auxiliary objective user-balanced rather than sample-count-balanced.

When Phase-CU is active there is exactly one validated auxiliary batch per Core task optimizer step. Every auxiliary batch must contain 64 unique segment IDs, exactly 8 selected classes, exactly 8 distinct users per selected class, and therefore exactly 7 valid same-class/different-user positives for every anchor.

## Losses

The total objective is

[
L = L_{WCCE}(B_{task})
  + lambda_a L_{prefix}(B_{task})
  + lambda_p L_{phase}(B_{aux}).
]

### Phase-CU

This is unchanged from Exp14 C2.

At normalized phases 25%, 50%, 75%, and 100%, take the L2 continuous pre-reset state, project

[
128 ightarrow 64 ightarrow ReLU ightarrow 32,
]

L2-normalize it, and use same-class/different-user positives in supervised contrastive loss.

The projector is training-only.

### Prefix-WCCE

For native evidence (e_t), define the prefix mean at phase (p)

[
ar e_p = rac{1}{t_p}sum_{t=1}^{t_p} e_t,
qquad
t_p=lceil pTceil.
]

Only 50% and 75% are supervised:

[
L_{prefix}
=
rac13 CE(ar e_{0.5},y)
+
rac23 CE(ar e_{0.75},y).
]

There is no 25% prefix term and no 100% prefix term. Full-sequence 100% supervision remains the ordinary WCCE term.

This is not TSCE: each prefix receives one CE after temporal averaging.

## Cases

Phase 1:

- C0_dual_null: WCCE only. It does not construct/iterate the auxiliary sampler, records the first three deterministic Core task-batch ID groups for debugging, and must reproduce CoreBenchmark O0 bitwise for the selected checkpoint.
- P_phase_cu: phase-CU only, lambda_phase in {0.01, 0.03, 0.06, 0.10}.
- A_prefix_wcce: Prefix-WCCE only, lambda_prefix in {0.10, 0.25, 0.50}.

Phase 2 uses validation-selected lambdas:

- J0_combined: full phase + full prefix;
- Jp_half_phase: half phase + full prefix;
- Ja_half_prefix: full phase + half prefix;
- Jb_half_both: half phase + half prefix.

These four are preregistered interaction/sensitivity cases. Test performance never selects among them.

A family with no validation-eligible candidate is a valid negative result, not a pipeline error. Its preregistered retention threshold is never relaxed after observing validation results. If either Phase-CU or Prefix-WCCE has no eligible candidate, combined cases are marked `not_applicable` and skipped. Final test evaluation contains only C0 plus families with a validation-selected candidate. Fixed Slurm array slots that have no mapped preregistered case exit successfully with `status=SKIPPED`.

## Prepare-time sampler feasibility

Before submitting training arrays, `prepare` validates the complete auxiliary sampling schedule across all benchmark seeds and all possible training epochs. For every class it records the number of train users with at least one segment, total unique training samples, and the minimum/maximum samples per eligible user. The experiment hard-fails before the phase-1 array if any class has fewer than 8 eligible train users.

## Scheduling

Both auxiliary weights:

- equal zero through epoch 10;
- ramp linearly from epochs 11–30;
- remain fixed after epoch 30.

Checkpoint selection is unchanged from CoreBenchmark:

1. higher held-out-user validation BA;
2. lower validation mean-logit CE;
3. earliest epoch.

## Phase-1 validation selection

No test arrays are used before selection.

### Phase lambda

First require

[
BA_{native,val} ge BA_{C0,val}-1pp.
]

Among eligible lambdas maximize L2 spike train-to-validation cross-user retrieval BA. Lambdas within 0.5 pp of the best retrieval choose the smaller lambda.

### Prefix lambda

Require both

[
BA_{native,val} ge BA_{C0,val}-1pp
]

and

[
BA_{relative10,val} ge BA_{C0,relative10,val}-1pp.
]

Then maximize L2 spike whole-count no-bias validation BA. Ties choose the smaller collapse gap and then the smaller lambda.

Define

[
G_{collapse}
=
BA_{relative10,no-bias}
-
BA_{wholecount,no-bias}.
]

The desired mechanism is whole-count improvement while relative10 remains approximately preserved, so the gap shrinks by raising whole-count rather than destroying phase-dependent information.

## Gradient diagnostics

At deterministic checkpoints, record on L2 weights. WCCE and Prefix-WCCE gradients are averaged over four fixed Core-distribution task batches; Phase-CU gradients are averaged over four fixed structured auxiliary batches before cosine computation:

- WCCE gradient norm;
- phase-CU gradient norm;
- Prefix-WCCE gradient norm;
- weighted phase/WCCE norm ratio;
- weighted prefix/WCCE norm ratio;
- cosine(phase, prefix);
- cosine(phase, WCCE);
- cosine(prefix, WCCE).

These are observational only. No dynamic weighting, GradNorm, PCGrad, optimizer modification, memory bank, or extra evidence-organization training loss is allowed.

## Direct prefix/evidence diagnostics

Prefix-WCCE is interpreted narrowly: it tests whether the actual trained shared native readout can decode accumulated evidence at 50%, 75%, and 100% horizons. Final evaluation therefore reports native Prefix50/Prefix75/Prefix100 BA in addition to whole-count and relative10 probes.

To distinguish "the model simply knows the answer earlier" from progressively compatible accumulation, final evaluation also records:

- true-class accumulated margin at 50%, 75%, and 100%;
- cosine similarity of cumulative evidence vectors at 50%↔75% and 75%↔100%;
- fraction of samples whose true-class margin is monotonic across 50%→75%→100%;
- sign-reversal rate of true-class support across the 0–50%, 50–75%, and 75–100% segments;
- a support cancellation ratio comparing net support with the sum of absolute segment support.

Native test BA improvement is a bonus outcome rather than a necessary success condition. The preservation gate remains native BA no worse than C0 by more than 1 pp; the core positive pattern is retrieval up, whole-count up, and relative10 approximately preserved.

## Execution

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_14_1_cpu.bash
```

Execution graph:

```text
prepare
  -> phase1 24-task CPU array
  -> validation-only lambda selection
  -> phase2 12-task CPU array
  -> final-eval 21-task CPU array
  -> aggregate-only finalizer
```

Each task uses one CPU core. Array concurrency defaults to 20 and is capped at 50.

Final-eval contains only C0, selected P, selected A, and the four combined cases across three seeds. Full timestep trace archives are not persisted.

## Main outputs

```text
notebooks/artifacts/
experiment_14_1_dual_loader_accumulation/
dual_loader_accumulation_v1/
```

Key aggregate files:

- native_summary.csv
- probe_summary.csv
- collapse_gap.csv
- collapse_gap_summary.csv
- cross_user_retrieval.csv
- cross_user_geometry.csv
- history_generalization.csv
- gradient_diagnostics.csv
- native_prefix_metrics.csv
- evidence_organization_diagnostics.csv
- paired_delta_vs_c0.csv
- manifest.json

The notebook `notebooks/experiment_14_1_dual_loader_accumulation.ipynb` is analysis-only.

## Validation

```bash
python -m pytest -q tests/test_experiment_14_1_dual_loader_accumulation_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
