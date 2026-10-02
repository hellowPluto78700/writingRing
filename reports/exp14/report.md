# Exp14 — History Organization and Dual-Loader Accumulation

## Report Scope and Evidence Cutoff

This report covers the original `experiment_14_history_organization` (`history_organization_v1`) and the redesigned `experiment_14_1_dual_loader_accumulation` (`dual_loader_accumulation_v1`). These are related but distinct training protocols. Original Exp14's structured task batches are not pooled with Exp14.1's ordinary task loader plus independent auxiliary loader. All original objective strengths are reported; Exp14.1 reports validation screening for all 24 candidates and final test evidence only for the six selected evaluations.

Evidence was inspected on 2026-10-02. Implementation was read from GitHub snapshot `3f0e349e48f46f0c14b536e1923488c27381dcbc`; Unity results were read from `/home/zhaolongwei_umass_edu/projects/writingRing` at tracked HEAD `ef12b4e22de64693f3af317e4757ba913bf8ab12`. The two snapshots are recorded separately: current source explains the final contract; commits and run manifests establish revisions. Unity files and checkpoint metadata determine quantitative outcomes. No training or cluster files were changed for this reconstruction.

Project history was actively searched by experiment IDs and mechanism terms, including discussions before design, during implementation, after results, and in successor experiments. The accessible retrievals contain dated user statements and prior assistant summaries, rather than a complete exported Project archive. They support the reasoning chronology but do not establish run completion or numerical truth. Titles are reported only where recovered; other entries are labeled by topic with title unavailable. Missing full transcripts and original failed-job stdout/stderr are evidence gaps, not reconstructed quotations.

All BA values are balanced accuracy, reported as percentages; differences are percentage points (pp). Unless explicitly labeled otherwise, `mean ± SD` is an equally weighted mean and sample SD (`ddof=1`) over model seeds 11, 23, 37, **n=3**. Shuffle replicates, batches, neurons, strokes, and users are not additional model seeds. No significance claim is made from these three paired seeds.

## 1. Experiment Identification

| Alias | Canonical ID / version | Genuine substructure |
| --- | --- | --- |
| Exp14, history organization | `experiment_14_history_organization` / `history_organization_v1` | C0 WCCE; C1 Whole-CU; C2 Phase-CU; C3 History-Delta; C4 selected combination/full and half strength; C5 shuffled auxiliary labels |
| Exp14.1, dual loader | `experiment_14_1_dual_loader_accumulation` / `dual_loader_accumulation_v1` | C0 dual-null; P Phase-CU grid; A Prefix-WCCE grid; conditional combined phase 2; selected final evaluations |
| Phase-CU, class consistency | Objective alias | Same-class/different-user supervised contrastive features, not a separate dataset or experiment |
| Prefix | Objective in 14.1 | Early accumulated-evidence CE at 50% and 75%; not present in original Exp14 or Exp15 |

The authoritative implementations are [Exp14](../../scripts/experiment_14_history_organization.py) and [Exp14.1](../../scripts/experiment_14_1_dual_loader_accumulation.py); the latter has a [task specification](../../docs/plans/experiment_14_1_dual_loader_accumulation_task_spec.md). Artifacts use `notebooks/artifacts/<canonical ID>/<version>/runs/<LossSpec.key>/`. Slurm names `exp14_*` and `exp14_1_*` denote execution stages, not independent hypotheses. The direct predecessor is the Exp13/Core diagnosis of contextual representation versus additive readout; the successor is Exp15's intervention on the write mechanism.

## 2. Scientific Context

Exp13/13.1 showed that deeper state can express more old context without monotonic gains in cross-user retrieval or native accumulated-spike classification. Separately, standardized Core O0 gives no-bias L2 WholeCount **57.57 ± 4.24%** versus ordered Relative10 **67.04 ± 0.73%**, while native O0 is **57.93 ± 2.23%**. These are historical Core checkpoints, not Exp14 controls [E14-1]. The gap suggests that useful class structure remains accessible to temporal readouts but incompletely consolidated into the native accumulator.

The first hypothesis was an organization problem: train the representation so same-character states across writers become compatible. The later dual-loader design addressed a different confound: contrastive positives require structured samples, but replacing the native task loader changes the baseline training distribution. A still later prefix objective tested whether earlier consistent class evidence could make accumulation more useful. These motivations are chronological, not one design that existed from the start.

## 3. Motivation and Open Question

Same-class/different-user positives target task consistency across nuisance style. A phase-conditioned objective should align comparable progress without requiring an entire trajectory to share one state; a history-delta objective should organize the component induced by older input. However, a representation can satisfy contrastive geometry through a projection while the native head still accumulates poorly. The report therefore distinguishes geometry, diagnostic probe BA, and native BA.

The 2026-09-29 03:20 discussion initially interpreted Phase-CU as improving geometry without reliably improving native performance. At 15:47–16:49, the design shifted toward preserving ordinary WCCE sampling, using fresh separate auxiliary batches, adding prefix supervision, measuring gradient scales, and treating “no eligible candidate” as a scientific outcome. A stale memory bank was considered and rejected in favor of fresh dual-loader features. The distinction matters because auxiliary gradients need to reach the current representation, not an old cache.

## 4. Hypotheses

| Hypothesis | Predicted result | Diagnostic/control |
| --- | --- | --- |
| Poor cross-user organization limits transfer | CU supervision improves held-out retrieval / geometry, potentially native BA | True auxiliary labels versus C5 shuffled labels; class/user matched positives |
| Phase mismatch matters | Phase-CU outperforms whole-trajectory CU | C1 versus C2 under matched original batches |
| Old-history component is poorly organized | History-Delta or its combination adds transferable benefit | C3 and validation-selected C4 |
| Structured task sampling is a confound | Original C0 can differ from ordinary Core training | Separate 14.1 task loader and current-contract C0 |
| Early evidence is inconsistent | Prefix raises useful early evidence and closes Relative10–WholeCount gap while preserving final BA | Native and Relative10 validation safeguards; evidence diagnostics |
| More auxiliary pressure solves the problem | Larger λ improves target metrics without native loss | Full λ grids; this prediction receives negative evidence |

## 5. Experimental Design

### 5.1 Original Exp14

Every case is trained from paired fresh SNN initialization. **All cases, including C0, use a structured 8 classes × 4 users × 4 segments = 128 batch.** The sampler chooses classes and distinct users without replacement but samples segments with replacement if a class-user cell contains fewer than four. This is internally matched among Exp14 cases but differs from the ordinary Core DataLoader.

C1 applies CU to projected valid-mean L2 pre-reset state. C2 averages CU losses at 25/50/75/100% progress. C3 contrasts the endpoint difference between full-history L2 Upre and a rerun with both layers' state reset to retain the last 250 ms. C4 combines validation-selected C2 and C3 strengths and separately tests half strength. C5 shuffles only auxiliary labels while preserving true labels for WCCE. λ is 0.01, 0.03, 0.1, 0.3 for C1–C3. There are 14 configurations × three seeds = 42 phase-1 runs, plus six C4 runs.

### 5.2 Exp14.1 redesign

WCCE retains Core's shuffled DataLoader, batch 128. An independent structured auxiliary loader uses **8 classes × 8 distinct users × 1 unique segment = 64**. There are five task steps per epoch for 581 samples; one fresh auxiliary batch accompanies each task step. Sampling selects users first, then one actual segment from each chosen class-user cell. There are seven cross-user positives per anchor and no duplicated segment in a batch. Validation covered 303 seed/epoch pairs and 1515 auxiliary batches.

Each optimizer step combines task WCCE, Phase-CU on the independent auxiliary batch when enabled, and Prefix CE on the task batch when enabled. C0 bypasses auxiliary feature computation entirely. The phase grid is 0.01, 0.03, 0.06, 0.1; prefix grid 0.1, 0.25, 0.5. Planned combined cases were conditional on both families yielding eligible candidates. Final execution contains **no combined cases**, because Prefix had no eligible candidate.

## 6. Implementation

The inherited Core cache uses action 0 and action 1 handwriting, 64 Hz, 256 padded steps, and valid lengths. The actual schema is `custom_wavelet_polarity_split_abs_events_v1`: 30 unsigned event channels enter the SNN; six additional continuous channels from the 36-channel producer are not SNN inputs. The class order is A, B, C, D, E, G, H, I, J, K, L, X. Train users are 0, 1, 2, 5, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20; validation users 4, 9, 16; test users 3, 6, 10. There are 581/126/146 train/validation/test sequences. Split seed is 12345; model seeds are 11/23/37.

The inspected lock identifies `core_benchmark_v1.0`, identity `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`, dataset SHA-256 `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a`. Experiment loaders accept v1.0/v1.1 locks and adapt the protocol object; this does not mean the historical cache was regenerated as v1.1. Data roots are `outputs/action{0,1}_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded`.

The two-layer baseline has width 128 per layer and shifts 234×234: α = 1−2^(−shift) gives 0.75, 0.875, 0.9375, allocated 43/43/42 neurons per layer. At Δt=15.625 ms these imply synaptic e-folding times approximately 54.31, 117.01, 242.10 ms. Membrane τ is 22.54 ms, threshold 0.5, spike cap 1, subtractive reset. Signed bias-free feedforward matrices communicate spikes; there is no trainable recurrent matrix. Synaptic current and membrane carry temporal state.

The update is **unnormalized**: Iₜ = αIₜ₋₁ + Wxₜ, without a (1−α) drive factor. The output is signed bias-free evidence eₜ=Wout sᴸ²ₜ with an accumulator, no extra output synaptic filtering. Whole-sequence cross-entropy (WCCE) is CE of **valid-mean logits**; native prediction is argmax of valid-summed logits, which has the same argmax as their mean. Timestep CE (TSCE) instead computes CE at each valid timestep; Core averages within each sequence, then across sequences so longer sequences do not receive extra weight. Exp14/15 use WCCE, not TSCE. Adam uses learning rate 0.001, weight decay 0, ordinary task batch size 128 where preserved, maximum 100 epochs, minimum 20, patience 30. Epoch 0 is eligible. Checkpoints maximize held-out-user validation native BA, then minimize validation valid-mean-logit CE, then prefer the earliest exact tie. Training gradients minimize the training objective; neither test BA nor a fitted test probe selects checkpoints.

The train-only projection is 128→64 with ReLU→32, then L2 normalization. It does not enter native inference. For anchor i, positives P(i) share the true class and have a different user. Supervised contrastive loss averages −log[exp(zᵢ·zₚ/T)/Σⱼ≠ᵢ exp(zᵢ·zⱼ/T)] over those positives, temperature T=0.1. The denominator contains all nonself examples, including same-class/same-user rows that are not designated positives. That implementation choice is part of the protocol, not a generic claim about invariance.

Auxiliary weights are zero through epoch 10, ramp during epochs 11–30, then reach the configured λ. Exp14.1 Prefix is **0.5 CE(mean valid logits through 50%) + 0.5 CE(mean valid logits through 75%)**. Prefix step counts use `ceil(phase×length)`, minimum one. The initial README's 1/3 and 2/3 weights are stale; source, corrective commit `9f1da3c`, and Unity `protocol.json` agree on equal weights. No changed readout or inference-time projection is added.

Original Exp14 selects C2/C3 λ by mean native validation BA, then prefers smaller λ on ties. It selects C2=0.03 and C3=0.01; C4 full/half strengths follow. Exp14.1 has stronger family-specific eligibility rules (Section 9). Native checkpoint selection remains validation BA/CE, not retrieval or test-probe optimization within an individual training run.

## 7. Executed Runs and Validity Audit

Original Exp14 first preparation `65011225` failed exit 127. Its original log is missing, so a particular missing module/command is not asserted. The next preparation completed. The initial phase-1 array `65011247` had 37 failed tasks and five completed tasks; the train-only projector initializer caused auxiliary cases to fail. After correction, `65011321` recorded 42 completed tasks, then selection, six C4 runs, and finalization completed. The script returns early when a checkpoint already exists: five valid outputs from the first launch could be reused. A completed rerun task is therefore not proof of retraining that case from scratch. Only complete final artifacts under their case keys enter the 48-run result set.

Exp14.1 first array `65032380` failed all 24 tasks under an infeasible no-replacement four-segment auxiliary contract, including C0 diagnostic validation. PR89 fixed sampling to 8×8×1. The second array `65035050` completed 21 auxiliary cases but failed C0 tasks 0/8/16 on a historical checkpoint bitwise-reproduction assertion. A control correction made current initialization/task-contract checks authoritative and retained historical O0 as a nonblocking reference. Replacement C0 tasks `65037600_0/8/16` completed.

Selection `65037601` then failed because no Prefix candidate met safeguards and the old code raised an exception. The corrected outcome handling produced selection `65039702`, six selected final evaluations `65039703`, and successful finalization `65039704`. The final aggregate explicitly records 24 phase-1 candidates, **phase2_count=0**, **final_eval_count=6**, and only C0/P. An older protocol metadata field `final_eval_count=21` represents planned maximum topology and is superseded by the final manifest, not evidence of 21 executed evaluations.

## 8. Evaluation and Diagnostics

Core temporal probes freeze the SNN and train a separate multiclass logistic readout. WholeCount sums valid spikes (dimension H); Fixed250 concatenates 16 fixed 16-step/250-ms bins (dimension 16H); Relative10 concatenates 10 normalized-progress bins (dimension 10H, integer endpoints from `linspace`). Ordered probes preserve bin positions. Fixed250 shuffle permutes only complete valid bins, leaving a partial final bin and padding fixed; Relative10 shuffle permutes all ten bins. Each sample uses a stable sample-ID-keyed permutation. Train, validation, and test are all transformed under the same matched shuffle protocol and the decoder is refitted for each replicate. This measures access to order-specific information, not corruption of test data under an ordered-only decoder.

The C grid is 0.001, 0.01, 0.1, 1, 10, 100, selected on validation BA among converged candidates, favoring smaller C on ties. Both Core decoders use scale-only `StandardScaler(with_mean=False)`; `no_bias` sets `fit_intercept=False`, `affine` sets it true. Thus no-bias is origin-preserving; affine adds a constant class offset. A fitted probe is not the trained native head. Five shuffle seeds 101, 211, 307, 401, 503 are averaged within each model seed before the n=3 aggregate. Primary comparisons below are L2 spikes with no bias. Affine outputs are supplementary and are not silently substituted for this contract.

The Core description defines the common feature contract. Original Exp14 evaluates ordered/shuffled temporal probes; the final Exp14.1 aggregate evaluates WholeCount, ordered Fixed250, and ordered Relative10 only. Missing shuffled columns in the latter are marked “Not evaluated,” not filled from historical Core or original Exp14.

Original Exp14 geometry interpolates valid trajectories to 64 phase points and averages cosine distances. It samples one same-class/cross-user and one different-class/cross-user partner per eligible query; R is their mean-distance ratio. Retrieval uses **mean training class trajectories as prototypes**, not Exp13.1's class/user medoids. Exp14.1 validation retrieval queries validation trajectories against train prototypes; final retrieval uses held-out test writers.

Original Exp14's `history_generalization` diagnostic is easy to misread. Its column named `id_ba` uses three folds formed by **groups of training users**, not within-user segment folds as in Exp13.1. It uses centered affine logistic regression at fixed C=1 and balanced class weights on endpoint L2 Upre. Its “ID” evaluates held-out subsets of the training-user pool. Therefore `excess_seen_gain_pp` is a stored column name; it does not establish same-writer memorization or directly reproduce Exp13.1 H. Exp14.1 delegates to this diagnostic and inherits the same limitation. No cross-family averaging of those history-gain metrics is performed.

Exp14.1 evidence diagnostics preserve the native signed head. Prefix vectors are cumulative valid-summed evidence, margins are true-class logit minus maximum competing logit, and evidence-direction consistency is cosine between prefix vectors. Additive support is true-class evidence minus mean competing evidence per timestep, summed over [0,50%], (50,75%], (75,100%]. Cancellation is 1−|Σsegment support|/Σ|segment support|; adjacent sign reversal counts negative products among nonzero pairs. This separates additive cancellation from a max-competitor margin, whose identity can change over time.

## 9. Quantitative Results

### 9.1 Original Exp14 native classification: complete grid

All values are n=3 seed mean±SD, independently grouped by **case and both λ values**, from Unity `native_runs.csv` [E14-2]. Test results for the grid are exploratory; test BA did not select λ.

| case | lambda_class | lambda_history | Train BA (%) | Validation BA (%) | Test BA (%) |
| --- | --- | --- | --- | --- | --- |
| C0_wcce | 0.0 | 0.0 | 74.75 ± 5.86 | 52.65 ± 2.52 | 50.48 ± 4.02 |
| C1_whole_cu | 0.01 | 0.0 | 76.74 ± 4.31 | 53.79 ± 5.28 | 52.69 ± 4.85 |
| C2_phase_cu | 0.01 | 0.0 | 72.53 ± 4.82 | 52.79 ± 3.00 | 51.37 ± 2.01 |
| C3_history_delta | 0.0 | 0.01 | 74.47 ± 5.30 | 52.96 ± 2.14 | 51.27 ± 3.68 |
| C1_whole_cu | 0.03 | 0.0 | 72.59 ± 7.68 | 50.11 ± 4.68 | 52.78 ± 5.65 |
| C2_phase_cu | 0.03 | 0.0 | 78.19 ± 2.66 | 54.22 ± 3.70 | 52.06 ± 4.07 |
| C3_history_delta | 0.0 | 0.03 | 74.63 ± 1.72 | 52.14 ± 3.38 | 51.28 ± 2.87 |
| C1_whole_cu | 0.1 | 0.0 | 64.98 ± 5.47 | 49.90 ± 4.93 | 45.04 ± 1.03 |
| C2_phase_cu | 0.1 | 0.0 | 72.62 ± 4.19 | 52.82 ± 3.78 | 52.49 ± 2.98 |
| C3_history_delta | 0.0 | 0.1 | 66.93 ± 2.17 | 48.34 ± 4.51 | 45.83 ± 3.09 |
| C1_whole_cu | 0.3 | 0.0 | 60.42 ± 5.06 | 41.40 ± 3.46 | 42.78 ± 3.14 |
| C2_phase_cu | 0.3 | 0.0 | 63.17 ± 5.19 | 49.76 ± 1.93 | 44.29 ± 1.79 |
| C3_history_delta | 0.0 | 0.3 | 53.72 ± 3.81 | 41.52 ± 2.52 | 39.54 ± 3.37 |
| C5_shuffled_phase_control | 0.1 | 0.0 | 73.96 ± 7.83 | 53.42 ± 4.55 | 49.08 ± 4.41 |
| C4_combined_best | 0.03 | 0.01 | 76.04 ± 4.47 | 54.67 ± 2.73 | 49.60 ± 0.95 |
| C4_combined_half | 0.015 | 0.005 | 77.74 ± 3.20 | 54.70 ± 3.34 | 52.67 ± 3.32 |

C0 is **50.48 ± 4.02%** test BA under structured task sampling, substantially different from historical Core O0. Low-strength C1/C2/C3 yield small improvements over this C0; stronger supervision often damages both validation and test. Selected C2=0.03 reaches **52.06 ± 4.07%**; C4 full strength falls to **49.60 ± 0.95%**, while half strength reaches **52.67 ± 3.32%**. C1=0.03's favorable test value is not a validation-selected universal winner. C5's **49.08 ± 4.41%** provides a negative auxiliary-label control.

### 9.2 Original selected representation diagnostics

The original aggregate geometry/retrieval/history files omit λ in their row labels. Probe `run_key` also omits λ. This report resolves ambiguity from the **parent run directory** and never pools the whole λ grid under one case. Selected configurations use C2 .03, C3 .01, C4 .03/.01 and .015/.005, C5 .1 [E14-3]. No-bias L2 spike probe BA (%):

| Selected configuration | WholeCount | Fixed250 order | Fixed250 shuffle | Relative10 order | Relative10 shuffle |
| --- | --- | --- | --- | --- | --- |
| C0_wcce | 56.84 ± 2.48 | 54.50 ± 3.10 | 47.50 ± 1.68 | 65.18 ± 1.22 | 45.51 ± 1.84 |
| C2_phase_cu__lc0p03 | 53.92 ± 1.43 | 57.30 ± 0.46 | 48.64 ± 2.32 | 66.22 ± 1.81 | 47.30 ± 0.37 |
| C3_history_delta__lh0p01 | 55.99 ± 1.39 | 56.27 ± 1.55 | 48.20 ± 0.79 | 66.59 ± 0.53 | 45.41 ± 1.40 |
| C4_combined_best__lc0p03__lh0p01 | 53.80 ± 5.98 | 56.46 ± 1.38 | 48.20 ± 1.63 | 68.06 ± 3.24 | 46.05 ± 1.74 |
| C4_combined_half__lc0p015__lh0p005 | 56.13 ± 0.18 | 57.56 ± 1.48 | 48.39 ± 0.67 | 67.57 ± 2.06 | 47.31 ± 0.54 |
| C5_shuffled_phase_control__lc0p1 | 54.09 ± 0.60 | 57.07 ± 2.02 | 47.02 ± 1.72 | 65.61 ± 2.20 | 44.93 ± 0.67 |

Training-prototype retrieval (%), same selected run directories:

| Selected configuration | L2 spike retrieval (%) | L2 Upre retrieval (%) |
| --- | --- | --- |
| C0_wcce | 59.11 ± 1.15 | 59.14 ± 3.47 |
| C2_phase_cu__lc0p03 | 62.96 ± 1.41 | 62.88 ± 3.69 |
| C3_history_delta__lh0p01 | 60.79 ± 1.23 | 61.02 ± 3.14 |
| C4_combined_best__lc0p03__lh0p01 | 62.82 ± 1.40 | 62.26 ± 3.99 |
| C4_combined_half__lc0p015__lh0p005 | 61.85 ± 1.46 | 63.63 ± 2.98 |
| C5_shuffled_phase_control__lc0p1 | 57.28 ± 2.00 | 56.57 ± 2.75 |

Phase-CU improves L2 spike retrieval from ≈59.11% to ≈62.96%, and Upre retrieval from ≈59.14% to ≈62.88%. Yet WholeCount falls from ≈56.84% to ≈53.92%; ordered Relative10 changes modestly. Test geometry ratio R rises from C0 ≈1.2072 to C2≈1.2208 for spikes and 1.2905→1.3220 for Upre. C4 full has stronger diagnostic Relative10/retrieval but worse native test BA. This is the observed representation/readout mismatch, not evidence that a high-dimensional probe was learned by the native head.

For the original endpoint history diagnostic, full-history gains relative to 50 ms are C0 train-pool CV/OOD **37.07/39.87 pp**, C2 **43.23/42.16 pp**; the stored excess becomes −2.80→+1.07 pp. OOD history benefit rises by ≈2.29 pp, but train-pool CV benefit rises more. Because folds hold out training-user groups, this is not a direct seen-user memorization result.

### 9.3 Exp14.1 validation screening and selected cases

The table is recomputed from all 24 `validation_summary.json` files, grouped by configuration with n=3. These are **validation percentages**, not test outcomes [E14-4].

| Case | λ phase | λ prefix | Native val | WC val | Relative10 val | Prefix 50 val | Prefix 75 val | Retrieval val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_prefix_wcce | 0.0 | 0.1 | 55.70 ± 0.92 | 55.66 ± 0.73 | 66.20 ± 1.26 | 26.86 ± 2.42 | 45.47 ± 4.44 | 64.03 ± 3.92 |
| A_prefix_wcce | 0.0 | 0.25 | 55.36 ± 0.28 | 55.30 ± 1.52 | 66.16 ± 1.76 | 25.94 ± 2.84 | 44.46 ± 2.16 | 64.13 ± 3.62 |
| A_prefix_wcce | 0.0 | 0.5 | 52.84 ± 0.81 | 52.29 ± 1.40 | 63.85 ± 3.00 | 28.02 ± 2.94 | 45.69 ± 6.44 | 60.86 ± 1.44 |
| C0_dual_null | 0.0 | 0.0 | 57.46 ± 0.77 | 57.34 ± 1.90 | 67.60 ± 2.34 | 24.77 ± 1.67 | 43.37 ± 0.89 | 66.01 ± 2.36 |
| P_phase_cu | 0.01 | 0.0 | 56.97 ± 2.84 | 56.47 ± 4.71 | 67.05 ± 0.31 | 23.69 ± 2.29 | 42.48 ± 0.47 | 65.51 ± 4.36 |
| P_phase_cu | 0.03 | 0.0 | 57.18 ± 1.99 | 57.12 ± 0.76 | 68.84 ± 0.75 | 23.23 ± 2.08 | 41.03 ± 1.01 | 64.92 ± 4.48 |
| P_phase_cu | 0.06 | 0.0 | 52.98 ± 2.85 | 55.82 ± 3.04 | 67.11 ± 4.35 | 22.67 ± 2.48 | 38.00 ± 4.89 | 60.91 ± 8.16 |
| P_phase_cu | 0.1 | 0.0 | 53.48 ± 0.24 | 53.02 ± 1.11 | 68.37 ± 2.05 | 22.99 ± 1.92 | 39.54 ± 1.07 | 61.68 ± 4.33 |

Phase eligibility requires mean native validation BA ≥ C0−1 pp. Among eligible candidates, select maximal validation L2-spike retrieval; candidates within 0.5 pp of the maximum favor smaller λ. Phase .01 and .03 preserve native BA sufficiently; .01 has the stronger retrieval and is selected. Its retrieval is below C0, so selection does not establish a successful auxiliary improvement.

Prefix eligibility requires both native validation BA and Relative10 BA ≥ C0−1 pp; then maximize WholeCount, tie-break by smaller collapse gap and λ. **No Prefix λ passes.** Even .1 loses ≈1.77 pp native validation BA, while .5 raises 50%-prefix BA but substantially lowers final BA. The final selection is Phase **λ=.01**, Prefix `no_eligible_candidate`, combined `not_applicable`. Conditional J0/Jp/Ja/Jb combined cases were proposed and implemented but not executed.

### 9.4 Exp14.1 final held-out test

Only the selected phase strength and matched fresh C0 receive final test evaluation [E14-5]:

| case | lambda_phase | lambda_prefix | Train BA (%) | Validation BA (%) | Test BA (%) |
| --- | --- | --- | --- | --- | --- |
| C0_dual_null | 0.0 | 0.0 | 91.49 ± 2.06 | 57.46 ± 0.77 | 54.60 ± 5.69 |
| P_phase_cu | 0.01 | 0.0 | 89.18 ± 6.38 | 56.97 ± 2.84 | 56.70 ± 3.20 |

Mean native improvement is **+2.10 pp**, with paired seed deltas **−2.08, +0.95, +7.43 pp**. It is not stable across all seeds. These checkpoints are not the historical O0 baseline used by Exp15.

| case | WholeCount | Fixed250 order | Fixed250 shuffle | Relative10 order | Relative10 shuffle |
| --- | --- | --- | --- | --- | --- |
| C0_dual_null | 55.39 ± 5.16 | 59.53 ± 3.67 | Not evaluated | 66.49 ± 0.86 | Not evaluated |
| P_phase_cu | 55.07 ± 5.41 | 59.06 ± 3.02 | Not evaluated | 66.34 ± 2.95 | Not evaluated |

WholeCount mean changes **55.39→55.07%** (−0.31 pp); Relative10 **66.49→66.34%** (−0.15 pp). Collapse gap, defined as Relative10−WholeCount, is **11.11→11.27 pp**. Thus the native mean increase is not accompanied by a demonstrated closing of the accumulation gap.

| Case | L2 spike retrieval (%) | L2 Upre retrieval (%) |
| --- | --- | --- |
| C0_dual_null | 67.50 ± 4.99 | 67.80 ± 3.24 |
| P_phase_cu | 67.94 ± 6.80 | 69.09 ± 3.22 |

L2 spike retrieval changes ≈67.50→67.94% (+0.44 pp), Upre≈67.81→69.09% (+1.29 pp). These small differences, with three seeds, do not replicate the original study's larger diagnostic gains under an identical training protocol; the training samplers differ.

## 10. Training / Mechanistic Diagnostics

Individual validation-selected best and stopped epochs for all 72 original/redesigned training runs are in Appendix A. In Exp14.1 C0, current-core initialization is bitwise matched, task objective/loader seeds are preserved, and the auxiliary path is bypassed. Saved current C0 epochs are **100/85/89**, versus historical O0 **93/99/99**. Historical comparison reports mismatches in `layers.0.weight`, `layers.1.weight`, `head.weight` for all three seeds. This shows numerical nonidentity; the evidence does not isolate its exact environmental or training-history cause. Historical O0 is a sanity reference, not a substitute for the new C0.

At selected P λ=.01 best epochs 93/66/97, four task and auxiliary diagnostic batches give the following L2 gradient norms [E14-6]:

| Seed | Best epoch | ‖∇WCCE‖ | ‖∇Phase‖ | λ‖∇Phase‖ / ‖∇WCCE‖ | Cosine Phase/WCCE |
| --- | --- | --- | --- | --- | --- |
| 11 | 93 | 0.05004 | 2.08122 | 0.41595 | 0.13477 |
| 23 | 66 | 0.06328 | 1.82843 | 0.28896 | 0.16836 |
| 37 | 97 | 0.07779 | 2.22047 | 0.28544 | 0.18126 |

The auxiliary reaches the intended L2 parameters at a material weighted scale. Low positive cosine shows weak alignment with the task gradient, not zero gradient or universal conflict. Nonzero gradient cannot establish useful history selection or improved transfer.

Native prefix test diagnostics, using the selected checkpoints:

| Case | 50% BA | 75% BA | 100% BA | 100% mean margin |
| --- | --- | --- | --- | --- |
| C0_dual_null | 22.92 ± 0.62 | 40.39 ± 3.81 | 54.60 ± 5.69 | 0.0637 ± 0.1639 |
| P_phase_cu | 22.10 ± 3.22 | 40.62 ± 2.23 | 56.70 ± 3.20 | 0.0690 ± 0.1924 |

Early BA is not improved consistently: 50% C0≈22.92%, P≈22.10%; 75%≈40.39→40.62%. Final mean margin changes ≈0.0637→0.0690. Evidence organization remains similar:

| Case | Cosine 50→75 | Cosine 75→100 | Monotonic margin fraction | Support sign reversal | Support cancellation |
| --- | --- | --- | --- | --- | --- |
| C0_dual_null | 0.8872 ± 0.0052 | 0.9411 ± 0.0039 | 0.5639 ± 0.0143 | 0.1518 ± 0.0105 | 0.1076 ± 0.0048 |
| P_phase_cu | 0.8829 ± 0.0049 | 0.9406 ± 0.0041 | 0.5548 ± 0.0411 | 0.1522 ± 0.0089 | 0.1032 ± 0.0079 |

Monotonic margin fraction is not increased, and cancellation/sign-reversal changes are small. The data do not support a substantial reorganization of additive evidence despite functioning auxiliary gradients. Gradient magnitude, target geometry, and native evidence organization are different links in the causal chain.

## 11. Negative and Null Results

Strong original CU/history supervision reduces native generalization. Selected original full-strength C4 improves some diagnostics while lowering native BA. Shuffled labels do not yield a consistent useful effect. In Exp14.1, all Prefix candidates fail final-preservation eligibility; combined training is consequently absent. Selected Phase has a modest mean native test gain but mixed seed signs, near-flat WholeCount/Relative10, and no closed collapse gap. “The dual loader solved generalization” is not a supported conclusion.

The absence of Prefix test results is intentional protocol screening, not a forgotten experiment. Its validation failure is itself evidence that earlier classification pressure can trade against final evidence. No nonexistent combined result or rejected prefix test result is supplied.

## 12. Interpretation

### 12.1 Direct observations

Original Phase-CU improves selected cross-user representation diagnostics under structured task sampling; native gains are small and strength-sensitive. Dual-loader Phase .01 yields a +2.10-pp mean test BA difference with a negative seed, no meaningful temporal-to-count gap reduction, and small retrieval differences. Prefix boosts some early validation BA while failing preservation constraints. L2 receives auxiliary gradients.

### 12.2 Mechanistic interpretation

The results are consistent with auxiliary objectives reshaping class geometry without reliably aligning temporal features for the signed accumulator. The original projection can facilitate contrastive similarity while leaving native evidence partly incompatible across time. Separating loaders removes an avoidable sampling confound but does not, by itself, resolve the representation/readout problem. The later September 30 discussion described zₜ=F(history) and eₜ=Wzₜ: changing class separation alone does not ensure that successive eₜ contribute compatible support.

The initial result interpretation—“Phase-CU helps geometry”—survives in the original protocol. The stronger claim that this solves transferable accumulation is weakened by C4, the dual-loader final metrics, and failed Prefix safeguards. This is a revision of the working hypothesis, not an implementation-only failure.

### 12.3 What the experiment does not establish

It does not prove harmful historical content is removed, that user invariance is achieved, or that larger λ would solve the bottleneck. A train-only projection's objective is not the deployed representation itself. Probe BA cannot be substituted for native inference. Historical O0 bitwise mismatch is not diagnosed as a hardware cause. A three-seed native mean gain with flat probes does not identify which mechanism improved. The `id_ba` name does not make this study's history CV a within-user seen-writer test.

## 13. Bugs, Corrections, and Reruns

| Issue | Affected jobs/runs | Consequence | Fix / implementation evidence | Final valid replacement |
| --- | --- | --- | --- | --- |
| Preparation exit 127 | 65011225 | No usable preparation from this launch | Exact cause unrecovered; later setup succeeds | 65011246/65011320 |
| Projection initializer | 37/42 tasks in 65011247 | Auxiliary cases fail; five completed outputs can remain valid | `2f0272d` fixes train-only projection initialization | 65011321 completed array plus valid existing outputs |
| 4 segments without replacement infeasible | All 24 tasks in 65032380 | Sparse class-user cells invalidate proposed sampling, including null diagnostics | PR89 / `25996d8`: 8 users × one actual unique segment | 21 auxiliary runs in 65035050 |
| Overstrict historical C0 reproduction | Tasks 0/8/16 in 65035050 | Fresh controls fail after training due to historical equality assertion | `198a175`: current contract authoritative, old checkpoint nonblocking | 65037600_0/8/16 |
| No-eligible Prefix treated as exception | Selection 65037601 | Valid negative selection blocked downstream finalization | `3c4ef7d`: record negative selection and dynamic final specs | 65039702/03/04, six final evaluations |
| Stale README/plan counts | Unequal prefix weights / historical bitwise requirement / maximum final count | Documentation could misstate executed design | Source/protocol/checkpoint/final manifest override stale text | Equal weights; current C0; final count six |

The sparse-cell concern was explicit in the Sep29 18:29–18:33 PR89 discussion: 68 of 168 train class-user cells contain fewer than four segments. Original Exp14 samples with replacement in those cells; the first 14.1 no-replacement plan is infeasible. The fix changes geometry of the auxiliary batch, not the native task batch. C0 bypasses auxiliary validation/feature computation during training after the control correction.

Failed outputs are excluded. Completed auxiliary checkpoints from the second 14.1 launch remain valid after the C0/selection-only corrections; there is no reason to combine first-array failed runs or overwrite them conceptually. Chat reports canceled dependency jobs after failures, but the available `sacct --allocations` capture does not provide a complete record of pending cancellations. Missing original failure logs prevent reproducing every traceback.

## 14. Relationship to Previous Experiments

| Component | Inherited | Modified in Exp14 / Exp14.1 |
| --- | --- | --- |
| Split/cache | Locked Core users, samples, hash, seeds | None |
| Backbone/dynamics | Core two-layer width 128, 234×234, bias-free accumulator | Train-only projection; backbone remains native architecture |
| Initialization | Paired fresh Core-style initialization | No continuation of O0; new C0 controls |
| Objective | WCCE task objective | CU / history-delta; then phase and prefix terms |
| Task sampler | Ordinary Core in 14.1 | Original Exp14 replaces it with structured sampling |
| Auxiliary sampler | No Core auxiliary loader | Original8×4×4; corrected14.1 independent8×8×1 |
| Probe | Core temporal features and bias distinction | Additional trajectory geometry/retrieval and separate history-CV protocol |
| Selection | Native validation BA/CE | Configuration/family validation safeguards; no test-driven λ choice |

Exp13 history-access results motivate organization but are not treated as a proof of nuisance invariance. Core historical native **57.93 ± 2.23%**, original structured C0 **50.48 ± 4.02%**, and fresh dual-null C0 **54.60 ± 5.69%** are different checkpoints/protocols. Only within-study matched controls support effect-size claims.

## 15. How This Led to the Next Experiment

After consistency training, the unresolved question was whether all incoming history should be written into L2 at every timestep. Auxiliary losses operate on resulting features; even a useful cross-user geometry does not choose which input deserves memory. Prefix supervision also failed to preserve final performance. A competing explanation remained that the backbone and native head could adapt to loss changes while retaining an indiscriminate history stream.

The September 30 19:49–20:12 discussions therefore moved toward a context-dependent scalar write gate. Exp15 adds it to an existing native baseline and tests both frozen-backbone gate training and joint training. Frozen training isolates what the gate alone can accomplish; ungated continuation controls extra optimization in the joint case. Inference interventions later distinguish learned timing/history input from simple gain. This directly targets memory writing rather than merely supervising class geometry. Prefix was deferred and **is not part of Exp15**. See [the Exp15 report](../exp15/report.md).

## 16. Final Conclusions

Exp14 demonstrates that auxiliary consistency can improve selected representation diagnostics without reliable native accumulated-evidence gains. The corrected dual-loader study removes a task-sampling confound, records Prefix rejection as a valid outcome, and obtains only a modest, seed-variable selected Phase effect. Nonzero gradients and improved class geometry are insufficient evidence that useful history has been selected or organized for native inference. That remaining mechanism motivated a direct write-gate test.

## Appendix A. Run Inventory

### A.1 Every final training checkpoint

Rows identify all 48 original Exp14 and 24 Exp14.1 checkpoints. Directory names encode full λ values. `checkpoint.pt` is in each linked directory; best/stopped epochs and full SHA-256 were read from Unity. Epochs are fresh-training epochs. For original C0/C5, the listed final rerun array may reuse an already complete checkpoint; this is explicitly separate from claiming a new training trajectory. Original cases have train/val/test evaluations; rejected 14.1 candidates remain validation-only.

| Run directory / family | Seed | Best epoch | Stopped epoch | Final valid task or source | Scope / validity | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [14 / H128 / C0_wcce__seed11](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C0_wcce__seed11) | 11 | 78 | 100 | 65011321_0 | Valid train/val/test | `b5d30d7deec64b16a793b778f21d314cfd5aff0332ac2df4ad1349f6696a8240` |
| [14 / H128 / C0_wcce__seed23](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C0_wcce__seed23) | 23 | 96 | 100 | 65011321_14 | Valid train/val/test | `90f469f3b9c762301de449f53bda83cd0e3f78276bdf48cdf1d00a64a81e7898` |
| [14 / H128 / C0_wcce__seed37](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C0_wcce__seed37) | 37 | 90 | 100 | 65011321_28 | Valid train/val/test | `8ca380cb3508f3304c05261996f5f3a6508625ac135a8749ae468872b8c5aa21` |
| [14 / H128 / C1_whole_cu__seed11__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed11__lc0p01) | 11 | 97 | 100 | 65011321_1 | Valid train/val/test | `de2c54d58ed57d32e54990524841152c826ce753d785c26d33e2247f9d317c89` |
| [14 / H128 / C1_whole_cu__seed11__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed11__lc0p03) | 11 | 78 | 100 | 65011321_4 | Valid train/val/test | `7317f6b0b797fad579f2be1dfb558ddb59edc2cbb17824e405d6d4a10b9256e1` |
| [14 / H128 / C1_whole_cu__seed11__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed11__lc0p1) | 11 | 81 | 100 | 65011321_7 | Valid train/val/test | `758d298128c8544bf77aac76fe8ccf33ef5a30641524c062170bd925ae138c44` |
| [14 / H128 / C1_whole_cu__seed11__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed11__lc0p3) | 11 | 98 | 100 | 65011321_10 | Valid train/val/test | `45e61f8d6a6a0ad6e2838284b58177e84eb3d95c807449d46d82b210c8e251ab` |
| [14 / H128 / C1_whole_cu__seed23__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed23__lc0p01) | 23 | 100 | 100 | 65011321_15 | Valid train/val/test | `29794be1b784252187d580f7ee8b0255a22aa74a6b46b58754bc1c38ba185523` |
| [14 / H128 / C1_whole_cu__seed23__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed23__lc0p03) | 23 | 99 | 100 | 65011321_18 | Valid train/val/test | `a9d107eb747669bbc9e7a61652c86052bff65b6072f5cf211f7073d96778266b` |
| [14 / H128 / C1_whole_cu__seed23__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed23__lc0p1) | 23 | 95 | 100 | 65011321_21 | Valid train/val/test | `158fdb19c29750f560c4f71ce05a390e9f22a79be591cc41ffef1e500fe89d74` |
| [14 / H128 / C1_whole_cu__seed23__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed23__lc0p3) | 23 | 100 | 100 | 65011321_24 | Valid train/val/test | `b8d19a4d76a4d9158d2f571c0250a6d8cb90af6cd4a03722f1945734a3f9d22d` |
| [14 / H128 / C1_whole_cu__seed37__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed37__lc0p01) | 37 | 78 | 100 | 65011321_29 | Valid train/val/test | `b18e04ba5887dcce0f0fdda4d815d76c9ce8d89c5ac6e2e704e428111e3fad84` |
| [14 / H128 / C1_whole_cu__seed37__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed37__lc0p03) | 37 | 73 | 100 | 65011321_32 | Valid train/val/test | `00a3c623bfbf37bbee6cca3974ba89b03b438603557224a7bdb8b34695931592` |
| [14 / H128 / C1_whole_cu__seed37__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed37__lc0p1) | 37 | 84 | 100 | 65011321_35 | Valid train/val/test | `72719da08d3c7afa6406924da5a7ac0919ad97c301b42da14a5b410f10ac0ea6` |
| [14 / H128 / C1_whole_cu__seed37__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C1_whole_cu__seed37__lc0p3) | 37 | 98 | 100 | 65011321_38 | Valid train/val/test | `7146b0d090e68c6ec4427e21a66971c27117f9ac8d8cf8c923d749fe01ee6abb` |
| [14 / H128 / C2_phase_cu__seed11__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed11__lc0p01) | 11 | 77 | 100 | 65011321_2 | Valid train/val/test | `0d9f412d1795e086c7b9dfb59371aef20c433e6edee1639461f1231178538f3c` |
| [14 / H128 / C2_phase_cu__seed11__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed11__lc0p03) | 11 | 98 | 100 | 65011321_5 | Valid train/val/test | `38d68c4f57cb2aa38a93c8ba37bb844900ebbe3bb631596db1b6653d10b5b49c` |
| [14 / H128 / C2_phase_cu__seed11__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed11__lc0p1) | 11 | 97 | 100 | 65011321_8 | Valid train/val/test | `72a46c8ec9bbf868aa9cd318967ca42e754d3ef71a8b6b5ee038a6dda4b32a65` |
| [14 / H128 / C2_phase_cu__seed11__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed11__lc0p3) | 11 | 80 | 100 | 65011321_11 | Valid train/val/test | `eb0d34dd4b79e3c701d35ad3c2d561ebac63056157ded721d723bc1a286f3447` |
| [14 / H128 / C2_phase_cu__seed23__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed23__lc0p01) | 23 | 90 | 100 | 65011321_16 | Valid train/val/test | `24ced071c82c13c34bf80e9fb905c426841ea454c4b96c52a2b1c5a611d73608` |
| [14 / H128 / C2_phase_cu__seed23__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed23__lc0p03) | 23 | 100 | 100 | 65011321_19 | Valid train/val/test | `a300c843161392dd27951b519fc762ce26d80c19500316071a4be72afcde0b6e` |
| [14 / H128 / C2_phase_cu__seed23__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed23__lc0p1) | 23 | 100 | 100 | 65011321_22 | Valid train/val/test | `8cbbaa6c253045da341451426d90e2e1c932f479f749a20c149b676be3bbbc83` |
| [14 / H128 / C2_phase_cu__seed23__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed23__lc0p3) | 23 | 87 | 100 | 65011321_25 | Valid train/val/test | `d3f73cccbe641d9fd32dc75cec73fc0fef2c629935ca299ddb3a8de66e204104` |
| [14 / H128 / C2_phase_cu__seed37__lc0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed37__lc0p01) | 37 | 75 | 100 | 65011321_30 | Valid train/val/test | `c1d0568ac79a8f129b24f3aa7ef3774cdc2aec101218565c8db1bcbf505edf9d` |
| [14 / H128 / C2_phase_cu__seed37__lc0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed37__lc0p03) | 37 | 90 | 100 | 65011321_33 | Valid train/val/test | `7a086cb4a2a23055eb61919ca66686c312b1a175b30b6dcb79614fb9cc7a841f` |
| [14 / H128 / C2_phase_cu__seed37__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed37__lc0p1) | 37 | 77 | 100 | 65011321_36 | Valid train/val/test | `8ddd2bab75b121e3028a1c0cbc900324302016e996bfd86ab8977f682293e8bb` |
| [14 / H128 / C2_phase_cu__seed37__lc0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C2_phase_cu__seed37__lc0p3) | 37 | 91 | 100 | 65011321_39 | Valid train/val/test | `8f13315b163d6cbb365b43ee3355d215263b9c3f64b8484e3051880682e598e6` |
| [14 / H128 / C3_history_delta__seed11__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed11__lh0p01) | 11 | 78 | 100 | 65011321_3 | Valid train/val/test | `f0b7b430766ba965d73d244dcca0d02de3acf6a6acd85a761c897f4a00ee6113` |
| [14 / H128 / C3_history_delta__seed11__lh0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed11__lh0p03) | 11 | 98 | 100 | 65011321_6 | Valid train/val/test | `02dcafbda5fbe1403b647956bf27408ac733f2e5abe62807b5f1832e137535ad` |
| [14 / H128 / C3_history_delta__seed11__lh0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed11__lh0p1) | 11 | 95 | 100 | 65011321_9 | Valid train/val/test | `6067a196228e309a43fb4913d58f3792a7e2ebdc159b526d7a073df2ee0f4573` |
| [14 / H128 / C3_history_delta__seed11__lh0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed11__lh0p3) | 11 | 91 | 100 | 65011321_12 | Valid train/val/test | `996340cc8c0295ea31ee09dd56b63a58fd36145e6696e8487dcd8bc118d9d000` |
| [14 / H128 / C3_history_delta__seed23__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed23__lh0p01) | 23 | 96 | 100 | 65011321_17 | Valid train/val/test | `dd6ce7897796a88800cb3950a5e30e8cd55fd3c20d500dac510628789eafef66` |
| [14 / H128 / C3_history_delta__seed23__lh0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed23__lh0p03) | 23 | 96 | 100 | 65011321_20 | Valid train/val/test | `c0d8dfad876b26db26cb96b286efb2dd6b5250077f8fe4e8c59565e7e01b9716` |
| [14 / H128 / C3_history_delta__seed23__lh0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed23__lh0p1) | 23 | 97 | 100 | 65011321_23 | Valid train/val/test | `f01f862e58a75d13e11def383c1d9df82d8352e91fadb8cdfdb225637c2e6343` |
| [14 / H128 / C3_history_delta__seed23__lh0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed23__lh0p3) | 23 | 99 | 100 | 65011321_26 | Valid train/val/test | `7c83bf591b20ef033a05efff2865cb92564b64cea9347ad4a986574712178f39` |
| [14 / H128 / C3_history_delta__seed37__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed37__lh0p01) | 37 | 90 | 100 | 65011321_31 | Valid train/val/test | `3ae1f4a946c40ce45509e465d6b73d789ddebe4999aa27274f52bd371c782fed` |
| [14 / H128 / C3_history_delta__seed37__lh0p03](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed37__lh0p03) | 37 | 90 | 100 | 65011321_34 | Valid train/val/test | `bf873d5792de958b15ffc427838cfa44b77a547bb275716b6a29543101bf13e9` |
| [14 / H128 / C3_history_delta__seed37__lh0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed37__lh0p1) | 37 | 97 | 100 | 65011321_37 | Valid train/val/test | `d62696c38cf8b29c30e8a62dd306f4858ba85ed66e33c010cbcf05d30d149cf3` |
| [14 / H128 / C3_history_delta__seed37__lh0p3](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C3_history_delta__seed37__lh0p3) | 37 | 75 | 100 | 65011321_40 | Valid train/val/test | `933cb43a778d40b2571de092f77a7c9d356ba56e900aed061d3ba7b29ad70d6c` |
| [14 / H128 / C4_combined_best__seed11__lc0p03__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_best__seed11__lc0p03__lh0p01) | 11 | 98 | 100 | 65011323_0 | Valid train/val/test | `d2bbf75cf372c9a345955f1f7cf26242dca633c370854f699acedd4b63b44c43` |
| [14 / H128 / C4_combined_best__seed23__lc0p03__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_best__seed23__lc0p03__lh0p01) | 23 | 97 | 100 | 65011323_2 | Valid train/val/test | `2dfe2a82f773719837497dc179683bbf1f5ad81458599109b6c6406d1ea406fe` |
| [14 / H128 / C4_combined_best__seed37__lc0p03__lh0p01](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_best__seed37__lc0p03__lh0p01) | 37 | 77 | 100 | 65011323_4 | Valid train/val/test | `d69f91ea1d30b6a5988beb105c885717ba4ac8523b9e7ca8f7ac54e9f0e08f33` |
| [14 / H128 / C4_combined_half__seed11__lc0p015__lh0p005](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_half__seed11__lc0p015__lh0p005) | 11 | 98 | 100 | 65011323_1 | Valid train/val/test | `2179ac74ac069b418f345dfa05e75d07950fe7600ecb6ab277df23aca8c232ae` |
| [14 / H128 / C4_combined_half__seed23__lc0p015__lh0p005](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_half__seed23__lc0p015__lh0p005) | 23 | 100 | 100 | 65011323_3 | Valid train/val/test | `50bf853e0ca6d8875e1b19b5fef97f1a26c348f47908f06b98c2354fc18b27c0` |
| [14 / H128 / C4_combined_half__seed37__lc0p015__lh0p005](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C4_combined_half__seed37__lc0p015__lh0p005) | 37 | 90 | 100 | 65011323_5 | Valid train/val/test | `f958cd2c2b3cca1c186695ed71cf847e9ba67b3c6ac82beeb7813fc52e534d71` |
| [14 / H128 / C5_shuffled_phase_control__seed11__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C5_shuffled_phase_control__seed11__lc0p1) | 11 | 77 | 100 | 65011321_13 | Valid train/val/test | `f6f0920bcec3d1927973b0c6b3b76b4cb4c8a03cee33617487a0d3c18c1e18a6` |
| [14 / H128 / C5_shuffled_phase_control__seed23__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C5_shuffled_phase_control__seed23__lc0p1) | 23 | 100 | 100 | 65011321_27 | Valid train/val/test | `f17b864e02fc32f99cf4dea2a474fe453deefe7c5102694f9cf8a65e429c9036` |
| [14 / H128 / C5_shuffled_phase_control__seed37__lc0p1](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/runs/C5_shuffled_phase_control__seed37__lc0p1) | 37 | 90 | 100 | 65011321_41 | Valid train/val/test | `cfa3dcb3d748064a3e24a32cf13e601cc360237fa9b4585250b388bc775edcf2` |
| [14.1 / H128 / A_prefix_wcce__seed11__la0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed11__la0p1) | 11 | 95 | 100 | 65035050_5 | Valid train/val only; not selected for final test | `1edf85d719cad42dcf6ebe03e3426c01ef0904a31b0dd2eb023b04c3db3f6996` |
| [14.1 / H128 / A_prefix_wcce__seed11__la0p25](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed11__la0p25) | 11 | 98 | 100 | 65035050_6 | Valid train/val only; not selected for final test | `ab048012f888e0a7953a0c6ada48d96697bb0bece64303bddbd025a95ef284a6` |
| [14.1 / H128 / A_prefix_wcce__seed11__la0p5](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed11__la0p5) | 11 | 93 | 100 | 65035050_7 | Valid train/val only; not selected for final test | `f96bc0ae3909dad4c27e9ac44f1cad3b53064285f5721eef34229ec4a66aaa5d` |
| [14.1 / H128 / A_prefix_wcce__seed23__la0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed23__la0p1) | 23 | 78 | 100 | 65035050_13 | Valid train/val only; not selected for final test | `cebf102ba86894cc27fa8a72b36085ac999e321cb5292df81562829523ab9bc2` |
| [14.1 / H128 / A_prefix_wcce__seed23__la0p25](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed23__la0p25) | 23 | 92 | 100 | 65035050_14 | Valid train/val only; not selected for final test | `b22207b86165a45ccfe31eafe341c40740913ff52d2d5b0944f9d2f57372df0f` |
| [14.1 / H128 / A_prefix_wcce__seed23__la0p5](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed23__la0p5) | 23 | 90 | 100 | 65035050_15 | Valid train/val only; not selected for final test | `508edd04e5fcdbbf0524d6e5f1ac0d1638b64ae12426c7795a06fc17c3ad9dd7` |
| [14.1 / H128 / A_prefix_wcce__seed37__la0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed37__la0p1) | 37 | 99 | 100 | 65035050_21 | Valid train/val only; not selected for final test | `8ca8ee1e85718e55178de0d519143ba55fe094207e4124afcfc9112c50e2a4bb` |
| [14.1 / H128 / A_prefix_wcce__seed37__la0p25](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed37__la0p25) | 37 | 82 | 100 | 65035050_22 | Valid train/val only; not selected for final test | `5eb88ceca50c57fae1195fd9be0e1d3d2c9cfad0d43a160fd94191496ea23911` |
| [14.1 / H128 / A_prefix_wcce__seed37__la0p5](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/A_prefix_wcce__seed37__la0p5) | 37 | 97 | 100 | 65035050_23 | Valid train/val only; not selected for final test | `756af7f45e523bc6e503e5665a50d650fcc059307e0e7d0c6f39bbc19e8f9333` |
| [14.1 / H128 / C0_dual_null__seed11](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/C0_dual_null__seed11) | 11 | 100 | 100 | 65037600_0 | Valid train/val/test | `c17e81571748dff14563bdb7d4855a1c677c4c9b4bf8c6acacb3877c4da9a6f7` |
| [14.1 / H128 / C0_dual_null__seed23](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/C0_dual_null__seed23) | 23 | 85 | 100 | 65037600_8 | Valid train/val/test | `f271b1a974793310456e20204dace7e2642d12f2c00edbb6f06b09e1c19f4609` |
| [14.1 / H128 / C0_dual_null__seed37](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/C0_dual_null__seed37) | 37 | 89 | 100 | 65037600_16 | Valid train/val/test | `d2742b41e575e59f7065b74f56acf8b3e1c7d3b20abf19c1d9e0ba2d9419390a` |
| [14.1 / H128 / P_phase_cu__seed11__lp0p01](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed11__lp0p01) | 11 | 93 | 100 | 65035050_1 | Valid train/val/test | `3aac693835b041d8a88f4257b0cc958a9e4bf421d9701ff55753c38c84fed498` |
| [14.1 / H128 / P_phase_cu__seed11__lp0p03](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed11__lp0p03) | 11 | 99 | 100 | 65035050_2 | Valid train/val only; not selected for final test | `871b819c32f6954648601c69b15d8b496e143725b0c9395220d2a38ad4ca3263` |
| [14.1 / H128 / P_phase_cu__seed11__lp0p06](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed11__lp0p06) | 11 | 93 | 100 | 65035050_3 | Valid train/val only; not selected for final test | `27b9a48b2ceea5f297f153b384459105f7e42f5c0ae1805e47715ffb79a2fd47` |
| [14.1 / H128 / P_phase_cu__seed11__lp0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed11__lp0p1) | 11 | 98 | 100 | 65035050_4 | Valid train/val only; not selected for final test | `a730fc4d48629074b557893c2f128783c775bb6f0590c01a22a1649531fe1b57` |
| [14.1 / H128 / P_phase_cu__seed23__lp0p01](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed23__lp0p01) | 23 | 66 | 96 | 65035050_9 | Valid train/val/test | `6817a1cf48fc9ebd7ec7d201b1b619ac51f987dcae03b70a46ecc3b50e4dc0e7` |
| [14.1 / H128 / P_phase_cu__seed23__lp0p03](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed23__lp0p03) | 23 | 83 | 100 | 65035050_10 | Valid train/val only; not selected for final test | `fb82f641f4db8a222eb811e922e44b6498f6bc79810958350b5ec1f38c2b239c` |
| [14.1 / H128 / P_phase_cu__seed23__lp0p06](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed23__lp0p06) | 23 | 72 | 100 | 65035050_11 | Valid train/val only; not selected for final test | `392f937792e63667aa2254aec237d24354528a5c66551ebeaf16718a778b5a12` |
| [14.1 / H128 / P_phase_cu__seed23__lp0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed23__lp0p1) | 23 | 100 | 100 | 65035050_12 | Valid train/val only; not selected for final test | `f76b32af05ec1cae543aa14e3c42641c4e7bf42b6e0159c2d08af3f721c2f178` |
| [14.1 / H128 / P_phase_cu__seed37__lp0p01](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed37__lp0p01) | 37 | 97 | 100 | 65035050_17 | Valid train/val/test | `26b357feadf50200df11ab12311a1a889db959231ea9dba74a06d32e9d46055e` |
| [14.1 / H128 / P_phase_cu__seed37__lp0p03](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed37__lp0p03) | 37 | 93 | 100 | 65035050_18 | Valid train/val only; not selected for final test | `3ffed6380851db644d60ac263adc02ff61802a2f29831273fd36f31fd2c03ea2` |
| [14.1 / H128 / P_phase_cu__seed37__lp0p06](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed37__lp0p06) | 37 | 37 | 67 | 65035050_19 | Valid train/val only; not selected for final test | `9599c40b878ccb718af65ae3122fb0284ab77e9fdc0c3cd5ec9c829bdb76331d` |
| [14.1 / H128 / P_phase_cu__seed37__lp0p1](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/runs/P_phase_cu__seed37__lp0p1) | 37 | 86 | 100 | 65035050_20 | Valid train/val only; not selected for final test | `d17b08dfd4c97e5689e07330ad33cdd0784e5aa16bd3f6b2f883a4c8c47bf6f0` |

### A.2 Scheduler accounting

| Job / array base | Name | Recorded tasks | States | Exit codes | First start (Unity local timestamp) | Last end |
| --- | --- | --- | --- | --- | --- | --- |
| 65011225 | exp14_prepare | 1 | FAILED: 1 | 127:0 | 2026-09-28T22:20:02 | 2026-09-28T22:20:04 |
| 65011246 | exp14_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-28T22:22:12 | 2026-09-28T22:22:48 |
| 65011247 | exp14_p1 | 42 | COMPLETED: 5; FAILED: 37 | 0:0, 1:0 | 2026-09-28T22:23:17 | 2026-09-28T22:53:46 |
| 65011320 | exp14_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-28T22:25:27 | 2026-09-28T22:25:39 |
| 65011321 | exp14_p1 | 42 | COMPLETED: 42 | 0:0 | 2026-09-28T22:26:00 | 2026-09-28T23:14:10 |
| 65011322 | exp14_select | 1 | COMPLETED: 1 | 0:0 | 2026-09-28T23:14:14 | 2026-09-28T23:14:27 |
| 65011323 | exp14_p2 | 6 | COMPLETED: 6 | 0:0 | 2026-09-28T23:14:46 | 2026-09-28T23:47:31 |
| 65011324 | exp14_final | 1 | COMPLETED: 1 | 0:0 | 2026-09-28T23:47:49 | 2026-09-28T23:48:05 |
| 65032379 | exp14_1_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-29T17:26:10 | 2026-09-29T17:27:04 |
| 65032380 | exp14_1_p1 | 24 | FAILED: 24 | 1:0 | 2026-09-29T17:27:15 | 2026-09-29T17:29:03 |
| 65035049 | exp14_1_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-29T18:32:52 | 2026-09-29T18:33:16 |
| 65035050 | exp14_1_p1 | 24 | COMPLETED: 21; FAILED: 3 | 0:0, 1:0 | 2026-09-29T18:33:56 | 2026-09-29T18:46:29 |
| 65037600 | exp14_1_p1 | 3 | COMPLETED: 3 | 0:0 | 2026-09-29T19:19:31 | 2026-09-29T19:24:28 |
| 65037601 | exp14_1_select | 1 | FAILED: 1 | 1:0 | 2026-09-29T19:24:59 | 2026-09-29T19:25:13 |
| 65039702 | exp14_1_select | 1 | COMPLETED: 1 | 0:0 | 2026-09-29T20:24:10 | 2026-09-29T20:24:25 |
| 65039703 | exp14_1_eval | 6 | COMPLETED: 6 | 0:0 | 2026-09-29T20:24:42 | 2026-09-29T20:39:17 |
| 65039704 | exp14_1_final | 1 | COMPLETED: 1 | 0:0 | 2026-09-29T20:39:22 | 2026-09-29T20:39:47 |

### A.3 Task mapping and selected evaluation scope

Original phase1 orders each seed block of 14 as C0, then [C1,C2,C3] for λ=.01/.03/.1/.3, then C5. Array index is 14s+configuration index. Phase2 orders each seed's C4 best/half, index 2s+strength. Exp14.1 orders each seed block of eight as C0, P .01/.03/.06/.1, A .1/.25/.5. Initial/final candidate index is 8s+configuration index; C0 replacements use indices 0,8,16. Final evaluation `65039703` orders [C0,P .01] for each seed, indices 2s and 2s+1. Every selected evaluation preserves its source checkpoint; it is not additional training.

Unity-native task statuses establish completion; source specs and run directory keys map task identities. Selection and final manifests determine validity. `phase1_count=24` is not `final_eval_count=24`. Test screening was not performed for rejected Prefix or unselected phase strengths in Exp14.1.

### A.4 Numerical audit

Native tables group case/λ and require exactly seeds11/23/37. Original diagnostic tables load the selected per-run directories because aggregate rows omit λ. Shuffle probe replicates are averaged within model seed. All 14.1 validation candidates are kept separate from final test candidates. Checkpoint metadata records the current C0 contract and historical mismatch rather than substituting old O0 numbers. The full failed arrays and selection failures are preserved in accounting, and none enters final metric averages.

## Appendix B. Configuration / Case Definitions

| Case/diagnostic | Definition and boundary |
| --- | --- |
| C0_wcce (original) | Fresh native WCCE with structured8×4×4 task batch, no auxiliary objective |
| C1_whole_cu | Contrastive valid-mean projected L2 Upre |
| C2_phase_cu | Mean of contrastive losses at25/50/75/100% Upre |
| C3_history_delta | Projected endpoint Upre(full)−Upre(reset retaining250ms) |
| C4_combined_best | C2 λ=.03 plus C3 λ=.01, selected using native validation |
| C4_combined_half | C2 λ=.015 plus C3 λ=.005 |
| C5_shuffled_phase_control | C2-style λ=.1 with shuffled auxiliary labels, true WCCE labels |
| C0_dual_null | Fresh current Core task initialization/loader/objective with auxiliary bypass |
| P_phase_cu | Original phase feature definition on separate corrected64-sample aux batch |
| A_prefix_wcce | Equal50%/75% CE on task batch; λ=.1/.25/.5 |
| J0/Jp/Ja/Jb | Conditional full/half phase/prefix combinations; not executed |
| Collapse gap | Ordered Relative10 no-bias BA−WholeCount no-bias BA, in pp |
| Margin monotonicity | Fraction with margin50≤margin75≤margin100; not monotonic every timestep |
| Cancellation | 1−abs(sum three supports)/max(sumabs,ε); all-zero support maps to1 by formula, not evidence of opposing signs |
| Sign reversal | Adjacent segment products<0 among nonzero products; zeros excluded |
| Gradient ratio | Ramped auxiliary weight × auxiliary L2 gradient norm / WCCE L2 norm |
| History CV “ID” | Holds out groups of train users; fixed C=1 centered affine balanced logistic; differs from Exp13.1 within-user ID |

## Appendix C. Evidence Map

| ID / central claim | Implementation / revision | Unity quantitative or execution source | Project reasoning coverage |
| --- | --- | --- | --- |
| E14-1: access/accumulator gap | [Core model](../../core_benchmark_v1/model.py), [probes](../../core_benchmark_v1/probes.py) | `core_benchmark_v1/results/main/aggregate/native_runs.csv`, `probe_summary.csv`, locked dataset metadata | Sep27 benchmark; Sep29 organization motivation |
| E14-2: complete original grid | Exp14 `phase1_specs`, `train_one`, `select_phase1` | [native_runs.csv](../../notebooks/artifacts/experiment_14_history_organization/history_organization_v1/aggregate/native_runs.csv), checkpoint/history metadata | Sep29 03:20 result interpretation |
| E14-3: geometry/readout mismatch | Exp14 `_geometry`, `_retrieval`, `_history_generalization` | Selected `runs/<key>/probes.json`, `cross_user_geometry.csv`, `cross_user_retrieval.csv`, `history_generalization.csv`; directory restores λ | Sep29 initial result and follow-up consistency discussion |
| E14-4: corrected sampler / negative prefix selection | Exp14.1 `_auxiliary_batches`, `_validation_artifacts`, selection; [PR89](https://github.com/hellowPluto78700/writingRing/pull/89) | All24 `validation_summary.json`; [selection.json](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/selection.json); `protocol.json` sampler checks | Sep29 15:47–18:33 design and review |
| E14-5: six final evaluations | Exp14.1 `final_eval_specs`, finalizer | [native_runs.csv](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/native_runs.csv), `probe_summary.csv`, `paired_delta_vs_c0.csv`, `collapse_gap.csv`, retrieval, final manifest | Sep29 19:33 preliminary failure; later final artifacts override stale interpretation |
| E14-6: gradients/evidence organization | Exp14.1 `_gradient_diagnostic`, `_native_prefix_diagnostics` | [gradient_diagnostics.csv](../../notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/gradient_diagnostics.csv), `native_prefix_metrics.csv`, `evidence_organization_diagnostics.csv` | Sep29 16:49 gradient-scale control; Sep30 03:49 shared evidence mapping |
| Original implementation/fix | [7d43034](https://github.com/hellowPluto78700/writingRing/commit/7d43034c252bee6af1d04a704486fe4d1a3d4ab8), [2f0272d](https://github.com/hellowPluto78700/writingRing/commit/2f0272d06d1d37de38ae541ba87f47f7397345d6), [artifact retention 1f85015](https://github.com/hellowPluto78700/writingRing/commit/1f8501515716b82e8ae5850b710f50bba4379691) | `sacct` and final 48 checkpoint metadata | Failed execution separated from scientific negative outcome |
| 14.1 protocol corrections | [ad170bb](https://github.com/hellowPluto78700/writingRing/commit/ad170bbc3023651d3118f59dd2d408e45c09c54b), [equal prefix/control 9f1da3c](https://github.com/hellowPluto78700/writingRing/commit/9f1da3cce5d562ac6672d773349b14aa11d35cd3), [sampler 25996d8](https://github.com/hellowPluto78700/writingRing/commit/25996d8a64a12d013373467be97fd451fc9026fb), [current C0 198a175](https://github.com/hellowPluto78700/writingRing/commit/198a175f59b98eadc081561c355c79d151a36251), [negative selection 3c4ef7d](https://github.com/hellowPluto78700/writingRing/commit/3c4ef7de7ee52052adf18ef7956e0a0cbc449031) | Appendix A accounting; selected current checkpoints and final manifest | Sep29 PR89 review and no-eligible discussion |
| Transition to write mechanism | Exp15 proposal cross-checked against its actual source, not imported results | No extra Exp14 gate runs | Sep30 19:49–20:12 history-content / frozen-versus-joint design |

All four required chat stages were searched: motivation, design/implementation, result interpretation, and successor transition. Retrieved topic titles include “Dual Loader Loss” and “验证修改方案合理性”; exact titles for several timed retrievals were unavailable. Later “exp16” discussions are retrospective context, not evidence that prefix or gating was in original Exp14. Available chat summaries do not constitute a complete archive. Original failure log paths checked on Unity were absent; the failure reasons stated here are bounded by accounting, implementation changes, and contemporaneous review rather than fabricated stderr.

## Appendix D. Chronological Reasoning Trace

| UTC date/time or execution interval | Scientific reasoning / action | What changed in the inference |
| --- | --- | --- |
| Sep24–28 predecessor discussion | More old context/depth does not ensure transferable native accumulation | Sets the organization question without declaring semantic abstraction |
| Original Exp14 execution before Sep29 review | CU / phase / delta / combination with structured task sampling; projector fix and final 48 outputs | Valid original study, but its control is not ordinary Core O0 |
| Sep29 03:20 | First result synthesis: Phase-CU improves geometry; C4 shows representation/readout mismatch | Native improvement is not inferred from probe improvement |
| Sep29 15:47–16:14 | Preserve WCCE task batch; independent fresh auxiliary data instead of stale memory bank | Dual-loader redesign isolates objective from task sampling |
| Sep29 16:37–16:49 | Add phase consistency/prefix evidence; require native preservation, measure gradient scales; record no eligibility | Early evidence and final evidence become separate falsifiable goals |
| Sep29 18:29–18:33 | User requests PR89 review; sparse cells make four unique segments infeasible | Replace auxiliary design with8×8×1, preserve task loader |
| 65035050 / 65037600 execution | Auxiliary cases complete, C0 fails historical bitwise check then is replaced after current-contract correction | Historical reference retained but no longer blocks a valid new control |
| Sep29 19:33 and selection execution | Initial no-eligible-prefix exception appears as a pipeline failure | Later `3c4ef7d` makes rejection an explicit negative result |
| Final65039702–04 | Phase .01 selected; Prefix none; no combined cases; six final tests | Modest native mean gain with flat count/temporal gap, not robust consolidation |
| Sep29 21:35 and Sep30 03:49 | Earlier-prefix classification does not guarantee final accumulation; z(history) and shared W can remain incompatible | Reframe beyond geometry alone |
| Sep30 19:49–20:12 | Stage synthesis and gate design; user requires frozen/joint comparisons | Next test intervenes on writing incoming history; Prefix deferred from Exp15 |

The chronology retains both scientific negative outcomes and implementation failures. Current final artifacts correct the earlier chat description of “selection failed” without erasing why that failure changed the pipeline.
