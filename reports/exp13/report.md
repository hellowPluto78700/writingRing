# Exp13 — Hierarchical Temporal Representation

## Report Scope and Evidence Cutoff

This report covers `experiment_13_hierarchical_temporal_representation` (v2) and its genuine extension `experiment_13_1_abstraction_generalization` (v1). It reconstructs frozen-checkpoint analyses, not a new Exp13 training sweep. Exp7.3/7.3.9 provide source networks. Later CoreBenchmark results provide a separately labeled standardized comparison; Exp14, Exp15, and Exp16 outcomes are outside this family's primary result set.

Evidence was inspected on 2026-10-02. Implementation was read from GitHub snapshot `3f0e349e48f46f0c14b536e1923488c27381dcbc`; Unity results were read from `/home/zhaolongwei_umass_edu/projects/writingRing` at tracked HEAD `ef12b4e22de64693f3af317e4757ba913bf8ab12`. The two snapshots are recorded separately: current source explains the final contract; commits and run manifests establish revisions. Unity files and checkpoint metadata determine quantitative outcomes. No training or cluster files were changed for this reconstruction.

Project history was actively searched by experiment IDs and mechanism terms, including discussions before design, during implementation, after results, and in successor experiments. The accessible retrievals contain dated user statements and prior assistant summaries, rather than a complete exported Project archive. They support the reasoning chronology but do not establish run completion or numerical truth. Titles are reported only where recovered; other entries are labeled by topic with title unavailable. Missing full transcripts and original failed-job stdout/stderr are evidence gaps, not reconstructed quotations.

All BA values are balanced accuracy, reported as percentages; differences are percentage points (pp). Unless explicitly labeled otherwise, `mean ± SD` is an equally weighted mean and sample SD (`ddof=1`) over model seeds 11, 23, 37, **n=3**. Shuffle replicates, batches, neurons, strokes, and users are not additional model seeds. No significance claim is made from these three paired seeds.

## 1. Experiment Identification

| Chat/job alias | Canonical repository ID | Protocol / artifact directory | Role |
| --- | --- | --- | --- |
| Exp13, exp13_a1…exp13_e | `experiment_13_hierarchical_temporal_representation` | `hierarchical_temporal_representation_v2` | A1/A2/A3/B/C/D/E temporal and causal analyses |
| Exp13.1, exp13_1_f/g/h | `experiment_13_1_abstraction_generalization` | `abstraction_generalization_v1` | F/G/H transfer, nuisance identity, history-gain analyses |
| A2, a2 | Exp7.3 `A2_e2e_linear_wcce` | `training_strategy_decomposition_v1` | Frozen trained two-layer primary source |
| C1, c1 | Exp7.3.9 `C1_frozen_a2_train_l3` | `pretrained_a2_depth_extension_v1` | Added L3 with inherited A2 L1/L2 frozen; primary controlled depth comparison in 13.1 |
| C2, c2 | Exp7.3.9 `C2_c1_init_e2e_3layer` | Same predecessor protocol | Jointly adapted three-layer replication/control |
| random_a2 | Matched random A2 architecture | No trained source checkpoint | Dynamics/filtering control in A1/E |

Artifact roots are `notebooks/artifacts/<canonical ID>/<protocol>/`. The implementation lives in [Exp13](../../scripts/experiment_13_hierarchical_temporal_representation.py) and [Exp13.1](../../scripts/experiment_13_1_abstraction_generalization.py), with corresponding subdirectories containing READMEs and Slurm launchers. “Hierarchy,” “temporal abstraction,” “longer TRW,” and “history mixing” are interpretation aliases, not additional experiments. `docs/plans/Done13.md`-style numbering must not be used to assign family membership.

## 2. Scientific Context

Earlier multi-τ networks contained temporal state but did not automatically close the gap between native accumulated-spike classification and richer temporal probes. Increasing depth also failed to guarantee native improvement. The inherited Exp7.3.9 native test BA was A2 **56.45 ± 3.28%**, C1 **47.55 ± 6.30%**, C2 **50.42 ± 2.58%** (n=3). These are predecessor outcomes, not retrained Exp13 results [E13-1]. They motivated asking what the added state actually represented.

Before Exp13, the hierarchy hypothesis was that event-level activity might become local motion primitives, then strokes, then character context. The 2026-09-24 14:54–14:58 UTC discussions separated three notions: intrinsic τ, a functional temporal receptive window, and abstraction. A long decay constant supplies memory capacity; it does not demonstrate learned integration, task-relevant communication, or nuisance invariance. Lag similarity alone can confuse smooth filtering with useful context.

## 3. Motivation and Open Question

The initial question was whether deeper layers merely filter local input or transform past input into a currently communicated representation. A second question followed: if the layer contains more history, does that history help unseen writers or mainly improve discrimination within familiar writing styles?

Competing explanations were explicitly retained. A layer might be slow without encoding useful information; its analog current/membrane might preserve history that spikes fail to communicate; or its spikes might carry history-dependent class evidence without organizing that evidence into transferable, WholeCount-readable content. The last explanation motivated Exp13.1 rather than being assumed from the beginning.

## 4. Hypotheses

| Hypothesis | Distinguishing prediction | Necessary evidence |
| --- | --- | --- |
| H0: temporal filtering only | Matched random dynamics explain temporal behavior; truncation effects need not be class-informative | A1/E plus task probes, not τ alone |
| H1: internal history without effective spike expression | I/U depend on old input but spike features or shared class probes do not | A2 across states, B shared/retrained probes |
| H2: communicated contextual representation | Identical recent input yields different current spike features after different history; class information is affected | A2/A3/B/D |
| Stronger abstraction hypothesis | Added depth improves cross-user task geometry and transfer, rather than only context dependence | C and F/H, supported by nuisance controls |
| Seen-context specialization | Long history benefits seen-user folds more than held-out writers, especially L3 | H phase-controlled excess-gain contrasts; G is a separate nuisance diagnostic |

The stronger abstraction hypothesis was testable and was not treated as a definition of depth. User identity and task utility can coexist; a user-ID probe cannot by itself classify history as harmful.

## 5. Experimental Design

Exp13 freezes A2 and C2. A1 compares trained A2, matched random A2, and C2 across lag, state, and layer. A2 replays exactly the same recent suffix from zero state, removing older history. A3 resets L1, L2, or both within an otherwise identical input sequence and measures downstream state recovery. B asks whether the history removed by A2 is class-accessible using a fixed full-history decoder and a separately refitted truncated-history decoder. C checks weak stroke-boundary organization against random and motion-matched pseudo-boundaries. D permutes intact stroke waveforms and realigns responses back to stroke identity. E probes impulse and real-motif responses to separate architecture-induced delays from learned communication.

Exp13.1 adds F/G/H. Its primary C1 comparison preserves inherited A2 L1/L2 while changing L3, limiting the confound of retraining earlier layers. C2 checks whether the pattern survives joint adaptation. F tests task geometry and held-out-user retrieval; G tests residual user identity; H tests how added history changes ID versus OOD class decoding on a common eligible cohort. The corrected F/H protocol, not the first completed version, defines final 13.1 evidence.

## 6. Implementation

The predecessor input is 30 unsigned events from the same 36-channel wavelet producer at 64 Hz. The loader is [Exp3.0.1](../../scripts/experiment_3_0_1_single_tau_objectives.py), reused through Exp7.x. Its `LABELS` allowlist places X earlier, but actual `data.labels` sorts the retained labels: A, B, C, D, E, G, H, I, J, K, L, X, matching the executed sample manifest and Core class order. The split seed is 12345. Train users are 0, 1, 2, 5, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20; validation users 4, 9, 16; test users 3, 6, 10. The source data contain 581/126/146 train/validation/test sequences; Exp13 source metadata separately records 1187/262/296 weakly annotated strokes. Stroke annotations are not additional labeled classification samples. Matching users/samples does not make the historical checkpoints Core-trained or establish identical serialized dataset hashes.

A2 has 128-neuron layers with 234×234 synaptic shifts, membrane τ=22.54 ms, signed bias-free communication, and unnormalized current recurrence Iₜ=αIₜ₋₁+Wxₜ. It was trained with a linear accumulated-spike head and WCCE. C1/C2 inherit the depth-extension architecture and source-checkpoint selection. No Exp13 loss updates these networks.

Here source WCCE is CE of valid-mean evidence with CE gain=1. TSCE computes timestep classification CE, but its historical Exp7.3 implementation pools valid timesteps, unlike later Core's equal-per-sequence average. Only WCCE source cases enter this report; no TSCE result is pooled across those contracts.

Primary state `spike50` is a **four-step rolling spike sum**: at 64 Hz it spans 62.5 ms, despite the historical name. `syn_current` is I and `pre_reset` is pre-reset membrane U. Neither is interchangeable with the communicated spike feature. A2 histories 50/100/250/500/750/1000 ms become 3/6/16/32/48/64 steps after rounding. Both I and U start at zero for suffix replay. Phase anchors are 25/50/75/100% and additional weak stroke midpoint/end anchors; the endpoint tables below select only test/phase=100%.

A3 resets I/U at 25/50/75%, then uses identical future events. Spike smoothing can retain pre-reset spikes for several steps, so near-zero-delay correlations are not evidence that reset had no effect. D preserves within-stroke waveform and permutes individual strokes or adjacent two-stroke blocks; nonzero inserted gaps exclude two onset steps during alignment. E uses one-channel impulses and six-step real motifs followed by a 64-step zero tail.

## 7. Executed Runs and Validity Audit

The Exp13 manifest lists 9 A1, 36 A2, 12 A3, 36 B, 3 C, 24 D, and 9 E tasks. All analysis tasks completed in Slurm. The first finalizer failed; the replacement completed after empty-file handling was corrected. **Twelve D multi-stroke artifacts are empty** because no eligible block-reordering sequences yielded rows. They are a coverage gap, not zero sensitivity, and no multi-stroke conclusion is drawn.

The first Exp13.1 launch completed 6 caches, 18 F metrics, 18 G, 18 H-feature, 18 H tasks, and finalization. A subsequent protocol correction reran all 18 F metrics and 18 H tasks. G and compatible feature caches were retained. Final numbers use corrected F/H outputs; first-pass metrics are superseded, not averaged into them. Appendix A lists accounting and exact task-index rules.

## 8. Evaluation and Diagnostics

Lag recurrence is Pearson correlation across neurons between state vectors at two valid times, then averaged over valid pairs. It assesses temporal similarity, not memory causality. Cosine and normalized L2 are companion measures. Undefined correlation for constant vectors is excluded; normalized L2 divides by the reference norm plus ε and can explode for silent references, so it is not used as the primary effect size.

Common-suffix similarity compares full-history and zero-state suffix representations at matched anchors. The shared B probe is trained on full-history train features and selected on full-history validation features, then held fixed on test truncations. The retrained probe learns from truncated train features with matching validation selection. A shared-probe drop mixes information change and geometry shift; a refitted-probe drop measures class accessibility under that probe family, not an information-theoretic loss.

B `no_bias` uses scale-only normalization and no intercept; B affine uses centering plus an intercept. **G/H differ:** `_fit_logistic` mean-centers (`with_mean=True`) and uses `fit_intercept=False`, balanced class weights. Its raw-space decision rule is W D⁻¹(x−μ), which includes the effective offset −W D⁻¹μ. It must not be described as Core's origin-preserving no-bias probe. Their C selection and fold structure also differ.

F normalizes each valid trajectory to 64 phase points. Distances are mean cosine distance under fixed alignment or bounded DTW with a 20% band. Pair sets distinguish same class/same user (SC_SU), same class/cross user (SC_CU), and different class/cross user (DC_CU). The corrected negative pairs are globally class-balanced and duration-matched. R=mean(DC_CU distance)/mean(SC_CU distance). Retrieval uses training-user class/user medoids as a gallery and held-out-user queries, scoring BA. It is not a fitted native accumulator.

G residualizes character centroids using only the appropriate training fold, predicts writer identity with nested C selection, and subtracts a within-character permutation baseline (100 permutations). H uses three deterministic folds stratified within user×class, selects C once on full-history ID data for each feature setting, and fixes it across the sweep. Eligibility is phase-specific and shared across all compared histories: phase 50% uses histories through 500 ms, 75% through 750 ms, endpoint through 1000 ms, with all 12 classes present. This avoids class-drop and cohort changes masquerading as a history effect.

## 9. Quantitative Results

### 9.1 Lag similarity and functional history window

The A1 repository summary pools two test-batch rows per seed, so its count=6 is not six independent seeds. The following table recomputes each seed using finite-correlation counts as batch weights, then reports n=3 mean±SD. It therefore intentionally differs slightly from the original batch-weighted summary [E13-2].

| Model | Layer | Lag 1 (15.6 ms) | Lag 4 (62.5 ms) | Lag 16 (250 ms) | Lag 64 (1000 ms) |
| --- | --- | --- | --- | --- | --- |
| a2 | L1 | 0.936 ± 0.002 | 0.798 ± 0.005 | 0.665 ± 0.012 | 0.476 ± 0.008 |
| a2 | L2 | 0.963 ± 0.001 | 0.796 ± 0.006 | 0.424 ± 0.018 | -0.067 ± 0.020 |
| c2 | L1 | 0.937 ± 0.003 | 0.799 ± 0.008 | 0.667 ± 0.015 | 0.476 ± 0.015 |
| c2 | L2 | 0.965 ± 0.001 | 0.804 ± 0.006 | 0.445 ± 0.018 | -0.034 ± 0.031 |
| c2 | L3 | 0.981 ± 0.002 | 0.870 ± 0.020 | 0.555 ± 0.076 | 0.075 ± 0.115 |

A2 L2 is highly similar at one step but decorrelates faster than L1 over longer lags. C2 L3 is smoother than C2 L2 at every displayed lag. Thus “every deeper layer changes faster” and “every deeper layer is slower” are both unsupported blanket statements.

The A2 saved summary pools train, validation, and test because split is absent from its grouping. The table below recomputes **test only**, endpoint `spike50`, one row per seed [E13-3].

| Retained history (nominal ms) | Steps | A2 L1 correlation | A2 L2 correlation | C2 L3 correlation |
| --- | --- | --- | --- | --- |
| 50 | 3 | 0.270 ± 0.035 | 0.088 ± 0.055 | 0.170 ± 0.152 |
| 100 | 6 | 0.398 ± 0.025 | 0.111 ± 0.034 | 0.107 ± 0.050 |
| 250 | 16 | 0.774 ± 0.016 | 0.353 ± 0.007 | 0.254 ± 0.103 |
| 500 | 32 | 0.937 ± 0.004 | 0.722 ± 0.003 | 0.559 ± 0.082 |
| 750 | 48 | 0.970 ± 0.002 | 0.910 ± 0.001 | 0.823 ± 0.037 |
| 1000 | 64 | 0.986 ± 0.003 | 0.972 ± 0.001 | 0.944 ± 0.013 |

At 500 ms, A2 L1 approximately reproduces its full-history endpoint (r≈0.937), while L2 is still at r≈0.722. At 750 ms L2 reaches ≈0.910; C2 L3 needs still more history. “L1≈500 ms, L2≈750 ms” is a descriptive recovery threshold, not an estimated intrinsic τ or a sharp memory boundary. The apparent paradox is resolved by measuring different quantities: a history-conditioned state can change quickly while depending on a long past.

### 9.2 Class-accessible context and stroke order

Endpoint spike B probes on A2, no bias, test BA [E13-4]:

| History (ms) | Layer | Full-history BA (%) | Truncated, shared probe | Truncated, retrained probe |
| --- | --- | --- | --- | --- |
| 250 | L1 | 27.47 ± 2.96 | 14.03 ± 4.60 | 19.92 ± 1.90 |
| 250 | L2 | 44.54 ± 2.33 | 15.73 ± 3.15 | 20.84 ± 2.75 |
| 500 | L1 | 27.47 ± 2.96 | 22.98 ± 4.12 | 26.10 ± 0.31 |
| 500 | L2 | 44.54 ± 2.33 | 32.62 ± 0.90 | 31.79 ± 6.55 |
| 1000 | L1 | 27.47 ± 2.96 | 26.95 ± 1.28 | 27.14 ± 1.56 |
| 1000 | L2 | 44.54 ± 2.33 | 44.70 ± 2.36 | 43.14 ± 2.61 |

L2 full-history instantaneous decoding exceeds L1. Restricting history to 250 ms reduces L2 BA even after refitting; the shared decoder alone would overstate information loss. At 1000 ms the shared L2 probe nearly recovers full BA, while independently retrained values need not be identical. These are endpoint feature probes; they do not report native predictions after state reset.

D sensitivity is 1−correlation after realigning the same stroke in original versus scrambled context. The table first averages eligible stroke rows within each seed, then aggregates seeds (n=3). The saved D summary instead pools stroke rows and its SD is not seed uncertainty [E13-5].

| Gap (nominal ms) | L1 sensitivity | L2 sensitivity |
| --- | --- | --- |
| 0 | 0.206 ± 0.006 | 0.471 ± 0.012 |
| 50 | 0.188 ± 0.008 | 0.436 ± 0.015 |
| 100 | 0.167 ± 0.009 | 0.398 ± 0.018 |
| 250 | 0.112 ± 0.007 | 0.267 ± 0.015 |

At a nominal 50-ms gap, L2 sensitivity is approximately 2.32× L1. Persistence across gaps supports sensitivity to preceding context rather than only immediate waveform mismatch. Reordering alters global trajectory and state, so this does not isolate a semantic compositional code.

### 9.3 Cross-user transfer and depth (Exp13.1)

F held-out-user `spike50` retrieval, corrected protocol, three seeds [E13-6]:

| Model | Layer | Alignment | Held-out-user retrieval BA (%) |
| --- | --- | --- | --- |
| c1 | L1 | dtw | 37.55 ± 6.07 |
| c1 | L1 | fixed | 44.97 ± 3.11 |
| c1 | L2 | dtw | 56.00 ± 2.90 |
| c1 | L2 | fixed | 57.11 ± 1.03 |
| c1 | L3 | dtw | 51.13 ± 3.62 |
| c1 | L3 | fixed | 52.38 ± 3.83 |
| c2 | L1 | dtw | 38.82 ± 4.44 |
| c2 | L1 | fixed | 44.55 ± 4.67 |
| c2 | L2 | dtw | 53.63 ± 1.86 |
| c2 | L2 | fixed | 56.84 ± 3.53 |
| c2 | L3 | dtw | 53.04 ± 4.64 |
| c2 | L3 | fixed | 53.67 ± 4.88 |

C1's controlled L1→L2 comparison raises fixed-alignment retrieval from ≈44.97% to ≈57.11%; adding L3 reduces it to ≈52.38%. C2 has a similar nonmonotonic pattern. The DTW condition does not restore a monotonic hierarchy. F is compatible with a useful L2 context transformation and does not support universal transfer improvement with depth.

G endpoint excess writer-ID BA is C1 L1/L2/L3 **18.38/23.29/14.67 pp**, C2 **20.38/22.79/14.26 pp**. L3 does **not** show a monotonic increase in this nuisance probe. The shorthand “L3 is more user-specific” must therefore refer to H's relative benefit from seen-context history, not an established rise in residual writer decodability [E13-7].

H endpoint full-history results and gains relative to 50 ms, corrected common cohort [E13-8]:

| Model | Layer | Full ID BA (%) | Full OOD BA (%) | ID gain (pp) | OOD gain (pp) | Excess seen gain (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | L1 | 24.78 ± 1.76 | 26.46 ± 3.45 | 16.01 | 17.92 | -1.91 ± 3.97 |
| c1 | L2 | 52.97 ± 2.97 | 49.67 ± 7.75 | 44.58 | 41.34 | 3.24 ± 4.32 |
| c1 | L3 | 66.30 ± 2.48 | 53.10 ± 4.12 | 58.42 | 44.77 | 13.65 ± 1.41 |
| c2 | L1 | 25.40 ± 1.29 | 27.77 ± 6.84 | 16.34 | 19.24 | -2.90 ± 5.42 |
| c2 | L2 | 51.60 ± 3.55 | 53.28 ± 7.17 | 43.16 | 44.94 | -1.79 ± 6.20 |
| c2 | L3 | 71.56 ± 3.67 | 56.15 ± 2.36 | 63.36 | 47.82 | 15.54 ± 2.41 |

L3 has a substantial positive excess seen gain (≈13.65/15.54 pp), reproduced in C1/C2. However, its endpoint OOD BA also rises from L2 to L3. “Added L3 always harms OOD” would conflict with these measurements. The effect is diagnostic-specific: F whole-trajectory retrieval peaks at L2, while H endpoint class decoding improves but improves more in seen-user folds. At 75% phase, excess gain is already large in L2 (C1≈17.18 pp, C2≈21.50 pp) and L3 (≈18.18/23.26 pp); the strong endpoint L2-versus-L3 separation is not phase-invariant.

### 9.4 Readout access: predecessor and later standardized evidence

Inherited Exp7.3.9 no-bias sequence probes give A2 L1 WholeCount **56.48 ± 4.00%**, Fixed250 **55.15 ± 2.03%**; A2 L2 **54.10 ± 3.69%** and **58.05 ± 1.65%**. C1 L3 WholeCount **51.22 ± 2.89%**, Fixed250 **54.43 ± 1.12%**; C2 L3 **52.48 ± 2.13%** and **57.68 ± 2.00%** [E13-1]. These do not show a native/WholeCount gain from adding L3. They also do not prove that L3 lacks class information, because temporally structured and endpoint probes access different features.

Fixed250 order/shuffle and Relative10 were standardized later under Core, not run as an Exp13 A–H subcase. For separately trained Core O0, no-bias L1/L2 test BA is [E13-9]:

| Feature | L1 | L2 |
| --- | --- | --- |
| WholeCount | 55.08 ± 2.23 | 57.57 ± 4.24 |
| Fixed250 ordered | 56.28 ± 1.47 | 58.62 ± 2.69 |
| Fixed250 shuffled, replicate mean | 44.86 ± 2.31 | 53.79 ± 3.05 |
| Relative10 ordered | 71.73 ± 1.83 | 67.04 ± 0.73 |
| Relative10 shuffled, replicate mean | 35.50 ± 0.71 | 54.24 ± 3.66 |

Fixed250 mean order–shuffle gap shrinks from **11.42 to 4.83 pp**; Relative10 from **36.24 to 12.79 pp**. Under matched decoder refitting, L2 retains more class access when coarse bin order is removed. This is consistent with some temporal context being embedded inside later features. It is not proof of temporal-order invariance or abstract composition: the ordered–shuffled gap remains positive, and Relative10 exposes class information beyond WholeCount/native accumulation. Core native O0 is **57.93 ± 2.23%**, a distinct checkpoint family from Exp13 A2 and from fresh Exp14.1 C0.

## 10. Training / Mechanistic Diagnostics

Exp13 has no training curve of its own. Source checkpoint epochs, stopping points, hashes, and provenance are listed in Appendix A. Random A2 is architecture-matched but not firing-statistics-matched; a silent random deeper layer weakens direct spike-level comparisons.

A3 L2 spike recovery after reset at 50%, test only, weighted within seed by valid correlation count [E13-10]:

| Reset at 50% | 62.5 ms later | 125 ms later | 250 ms later | 500 ms later |
| --- | --- | --- | --- | --- |
| intact | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| reset_l1 | 0.748 ± 0.009 | 0.576 ± 0.006 | 0.577 ± 0.008 | 0.776 ± 0.007 |
| reset_l2 | 0.621 ± 0.015 | 0.799 ± 0.010 | 0.924 ± 0.008 | 0.982 ± 0.003 |
| reset_both | 0.116 ± 0.024 | 0.248 ± 0.025 | 0.475 ± 0.021 | 0.758 ± 0.005 |

Resetting L2 alone recovers quickly under intact L1 input. Resetting L1 causes a delayed and persistent L2 change, because the upstream stream supplying L2 is itself altered. Resetting both has the largest sustained effect. This supports history distributed across a dynamical cascade, rather than only a single slow L2 compartment. These correlations do not quantify native test BA or the necessity of each layer for the final decision.

C is mixed. At the exact weak press boundary, pooled `spike50` transition 1−r is L2 ≈0.0571 for real boundaries versus ≈0.0339 motion-matched and ≈0.0320 random; at lift, real ≈0.0283 is below motion-matched ≈0.0427 and random ≈0.0323. L1 shows analogous nonuniform behavior. Boundary rows pool events and finite-state matches, not model-seed means. This fails to establish a uniform learned stroke-boundary hierarchy. Weak annotations and imperfect motion matching limit the test.

E pooled pre-reset impulse peak latency is approximately 30.95 ms in trained A2 L1, 41.15 ms in L2, and 61.26 ms in C2 L3. Random L1 is approximately 31.13 ms; some random deeper responses are silent. The code assigns zero peak latency to a zero response through `argmax`, so zero is not an active instantaneous peak. Latency/width summaries pool neurons and stimuli; they are not n=3 effect estimates. These observations support architecture-induced response shaping and learned communication, not a demonstrated learned bank of delay decoders.

## 11. Negative and Null Results

Depth did not monotonically increase slowness, writer invariance, retrieval, or native accumulation. C provides only weak/mixed stroke organization evidence. D's multi-stroke condition has no usable rows. E cannot establish that trained networks discovered a delay-field mechanism. H's excess gain is phase-dependent; G contradicts a simple monotonic writer-leakage story. The preserved negative findings are central to the revision from “deeper means abstract” to “deeper changes how history is represented.”

## 12. Interpretation

### 12.1 Direct observations

Under matched recent input, L2 and L3 require more past input to reproduce their current state. A2 L2's communicated spike features contain class-accessible context, and stroke-aligned L2 responses depend more on preceding order than L1. F transfer peaks at L2; H L3 gains disproportionately from seen-user history at the endpoint. Native and WholeCount performance do not monotonically improve with depth.

### 12.2 Mechanistic interpretation

The measurements are consistent with L1 providing relatively local dynamical features and L2 transforming a longer history into current, task-informative spike context. Added L3 reorganizes more past information without reliably consolidating it into a transferable accumulated-spike representation. “History mixing” is a working interpretation of these combined tests, not a directly measured latent variable.

The chronology matters. The September 24 first interpretation emphasized a longer functional window despite faster long-lag change. The September 27 benchmark discussion then demanded locked splits, three seeds, and bias-separated probes. September 28 analysis emphasized L2 transfer versus L3 seen-context gains. Subsequent September 29–30 discussion reframed the bottleneck as useful history organization/content selection. The later phrase “L3 more user-specific” is narrowed here because G and endpoint OOD results do not support its strongest reading.

### 12.3 What the experiment does not establish

It does not identify semantic motion primitives, prove abstract compositional representations, show that all old history is beneficial, or measure mutual information. State perturbations establish consequences for representations; Exp13 A3 does not directly establish a change in final native prediction. B demonstrates class accessibility of different endpoint states using fitted probes. Later Core prediction-reset evidence is a separate protocol and is not relabeled as A3. Reduced order–shuffle gaps can arise from history internalization, smoothing, rate effects, and feature remapping; they do not alone identify an abstraction mechanism.

## 13. Bugs, Corrections, and Reruns

| Issue | Affected executions | Consequence | Repository correction | Valid replacement |
| --- | --- | --- | --- | --- |
| Empty D CSV handling | `64838820` finalizer failed exit 1; D analysis tasks completed | Missing eligible multi-stroke rows could abort finalization | `be1d57a` handles empty artifacts while retaining coverage count | `64840087` finalizer; only populated stroke rows analyzed |
| F negative-pair balance / duration control | Initial F job `64885716` | Earlier pair geometry was not final protocol | `4503d3c` fixes pairing and H protocol together | F `64891342` tasks 0–17 |
| H phase eligibility / all-class coverage | Initial H job `64885719` | Incomparable cohorts could confound history gains | `4503d3c`: phase-specific common eligibility, label/class checks and fixed sweep selection | H `64891343` tasks 0–17; `64891344` finalizer |
| Aggregate unit ambiguity | Saved A1/A2/D/E summaries | Batch/split/stroke/neuron pooling could be mistaken for seed uncertainty | Reporting correction using raw rows where available | A1/A2/A3/D recomputed separately as stated |

Original failure stdout/stderr was not recovered. Slurm state/exit code and commit changes support the failure/fix sequence; this report does not invent the original traceback. A changed protocol can invalidate a previously completed metric task without invalidating its compatible raw feature cache.

## 14. Relationship to Previous Experiments

| Component | Exp13 inheritance | Difference from later Core |
| --- | --- | --- |
| Data and split | Exp3/Exp7 loader, fixed seed and user split, sorted executed labels | No claim that the source checkpoints were Core-trained or used the later serialized Core cache |
| Backbone/head | Exp7.3 A2 and Exp7.3.9 C1/C2 | Frozen sources rather than Core O0 checkpoints |
| Objective | Source WCCE; no new network objective | Analysis only |
| Readout | Inherited native accumulator plus diagnostic probes | Endpoint B versus sequence-level Core temporal features |
| Probe | B bias-separated; F distance retrieval; G/H centered logistic | G/H are not strict origin-preserving no-bias Core probes |
| Selection | Source validation-selected checkpoints; probe/fold validation | Test never chooses the network or history condition |

The standardized Core comparisons above explain the readout-access ambiguity but cannot be numerically pooled with Exp13 or its predecessor results.

## 15. How This Led to the Next Experiment

Exp13 showed that adding remembered context is not enough: context can be communicated and class-accessible while its transferable geometry and additive readout remain inadequate. Two explanations remained: the representation needed explicit cross-user/class organization, or the network was retaining/writing the wrong history. Exp14 first tested the organization explanation by supervising same-class/different-user representations, including phase and history-delta features. The September 29 follow-up discussed whether improved geometry would transfer to the native accumulator, making representation/readout mismatch an explicit prediction rather than a post hoc excuse.

Exp14.1 later introduced prefix evidence and a preserved ordinary task loader because auxiliary training alone could change sampling or improve a diagnostic without improving accumulated evidence. The sequence is not “Exp13 proved abstraction, then Exp14 optimized it”; it is “Exp13 separated memory from transferable organization, then Exp14 tried to make the latter learnable.” See [the separate Exp14 report](../exp14/report.md).

## 16. Final Conclusions

Exp13/13.1 establish stronger and more task-accessible history dependence in deeper representations, with useful L1→L2 transfer under the tested frozen checkpoints. They reject a universal monotonic hierarchy of slowness, invariance, and native accumulation. L3 can improve endpoint class access while benefiting seen-context history more strongly and reducing trajectory retrieval. The supported conclusion is contextual transformation with incomplete transferable consolidation, not demonstrated semantic abstraction.

## Appendix A. Run Inventory

### A.1 Frozen source checkpoints

The directory links below identify the exact saved checkpoint locations. Checkpoint hashes were read on Unity; training job IDs of these predecessor runs were not recovered. Best/stopped epochs are source training epochs, not analysis epochs. Random A2 has no trained checkpoint.

| Run directory / family | Seed | Best epoch | Stopped epoch | Final valid task or source | Scope / validity | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [Inherited / H128 / 234x234__A2_e2e_linear_wcce__task_only__seed11](../../notebooks/artifacts/experiment_7_3_training_strategy_decomposition/training_strategy_decomposition_v1/e2e_checkpoints) | 11 | 94 | 100 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `1411e046f7b453682a67dcf04512702c35c6b3e217ba13dd3a29ed7fe874f6e6` |
| [Inherited / H128 / 234x234__A2_e2e_linear_wcce__task_only__seed23](../../notebooks/artifacts/experiment_7_3_training_strategy_decomposition/training_strategy_decomposition_v1/e2e_checkpoints) | 23 | 89 | 100 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `3cbf3282bcbd97fbd9a0be503373d3eb234af51aee4ed69ff2c559aeca23c94a` |
| [Inherited / H128 / 234x234__A2_e2e_linear_wcce__task_only__seed37](../../notebooks/artifacts/experiment_7_3_training_strategy_decomposition/training_strategy_decomposition_v1/e2e_checkpoints) | 37 | 90 | 100 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `02b971ed40b386783de9d05e2dad34e572ea653e382b6b1ffc33584842a48223` |
| [Inherited / H128 / C1_frozen_a2_train_l3__seed11](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C1_frozen_a2_train_l3/checkpoints) | 11 | 9 | 39 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `0dca5ca5245c57584f4ebea7df65b357c915352b22a721e404ae96c18ec418e8` |
| [Inherited / H128 / C1_frozen_a2_train_l3__seed23](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C1_frozen_a2_train_l3/checkpoints) | 23 | 47 | 77 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `cfe058afec1c035e0a34917a590557e3cd41cdc60b796614582fb9dd935eb3ad` |
| [Inherited / H128 / C1_frozen_a2_train_l3__seed37](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C1_frozen_a2_train_l3/checkpoints) | 37 | 10 | 40 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `3704b2d7c9b38d67c362c9f7ca9f338a7a65ae50f1ebcb19ac56739d1bac712f` |
| [Inherited / H128 / C2_c1_init_e2e_3layer__seed11](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C2_c1_init_e2e_3layer/checkpoints) | 11 | 0 | 30 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `d834376af2fee50157b81508154a1f4933c75ccc7f1309a0bbd70343acaea0ad` |
| [Inherited / H128 / C2_c1_init_e2e_3layer__seed23](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C2_c1_init_e2e_3layer/checkpoints) | 23 | 18 | 48 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `b074ca6a4677814002ea234afec609bddedc909eeb5177c2de105d3f5060bf0c` |
| [Inherited / H128 / C2_c1_init_e2e_3layer__seed37](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/C2_c1_init_e2e_3layer/checkpoints) | 37 | 58 | 88 | Inherited; training job not recovered | Frozen source; no Exp13 retraining | `9f49c1419888dcbb549ba662eb4798e4c16daa12c60bd539aaec91aeeb8dc4ce` |

### A.2 Every analysis task: index-to-configuration mapping

Let s be the index in seeds [11,23,37], m the index in the stated model list, h the index in [50,100,250,500,750,1000], r the index in [intact,reset_l1,reset_l2,reset_both], g the index in gaps [0,50,100,250], k the index in [stroke,multi_stroke], and v the index in states [spike50,syn_current,pre_reset]. Task IDs start at zero. A single task emits all applicable layers/states/anchors unless the state is part of its spec.

| Analysis | Final valid job base | Array index | Configuration / evaluation split | Artifact directory |
| --- | --- | --- | --- | --- |
| A1 | 64838811 | 3s+m; m=[a2,random_a2,c2] | Lag grid and activity, test | `A1_recurrence` |
| A2 | 64838812 | 12s+6m+h; m=[a2,c2] | Truncation, train/val/test separately | `A2_history_truncation` |
| A3 | 64838813 | 4s+r | A2 reset, test | `A3_layer_reset` |
| B | 64838814 | 12s+6m+h; m=[a2,c2] | Fit train/select val/report test | `B_context_expression` |
| C | 64838815 | s | A2 boundary and lag controls, test | `C_stroke_organization` |
| D | 64838816 | 8s+2g+k | A2 stroke-aligned test; k=1 empty | `D_stroke_scrambling` |
| E | 64838818 | 3s+m; m=[a2,random_a2,c2] | Synthetic impulse / train-derived motifs | `E_cross_tau_decoding` |
| F cache | 64885715 | 2s+m; m=[c1,c2] | Frozen trajectories, all splits | `F_cross_user_geometry/cache` |
| F final metrics | 64891342 | 6s+3m+v | Pair domains; held-out-user retrieval | `F_cross_user_geometry` |
| G | 64885717 | 6s+3m+v | Nested writer-ID controls | `G_user_leakage` |
| H features | 64885718 | 6s+h | C1 suffix features; C2 compatible Exp13 cache | `H_history_generalization` |
| H final metrics | 64891343 | 6s+3m+v | ID folds and OOD test at three phases | `H_history_generalization` |

These rules enumerate every analysis configuration and Slurm array task without counting feature-cache tasks as independent outcomes. Artifact stems are generated by each spec's `.key` in the linked scripts; `source_manifest.json` fixes source checkpoints. Primary tables use test-only rows and seed-specific outputs from these directories.

### A.3 Scheduler accounting (failed and completed launches)

| Job / array base | Name | Recorded tasks | States | Exit codes | First start (Unity local timestamp) | Last end |
| --- | --- | --- | --- | --- | --- | --- |
| 64838810 | exp13_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-24T22:11:06 | 2026-09-24T22:11:45 |
| 64838811 | exp13_a1 | 9 | COMPLETED: 9 | 0:0 | 2026-09-24T22:12:11 | 2026-09-24T22:13:37 |
| 64838812 | exp13_a2 | 36 | COMPLETED: 36 | 0:0 | 2026-09-24T22:13:49 | 2026-09-24T22:15:38 |
| 64838813 | exp13_a3 | 12 | COMPLETED: 12 | 0:0 | 2026-09-24T22:15:59 | 2026-09-24T22:16:22 |
| 64838814 | exp13_b | 36 | COMPLETED: 36 | 0:0 | 2026-09-24T22:16:31 | 2026-09-24T22:19:18 |
| 64838815 | exp13_c | 3 | COMPLETED: 3 | 0:0 | 2026-09-24T22:19:39 | 2026-09-24T22:20:28 |
| 64838816 | exp13_d | 24 | COMPLETED: 24 | 0:0 | 2026-09-24T22:20:38 | 2026-09-24T22:21:50 |
| 64838818 | exp13_e | 9 | COMPLETED: 9 | 0:0 | 2026-09-24T22:21:56 | 2026-09-24T22:22:30 |
| 64838820 | exp13_fin | 1 | FAILED: 1 | 1:0 | 2026-09-24T22:22:38 | 2026-09-24T22:22:56 |
| 64840087 | exp13_fin | 1 | COMPLETED: 1 | 0:0 | 2026-09-24T22:29:32 | 2026-09-24T22:29:59 |
| 64885714 | exp13_1_prep | 1 | COMPLETED: 1 | 0:0 | 2026-09-25T23:07:28 | 2026-09-25T23:08:14 |
| 64885715 | exp13_1_fc | 6 | COMPLETED: 6 | 0:0 | 2026-09-25T23:08:33 | 2026-09-25T23:10:38 |
| 64885716 | exp13_1_fm | 18 | COMPLETED: 18 | 0:0 | 2026-09-25T23:10:43 | 2026-09-25T23:13:47 |
| 64885717 | exp13_1_g | 18 | COMPLETED: 18 | 0:0 | 2026-09-25T23:13:58 | 2026-09-25T23:21:03 |
| 64885718 | exp13_1_hf | 18 | COMPLETED: 18 | 0:0 | 2026-09-25T23:21:33 | 2026-09-25T23:22:48 |
| 64885719 | exp13_1_h | 18 | COMPLETED: 18 | 0:0 | 2026-09-25T23:23:11 | 2026-09-25T23:23:56 |
| 64885720 | exp13_1_fin | 1 | COMPLETED: 1 | 0:0 | 2026-09-25T23:24:16 | 2026-09-25T23:24:38 |
| 64891341 | exp13_1_prep | 1 | COMPLETED: 1 | 0:0 | 2026-09-26T04:26:26 | 2026-09-26T04:27:26 |
| 64891342 | exp13_1_fm | 18 | COMPLETED: 18 | 0:0 | 2026-09-26T04:27:30 | 2026-09-26T04:30:16 |
| 64891343 | exp13_1_h | 18 | COMPLETED: 18 | 0:0 | 2026-09-26T04:30:42 | 2026-09-26T04:31:21 |
| 64891344 | exp13_1_fin | 1 | COMPLETED: 1 | 0:0 | 2026-09-26T04:31:47 | 2026-09-26T04:32:08 |

### A.4 Aggregation integrity

A1 and A3 weight each batch by finite-correlation `count`, then average the three seeds. A2 uses one test/endpoint row per seed. D averages stroke rows within a seed without neuron/pair reweighting. B/F/H use their seed-level BA. C/E numbers are explicitly pooled descriptive summaries. The saved `count` fields in pooled files are never interpreted as seed counts. First-pass F/H are superseded. Multi-stroke empties do not enter means.

## Appendix B. Configuration / Case Definitions

| Term | Exact operational definition |
| --- | --- |
| Pearson r | Cosine between neuron-centered vectors; undefined constant vectors excluded |
| Context sensitivity | 1−r, not a BA difference; r may be negative |
| Normalized L2 | ‖a−b‖/(‖a‖+10⁻¹²), asymmetric and unstable for zero reference norms |
| Shared drop | 100×(full-history test BA−truncated test BA under fixed full-history probe) |
| Refitted loss | 100×(full-history BA−truncated BA under independently refitted probe); a probe-access contrast |
| Geometry shift | Refitted truncated BA−shared truncated BA, in pp; can be negative |
| R cross-user geometry | Mean different-class/cross-user trajectory distance divided by mean same-class/cross-user distance |
| User excess | Writer-ID BA minus mean within-character permutation BA, using fold-local residualization |
| ID history gain | Within-user/class-fold class BA at a history condition minus its 50-ms BA |
| OOD history gain | Held-out-writer class BA at condition minus 50-ms BA |
| Excess seen gain | ID history gain−OOD history gain; not absolute ID−OOD gap |
| Impulse width/duration | Number of response steps ≥50% / ≥10% peak; not necessarily one contiguous interval |

Relative10 and Fixed250 Core terms use valid spike bin sums, not endpoint U/I. No reported endpoint result is relabeled WholeCount. No temporal probe is claimed to be a new trained native head.

## Appendix C. Evidence Map

Unity artifact references are repository-relative where corresponding artifacts are retained; large PT/NPZ files may exist only on Unity. Absolute Unity root is given in Scope. GitHub source snapshot and checkpoint hashes make the origin explicit.

| ID / central claim | Implementation | Quantitative or execution evidence | Reasoning source |
| --- | --- | --- | --- |
| E13-1: depth/native/readout predecessor | [Exp7.3.9](../../scripts/experiment_7_3_9_pretrained_a2_depth_extension.py) | [native_runs.csv](../../notebooks/artifacts/experiment_7_3_9_pretrained_a2_depth_extension/pretrained_a2_depth_extension_v1/native_runs.csv), `probe_summary.csv`, source checkpoint metadata; [sample_manifest.csv](../../notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/sample_manifest.csv) establishes executed label order/split | Sep24 depth/history motivation; Sep28 depth interpretation |
| E13-2: lag recurrence | Exp13 `_recurrence_rows`, `_row_corr` | `A1_recurrence/A1_recurrence_all_runs.csv`; [summary](../../notebooks/artifacts/experiment_13_hierarchical_temporal_representation/hierarchical_temporal_representation_v2/A1_recurrence/A1_recurrence_summary.csv) | Sep24 14:58, 22:37 UTC |
| E13-3: common suffix | Exp13 `_suffix_features`, `run_a2` | `A2_history_truncation/A2_history_truncation_all_runs.csv`, filtered test/endpoint | Sep24 18:17 controls; 22:37 result interpretation |
| E13-4: class-accessible context | Exp13 `_fit_probe_model`, `run_b` | `B_context_expression/B_context_expression_all_runs.csv`, no_bias/endpoint | Sep24 control chain; Sep27 bias distinction |
| E13-5: preceding stroke sensitivity | Exp13 `_scramble_order`, `run_d` | `D_stroke_scrambling/D_stroke_scrambling_all_runs.csv`, [manifest](../../notebooks/artifacts/experiment_13_hierarchical_temporal_representation/hierarchical_temporal_representation_v2/manifest.json) empty count | Sep24 18:17 and 22:37 |
| E13-6: transfer peaks at L2 | Exp13.1 `_pair_distances`, `_retrieval_rows` | [F_retrieval_summary.csv](../../notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/F_cross_user_geometry/F_retrieval_summary.csv), per-seed file, corrected F pairs | Sep28 01:46 interpretation |
| E13-7: nonmonotonic writer decoding | Exp13.1 `_fit_logistic`, residualization and `run_g` | [G_user_leakage_summary.csv](../../notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/G_user_leakage/G_user_leakage_summary.csv) | Sep28 user-specific interpretation, narrowed by artifacts |
| E13-8: disproportionate L3 seen gain | Exp13.1 `_make_h_folds`, `run_h` | [H_history_summary.csv](../../notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/H_history_generalization/H_history_summary.csv), phase/common-cohort manifest | Sep28 analysis; Sep29–30 organization revision |
| E13-9: separately standardized order/access | [Core probes](../../core_benchmark_v1/probes.py) | `core_benchmark_v1/results/main/aggregate/probe_summary.csv`, O0 rows; `native_runs.csv` | Sep27 20:15/20:16 benchmark proposal |
| E13-10: cascade reset and boundary/impulse limits | Exp13 `run_a3`, `run_c`, `_response_metrics` | `A3_layer_reset_all_runs.csv`, `C_boundary_summary.csv`, `E_cross_tau_decoding_summary.csv` | Sep24 causal versus architectural interpretation |
| Execution and corrections | [implementation commit 254d724](https://github.com/hellowPluto78700/writingRing/commit/254d72421a4833c74bd858a8610c8b2ad579b5e0), [empty fix be1d57a](https://github.com/hellowPluto78700/writingRing/commit/be1d57a63b70785fa21d3550d8c1a7ecc3af8a5f), [13.1 implementation 7a6b5e6](https://github.com/hellowPluto78700/writingRing/commit/7a6b5e6a396f9de6213cf8ea16cba7b53e2e1cb1), [protocol fix 4503d3c](https://github.com/hellowPluto78700/writingRing/commit/4503d3cda621251a823ec722ee8b96f4a3e307bb) | Unity `sacct` task states in Appendix A; final manifests | Design/execution topics, full original failure logs unavailable |

Chat coverage audit: motivation (Sep24 temporal hierarchy), design (Sep24 A–E acceptance and controls), interpretation (Sep24, Sep28), successor transition (Sep29–30 organization/content selection) were all searched. The Sep27 titled thread “分支 · 评估基准实验方案” supplies standardized benchmark intent; the Sep24 topic “设计局部运动原语” supplies hierarchy motivation. Other retrievals did not reliably expose exact titles. No bibliography or remembered literature claim is used as a substitute for the experimental evidence.

## Appendix D. Chronological Reasoning Trace

| UTC date/time | Contemporaneous reasoning / decision | Evidence status and subsequent revision |
| --- | --- | --- |
| Sep24 14:54–14:58 | Ask whether L2 is learned hierarchy or longer filtering; separate τ, functional window, abstraction | Pre-run scientific motivation recovered from Project history |
| Sep24 17:57–18:17 | Propose lag, suffix/reset, spike probes, stroke controls, impulse response | Proposed ordering/names evolved; final repository A1–E definitions take precedence |
| Sep24 18:29 | User accepts v2 and separate A–E outputs on Slurm | Implementation `254d724`, executed task arrays |
| Sep24 22:22–22:37 | First finalizer fails; fixed finalizer succeeds; interpret longer L2 window without slower long-lag state | Scheduler plus `be1d57a`; representation resets do not directly measure prediction |
| Sep27 20:15–20:16 | User requests locked user split, three seeds and both bias modes before broad conclusions | Motivation for later Core standardized order/readout comparisons, not retroactive Exp13 training |
| Sep25–26 execution; Sep27–28 review | Exp13.1 F/G/H and correction complete | Original F/H completed but superseded by `4503d3c` reruns; accounting dates distinguish execution from later discussion |
| Sep28 01:46 | First synthesis: L2 transfer improves; L3 seen-user history gains stronger | Supported in F/H, but narrowed: G writer excess falls in L3 and H endpoint OOD also improves |
| Sep29 15:11 and later review | Reconsider history organization and representation/readout mismatch | Exp14 tests consistency; no assumption that more depth alone solves transfer |
| Sep30 19:49–20:10 | Stage report through Exp14.1 frames remaining issue as content selection/organization | Retrospective synthesis; not an initial Sep24 belief and not extra Exp13 execution |

The scientific chain is therefore preserved with its revisions: architecture supplies state → causal replay shows longer communicated context → cross-user tests limit abstraction claims → consistency training becomes a testable next step. Incomplete full-chat access prevents claiming this is an exhaustive transcript chronology, but all four requested reasoning stages have recovered evidence.
