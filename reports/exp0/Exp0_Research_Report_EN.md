# Exp0 — Direct Classification with Multi-timescale SNNs, Firing Regularization, and Endpoint-tail Control

**Verification date:** 2026-10-02  
**Primary experiment period:** 2026-09-08 to 2026-09-11. The older growing-prefix branch was implemented on 2026-08-28.

This report is based on the existing Unity artifacts, the GitHub implementation, and historical-discussion retrieval. Values from Unity files take precedence. Balanced accuracy (BA) is reported as a percentage; ± is the sample standard deviation across seeds, not a confidence interval.

**Scope and numbering.** The repository does not contain a single experiment series that represents all work called “Exp0.” The main line is `experiment_0_1_general_comparison` → 0.1.1–0.1.5 → `experiment_0_2_endpoint_tail_regularization` → shift-resolved analysis → 0.2.2. The older `experiment_0_1_growing_prefix_relative10` and `withGyro/experiment_0_*` are independent research threads that reused the numbering. They are reported separately and are never averaged with the main line. No independently verifiable training series named `experiment_0_2_1` was found; this report does not invent an experiment to fill that numbering gap.

## 1. Executive Summary

The main Exp0 question was whether a feed-forward multi-τ SNN could turn raw event trajectories into class evidence readable by a simple WholeCount head by progressively extending the timescale across layers. It also asked whether suppressing sustained firing in longer-timescale layers could improve classification or restore information propagation.

The architecture comparison produced two clear findings. Binary `short_mid` with WholeCount CE reached **45.72±2.70%** test BA, `mid_long` reached **40.08±3.06%**, and the three-layer `short_mid_long` reached **21.87±1.57%**. A local `(234)→(234)` spiking-output WholeCount control reached **50.09±2.50%**. A separate local backbone trained with a temporary analog head reached **59.54±3.21%** after a frozen Fixed250+Linear readout, while the fixed-split Raw Relative10+Linear baseline reached **73.95%**. Thus, simply adding depth or longer τ did not automatically produce a more readable representation. An analog temporal decoder could extract additional class information, although that comparison also changes the training head, τ configuration, and readout, so its gain cannot be attributed solely to the readout.

No regularization series produced a stable BA improvement. Direct P2+A1 and the documentation-faithful formulation both collapsed many conditions to near chance. Normalization, warmup, and initial gradient calibration moderated part of the collapse but did not restore the three-layer long-τ network. Endpoint-tail penalties reduced tail activity and valid activity together, with incompletely matched compute budgets. Exp0.2.2 also exposed intervention-validity failures: `rho_max=1` makes the saturation hinge identically zero, relative-tail calibration has small-denominator gradient explosions, and the capacity floor did not demonstrate preservation of useful representations.

The strongest conclusion is that, under this protocol, reducing activity and improving classification are different objectives. The tested global and tail firing penalties did not reliably repair the disadvantage of long-τ architectures. Exp0 did not show that persistent activity is useless, and it did not directly test whether its information transfers across users.

## 2. Context and Motivation

Historical discussion retrieval identifies a new general-comparison thread on 2026-09-08. Raw Fixed250/Relative10 linear temporal baselines and local SNN representations with external temporal decoders already existed. The new experiments aligned inputs, user split, seeds, and communication capacity to test whether a feed-forward hierarchy from local to medium to long timescales could directly create class evidence.

The initial comparison internalized local temporal processing and longer evidence accumulation in the SNN instead of applying Relative10 or Fixed250 to raw input first. Performance and firing-pattern checks then raised the sustained-activity question. This led from direct P2+A1 regularization to the faithful formulation, dimensional fixes, gradient calibration, and finally an endpoint-following zero-input tail.

The older Exp0.1 had a different motivation. Full-gesture Relative10 was strong, but its bins depend on the true endpoint. It asked whether a frozen full-gesture classifier would still work when ten bins were recomputed for each increasingly long prefix. It did not train an SNN.

Historical context is evidence-bounded: retrieval returned dated summaries rather than complete transcripts, stable conversation URLs, or all discussion turns. This report therefore keeps the reconstructed motivation separate from questions explicitly stated in code and READMEs.

## 3. Research Questions and Hypotheses

| Hypothesis | Falsifiable prediction | Operation and control | Main outcomes | Competing explanation |
| --- | --- | --- | --- | --- |
| H1: Longer-τ hierarchy creates longer-range direct evidence | Adding a long layer or longer shift group increases direct test BA | `short_mid`, `mid_long`, `short_mid_long`; same split, seeds, objective, and cap | valid Output WholeCount BA | deeper optimization difficulty, spiking head limitation, state mixing, or communication saturation |
| H2: Hidden event capacity is the main bottleneck | cap31 consistently beats cap1 in matched settings | binary vs `multi_h`; output cap fixed at 1 | BA and hidden event rate | more events amplify drive or redundancy rather than transferable information |
| H3: Abnormal firing can be repaired by P2+A1 | firing falls while BA holds or improves | paired regularization-on vs frozen-off | BA, dead fraction, rate, gradient scale | useful signal is also suppressed; scale or gradient is too strong |
| H4: Failure mainly comes from mathematical scale and early intervention | temporal/τ normalization, warmup, or calibration recovers classification | 0.1.4 equal-30-epoch `no_reg`; 0.1.5 targets 5/10/20% | BA, gradient ratio, cosine | initial calibration drifts; the objective does not distinguish useful activity |
| H5: Endpoint tail can be separated from valid computation | tail-only loss reduces tail with less valid-BA damage | `a1`, `a1_p2`, `tail`, `a1_tail`, frozen `none` | tail/valid activity and BA | shared weights couple valid and tail computation; training budgets differ |
| H6: Long-only, relative-tail, and capacity-floor terms avoid global silence | long tail/valid ratio improves while valid FR and BA remain | 0.2.2 five conditions from a shared epoch-5 warmup fork | BA, s6/s7 FR, ratio, output-weight use | hinge is inactive; small denominator; composite calibration overwhelms the floor |

These are operational hypotheses reconstructed from the implementation and historical discussion; they were not necessarily preregistered in this exact H1–H6 form. Depth and timescale are not fully orthogonal in H1, so the design cannot identify one factor in isolation.

## 4. Experimental Design

The general comparison used 3 architectures × 2 objectives × 2 hidden caps × 5 seeds: 60 direct runs. It also used a local `(234)→(234)` backbone with a temporary analog head and timestep CE, giving 2 caps × 5 seeds = 10 runs, plus two deterministic Raw Linear baselines.

Exp0.1.1 added the direct local234 spiking-output control (2 objectives × 2 caps × 5 seeds = 20). Exp0.1.2 trained 60 adapted-regularization-on runs and reused the original 60 off controls. Exp0.1.3 trained 30 faithful-regularization-on binary runs. Exp0.1.4 used 6 conditions × 3 architectures × 2 objectives × 5 seeds = 180 paired 30-epoch runs. Exp0.1.5 used 3 strengths × 3 architectures × 2 objectives × 5 seeds = 90 runs with 50–100 epoch budgets.

Exp0.2 trained 3 architectures × 2 objectives × 4 new profiles × 3 seeds = 72 new 50-epoch runs and reevaluated 18 frozen `none` controls. Shift-resolved analysis reevaluated 90 existing checkpoints; it was not 90 new training runs. Exp0.2.2 used 2 architectures × 5 conditions × 3 seeds = 30 runs, sharing six epoch-5 warmup models and a train-only healthy reference.

Important control limitations: the correct equal-budget control for 0.1.4 is its own 30-epoch `no_reg`; 0.1.5 compares partly early-stopped new models with original 100-epoch models; and 0.2 new runs used 50 epochs while `none` checkpoints came from 100 epochs. The latter comparisons are not claimed as strict equal-compute causal effects.

## 5. Implementation

### 5.1 Data, split, and randomness

The main series uses 64 Hz, 30-channel unsigned weighted-wavelet events (Raw64), Action0+Action1, and 12 classes. Physical IMU channels are not SNN inputs. Labels are A/B/C/D/E/G/H/I/J/K/L/X. The main series has a fixed user-disjoint split with `split_seed=12345`, 853 samples, and artifact counts of 581/126/146 train/validation/test samples across 14/3/3 users. Training seeds 11/23/37/53/71 change model initialization and loader order; **they are not five user splits**. Exp0.2/0.2.2 use only seeds 11/23/37.

Train users: `user_0`, `user_1`, `user_11`, `user_12`, `user_13`, `user_14`, `user_15`, `user_18`, `user_19`, `user_2`, `user_20`, `user_5`, `user_7`, `user_8`. Validation users: `user_16`, `user_4`, `user_9`. Test users: `user_10`, `user_3`, `user_6`.

The old growing-prefix experiment uses 853 samples, 20 users, and 12 classes, but its five seeds are five distinct user-disjoint 12/4/4 user splits. It must not be confused with the fixed-split main series. withGyro is an independent Action0 event schema and is discussed separately.

### 5.2 Backbone, losses, and checkpoint selection

Each hidden layer has width 128 and bias-free matrices. For shift group `s`, α_s = 1−2^(−s) and τ_s = −Δt/log(α_s). At 64 Hz, s2–s7 correspond approximately to 54.31, 117.01, 242.10, 492.15, 992.17, and 1992.18 ms. `tau_mem=22.54 ms`, threshold = 0.5, and surrogate slope = 25. The recurrence is `I_t = α ⊙ I_(t−1) + W z_(t−1 layer)`, followed by membrane update, capped multi-threshold spiking, and subtractive reset. State is reset at the start of each sequence. Binary cap1 and multi_h cap31 use the same reset/update semantics.

The direct head is `Linear(128,12,bias=False)` plus binary MacroLIF. WCCE trains `5 × mean_valid(output_spikes / output_cap)` once per gesture; deployment uses the valid spike sum. TSCE trains `5 × output_spike_t / cap` at each valid timestep, so longer sequences contribute more terms. The temporary analog probe head has a bias, and its external `LogisticRegression` has the default affine intercept.

Adam uses lr=0.001, weight_decay=0, batch_size=128. Main 0.1/0.1.1/0.1.2/0.1.3 runs nominally use 100 epochs. Direct checkpoints are selected by highest validation WholeCount BA, with lower normalized WholeCount CE as tiebreaker. Training gradients come from train objectives; validation selects the checkpoint; test never selects it. The analog-probe family selects a backbone by analog validation metrics before freezing it for a linear probe.

Fixed250 uses 16 fixed 64-Hz bins and train-only `StandardScaler` plus validation-selected logistic-regression C. Raw feature dimension is 480; L2 feature dimension is 2048. Relative10 splits each valid full-gesture prefix into ten chunks and sums each chunk (Raw feature dimension 300). Relative10 relies on the full gesture endpoint and is not an online causal phase estimate.

### 5.3 Regularization evolution

Exp0.1.2 uses hidden pre-reset membrane: `U_t=0.99U_(t−1)+v_pre_t`, updated only on valid timesteps and frozen after endpoint. P2 is averaged by batch/neuron then layer; A1 is cap-normalized hidden activity mean; `L=L_task+0.01P2+0.1A1`; output is not penalized.

Exp0.1.3 uses post-reset membrane, direct P2 summation over all spiking layers including output, and last-hidden-only A1 after per-sample valid-time division. It removes batch/neuron/cap normalization while retaining coefficients 0.01/0.1. This is a bundled mathematical change; it cannot isolate pre- versus post-reset effects. The cited SAE-Dense source document was not recovered in full, so “faithful” means faithful to the implementation and README claim, not independently verified paper equivalence.

Exp0.1.4 adds temporal normalization by valid EW mass, τ-gain normalization only in the P2 path, and a `min(1, epoch/10)` warmup. `all_fixes` combines these while leaving A1 unchanged. Exp0.1.5 measures the initial hidden incoming-weight gradient ratio on five train batches, sets frozen κ = ρ/r0 for ρ=5/10/20%, and ramps it for ten epochs. This is initialization calibration, not a train-time constrained budget. Early stopping uses min50/max100/patience30; the final checkpoint remains validation WholeCount BA.

Exp0.2 explicitly zeroes input after endpoint for about 600 ms. The three tail phases 0–200/200–400/400–600 ms have weights 1/2/4 divided by 7. Loss is hidden-only and classification remains endpoint-truncated. It uses train-only 5% initial gradient calibration and a 10-epoch ramp. Exp0.2.2 defines “long” as final hidden s6+s7, excluding s5. Epochs 1–5 are shared WCCE warmup; 6–15 ramp and 16–50 full strength. The healthy short_mid train-only s4/s5 occupancy P95 gives `rho_max`; tail/valid P90 gives gamma; the capacity floor is 0.7× warmup FR matched by sample and shift. Actual reference values are `rho_max=1` and gamma ≈ 2.4155/1.4434/0.8457. The repair for zero regularizer gradient sets κ=1 when inactive and records the status; those cases did not achieve 5% gradient calibration.

## 6. Execution History

| Stage | Job / array | Scheduler status | Analysis eligibility |
| --- | --- | --- | --- |
| 0.1 general | 64110128 | COMPLETED ×70 | successful artifacts; reused controls not double-counted |
| Raw baseline | 64110129 | COMPLETED ×1 | successful artifact |
| 0.1.1 | 64133787 | COMPLETED ×20 | successful artifacts |
| 0.1.2 | 64168650 | COMPLETED ×60 | successful artifacts |
| 0.1.3 first / second attempts | 64187056 / 64187136 | FAILED ×30 each | failed tasks excluded |
| 0.1.3 valid run | 64187361 | COMPLETED ×30 | successful artifacts |
| 0.1.4 | 64192803 | COMPLETED ×180 | successful artifacts |
| 0.1.5 first attempt | 64216785 | FAILED ×89; CANCELLED ×1 | failed tasks excluded |
| 0.1.5 second attempt | 64217820 | COMPLETED ×89; FAILED ×1 | strict aggregate uses 89; task 77 reported separately |
| 0.2 new training | 64288918 | COMPLETED ×72 | successful artifacts |
| shift analysis | 64299539 | COMPLETED ×90 | successful artifacts |
| 0.2.2 initial main | 64300930 | COMPLETED ×24; FAILED ×6 | failed tasks excluded |
| 0.2.2 completion array | 64301942 | COMPLETED ×30 | 24 reused artifacts plus 6 completed sat runs |
| 0.2.2 finalizer | 64301943 | COMPLETED ×1 | valid summary |

The exact cause of the two Exp0.1.3 failed rounds is **not verified from available artifacts**. In 0.1.5, failed task 77 maps to short_mid/TSCE/20%/seed37. It has a parseable evaluation (61 epochs, selected best epoch 61, test BA 12.7381%) but no verified successful rerun. It is excluded from strict aggregation while retained as an artifact. In 0.2, the field named `epochs_trained` is actually the checkpoint `best_epoch`, not total epochs; each new history has 50 rows ending at epoch 50. In 0.2.2 the first finalizer 64300931 was CANCELLED with Start=None. Commit `f9c0f2b2` repaired inactive-gradient handling; the 24 previously successful cases are legacy artifacts and only the six saturation cases were rerun under the repaired code.

## 7. Results

### 7.1 General comparison and local234 controls

| Family | Architecture | Objective | Variant | Test BA (%) |
| --- | --- | --- | --- | --- |
| direct_snn | mid_long | timestep_ce | binary | 30.40±4.50 |
| direct_snn | mid_long | timestep_ce | multi_h | 23.87±5.88 |
| direct_snn | mid_long | whole_count_ce | binary | 40.08±3.06 |
| direct_snn | mid_long | whole_count_ce | multi_h | 41.38±1.17 |
| direct_snn | short_mid | timestep_ce | binary | 28.15±2.52 |
| direct_snn | short_mid | timestep_ce | multi_h | 29.22±2.71 |
| direct_snn | short_mid | whole_count_ce | binary | 45.72±2.70 |
| direct_snn | short_mid | whole_count_ce | multi_h | 43.46±2.41 |
| direct_snn | short_mid_long | timestep_ce | binary | 13.21±1.57 |
| direct_snn | short_mid_long | timestep_ce | multi_h | 13.60±1.35 |
| direct_snn | short_mid_long | whole_count_ce | binary | 21.87±1.57 |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 18.85±2.88 |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 59.54±3.21 |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 60.10±1.44 |
| raw_linear_baseline | fixed250 | posthoc_linear | deterministic | 54.85 |
| raw_linear_baseline | relative10 | posthoc_linear | deterministic | 73.95 |

The direct local234 control reached 50.09±2.50% (WCCE/binary), 47.71±2.35% (WCCE/multi_h), 32.85±2.93% (TSCE/binary), and 32.42±4.35% (TSCE/multi_h). Raw Fixed250 and Relative10 are single deterministic fixed-user-split results, so no training-seed SD is assigned. The analog-trained local backbone had native BA 49.82±3.13%/49.64±2.32% (binary/multi_h) and Fixed250-probe BA 59.54±3.21%/60.10±1.44%, a same-checkpoint gain of about 9.73/10.46 pp.

### 7.2 Adapted, faithful, and repaired regularization

Adapted regularization collapsed most conditions to approximately 8.33% balanced chance. For short_mid/WCCE/binary, end-layer FR fell from 9.918 to 0.0408 events/neuron/s while BA fell from 45.72% to 8.33%. Faithful regularization similarly gave short_mid 8.33% and mid_long 6.75±1.54%; finite samples and biased predictions allow BA below nominal chance.

In the 30-epoch paired repair study, `all_fixes` reduced the original collapse but did not beat the equal-budget `no_reg` condition for any architecture/objective. For example, short_mid/WCCE was 18.66±2.69% with all fixes versus 22.66±3.85% with no regularization. In gradient-calibrated WCCE, target 5% gave short_mid 43.84±5.67% (frozen original 45.72±2.70%), mid_long 37.36±5.27% (40.08±3.06%), and short_mid_long 8.97±0.93% (21.87±1.57%). Increasing strength from 5% to 20% did not yield a stable recovery.

### 7.3 Endpoint-tail and capacity-preserving losses

| Architecture | Endpoint profile | Test BA (%) |
| --- | --- | --- |
| mid_long | none / tail | 41.68±1.54 / 39.26±2.86 |
| short_mid | none / tail | 46.58±2.28 / 36.55±3.78 |
| short_mid_long | none / tail | 22.97±0.51 / 15.03±0.71 |

These are n=3 artifact comparisons where `none` comes from 100-epoch checkpoints and new profiles use 50 epochs, so they are not equal-budget effect estimates. Tail-only loss reduced tail more than some alternatives but provided no stable test gain. For short_mid/WCCE, valid hidden activity fell 0.08885→0.05150 and final-hidden tail area 3.759→1.830, alongside BA 46.58→36.55%.

In Exp0.2.2, `mid_long` WCCE-only and saturation had exactly the same per-seed BA, 38.86±5.19%. Because binary occupancy is at most 1 and `rho_max=1`, the saturation hinge is always zero; train history confirms maximum `train_sat_loss=0`. It is an inactive-intervention control, not evidence against saturation control. Relative-tail and the capacity combination did not improve mean BA. For mid_long, valid long FR fell 15.99→11.21 Hz and tail FR 25.99→19.21 Hz, but tail/valid ratio worsened 1.626→1.711. Relative-tail raw gradient ratios reached 10^9–10^10, yielding correspondingly tiny κ; a large raw regularizer value therefore did not imply a large effective training signal.

### 7.4 Older growing-prefix result

The frozen full-gesture Relative10 decoder obtained **71.47±7.79%** test BA across five distinct user splits. This is not the same estimand as the main-series single-split 73.95%. At 10/20/30 samples (156/313/469 ms), prefix BA was approximately 9.03/8.82/9.27% with 100% coverage. At 80 samples it was 26.85±6.81% with 68.04% coverage; at 110 samples, 56.08±11.87% with 23.27% coverage. Later points contain only a long-gesture active cohort and must be read with matched full-gesture cohort performance and coverage. The result shows that the full-gesture Relative10 decoder is not a reliable early-prefix decoder; it does not show that every online or causal decoder must fail.

## 8. Diagnostics, Interpretation, and Negative Evidence

Initial gradient calibration did not maintain its target across training. In 0.1.5, mid_long/TSCE effective ratio averaged about 0.995 at epoch 5 and 0.650 at epoch 10 despite a 5% target; mid_long/WCCE was about 0.1725 at epoch 100. Measurements are first-batch snapshots at specified epochs, not every batch.

The evidence supports a configuration disadvantage for deeper/longer direct models but cannot uniquely identify τ, depth, information loss, or optimization as its cause. Multi_h did not provide a stable gain, so greater hidden spike capacity is not a sufficient repair. The analog local-backbone probe directly supports a readability gap: trajectory information can be decoded externally. It does not prove that the native model never uses history.

The tested firing penalties reduce activity but do not stably improve held-out-user BA. Tail-only loss remains coupled to valid computation through shared parameters. Saturation was never actually tested because its hinge was inactive; relative-tail and the capacity floor did not stably retain classification while suppressing persistence. These are negative results about the tested parameterizations, not proof that all persistent spikes are useless.

Exp0 did not measure persistent-group class selectivity, gating, Reset/Replay, seen-vs-OOD retrieval, or paired Fixed250 order-versus-shuffle probes. It did not establish that long history is lost from L2, that higher abstractions stabilize, that all selective regularizers or gating fail, or that a universally optimal τ/readout exists. Exact run SHA for every artifact, 0.1.3 failed stderr, the source of a successful rerun for task 77, and old prefix/withGyro scheduler IDs remain **not verified from available artifacts**.

## 9. Evidence and Traceability

Unity root: `/home/zhaolongwei_umass_edu/projects/writingRing`. Paths in the source evidence are relative to that root. GitHub: [hellowPluto78700/writingRing](https://github.com/hellowPluto78700/writingRing). Tables were generated from downloaded CSVs and cross-checked against per-seed means/sample SD, scheduler state, checkpoint-selection logic, healthy thresholds, 50-epoch histories, inactive loss, and artifact counts. No retraining, job submission, Unity-data modification, or unrelated CI was performed for this report.

The principal implementation commits are: `3cc558fe` (old prefix), `2b277af5` (general comparison), `2ed3e52e` (local234 direct), `79b7dda3` (adapted regularizer), `aee2558e` (faithful runner), `11f25ddd` (parallel repair), `fea78afc` and `2965dc9f` (gradient calibration), `d7f3ebd1` (endpoint tail), `03a3a0e1` (long group s6/s7), `070c9b3c` (capacity-preserving runner), and `f9c0f2b2` (inactive-gradient handling).

## Appendix A. Complete Per-seed Results

The following translated data appendix preserves the available selected-checkpoint rows for the main series. It is intentionally separate from the primary conclusions: it includes historical artifacts and records strict-inclusion status.

## Appendix A. Complete Per-seed Results for the Main Series

Each row contains the seed, validation-selected best epoch, and train/validation/test BA. These are selected-checkpoint metrics, not final-epoch training states. Rows marked strict = no are retained as historical artifacts only and are excluded from strict aggregates.


### experiment_0_1_1_local234_wholecount

Source: `notebooks/artifacts/experiment_0_1_1_local234_wholecount/local234_wholecount_v1/runs.csv`.

| architecture | objective | variant | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_234x2 | whole_count_ce | binary | 11 | 84 | 85.759 | 47.591 | 47.750 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | binary | 23 | 83 | 88.007 | 44.740 | 52.326 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | binary | 37 | 88 | 89.106 | 41.707 | 50.404 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | binary | 53 | 94 | 87.566 | 47.226 | 52.661 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | binary | 71 | 95 | 91.138 | 50.679 | 47.294 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | multi_h | 11 | 77 | 81.936 | 45.557 | 47.131 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | multi_h | 23 | 87 | 85.977 | 47.858 | 51.501 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | multi_h | 37 | 85 | 82.848 | 45.569 | 48.191 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | multi_h | 53 | 92 | 83.495 | 44.154 | 45.499 | yes / artifact-only legacy prefix |
| local_234x2 | whole_count_ce | multi_h | 71 | 93 | 87.404 | 50.887 | 46.203 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | binary | 11 | 99 | 54.657 | 31.320 | 35.943 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | binary | 23 | 97 | 55.138 | 32.169 | 34.466 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | binary | 37 | 100 | 49.117 | 24.775 | 29.764 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | binary | 53 | 96 | 49.936 | 29.067 | 34.413 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | binary | 71 | 95 | 54.099 | 27.480 | 29.671 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | multi_h | 11 | 92 | 48.236 | 29.563 | 34.621 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | multi_h | 23 | 98 | 52.963 | 32.358 | 30.484 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | multi_h | 37 | 88 | 47.489 | 30.291 | 26.225 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | multi_h | 53 | 98 | 49.326 | 27.249 | 33.032 | yes / artifact-only legacy prefix |
| local_234x2 | timestep_ce | multi_h | 71 | 100 | 53.575 | 31.617 | 37.751 | yes / artifact-only legacy prefix |

### experiment_0_1_2_regularized_general_comparison

Source: `notebooks/artifacts/experiment_0_1_2_regularized_general_comparison/regularized_general_comparison_v1/runs.csv`.

| family | architecture | objective | variant | regularization | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_snn | short_mid_long | whole_count_ce | binary | on | 11 | 1 | 7.678 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | on | 23 | 1 | 6.875 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | on | 37 | 17 | 8.494 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | on | 53 | 2 | 9.061 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | on | 71 | 3 | 7.960 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | on | 11 | 1 | 8.175 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | on | 23 | 1 | 7.402 | 9.722 | 7.413 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | on | 37 | 3 | 7.594 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | on | 53 | 2 | 8.730 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | on | 71 | 4 | 8.143 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | on | 11 | 5 | 8.542 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | on | 23 | 1 | 6.875 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | on | 37 | 3 | 7.797 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | on | 53 | 2 | 8.900 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | on | 71 | 3 | 8.150 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | on | 11 | 1 | 8.175 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | on | 23 | 1 | 7.402 | 9.722 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | on | 37 | 3 | 7.781 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | on | 53 | 2 | 8.521 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | on | 71 | 3 | 8.341 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | on | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | on | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | on | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | on | 53 | 1 | 8.525 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | on | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | on | 23 | 2 | 8.368 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | on | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | on | 53 | 1 | 8.525 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | on | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | on | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | on | 37 | 1 | 8.333 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | on | 53 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | on | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | on | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | on | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | on | 53 | 1 | 8.525 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | on | 11 | 15 | 11.280 | 12.698 | 13.366 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | on | 23 | 13 | 14.580 | 16.865 | 8.788 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | on | 37 | 3 | 10.711 | 12.897 | 7.998 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | on | 53 | 4 | 6.413 | 10.417 | 9.037 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | on | 71 | 4 | 9.318 | 11.250 | 5.736 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | on | 11 | 7 | 10.164 | 13.532 | 11.374 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | on | 23 | 13 | 13.308 | 14.980 | 7.359 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | on | 37 | 4 | 11.194 | 13.115 | 5.249 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | on | 53 | 5 | 7.872 | 11.806 | 9.470 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | on | 71 | 2 | 7.605 | 11.647 | 8.216 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | on | 11 | 3 | 8.019 | 9.167 | 6.061 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | on | 23 | 32 | 10.823 | 11.806 | 12.933 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | on | 37 | 6 | 9.218 | 11.528 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | on | 53 | 3 | 6.976 | 9.028 | 7.652 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | on | 71 | 17 | 8.667 | 9.722 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | on | 11 | 3 | 7.241 | 9.226 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | on | 23 | 14 | 8.831 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | on | 37 | 4 | 10.801 | 11.250 | 6.061 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | on | 53 | 4 | 7.964 | 10.417 | 9.037 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | on | 71 | 3 | 8.144 | 10.417 | 7.251 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | not_applicable | 11 | 100 | 96.490 | 55.341 | 62.921 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | not_applicable | 23 | 73 | 95.143 | 54.601 | 60.274 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | not_applicable | 37 | 93 | 98.103 | 50.530 | 60.512 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | not_applicable | 53 | 88 | 98.418 | 52.253 | 59.777 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | not_applicable | 71 | 59 | 99.001 | 56.086 | 54.235 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | not_applicable | 11 | 88 | 93.724 | 51.171 | 61.013 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | not_applicable | 23 | 86 | 92.610 | 49.934 | 61.979 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | not_applicable | 37 | 87 | 97.609 | 53.658 | 58.482 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | not_applicable | 53 | 98 | 91.412 | 52.698 | 58.974 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | not_applicable | 71 | 67 | 92.869 | 48.095 | 60.039 | yes / artifact-only legacy prefix |

### experiment_0_1_3_doc_faithful_regularization

Source: `notebooks/artifacts/experiment_0_1_3_doc_faithful_regularization/doc_faithful_sae_dense_binary_v1/runs.csv`.

| family | architecture | objective | variant | regularization | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_snn | short_mid_long | whole_count_ce | binary | doc_sae_dense_on | 11 | 2 | 7.977 | 8.333 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | doc_sae_dense_on | 23 | 1 | 7.038 | 8.333 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | doc_sae_dense_on | 37 | 3 | 7.091 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | doc_sae_dense_on | 53 | 2 | 8.836 | 8.333 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | doc_sae_dense_on | 71 | 2 | 7.629 | 8.333 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | doc_sae_dense_on | 11 | 2 | 7.977 | 8.333 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | doc_sae_dense_on | 23 | 1 | 7.038 | 8.333 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | doc_sae_dense_on | 37 | 3 | 7.091 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | doc_sae_dense_on | 53 | 2 | 8.836 | 8.333 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | doc_sae_dense_on | 71 | 2 | 7.629 | 8.333 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | doc_sae_dense_on | 11 | 1 | 8.320 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | doc_sae_dense_on | 23 | 2 | 8.317 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | doc_sae_dense_on | 37 | 1 | 8.333 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | doc_sae_dense_on | 53 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | doc_sae_dense_on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | doc_sae_dense_on | 11 | 1 | 8.320 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | doc_sae_dense_on | 23 | 2 | 8.317 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | doc_sae_dense_on | 37 | 1 | 8.333 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | doc_sae_dense_on | 53 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | doc_sae_dense_on | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | doc_sae_dense_on | 11 | 5 | 8.108 | 12.778 | 7.643 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | doc_sae_dense_on | 23 | 84 | 8.160 | 9.167 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | doc_sae_dense_on | 37 | 7 | 6.364 | 12.698 | 5.835 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | doc_sae_dense_on | 53 | 4 | 6.925 | 9.028 | 4.545 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | doc_sae_dense_on | 71 | 6 | 7.589 | 11.607 | 7.413 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | doc_sae_dense_on | 11 | 5 | 8.108 | 12.778 | 7.643 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | doc_sae_dense_on | 23 | 84 | 8.160 | 9.167 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | doc_sae_dense_on | 37 | 7 | 6.364 | 12.698 | 5.835 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | doc_sae_dense_on | 53 | 4 | 6.925 | 9.028 | 4.545 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | doc_sae_dense_on | 71 | 6 | 7.589 | 11.607 | 7.413 | yes / artifact-only legacy prefix |

### experiment_0_1_4_parallel_regularization_ablation

Source: `notebooks/artifacts/experiment_0_1_4_parallel_regularization_ablation/parallel_regularization_ablation_v1/runs.csv`.

| family | architecture | objective | variant | condition | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_snn | short_mid_long | whole_count_ce | binary | no_reg | 11 | 30 | 26.100 | 12.004 | 13.799 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | no_reg | 23 | 30 | 27.462 | 15.615 | 12.121 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | no_reg | 37 | 29 | 26.636 | 14.921 | 10.381 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | no_reg | 53 | 27 | 23.887 | 15.278 | 13.203 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | no_reg | 71 | 29 | 26.212 | 12.235 | 10.227 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | no_reg | 11 | 1 | 8.375 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | no_reg | 23 | 30 | 20.524 | 15.972 | 16.667 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | no_reg | 37 | 12 | 11.283 | 11.111 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | no_reg | 53 | 27 | 16.752 | 11.806 | 9.957 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | no_reg | 71 | 29 | 15.621 | 11.111 | 9.199 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | no_reg | 11 | 30 | 29.428 | 20.119 | 18.786 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | no_reg | 23 | 29 | 34.251 | 22.414 | 19.663 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | no_reg | 37 | 28 | 31.747 | 19.266 | 21.577 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | no_reg | 53 | 29 | 36.463 | 19.299 | 27.799 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | no_reg | 71 | 27 | 33.262 | 21.488 | 25.460 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | no_reg | 11 | 30 | 17.785 | 9.722 | 11.742 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | no_reg | 23 | 29 | 16.893 | 11.111 | 11.223 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | no_reg | 37 | 30 | 21.119 | 15.278 | 13.203 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | no_reg | 53 | 29 | 17.703 | 11.111 | 10.823 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | no_reg | 71 | 27 | 21.818 | 16.898 | 19.123 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | no_reg | 11 | 27 | 63.228 | 30.007 | 34.995 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | no_reg | 23 | 30 | 69.020 | 30.146 | 35.491 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | no_reg | 37 | 27 | 56.378 | 25.238 | 25.170 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | no_reg | 53 | 30 | 61.811 | 33.757 | 33.400 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | no_reg | 71 | 30 | 70.653 | 32.282 | 41.620 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | no_reg | 11 | 29 | 28.521 | 22.282 | 17.136 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | no_reg | 23 | 27 | 32.179 | 19.524 | 22.628 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | no_reg | 37 | 29 | 28.544 | 19.960 | 18.539 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | no_reg | 53 | 27 | 24.339 | 18.016 | 19.621 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | no_reg | 71 | 26 | 27.970 | 20.271 | 18.297 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | original_reg | 11 | 1 | 7.678 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | original_reg | 23 | 1 | 6.875 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | original_reg | 37 | 17 | 8.333 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | original_reg | 53 | 2 | 9.061 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | original_reg | 71 | 3 | 7.960 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | original_reg | 11 | 5 | 8.542 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | original_reg | 23 | 1 | 6.875 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | original_reg | 37 | 3 | 7.797 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | original_reg | 53 | 2 | 8.900 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | original_reg | 71 | 3 | 8.150 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | original_reg | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | original_reg | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | original_reg | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | original_reg | 53 | 1 | 8.525 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | original_reg | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | original_reg | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | original_reg | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | original_reg | 37 | 1 | 8.333 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | original_reg | 53 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | original_reg | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | original_reg | 11 | 15 | 11.280 | 12.698 | 13.366 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | original_reg | 23 | 13 | 14.580 | 16.865 | 8.788 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | original_reg | 37 | 3 | 10.711 | 12.897 | 7.998 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | original_reg | 53 | 4 | 6.413 | 10.417 | 9.037 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | original_reg | 71 | 4 | 9.318 | 11.250 | 5.736 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | original_reg | 11 | 3 | 8.019 | 9.167 | 6.061 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | original_reg | 23 | 30 | 9.946 | 11.806 | 12.338 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | original_reg | 37 | 6 | 9.218 | 11.528 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | original_reg | 53 | 3 | 6.976 | 9.028 | 7.652 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | original_reg | 71 | 17 | 8.667 | 9.722 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | temporal_norm | 11 | 1 | 8.388 | 9.028 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | temporal_norm | 23 | 1 | 9.171 | 9.722 | 7.846 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | temporal_norm | 37 | 3 | 9.981 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | temporal_norm | 53 | 30 | 15.098 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | temporal_norm | 71 | 27 | 15.236 | 9.722 | 9.361 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | temporal_norm | 11 | 1 | 7.903 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | temporal_norm | 23 | 1 | 7.777 | 10.417 | 6.656 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | temporal_norm | 37 | 1 | 7.763 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | temporal_norm | 53 | 2 | 9.564 | 8.333 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | temporal_norm | 71 | 1 | 8.127 | 9.028 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | temporal_norm | 11 | 28 | 13.621 | 9.028 | 10.119 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | temporal_norm | 23 | 29 | 20.185 | 15.417 | 16.388 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | temporal_norm | 37 | 30 | 16.759 | 9.722 | 12.159 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | temporal_norm | 53 | 29 | 16.644 | 10.417 | 16.385 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | temporal_norm | 71 | 30 | 17.316 | 9.722 | 9.715 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | temporal_norm | 11 | 1 | 8.679 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | temporal_norm | 23 | 1 | 8.229 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | temporal_norm | 37 | 30 | 13.546 | 9.722 | 9.167 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | temporal_norm | 53 | 1 | 9.458 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | temporal_norm | 71 | 1 | 8.727 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | temporal_norm | 11 | 28 | 49.317 | 21.627 | 27.285 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | temporal_norm | 23 | 26 | 47.252 | 24.035 | 23.689 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | temporal_norm | 37 | 19 | 38.404 | 22.421 | 22.613 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | temporal_norm | 53 | 28 | 52.341 | 25.886 | 29.038 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | temporal_norm | 71 | 9 | 26.930 | 17.817 | 18.285 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | temporal_norm | 11 | 30 | 23.976 | 15.972 | 14.708 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | temporal_norm | 23 | 27 | 22.932 | 22.361 | 17.677 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | temporal_norm | 37 | 30 | 26.855 | 19.167 | 15.823 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | temporal_norm | 53 | 22 | 17.445 | 14.583 | 14.394 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | temporal_norm | 71 | 30 | 20.806 | 14.028 | 13.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | tau_gain_norm | 11 | 1 | 8.256 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | tau_gain_norm | 23 | 1 | 7.244 | 9.722 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | tau_gain_norm | 37 | 5 | 9.913 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | tau_gain_norm | 53 | 2 | 9.387 | 8.333 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | tau_gain_norm | 71 | 3 | 9.243 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | tau_gain_norm | 11 | 1 | 8.269 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | tau_gain_norm | 23 | 1 | 7.611 | 9.722 | 8.604 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | tau_gain_norm | 37 | 4 | 8.809 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | tau_gain_norm | 53 | 2 | 9.035 | 8.333 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | tau_gain_norm | 71 | 3 | 8.683 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | tau_gain_norm | 11 | 1 | 8.839 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | tau_gain_norm | 23 | 1 | 8.213 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | tau_gain_norm | 37 | 1 | 8.811 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | tau_gain_norm | 53 | 1 | 9.263 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | tau_gain_norm | 71 | 1 | 8.727 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | tau_gain_norm | 11 | 1 | 8.494 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | tau_gain_norm | 23 | 2 | 8.843 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | tau_gain_norm | 37 | 1 | 8.333 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | tau_gain_norm | 53 | 1 | 9.250 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | tau_gain_norm | 71 | 1 | 8.679 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | tau_gain_norm | 11 | 27 | 32.037 | 17.282 | 21.357 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | tau_gain_norm | 23 | 22 | 33.901 | 18.988 | 19.215 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | tau_gain_norm | 37 | 25 | 37.944 | 20.337 | 20.435 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | tau_gain_norm | 53 | 22 | 29.079 | 20.893 | 16.022 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | tau_gain_norm | 71 | 27 | 37.333 | 22.480 | 16.593 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | tau_gain_norm | 11 | 30 | 16.742 | 15.278 | 14.794 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | tau_gain_norm | 23 | 29 | 18.491 | 12.143 | 11.580 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | tau_gain_norm | 37 | 28 | 18.405 | 17.917 | 9.351 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | tau_gain_norm | 53 | 1 | 7.795 | 10.198 | 6.006 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | tau_gain_norm | 71 | 30 | 16.024 | 10.556 | 6.818 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | warmup | 11 | 1 | 7.518 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | warmup | 23 | 1 | 6.557 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | warmup | 37 | 9 | 8.491 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | warmup | 53 | 2 | 9.061 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | warmup | 71 | 3 | 8.307 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | warmup | 11 | 4 | 8.171 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | warmup | 23 | 1 | 6.714 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | warmup | 37 | 9 | 8.333 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | warmup | 53 | 2 | 8.903 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | warmup | 71 | 3 | 8.143 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | warmup | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | warmup | 23 | 2 | 8.368 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | warmup | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | warmup | 53 | 1 | 8.897 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | warmup | 71 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | warmup | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | warmup | 23 | 2 | 8.160 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | warmup | 37 | 1 | 8.491 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | warmup | 53 | 1 | 8.699 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | warmup | 71 | 1 | 8.542 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | warmup | 11 | 5 | 8.920 | 11.944 | 8.994 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | warmup | 23 | 11 | 8.690 | 11.111 | 7.413 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | warmup | 37 | 3 | 11.075 | 11.369 | 8.074 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | warmup | 53 | 4 | 6.738 | 9.722 | 7.251 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | warmup | 71 | 3 | 9.742 | 11.806 | 6.656 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | warmup | 11 | 2 | 7.593 | 10.417 | 7.102 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | warmup | 23 | 11 | 8.314 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | warmup | 37 | 5 | 9.410 | 10.694 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | warmup | 53 | 5 | 7.449 | 8.333 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | warmup | 71 | 3 | 7.161 | 9.028 | 6.061 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | all_fixes | 11 | 28 | 17.587 | 10.417 | 9.957 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | all_fixes | 23 | 25 | 24.903 | 13.393 | 14.232 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | all_fixes | 37 | 24 | 22.621 | 13.194 | 10.768 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | all_fixes | 53 | 28 | 22.953 | 14.583 | 10.823 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | all_fixes | 71 | 26 | 22.415 | 13.624 | 10.465 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | all_fixes | 11 | 1 | 7.376 | 9.028 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | all_fixes | 23 | 28 | 15.323 | 13.194 | 11.147 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | all_fixes | 37 | 26 | 12.259 | 10.417 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | all_fixes | 53 | 27 | 14.718 | 9.028 | 7.489 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | all_fixes | 71 | 10 | 10.391 | 9.722 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | all_fixes | 11 | 29 | 24.220 | 17.361 | 15.134 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | all_fixes | 23 | 30 | 30.360 | 20.919 | 21.304 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | all_fixes | 37 | 29 | 28.068 | 18.730 | 17.840 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | all_fixes | 53 | 26 | 23.438 | 17.560 | 17.600 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | all_fixes | 71 | 28 | 29.890 | 25.562 | 21.422 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | all_fixes | 11 | 29 | 12.943 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | all_fixes | 23 | 27 | 13.958 | 8.333 | 10.195 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | all_fixes | 37 | 30 | 17.085 | 13.889 | 13.249 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | all_fixes | 53 | 30 | 14.009 | 10.417 | 11.147 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | all_fixes | 71 | 25 | 15.147 | 11.250 | 9.361 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | all_fixes | 11 | 24 | 55.745 | 28.142 | 33.867 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | all_fixes | 23 | 30 | 63.657 | 25.390 | 28.520 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | all_fixes | 37 | 25 | 51.251 | 20.714 | 21.760 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | all_fixes | 53 | 30 | 60.787 | 27.612 | 34.744 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | all_fixes | 71 | 30 | 66.430 | 23.019 | 35.754 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | all_fixes | 11 | 30 | 25.611 | 16.369 | 15.920 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | all_fixes | 23 | 29 | 34.030 | 21.746 | 22.920 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | all_fixes | 37 | 29 | 25.641 | 18.948 | 13.496 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | all_fixes | 53 | 24 | 19.734 | 18.175 | 15.812 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | all_fixes | 71 | 29 | 25.260 | 17.282 | 16.813 | yes / artifact-only legacy prefix |

### experiment_0_1_5_gradient_calibrated_all_fixes

Source: `notebooks/artifacts/experiment_0_1_5_gradient_calibrated_all_fixes/gradient_calibrated_all_fixes_v1/runs.csv`.

| family | architecture | objective | variant | target_grad_ratio | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.05 | 11 | 1 | 7.521 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.05 | 23 | 13 | 16.390 | 12.500 | 9.199 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.05 | 37 | 47 | 20.021 | 11.111 | 10.132 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.05 | 53 | 5 | 12.624 | 9.028 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.05 | 71 | 15 | 15.626 | 11.806 | 9.199 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.05 | 11 | 1 | 6.882 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.05 | 23 | 8 | 10.191 | 10.417 | 10.714 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.05 | 37 | 10 | 10.354 | 9.722 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.05 | 53 | 4 | 9.683 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.05 | 71 | 52 | 12.559 | 9.722 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.05 | 11 | 97 | 81.709 | 42.237 | 41.517 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.05 | 23 | 90 | 74.805 | 43.609 | 53.057 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.05 | 37 | 93 | 82.942 | 47.161 | 45.047 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.05 | 53 | 95 | 70.561 | 38.963 | 38.365 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.05 | 71 | 98 | 85.274 | 41.416 | 41.237 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.05 | 11 | 99 | 26.597 | 16.667 | 20.438 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.05 | 23 | 100 | 20.772 | 13.194 | 16.464 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.05 | 37 | 90 | 37.211 | 24.028 | 24.336 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.05 | 53 | 93 | 22.009 | 15.556 | 19.394 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.05 | 71 | 99 | 34.750 | 20.139 | 25.422 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.05 | 11 | 99 | 89.266 | 35.444 | 43.793 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.05 | 23 | 97 | 89.553 | 43.731 | 41.304 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.05 | 37 | 58 | 77.206 | 25.563 | 30.573 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.05 | 53 | 89 | 81.493 | 32.309 | 36.541 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.05 | 71 | 95 | 90.948 | 37.514 | 34.599 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.05 | 11 | 94 | 34.651 | 19.782 | 22.132 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.05 | 23 | 96 | 52.279 | 28.499 | 32.399 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.05 | 37 | 10 | 9.799 | 10.417 | 9.407 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.05 | 53 | 63 | 21.831 | 18.611 | 14.621 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.05 | 71 | 100 | 47.555 | 20.694 | 24.030 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.1 | 11 | 1 | 7.216 | 9.028 | 8.766 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.1 | 23 | 27 | 16.447 | 11.111 | 9.957 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.1 | 37 | 17 | 12.971 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.1 | 53 | 5 | 11.406 | 9.722 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.1 | 71 | 6 | 11.644 | 10.417 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.1 | 11 | 1 | 7.216 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.1 | 23 | 1 | 8.486 | 10.417 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.1 | 37 | 4 | 8.842 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.1 | 53 | 2 | 9.699 | 9.028 | 9.091 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.1 | 71 | 7 | 10.186 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.1 | 11 | 97 | 75.779 | 41.311 | 38.377 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.1 | 23 | 91 | 65.517 | 32.203 | 38.762 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.1 | 37 | 74 | 73.154 | 42.839 | 43.044 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.1 | 53 | 100 | 41.488 | 23.393 | 28.495 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.1 | 71 | 95 | 79.102 | 43.192 | 45.718 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.1 | 11 | 100 | 12.003 | 8.333 | 9.762 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.1 | 23 | 1 | 8.229 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.1 | 37 | 97 | 35.576 | 23.889 | 23.054 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.1 | 53 | 1 | 9.612 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.1 | 71 | 98 | 24.922 | 12.500 | 15.823 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.1 | 11 | 88 | 82.535 | 34.615 | 38.811 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.1 | 23 | 96 | 86.475 | 37.346 | 39.199 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.1 | 37 | 90 | 75.405 | 24.746 | 34.393 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.1 | 53 | 90 | 75.817 | 31.925 | 39.207 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.1 | 71 | 81 | 86.634 | 27.451 | 37.239 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.1 | 11 | 5 | 10.915 | 11.806 | 9.307 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.1 | 23 | 80 | 37.053 | 21.448 | 22.101 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.1 | 37 | 11 | 9.286 | 8.333 | 8.974 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.1 | 53 | 5 | 9.815 | 9.167 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.1 | 71 | 24 | 10.823 | 10.556 | 9.167 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.2 | 11 | 1 | 7.517 | 9.028 | 8.171 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.2 | 23 | 2 | 10.357 | 11.111 | 10.552 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.2 | 37 | 9 | 11.056 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.2 | 53 | 4 | 11.131 | 9.722 | 7.576 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 0.2 | 71 | 6 | 10.924 | 9.722 | 9.524 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.2 | 11 | 1 | 7.409 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.2 | 23 | 1 | 8.282 | 10.417 | 7.413 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.2 | 37 | 6 | 8.990 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.2 | 53 | 1 | 9.224 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 0.2 | 71 | 3 | 9.368 | 9.028 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.2 | 11 | 92 | 47.829 | 27.923 | 32.627 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.2 | 23 | 96 | 38.299 | 25.344 | 25.466 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.2 | 37 | 99 | 80.062 | 41.992 | 48.588 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.2 | 53 | 1 | 9.621 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 0.2 | 71 | 93 | 70.959 | 36.350 | 39.577 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.2 | 11 | 1 | 8.654 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.2 | 23 | 2 | 8.840 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.2 | 37 | 61 | 14.602 | 9.028 | 12.738 | no: Slurm FAILED |
| direct_snn | short_mid | timestep_ce | binary | 0.2 | 53 | 1 | 9.073 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 0.2 | 71 | 1 | 8.676 | 8.333 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.2 | 11 | 96 | 77.154 | 27.431 | 35.786 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.2 | 23 | 93 | 76.026 | 29.995 | 34.506 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.2 | 37 | 45 | 37.513 | 22.083 | 20.660 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.2 | 53 | 99 | 64.741 | 33.175 | 30.983 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 0.2 | 71 | 90 | 80.499 | 25.225 | 37.108 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.2 | 11 | 5 | 8.950 | 10.417 | 8.333 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.2 | 23 | 49 | 12.481 | 14.167 | 13.171 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.2 | 37 | 3 | 9.127 | 11.111 | 9.199 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.2 | 53 | 5 | 9.414 | 9.028 | 8.929 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 0.2 | 71 | 6 | 8.808 | 9.167 | 8.333 | yes / artifact-only legacy prefix |

### experiment_0_1_general_comparison

Source: `notebooks/artifacts/experiment_0_1_general_comparison/general_comparison_v1/runs.csv`.

| family | architecture | objective | variant | seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_snn | short_mid_long | whole_count_ce | binary | 11 | 99 | 49.378 | 21.005 | 23.357 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 23 | 100 | 47.897 | 22.235 | 23.147 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 37 | 93 | 47.826 | 18.889 | 22.396 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 53 | 79 | 40.167 | 16.481 | 19.788 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | binary | 71 | 100 | 51.515 | 20.648 | 20.674 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 11 | 87 | 33.237 | 17.930 | 18.967 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 23 | 78 | 35.110 | 20.430 | 22.535 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 37 | 99 | 40.604 | 18.056 | 20.720 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 53 | 53 | 28.421 | 16.204 | 16.342 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | whole_count_ce | multi_h | 71 | 79 | 41.471 | 18.565 | 15.706 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 11 | 92 | 28.123 | 13.889 | 10.985 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 23 | 68 | 26.761 | 16.667 | 13.279 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 37 | 98 | 31.614 | 17.004 | 15.260 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 53 | 69 | 26.803 | 15.509 | 13.844 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | binary | 71 | 83 | 25.690 | 13.194 | 12.684 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | 11 | 82 | 18.687 | 17.361 | 14.394 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | 23 | 82 | 22.929 | 15.972 | 15.152 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | 37 | 97 | 20.783 | 13.333 | 11.656 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | 53 | 92 | 23.356 | 15.972 | 13.874 | yes / artifact-only legacy prefix |
| direct_snn | short_mid_long | timestep_ce | multi_h | 71 | 90 | 23.231 | 16.204 | 12.933 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 11 | 94 | 84.637 | 46.113 | 44.163 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 23 | 99 | 85.726 | 46.546 | 48.692 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 37 | 73 | 77.956 | 43.659 | 46.890 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 53 | 70 | 73.255 | 40.427 | 41.864 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | binary | 71 | 88 | 79.885 | 43.685 | 46.996 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | 11 | 84 | 83.615 | 49.787 | 47.593 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | 23 | 100 | 79.964 | 42.393 | 41.714 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | 37 | 92 | 73.999 | 40.272 | 42.839 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | 53 | 93 | 80.516 | 45.547 | 43.302 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | whole_count_ce | multi_h | 71 | 98 | 81.110 | 46.314 | 41.839 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 11 | 100 | 44.015 | 29.444 | 25.708 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 23 | 94 | 47.071 | 25.636 | 31.315 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 37 | 93 | 45.446 | 27.037 | 25.435 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 53 | 84 | 46.379 | 24.147 | 28.996 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | binary | 71 | 97 | 52.465 | 31.369 | 29.277 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | 11 | 100 | 45.408 | 29.845 | 28.596 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | 23 | 77 | 37.943 | 27.037 | 28.130 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | 37 | 96 | 46.140 | 30.106 | 26.776 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | 53 | 99 | 49.761 | 29.997 | 28.708 | yes / artifact-only legacy prefix |
| direct_snn | short_mid | timestep_ce | multi_h | 71 | 100 | 46.641 | 28.591 | 33.874 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 11 | 97 | 94.538 | 43.074 | 43.200 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 23 | 53 | 84.073 | 43.731 | 41.713 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 37 | 71 | 90.111 | 32.663 | 40.121 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 53 | 69 | 86.860 | 39.055 | 35.070 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | binary | 71 | 94 | 93.667 | 38.335 | 40.290 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | 11 | 92 | 87.444 | 40.167 | 43.358 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | 23 | 59 | 80.139 | 41.099 | 41.453 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | 37 | 89 | 83.605 | 38.702 | 40.962 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | 53 | 84 | 84.593 | 40.275 | 40.471 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | whole_count_ce | multi_h | 71 | 66 | 85.709 | 45.603 | 40.637 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 11 | 62 | 49.792 | 30.817 | 34.618 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 23 | 100 | 62.917 | 28.909 | 32.818 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 37 | 100 | 60.245 | 26.257 | 24.992 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 53 | 87 | 54.386 | 26.091 | 26.096 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | binary | 71 | 78 | 59.781 | 31.118 | 33.494 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | 11 | 100 | 49.680 | 32.560 | 25.609 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | 23 | 61 | 25.901 | 28.948 | 14.665 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | 37 | 77 | 37.047 | 28.820 | 23.374 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | 53 | 58 | 35.338 | 24.749 | 24.784 | yes / artifact-only legacy prefix |
| direct_snn | mid_long | timestep_ce | multi_h | 71 | 80 | 51.609 | 32.090 | 30.907 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 11 | 100 | 96.490 | 55.341 | 62.921 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 23 | 73 | 95.143 | 54.601 | 60.274 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 37 | 93 | 98.103 | 50.530 | 60.512 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 53 | 88 | 98.418 | 52.253 | 59.777 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | binary | 71 | 59 | 99.001 | 56.086 | 54.235 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 11 | 88 | 93.724 | 51.171 | 61.013 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 23 | 86 | 92.610 | 49.934 | 61.979 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 37 | 87 | 97.609 | 53.658 | 58.482 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 53 | 98 | 91.412 | 52.698 | 58.974 | yes / artifact-only legacy prefix |
| snn_fixed250_linear | local_234x2 | timestep_ce | multi_h | 71 | 67 | 92.869 | 48.095 | 60.039 | yes / artifact-only legacy prefix |

### experiment_0_1_growing_prefix_relative10

Source: `notebooks/artifacts/experiment_0_1_growing_prefix_relative10/frozen_full_linear_v1/experiment_0_1_results.csv`.

| seed | selected best epoch | train BA | val BA | test BA | strict inclusion |
| --- | --- | --- | --- | --- | --- |
| 11 | — | 100.000 | 52.991 | 81.782 | yes / artifact-only legacy prefix |
| 23 | — | 100.000 | 68.662 | 68.365 | yes / artifact-only legacy prefix |
| 37 | — | 100.000 | 74.995 | 72.862 | yes / artifact-only legacy prefix |
| 53 | — | 100.000 | 77.477 | 60.548 | yes / artifact-only legacy prefix |
| 71 | — | 100.000 | 83.940 | 73.813 | yes / artifact-only legacy prefix |

### 0.2 / 0.2.2 Per-seed Results

For 0.2, selected best epoch is taken from the original `epochs_trained` field; actual history rows provide the true training length. Frozen `none` histories belong to the parent experiment; zero means this directory contains no retraining history, not that the model was untrained.
| architecture | objective | profile | seed | selected best epoch | history rows in this directory | train BA | val BA | test BA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mid_long | timestep_ce | a1 | 11 | 47 | 50 | 22.240 | 16.171 | 13.728 |
| mid_long | timestep_ce | a1 | 23 | 47 | 50 | 34.869 | 21.475 | 27.080 |
| mid_long | timestep_ce | a1 | 37 | 46 | 50 | 12.281 | 11.806 | 10.119 |
| mid_long | timestep_ce | a1_p2 | 11 | 40 | 50 | 20.006 | 17.560 | 13.041 |
| mid_long | timestep_ce | a1_p2 | 23 | 47 | 50 | 35.950 | 22.246 | 24.965 |
| mid_long | timestep_ce | a1_p2 | 37 | 39 | 50 | 11.787 | 10.417 | 11.355 |
| mid_long | timestep_ce | a1_tail | 11 | 50 | 50 | 29.159 | 18.869 | 22.175 |
| mid_long | timestep_ce | a1_tail | 23 | 45 | 50 | 41.495 | 25.007 | 24.564 |
| mid_long | timestep_ce | a1_tail | 37 | 50 | 50 | 26.430 | 16.171 | 19.548 |
| mid_long | timestep_ce | none | 11 | 62 | 0 | 49.792 | 30.817 | 34.618 |
| mid_long | timestep_ce | none | 23 | 100 | 0 | 62.917 | 28.909 | 32.818 |
| mid_long | timestep_ce | none | 37 | 100 | 0 | 60.245 | 26.257 | 24.992 |
| mid_long | timestep_ce | tail | 11 | 42 | 50 | 32.971 | 18.730 | 19.981 |
| mid_long | timestep_ce | tail | 23 | 50 | 50 | 46.993 | 28.000 | 28.470 |
| mid_long | timestep_ce | tail | 37 | 45 | 50 | 31.832 | 20.179 | 17.630 |
| mid_long | whole_count_ce | a1 | 11 | 47 | 50 | 78.118 | 30.289 | 41.613 |
| mid_long | whole_count_ce | a1 | 23 | 49 | 50 | 75.825 | 28.513 | 35.752 |
| mid_long | whole_count_ce | a1 | 37 | 49 | 50 | 71.283 | 26.320 | 29.870 |
| mid_long | whole_count_ce | a1_p2 | 11 | 39 | 50 | 73.150 | 26.690 | 47.240 |
| mid_long | whole_count_ce | a1_p2 | 23 | 50 | 50 | 79.234 | 33.626 | 38.557 |
| mid_long | whole_count_ce | a1_p2 | 37 | 35 | 50 | 62.043 | 22.722 | 26.094 |
| mid_long | whole_count_ce | a1_tail | 11 | 33 | 50 | 69.708 | 28.417 | 41.271 |
| mid_long | whole_count_ce | a1_tail | 23 | 49 | 50 | 81.441 | 36.849 | 41.713 |
| mid_long | whole_count_ce | a1_tail | 37 | 49 | 50 | 74.608 | 27.124 | 33.728 |
| mid_long | whole_count_ce | none | 11 | 97 | 0 | 94.538 | 43.074 | 43.200 |
| mid_long | whole_count_ce | none | 23 | 53 | 0 | 84.073 | 43.731 | 41.713 |
| mid_long | whole_count_ce | none | 37 | 71 | 0 | 90.111 | 32.663 | 40.121 |
| mid_long | whole_count_ce | tail | 11 | 49 | 50 | 81.086 | 30.950 | 41.993 |
| mid_long | whole_count_ce | tail | 23 | 49 | 50 | 81.690 | 36.462 | 36.292 |
| mid_long | whole_count_ce | tail | 37 | 46 | 50 | 77.854 | 31.766 | 39.510 |
| short_mid | timestep_ce | a1 | 11 | 35 | 50 | 12.763 | 8.333 | 9.167 |
| short_mid | timestep_ce | a1 | 23 | 46 | 50 | 14.228 | 9.028 | 9.199 |
| short_mid | timestep_ce | a1 | 37 | 46 | 50 | 25.664 | 17.560 | 17.362 |
| short_mid | timestep_ce | a1_p2 | 11 | 49 | 50 | 15.082 | 8.333 | 10.790 |
| short_mid | timestep_ce | a1_p2 | 23 | 43 | 50 | 13.539 | 9.028 | 10.195 |
| short_mid | timestep_ce | a1_p2 | 37 | 45 | 50 | 24.294 | 17.361 | 15.693 |
| short_mid | timestep_ce | a1_tail | 11 | 46 | 50 | 16.443 | 10.417 | 10.435 |
| short_mid | timestep_ce | a1_tail | 23 | 44 | 50 | 14.992 | 9.722 | 13.171 |
| short_mid | timestep_ce | a1_tail | 37 | 50 | 50 | 27.950 | 18.056 | 17.167 |
| short_mid | timestep_ce | none | 11 | 100 | 0 | 44.015 | 29.444 | 25.708 |
| short_mid | timestep_ce | none | 23 | 94 | 0 | 47.071 | 25.636 | 31.315 |
| short_mid | timestep_ce | none | 37 | 93 | 0 | 45.446 | 27.037 | 25.435 |
| short_mid | timestep_ce | tail | 11 | 49 | 50 | 21.358 | 11.806 | 14.483 |
| short_mid | timestep_ce | tail | 23 | 50 | 50 | 23.830 | 15.278 | 16.896 |
| short_mid | timestep_ce | tail | 37 | 38 | 50 | 23.952 | 16.667 | 15.738 |
| short_mid | whole_count_ce | a1 | 11 | 50 | 50 | 51.314 | 26.085 | 25.546 |
| short_mid | whole_count_ce | a1 | 23 | 48 | 50 | 49.475 | 24.990 | 33.453 |
| short_mid | whole_count_ce | a1 | 37 | 49 | 50 | 54.788 | 33.926 | 39.262 |
| short_mid | whole_count_ce | a1_p2 | 11 | 50 | 50 | 50.925 | 28.922 | 27.956 |
| short_mid | whole_count_ce | a1_p2 | 23 | 50 | 50 | 49.849 | 27.524 | 35.086 |
| short_mid | whole_count_ce | a1_p2 | 37 | 47 | 50 | 53.511 | 31.786 | 31.845 |
| short_mid | whole_count_ce | a1_tail | 11 | 50 | 50 | 53.146 | 25.172 | 29.029 |
| short_mid | whole_count_ce | a1_tail | 23 | 49 | 50 | 47.747 | 25.853 | 32.871 |
| short_mid | whole_count_ce | a1_tail | 37 | 48 | 50 | 54.789 | 31.429 | 29.866 |
| short_mid | whole_count_ce | none | 11 | 94 | 0 | 84.637 | 46.113 | 44.163 |
| short_mid | whole_count_ce | none | 23 | 99 | 0 | 85.726 | 46.546 | 48.692 |
| short_mid | whole_count_ce | none | 37 | 73 | 0 | 77.956 | 43.659 | 46.890 |
| short_mid | whole_count_ce | tail | 11 | 47 | 50 | 51.414 | 27.996 | 32.245 |
| short_mid | whole_count_ce | tail | 23 | 49 | 50 | 56.202 | 32.603 | 39.317 |
| short_mid | whole_count_ce | tail | 37 | 48 | 50 | 56.920 | 33.092 | 38.094 |
| short_mid_long | timestep_ce | a1 | 11 | 1 | 50 | 7.203 | 9.028 | 9.524 |
| short_mid_long | timestep_ce | a1 | 23 | 1 | 50 | 8.258 | 9.722 | 7.251 |
| short_mid_long | timestep_ce | a1 | 37 | 6 | 50 | 9.372 | 9.028 | 8.333 |
| short_mid_long | timestep_ce | a1_p2 | 11 | 1 | 50 | 7.056 | 9.028 | 8.929 |
| short_mid_long | timestep_ce | a1_p2 | 23 | 49 | 50 | 12.908 | 10.417 | 9.957 |
| short_mid_long | timestep_ce | a1_p2 | 37 | 9 | 50 | 9.705 | 10.417 | 8.333 |
| short_mid_long | timestep_ce | a1_tail | 11 | 1 | 50 | 6.876 | 9.028 | 9.686 |
| short_mid_long | timestep_ce | a1_tail | 23 | 36 | 50 | 13.767 | 11.111 | 11.742 |
| short_mid_long | timestep_ce | a1_tail | 37 | 12 | 50 | 10.665 | 9.722 | 8.333 |
| short_mid_long | timestep_ce | none | 11 | 92 | 0 | 28.123 | 13.889 | 10.985 |
| short_mid_long | timestep_ce | none | 23 | 68 | 0 | 26.761 | 16.667 | 13.279 |
| short_mid_long | timestep_ce | none | 37 | 98 | 0 | 31.614 | 17.004 | 15.260 |
| short_mid_long | timestep_ce | tail | 11 | 1 | 50 | 7.216 | 9.028 | 10.281 |
| short_mid_long | timestep_ce | tail | 23 | 50 | 50 | 20.318 | 11.806 | 12.753 |
| short_mid_long | timestep_ce | tail | 37 | 41 | 50 | 15.585 | 11.111 | 12.143 |
| short_mid_long | whole_count_ce | a1 | 11 | 1 | 50 | 7.542 | 9.028 | 8.171 |
| short_mid_long | whole_count_ce | a1 | 23 | 45 | 50 | 20.222 | 11.806 | 10.390 |
| short_mid_long | whole_count_ce | a1 | 37 | 42 | 50 | 21.499 | 11.806 | 9.894 |
| short_mid_long | whole_count_ce | a1_p2 | 11 | 1 | 50 | 7.213 | 9.028 | 8.766 |
| short_mid_long | whole_count_ce | a1_p2 | 23 | 15 | 50 | 16.419 | 13.194 | 10.390 |
| short_mid_long | whole_count_ce | a1_p2 | 37 | 49 | 50 | 22.923 | 10.556 | 9.136 |
| short_mid_long | whole_count_ce | a1_tail | 11 | 29 | 50 | 17.267 | 11.111 | 8.171 |
| short_mid_long | whole_count_ce | a1_tail | 23 | 44 | 50 | 21.958 | 13.889 | 14.719 |
| short_mid_long | whole_count_ce | a1_tail | 37 | 34 | 50 | 26.932 | 14.583 | 11.030 |
| short_mid_long | whole_count_ce | none | 11 | 99 | 0 | 49.378 | 21.005 | 23.357 |
| short_mid_long | whole_count_ce | none | 23 | 100 | 0 | 47.897 | 22.235 | 23.147 |
| short_mid_long | whole_count_ce | none | 37 | 93 | 0 | 47.826 | 18.889 | 22.396 |
| short_mid_long | whole_count_ce | tail | 11 | 48 | 50 | 26.673 | 13.889 | 15.297 |
| short_mid_long | whole_count_ce | tail | 23 | 45 | 50 | 27.824 | 16.171 | 14.232 |
| short_mid_long | whole_count_ce | tail | 37 | 38 | 50 | 29.550 | 16.310 | 15.574 |

#### experiment_0_2_endpoint_tail_regularization

| architecture | objective | profile | seed | test_balanced_accuracy |
| --- | --- | --- | --- | --- |
| mid_long | timestep_ce | a1 | 11 | 13.728 |
| mid_long | timestep_ce | a1 | 23 | 27.080 |
| mid_long | timestep_ce | a1 | 37 | 10.119 |
| mid_long | timestep_ce | a1_p2 | 11 | 13.041 |
| mid_long | timestep_ce | a1_p2 | 23 | 24.965 |
| mid_long | timestep_ce | a1_p2 | 37 | 11.355 |
| mid_long | timestep_ce | a1_tail | 11 | 22.175 |
| mid_long | timestep_ce | a1_tail | 23 | 24.564 |
| mid_long | timestep_ce | a1_tail | 37 | 19.548 |
| mid_long | timestep_ce | none | 11 | 34.618 |
| mid_long | timestep_ce | none | 23 | 32.818 |
| mid_long | timestep_ce | none | 37 | 24.992 |
| mid_long | timestep_ce | tail | 11 | 19.981 |
| mid_long | timestep_ce | tail | 23 | 28.470 |
| mid_long | timestep_ce | tail | 37 | 17.630 |
| mid_long | whole_count_ce | a1 | 11 | 41.613 |
| mid_long | whole_count_ce | a1 | 23 | 35.752 |
| mid_long | whole_count_ce | a1 | 37 | 29.870 |
| mid_long | whole_count_ce | a1_p2 | 11 | 47.240 |
| mid_long | whole_count_ce | a1_p2 | 23 | 38.557 |
| mid_long | whole_count_ce | a1_p2 | 37 | 26.094 |
| mid_long | whole_count_ce | a1_tail | 11 | 41.271 |
| mid_long | whole_count_ce | a1_tail | 23 | 41.713 |
| mid_long | whole_count_ce | a1_tail | 37 | 33.728 |
| mid_long | whole_count_ce | none | 11 | 43.200 |
| mid_long | whole_count_ce | none | 23 | 41.713 |
| mid_long | whole_count_ce | none | 37 | 40.121 |
| mid_long | whole_count_ce | tail | 11 | 41.993 |
| mid_long | whole_count_ce | tail | 23 | 36.292 |
| mid_long | whole_count_ce | tail | 37 | 39.510 |
| short_mid | timestep_ce | a1 | 11 | 9.167 |
| short_mid | timestep_ce | a1 | 23 | 9.199 |
| short_mid | timestep_ce | a1 | 37 | 17.362 |
| short_mid | timestep_ce | a1_p2 | 11 | 10.790 |
| short_mid | timestep_ce | a1_p2 | 23 | 10.195 |
| short_mid | timestep_ce | a1_p2 | 37 | 15.693 |
| short_mid | timestep_ce | a1_tail | 11 | 10.435 |
| short_mid | timestep_ce | a1_tail | 23 | 13.171 |
| short_mid | timestep_ce | a1_tail | 37 | 17.167 |
| short_mid | timestep_ce | none | 11 | 25.708 |
| short_mid | timestep_ce | none | 23 | 31.315 |
| short_mid | timestep_ce | none | 37 | 25.435 |
| short_mid | timestep_ce | tail | 11 | 14.483 |
| short_mid | timestep_ce | tail | 23 | 16.896 |
| short_mid | timestep_ce | tail | 37 | 15.738 |
| short_mid | whole_count_ce | a1 | 11 | 25.546 |
| short_mid | whole_count_ce | a1 | 23 | 33.453 |
| short_mid | whole_count_ce | a1 | 37 | 39.262 |
| short_mid | whole_count_ce | a1_p2 | 11 | 27.956 |
| short_mid | whole_count_ce | a1_p2 | 23 | 35.086 |
| short_mid | whole_count_ce | a1_p2 | 37 | 31.845 |
| short_mid | whole_count_ce | a1_tail | 11 | 29.029 |
| short_mid | whole_count_ce | a1_tail | 23 | 32.871 |
| short_mid | whole_count_ce | a1_tail | 37 | 29.866 |
| short_mid | whole_count_ce | none | 11 | 44.163 |
| short_mid | whole_count_ce | none | 23 | 48.692 |
| short_mid | whole_count_ce | none | 37 | 46.890 |
| short_mid | whole_count_ce | tail | 11 | 32.245 |
| short_mid | whole_count_ce | tail | 23 | 39.317 |
| short_mid | whole_count_ce | tail | 37 | 38.094 |
| short_mid_long | timestep_ce | a1 | 11 | 9.524 |
| short_mid_long | timestep_ce | a1 | 23 | 7.251 |
| short_mid_long | timestep_ce | a1 | 37 | 8.333 |
| short_mid_long | timestep_ce | a1_p2 | 11 | 8.929 |
| short_mid_long | timestep_ce | a1_p2 | 23 | 9.957 |
| short_mid_long | timestep_ce | a1_p2 | 37 | 8.333 |
| short_mid_long | timestep_ce | a1_tail | 11 | 9.686 |
| short_mid_long | timestep_ce | a1_tail | 23 | 11.742 |
| short_mid_long | timestep_ce | a1_tail | 37 | 8.333 |
| short_mid_long | timestep_ce | none | 11 | 10.985 |
| short_mid_long | timestep_ce | none | 23 | 13.279 |
| short_mid_long | timestep_ce | none | 37 | 15.260 |
| short_mid_long | timestep_ce | tail | 11 | 10.281 |
| short_mid_long | timestep_ce | tail | 23 | 12.753 |
| short_mid_long | timestep_ce | tail | 37 | 12.143 |
| short_mid_long | whole_count_ce | a1 | 11 | 8.171 |
| short_mid_long | whole_count_ce | a1 | 23 | 10.390 |
| short_mid_long | whole_count_ce | a1 | 37 | 9.894 |
| short_mid_long | whole_count_ce | a1_p2 | 11 | 8.766 |
| short_mid_long | whole_count_ce | a1_p2 | 23 | 10.390 |
| short_mid_long | whole_count_ce | a1_p2 | 37 | 9.136 |
| short_mid_long | whole_count_ce | a1_tail | 11 | 8.171 |
| short_mid_long | whole_count_ce | a1_tail | 23 | 14.719 |
| short_mid_long | whole_count_ce | a1_tail | 37 | 11.030 |
| short_mid_long | whole_count_ce | none | 11 | 23.357 |
| short_mid_long | whole_count_ce | none | 23 | 23.147 |
| short_mid_long | whole_count_ce | none | 37 | 22.396 |
| short_mid_long | whole_count_ce | tail | 11 | 15.297 |
| short_mid_long | whole_count_ce | tail | 23 | 14.232 |
| short_mid_long | whole_count_ce | tail | 37 | 15.574 |

#### experiment_0_2_2_capacity_preserving_loss

| architecture | condition | seed | best_epoch | best_val_balanced_accuracy | test_balanced_accuracy | test_long_valid_fr_hz | test_long_tail_fr_hz | test_long_tail_valid_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mid_long | wc_only | 11 | 48 | 37.531 | 34.825 | 14.0166 | 23.0184 | 1.6422 |
| mid_long | wc_only | 23 | 45 | 37.253 | 37.040 | 16.9382 | 25.9707 | 1.5333 |
| mid_long | wc_only | 37 | 50 | 38.241 | 44.721 | 17.0106 | 28.9697 | 1.7030 |
| mid_long | sat | 11 | 48 | 37.531 | 34.825 | 14.0166 | 23.0184 | 1.6422 |
| mid_long | sat | 23 | 45 | 37.253 | 37.040 | 16.9382 | 25.9707 | 1.5333 |
| mid_long | sat | 37 | 50 | 38.241 | 44.721 | 17.0106 | 28.9697 | 1.7030 |
| mid_long | relative_tail | 11 | 42 | 33.360 | 33.274 | 13.7013 | 23.0274 | 1.6807 |
| mid_long | relative_tail | 23 | 50 | 36.483 | 34.017 | 9.2756 | 15.5438 | 1.6758 |
| mid_long | relative_tail | 37 | 37 | 36.525 | 38.763 | 16.5380 | 30.0571 | 1.8175 |
| mid_long | sat_relative_tail | 11 | 43 | 30.242 | 32.322 | 12.6566 | 20.5343 | 1.6224 |
| mid_long | sat_relative_tail | 23 | 37 | 31.608 | 29.298 | 7.1167 | 12.2262 | 1.7180 |
| mid_long | sat_relative_tail | 37 | 49 | 37.792 | 46.263 | 16.3565 | 29.1875 | 1.7845 |
| mid_long | sat_relative_tail_capacity | 11 | 48 | 32.325 | 34.180 | 12.8924 | 20.3132 | 1.5756 |
| mid_long | sat_relative_tail_capacity | 23 | 40 | 30.506 | 30.734 | 4.5163 | 7.8811 | 1.7450 |
| mid_long | sat_relative_tail_capacity | 37 | 37 | 35.275 | 42.199 | 16.2301 | 29.4218 | 1.8128 |
| short_mid_long | wc_only | 11 | 45 | 12.698 | 12.608 | 4.5709 | 9.2563 | 2.0250 |
| short_mid_long | wc_only | 23 | 17 | 17.897 | 19.021 | 6.1337 | 12.0910 | 1.9712 |
| short_mid_long | wc_only | 37 | 45 | 17.361 | 18.065 | 5.8985 | 11.0391 | 1.8715 |
| short_mid_long | sat | 11 | 45 | 15.509 | 13.636 | 4.2610 | 9.6298 | 2.2600 |
| short_mid_long | sat | 23 | 17 | 17.698 | 17.091 | 6.2076 | 11.8892 | 1.9153 |
| short_mid_long | sat | 37 | 30 | 15.417 | 11.851 | 4.3842 | 8.8665 | 2.0224 |
| short_mid_long | relative_tail | 11 | 35 | 15.417 | 12.608 | 3.0496 | 7.0288 | 2.3048 |
| short_mid_long | relative_tail | 23 | 8 | 15.972 | 10.173 | 2.3866 | 5.4470 | 2.2824 |
| short_mid_long | relative_tail | 37 | 43 | 14.921 | 17.262 | 5.2719 | 9.8943 | 1.8768 |
| short_mid_long | sat_relative_tail | 11 | 23 | 14.583 | 11.255 | 3.7286 | 8.6611 | 2.3229 |
| short_mid_long | sat_relative_tail | 23 | 50 | 17.560 | 17.335 | 6.7437 | 12.3761 | 1.8352 |
| short_mid_long | sat_relative_tail | 37 | 39 | 12.639 | 10.985 | 1.1410 | 2.4714 | 2.1660 |
| short_mid_long | sat_relative_tail_capacity | 11 | 26 | 13.194 | 8.874 | 2.9942 | 6.6119 | 2.2082 |
| short_mid_long | sat_relative_tail_capacity | 23 | 11 | 17.758 | 15.206 | 5.3014 | 10.4752 | 1.9759 |
| short_mid_long | sat_relative_tail_capacity | 37 | 42 | 15.556 | 15.955 | 4.4682 | 8.5466 | 1.9128 |

### 0.1.5 Scheduler-conflict Sensitivity

| group | n | mean±sample SD BA (%) |
| --- | --- | --- |
| short_mid / TSCE / 20%：original artifact | 5 | 9.21±1.97 |
| same group: excluded FAILED task 77 | 4 | 8.33±0.00 |

This is not a BA-based selective exclusion: the excluded rule comes from scheduler status. The five-seed artifact result is retained in the table and CSV; the strict four-seed statistic is explicitly distinct from the original five-seed statistic.


