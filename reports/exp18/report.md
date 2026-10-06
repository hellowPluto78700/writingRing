# Exp18 — History Carrier, Count-Loss Geometry, and Native Evidence Accessibility

Evidence cutoff: **2026-10-06**. Inspected repository and Unity HEAD: `0480ef082898dbbc162bf83f2c7e9c8e5b835773`; repository `hellowPluto78700/writingRing`; Unity checkout `/home/zhaolongwei_umass_edu/projects/writingRing`. This standalone report covers canonical Exp18, Exp18.1, Exp18.2 and Exp18.3. BA values are percent, contrasts pp, and mean ± sample SD describes seeds **11/23/37, n=3**, on one locked cross-user split. Reused references are counted once. Numerical completeness and causal validity are evaluated separately.

**Material pairing correction.** The reused I baseline is CoreBenchmark **v1.0**, while new models use **v1.1 initialization streams**. Direct inspection of all three `initial.pt` files shows different L1, L2 and head weights between I_REF and U_NORMAL for every seed. Fresh U_NORMAL/U_DETACH, UI/IU, and I/U-MWCCE initial common weights match within each seed. Accordingly, I_REF-versus-new-model and factorial contrasts touching II_REF are descriptive cross-version comparisons, not strictly paired carrier-only interventions. The READMEs' four-corner initialization claim is stronger than the executed evidence supports. This report preserves the results and corrects their causal interpretation. [E18-INIT]

## 1. Executive Summary

Exp18 asks why SNN optimization recruits persistent/high-rate communication, separating the **state carrying history** from the **objective and head that turn spikes into class evidence**. It first moves the slow pole from unreset synaptic current I to resettable membrane U; then localizes the carrier by layer, changes count-loss geometry, and analyzes rate/selectivity/head alignment plus zero-versus-train-mean feature replacement.

The executed baseline/new-U comparison is **57.93 ± 2.23%** native test BA for reused I_REF, **51.05 ± 0.38%** for U_NORMAL, and **47.74 ± 2.42%** for U_DETACH. U reset detachment fails to improve classification despite larger retained suffix gradients in some bins. UI and IU achieve **53.57 ± 1.67%** and **56.31 ± 3.92%**. The clean new-model contrast IU−UU is **+5.27 ± 4.30 pp**, consistent with a benefit from restoring I in the first layer when the second remains U; a general layer ranking remains uncertain. [E18; E18.1]

U-based WCCE remains dense: pooled-time L2 mean rates are **26.70 ± 0.69 Hz**, versus **23.94 ± 0.52 Hz** for I_REF. Margin-WCCE reduces them to **5.52 ± 1.51 Hz** for U and **5.09 ± 4.29 Hz** for I, but native BA falls to **38.82 ± 1.98%** and **32.50 ± 17.37%**. The I average includes a completed but near-collapsed seed11; its result is retained as a training failure mode. NWCCE produces near-silent communication, with an additional U scale-calibration confound. [E18.2]

Exp18.3 shows that rate is not a direct measure of class selectivity or native utility under WCCE. Top-rate mean replacement drops test BA **3.97 pp (I)** and **1.64 pp (U)** on average; zero replacement drops **17.94** and **26.92 pp**. This supports sensitivity of the fixed head to a nonzero feature operating point, plus class-dependent modulation. It is a feature-level/readout result, not proof of an internal dynamical requirement. U-MWCCE Relative10 reaches **63.13 ± 8.57%**, well above its native **38.82%**, yet the richer decoder changes both temporal structure and capacity. [E18.3]

The family rules out slow I as a necessary cause of high-rate communication and demonstrates that objective geometry can change firing strongly in U models with matched initialization. It does not prove that lowering rate improves cross-user transfer or that the accumulator causes the full gap. The remaining problem is how to organize transferable evidence without discarding useful context or destabilizing the native operating point.

## 2. Context and Motivation

Exp17 found a broad high-rate population, changing rate identities, distributed class information, and low-group auxiliary selectivity without native or fixed-group transfer gain. Its Exp17.2 input-influence criterion did not support simple history drowning. Those findings motivate a narrower comparison: is persistent firing primarily an outcome of a particular memory carrier, or of optimization over repeated class evidence?

Project discussion at **2026-10-03 02:21:19 UTC** required the CoreBenchmark data/split/seeds and bias-free accumulator reference. The first valid-result discussion at **03:52:26 UTC** rejected the hoped-for U improvement; **04:03:55 UTC** defined the two missing UI/IU corners. Discussion at **07:16:03 UTC** shifted toward removing radial count incentives and saturating margins. The next available canonical number was Exp18.2, because Exp18.1 already denoted layer localization. Later proposals for TSCE/hybrid, signed output, adaptive thresholds, and contextual/evidence routing are not silently inserted into this executed family.

The architecture-specific correction is retained throughout: with I-based layers, **slow I stores history and spikes communicate it**; fast membrane decay does not mean history is absent, and repeated spikes are not themselves the storage state.

## 3. Research Questions

1. Does placing the same long pole in resettable U improve native/count organization or remove high-rate communication?
2. Does reset-branch surrogate differentiation impair useful temporal credit, and does detachment repair it?
3. Which layer's carrier most affects native and temporal-probe outcomes?
4. Can normalization or margin saturation remove firing pressure while retaining discrimination?
5. Does high rate imply class information, head alignment, or cross-user profile stability?
6. How much zero-ablation damage comes from removing the mean operating contribution versus sample-varying information?
7. What part of the train–test discrepancy reflects available representation versus native accessibility?

## 4. Competing Hypotheses

| Hypothesis | Control | Outcome |
| --- | --- | --- |
| Unreset slow I is the necessary cause of dense communication | U_NORMAL under the same count objective | Falsified as a necessity claim: U remains high-rate |
| Consuming membrane history improves native evidence | I_REF vs U variants | No observed improvement; initialization confound limits carrier-only attribution |
| Reset backward attenuation is the main U bottleneck | U_DETACH vs U_NORMAL with matched initial weights | Detachment does not improve native BA |
| Early-layer history loss matters most | II/UI/IU/UU and clean IU−UU comparison | Compatible with early-layer importance; not universal localization |
| CE count geometry sustains excessive rate | WCCE vs NWCCE/MWCCE | Rate changes sharply; native performance deteriorates |
| High-rate neurons intrinsically have the strongest class information | Rate–η² / alignment / utility diagnostics | WCCE correlations are weak or negative |
| Zero-ablation damage measures unique class information | Mean replacement and matched random controls | Zero is substantially more damaging; mean operating point matters to frozen head |
| All transferable information is absent from weak-native models | WholeCount/Fixed250/Relative10 | Temporal probes retain substantial information, with changed decoder capacity |

## 5. Experimental Design

### 5.1 Canonical experiments and status

| Alias | Canonical ID / protocol | New work / references |
| --- | --- | --- |
| Exp18 (18.0) | `experiment_18_membrane_history` / `membrane_history_v1` | Six new U_NORMAL/U_DETACH runs; three I_REF gradient diagnostics; Core O0 reused |
| Exp18.1 | `experiment_18_1_layerwise_memory_carrier` / `layerwise_memory_carrier_v1` | Six UI/IU runs; II_REF=Core O0; UU_REF=U_NORMAL |
| Exp18.2 | `experiment_18_2_loss_geometry` / `loss_geometry_v1` | 12 new I/U×NWCCE/MWCCE runs; WCCE references reused |
| Exp18.3 | `experiment_18_3_discriminative_coordinate` / `discriminative_coordinate_v1` | 12 artifact-only diagnostics of I/U×WCCE/MWCCE; no model/head retraining |
| Earlier “Exp18 context/evidence decoupling” plan | Edge interventions, head refit, late rerouting and soft routing | Proposed scientific predecessor; not canonical executed carrier cases |
| WCCE/TSCE/hybrid carrier comparison | Later conversational plan | No matching executed formal cases in the four inspected canonical modules |

All four final aggregates report PASS. Every one of the **36 new-run/diagnostic complete.json certificates** (6+6+12+12) was checked against its declared file hashes on Unity, with zero mismatches. This establishes retained-file integrity; it does not repair initialization pairing or prove hypothetical controls were executed.

### 5.2 Dataset, architecture and selection

Both action0/action1 64 Hz aligned-board wavelet-event inputs are used. The source has 36 channels; only the 30 polarity-split event channels enter the network. Selected labels: **A B C D E G H I J K L X**. Valid lengths are preserved in a padded 256-step horizon; padding is masked in hidden evolution, loss and probes.

| Split | Fixed users | Samples |
| --- | --- | --- |
| Train | user_0, user_1, user_11, user_12, user_13, user_14, user_15, user_18, user_19, user_2, user_20, user_5, user_7, user_8 | 581 |
| Val | user_16, user_4, user_9 | 126 |
| Test | user_10, user_3, user_6 | 146 |

Split seed=12345; model seeds=11/23/37; dataset SHA256 `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a`; Core identity `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`. BA pools samples across the three test users before computing class recalls. Training BA is in-sample and does not measure held-out performance on training users.

Hidden widths=128/128; shifts=(2,3,4)/(2,3,4), grouped 43/43/42. Threshold=.5; slope=25; binary cap=1 spike/timestep; all layer/head weights are bias-free. Δt=15.625 ms; slow α=.75/.875/.9375 gives **54.313/117.014/242.103 ms**. Fast β=exp(−Δt/22.54 ms)≈.49997. No new time constant is introduced by swapping poles. “249 ms” in conversation is approximate and is not the exact shift4 pole.

Adam lr=.001, weight_decay=0, batch=128, max=100 epochs, min=20, patience=30. Epoch0 is eligible. Selection uses **native validation BA→native valid-mean-logit CE→earliest epoch**. No test or probe value selects a checkpoint. Exp18 uses a shuffled Core loader held across epochs; newer hybrid/loss modules follow the same loader contract. This is distinct from the deterministic epoch sampler used in Exp17/Exp16.3; equal seed labels do not make their checkpoints identical.

### 5.3 Exact carrier equations

For I carrier:

\[
I_t=\alpha_{slow}I_{t-1}+Wa_t,
\quad P_t=\beta_{fast}U_{t-1}+I_t,
\quad s_t=H(P_t-\theta),\quad U_t=P_t-\theta s_t.
\]

For U carrier:

\[
I_t=\beta_{fast}I_{t-1}+Wa_t,
\quad P_t=\alpha_{slow}U_{t-1}+I_t,
\quad s_t=H(P_t-\theta),\quad U_t=P_t-\theta s_t.
\]

These are cascaded slow/fast filters with **both states retained**. The U implementation is not a pure single-state instantaneous-current integrate-and-fire recurrence. Before reset, their linear transfer functions agree because the two poles commute. After threshold/reset, the same subtraction acts on different long-lived state locations; equivalence no longer follows.

U_NORMAL differentiates the surrogate spike in reset. U_DETACH uses U_t=P_t−θ stopgrad(s_t), with identical forward subtraction and a differentiable communication/evidence spike branch. It does not detach spikes from all training. Exp18.1 uses normal reset in every U layer; Exp18.2 does not add reset detachment.

### 5.4 Exact objectives and native geometry

For valid L2 counts C_i=Σ_t s_it and occupancy q_i=C_i/T_i:

\[
z_i=Rq_i=\frac1{T_i}\sum_t Rs_{i,t},
\quad L_{WC}=\frac1N\sum_i CE(z_i,y_i).
\]

**NWCCE** uses differentiable count-L1 normalization, not detached denominators:

\[
\widehat C_i=\begin{cases}C_i/\lVert C_i\rVert_1&\lVert C_i\rVert_1>0\\0&\text{otherwise},\end{cases}
\quad L_{NWC}=\frac1N\sum_i CE(\gamma R\widehat C_i,y_i).
\]

γ is fixed per carrier/seed by label-free epoch0 train logit-RMS matching. For nonzero C, normalized logits and native mean logits differ by a **positive sample-wise scalar**, preserving argmax but changing CE scale/gradients. Native checkpoint CE still uses the unnormalized valid-mean geometry.

**MWCCE** uses one strongest competitor, not a sum over all competitors:

\[
m_i=z_{i,y_i}-\max_{k\ne y_i}z_{i,k},
\qquad L_{MWC}=\frac1N\sum_i\max(0,1-m_i).
\]

The fixed target is 1.0; no margin sweep occurred. NWCCE/MWCCE change the loss, not add a firing penalty. Their different weight norms are diagnostic, not controlled weight-normalization treatments.

### 5.5 Probes, diagnostics and aggregation

Canonical probes inspect L1/L2 spikes and pre-reset states with no-bias or affine decoders. Primary tables here use **spike/no-bias**. WholeCount sums valid activity; Fixed250 uses 16-step non-overlapping absolute-time bins within a 256-step horizon; Relative10 splits each valid sequence into ten equal-phase bins. Their flattened dimensions are **128, 2048, 1280** at one layer. Shuffled variants permute bins within sample with five fixed seeds; repetitions are averaged within model seed. Train-only scale fitting uses `StandardScaler(with_mean=False)`; C grid=.001/.01/.1/1/10/100; convergence required; validation-only BA chooses C, tie retains smaller C.

Exp18 gradient diagnostics use validation only, ordinary WCCE and diagnostic suffix CE over the last 16 valid steps. Rows are stratified by layer, τ and future-reset bucket. They compare selected checkpoints that may differ in activation and gradient scale; larger derivatives do not isolate a longer causal memory horizon.

Exp18.2 rate diagnostics use **pooled-time** counts / total valid duration. Exp18.3 computes **sample-mean** occupancy×64 Hz because occupancy is the native feature. The rate means are therefore not interchangeable. Tail thresholds in Exp18.2 are ≥10/≥20 Hz; conversational >10 Hz is a separate operational convention.

Exp18.3 ranks train-only rate, class η², aligned signal, or margin; selects ceil(.30×128)=39 neurons, plus low-rate39 and 20 matched random groups. It applies mean replacement or zero to **occupancy features at the frozen native head**. No internal states or model parameters are changed. Per-neuron class-profile transfer uses test labels only after training/ranking; it is a diagnostic, not a deployable feature selector.

### 5.6 Provenance correction and admissible contrasts

`core_benchmark_v1/results/main/protocol.lock.json` records **core_benchmark_v1.0**, source `96088fe7725f6d0748aca7027abfe8fc438bad2a`. `exp16._load_core` copies the lock but overrides `version` to current `Protocol().version` (v1.1). `paired_seed` hashes the version string. Direct tensor checks confirm all nine I_REF/U_NORMAL common-weight comparisons (3 tensors×3 seeds) differ; maximum absolute differences are approximately .357–.358 for L1, .175–.176 for L2, and .171–.175 for the head.

Fresh U_DETACH, UI, IU, U_MWCCE and I_MWCCE common initial tensors each equal U_NORMAL's same-seed tensors. Thus U_DETACH−U_NORMAL and U-MWCCE−U-WCCE avoid this initialization confound. IU−UU and UI−UU are also within the fresh initialization family. II_REF-based four-corner contrasts, I-loss changes versus reused I-WCCE, and I-versus-U WCCE are **cross-version descriptive**. No matched newly trained v1.1 II-WCCE control was executed. Artifact PASS denotes integrity, not equivalence to such an absent control.

## 6. Implementation and Execution History

| Stage | Repository history / jobs | Recovered outcome |
| --- | --- | --- |
| Exp18 implementation | PR #103; `4005d25`; paired loader/probe fix `6f7b046` | Implemented |
| First formal U array | 65177120 prepare; 65177121_0–5 | All six FAILED 1:0; retained stdout shows `NameError: train_loader is not defined`; these are not results |
| Loader fix / rerun | `4da0164`; 65177298_0–5; 65177299 final; I diagnostics 65177122_0–2 | All rerun/ref rows COMPLETED 0:0; six current completion hashes valid |
| Environment/smoke hardening | `df0de00`, `09f983f`, `7b14c10`; smoke 65178164 failed; `b1961d7` fix | Failed smoke stdout shows missing `state_hash`; 65178144/65178222 pass afterward |
| Exp18.1 | PR #104; `8c8d279`, `35f441c`, test fix `c585cfc`; smoke 65178723; prepare 65178724; array 65178725_0–5; final 65178726 | Six new runs completed |
| Exp18.1 second launcher | 65179988 smoke; 65179989 prepare; 65179990_0–5; 65179991 final | Completed; inspected task0 says `already_complete`; not six new models |
| Exp18.2 | PR #105; `e5c8107`; smoke 65181327; prepare 65181328; array 65181329_0–11; final 65181330 | Twelve completed formal runs; numerical collapse retained as outcome |
| Exp18.3 | PR #106; `80675b9`; smoke 65185368; prepare 65185369; diagnostics 65185370_0–11; final 65185371 | Twelve completed artifact-only diagnostics; PASS aggregate |

The original valid U runs finished before the later smoke failures/passes. Therefore current smoke-gated READMEs must not be used to claim those already-executed runs were originally smoke-gated. Environment bootstrapping was hardened against nounset-sensitive `LMOD_DO_PURGE`; Git/README document that failure mode. The retained first formal array's immediate cause is instead the missing loader, and the retained smoke failure is the missing import. Neither is reassigned to Lmod without log evidence.

Scheduler accounting was inspected from October1–6 (UTC timestamps). `scontrol` no longer retained the first failed job; its stdout was recovered at `slurm-65177121_0.out`, and smoke stdout at `slurm-65178164.out`. No invalid pre-fix numerical result is averaged with final results. These reports do not retrain, rewrite locks, or modify artifacts to repair retrospective comparability.

## 7. Main Results

### 7.1 Original carrier/reset comparison

| Case | Train Ba (%) | Val Ba (%) | Test Ba (%) |
| --- | --- | --- | --- |
| I_REF | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 |
| U_NORMAL | 64.44 ± 2.00 | 48.73 ± 4.93 | 51.05 ± 0.38 |
| U_DETACH | 60.35 ± 5.24 | 48.47 ± 4.88 | 47.74 ± 2.42 |

BA %, mean ± sample SD. U_NORMAL−I_REF is **−6.88 pp** descriptively; clean U_DETACH−U_NORMAL is **−3.31 pp**, negative in every seed. U_DETACH native test is 48.67/49.55/44.99% across seeds; U_NORMAL is 50.69/51.00/51.45%. Detachment does not improve discrimination. U has lower train BA too, so its smaller train–test gap is not a generalization success.

### 7.2 Layer-wise carrier factorial

| Case | Train Ba (%) | Val Ba (%) | Test Ba (%) |
| --- | --- | --- | --- |
| II_REF | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 |
| UI | 86.36 ± 2.92 | 53.89 ± 2.90 | 53.57 ± 1.67 |
| IU | 81.27 ± 1.47 | 52.99 ± 3.04 | 56.31 ± 3.92 |
| UU_REF | 64.44 ± 2.00 | 48.73 ± 4.93 | 51.05 ± 0.38 |

II_REF and UU_REF reuse the original references. UI means L1=U/L2=I; IU means L1=I/L2=U. Seed values:

| case | seed | train_ba | val_ba | test_ba |
| --- | --- | --- | --- | --- |
| II_REF | 11 | 93.078 | 55.047 | 55.975 |
| UI | 11 | 89.54 | 57.181 | 54.185 |
| IU | 11 | 81.009 | 49.68 | 59.731 |
| UU_REF | 11 | 66.009 | 48.755 | 50.689 |
| II_REF | 23 | 93.001 | 52.815 | 57.459 |
| UI | 23 | 83.798 | 51.726 | 54.849 |
| IU | 23 | 82.848 | 55.672 | 57.171 |
| UU_REF | 23 | 65.113 | 53.653 | 51.0 |
| II_REF | 37 | 92.838 | 53.249 | 60.356 |
| UI | 37 | 85.733 | 52.756 | 51.687 |
| IU | 37 | 79.952 | 53.615 | 52.035 |
| UU_REF | 37 | 62.194 | 43.785 | 51.449 |

The clean **IU−UU** native contrasts are **+9.042/+6.170/+0.586 pp** (mean **+5.27 ± 4.30**); clean **UI−UU** contrasts are **+3.495/+3.849/+0.238 pp** (mean **+2.53 ± 1.99**). Restoring I in either layer helps relative to fresh UU in these seeds, with a larger mean for restoring L1. UI versus IU changes two layers simultaneously; these results do not prove early-layer importance in every background.

For completeness, the reported factorial native means UI−II=−4.36, UU−IU=−5.27, IU−II=−1.62, UU−UI=−2.53, interaction=−.91 pp. Contrasts involving II carry the version confound. The native interaction by seed is **−7.25/−3.56/+8.08 pp**, so a universal non-additive interaction is not supported.

### 7.3 Loss geometry: native BA and pooled-time firing

| Case | Train Ba (%) | Val Ba (%) | Test Ba (%) | L2 pooled Hz | L2 P95 Hz | L2 ≥10 Hz (%) | L2 ≥20 Hz (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_WCCE_REF | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 | 23.94 ± 0.52 | 33.93 ± 1.46 | 98.96 ± 0.45 | 74.22 ± 3.58 |
| I_NWCCE | 33.79 ± 3.65 | 29.48 ± 4.99 | 24.14 ± 2.11 | 0.11 ± 0.03 | 0.08 ± 0.10 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| I_MWCCE | 56.27 ± 31.85 | 33.42 ± 12.45 | 32.50 ± 17.37 | 5.09 ± 4.29 | 9.85 ± 7.94 | 16.93 ± 14.71 | 0.52 ± 0.90 |
| U_WCCE_REF | 64.44 ± 2.00 | 48.73 ± 4.93 | 51.05 ± 0.38 | 26.70 ± 0.69 | 38.19 ± 0.48 | 100.00 ± 0.00 | 84.90 ± 4.30 |
| U_NWCCE | 12.93 ± 1.06 | 14.63 ± 0.56 | 11.11 ± 2.18 | 0.01 ± 0.01 | 0.04 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| U_MWCCE | 59.60 ± 3.06 | 44.32 ± 3.44 | 38.82 ± 1.98 | 5.52 ± 1.51 | 14.02 ± 2.34 | 17.45 ± 6.36 | 0.78 ± 1.35 |

Rates are Hz; ≥10/≥20 columns are percent of L2 neurons. NWCCE communication nearly collapses. I-MWCCE's large SD reflects a retained seed-specific failure, not an uncertainty band around a stable 5 Hz solution. U-MWCCE remains a lower-rate yet poorly native-decoded regime.

Full seed endpoints (BA %, rate Hz):

| case | seed | train_ba | val_ba | test_ba | l2_mean_firing_hz | l2_relative10_ordered_test_ba |
| --- | --- | --- | --- | --- | --- | --- |
| I_WCCE_REF | 11 | 93.078 | 55.047 | 55.975 | 24.186 | 67.367 |
| I_NWCCE | 11 | 29.731 | 28.476 | 25.474 | 0.135 | 35.885 |
| I_MWCCE | 11 | 19.616 | 19.226 | 12.892 | 0.133 | 14.764 |
| U_WCCE_REF | 11 | 66.009 | 48.755 | 50.689 | 27.442 | 66.526 |
| U_NWCCE | 11 | 12.234 | 15.278 | 9.957 | 0.008 | 10.24 |
| U_MWCCE | 11 | 60.877 | 48.172 | 40.503 | 7.033 | 69.636 |
| I_WCCE_REF | 23 | 93.001 | 52.815 | 57.459 | 24.293 | 66.196 |
| I_NWCCE | 23 | 36.792 | 34.901 | 21.71 | 0.082 | 26.763 |
| I_MWCCE | 23 | 72.055 | 38.532 | 45.953 | 7.727 | 45.219 |
| U_WCCE_REF | 23 | 65.113 | 53.653 | 51.0 | 26.599 | 68.124 |
| U_NWCCE | 23 | 14.152 | 14.286 | 9.74 | 0.01 | 22.091 |
| U_MWCCE | 23 | 61.804 | 41.535 | 39.325 | 5.512 | 66.334 |
| I_WCCE_REF | 37 | 92.838 | 53.249 | 60.356 | 23.338 | 67.542 |
| I_NWCCE | 37 | 34.839 | 25.067 | 25.236 | 0.112 | 27.344 |
| I_MWCCE | 37 | 77.145 | 42.51 | 38.667 | 7.399 | 48.852 |
| U_WCCE_REF | 37 | 62.194 | 43.785 | 51.449 | 26.069 | 64.54 |
| U_NWCCE | 37 | 12.408 | 14.319 | 13.626 | 0.018 | 20.951 |
| U_MWCCE | 37 | 56.107 | 43.264 | 36.647 | 4.017 | 53.417 |

I-MWCCE seed11 reaches **12.89% native BA / .133 pooled Hz**, while seeds23/37 reach **45.95%/7.727 Hz** and **38.67%/7.399 Hz**. It completed correctly and is a valid negative optimization outcome, not an artifact error. Excluding it post hoc would bias the comparison; any two-seed description must be explicitly restricted rather than replace the formal three-seed result.

## 8. Mechanistic Diagnostics

### 8.1 U-based history does not eliminate high-rate communication

Under WCCE, all U L2 neurons have pooled train rate ≥10 Hz; 84.90% average are ≥20 Hz. I_REF has 98.96% ≥10 and 74.22% ≥20. U rates remain high despite subtractive reset acting on the slow state. This falsifies **slow unreset I is necessary for high rate**. It does not isolate whether the cause is head geometry, input drive, learned gain, binary cap, or their interaction.

The one-spike cap can leave positive pre-reset charge after subtracting θ; persistent incoming drive can replenish it. Carrier reset therefore cannot by itself be interpreted as a communication-rate budget. This is a structural explanation consistent with the observations, not a separately executed cap/backlog causal ablation in Exp18.

### 8.2 Temporal information survives weak native results

Exp18.1 L2 spike/no-bias probes:

| Case | WholeCount (%) | Fixed250 order (%) | Fixed250 shuffle (%) | Relative10 order (%) | Relative10 shuffle (%) |
| --- | --- | --- | --- | --- | --- |
| II_REF | 57.57 ± 4.24 | 58.62 ± 2.69 | 53.79 ± 3.04 | 67.04 ± 0.73 | 54.24 ± 3.66 |
| UI | 53.60 ± 3.58 | 56.20 ± 1.16 | 48.34 ± 0.93 | 67.54 ± 3.57 | 48.11 ± 0.98 |
| IU | 56.54 ± 4.54 | 58.52 ± 1.25 | 49.36 ± 3.37 | 63.38 ± 4.84 | 48.58 ± 4.12 |
| UU_REF | 53.32 ± 0.97 | 58.59 ± 2.64 | 45.88 ± 2.36 | 66.40 ± 1.80 | 41.49 ± 0.49 |

For I_REF, Fixed250 order−shuffle is **4.83 pp**; for UU it is **12.71 pp**. Relative10 order−shuffle grows **12.79→24.91 pp**. Ordered Fixed250 remains nearly unchanged descriptively (**58.62→58.59%**), while shuffled performance decreases. U representations retain useful temporal structure yet are less accessible to an order-insensitive count decoder. This is compatible with altered temporal organization; initialization and independently fitted probe geometry limit pure carrier causality.

L1/L2 results for new U models, including the detached-reset condition:

| case | layer | aggregation | Test BA (%) |
| --- | --- | --- | --- |
| U_DETACH | L1 | whole_count | 48.43 ± 7.62 |
| U_DETACH | L1 | fixed250_ordered | 56.05 ± 2.28 |
| U_DETACH | L1 | fixed250_shuffled | 37.77 ± 1.84 |
| U_DETACH | L1 | relative10_ordered | 70.88 ± 0.71 |
| U_DETACH | L1 | relative10_shuffled | 28.85 ± 1.37 |
| U_DETACH | L2 | whole_count | 50.12 ± 3.68 |
| U_DETACH | L2 | fixed250_ordered | 57.56 ± 2.11 |
| U_DETACH | L2 | fixed250_shuffled | 43.88 ± 0.82 |
| U_DETACH | L2 | relative10_ordered | 69.04 ± 2.52 |
| U_DETACH | L2 | relative10_shuffled | 39.58 ± 1.05 |
| U_NORMAL | L1 | whole_count | 49.45 ± 4.61 |
| U_NORMAL | L1 | fixed250_ordered | 55.60 ± 3.58 |
| U_NORMAL | L1 | fixed250_shuffled | 37.83 ± 0.88 |
| U_NORMAL | L1 | relative10_ordered | 71.58 ± 2.40 |
| U_NORMAL | L1 | relative10_shuffled | 30.81 ± 0.75 |
| U_NORMAL | L2 | whole_count | 53.32 ± 0.97 |
| U_NORMAL | L2 | fixed250_ordered | 58.59 ± 2.64 |
| U_NORMAL | L2 | fixed250_shuffled | 45.88 ± 2.36 |
| U_NORMAL | L2 | relative10_ordered | 66.40 ± 1.80 |
| U_NORMAL | L2 | relative10_shuffled | 41.49 ± 0.49 |

These are not hidden-layer ablation results. L1 and L2 are representations jointly optimized in the same model. Affine and pre-reset probe outputs exist but are not mixed into the primary spike/no-bias tables.

### 8.3 Reset detachment: gradient transport without native gain

Validation suffix CE excludes direct early-evidence credit. In future-reset bucket **10+**, the count-weighted mean absolute retained pre-reset gradients, pooled descriptively across seeds and τ groups, are:

| Carrier/reset | L1 suffix gradient | L2 suffix gradient |
| --- | --- | --- |
| I_REF | 3.834e−6 | 3.114e−7 |
| U_NORMAL | 3.198e−6 | 1.248e−6 |
| U_DETACH | 1.709e−5 | 1.873e−6 |

This table is an observation-weighted descriptive diagnostic, **not** mean±SD over independent seeds or a significance test. It uses `n_values` weights. Selected-checkpoint activity, τ occupancy and derivative distributions differ. Detachment increases these derivatives relative to U_NORMAL but has worse native BA in all seeds. A backward reset path can matter numerically without being the sufficient explanation of task performance.

### 8.4 Objective changes strongly affect communication, with confounds retained

The clean U WCCE→MWCCE contrast reduces pooled rate **26.70→5.52 Hz**, test BA **51.05→38.82%**, and head Frobenius norm **8.13→5.05**. Thus lowering rate is not merely compensated by increased head magnitude in this endpoint. I WCCE→MWCCE gives **23.94→5.09 Hz** and **57.93→32.50%**, but combines version differences with collapse. No result demonstrates a free rate reduction at constant native accuracy.

NWCCE γ values are **2.624/2.267/1.659** for I and **.07256/.07914/.06507** for U. Only approximately **33–34%** of U epoch0 training samples have nonzero L2 counts, versus **67–72%** for I. Matching a sparse initial raw-logit RMS locks a much smaller U training scale. Therefore U-NWCCE collapse is not a clean isolated test of count-magnitude invariance; calibration and near-zero activity interact. No corrected-γ formal rerun is present in the inspected family. I normalization also becomes near-silent, but this does not prove every normalized-count objective must fail.

### 8.5 Rate is not equivalent to selectivity or utility

Exp18.3 occupancy-based coordinates:

| case | mean_train_class_eta2 | mean_test_class_eta2 | mean_train_alignment_cos | mean_test_alignment_cos | mean_profile_transfer_cos |
| --- | --- | --- | --- | --- | --- |
| I_WCCE | 0.45 ± 0.01 | 0.43 ± 0.00 | 0.91 ± 0.01 | 0.74 ± 0.01 | 0.86 ± 0.00 |
| I_MWCCE | 0.14 ± 0.10 | 0.23 ± 0.09 | 0.58 ± 0.12 | 0.23 ± 0.16 | 0.57 ± 0.34 |
| U_WCCE | 0.35 ± 0.01 | 0.44 ± 0.01 | 0.89 ± 0.00 | 0.78 ± 0.01 | 0.88 ± 0.00 |
| U_MWCCE | 0.15 ± 0.01 | 0.27 ± 0.01 | 0.66 ± 0.02 | 0.34 ± 0.02 | 0.73 ± 0.01 |

Under I-WCCE mean rate–train class η² Spearman is **−.111**, and rate–test class η² is **−.006**. Under U-WCCE they are **+.071** and **+.019**. Rate–test mean-replacement ΔCE is **−.169 (I)** and **−.263 (U)**; rate–test margin contribution is **−.139 (I)** and **−.493 (U)**. These are mean within-model Spearman correlations across three seeds, not pooled neuron-level significance claims.

By contrast MWCCE rate–test-utility correlations are **+.405 (I)** and **+.549 (U)**, consistent with active neurons carrying much of the residual useful communication in a sparse regime. The relationship depends on loss/operating regime. The high-rate-WCCE population need not be the most class-selective population.

Descriptive standardized OLS predicts test single-neuron mean-replacement ΔCE from log1p(rate), train η², train alignment, head contrast norm and train→test profile transfer. Under WCCE, rate coefficients are negative in all seeds: I **−.193/−.339/−.174**, U **−.148/−.294/−.100**. Profile-transfer coefficients are positive: I **.571/.707/.708**, U **.662/.809/.698**. This does not estimate a causal effect of rate; neurons are dependent, the outcome and transfer predictor use test labels, and collinearity/finite profiles remain. It is evidence against treating rate alone as a useful OOD selector.

Exp18.3 does not implement a separate user-identity decoder or user η² matrix. Profile stability across train/test users is available; persistent-specific user nuisance remains an interpretation requiring dedicated controls.

### 8.6 Mean operating point versus sample-varying modulation

Let q_j=μ_j+δ_j, with μ_j the train mean occupancy. Zero replacement sets q_j=0; mean replacement sets q_j=μ_j. Neither intervention removes a neuron's contribution from hidden computation; both alter the frozen native feature vector.

Primary top-rate and random group BA drops (pp):

| Case | Group | Mean-replacement drop | Zero drop |
| --- | --- | --- | --- |
| I_WCCE | high_rate30 | 3.97 ± 2.18 | 17.94 ± 4.15 |
| I_WCCE | random30 | 2.90 ± 1.06 | 6.07 ± 1.09 |
| I_MWCCE | high_rate30 | 7.20 ± 5.90 | 7.66 ± 5.96 |
| I_MWCCE | random30 | 2.03 ± 1.72 | 2.28 ± 1.88 |
| U_WCCE | high_rate30 | 1.64 ± 2.65 | 26.92 ± 6.94 |
| U_WCCE | random30 | 1.69 ± 0.56 | 8.01 ± 1.67 |
| U_MWCCE | high_rate30 | 10.47 ± 3.63 | 25.28 ± 4.97 |
| U_MWCCE | random30 | 2.34 ± 2.18 | 6.55 ± 1.64 |

Random masks are averaged within seed. For WCCE, high-rate mean replacement is not uniformly more damaging than random: mean ΔCE is **.0930 versus .1176 (I)** and **.1097 versus .1177 (U)**. Yet zero replacement is much more damaging. This weakens the claim that top-rate neurons uniquely concentrate sample-varying native discrimination.

For a frozen bias-free head,

\[
z=R\mu+R\delta.
\]

The term Rμ is a class-dependent operating offset, even though there is no learned bias parameter. Removing it changes class balance and cancellation among feature contributions. Zero-versus-mean therefore supports a **readout operating-point/cancellation** explanation. It does not show that tonic spikes are biologically necessary or that a centered/retrained head would retain the same BA. Also, the zero-minus-mean BA difference is not an additive information decomposition because argmax and CE are nonlinear.

The result is not limited to high-rate coordinates: U-WCCE low-rate30 zero replacement also drops BA strongly, and selectivity/alignment-ranked groups can be more damaging than rate-ranked groups. Full controls are retained below.

| case | family | method | BA drop (pp) | ΔCE |
| --- | --- | --- | --- | --- |
| I_WCCE | high_rate30 | mean_replacement | 3.97 ± 2.18 | 0.09 ± 0.02 |
| I_WCCE | high_rate30 | zero | 17.94 ± 4.15 | 0.51 ± 0.12 |
| I_WCCE | low_rate30 | mean_replacement | 5.41 ± 1.23 | 0.14 ± 0.05 |
| I_WCCE | low_rate30 | zero | 12.23 ± 9.47 | 0.26 ± 0.10 |
| I_WCCE | high_selectivity30 | mean_replacement | 5.42 ± 2.45 | 0.23 ± 0.05 |
| I_WCCE | high_selectivity30 | zero | 28.54 ± 2.08 | 0.83 ± 0.07 |
| I_WCCE | high_alignment30 | mean_replacement | 6.04 ± 4.36 | 0.19 ± 0.05 |
| I_WCCE | high_alignment30 | zero | 22.62 ± 2.18 | 0.52 ± 0.05 |
| I_WCCE | high_margin30 | mean_replacement | 4.67 ± 1.49 | 0.15 ± 0.06 |
| I_WCCE | high_margin30 | zero | 18.10 ± 1.80 | 0.40 ± 0.12 |
| I_WCCE | random30 | mean_replacement | 2.90 ± 1.06 | 0.12 ± 0.03 |
| I_WCCE | random30 | zero | 6.07 ± 1.09 | 0.16 ± 0.03 |
| I_MWCCE | high_rate30 | mean_replacement | 7.20 ± 5.90 | 0.16 ± 0.14 |
| I_MWCCE | high_rate30 | zero | 7.66 ± 5.96 | 0.16 ± 0.14 |
| I_MWCCE | low_rate30 | mean_replacement | 0.40 ± 2.93 | 0.02 ± 0.02 |
| I_MWCCE | low_rate30 | zero | 0.29 ± 1.68 | 0.02 ± 0.02 |
| I_MWCCE | high_selectivity30 | mean_replacement | 9.11 ± 2.56 | 0.18 ± 0.16 |
| I_MWCCE | high_selectivity30 | zero | 10.46 ± 6.45 | 0.20 ± 0.17 |
| I_MWCCE | high_alignment30 | mean_replacement | 8.96 ± 7.74 | 0.20 ± 0.17 |
| I_MWCCE | high_alignment30 | zero | 10.51 ± 9.12 | 0.21 ± 0.18 |
| I_MWCCE | high_margin30 | mean_replacement | 10.38 ± 8.61 | 0.17 ± 0.15 |
| I_MWCCE | high_margin30 | zero | 16.28 ± 14.73 | 0.21 ± 0.18 |
| I_MWCCE | random30 | mean_replacement | 2.03 ± 1.72 | 0.08 ± 0.07 |
| I_MWCCE | random30 | zero | 2.28 ± 1.88 | 0.09 ± 0.07 |
| U_WCCE | high_rate30 | mean_replacement | 1.64 ± 2.65 | 0.11 ± 0.01 |
| U_WCCE | high_rate30 | zero | 26.92 ± 6.94 | 0.62 ± 0.29 |
| U_WCCE | low_rate30 | mean_replacement | 6.99 ± 4.14 | 0.16 ± 0.03 |
| U_WCCE | low_rate30 | zero | 25.86 ± 2.71 | 0.51 ± 0.08 |
| U_WCCE | high_selectivity30 | mean_replacement | 3.70 ± 2.56 | 0.15 ± 0.02 |
| U_WCCE | high_selectivity30 | zero | 20.71 ± 1.40 | 0.40 ± 0.03 |
| U_WCCE | high_alignment30 | mean_replacement | 4.58 ± 4.45 | 0.16 ± 0.01 |
| U_WCCE | high_alignment30 | zero | 7.32 ± 4.11 | 0.22 ± 0.03 |
| U_WCCE | high_margin30 | mean_replacement | 5.63 ± 4.17 | 0.17 ± 0.00 |
| U_WCCE | high_margin30 | zero | 25.36 ± 5.53 | 0.71 ± 0.23 |
| U_WCCE | random30 | mean_replacement | 1.69 ± 0.56 | 0.12 ± 0.01 |
| U_WCCE | random30 | zero | 8.01 ± 1.67 | 0.17 ± 0.02 |
| U_MWCCE | high_rate30 | mean_replacement | 10.47 ± 3.63 | 0.09 ± 0.01 |
| U_MWCCE | high_rate30 | zero | 25.28 ± 4.97 | 0.09 ± 0.02 |
| U_MWCCE | low_rate30 | mean_replacement | 1.52 ± 2.23 | 0.01 ± 0.01 |
| U_MWCCE | low_rate30 | zero | -1.29 ± 2.54 | 0.01 ± 0.01 |
| U_MWCCE | high_selectivity30 | mean_replacement | 12.55 ± 4.22 | 0.11 ± 0.02 |
| U_MWCCE | high_selectivity30 | zero | 23.24 ± 6.81 | 0.13 ± 0.03 |
| U_MWCCE | high_alignment30 | mean_replacement | 17.94 ± 2.77 | 0.12 ± 0.02 |
| U_MWCCE | high_alignment30 | zero | 24.63 ± 5.26 | 0.13 ± 0.03 |
| U_MWCCE | high_margin30 | mean_replacement | 7.94 ± 4.41 | 0.09 ± 0.02 |
| U_MWCCE | high_margin30 | zero | 26.69 ± 5.31 | 0.13 ± 0.03 |
| U_MWCCE | random30 | mean_replacement | 2.34 ± 2.18 | 0.05 ± 0.01 |
| U_MWCCE | random30 | zero | 6.55 ± 1.64 | 0.05 ± 0.01 |

### 8.7 Representation/native accessibility under altered loss

| Case | WholeCount (%) | Fixed250 order (%) | Fixed250 shuffle (%) | Relative10 order (%) | Relative10 shuffle (%) |
| --- | --- | --- | --- | --- | --- |
| I_WCCE_REF | 57.57 ± 4.24 | 58.62 ± 2.69 | 53.79 ± 3.04 | 67.04 ± 0.73 | 54.24 ± 3.66 |
| I_NWCCE | 24.64 ± 4.91 | 29.33 ± 4.12 | 24.00 ± 1.97 | 30.00 ± 5.11 | 21.56 ± 2.62 |
| I_MWCCE | 32.51 ± 17.79 | 35.62 ± 19.72 | 31.52 ± 17.09 | 36.28 ± 18.72 | 29.41 ± 15.39 |
| U_WCCE_REF | 53.32 ± 0.97 | 58.59 ± 2.64 | 45.88 ± 2.36 | 66.40 ± 1.80 | 41.49 ± 0.49 |
| U_NWCCE | 15.17 ± 2.68 | 15.65 ± 4.77 | 10.54 ± 2.44 | 17.76 ± 6.54 | 10.30 ± 0.30 |
| U_MWCCE | 47.44 ± 0.61 | 46.59 ± 1.48 | 37.45 ± 2.01 | 63.13 ± 8.57 | 34.61 ± 0.62 |

U-MWCCE retains **63.13 ± 8.57%** Relative10, **47.44 ± .61%** WholeCount and **46.59 ± 1.48%** Fixed250, compared with **38.82 ± 1.98%** native. The Relative10−native gap is **24.31 pp** descriptively. Crucially, even a refitted WholeCount head gains **8.62 pp** on the same checkpoint's spikes, while relative-phase information supplies an additional decoder opportunity.

This does not mean native could attain 63% by changing only its weights: Relative10 has ten time-specific weight blocks and uses completed-sequence phase/length. WholeCount uses a separately regularized/scaled logistic optimizer, and native uses end-to-end surrogate training on mean occupancy. These differences distinguish decoder opportunity from an identified causal readout defect.

## 9. Mathematical Interpretation

### 9.1 Temporal support under the actual valid-mean objective

The conversational sum-logit story z_k=Σ_t e_tk yields margin Tδ for repeated evidence. The executed objective instead uses:

\[
z_y-z_j=\frac1T\sum_t(e_{t,y}-e_{t,j}).
\]

Evidence δ aligned for r valid timesteps contributes rδ/T. At fixed T, increasing the fraction of supported timesteps enlarges the margin; increasing T while repeating the same δ throughout leaves the mean margin at δ. **Duration alone is not the implemented T-fold amplification.** Sum and mean have the same inference argmax but different optimization incentives.

For CE residual d=softmax(Rq)−onehot(y):

\[
\frac{\partial L}{\partial s_{t,j}}=\frac1T R_{:,j}^{\top}d.
\]

Time-independent direct credit makes repeated class-aligned communication an accessible solution, but extra spikes in unaligned directions can hurt. Coupled surrogate/state dynamics, signed projections, and gain determine the resulting code. Objective ablations support a role for this geometry, especially the matched U WCCE/MWCCE contrast; they do not prove it acts alone.

NWCCE removes uniform radial scaling of C for C≠0. Its Jacobian is (I−Ĉ1ᵀ)/||C||₁, so gradient directions and magnitude change as well as count-scale reward. Near zero counts, binary surrogate dynamics and the zero-vector branch introduce a difficult operating region. MWCCE instead stops direct margin reward beyond 1, but changes competition to the strongest rival and does not constrain rates or head norm directly. Their failures cannot be reduced to a single isolated “remove high-rate reward” factor.

### 9.2 Carrier/reset coupling

Before firing/reset, the cascaded filters commute; afterward slow-U resets alter retained history. Locally,

\[
\frac{\partial U_t}{\partial P_t}=1-\theta\phi'(P_t-\theta)
\]

under normal surrogate reset, versus 1 under detached reset. This changes gradient transport but also does not guarantee a useful learned class geometry. The clean detached-reset failure establishes that this modification alone does not rescue U classification under the tested protocol.

### 9.3 Output evidence, residual and signed support

Hidden spikes are nonnegative, but class evidence e_t=Rs_t is signed. An analog bias-free accumulator retains E=Σ_t e_t, with prediction argmax E. For an optional spiking output with subtractive reset,

\[
V_t=\beta_oV_{t-1}+e_t-\theta_o r_t,
\]

zero initial state implies, within the valid window,

\[
\sum_t e_t=\theta_o\sum_t r_t+V_T
 +(1-\beta_o)\sum_tV_{t-1}.
\]

At β_o=1, spike count alone discards signed residual V_T. Leakage adds the cumulative state term; it can erase useful negative and positive evidence depending on the trajectory. Positive-only draining can emit outstanding positive charge but cannot encode negative residual as negative spikes. This explains why an accumulator can be effective without proving it causes user-specific overfitting.

The hidden **carrier** β/α swap is distinct from an output-neuron leakage sweep. Core output extensions are related context, not additional Exp18 treatments. The inspected family does not execute signed output or adaptive-threshold readout, so neither is credited as a demonstrated improvement.

## 10. Negative Evidence and Failed Hypotheses

The hoped-for U-native improvement did not occur; detachment made native BA worse in every seed. U WCCE remained dense, weakening a carrier-only explanation. Margin saturation lowered firing but also reduced native BA; I seed11 nearly collapsed. NWCCE became near-silent and U calibration is confounded. Rate-selectivity/utility diagnostics do not support high-rate neurons as intrinsically superior class coordinates, and mean-replacement top-rate damage is not uniformly greater than random.

These outcomes justify preserving the effective accumulator reference while studying evidence organization. They do not justify another nominally successful sparse regime on the basis of rate alone.

## 11. Interpretation

**Observation.** Changing carrier/reset changes native and temporal-probe outcomes; U remains high-rate; a matched U loss change reduces rates strongly; class/profile/head diagnostics distinguish firing from native utility; zero replacement is much more damaging than mean replacement.

**Mechanistic interpretation.** The carrier determines which state survives threshold/reset and how temporal structure is represented. Objective/head geometry influences the communication operating regime and the formatting of evidence. A tonic mean component plus smaller class-dependent modulations can support the fixed bias-free classifier, with signed feature cancellation shaping prediction.

**Alternative explanations.** Gain, optimizer trajectories, count normalization scale, hinge competitor gradients, parameter norms, binary cap, finite samples, user/class coverage and initialization differences can contribute. Stronger temporal probes partly reflect extra dimensions and completed-sequence phase access. Profile drift is compatible with user/style/duration effects, but does not uniquely identify semantic nuisance.

**Confidence.** High that slow I is unnecessary for dense WCCE communication and that U margin loss changes firing; high for fixed-head zero/mean differences and recovered initialization mismatch; moderate that objective/head geometry organizes the code; limited for general I-versus-U superiority, universal layer ranking, or an isolated accumulator-induced transfer failure.

## 12. What This Experiment Established

1. Executed U models retain high-rate communication under WCCE.
2. Reset detachment alone does not improve U native classification.
3. Restoring I in L1 of fresh UU improves native BA in all three seeds, with seed-dependent magnitude; restoring I in L2 also gives positive but smaller average gain.
4. A matched U WCCE/MWCCE comparison changes firing strongly while decreasing native BA.
5. Weak-native U and margin-trained checkpoints retain information accessible to richer or refitted probes.
6. High firing is not synonymous with class selectivity, transfer stability or sample-varying native utility.
7. The frozen head relies on both mean feature operating contributions and discriminative modulation; zero replacement conflates them.

## 13. What This Experiment Did NOT Establish

It did not execute a fully matched v1.1 II-WCCE anchor, establish carrier-only causality from I_REF/U differences, prove that accumulator replacement resolves transfer, or demonstrate stable accuracy-preserving sparse communication. It did not identify a universal first-layer carrier mechanism, prove all NWCCE formulations fail, separate biological memory from every gain effect, or show signed output/adaptive thresholds improve BA. It did not establish that persistent activity is solely user-specific, nor that all probe-native gaps can be closed by native-head refitting.

## 14. Implications for the Train–Test Gap

The reused I-WCCE reference has **92.97% train**, **67.04% Relative10 test**, and **57.93% native test** BA. The arithmetic split is **25.93 pp** from native train to temporal probe and **9.11 pp** from that probe to native test. It is a descriptive decomposition, not two identified causal losses: training and probe classifiers differ and Relative10 is not a representation upper bound.

Fresh U-WCCE has **64.44% train**, **66.40% Relative10 test**, and **51.05% native test**. Probe test exceeds native train, demonstrating why train→probe cannot universally be interpreted as “information lost.” U-MWCCE likewise has **59.60% native train**, **63.13% Relative10 test**, and **38.82% native test**. Its large native-accessibility opportunity coexists with weaker representation metrics and training optimization.

The most defensible conclusion is that both representation and decoder geometry matter, with distinct, checkpoint-specific diagnostic contrasts. Higher train η² and lower user η² in Exp17.1 were insufficient for transfer; lower firing in Exp18.2 was insufficient for native performance. A readout that preserves signed evidence is useful, but current evidence does not isolate it as the source of the large unseen-user gap.

## 15. Transition to the Next Experiment

The next test should separate **class-varying transferable modulation**, **mean operating-point/cancellation**, and **native-head adaptation** before reducing persistent communication. An informative proposed control is a frozen-backbone comparison of the existing head with a train-derived centered/offset-aware refit, using fixed train-ranked groups and matched zero/mean/random interventions. It would test whether the zero-ablation dependence is mainly geometric or whether altered internal communication adds a further deficit.

A matched v1.1 II-WCCE anchor would repair the current carrier comparison; a deliberately recalibrated NWCCE protocol would test whether its collapse depends on scale. These are **future experiments**, not authorized reruns performed for this report. Signed output and adaptive thresholds address realization of signed/residual evidence; context-versus-evidence routing addresses hidden representation organization. Each needs its own controlled experiment rather than being described as an already successful solution.

## 16. Retrospective Interpretation

Earlier discussion proposed “carrier affects history quality, objective affects coding format.” The executed family partially supports this decomposition, but not a strict separation: changing loss also changes selectivity, alignment, profile transfer and parameter norms; changing carrier affects firing and native optimization. The initialization mismatch further weakens a clean carrier-only narrative.

The later Exp18.3 endpoint evidence sharpens the interpretation of Exp18.0–18.2 without being attributed to their original design. Broad high-rate WCCE activity can form a useful operating point whose most active coordinates are not the most informative per unit modulation. Removing that point with zero features can be much more damaging than removing sample-varying information with mean replacement. This favors studying organization and cancellation, while leaving hidden dynamic necessity unresolved.

| Emerging story element | Evidence status |
| --- | --- |
| Slow internal states preserve context | Direct implemented mechanism; earlier history tests provide context |
| Sequence-level count supervision can favor broad evidence support | Mathematical interpretation, partially supported by matched U loss contrast |
| Repeated communication is a useful learned regime | Direct high-rate endpoints; no universal necessity claim |
| Slow I alone causes high rate | Rejected as a necessity claim |
| Rate is an information category | Not supported by η²/utility controls |
| Class/user/context entanglement fully explains the gap | Unresolved semantic/causal hypothesis |
| Transferable information exists beyond native access in some checkpoints | Supported as a decoder opportunity; capacity/phase caveat retained |
| Organize/route evidence rather than simply suppress firing | Supported research direction, not an established performance solution |

## 17. Evidence Traceability

All Unity paths are relative to `/home/zhaolongwei_umass_edu/projects/writingRing/`. Raw results are not all Git-tracked. Source, protocol locks, per-run native/probe/activity/loss JSON, histories, aggregate CSV, complete certificates, Git changes, selected stdout and scheduler accounting were inspected. Project-history search returned dated decisions and earlier assistant summaries rather than a full raw conversation export; history is used for motivation/interpretation, not numerical truth.

| ID / claim | Evidence source / exact coordinate |
| --- | --- |
| E18 equations / reset / selection | [implementation](../../scripts/experiment_18_membrane_history.py), [README](../../scripts/experiment_18_membrane_history/README.md); `notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1/protocol.lock.json` |
| E18 native / probes / gradients | Same root `aggregate.json`; `runs/{U_NORMAL,U_DETACH}__seed{11,23,37}/{native,probes,history,complete}.json`; `diagnostics/{I_REF,U_NORMAL,U_DETACH}__seed*/gradient_diagnostics.json`; suffix table filters `loss_kind=suffix`, bucket=`10+`, weights `n_values` |
| E18.1 layer mapping / metrics | [implementation](../../scripts/experiment_18_1_layerwise_memory_carrier.py), [README](../../scripts/experiment_18_1_layerwise_memory_carrier/README.md); `notebooks/artifacts/experiment_18_1_layerwise_memory_carrier/layerwise_memory_carrier_v1/aggregate.json` rows and factorial_contrasts |
| E18.2 exact losses / calibration | [implementation](../../scripts/experiment_18_2_loss_geometry.py), [README](../../scripts/experiment_18_2_loss_geometry/README.md); `notebooks/artifacts/experiment_18_2_loss_geometry/loss_geometry_v1/runs/*/{loss_config,activity,native,history,complete}.json`; root `aggregate.json` |
| E18.3 rate/η²/alignment/profile utility | [implementation](../../scripts/experiment_18_3_discriminative_coordinate.py), [README](../../scripts/experiment_18_3_discriminative_coordinate/README.md); `notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate/{case_summary,run_summary,correlations,utility_ols}.csv` |
| E18.3 zero/mean controls | Same root `aggregate/group_ablation.csv`: split=test, family/method; random masks averaged within seed before sample SD; positive `delta_ba_pp` denotes baseline minus ablated |
| E18-INIT version and actual pairing | `core_benchmark_v1/results/main/protocol.lock.json`, `runs/O0__seed*/initial.pt`; [version override](../../scripts/experiment_16_prefix_supervised_selective_memory.py), [paired_seed](../../core_benchmark_v1/protocol.py); read-only `torch.equal` on layers.0.weight/layers.1.weight/head.weight versus fresh family initial.pt |
| Integrity / execution | 36 complete certificates rehashed with zero mismatches; root PASS aggregates; `sacct` Oct1–6 with fixed user; job IDs in Section6 |
| Actual failed causes | `slurm-65177121_0.out` loader NameError; `slurm-65178164.out` state_hash NameError; `slurm-65179990_0.out` already_complete; commits `4da0164`, `b1961d7`; Lmod scope from bootstrap commits/README |
| H18-1 original contract | Retrieved user discussion 2026-10-03 02:21:19 UTC; original-result discussion 03:52:26 UTC |
| H18-2 change of direction | Retrieved carrier-localization discussion 2026-10-03 04:03:55 UTC; objective direction 07:16:03 UTC; loss-result discussion 12:36:34 UTC; Exp18.3 design 13:12:55 UTC |
| H18-3 later train–test interpretation | Project conversations “exp18”, “Branch · 分支 · exp18”, “梳理CoreBenchmark到Exp18故事” on Oct3–5; retrieval 2026-10-03 23:55:12 UTC and visible Project context; no complete conversation archive claimed |
| Related output geometry | [CoreBenchmark report](../core_benchmark_v1/report.md); [output residual extension](../../core_benchmark_v1/extensions/output_residual_leakage/README.md), [output drain](../../core_benchmark_v1/extensions/output_spike_drain/README.md); contextual mechanism, not Exp18-family quantitative treatment |

Artifact hashes:

| Artifact | SHA256 |
| --- | --- |
| notebooks/artifacts/experiment_18_membrane_history/membrane_history_v1/aggregate.json | f47951fae4122bdd3adabe14f623e1bfcce1e2f37355416b1260c925a579cf5b |
| notebooks/artifacts/experiment_18_1_layerwise_memory_carrier/layerwise_memory_carrier_v1/aggregate.json | 1d1756e0ede9e84ecabd4029aef2e53274b986602dbbcb511b06df86408b6c9f |
| notebooks/artifacts/experiment_18_2_loss_geometry/loss_geometry_v1/aggregate.json | b0bff03258ee00d05f5db49b5d88fb32c7baf24d457defed4a37f640a131a9eb |
| notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate.json | 81923f00430588d8a66c34c4166a0316d0b7309c1c0ef5fa548152f91e9ae954 |
| notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate/case_summary.csv | 6963ca1ba91110ffdaf38b431dbe1969250eceb7dd733ce29e35bda91823c4f7 |
| notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate/group_ablation.csv | 983487e771c4818b27b385264207015127a14512f5d5562efb62945f8d0bff64 |
| notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate/correlations.csv | 1322b914865a8432ae2e16d8c15f6bc08913d165622ea297f2aff85f4a7c2fc4 |
| notebooks/artifacts/experiment_18_3_discriminative_coordinate/discriminative_coordinate_v1/aggregate/utility_ols.csv | 333552310c268e9eb85d9f235396d69dfd08fdb4a344d867dd332cf12f435e0e |
