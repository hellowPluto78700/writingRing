# Exp17 — Persistent Firing, Distributed Discrimination, and Low-Rate Recruitment

Evidence cutoff: **2026-10-06**. Repository and Unity inspected HEAD: `0480ef082898dbbc162bf83f2c7e9c8e5b835773`; repository: `hellowPluto78700/writingRing`; Unity checkout: `/home/zhaolongwei_umass_edu/projects/writingRing`. This report covers the canonical Exp17 family, including Exp17.1 and Exp17.2. Exp18 numerical results are confined to Section 16. BA is balanced accuracy in percent; drops and contrasts are percentage points (pp). Unless otherwise stated, ± denotes **sample SD over optimization seeds 11/23/37, n=3 on one locked user split**. Random-mask replicates and user-CV folds are averaged within case/seed before seed statistics. Different source cases are never treated as exchangeable model seeds.

## 1. Executive Summary

Exp17 asked whether persistent/high-rate L2 activity was a nuisance shortcut that could be removed to improve unseen-user classification, or part of the learned discriminative code. It combined fresh baseline training trajectories, dimension-matched class/user subset probes, and frozen-versus-refitted feature interventions. Exp17.1 then explicitly supervised a fixed epoch-20 low-occupancy quartile; Exp17.2 replayed the source states to test whether history increasingly drowned out incoming input.

Three findings change the original story. First, persistence expands during ordinary WCCE training, but the highest-occupancy quartile's native-head weight share falls from approximately 25% to 23%; class selectivity migrates toward the dynamically lower-rate group. Second, both high and low subsets support class decoding, and high-occupancy pruning is not uniformly better than matched random pruning. In the inherited Exp16.2 C0 source case, top-30% mean replacement drops the frozen WholeCount probe by **5.09 ± 5.14 pp**, reduced to **2.24 ± 2.42 pp** after refitting; bottom-30% drops are **6.92 ± 1.58** and **0.54 ± 2.32 pp**. These are post-hoc probe interventions, not live neuron deletion. Third, Exp17.1 changes train-side selectivity without solving transfer: fixed-low class η² rises **0.366→0.549**, but native test BA changes **56.82→56.27%**, and its fixed-low probe changes **55.21→54.92%**. [E17-A/B/C; E17.1]

The fixed-low population starts at **4.06 Hz** on average and reaches **21.70 Hz** under baseline continuation and **24.52 Hz** under auxiliary supervision. At Exp17-A selected checkpoints, 265 of 267 neurons remaining after top-30% exclusion still exceed the operational **>10 Hz** threshold. High-rate communication is therefore a broad operating regime, not a small pathological caste. Exp17.2 further rejects the specific drowning-out hypothesis: incoming-drive spike-flip probability increases by **2.77–3.16 pp** from epoch 20 to selected checkpoints, alongside increasing history-sufficient spiking. [E17-A; E17.1; E17.2]

The evidence supports distributed discriminative coding and changing neuron roles. It does not establish that high firing causes the train–test gap or that rescuing low-rate neurons creates transferable evidence. The next question is why this regime is attractive under the architecture and count-based objective.

## 2. Context and Motivation

Earlier CoreBenchmark/Exp13 work established substantial state-dependent history and a distinction between temporal-probe information and native accessibility. Exp14/14.1 supervision did not reliably convert improved representations into native transfer. Exp15/16 write gates often remained open or were compensated by representation/readout changes. Exp16.3 reduced within-episode repeated-spike reward, yet persistence was not suppressed and classification deteriorated. The predecessor evidence is documented in [Exp16](../exp16/report.md); its numbers are not new Exp17 results.

The original Exp17 design explicitly sought closure of three arrows: increasing persistent-readout reliance during optimization; preferential class-plus-user information in persistent dimensions; and persistence-specific pruning benefits beyond generic dimensionality reduction. Project retrieval dated **2026-10-01 22:35:14 UTC** records this motivating story. The results discussion at **2026-10-02 00:53:11 UTC** already weakened it: persistence increased, but reliance and persistent-specific user information did not. Later discussion reframed the useful observation as early high-rate recruitment followed by role differentiation. This chronology matters: the original experiment did not start by knowing that high firing was merely a phenotype. [H17-1]

## 3. Research Questions

1. Does ordinary WCCE training increase persistent activity, selective readout reliance, or both?
2. Do high-occupancy neurons uniquely carry class information or disproportionately encode class-controlled user identity?
3. Does an intervention damage information, misalign the fitted head, or expose redundancy in remaining coordinates?
4. Is the trained population broadly high-rate after excluding its highest-rate neurons?
5. Can an independently supervised early-low group remain low-rate and improve native or probe transfer?
6. Does stronger history reduce incoming-input control over spike decisions?

## 4. Competing Hypotheses

| Hypothesis | Discriminating prediction | Outcome |
| --- | --- | --- |
| Small high-rate nuisance subset drives poor transfer | Removing it improves OOD and outperforms random removal | Not generally supported |
| Persistent dimensions have unique class information | Frozen loss remains large after matched refitting | Substantial recovery; uniqueness not established |
| Persistence increasingly monopolizes the head | Top-quartile weight share and frozen dependence rise consistently | Activity rises; weight share does not |
| Early persistent success starves an alternative route | Fixed-low auxiliary CE increases transferable selectivity without recreating high rate | Train selectivity improves, rate rises, OOD does not improve |
| High-rate groups are fixed neuron identities | Early-low neurons remain low through optimization | Rejected by tracked fixed groups |
| History drowns out incoming evidence | Firing/history sufficiency rise while input-flip probability falls | Last requirement fails in all seeds |

## 5. Experimental Design

### 5.1 Canonical identity and family boundaries

| Conversational name | Canonical experiment / protocol | Implemented and executed scope |
| --- | --- | --- |
| Exp17, 17-A/B/C | `experiment_17_persistent_pathway_story` / `persistent_pathway_story_v1` | Three new trajectories; 24 subset tasks; 24 pruning tasks on eight inherited source cases |
| Exp17.1 | `experiment_17_1_gradient_starvation` / `gradient_starvation_v1` | Seed101 calibration, three shared bootstraps, 12 continuations, 12 post-hoc group probes |
| Exp17.2 | `experiment_17_2_history_input_competition` / `history_input_competition_v1` | Three artifact replay tasks and three short-horizon tasks; no new model |
| Earlier proposed “Exp18” edge/routing plan | Proposed successor communication-edge/refit/rerouting experiments | Not the canonical Exp18 carrier experiment; not counted as executed Exp17 results |

Implementations and READMEs have the same canonical names under `scripts/`. Artifact roots are `notebooks/artifacts/<canonical experiment>/<protocol>/`. Eight 17-B/C source cases are **C0, GZ0, S90, G90, S70, G70, S50, G50**, from Exp16.2. Their S/G names reflect that earlier implemented budget protocol; they are not renamed to the ρ values in later proposals. Inherited results and new trajectories remain distinct cohorts.

### 5.2 Data and training contract

Inputs use both action0/action1 64 Hz aligned-board wavelet-event datasets. The model takes the **30 event channels**, excluding six appended IMU channels from the 36-channel source. Labels are **A, B, C, D, E, G, H, I, J, K, L, X**, rather than the full alphabet. Padding horizon is 256; every sequence has an explicit valid length and padded timesteps contribute no spikes/evidence/features.

| Split | Users | Samples |
| --- | --- | --- |
| Train | user_0, user_1, user_11, user_12, user_13, user_14, user_15, user_18, user_19, user_2, user_20, user_5, user_7, user_8 | 581 |
| Validation | user_16, user_4, user_9 | 126 |
| Test | user_10, user_3, user_6 | 146 |

Split seed is 12345. Cache SHA256 is `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a`; Core identity is `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`. BA pools samples before class recalls; it is not mean per-user BA. Train metrics are in-sample, not held-out seen-user generalization.

The fresh backbone is **30→128(234)→128(234)→12**, with fully connected bias-free weights, threshold θ=0.5, surrogate slope 25, and at most one spike per neuron per timestep. Each layer has slow-pole groups of **43/43/42 neurons**. At Δt=15.625 ms, α=(0.75,0.875,0.9375) corresponds to τ=(54.313,117.014,242.103) ms; the often-used conversational “249 ms” is not this exact implementation. The fast membrane pole is β=exp(−15.625/22.54)≈0.49997.

For layer input a_t (raw events at L1, same-timestep L1 spikes at L2):

\[
I_t=\alpha I_{t-1}+Wa_t,\quad P_t=\beta U_{t-1}+I_t,
\quad s_t=\mathbf1[P_t\ge\theta],\quad U_t=P_t-\theta s_t.
\]

There is **no (1−α) input multiplier**. Slow I carries context; spike reset consumes membrane, not I. Spikes communicate state-dependent evidence to the next layer. Persistence of spikes is not the primary storage variable.

The bias-free head computes e_t=Rs_t and valid-mean sequence logits:

\[
q_i=\frac1{T_i}\sum_{t<T_i}s_{i,t},\quad z_i=Rq_i,
\quad L_{WC}=\frac1N\sum_i CE(z_i,y_i).
\]

Accumulator sum and mean have identical per-sample argmax, but CE training scales differ. The actual WCCE uses valid **mean** logits. Adam lr=.001, weight decay=0, batch=128, max epochs=100, min=20, patience=30. Selection maximizes native validation BA, then minimizes validation mean-logit CE, then selects the earliest tied epoch. Test does not select checkpoints. Exp17 uses the Exp16.3 deterministic epoch sampler; Exp17.1 branches from matched serialized epoch-20 checkpoints.

Version caveat: the inherited Core lock is v1.0, while `exp16._load_core` substitutes the current v1.1 protocol version for fresh model construction. `paired_seed` includes that version. Fresh Exp17 checkpoints are not the original Core O0 checkpoints, even with equal seeds/data. Within-family matched conditions remain the intended unit of comparison.

### 5.3 Grouping, intervention, and retraining semantics

17-A ranks **train sample-mean occupancy**, q̄_j=mean_i(C_ij/T_i), at each snapshot. Dynamic high25/low25 contain 32 neurons; high30 contains ceil(.30×128)=39. Rankings are per model/seed and change with epoch. The native-head frozen intervention replaces selected occupancy features with their training mean. It does not suppress internal firing or re-run altered hidden dynamics.

17-B trains bias-free WholeCount logistic probes on high30, low30, and five dimension-matched random30 subsets. User identity is decoded only within training users with five user-stratified CV folds: one test fold, a different validation fold, three training folds. Class residualization subtracts centroids fitted only on each CV training fold. This is a first-order class control, not complete class/user disentanglement.

17-C compares 10/20/30% train-ranked high occupancy, low occupancy, standardized WholeCount probe coefficient norm, and 20 deterministic random masks. `frozen_mean_replacement` keeps the full fitted probe and replaces selected raw feature dimensions by train means; `retrained_without_neurons` drops those feature columns and refits a **new no-bias probe**. Its reference is the full probe, not the native head. Refitting permits scaler and C reselection under the same train/validation contract.

Probes use train-fitted `StandardScaler(with_mean=False)` and bias-free multinomial logistic regression. C∈{.001,.01,.1,1,10,100}; converged candidates maximize validation BA, retaining smaller C on ties; max_iter=3000, tol=1e−4. Test never selects rankings, masks, hyperparameters, or user centroids.

17-A/17-B class η² is computed from **raw WholeCount**, whereas Exp17.1 uses **occupancy**. Both are one-way between-label variance/total variance, but they differ under duration variability. Their numerical values must not be pooled as one selectivity measure. User η² alone also cannot establish nuisance-specific semantics.

### 5.4 Fixed-low specialization and input-competition controls

Exp17.1 fixes the epoch-20 train-ranked low/high/random quartiles permanently (32 neurons each). C0 continues baseline; C1 adds low-only auxiliary CE; C2 adds high-only auxiliary CE; C3 adds random-only auxiliary CE. All heads are bias-free and consume valid-mean L2 spikes; only the main head defines native metrics. Seed101 matches low-group main/aux gradient norms over five available batches and locks **λ=1.3061355096**, with no BA-based tuning. The hypothesized later persistent-head attenuation Phase2 was intentionally not implemented.

Exp17.2 decomposes P_t=M_t+H_t+N_t, where M_t=βU_{t−1}, H_t=αI_{t−1}, N_t=Wa_t. It holds factual prior states fixed while removing current drive, synaptic carry, or membrane carry at one timestep. Its incoming flip metric is P(s_full≠s_no-new), while history sufficiency is P(s_no-new=1 | s_full=1). These have different denominators and can both increase. A separate raw-x_t=0 counterfactual propagates altered current L1 spikes to L2 while retaining factual prior states. The short-horizon test drops one L2 input drive, restores subsequent factual L1 drives, and propagates counterfactual L2 states over 1/2/4/8/16 steps, with stride-eight anchors and eligible-sample accounting.

Exp17.2 rate is **pooled-time** fs×Σ_iC_ij/Σ_iT_i, whereas 17-A/17.1 rate is **sample-mean** fs×mean_i(C_ij/T_i). The audit's nonzero epoch20 occupancy discrepancies (0.02295/0.02280/0.02856 for seeds 11/23/37) follow these definitions; factual instrumented forward replay is exact in all three audits. Fixed epoch20 membership is taken from Exp17 source ranking rather than redefining it from pooled rates.

## 6. Implementation and Execution History

| Stage | Git / Slurm evidence | Status |
| --- | --- | --- |
| Main Exp17 closure | Initial commits `7eda15c`, `0766b91`; prepare 65146351; trajectory 65146352_0–2; subsets 65146353_0–23; pruning 65146354_0–23; final 65146355 | All inspected accounting rows COMPLETED, exit 0:0; aggregate manifest has 65 snapshots, 168 class rows, 1680 user rows, 3312 pruning rows |
| Exp17.1 implementation | PR #96; `f0ba6ec`; metadata contamination repaired by `6f0f6aa`/`8d7546f` | Syntax/source repair, not a scientific treatment |
| Exp17.1 formal | Calibration 65154082; bootstrap 65154083_0–2; continuations 65154084_0–11; final 65154085 | All COMPLETED 0:0; 12 metrics/histories present |
| Exp17.1 post-hoc probe | Documentation `0a956a1`; tasks 65155590_0–11; final 65155591 | All COMPLETED 0:0; aggregate has all 12 case/seed probes |
| Exp17.2 | PR #101; `aa8defc`; factual-threshold margin fix `815331a`; prepare 65171913; replay 65171914_0–2; horizon 65171915_0–2; final 65171916 | All COMPLETED 0:0; all factual replay audits true |
| Attenuation/rerouting successor plans | Project discussion | Proposed, not valid Exp17 results |

Scheduler timestamps place main Exp17 on October 1, Exp17.1 on October 2, and Exp17.2 on October 2 (UTC accounting). Current aggregate artifacts, histories, manifests and source were inspected; these older families do not all provide Exp18-style hash-gated completion certificates. There is no recovered scientific invalidation of these final aggregates. The contamination and threshold fixes are recorded without inventing numerical results from pre-fix revisions. No training, dataset, checkpoint, or result artifact was changed for this report.

## 7. Main Results

### 7.1 Fresh 17-A baseline, selected checkpoints

| seed | epoch | train_native_ba | val_native_ba | test_native_ba | persistent_top25_head_weight_share | test_native_frozen_remove_high30_drop_pp | test_wholecount_frozen_remove_high30_drop_pp | test_wholecount_retrained_remove_high30_drop_pp | L2 sample-mean Hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 95 | 93.607 | 56.334 | 60.097 | 23.39 | 2.832 | -0.127 | -1.786 | 23.464 |
| 23 | 96 | 92.728 | 58.11 | 54.536 | 22.955 | 1.52 | 0.366 | -1.902 | 22.981 |
| 37 | 84 | 91.245 | 55.332 | 55.071 | 22.967 | 5.441 | -1.724 | -0.469 | 22.794 |

All BA columns are percent; the three intervention columns are **positive drop** pp, with negative drop denoting improvement. Selected epochs are 95/96/84; epoch100 is a stopped/last snapshot, not the selected result. Mean selected train/val/test BA is **92.53 ± 1.19 / 56.59 ± 1.41 / 56.57 ± 3.07%**. Native high30 mean replacement drops by **3.26 ± 2.00 pp**. The full WholeCount probe and its refit can respond differently from the native head; the selected high30 refit improves the full-probe result in all three seeds. This does not establish that live high-rate dynamics are harmful.

### 7.2 Matched continuation in Exp17.1

| case | Train BA (%) | Val BA (%) | Native test BA (%) | Fixed-low rate (Hz) | Fixed-low class η² | Fixed-low user η² |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 93.43 ± 0.54 | 58.54 ± 2.47 | 56.82 ± 2.99 | 21.70 ± 0.73 | 0.37 ± 0.03 | 0.16 ± 0.03 |
| C1 | 91.46 ± 0.95 | 57.81 ± 2.16 | 56.27 ± 1.19 | 24.52 ± 1.33 | 0.55 ± 0.03 | 0.10 ± 0.01 |
| C2 | 88.93 ± 6.48 | 58.53 ± 1.75 | 54.79 ± 5.02 | 21.21 ± 0.45 | 0.35 ± 0.07 | 0.17 ± 0.05 |
| C3 | 89.67 ± 5.03 | 56.18 ± 2.52 | 57.28 ± 4.08 | 22.16 ± 0.96 | 0.39 ± 0.07 | 0.15 ± 0.05 |

η² values are dimensionless; rates are sample-mean Hz. C1−C0 native test is **−0.54 pp** on average, not a gain. Auxiliary supervision improves train-side class separation and reduces one-way user η² in the fixed-low subspace, but this is insufficient evidence of useful unseen-user geometry.

Seed-level values (BA %, fixed-low Hz):

| case | seed | best_epoch | train_ba | val_ba | test_ba | Fixed-low Hz |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 11 | 99 | 94.042 | 56.026 | 60.114 | 21.646 |
| C1 | 11 | 92 | 90.548 | 56.982 | 55.262 | 24.759 |
| C2 | 11 | 97 | 93.044 | 57.65 | 60.583 | 21.649 |
| C3 | 11 | 97 | 93.086 | 53.54 | 61.744 | 23.262 |
| C0 | 23 | 98 | 93.203 | 60.963 | 56.061 | 22.448 |
| C1 | 23 | 98 | 91.395 | 56.174 | 55.98 | 25.722 |
| C2 | 23 | 66 | 81.466 | 60.544 | 52.002 | 20.747 |
| C3 | 23 | 73 | 83.893 | 58.553 | 53.748 | 21.778 |
| C0 | 37 | 95 | 93.045 | 58.635 | 54.275 | 20.993 |
| C1 | 37 | 100 | 92.442 | 60.259 | 57.581 | 23.091 |
| C2 | 37 | 100 | 92.279 | 57.385 | 51.778 | 21.244 |
| C3 | 37 | 98 | 92.027 | 56.456 | 56.347 | 21.453 |

### 7.3 Dimension-matched subset decoding

| case | subset | Train BA (%) | Test BA (%) |
| --- | --- | --- | --- |
| C0 | high30 | 89.32 ± 5.70 | 54.89 ± 3.89 |
| C0 | low30 | 90.20 ± 6.64 | 53.16 ± 1.83 |
| C0 | random30 | 92.36 ± 4.27 | 52.79 ± 3.90 |
| GZ0 | high30 | 90.21 ± 10.06 | 50.36 ± 7.35 |
| GZ0 | low30 | 84.98 ± 10.46 | 50.69 ± 3.24 |
| GZ0 | random30 | 87.67 ± 11.74 | 51.07 ± 4.89 |
| S90 | high30 | 90.56 ± 5.34 | 53.61 ± 2.99 |
| S90 | low30 | 91.09 ± 8.04 | 54.24 ± 1.72 |
| S90 | random30 | 94.63 ± 2.77 | 53.56 ± 3.80 |
| G90 | high30 | 93.03 ± 5.25 | 52.23 ± 3.59 |
| G90 | low30 | 92.01 ± 4.12 | 53.66 ± 1.70 |
| G90 | random30 | 94.07 ± 2.20 | 54.05 ± 3.99 |
| S70 | high30 | 94.62 ± 0.43 | 53.16 ± 1.56 |
| S70 | low30 | 93.49 ± 4.92 | 55.31 ± 0.23 |
| S70 | random30 | 95.85 ± 0.85 | 53.77 ± 2.30 |
| G70 | high30 | 93.90 ± 5.34 | 52.37 ± 7.09 |
| G70 | low30 | 90.31 ± 5.29 | 54.33 ± 5.15 |
| G70 | random30 | 93.34 ± 1.49 | 52.84 ± 5.82 |
| S50 | high30 | 98.72 ± 0.97 | 50.02 ± 3.37 |
| S50 | low30 | 89.84 ± 2.43 | 55.94 ± 3.58 |
| S50 | random30 | 94.82 ± 2.51 | 53.87 ± 3.93 |
| G50 | high30 | 89.97 ± 0.99 | 52.20 ± 3.23 |
| G50 | low30 | 88.89 ± 0.23 | 53.06 ± 1.47 |
| G50 | random30 | 91.00 ± 2.15 | 53.96 ± 3.15 |

Each row summarizes three source-model seeds, with random subset replicates averaged within seed. Results are kept separate by inherited Exp16.2 condition. Both high30 and low30 are strongly class-decodable relative to 12-class chance (8.33%). The mixed source-case headline mean (52.35% high30 versus 53.80% low30) in `story_summary.json` is descriptive across heterogeneous cases, not an independent-seed inferential comparison.

## 8. Mechanistic Diagnostics

### 8.1 Activity increases while relative readout concentration does not

| epoch | L2 occupancy | High25 head share (%) | High25 class η² | Low25 class η² | High25 user η² | Low25 user η² |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.02 ± 0.00 | 24.98 ± 0.56 | 0.08 ± 0.01 | 0.02 ± 0.00 | 0.18 ± 0.01 | 0.04 ± 0.01 |
| 20 | 0.19 ± 0.02 | 25.53 ± 0.11 | 0.32 ± 0.00 | 0.10 ± 0.00 | 0.20 ± 0.01 | 0.17 ± 0.01 |
| 50 | 0.32 ± 0.00 | 24.05 ± 0.26 | 0.33 ± 0.02 | 0.40 ± 0.04 | 0.15 ± 0.01 | 0.13 ± 0.00 |
| 100 | 0.37 ± 0.00 | 22.75 ± 0.16 | 0.38 ± 0.01 | 0.53 ± 0.01 | 0.11 ± 0.01 | 0.08 ± 0.01 |

Dynamic high/low membership changes at every epoch. At epoch20 high25 class η² is approximately .324 and low25 .097; by epoch100 high25 is .377 and low25 .534. Within-neuron occupancy/class-η² Pearson correlations change from **+.548/+.643/+.541** at epoch20 to **−.397/−.443/−.501** at epoch100. Top25 head share changes from approximately .256 at epoch20 to .227 at epoch100. Therefore “more persistent neurons” is directly supported, while “optimization progressively allocates more native-head weight to the persistent quartile” is not. Head norm share measures coefficients, not actual logit or causal contribution.

### 8.2 Remaining neurons are still mostly medium/high-rate

At selected checkpoints, train sample-mean rates are:

| Seed / epoch | Full mean Hz | Full median Hz | Full >10 Hz | Remaining mean Hz after high30 exclusion | Remaining median Hz | Remaining >10 Hz |
| --- | --- | --- | --- | --- | --- | --- |
| 11 / 95 | 23.46 | 22.81 | 127/128 | 20.50 | 20.79 | 88/89 |
| 23 / 96 | 22.98 | 22.43 | 127/128 | 19.94 | 20.24 | 88/89 |
| 37 / 84 | 22.79 | 22.44 | 128/128 | 20.05 | 20.56 | 89/89 |

Removed top39 mean rates are 30.22/29.93/29.05 Hz; bottom39 means are 17.26/16.78/16.97 Hz. Their “low” label is relative rank, not sparse activity. The 265/267 remaining >10 Hz figure is a population description, not 267 independent experimental replicates. The threshold >10 Hz is an operational interpretation from the Project discussion; it is not the train-occupancy ranking rule or the original >.5/>.8 occupancy diagnostic.

### 8.3 Frozen versus refitted probe: specificity controls

The full top30/fraction-matched endpoint comparison is:

| Case | Ranking | Frozen mean replacement drop (pp) | Refitted probe drop (pp) |
| --- | --- | --- | --- |
| C0 | high_occupancy | 5.09 ± 5.14 | 2.24 ± 2.42 |
| C0 | low_occupancy | 6.92 ± 1.58 | 0.54 ± 2.32 |
| C0 | high_readout_weight | 8.72 ± 3.10 | 0.12 ± 1.90 |
| C0 | random | 4.86 ± 2.99 | 1.01 ± 0.94 |
| GZ0 | high_occupancy | 7.63 ± 1.61 | 1.46 ± 1.74 |
| GZ0 | low_occupancy | 8.53 ± 4.99 | 2.20 ± 3.49 |
| GZ0 | high_readout_weight | 13.40 ± 5.70 | -0.27 ± 1.91 |
| GZ0 | random | 7.94 ± 3.42 | 2.21 ± 2.56 |
| S90 | high_occupancy | 7.05 ± 2.52 | -2.32 ± 1.47 |
| S90 | low_occupancy | 6.45 ± 0.81 | 2.58 ± 5.87 |
| S90 | high_readout_weight | 12.94 ± 7.17 | -0.15 ± 1.11 |
| S90 | random | 5.56 ± 2.70 | 0.46 ± 0.63 |
| G90 | high_occupancy | 3.22 ± 2.52 | 1.42 ± 2.64 |
| G90 | low_occupancy | 9.85 ± 6.24 | 0.57 ± 1.70 |
| G90 | high_readout_weight | 7.65 ± 5.32 | 1.59 ± 1.11 |
| G90 | random | 5.66 ± 2.05 | 1.38 ± 1.30 |
| S70 | high_occupancy | 2.09 ± 2.49 | -3.08 ± 5.43 |
| S70 | low_occupancy | 9.37 ± 0.77 | -0.63 ± 1.87 |
| S70 | high_readout_weight | 5.56 ± 0.68 | -1.61 ± 5.27 |
| S70 | random | 5.37 ± 3.13 | -0.27 ± 1.77 |
| G70 | high_occupancy | 4.80 ± 5.69 | 1.25 ± 1.51 |
| G70 | low_occupancy | 11.83 ± 7.97 | -0.53 ± 2.12 |
| G70 | high_readout_weight | 13.14 ± 7.51 | 2.08 ± 2.41 |
| G70 | random | 6.72 ± 5.28 | 0.94 ± 1.95 |
| S50 | high_occupancy | 4.08 ± 0.51 | 5.44 ± 3.85 |
| S50 | low_occupancy | 11.99 ± 2.64 | -0.29 ± 2.41 |
| S50 | high_readout_weight | 5.04 ± 3.43 | 0.78 ± 0.61 |
| S50 | random | 6.06 ± 1.07 | 0.84 ± 0.80 |
| G50 | high_occupancy | 5.13 ± 3.09 | 1.66 ± 2.29 |
| G50 | low_occupancy | 14.92 ± 4.49 | 0.19 ± 0.79 |
| G50 | high_readout_weight | 12.90 ± 3.08 | 1.27 ± 2.05 |
| G50 | random | 9.60 ± 1.98 | 0.91 ± 0.59 |

Positive drop means full probe minus intervened probe. Random values average 20 masks per seed before SD. High30 refitting improves some source cases (S90, S70) but hurts others; C0 high30 remains numerically worse than random. Low30 and high-readout-weight removal also recover substantially after refitting. Hence recovery is evidence for accessible information and head adaptation in remaining dimensions, not proof of a uniquely redundant high-rate pathway. Refitting changes both selected regularization and decision geometry and cannot isolate their contributions.

### 8.4 User/context information is broadly distributed

Class-controlled user-CV BA:

| case | subset | CV user BA (%) |
| --- | --- | --- |
| C0 | high30 | 34.74 ± 0.96 |
| C0 | low30 | 33.66 ± 0.84 |
| C0 | random30 | 36.68 ± 1.38 |
| GZ0 | high30 | 36.91 ± 2.32 |
| GZ0 | low30 | 36.24 ± 1.44 |
| GZ0 | random30 | 37.72 ± 2.57 |
| S90 | high30 | 36.82 ± 3.33 |
| S90 | low30 | 37.18 ± 2.90 |
| S90 | random30 | 37.36 ± 1.83 |
| G90 | high30 | 34.24 ± 0.80 |
| G90 | low30 | 36.20 ± 1.10 |
| G90 | random30 | 36.36 ± 0.27 |
| S70 | high30 | 37.84 ± 2.30 |
| S70 | low30 | 37.64 ± 2.15 |
| S70 | random30 | 37.82 ± 1.31 |
| G70 | high30 | 36.30 ± 3.87 |
| G70 | low30 | 36.00 ± 3.04 |
| G70 | random30 | 38.17 ± 1.38 |
| S50 | high30 | 36.13 ± 1.09 |
| S50 | low30 | 35.98 ± 2.65 |
| S50 | random30 | 38.25 ± 0.76 |
| G50 | high30 | 38.23 ± 2.20 |
| G50 | low30 | 36.54 ± 2.11 |
| G50 | random30 | 38.82 ± 1.12 |

These are training-user identity probes, not unseen-user task BA. Neither high30 nor low30 consistently dominates random30; the across-case descriptive residualized means are 36.40%, 36.18%, and 37.65%. Persistent activity is compatible with mixed class/user content, but the proposed **persistent-specific** contextual contamination is not established. Class residualization removes only fitted first-order class means, so residual user decodability does not identify its biological or task meaning.

### 8.5 Fixed-low supervision produces another active route

The fixed-low group starts at **4.55/3.87/3.76 Hz** across seeds. Selected C0 rates are 21.65/22.45/20.99 Hz; C1 rates are 24.76/25.72/23.09 Hz. At the selected C1 checkpoint, fixed-low class η² is higher and user η² lower than C0, but native-head low-column norm is **4.27 versus 4.92**, and mean low-group logit norm is **3.30 versus 3.27**. Higher selectivity does not automatically mean the main head increases its reliance on the group.

Post-hoc group probe results:

| Case | Fixed group | train_ba | val_ba | test_ba |
| --- | --- | --- | --- | --- |
| C0 | high | 91.73 ± 3.79 | 51.45 ± 1.63 | 52.23 ± 3.99 |
| C0 | low | 91.46 ± 1.34 | 51.24 ± 4.06 | 55.21 ± 2.99 |
| C1 | high | 93.08 ± 1.92 | 51.95 ± 4.91 | 54.90 ± 0.86 |
| C1 | low | 88.42 ± 2.93 | 52.72 ± 0.99 | 54.92 ± 4.50 |
| C2 | high | 88.58 ± 3.52 | 53.38 ± 1.19 | 48.73 ± 3.70 |
| C2 | low | 89.92 ± 9.06 | 52.32 ± 3.06 | 53.49 ± 4.82 |
| C3 | high | 93.59 ± 4.93 | 52.14 ± 4.85 | 51.56 ± 4.47 |
| C3 | low | 93.46 ± 0.81 | 54.69 ± 3.39 | 50.78 ± 8.02 |

The primary fixed-low C1−C0 test contrast is **−0.292 ± 1.520 pp**, with seed changes **+1.360, −0.605, −1.631 pp**. This fails the predicted consistent transferable low-route rescue. The fixed-high probe gains 2.666 pp on average, despite the auxiliary head targeting low neurons, indicating distributed adaptation rather than a neatly isolated semantic route.

### 8.6 Exp17.2 falsifies simple history drowning

Epoch20→selected-best paired changes (probability changes shown in pp):

| seed | from_epoch | to_selected_best_epoch | delta_firing_rate_hz | delta_history_sufficient_spike_fraction | delta_incoming_drive_flip_fraction | delta_raw_input_flip_given_active |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 20 | 95 | 11.3731 | 7.92 | 3.156 | 2.5777 |
| 23 | 20 | 96 | 11.9659 | 10.1253 | 2.9752 | 2.454 |
| 37 | 20 | 84 | 9.4823 | 7.1182 | 2.7674 | 2.3197 |

History-sufficient spike fraction rises by 7.12–10.13 pp, but current incoming-drive flip fraction **also rises** by 2.77–3.16 pp, and raw-input influence conditional on active x rises by 2.32–2.58 pp. For the fixed epoch20 low group, incoming-flip changes are +6.37/+6.22/+6.47 pp. The sign contradicts the preregistered conjunction q↑, S_H↑, F_new↓. More history-dependent spiking does not mean loss of current-input control.

Short-horizon replay and relative-time/tau summaries were executed and retained, but they are supporting state-influence diagnostics, not a new classification intervention. No result here establishes that old history prevents useful state updates. Exact replay establishes instrument fidelity, not a causal decomposition of semantic information.

## 9. Mathematical Interpretation

For z=Rq and δ=softmax(z)−onehot(y),

\[
\frac{\partial L}{\partial q_j}=R_{:,j}^{\top}\delta,
\qquad \frac{\partial L}{\partial s_{t,j}}
=\frac1T R_{:,j}^{\top}\delta.
\]

Direct classification credit is shared across valid timesteps. A class-aligned change sustained over many steps can increase the sequence's average evidence, making temporal support useful. However, an extra spike is rewarded only when its head column aligns with the current residual; CE does not universally reward every neuron firing more. Surrogate/state Jacobians and signed interlayer weights determine realized gradients.

An independent group head supplies δ_aux when the main-head residual is small:

\[
L=L_{WC}(Rq,y)+\lambda CE(Aq_S,y).
\]

This is a plausible reason C1 increases group selectivity. Yet the intervention also changes firing and whole-network optimization, so it is not a pure causal test of CE gradient starvation. The necessary task-transfer prediction fails.

Frozen feature replacement changes z by R_S(μ_S−q_S), whereas refitting seeks a new decision rule on q_−S. A frozen BA loss can therefore arise from destroyed variation, altered class offset, or coefficient mismatch. Recovery establishes an available alternative code under the probe protocol; it does not prove restoration of the original evidence or neuron dispensability in hidden dynamics.

## 10. Negative Evidence and Failed Hypotheses

The strongest original closure story does not survive all three controls. High-rate activity expands, but its relative coefficient share declines; user identity is not confined to the persistent subset; pruning specificity is inconsistent. Low-only auxiliary CE increases train selectivity, but neither native nor fixed-low test probes improve consistently, and the group becomes high-rate. Exp17.2's proposed loss of current-input leverage has the wrong longitudinal sign.

These are substantive negative results. They prevent concluding that selectively removing a few persistent neurons, supplying more low-group gradients, or increasing forgetting is already a supported solution to cross-user generalization.

## 11. Interpretation

**Observation.** Firing increases broadly; dynamically lower-rate neurons become more class-selective late; fixed early-low neurons become active; frozen feature losses recover after refitting; input influence does not collapse.

**Mechanistic interpretation.** The architecture supplies slow contextual states, while sequence-level optimization can use repeated communication and recruit changing coordinates. Rate is a property of the learned coding regime, not a semantic label identifying nuisance or transferable neurons.

**Alternatives.** Duration, state gain, class correlations, optimization version, probe scaling/regularization, and redundant class directions can all contribute. Changes in train η² can reflect altered duration-normalized separation without preserving class geometry across users. Native and probe changes need not have one common cause.

**Confidence.** High for measured role/rate migration and the failure of the simple drowning conjunction; moderate for distributed information/readout adaptation; limited for CE starvation as a primary causal mechanism and for high firing as the cause of the gap. Three optimization seeds on one user split do not establish population-level user generalization.

## 12. What This Experiment Established

1. Both high- and low-ranked L2 subsets are class-decodable under matched subset dimensions.
2. Mean-replacement damage and refitting recovery are different endpoints; original-head dependence does not establish unique information.
3. The selected trained population remains mostly >10 Hz after top30 exclusion.
4. Rate identities are dynamic: an early-low group becomes medium/high-rate even under C0.
5. Improving fixed-low train selectivity does not consistently improve its held-out-user probe or the native classifier.
6. Increasing history sufficiency does not entail declining current-input spike influence in this diagnostic.

## 13. What This Experiment Did NOT Establish

It did not prove high-rate activity is intrinsically harmful, show that low rate defines transferability, isolate an irreversible information category, or demonstrate that live high-rate suppression would reproduce feature-column interventions. It did not establish persistent-specific user contamination, unique high-rate class information, sufficient low-group gradient starvation, or a complete causal explanation of the train–test gap. Proposed attenuation/rerouting/spectral-decoupling directions are not executed results here.

## 14. Implications for the Train–Test Gap

Exp17.1 C0 has train/native test BA **93.43/56.82%**, a **36.61 pp** difference. C1 has **91.46/56.27%**, a **35.19 pp** difference; its slightly narrower gap accompanies lower train performance and no test improvement. Reporting gap reduction alone would misstate progress.

The fixed-low C0 probe reaches 55.21% and C1 54.92%, so this matched diagnostic does not expose a rescued transferable subspace that only the native readout fails to use. It constrains a routing-only explanation for **that** intervention. It does not exclude temporal accessibility failures elsewhere.

The conceptual decomposition train→best temporal probe→native is useful, but Exp17's fixed-group WholeCount probes are not a matched temporal-probe upper bound. No exact two-part decomposition is claimed from cross-family approximate 98/68/56% figures. High-rate recruitment can explain how the model forms easily supervised evidence; these experiments do not establish why the associated class geometry transfers poorly.

## 15. Transition to the Next Experiment

High-rate activity is not simply removable nuisance, and forcing early-low neurons to discriminate reconstructs another active route. The unresolved question is **why this operating regime is preferred**. A carrier hypothesis attributes it to slow synaptic dynamics; an objective/readout hypothesis attributes it to temporally supported evidence that is easy for the count head to use. The successor experiment should change the location of the slow state and the count-loss geometry, while preserving data and separating within-version pairing from inherited references.

## 16. Retrospective Interpretation

Later canonical Exp18 places the slow pole in resettable U, then changes count-loss geometry and decomposes rate/selectivity/alignment. Its WCCE U model remains high-rate (26.70 Hz), so slow I alone cannot explain dense communication. That later evidence strengthens the operating-regime interpretation; it is **not** original Exp17 evidence. Exp18.3 also separates mean operating contribution from sample-varying modulation. See [Exp18](../exp18/report.md) for the numerical matrices, initialization confound and limits of causal attribution.

The emerging story is graded rather than assumed:

| Element | Evidence status within Exp17 |
| --- | --- |
| Slow states retain substantial context | Implemented structure; prior history tests as context |
| Discrimination can recruit repeated communication | Direct rate migration; plausible optimization interpretation |
| High firing is a broad coding regime | Directly supported |
| Persistent-specific user contamination causes transfer failure | Unresolved; simple preferential-subset version weakened |
| Low-group gradient rescue improves transfer | Not supported by executed result |
| Native-accessibility limitations exist in every Exp17 intervention | Not established; matched low-probe test fails to show rescue |
| Organizing/routing evidence is a useful next question | Research implication, not solved result |

## 17. Evidence Traceability

Paths below are relative to `/home/zhaolongwei_umass_edu/projects/writingRing/` on Unity. Artifacts remain the numerical authority and are not all published with this report. The reconstruction used 253 exported family JSON/CSV files, additional read-only neuron-distribution/initialization inspections, Git history and scheduler accounting. Historical retrieval returned dated decisions and prior assistant summaries, **not a complete raw conversation export**; no fabricated conversation URL or verbatim quotation is supplied.

| ID / claim | Exact evidence and audit coordinate |
| --- | --- |
| E17-A architecture / trajectory | [implementation](../../scripts/experiment_17_persistent_pathway_story.py), [README](../../scripts/experiment_17_persistent_pathway_story/README.md); `notebooks/artifacts/experiment_17_persistent_pathway_story/persistent_pathway_story_v1/protocol.json` |
| E17-A selected BA / role trajectory | Same root `aggregate/trajectory_metrics.csv`; filter `snapshot_roles` containing `selected_best`, or exact epoch0/20/50/100; neuron correlations/rate tails from `aggregate/trajectory_neurons.csv` |
| E17-B class/user decoding | Same root `aggregate/subset_class_decoding.csv`, `aggregate/subset_user_decoding.csv`; group by case/seed/subset; user mode=`class_residualized`; average CV/masks within seed |
| E17-C frozen/refit | Same root `aggregate/pruning_specificity.csv`; fraction_removed=.30; `test_delta_ba_pp_vs_full` is full minus intervention; random replicate average precedes seed SD |
| E17.1 formulas / grouping | [implementation](../../scripts/experiment_17_1_gradient_starvation.py), [README](../../scripts/experiment_17_1_gradient_starvation/README.md), [probe implementation](../../scripts/analyze_experiment_17_1_group_probes.py) |
| E17.1 calibration / rates / BA | `notebooks/artifacts/experiment_17_1_gradient_starvation/gradient_starvation_v1/calibration.json`; `bootstrap/seed{11,23,37}/groups.json`; `formal/C{0,1,2,3}__seed{11,23,37}/{metrics,history}.json`; `aggregate/summary.json` |
| E17.1 fixed-group transfer | Same root `posthoc_group_probes/aggregate/summary.json`, `paired_vs_C0.C1.low.test_ba`; per-run `probes.json` |
| E17.2 replay / contrasts | [implementation](../../scripts/experiment_17_2_history_input_competition.py), [README](../../scripts/experiment_17_2_history_input_competition/README.md); `notebooks/artifacts/experiment_17_2_history_input_competition/history_input_competition_v1/aggregate/hypothesis_summary.json`; `trajectory_summary.csv`, `epoch20_fixed_group.csv`, `short_horizon_summary.csv`; `seed*/audit.json` |
| Split/cache/version | `core_benchmark_v1/results/main/protocol.lock.json`; [loader version override](../../scripts/experiment_16_prefix_supervised_selective_memory.py); [RNG protocol](../../core_benchmark_v1/protocol.py) |
| Execution outcome | `sacct -u zhaolongwei_umass_edu -S 2026-10-01 -E 2026-10-06 --parsable2 --format=JobID,JobName,State,ExitCode,Start,End`; family job IDs in Section 6 |
| H17-1 original motivation / weakening | Project-history retrieval: user 2026-10-01 22:35:14 UTC; assistant discussion 2026-10-02 00:53:11 UTC |
| H17-2 low-route decision / result interpretation | User implementation authorization 2026-10-02 05:15:37 UTC; discussion 05:25:41/05:33:57, 14:29:03/16:02:06 UTC |
| H17-3 >10Hz / state distinction | Retrieved discussions 2026-10-02 16:08:31/16:15:40/19:34:50/20:11:35/20:18:19 UTC; visible Project context includes user's fast-U/slow-I correction; exact timestamp of that correction was not recovered |
| Retrospective context | [Exp18 report](../exp18/report.md), Section 16 only; later mechanism discussion 2026-10-03 23:55:12 UTC |

Selected artifact SHA256 values allow source-byte verification:

| Artifact | SHA256 |
| --- | --- |
| notebooks/artifacts/experiment_17_persistent_pathway_story/persistent_pathway_story_v1/aggregate/trajectory_metrics.csv | cabacb7dec76d77ef65a3ac2a5818c840035ee5dd15c21b9185014cf02b94ed9 |
| notebooks/artifacts/experiment_17_persistent_pathway_story/persistent_pathway_story_v1/aggregate/pruning_specificity.csv | c8dd425d0b3e5d9e1039e7a094e056de99958dec5a0a216cfb315b09dc465a8f |
| notebooks/artifacts/experiment_17_1_gradient_starvation/gradient_starvation_v1/aggregate/summary.json | 7ea6c3be30909c5c19ea94b5b58f31b265229b601af30f4d532886e484aaacd9 |
| notebooks/artifacts/experiment_17_1_gradient_starvation/gradient_starvation_v1/posthoc_group_probes/aggregate/summary.json | e18f21132bf5c27aed3cc023d5cc43b434891381b4875809ff74ba8fe5420662 |
| notebooks/artifacts/experiment_17_2_history_input_competition/history_input_competition_v1/aggregate/hypothesis_summary.json | dad74a85cabe2964935eda9420d13d7588e46accefc49e70385afaa1717d1dcd |
