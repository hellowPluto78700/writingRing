# CoreBenchmark v1 — A Controlled Protocol for Temporal Representation and Readout Evaluation

Report date: October 2, 2026. Primary numerical scope: the finalized **core_benchmark_v1.0** Unity production run, plus a separately identified **v1.1 Core04 end-to-end LIF extension**. All BA values in prose and Markdown tables are percentages; differences are percentage points (pp). CSV metrics retain the original fractional scale. Unless labeled otherwise, representation results use **spikes and no-bias probes**.

## 1. Executive Summary

CoreBenchmark was created to resolve an experimental comparability problem. Historical SNN experiments varied user splits, optimization seeds, supervision, temporal aggregation, and probe geometry. Their differences could not safely be attributed to a single architectural mechanism. The benchmark introduced one immutable input cache and user split, three optimization seeds, independently controlled objective/tau/depth blocks, common checkpoint selection, and matched temporal and bias/no-bias probes.

The original 33-task production matrix completed: 27 backbone runs and six O0-derived readout tasks. O0 WCCE achieved native train/validation/test BA of **92.97 ± 0.12 / 53.70 ± 1.18 / 57.93 ± 2.23%**. O1 TSCE achieved **53.72 ± 3.92%** native test BA, while its L2 Fixed250-order probe reached **63.74 ± 1.37%**. Thus the best native objective and the representation most useful to a temporal probe were different outcomes. The three-layer D1 achieved **46.86 ± 3.90%** native test BA.

For O0, L1→L2 WholeCount increased from **55.08 to 57.57%**, Fixed250-order increased from **56.28 to 58.62%**, and Fixed250-shuffle increased from **44.86 to 53.79%**. The order–shuffle gap narrowed from **11.42 to 4.83 pp**. This corrects the remembered claim that O0 Fixed250-order itself fell at L2. Relative10-order fell from **71.73 to 67.04%**, yet remained substantially above native performance. The evidence supports changed decoder accessibility and substantial contextualization, but does not uniquely establish abstraction or information loss.

Same-checkpoint accumulator-to-LIF conversion reduced test BA; threshold calibration and W-only adaptation did not recover O0 performance. The later E2E-LIF extension achieved **42.01 ± 2.49%**; replacing its LIF with an accumulator at the same selected checkpoint gave **43.08 ± 1.13%**. This extension used v1.1 initialization streams and must not be treated as a strictly paired contrast against the already executed v1.0 O0 runs.

The benchmark established a reproducible diagnostic foundation. Its unresolved question was how to organize task-relevant history so that a causal, shared, bias-free readout can exploit it across users. Simply adding depth, extending time constants, or changing to timestep supervision did not reliably solve that problem on this split.

## 2. Benchmark Identity and Scope

The canonical repository directory is **`core_benchmark_v1/`**, with README title “Core benchmark v1.” It is an independent measurement framework, explicitly separate from historical Exp7.x–13.x ablations; it is not a numbered Exp0, Exp13, or Exp14 subexperiment. Its blocks are `01_objective`, `02_tau`, `03_depth`, `04_readout`, `05_probes`, and `06_temporal_diagnostics`; `07_membrane` is optional.

The report follows the top-level `reports/` convention already used by `reports/exp0/`, while preserving the canonical benchmark directory name: **`reports/core_benchmark_v1/report.md`**.

The original result root is `core_benchmark_v1/results/main/`, protocol `core_benchmark_v1.0`, source commit `96088fe7725f6d0748aca7027abfe8fc438bad2a`. The extension root is `core_benchmark_v1/results/core04_e2e_lif_v1_1/`, protocol `core_benchmark_v1.1`, source commit `7bc6d6fa6c0caadea0685b95a51fadae3add01f6`. Current README defaults include the later E2E tasks; they do not retrospectively add those tasks to the original v1.0 manifest.

`REF`, `T0`, `D0`, `M0`, and `R0` alias O0, rather than representing additional independent checkpoints. Benchmark D1 denotes the three-layer architecture, not the unrelated preprocessing ablation also named D1.

## 3. Context and Motivation

The September 27 Project discussion explicitly requested a common standard before interpreting historical experiments: one user split, three random seeds, and both affine and no-bias probes. The motivating ambiguity was whether weaker gains from explicit temporal fusion at deeper layers reflected useful history integration, temporal information loss, or incompatible evaluation protocols. The user also questioned describing multi-tau L2 as purely local: its synaptic and membrane states already depend on past inputs.

The immediate research background included Exp7.3 objective/readout comparisons and Exp11–13 investigations of temporal fusion, primitive representations, and layerwise history. Those historical results supplied questions, not interchangeable numerical baselines. In particular, old TSCE could weight long sequences differently, and centered “no-bias” preprocessing could introduce an effective intercept. CoreBenchmark therefore standardized these operations explicitly rather than claiming exact reproduction of every earlier experiment.

The methodological purpose was to separate objective effects, time-constant effects, depth effects, and output realization effects against one anchor. Blocks were not stages in a winner-selection pipeline. A favorable tau configuration was not silently carried into the objective or depth block. Contemporary motivation is supported by the Project excerpts and retrieved context listed in Section 22; original conversation URLs are not available.

## 4. Research Questions

1. Under a common backbone and split, do WCCE, TSCE, and L1 auxiliary losses change native classification or only the information accessible to probes?
2. Does redistributing synaptic time constants improve classification and temporal decoding without changing other factors?
3. Does an added layer improve transferable classification or reduce useful decoder accessibility?
4. How much classification depends on temporal position and relative phase, beyond aggregate activity?
5. Which differences depend on a decoder intercept?
6. What is lost when an analog accumulator is realized as output spikes, and how much can output-only or full-network adaptation recover?
7. Does reduced order-probe advantage mean absent history, or does a common-suffix reset reveal state-dependent evidence?

These are controlled measurement questions. The benchmark did not train a selective memory gate or directly supervise abstract motion primitives.

## 5. Benchmark Contract

| Dimension | Locked choice | Deliberately varied |
| --- | --- | --- |
| Data | D0 action0+action1, one immutable 64 Hz event cache | None within the benchmark |
| Labels/input | 12 classes; first 30 event channels | None |
| Split | One user split, split seed 12345 | Model seeds do not change users |
| Optimization | Adam, lr 0.001, weight decay 0, batch 128 | Training objective in O0–O3 |
| Budget | Max 100 epochs, min 20, patience 30 | Early stopping may yield different actual epochs |
| Backbone | Bias-free feed-forward LIF layers, width 128 | Synaptic shift groups in T1–T4; three layers in D1 |
| Dynamics | Unnormalized input drive; fixed membrane tau 22.54 ms; threshold 0.5 | Synaptic alpha allocation |
| Native anchor | Bias-free analog accumulator | IF/LIF realization in readout controls |
| Selection | Validation native BA, then common mean-logit CE, then earliest epoch | Readout-specific validation scores in spike controls |
| Probe | Train-only scale normalization; validation-only C selection | Layer, state, temporal aggregation, intercept |
| Randomness | Optimization seeds 11/23/37 | Five within-seed shuffle replicates |

Every common parameter name receives its own deterministic initialization stream; adding an auxiliary head or layer does not shift shared parameters within the same protocol version. Loader randomness is likewise paired for compatible backbone cases. Protocol/source/cache/package identities are locked. This pairing is version-specific: the version string participates in `paired_seed`.

## 6. Dataset and Evaluation Protocol

The two producer roots are:

```text
outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
```

The loader requires unsigned events, producer event schema `custom_wavelet_polarity_split_abs_events_v1`, 30 event channels, total channel count 36, sampling rate 64 Hz, and padded horizon 256. It excludes the six appended IMU channels from model input. The selected labels are `A B C D E G H I J K L X`, not all 26 alphabet classes. Inputs preserve valid events and are exactly zero beyond valid length. Hidden evolution, loss, scores, and probes do not accumulate padded-tail evidence.

Users are lexicographically sorted, shuffled once using NumPy split seed 12345, then assigned floor(70%) training, floor(15%) validation, and the remainder test. The concrete 20-user assignment is:

| Split | Users | Samples | Valid length min–max | Mean valid steps |
| --- | --- | ---: | --- | ---: |
| Train | user_0, user_1, user_11, user_12, user_13, user_14, user_15, user_18, user_19, user_2, user_20, user_5, user_7, user_8 | 581 | 46–247 | 102.77 |
| Validation | user_16, user_4, user_9 | 126 | 49–171 | 90.40 |
| Test | user_10, user_3, user_6 | 146 | 44–131 | 82.68 |

All three splits contain all 12 classes. `sample_counts.csv` (Unity artifact; see Section 22) retain user/class coverage. The immutable cache SHA256 is `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a` in both result roots.

Balanced accuracy is mean per-class recall over classes present in the evaluated cohort. Accuracy and macro-F1 are also saved. Main test BA pools the held-out-user samples before computing class recalls; it is not a mean of per-user BA. Per-user metrics are retained in `native_per_user.csv` (Unity artifact; see Section 22). Training metrics are in-sample measurements on training users; validation and test users are both unseen during optimization. There is no independent held-out-segment evaluation of seen users in the main benchmark, so “seen-user performance” must not be interpreted as an independent generalization estimate.

All mean ± SD values summarize three optimization seeds on **one** user split. SD uses the sample convention, ddof=1. Five shuffle replicates are averaged inside each optimization seed before calculating mean/SD across seeds. These are not 15 independent models or three independent user splits.

## 7. Model and SNN Backbone

For layer l, with z^0_t=x_t, the implemented dynamics are:

$$I^l_t=\alpha_l\odot I^l_{t-1}+W_lz^{l-1}_t,$$
$$V^{l,pre}_t=\beta_l V^l_{t-1}+I^l_t,\quad z^l_t=\mathbf1[V^{l,pre}_t\geq\theta],$$
$$V^l_t=V^{l,pre}_t-\theta z^l_t.$$

There is **no (1−alpha) multiplier** on input drive. State starts at zero per sequence. Spike output is binary, capped at one per timestep; reset is immediate and subtractive; membrane remains signed with no zero clamp. There are no learned recurrent matrices: recurrence is through neuron state. The normalized fast-sigmoid surrogate derivative is `1 / [theta (1+slope |(Vpre−theta)/theta|)^2]`, slope 25.

The anchor is 30→128 L1(234)→128 L2(234)→12 bias-free analog outputs. Shift s gives alpha=1−2^(−s), and tau_syn=−15.625/log(alpha) ms. Groups distribute 128 neurons as 43/43/42. The membrane decay is beta=exp(−15.625/22.54), approximately 0.5, independently of synaptic groups.

| Shift | Alpha | Effective synaptic tau (ms) |
| --- | ---: | ---: |
| 1 | 0.5 | 22.54 |
| 2 | 0.75 | 54.31 |
| 3 | 0.875 | 117.01 |
| 4 | 0.9375 | 242.10 |
| 5 | 0.96875 | 492.16 |

Thus the locked 234 synaptic group uses **242.10 ms**, not a nominal 249/250 ms. The 250 ms probe width is a separate operation. Longer alpha also increases DC gain under the unnormalized update; tau interventions cannot be interpreted as changing memory horizon alone. `activity.csv` (Unity artifact; see Section 22) provide firing, silence, and pre-reset magnitude diagnostics.

The native readout computes e_t=R z_t and A_t=A_(t−1)+e_t, with no bias, leak, reset, or threshold. Classification is argmax A_T. L1/L2/L3 denote consecutive hidden layers, not independently trained networks within one run.

## 8. Training Objectives

For sequence i of valid length T_i, define e_it=R z_it from the final hidden layer and a_it=Q z^1_it from a separate L1 auxiliary head. The implemented objectives are:

$$L_{WC}(e)=\frac1N\sum_i CE\!\left(\frac1{T_i}\sum_{t<T_i}e_{it},y_i\right),$$
$$L_{TS}(e)=\frac1N\sum_i\frac1{T_i}\sum_{t<T_i}CE(e_{it},y_i).$$

| Case | Loss | Supervision | Scientific question |
| --- | --- | --- | --- |
| O0 | L_WC(e) | Final-layer bias-free accumulator evidence | Common sequence-level anchor |
| O1 | L_TS(e) | Final-layer instantaneous evidence | Does sample-balanced timestep supervision change representation/readout tradeoffs? |
| O2 | L_WC(e)+0.1 L_WC(a) | Main plus separate L1 sequence head | Does intermediate sequence supervision preserve useful L1 structure and improve final classification? |
| O3 | L_WC(e)+0.1 L_TS(a) | Main plus separate L1 timestep head | Does locally supervised L1 improve downstream sequence classification? |

Definitions reside in `model.sequence_loss`, `model.objective_loss`, and `protocol.runs`. The auxiliary head is training-only; its logits are not fused into native evaluation. WCCE averages logits before CE. Summed and averaged logits share argmax for this head, but their CE and gradients differ. TSCE averages valid timesteps **within each sample**, then samples; unlike historical flattened-timestep TSCE, it does not weight longer gestures more heavily.

Tau cases all use WCCE: T1=123/234, T2=123/123, T3=123/345, T4=234/345. D1 uses WCCE and 234/234/234. Optional D2 freezes O0 L1/L2 while fitting L3 and a new head; D3 unfreezes that D2 network. Optional M1/M2/M3 change L1 membrane tau. These controls exist in code but were disabled in the executed original lock and are **not executed in this report's result matrix**.

## 9. Diagnostic Probes

For any cached state z_it, WholeCount forms f_i=sum_(t<T_i) z_it. Fixed250 forms sums in 16-step (250 ms) absolute bins across the 256-step horizon, concatenating 16 width-128 vectors: 2,048 dimensions. Relative10 divides each valid sequence using integer boundaries from `linspace(0,T_i,11)` and concatenates ten bin sums: 1,280 dimensions. WholeCount is 128-dimensional. These are sums, not per-bin means.

The classifier is q_i=sum_b U_b f_ib+b, where b=0 for no-bias and b is fitted for affine. Temporal bins have distinct classifier columns/weight matrices; training is joint multiclass logistic regression on their concatenation. This differs from fitting independent per-bin classifiers and averaging decisions.

| Probe | Temporal support | Diagnostic meaning | Deployment relation |
| --- | --- | --- | --- |
| WholeCount | Sum over valid sequence | Information accessible after temporal pooling | No-bias scaling can be folded into a shared linear mapping; not proof of identical native training |
| Fixed250 order | Absolute bin positions preserved | Information accessible with explicit absolute-time resolution | Requires timed position-dependent mapping |
| Fixed250 shuffle | Complete valid bins independently permuted per sample | Benefit of consistent full-bin position/alignment | Diagnostic control |
| Relative10 order | Ten duration-normalized positions | Information accessible with explicit relative-phase alignment | Offline; requires final valid length |
| Relative10 shuffle | All ten normalized bins permuted per sample | Benefit of normalized-bin alignment | Diagnostic control |
| pre_reset state | Signed membrane before reset | Secondary analog-state accessibility | Not a spike-only deployable feature |

Fixed250 shuffling holds the final partial bin and padding bins in their original positions. It does not reorder input timesteps or rerun the backbone. Each sample gets its own deterministic permutation based on sample ID and shuffle seed. The same sample/replicate permutation is used across layers and states within a version. Every shuffled representation gets a **new fitted decoder**, with C selected on its shuffled validation representation; this is not applying the ordered decoder to corrupted test features. A global permutation would merely rename feature columns and would not test the intended hypothesis. All samples here have at least two full bins; partial-bin anchoring and duration cues remain.

Both geometries use train-fitted `StandardScaler(with_mean=False, with_std=True)`. No centering is permitted, because centering followed by a bias-free map introduces an effective offset. The sole geometric change is `fit_intercept`. Logistic regression uses lbfgs, max_iter=3000, tol=1e−4, C in {0.001,0.01,0.1,1,10,100}; validation BA chooses C, with ties retaining smaller C. Nonconverged candidates are excluded. No train+validation refit precedes testing.

Order–shuffle differences measure decoder-accessible alignment under these operations. They mix position-specific class evidence, residual temporal anchoring, fitting/regularization effects, and representation structure. They are neither mutual information nor a direct assay of abstract semantics. Relative10 is a useful reference, not a guaranteed upper bound.

## 10. Checkpoint Selection

Adam updates parameters to minimize the training objective. BA is not differentiated. At each epoch, a separate validation pass evaluates native accumulator BA and common valid-mean-logit CE. Checkpoints maximize validation BA; equal BA is broken by lower validation CE, then earliest epoch. Epoch zero is eligible. Early stopping requires at least 20 epochs and 30 epochs without a selected-checkpoint improvement.

This distinction resolves the question whether training “optimizes test BA”: **optimization uses train loss; checkpoint selection uses validation metrics; final test BA is evaluation only**. The backbone selection code has no probe or test input. Even O1 uses common validation accumulator metrics for checkpoint selection, not its timestep objective value alone.

R2 selects by validation spike-count BA, then spike-count CE, using summed output spikes. R3 selects by validation LIF BA, then mean-output-spike CE. Their loss scales are not identical: original R2 trains CE of spike counts; later R3 trains CE of valid mean output spikes. This is an additional reason not to interpret their difference as only a hidden-network-gradient intervention.

## 11. Execution History

The original production preparation job **64970891**, phase-1 array **64970892_0–26**, phase-2 array **64970893_0–5**, and finalizer **64970894** were all `COMPLETED`, exit `0:0`. Array-task provenance often records Slurm's individual internal job ID rather than `array_job_task`; the inventory below preserves both individual ID and task index. These are not evidence of reruns.

The original manifest reports production profile, **33 expected/33 completed tasks**, **27 backbone runs**, **6 readout tasks**, **2,964 raw probe rows**, and **PASS**. Shuffle averaging reduces those raw probes to **1,140 model-seed coordinates** and **380 aggregate coordinates**.

The later preparation job **64981383** completed. Only E2E tasks **64981384_9, _19, _29** were executed in its v1.1 result root, all `COMPLETED`, exit `0:0`. It has per-run completion records but no full aggregate manifest because the complete v1.1 default matrix was not run there. R4 is a second evaluation of each R3 checkpoint, not three additional training tasks.

A launch compatibility fix, commit `746c465b4c1e91813e469978eecf4bc3ba1db8cf`, temporarily disabled shell nounset while sourcing Unity `/etc/profile`. It changes environment setup, not neuron equations or loss. The valid original source commit includes the fix. Exact failed pre-fix job IDs and logs are **not verified from available artifacts**; no hypothetical earlier failure is included in aggregates. No invalid or superseded numerical run was found among the completed runs used here.

Inventory source: per-run `provenance.json`, `history.json`, completion records, and Slurm accounting. “Best” denotes selected epoch; readout R2 may select epoch zero. Every listed run is valid in its own version scope.

| Root | Run | Individual job ID | Array task | Status | Best / stopped epoch | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| main | D1__seed11 | 64970922 | 8 | COMPLETED | 81 / 100 | checkpoint.pt |
| main | D1__seed23 | 64970931 | 17 | COMPLETED | 89 / 100 | checkpoint.pt |
| main | D1__seed37 | 64970892 | 26 | COMPLETED | 46 / 76 | checkpoint.pt |
| main | O0__seed11 | 64970914 | 0 | COMPLETED | 93 / 100 | checkpoint.pt |
| main | O0__seed23 | 64970923 | 9 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | O0__seed37 | 64970932 | 18 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | O1__seed11 | 64970915 | 1 | COMPLETED | 88 / 100 | checkpoint.pt |
| main | O1__seed23 | 64970924 | 10 | COMPLETED | 94 / 100 | checkpoint.pt |
| main | O1__seed37 | 64970933 | 19 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | O2__seed11 | 64970916 | 2 | COMPLETED | 96 / 100 | checkpoint.pt |
| main | O2__seed23 | 64970925 | 11 | COMPLETED | 100 / 100 | checkpoint.pt |
| main | O2__seed37 | 64970934 | 20 | COMPLETED | 91 / 100 | checkpoint.pt |
| main | O3__seed11 | 64970917 | 3 | COMPLETED | 97 / 100 | checkpoint.pt |
| main | O3__seed23 | 64970926 | 12 | COMPLETED | 93 / 100 | checkpoint.pt |
| main | O3__seed37 | 64970935 | 21 | COMPLETED | 100 / 100 | checkpoint.pt |
| main | R_IF__seed11 | 64972050 | 0 | COMPLETED | 3 / 33 | head.pt |
| main | R_IF__seed23 | 64972072 | 2 | COMPLETED | 35 / 65 | head.pt |
| main | R_IF__seed37 | 64972078 | 4 | COMPLETED | 0 / 30 | head.pt |
| main | R_LIF__seed11 | 64972065 | 1 | COMPLETED | 57 / 87 | head.pt |
| main | R_LIF__seed23 | 64972073 | 3 | COMPLETED | 15 / 45 | head.pt |
| main | R_LIF__seed37 | 64970893 | 5 | COMPLETED | 1 / 31 | head.pt |
| main | T1__seed11 | 64970918 | 4 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | T1__seed23 | 64970927 | 13 | COMPLETED | 96 / 100 | checkpoint.pt |
| main | T1__seed37 | 64970936 | 22 | COMPLETED | 100 / 100 | checkpoint.pt |
| main | T2__seed11 | 64970919 | 5 | COMPLETED | 98 / 100 | checkpoint.pt |
| main | T2__seed23 | 64970928 | 14 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | T2__seed37 | 64970937 | 23 | COMPLETED | 100 / 100 | checkpoint.pt |
| main | T3__seed11 | 64970920 | 6 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | T3__seed23 | 64970929 | 15 | COMPLETED | 56 / 86 | checkpoint.pt |
| main | T3__seed37 | 64970938 | 24 | COMPLETED | 98 / 100 | checkpoint.pt |
| main | T4__seed11 | 64970921 | 7 | COMPLETED | 99 / 100 | checkpoint.pt |
| main | T4__seed23 | 64970930 | 16 | COMPLETED | 76 / 100 | checkpoint.pt |
| main | T4__seed37 | 64970939 | 25 | COMPLETED | 95 / 100 | checkpoint.pt |
| core04_e2e_lif_v1_1 | R_LIF_E2E__seed11 | 64981400 | 9 | COMPLETED | 98 / 100 | checkpoint.pt |
| core04_e2e_lif_v1_1 | R_LIF_E2E__seed23 | 64981401 | 19 | COMPLETED | 82 / 100 | checkpoint.pt |
| core04_e2e_lif_v1_1 | R_LIF_E2E__seed37 | 64981384 | 29 | COMPLETED | 65 / 95 | checkpoint.pt |

Result files are `<root>/runs/<run>/native.json` and `probes.json` for backbones, or `readout.json` for readout tasks. The full provenance table is `run_inventory.csv` (Unity artifact; see Section 22). No expensive experiment was rerun for this report.

## 12. Main Benchmark Results

### Native performance

Source: `main/aggregate/native_runs.csv`, cross-checked against all 27 `native.json` files. Mean ± sample SD over three seeds, percentages.

| Case | Train BA | Validation BA | Test BA | Train−test gap (pp) |
| --- | --- | --- | --- | --- |
| O0 | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 | 35.04 ± 2.35 |
| O1 | 89.34 ± 1.41 | 50.15 ± 1.97 | 53.72 ± 3.92 | 35.62 ± 4.91 |
| O2 | 92.78 ± 0.67 | 55.37 ± 3.10 | 57.23 ± 3.03 | 35.55 ± 3.42 |
| O3 | 93.45 ± 1.14 | 54.90 ± 3.85 | 57.34 ± 0.97 | 36.11 ± 0.74 |
| T1 | 89.09 ± 2.34 | 51.25 ± 0.47 | 56.22 ± 1.92 | 32.87 ± 3.08 |
| T2 | 84.14 ± 1.03 | 52.46 ± 3.01 | 58.17 ± 2.12 | 25.97 ± 2.57 |
| T3 | 84.47 ± 12.37 | 50.69 ± 3.10 | 52.93 ± 1.98 | 31.54 ± 13.10 |
| T4 | 93.19 ± 3.00 | 52.52 ± 1.92 | 54.83 ± 4.95 | 38.36 ± 5.22 |
| D1 | 90.79 ± 9.59 | 49.83 ± 3.53 | 46.86 ± 3.90 | 43.93 ± 7.35 |

### Every native run

These values describe the selected checkpoints, not last-epoch models.

| Case | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- |
| O0 | 11 | 93.08 | 55.05 | 55.97 |
| O1 | 11 | 90.86 | 49.73 | 49.84 |
| O2 | 11 | 93.46 | 58.91 | 54.25 |
| O3 | 11 | 94.21 | 59.23 | 58.42 |
| T1 | 11 | 90.98 | 51.53 | 57.74 |
| T2 | 11 | 85.29 | 54.90 | 56.99 |
| T3 | 11 | 92.55 | 54.23 | 50.75 |
| T4 | 11 | 94.97 | 54.27 | 50.65 |
| D1 | 11 | 95.52 | 51.10 | 45.37 |
| O0 | 23 | 93.00 | 52.82 | 57.46 |
| O1 | 23 | 88.07 | 52.29 | 53.64 |
| O2 | 23 | 92.75 | 53.14 | 60.31 |
| O3 | 23 | 92.14 | 53.57 | 56.56 |
| T1 | 23 | 86.48 | 51.50 | 56.85 |
| T2 | 23 | 83.82 | 53.39 | 60.62 |
| T3 | 23 | 70.23 | 49.35 | 53.45 |
| T4 | 23 | 89.72 | 52.82 | 53.53 |
| D1 | 23 | 97.10 | 52.56 | 51.29 |
| O0 | 37 | 92.84 | 53.25 | 60.36 |
| O1 | 37 | 89.10 | 48.42 | 57.68 |
| O2 | 37 | 92.13 | 54.05 | 57.12 |
| O3 | 37 | 93.99 | 51.89 | 57.04 |
| T1 | 37 | 89.80 | 50.70 | 54.06 |
| T2 | 37 | 83.31 | 49.09 | 56.91 |
| T3 | 37 | 90.64 | 48.49 | 54.60 |
| T4 | 37 | 94.87 | 50.47 | 60.30 |
| D1 | 37 | 79.75 | 45.84 | 43.92 |

### Primary layer/probe matrix

Spikes, no-bias, test BA. Every objective, tau, and depth case is included; shuffled values first average five permutation replicates inside each seed.

| Case | Layer | WholeCount | Fixed250 order | Fixed250 shuffle | Relative10 order | Relative10 shuffle |
| --- | --- | --- | --- | --- | --- | --- |
| O0 | L1 | 55.08 ± 2.23 | 56.28 ± 1.47 | 44.86 ± 2.31 | 71.73 ± 1.82 | 35.50 ± 0.71 |
| O0 | L2 | 57.57 ± 4.24 | 58.62 ± 2.69 | 53.79 ± 3.04 | 67.04 ± 0.73 | 54.24 ± 3.66 |
| O1 | L1 | 55.25 ± 3.77 | 55.70 ± 1.14 | 45.20 ± 2.51 | 70.74 ± 2.36 | 37.09 ± 2.33 |
| O1 | L2 | 58.55 ± 4.30 | 63.74 ± 1.37 | 55.98 ± 0.83 | 69.79 ± 7.34 | 52.92 ± 2.06 |
| O2 | L1 | 59.62 ± 1.26 | 56.26 ± 1.32 | 46.10 ± 3.12 | 72.22 ± 0.81 | 38.75 ± 1.81 |
| O2 | L2 | 54.06 ± 4.24 | 57.96 ± 1.15 | 51.57 ± 1.12 | 67.79 ± 2.20 | 52.36 ± 1.10 |
| O3 | L1 | 57.17 ± 3.17 | 57.93 ± 0.46 | 44.77 ± 2.69 | 71.47 ± 1.36 | 38.17 ± 1.91 |
| O3 | L2 | 58.48 ± 1.35 | 58.03 ± 3.07 | 53.87 ± 2.37 | 69.78 ± 2.66 | 52.59 ± 0.68 |
| T1 | L1 | 48.75 ± 0.34 | 53.90 ± 1.28 | 40.16 ± 1.21 | 71.86 ± 1.94 | 30.83 ± 0.33 |
| T1 | L2 | 58.29 ± 0.45 | 61.48 ± 4.36 | 52.84 ± 1.38 | 70.74 ± 1.00 | 51.59 ± 1.26 |
| T2 | L1 | 50.29 ± 6.19 | 53.11 ± 3.37 | 40.56 ± 0.94 | 70.86 ± 0.22 | 33.44 ± 0.39 |
| T2 | L2 | 56.72 ± 1.70 | 60.29 ± 5.31 | 53.10 ± 2.15 | 71.12 ± 2.35 | 50.54 ± 1.08 |
| T3 | L1 | 46.13 ± 1.18 | 50.12 ± 1.95 | 39.40 ± 1.99 | 69.57 ± 1.46 | 30.87 ± 1.81 |
| T3 | L2 | 54.48 ± 2.20 | 58.25 ± 3.22 | 51.20 ± 0.96 | 68.14 ± 3.65 | 48.97 ± 3.67 |
| T4 | L1 | 53.61 ± 3.22 | 52.83 ± 3.36 | 44.80 ± 1.48 | 69.64 ± 1.32 | 35.92 ± 1.30 |
| T4 | L2 | 55.13 ± 2.30 | 59.73 ± 3.75 | 52.41 ± 3.92 | 66.08 ± 3.16 | 54.23 ± 2.80 |
| D1 | L1 | 43.85 ± 1.08 | 46.90 ± 2.99 | 35.32 ± 1.12 | 65.46 ± 1.97 | 26.20 ± 1.67 |
| D1 | L2 | 50.70 ± 6.31 | 54.33 ± 1.86 | 45.72 ± 3.24 | 65.48 ± 1.55 | 42.29 ± 4.15 |
| D1 | L3 | 48.08 ± 0.59 | 47.99 ± 2.37 | 44.94 ± 2.77 | 55.60 ± 2.81 | 46.42 ± 3.59 |

The complete matrix includes both states, both decoder geometries, all layers, all five aggregations, and train/validation/test results. Appendix A contains all 380 aggregate coordinates. `probe_seed_means.csv` (Unity artifact; see Section 22) contains every one of the 1,140 per-seed coordinates; `probe_summary.csv` (Unity artifact; see Section 22) retains exact values. Neither native and probe scores nor v1.0 and v1.1 outcomes are pooled into a single aggregate.

## 13. Layer-Wise Representation Analysis

### O0 WCCE

WholeCount rises 55.08→57.57%, Fixed250-order rises 56.28→58.62%, and Fixed250-shuffle rises 44.86→53.79%. The smaller order gap is driven principally by the shuffle improvement, not declining ordered performance. The Fixed250 gain over WholeCount remains small: 1.20 pp at L1 and 1.04 pp at L2. Relative10-order decreases 71.73→67.04%, while Relative10-shuffle increases 35.50→54.24%. Its relative-phase gap decreases 36.24→12.79 pp.

This combination is consistent with context becoming encoded into features that remain useful after external bin permutations. It also indicates residual phase-dependent evidence that a shared native map does not exploit. It does not show that L2 has become a stable, user-independent stroke representation. Some alignment-sensitive information can weaken while pooled classification improves.

### O1 TSCE

WholeCount increases 55.25→58.55%, and Fixed250-order increases 55.70→63.74%; its L2 gain over WholeCount is 5.19 pp. The Fixed250 gap narrows from 10.50 to 7.75 pp, less strongly than under O0. Relative10-order is 70.74% at L1 and 69.79% at L2, with larger seed variability at L2 (SD 7.34 pp). Despite these favorable probe scores, native test BA is lower than O0. The probe's independently fitted decision geometry and temporal structure cannot be equated with the trained shared head.

### O2 and O3

O2 L1 WholeCount is 59.62%, above O0's 55.08%; however, its L2 WholeCount is 54.06%, below both its L1 and O0 L2. Auxiliary improvement at the supervised layer therefore does not guarantee downstream improvement. O3 L2 WholeCount is 58.48%, while Fixed250-order is 58.03%; explicitly resolved absolute time adds no aggregate advantage there. Both auxiliary cases remain near O0 native test BA rather than clearly improving it.

### D1 and the added third layer

Within D1, WholeCount rises 43.85→50.70% from L1 to L2, then falls to 48.08% at L3. Fixed250-order follows 46.90→54.33→47.99%; Relative10-order follows 65.46→65.48→55.60%. The order–shuffle gap narrows 11.58→8.61→3.05 pp, while the final representation and native classifier weaken. A narrowed gap is therefore not sufficient evidence of useful abstraction. D1 L1/L2 are jointly trained components of a three-layer network and must not be substituted for O0's identically named layers.

### Stability and causal history diagnostics

For O0 test spikes, correlation at one-step lag is 0.293 in L1 and 0.699 in L2; at 250 ms it is 0.385 versus 0.343, and at 1,000 ms it is 0.274 versus −0.062. L2 is locally smoother under this metric but less correlated at longer lags. Cosine and correlation differ because baseline activity affects cosine; both are preserved in `lag_similarity.csv` (Unity artifact; see Section 22). Lag curves are descriptive and cannot distinguish changing input from changing internal history.

The causal diagnostic resets **both** synaptic and membrane state immediately before an identical final input suffix, separately in L1, L2, or all layers. It evaluates both full-sequence evidence, which retains the prefix accumulator, and suffix-only evidence. All-layer reset over a 250 ms suffix reduces O0 test suffix BA from **35.21 to 13.05%**, a **22.17 pp** drop, and full-sequence BA by **8.24 pp**. At 50 ms the suffix drop is 21.24 pp, while full-sequence drop is only 1.04 pp because prefix evidence remains. Resetting L1 alone can improve suffix BA; history effects are not uniformly beneficial.

The 50/100/250/500 ms contrasts include all 146 test samples; the 1,000 ms contrast includes 111. Actual horizons are rounded to 3/6/16/32/64 steps, or 46.875/93.75/250/500/1,000 ms. Comparisons use identical eligible cohorts and report class coverage. The effect is endpoint state dependence, not a measurement of sliding-window memory capacity or a reset intervention on lag similarity. Full per-seed intervention coordinates are in `history_reset.csv` (Unity artifact; see Section 22).

## 14. Objective Comparison

O1's native test BA is 4.21 pp below O0, while its L2 no-bias WholeCount is 0.98 pp higher and Fixed250-order is 5.12 pp higher. Timestep supervision can therefore improve temporal-probe accessibility without improving the deployed classifier. It should not be dismissed as learning no useful information, nor selected as a better native objective from its best probe alone.

O2/O3 improve mean validation BA relative to O0 (55.37/54.90 versus 53.70%), but their test means are 57.23/57.34 versus 57.93%. The benchmark does not demonstrate an auxiliary-loss OOD gain. Their training BA remains around 93%, indicating a large in-sample/held-out-user gap. Three seeds are insufficient for sweeping claims of equality or superiority.

The tau block also resists a monotonic memory-horizon story. T2 123/123 reaches 58.17 ± 2.12% native test BA, only 0.24 pp above O0; T1, T3, and T4 achieve 56.22, 52.93, and 54.83%. The numerically best T2 is not a demonstrated robust winner. Longer time constants change signal gain, firing, and temporal mixtures simultaneously; the experiment does not identify a pure memory-length optimum. Blocks were analyzed against O0 independently rather than composing selected factors.

### Bias/no-bias effects

The benchmark has no implicit unlabeled geometry; this report's primary convention is spike/no-bias, and affine results are explicitly marked. For O0 spikes, affine−no-bias test gains vary by layer and aggregation, as shown below. A remembered universal “about six percentage points” improvement is not supported by this matrix.

| O0 layer | Aggregation | Affine−no-bias test BA (pp) |
| --- | --- | --- |
| L1 | whole_count | +4.67 |
| L1 | fixed250_ordered | -0.12 |
| L1 | fixed250_shuffled | +0.04 |
| L1 | relative10_ordered | -3.59 |
| L1 | relative10_shuffled | +4.48 |
| L2 | whole_count | +2.26 |
| L2 | fixed250_ordered | -2.08 |
| L2 | fixed250_shuffled | -1.66 |
| L2 | relative10_ordered | +0.45 |
| L2 | relative10_shuffled | +1.34 |

No-bias feature scaling can be folded into linear weights without a constant-input source. An affine intercept requires an added implementation mechanism, so its accessibility result is diagnostic rather than direct proof of compatibility with the existing bias-free accumulator. Even no-bias bin-dependent and analog-state probes require additional temporal routing or access beyond the native spike head.

## 15. Readout and Output-Layer Controls

### Original v1.0 O0-derived controls

All modes share the exact O0 L2 spike cache and parent checkpoint per seed. Output synaptic alpha=0; IF beta=1 and LIF beta=0.5; signed membrane, binary cap one, immediate subtractive reset. R0 is duplicated in the IF/LIF views but is one anchor, not six independently trained baselines.

| Output | Mode | Train BA | Validation BA | Test BA | Test ties (%) |
| --- | --- | --- | --- | --- | --- |
| R_IF | R0_accumulator | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 | 0.00 ± 0.00 |
| R_IF | R1_sameW_fixed | 53.21 ± 3.27 | 30.63 ± 0.81 | 28.15 ± 1.58 | 21.69 ± 4.56 |
| R_IF | R1_sameW_calibrated | 69.52 ± 6.41 | 37.68 ± 0.95 | 35.96 ± 2.74 | 10.73 ± 2.20 |
| R_IF | R2_adaptW | 76.13 ± 7.31 | 41.76 ± 3.08 | 38.07 ± 5.44 | 8.68 ± 4.81 |
| R_LIF | R0_accumulator | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 | 0.00 ± 0.00 |
| R_LIF | R1_sameW_fixed | 84.94 ± 2.34 | 47.72 ± 2.28 | 51.48 ± 3.53 | 4.79 ± 2.47 |
| R_LIF | R1_sameW_calibrated | 85.94 ± 3.17 | 49.89 ± 1.62 | 52.55 ± 1.06 | 4.34 ± 2.85 |
| R_LIF | R2_adaptW | 89.39 ± 3.68 | 51.97 ± 2.43 | 51.30 ± 1.83 | 6.16 ± 2.37 |

R1 fixed keeps W and threshold 0.5. R1 calibrated keeps W but chooses threshold from {0.125,0.25,0.5,1,2} using validation BA, preferring the native/nearest scale in a tie. R2 initializes W from O0 and trains only W on cached spikes at the calibrated threshold; the backbone cannot receive gradients. All-zero outputs and ties are retained in `readout_actual.csv` (Unity artifact; see Section 22); argmax ties consistently select the first class index.

LIF fixed/calibrated/adapted test BA is 51.48/52.55/51.30%, below the 57.93% analog anchor. W-only training improves validation over calibration (51.97 versus 49.89%) but not test (51.30 versus 52.55%). IF is substantially worse. These controlled outcomes establish a readout-realization cost for these checkpoints and dynamics. They do not establish a fundamental impossibility of spike readout, or uniquely attribute the cost to signed negative evidence rather than leak, clipping, quantization, calibration, or optimization.

### Post-CoreBenchmark extension: v1.1 E2E-LIF and accumulator swap

R3 trains L1, L2, and W jointly through an output LIF, beta=0.5, threshold=0.5, alpha_out=0, with CE on valid mean output spikes. R4 evaluates its selected checkpoint through the analog accumulator with identical hidden layers and W, without retraining or choosing a new checkpoint. Per-seed and aggregate results follow.

| Mode | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- |
| R3_e2e_LIF | 11 | 82.20 | 43.56 | 42.17 |
| R4_e2e_accumulator_swap | 11 | 79.45 | 37.01 | 44.06 |
| R3_e2e_LIF | 23 | 74.55 | 42.77 | 39.44 |
| R4_e2e_accumulator_swap | 23 | 71.34 | 43.82 | 41.84 |
| R3_e2e_LIF | 37 | 70.02 | 45.51 | 44.42 |
| R4_e2e_accumulator_swap | 37 | 66.99 | 42.01 | 43.34 |

| Mode | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- |
| R3_e2e_LIF | 75.59 ± 6.15 | 43.95 ± 1.41 | 42.01 ± 2.49 |
| R4_e2e_accumulator_swap | 72.59 ± 6.33 | 40.95 ± 3.53 | 43.08 ± 1.13 |

R4−R3 test changes by seed 11/23/37 are +1.89, +2.40, -1.08 pp; the paired mean is 1.07 ± 1.88 pp. Two seeds improve and one declines. A residual realization gap is modest and inconsistent after E2E adaptation; it does not explain the entire performance deficit.

The README says R3 uses paired O0 initialization, and its code verifies pairing with a newly instantiated O0 **inside v1.1**. However, `paired_seed` hashes `[VERSION,seed,role]`. The executed O0 checkpoint is v1.0, while R3 is v1.1, so initialization and loader RNG streams differ despite equal user-facing seeds. Data-cache hashes match exactly. Therefore O0-v1.0 versus R3-v1.1 is a descriptive comparison, not an exact paired readout-only intervention. R3→R4 remains the clean same-checkpoint contrast. A matched v1.1 O0 retraining would be needed to isolate the E2E training treatment, but was not performed for this report.

## 16. Negative Evidence

The benchmark preserves several results that weaken simple mechanisms. TSCE does not beat WCCE in native test BA despite a stronger temporal probe. L1 auxiliary losses do not yield a clear final OOD gain. Greater depth worsens native performance and loses Relative10 accessibility at L3. Longer synaptic traces do not produce monotonically better classification. Smaller order gaps coexist with both improving O0 and worsening D1 representations. Affine probes sometimes decrease held-out BA. Threshold calibration and W-only output adaptation do not recover the analog anchor. End-to-end LIF training in the attached extension also performs poorly, and the same-checkpoint accumulator swap only modestly changes test BA.

Most backbone cases reach the 100-epoch cap, and several best checkpoints are near its end. This benchmark therefore does not rule out a training-budget effect in every case. It nevertheless reports the actual common budget rather than assuming all models converged to an unrestricted optimum. D1 seed37 stops at epoch76 after selecting epoch46; T3 seed23 stops at86 after selecting56. These differences and D1's large training-seed variability remain visible.

## 17. Interpretation

| Claim | Observation | Interpretation | Alternative explanation | Evidence strength |
| --- | --- | --- | --- | --- |
| L2 reduces reliance on explicit bin alignment | O0 shuffle gains more than ordered; gaps shrink | Context is redistributed into pooled features | Alignment-sensitive detail can weaken; decoder fitting and duration cues also matter | Strong descriptive result; nonunique mechanism |
| History is used | Identical suffix, changed hidden state, altered scores/BA | Predictions depend causally on earlier state | Reset is an off-trajectory intervention and does not identify what information is stored | Strong state-dependence evidence; limited semantic attribution |
| L2 is abstract | Local smoothing but weaker long-lag correlation; Relative10 still useful | No simple stable-abstraction signature is established | Abstract features may change with primitives or be nonlinear | Benchmark cannot resolve the broad abstraction claim |
| TSCE is better | Higher L2 temporal probe, lower native test BA | Better accessibility for some decoders, weaker native outcome | Head optimization, loss scale and selection geometry can explain differences | Direct tradeoff; no universal ranking |
| Output spikes are the sole bottleneck | O0 conversion loses BA; E2E swap gives small mixed gain | Readout realization matters but is insufficient as sole explanation | Backbone learning and version-specific initialization differ | Strong within-O0 and R3/R4 controls; cross-version attribution limited |
| Longer memory is sufficient | Tau sweep lacks monotonic improvement | Memory horizon alone is not a reliable remedy under this protocol | Gain/activity and optimization confound a pure horizon interpretation | Strong negative observation; no pure horizon causal estimate |

## 18. What CoreBenchmark v1 Established

It established an auditable baseline, a fixed cross-user cohort, deterministic paired comparisons within a version, and common loss/selection/probe conventions. It demonstrated that native accuracy, pooled linear accessibility, explicit temporal accessibility, and output-spike realization can move differently. On this split, O0 is a strong native anchor among the objective cases, an added third layer does not help, and Relative10 exposes residual alignment-dependent classification potential. Common-suffix reset confirms history-sensitive predictions without requiring a speculative gate mechanism.

## 19. What CoreBenchmark v1 Did NOT Establish

It did not establish universal superiority across user splits; an abstract, stable, user-invariant stroke code; absence of history when order gaps shrink; a purely local L2; a causal identification of harmful versus beneficial persistent neurons; or that firing suppression/selective gating is necessary. It did not establish Relative10 as a deployable streaming solution or upper bound. It did not measure independent seen-user held-out-segment performance, execute D2/D3 or membrane sweeps, or provide a matched executed v1.1 O0 baseline for R3. These are scope limits of the measured protocol, not post-hoc exclusions based on performance.

## 20. Connection to Subsequent Experiments

The September 28–30 Project discussions explicitly used the benchmark contract as the reference for Exp14/14.1: common two-layer backbone, time constants, user split, and seeds, while studying supervision that organizes class/history evidence across users. The dual-loader proposal aimed to preserve task sampling while adding structured same-class/different-user supervision. Later selective-memory discussions kept WCCE as the native anchor and used the probe/reset signatures to ask what history should be stored rather than whether history exists.

Those descendants may reuse code, data, or conventions while changing objectives, training schedules, initialization versions, or checkpoint rules. Protocol inheritance is not automatic numerical identity. Their quantitative results are outside this report's original matrix. Exp13 historical and extension diagnostics provide related context; their retrieval and other tests are not silently attributed to CoreBenchmark itself.

## 21. Retrospective Interpretation

Later Project discussions reframed the problem as history content and organization: strong state dependence can coexist with incomplete transfer and residual phase-specific decoding benefit. Gate/prefix discussions further distinguished performance gain from a learned selective-write mechanism. Persistent-firing discussions raised the possibility that contextual activity contains useful class evidence together with user-specific information.

These are retrospective hypotheses from later conversations, not mechanisms proven by CoreBenchmark. The present report does not import later Exp15–17 numbers, pruning results, or gate behavior into its original conclusions. The benchmark's durable contribution is to make such follow-up questions testable against explicit shared controls.

## 22. Evidence and Traceability

### Primary numerical authority

Unity repository: `/home/zhaolongwei_umass_edu/projects/writingRing/`. Original artifacts are under `core_benchmark_v1/results/main/`; extension artifacts are under `core_benchmark_v1/results/core04_e2e_lif_v1_1/`. Both contain `protocol.lock.json`, immutable `dataset.npz`, and per-run provenance/completion/checkpoint/metric files. Main aggregate manifest status is PASS; extension has three individually completed runs and no full default-matrix manifest.

Main identity: `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`. Extension identity: `5a06cbe2e861f7ed18abe92cdc28678e34b9d5418c97f58fc45e8577ef2ea65b`.

Validation performed during report reconstruction: original 16 and extension 17 locked source files matched their recorded git snapshots; **255** selected JSON/checkpoint/initial/head artifact SHA256 checks had zero mismatches; all **27** native aggregates matched `native.json` exactly; all **1,140** per-seed probe means matched the **2,964** original `probes.json` rows to floating-point precision (maximum discrepancy 1.11e−16). Local mean/SD recomputation matched all **380** aggregate probe coordinates. Accounting confirmed completed tasks with exit 0:0. No synthetic execution results are included.

The following inventory identifies original Unity evidence. The report publishes aggregate and per-seed experimental outcomes only; per-user metric tables, sample-level counts, raw data, checkpoints, and the separate CSV evidence bundle are not included in this repository submission. During reconstruction, a transport copy of the readout CSV with a truncated header was discarded in favor of the verified full 27-column source.

| Report evidence file | Source and purpose |
| --- | --- |
| `native_runs.csv` (Unity artifact; see Section 22) | main/aggregate/native_runs.csv; all 27 native runs |
| `probe_seed_means.csv` (Unity artifact; see Section 22) | main/aggregate/probe_seed_means.csv; all 1,140 per-seed probe coordinates |
| `probe_summary.csv` (Unity artifact; see Section 22) | main/aggregate/probe_summary.csv; all 380 aggregate coordinates |
| `readout_actual.csv` (Unity artifact; see Section 22) | main/aggregate/readout_runs.csv; all 24 mode views, including duplicated R0 |
| `e2e_readout.csv` (Unity artifact; see Section 22) | v1.1 runs/*/readout.json; six R3/R4 evaluations |
| `run_inventory.csv` (Unity artifact; see Section 22) | Both roots: provenance/history/completion; 36 tasks |
| `history_reset.csv` (Unity artifact; see Section 22) | main/aggregate/history_reset.csv; all endpoint interventions |
| `lag_similarity.csv` (Unity artifact; see Section 22) | main/aggregate/lag_similarity.csv; pair coverage and metrics |
| `activity.csv` (Unity artifact; see Section 22) | main/aggregate/activity.csv; firing/pre-reset state diagnostics |
| `native_per_user.csv` (Unity artifact; see Section 22) | main/aggregate/native_per_user.csv; train/val/test per-user metrics |
| `sample_counts.csv` (Unity artifact; see Section 22) | main/aggregate/sample_counts.csv; cohort/class counts |

### Implementation authority

Current core model/probe/training/data/diagnostic files were diffed against the original source commit with no changes in those definitions. Protocol and readout extension changes were reconstructed separately. Pinned source links follow.

| Version | File | Authority |
| --- | --- | --- |
| v1.0 | [core_benchmark_v1/README.md](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/README.md) | Original implementation |
| v1.0 | [core_benchmark_v1/00_protocol/default.json](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/00_protocol/default.json) | Original implementation |
| v1.0 | [core_benchmark_v1/00_protocol/TASKSPEC.md](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/00_protocol/TASKSPEC.md) | Original implementation |
| v1.0 | [core_benchmark_v1/protocol.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/protocol.py) | Original implementation |
| v1.0 | [core_benchmark_v1/data.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/data.py) | Original implementation |
| v1.0 | [core_benchmark_v1/model.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/model.py) | Original implementation |
| v1.0 | [core_benchmark_v1/training.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/training.py) | Original implementation |
| v1.0 | [core_benchmark_v1/probes.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/probes.py) | Original implementation |
| v1.0 | [core_benchmark_v1/readout.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/readout.py) | Original implementation |
| v1.0 | [core_benchmark_v1/diagnostics.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/diagnostics.py) | Original implementation |
| v1.0 | [core_benchmark_v1/aggregate.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/aggregate.py) | Original implementation |
| v1.0 | [core_benchmark_v1/runner.py](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/runner.py) | Original implementation |
| v1.0 | [core_benchmark_v1/slurm/common.bash](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/slurm/common.bash) | Original implementation |
| v1.0 | [core_benchmark_v1/slurm/submit.bash](https://github.com/hellowPluto78700/writingRing/blob/96088fe7725f6d0748aca7027abfe8fc438bad2a/core_benchmark_v1/slurm/submit.bash) | Original implementation |
| v1.1 | [core_benchmark_v1/protocol.py](https://github.com/hellowPluto78700/writingRing/blob/7bc6d6fa6c0caadea0685b95a51fadae3add01f6/core_benchmark_v1/protocol.py) | E2E extension/version pairing |
| v1.1 | [core_benchmark_v1/e2e_readout.py](https://github.com/hellowPluto78700/writingRing/blob/7bc6d6fa6c0caadea0685b95a51fadae3add01f6/core_benchmark_v1/e2e_readout.py) | E2E extension/version pairing |
| v1.1 | [core_benchmark_v1/aggregate.py](https://github.com/hellowPluto78700/writingRing/blob/7bc6d6fa6c0caadea0685b95a51fadae3add01f6/core_benchmark_v1/aggregate.py) | E2E extension/version pairing |

Implementation-history commits include `36363fe8` (isolated benchmark), `746c465b` (Unity profile/nounset compatibility), `19f3a0b0` (E2E LIF control), `d735a14f` (run mapping), `75f39506` (version bump), `aed70211` (routing), `b0ba703e` (aggregation), and `a067280d` (documentation). Commit hashes are the traceability authority here; exact PR numbers for these changes are **not verified from available artifacts**.

### Contemporary chat evidence and limits

The provided Project excerpts identify “Branch · exp13” (September 27) as the explicit request for common split/seeds/bias probes; “R3/R4 Results Watch” (September 27) as the request to add E2E LIF and same-checkpoint accumulator swap and clarify shuffled-probe fitting; the benchmark/presentation discussion (September 27) as distinguishing reset effects on predictions from lag similarity; and Exp14/14.1 discussions (September 28–30) as protocol inheritance. Personal-context retrieval additionally recovered timestamped summaries at September 28 21:54 UTC (Exp14 contract), September 29 16:35 UTC (locked benchmark inheritance), and September 30 04:13 UTC (residual phase accessibility interpretation).

These retrieved summaries are historical context, not primary numerical sources or verbatim transcripts. Full unabridged Project history, original conversation URLs, exact initial proposal timestamps, and unavailable pre-fix failure logs are **not verified from available artifacts**. Where remembered statements disagree with Unity, this report uses Unity: in particular, O0 Fixed250-order increases at L2 and affine gains are representation-dependent. The later README pairing statement is qualified using executed version-specific RNG evidence.

### Report validation scope

This is a documentation-only repository change. Validation checks table coverage, unique coordinates, seed membership, mean/sample-SD recomputation, source references, and consistency with verified Unity metrics. No experiment code is modified; no training, expensive evaluation, unrelated pytest, or full CI suite is claimed.

## Appendix A. Complete Aggregate Probe Matrix

Each state/geometry section includes every case and available layer, all temporal aggregations, and train/validation/test BA. Each cell is mean ± sample SD over seeds 11/23/37. Shuffled rows use within-seed five-replicate means. The spike/no-bias test-only compact view in Section 12 is a subset of this complete matrix, not a separate analysis cohort.

### State: spike; decoder: no_bias

| Case | Layer | Aggregation | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 95.16 ± 2.92 | 52.39 ± 0.77 | 55.08 ± 2.23 |
| O0 | L1 | fixed250_ordered | 96.98 ± 1.96 | 54.34 ± 1.96 | 56.28 ± 1.47 |
| O0 | L1 | fixed250_shuffled | 95.74 ± 1.93 | 41.46 ± 2.00 | 44.86 ± 2.31 |
| O0 | L1 | relative10_ordered | 94.41 ± 3.63 | 69.83 ± 2.17 | 71.73 ± 1.82 |
| O0 | L1 | relative10_shuffled | 98.31 ± 0.45 | 34.80 ± 1.18 | 35.50 ± 0.71 |
| O0 | L2 | whole_count | 96.22 ± 2.85 | 57.27 ± 2.76 | 57.57 ± 4.24 |
| O0 | L2 | fixed250_ordered | 99.26 ± 0.87 | 58.82 ± 2.09 | 58.62 ± 2.69 |
| O0 | L2 | fixed250_shuffled | 98.94 ± 0.93 | 53.23 ± 3.30 | 53.79 ± 3.04 |
| O0 | L2 | relative10_ordered | 98.16 ± 2.78 | 66.46 ± 2.96 | 67.04 ± 0.73 |
| O0 | L2 | relative10_shuffled | 97.69 ± 0.81 | 50.76 ± 2.02 | 54.24 ± 3.66 |
| O1 | L1 | whole_count | 92.16 ± 4.16 | 49.17 ± 1.17 | 55.25 ± 3.77 |
| O1 | L1 | fixed250_ordered | 94.41 ± 0.98 | 54.02 ± 4.13 | 55.70 ± 1.14 |
| O1 | L1 | fixed250_shuffled | 95.43 ± 0.36 | 43.74 ± 1.20 | 45.20 ± 2.51 |
| O1 | L1 | relative10_ordered | 94.48 ± 5.62 | 66.43 ± 1.94 | 70.74 ± 2.36 |
| O1 | L1 | relative10_shuffled | 96.32 ± 1.37 | 34.28 ± 1.21 | 37.09 ± 2.33 |
| O1 | L2 | whole_count | 93.29 ± 1.67 | 49.15 ± 2.95 | 58.55 ± 4.30 |
| O1 | L2 | fixed250_ordered | 97.51 ± 1.18 | 57.55 ± 2.31 | 63.74 ± 1.37 |
| O1 | L2 | fixed250_shuffled | 95.97 ± 1.13 | 51.73 ± 1.82 | 55.98 ± 0.83 |
| O1 | L2 | relative10_ordered | 96.27 ± 3.10 | 63.39 ± 2.51 | 69.79 ± 7.34 |
| O1 | L2 | relative10_shuffled | 97.61 ± 1.00 | 47.73 ± 0.91 | 52.92 ± 2.06 |
| O2 | L1 | whole_count | 91.44 ± 5.53 | 52.21 ± 2.76 | 59.62 ± 1.26 |
| O2 | L1 | fixed250_ordered | 97.23 ± 2.11 | 57.15 ± 1.51 | 56.26 ± 1.32 |
| O2 | L1 | fixed250_shuffled | 95.96 ± 2.65 | 44.42 ± 1.13 | 46.10 ± 3.12 |
| O2 | L1 | relative10_ordered | 98.36 ± 1.00 | 70.07 ± 1.04 | 72.22 ± 0.81 |
| O2 | L1 | relative10_shuffled | 98.00 ± 1.05 | 37.53 ± 2.20 | 38.75 ± 1.81 |
| O2 | L2 | whole_count | 96.61 ± 3.29 | 56.69 ± 3.92 | 54.06 ± 4.24 |
| O2 | L2 | fixed250_ordered | 99.02 ± 0.66 | 62.65 ± 2.93 | 57.96 ± 1.15 |
| O2 | L2 | fixed250_shuffled | 98.67 ± 1.36 | 55.69 ± 2.03 | 51.57 ± 1.12 |
| O2 | L2 | relative10_ordered | 97.15 ± 2.33 | 64.59 ± 2.78 | 67.79 ± 2.20 |
| O2 | L2 | relative10_shuffled | 98.56 ± 1.73 | 54.01 ± 3.49 | 52.36 ± 1.10 |
| O3 | L1 | whole_count | 92.49 ± 4.22 | 52.35 ± 0.67 | 57.17 ± 3.17 |
| O3 | L1 | fixed250_ordered | 97.36 ± 1.71 | 54.94 ± 2.12 | 57.93 ± 0.46 |
| O3 | L1 | fixed250_shuffled | 92.74 ± 4.11 | 43.46 ± 1.97 | 44.77 ± 2.69 |
| O3 | L1 | relative10_ordered | 95.21 ± 5.67 | 67.48 ± 2.39 | 71.47 ± 1.36 |
| O3 | L1 | relative10_shuffled | 98.14 ± 0.53 | 36.22 ± 1.89 | 38.17 ± 1.91 |
| O3 | L2 | whole_count | 95.87 ± 3.24 | 55.05 ± 4.54 | 58.48 ± 1.35 |
| O3 | L2 | fixed250_ordered | 99.40 ± 0.62 | 58.93 ± 3.17 | 58.03 ± 3.07 |
| O3 | L2 | fixed250_shuffled | 98.91 ± 1.03 | 54.86 ± 3.81 | 53.87 ± 2.37 |
| O3 | L2 | relative10_ordered | 97.03 ± 2.32 | 66.38 ± 4.71 | 69.78 ± 2.66 |
| O3 | L2 | relative10_shuffled | 99.52 ± 0.19 | 52.86 ± 3.12 | 52.59 ± 0.68 |
| T1 | L1 | whole_count | 93.13 ± 5.72 | 45.97 ± 1.01 | 48.75 ± 0.34 |
| T1 | L1 | fixed250_ordered | 93.60 ± 1.06 | 49.73 ± 1.84 | 53.90 ± 1.28 |
| T1 | L1 | fixed250_shuffled | 94.03 ± 2.66 | 39.59 ± 1.18 | 40.16 ± 1.21 |
| T1 | L1 | relative10_ordered | 95.34 ± 0.85 | 66.06 ± 2.88 | 71.86 ± 1.94 |
| T1 | L1 | relative10_shuffled | 97.03 ± 0.55 | 32.53 ± 0.24 | 30.83 ± 0.33 |
| T1 | L2 | whole_count | 92.74 ± 1.38 | 53.42 ± 6.02 | 58.29 ± 0.45 |
| T1 | L2 | fixed250_ordered | 99.02 ± 1.13 | 59.28 ± 3.06 | 61.48 ± 4.36 |
| T1 | L2 | fixed250_shuffled | 95.14 ± 2.70 | 54.50 ± 2.88 | 52.84 ± 1.38 |
| T1 | L2 | relative10_ordered | 95.38 ± 2.18 | 69.49 ± 3.61 | 70.74 ± 1.00 |
| T1 | L2 | relative10_shuffled | 97.84 ± 0.68 | 49.26 ± 2.12 | 51.59 ± 1.26 |
| T2 | L1 | whole_count | 92.69 ± 7.44 | 50.51 ± 2.51 | 50.29 ± 6.19 |
| T2 | L1 | fixed250_ordered | 96.05 ± 3.43 | 50.70 ± 1.50 | 53.11 ± 3.37 |
| T2 | L1 | fixed250_shuffled | 92.77 ± 5.91 | 39.44 ± 0.35 | 40.56 ± 0.94 |
| T2 | L1 | relative10_ordered | 97.08 ± 2.02 | 66.01 ± 1.11 | 70.86 ± 0.22 |
| T2 | L1 | relative10_shuffled | 95.79 ± 1.19 | 32.62 ± 0.58 | 33.44 ± 0.39 |
| T2 | L2 | whole_count | 92.57 ± 6.83 | 54.96 ± 1.19 | 56.72 ± 1.70 |
| T2 | L2 | fixed250_ordered | 99.78 ± 0.11 | 59.74 ± 2.60 | 60.29 ± 5.31 |
| T2 | L2 | fixed250_shuffled | 99.12 ± 0.29 | 54.26 ± 3.77 | 53.10 ± 2.15 |
| T2 | L2 | relative10_ordered | 97.02 ± 3.28 | 69.33 ± 2.12 | 71.12 ± 2.35 |
| T2 | L2 | relative10_shuffled | 97.06 ± 2.08 | 48.98 ± 2.88 | 50.54 ± 1.08 |
| T3 | L1 | whole_count | 87.47 ± 5.24 | 46.43 ± 0.64 | 46.13 ± 1.18 |
| T3 | L1 | fixed250_ordered | 92.54 ± 6.21 | 47.63 ± 3.41 | 50.12 ± 1.95 |
| T3 | L1 | fixed250_shuffled | 94.24 ± 1.21 | 37.90 ± 3.33 | 39.40 ± 1.99 |
| T3 | L1 | relative10_ordered | 95.19 ± 3.31 | 62.37 ± 2.53 | 69.57 ± 1.46 |
| T3 | L1 | relative10_shuffled | 95.65 ± 1.37 | 29.51 ± 1.29 | 30.87 ± 1.81 |
| T3 | L2 | whole_count | 88.33 ± 13.43 | 50.03 ± 3.62 | 54.48 ± 2.20 |
| T3 | L2 | fixed250_ordered | 94.25 ± 5.16 | 54.32 ± 3.45 | 58.25 ± 3.22 |
| T3 | L2 | fixed250_shuffled | 94.22 ± 3.97 | 50.74 ± 3.93 | 51.20 ± 0.96 |
| T3 | L2 | relative10_ordered | 97.36 ± 1.79 | 66.45 ± 1.68 | 68.14 ± 3.65 |
| T3 | L2 | relative10_shuffled | 96.98 ± 1.32 | 46.87 ± 3.14 | 48.97 ± 3.67 |
| T4 | L1 | whole_count | 93.17 ± 5.63 | 49.83 ± 3.05 | 53.61 ± 3.22 |
| T4 | L1 | fixed250_ordered | 95.94 ± 2.76 | 54.80 ± 4.03 | 52.83 ± 3.36 |
| T4 | L1 | fixed250_shuffled | 96.54 ± 0.97 | 42.02 ± 5.41 | 44.80 ± 1.48 |
| T4 | L1 | relative10_ordered | 96.42 ± 0.09 | 68.75 ± 1.98 | 69.64 ± 1.32 |
| T4 | L1 | relative10_shuffled | 97.65 ± 0.56 | 33.76 ± 2.45 | 35.92 ± 1.30 |
| T4 | L2 | whole_count | 97.66 ± 1.67 | 52.81 ± 1.10 | 55.13 ± 2.30 |
| T4 | L2 | fixed250_ordered | 97.54 ± 1.83 | 55.34 ± 1.41 | 59.73 ± 3.75 |
| T4 | L2 | fixed250_shuffled | 97.24 ± 2.60 | 52.19 ± 0.60 | 52.41 ± 3.92 |
| T4 | L2 | relative10_ordered | 98.05 ± 1.70 | 62.55 ± 4.96 | 66.08 ± 3.16 |
| T4 | L2 | relative10_shuffled | 97.21 ± 0.95 | 49.91 ± 2.24 | 54.23 ± 2.80 |
| D1 | L1 | whole_count | 80.35 ± 10.17 | 40.69 ± 2.39 | 43.85 ± 1.08 |
| D1 | L1 | fixed250_ordered | 92.62 ± 5.27 | 48.55 ± 1.42 | 46.90 ± 2.99 |
| D1 | L1 | fixed250_shuffled | 91.55 ± 2.49 | 33.46 ± 1.75 | 35.32 ± 1.12 |
| D1 | L1 | relative10_ordered | 94.96 ± 2.04 | 63.12 ± 3.40 | 65.46 ± 1.97 |
| D1 | L1 | relative10_shuffled | 93.96 ± 1.13 | 27.78 ± 0.32 | 26.20 ± 1.67 |
| D1 | L2 | whole_count | 92.17 ± 4.55 | 46.15 ± 4.76 | 50.70 ± 6.31 |
| D1 | L2 | fixed250_ordered | 96.06 ± 4.13 | 52.19 ± 4.02 | 54.33 ± 1.86 |
| D1 | L2 | fixed250_shuffled | 94.26 ± 1.56 | 43.62 ± 2.43 | 45.72 ± 3.24 |
| D1 | L2 | relative10_ordered | 96.99 ± 2.23 | 60.70 ± 1.25 | 65.48 ± 1.55 |
| D1 | L2 | relative10_shuffled | 95.54 ± 2.17 | 41.04 ± 3.41 | 42.29 ± 4.15 |
| D1 | L3 | whole_count | 93.98 ± 4.39 | 49.42 ± 2.80 | 48.08 ± 0.59 |
| D1 | L3 | fixed250_ordered | 96.67 ± 3.50 | 49.57 ± 3.13 | 47.99 ± 2.37 |
| D1 | L3 | fixed250_shuffled | 94.19 ± 6.92 | 44.91 ± 3.10 | 44.94 ± 2.77 |
| D1 | L3 | relative10_ordered | 97.25 ± 1.00 | 52.14 ± 1.12 | 55.60 ± 2.81 |
| D1 | L3 | relative10_shuffled | 94.20 ± 5.05 | 45.83 ± 5.12 | 46.42 ± 3.59 |

### State: spike; decoder: affine

| Case | Layer | Aggregation | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 94.22 ± 3.80 | 55.66 ± 1.65 | 59.75 ± 4.40 |
| O0 | L1 | fixed250_ordered | 98.22 ± 0.10 | 52.03 ± 0.55 | 56.16 ± 0.42 |
| O0 | L1 | fixed250_shuffled | 98.76 ± 0.56 | 39.31 ± 2.28 | 44.90 ± 1.49 |
| O0 | L1 | relative10_ordered | 98.58 ± 0.00 | 65.15 ± 1.45 | 68.15 ± 2.83 |
| O0 | L1 | relative10_shuffled | 96.71 ± 0.41 | 37.33 ± 1.24 | 39.98 ± 1.44 |
| O0 | L2 | whole_count | 97.80 ± 1.36 | 56.92 ± 3.47 | 59.83 ± 4.42 |
| O0 | L2 | fixed250_ordered | 97.36 ± 4.28 | 58.94 ± 4.06 | 56.53 ± 2.62 |
| O0 | L2 | fixed250_shuffled | 99.21 ± 0.94 | 52.47 ± 4.38 | 52.13 ± 0.48 |
| O0 | L2 | relative10_ordered | 99.16 ± 0.64 | 65.06 ± 2.07 | 67.48 ± 1.46 |
| O0 | L2 | relative10_shuffled | 97.18 ± 1.13 | 51.91 ± 2.17 | 55.59 ± 1.11 |
| O1 | L1 | whole_count | 94.66 ± 0.30 | 50.92 ± 1.46 | 55.92 ± 1.93 |
| O1 | L1 | fixed250_ordered | 98.03 ± 1.03 | 49.81 ± 2.50 | 53.14 ± 5.01 |
| O1 | L1 | fixed250_shuffled | 97.99 ± 1.24 | 37.77 ± 1.59 | 40.50 ± 1.78 |
| O1 | L1 | relative10_ordered | 97.62 ± 0.04 | 62.17 ± 2.11 | 67.87 ± 2.69 |
| O1 | L1 | relative10_shuffled | 92.09 ± 5.98 | 35.83 ± 2.80 | 40.83 ± 2.57 |
| O1 | L2 | whole_count | 92.51 ± 2.84 | 52.10 ± 3.83 | 61.62 ± 2.63 |
| O1 | L2 | fixed250_ordered | 98.06 ± 1.79 | 55.41 ± 1.33 | 62.91 ± 1.12 |
| O1 | L2 | fixed250_shuffled | 97.32 ± 1.05 | 49.66 ± 1.68 | 54.49 ± 0.48 |
| O1 | L2 | relative10_ordered | 98.23 ± 0.66 | 61.00 ± 1.62 | 70.90 ± 2.07 |
| O1 | L2 | relative10_shuffled | 94.10 ± 1.29 | 48.69 ± 0.58 | 56.34 ± 1.50 |
| O2 | L1 | whole_count | 91.33 ± 5.01 | 54.83 ± 6.14 | 62.74 ± 1.44 |
| O2 | L1 | fixed250_ordered | 95.95 ± 2.81 | 53.45 ± 2.00 | 57.39 ± 1.26 |
| O2 | L1 | fixed250_shuffled | 98.51 ± 0.62 | 42.55 ± 0.59 | 44.93 ± 2.88 |
| O2 | L1 | relative10_ordered | 97.89 ± 1.94 | 66.98 ± 1.24 | 72.76 ± 1.94 |
| O2 | L1 | relative10_shuffled | 97.99 ± 1.38 | 39.85 ± 2.59 | 41.26 ± 1.76 |
| O2 | L2 | whole_count | 97.61 ± 1.94 | 56.67 ± 4.47 | 56.17 ± 3.08 |
| O2 | L2 | fixed250_ordered | 96.88 ± 4.09 | 62.56 ± 4.60 | 57.91 ± 1.82 |
| O2 | L2 | fixed250_shuffled | 99.25 ± 0.98 | 55.37 ± 1.46 | 51.84 ± 3.20 |
| O2 | L2 | relative10_ordered | 97.87 ± 2.65 | 63.91 ± 2.63 | 68.68 ± 5.35 |
| O2 | L2 | relative10_shuffled | 97.65 ± 0.98 | 54.80 ± 3.01 | 52.91 ± 2.11 |
| O3 | L1 | whole_count | 95.07 ± 3.98 | 55.68 ± 0.49 | 58.19 ± 1.61 |
| O3 | L1 | fixed250_ordered | 98.65 ± 0.26 | 51.51 ± 1.05 | 57.22 ± 0.74 |
| O3 | L1 | fixed250_shuffled | 98.33 ± 1.12 | 41.07 ± 1.58 | 45.26 ± 2.02 |
| O3 | L1 | relative10_ordered | 98.73 ± 0.25 | 64.90 ± 1.77 | 69.82 ± 2.53 |
| O3 | L1 | relative10_shuffled | 96.65 ± 1.48 | 38.76 ± 1.41 | 42.60 ± 0.95 |
| O3 | L2 | whole_count | 95.24 ± 2.25 | 55.12 ± 5.45 | 60.11 ± 1.34 |
| O3 | L2 | fixed250_ordered | 99.37 ± 0.81 | 60.85 ± 4.70 | 59.12 ± 3.84 |
| O3 | L2 | fixed250_shuffled | 99.01 ± 0.55 | 54.89 ± 3.80 | 52.58 ± 2.57 |
| O3 | L2 | relative10_ordered | 99.11 ± 0.65 | 65.42 ± 2.04 | 68.76 ± 5.20 |
| O3 | L2 | relative10_shuffled | 98.50 ± 0.36 | 53.10 ± 3.58 | 54.62 ± 1.04 |
| T1 | L1 | whole_count | 91.01 ± 6.24 | 48.57 ± 3.00 | 51.04 ± 2.96 |
| T1 | L1 | fixed250_ordered | 93.54 ± 5.65 | 48.10 ± 2.49 | 53.93 ± 1.16 |
| T1 | L1 | fixed250_shuffled | 97.90 ± 0.90 | 36.95 ± 2.00 | 39.77 ± 0.98 |
| T1 | L1 | relative10_ordered | 97.80 ± 0.45 | 64.98 ± 1.50 | 68.14 ± 3.46 |
| T1 | L1 | relative10_shuffled | 93.21 ± 1.85 | 33.90 ± 1.35 | 35.77 ± 1.12 |
| T1 | L2 | whole_count | 90.63 ± 2.65 | 55.52 ± 4.65 | 57.89 ± 1.15 |
| T1 | L2 | fixed250_ordered | 98.11 ± 1.03 | 59.92 ± 1.26 | 63.87 ± 3.37 |
| T1 | L2 | fixed250_shuffled | 98.04 ± 1.83 | 53.80 ± 3.37 | 53.33 ± 2.17 |
| T1 | L2 | relative10_ordered | 99.82 ± 0.17 | 66.78 ± 3.00 | 65.69 ± 2.77 |
| T1 | L2 | relative10_shuffled | 93.18 ± 2.46 | 51.61 ± 1.58 | 53.73 ± 0.51 |
| T2 | L1 | whole_count | 92.30 ± 4.16 | 54.72 ± 2.11 | 55.32 ± 1.41 |
| T2 | L1 | fixed250_ordered | 98.70 ± 0.89 | 50.63 ± 1.36 | 53.59 ± 1.69 |
| T2 | L1 | fixed250_shuffled | 97.73 ± 1.39 | 38.44 ± 0.84 | 40.96 ± 1.00 |
| T2 | L1 | relative10_ordered | 97.42 ± 3.10 | 64.26 ± 2.05 | 67.95 ± 4.93 |
| T2 | L1 | relative10_shuffled | 96.18 ± 2.26 | 35.01 ± 1.32 | 37.80 ± 2.31 |
| T2 | L2 | whole_count | 89.61 ± 7.55 | 57.30 ± 2.04 | 60.85 ± 3.38 |
| T2 | L2 | fixed250_ordered | 99.95 ± 0.09 | 62.00 ± 2.27 | 60.44 ± 4.47 |
| T2 | L2 | fixed250_shuffled | 99.65 ± 0.48 | 55.12 ± 3.94 | 51.79 ± 1.14 |
| T2 | L2 | relative10_ordered | 98.68 ± 1.16 | 68.59 ± 3.78 | 71.17 ± 3.08 |
| T2 | L2 | relative10_shuffled | 94.71 ± 2.36 | 51.53 ± 3.78 | 53.71 ± 1.69 |
| T3 | L1 | whole_count | 82.66 ± 10.51 | 48.22 ± 4.62 | 50.33 ± 3.25 |
| T3 | L1 | fixed250_ordered | 97.07 ± 2.13 | 45.92 ± 4.21 | 49.73 ± 3.81 |
| T3 | L1 | fixed250_shuffled | 96.21 ± 1.70 | 35.13 ± 2.49 | 38.76 ± 3.31 |
| T3 | L1 | relative10_ordered | 97.65 ± 1.84 | 61.88 ± 1.38 | 68.16 ± 3.47 |
| T3 | L1 | relative10_shuffled | 93.40 ± 0.91 | 31.12 ± 1.80 | 33.93 ± 3.76 |
| T3 | L2 | whole_count | 86.95 ± 10.51 | 51.37 ± 0.96 | 57.00 ± 1.43 |
| T3 | L2 | fixed250_ordered | 95.22 ± 5.96 | 56.03 ± 3.83 | 57.39 ± 0.66 |
| T3 | L2 | fixed250_shuffled | 96.66 ± 2.44 | 50.50 ± 3.77 | 52.56 ± 1.29 |
| T3 | L2 | relative10_ordered | 97.85 ± 0.59 | 64.96 ± 1.81 | 67.79 ± 3.22 |
| T3 | L2 | relative10_shuffled | 92.79 ± 3.82 | 49.01 ± 1.60 | 53.36 ± 1.49 |
| T4 | L1 | whole_count | 92.16 ± 6.65 | 48.70 ± 2.22 | 57.53 ± 3.80 |
| T4 | L1 | fixed250_ordered | 98.80 ± 0.80 | 51.31 ± 3.45 | 50.96 ± 1.11 |
| T4 | L1 | fixed250_shuffled | 98.10 ± 1.17 | 37.42 ± 2.50 | 41.53 ± 0.52 |
| T4 | L1 | relative10_ordered | 98.54 ± 0.85 | 65.73 ± 1.43 | 66.91 ± 4.94 |
| T4 | L1 | relative10_shuffled | 93.99 ± 0.74 | 34.71 ± 2.51 | 40.42 ± 2.82 |
| T4 | L2 | whole_count | 96.47 ± 2.14 | 52.59 ± 2.37 | 55.26 ± 4.80 |
| T4 | L2 | fixed250_ordered | 98.35 ± 1.20 | 56.04 ± 0.83 | 62.70 ± 3.57 |
| T4 | L2 | fixed250_shuffled | 98.88 ± 0.65 | 51.44 ± 1.73 | 52.62 ± 3.79 |
| T4 | L2 | relative10_ordered | 97.76 ± 2.05 | 61.64 ± 2.97 | 65.13 ± 5.02 |
| T4 | L2 | relative10_shuffled | 97.28 ± 0.30 | 49.54 ± 1.41 | 54.87 ± 1.17 |
| D1 | L1 | whole_count | 92.64 ± 0.50 | 44.48 ± 6.40 | 44.37 ± 5.50 |
| D1 | L1 | fixed250_ordered | 95.44 ± 1.34 | 45.86 ± 1.40 | 51.17 ± 1.95 |
| D1 | L1 | fixed250_shuffled | 95.45 ± 0.57 | 31.38 ± 2.99 | 35.60 ± 0.71 |
| D1 | L1 | relative10_ordered | 96.39 ± 1.15 | 60.59 ± 3.55 | 64.62 ± 3.97 |
| D1 | L1 | relative10_shuffled | 89.07 ± 4.21 | 30.92 ± 1.43 | 32.97 ± 2.33 |
| D1 | L2 | whole_count | 87.18 ± 12.97 | 48.07 ± 5.70 | 52.91 ± 7.97 |
| D1 | L2 | fixed250_ordered | 96.55 ± 1.21 | 50.97 ± 4.99 | 55.04 ± 2.43 |
| D1 | L2 | fixed250_shuffled | 95.45 ± 3.48 | 42.51 ± 2.98 | 46.51 ± 2.28 |
| D1 | L2 | relative10_ordered | 96.29 ± 0.73 | 58.73 ± 2.07 | 65.39 ± 4.52 |
| D1 | L2 | relative10_shuffled | 92.34 ± 2.46 | 42.48 ± 3.55 | 46.55 ± 4.67 |
| D1 | L3 | whole_count | 96.62 ± 0.74 | 48.22 ± 4.49 | 49.87 ± 2.92 |
| D1 | L3 | fixed250_ordered | 97.45 ± 2.69 | 49.90 ± 3.36 | 48.15 ± 2.48 |
| D1 | L3 | fixed250_shuffled | 95.39 ± 5.66 | 45.33 ± 4.69 | 46.34 ± 1.81 |
| D1 | L3 | relative10_ordered | 96.89 ± 2.04 | 52.07 ± 2.00 | 54.95 ± 4.80 |
| D1 | L3 | relative10_shuffled | 96.18 ± 3.48 | 45.04 ± 4.47 | 47.23 ± 4.16 |

### State: pre_reset; decoder: no_bias

| Case | Layer | Aggregation | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 99.61 ± 0.42 | 50.48 ± 3.95 | 51.43 ± 4.48 |
| O0 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.80 ± 1.04 | 61.15 ± 2.19 |
| O0 | L1 | fixed250_shuffled | 98.92 ± 1.16 | 39.77 ± 1.48 | 35.61 ± 0.64 |
| O0 | L1 | relative10_ordered | 100.00 ± 0.00 | 70.35 ± 0.49 | 68.59 ± 1.27 |
| O0 | L1 | relative10_shuffled | 91.73 ± 0.21 | 25.73 ± 0.92 | 24.37 ± 1.08 |
| O0 | L2 | whole_count | 91.67 ± 7.47 | 54.83 ± 1.74 | 56.47 ± 2.07 |
| O0 | L2 | fixed250_ordered | 98.84 ± 1.46 | 58.65 ± 3.48 | 58.09 ± 4.53 |
| O0 | L2 | fixed250_shuffled | 96.41 ± 2.38 | 52.99 ± 1.67 | 49.82 ± 1.01 |
| O0 | L2 | relative10_ordered | 94.62 ± 4.16 | 67.89 ± 2.95 | 67.77 ± 7.46 |
| O0 | L2 | relative10_shuffled | 94.71 ± 1.96 | 51.27 ± 1.61 | 49.44 ± 2.40 |
| O1 | L1 | whole_count | 92.03 ± 6.92 | 52.37 ± 1.31 | 50.02 ± 2.20 |
| O1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.69 ± 1.34 | 61.41 ± 1.41 |
| O1 | L1 | fixed250_shuffled | 96.98 ± 3.28 | 40.23 ± 0.56 | 35.40 ± 1.43 |
| O1 | L1 | relative10_ordered | 100.00 ± 0.00 | 72.08 ± 1.38 | 70.14 ± 2.31 |
| O1 | L1 | relative10_shuffled | 85.80 ± 0.96 | 25.85 ± 1.12 | 24.26 ± 1.60 |
| O1 | L2 | whole_count | 88.71 ± 3.98 | 52.21 ± 1.15 | 54.23 ± 4.84 |
| O1 | L2 | fixed250_ordered | 94.99 ± 7.41 | 58.80 ± 2.29 | 59.31 ± 2.72 |
| O1 | L2 | fixed250_shuffled | 92.75 ± 2.78 | 51.67 ± 0.98 | 51.63 ± 1.10 |
| O1 | L2 | relative10_ordered | 93.94 ± 4.08 | 65.10 ± 1.57 | 70.45 ± 2.19 |
| O1 | L2 | relative10_shuffled | 96.61 ± 1.66 | 44.85 ± 2.08 | 44.63 ± 0.81 |
| O2 | L1 | whole_count | 92.08 ± 6.86 | 54.78 ± 1.81 | 51.35 ± 3.22 |
| O2 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.12 ± 1.04 | 60.44 ± 2.43 |
| O2 | L1 | fixed250_shuffled | 98.49 ± 0.41 | 40.72 ± 1.55 | 34.92 ± 0.90 |
| O2 | L1 | relative10_ordered | 99.45 ± 0.50 | 69.85 ± 0.69 | 72.05 ± 2.37 |
| O2 | L1 | relative10_shuffled | 86.85 ± 3.85 | 26.02 ± 1.21 | 24.69 ± 1.07 |
| O2 | L2 | whole_count | 96.66 ± 5.37 | 57.03 ± 1.76 | 52.02 ± 4.70 |
| O2 | L2 | fixed250_ordered | 98.84 ± 1.16 | 62.37 ± 2.51 | 60.45 ± 3.32 |
| O2 | L2 | fixed250_shuffled | 96.85 ± 2.11 | 56.89 ± 1.42 | 50.13 ± 1.42 |
| O2 | L2 | relative10_ordered | 95.78 ± 5.36 | 66.48 ± 2.54 | 68.31 ± 6.35 |
| O2 | L2 | relative10_shuffled | 97.00 ± 1.94 | 53.94 ± 1.93 | 47.93 ± 3.07 |
| O3 | L1 | whole_count | 99.56 ± 0.26 | 53.79 ± 1.25 | 50.99 ± 5.41 |
| O3 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.77 ± 0.38 | 60.89 ± 1.45 |
| O3 | L1 | fixed250_shuffled | 98.09 ± 1.99 | 40.71 ± 1.52 | 35.21 ± 1.39 |
| O3 | L1 | relative10_ordered | 99.83 ± 0.29 | 70.04 ± 1.24 | 70.37 ± 3.63 |
| O3 | L1 | relative10_shuffled | 86.25 ± 3.80 | 25.32 ± 0.28 | 24.38 ± 0.60 |
| O3 | L2 | whole_count | 92.53 ± 3.76 | 57.46 ± 1.71 | 55.56 ± 1.83 |
| O3 | L2 | fixed250_ordered | 98.14 ± 1.56 | 62.19 ± 5.47 | 58.04 ± 3.44 |
| O3 | L2 | fixed250_shuffled | 94.94 ± 3.44 | 55.78 ± 2.83 | 52.18 ± 2.56 |
| O3 | L2 | relative10_ordered | 98.19 ± 1.23 | 69.87 ± 3.12 | 65.47 ± 5.05 |
| O3 | L2 | relative10_shuffled | 97.73 ± 1.63 | 51.76 ± 1.11 | 48.43 ± 0.50 |
| T1 | L1 | whole_count | 97.12 ± 0.86 | 48.82 ± 1.56 | 51.00 ± 2.72 |
| T1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 61.19 ± 1.30 | 60.83 ± 0.97 |
| T1 | L1 | fixed250_shuffled | 96.63 ± 2.59 | 37.28 ± 0.80 | 33.81 ± 0.25 |
| T1 | L1 | relative10_ordered | 99.84 ± 0.28 | 69.24 ± 0.80 | 69.07 ± 2.89 |
| T1 | L1 | relative10_shuffled | 89.22 ± 3.51 | 23.39 ± 0.51 | 21.78 ± 0.80 |
| T1 | L2 | whole_count | 85.57 ± 2.32 | 52.88 ± 2.28 | 53.26 ± 1.86 |
| T1 | L2 | fixed250_ordered | 90.34 ± 4.61 | 58.18 ± 2.13 | 59.42 ± 1.99 |
| T1 | L2 | fixed250_shuffled | 93.97 ± 2.47 | 50.58 ± 2.97 | 49.21 ± 2.04 |
| T1 | L2 | relative10_ordered | 94.68 ± 3.44 | 65.39 ± 2.37 | 70.51 ± 0.93 |
| T1 | L2 | relative10_shuffled | 94.16 ± 3.42 | 47.16 ± 2.02 | 46.92 ± 1.46 |
| T2 | L1 | whole_count | 88.71 ± 7.46 | 47.41 ± 0.63 | 53.14 ± 2.10 |
| T2 | L1 | fixed250_ordered | 99.66 ± 0.59 | 59.73 ± 0.66 | 59.85 ± 1.03 |
| T2 | L1 | fixed250_shuffled | 93.13 ± 3.77 | 37.21 ± 0.48 | 32.47 ± 1.57 |
| T2 | L1 | relative10_ordered | 100.00 ± 0.00 | 67.94 ± 0.45 | 66.89 ± 0.55 |
| T2 | L1 | relative10_shuffled | 88.78 ± 3.42 | 23.81 ± 0.38 | 22.67 ± 0.67 |
| T2 | L2 | whole_count | 81.67 ± 6.11 | 51.03 ± 2.40 | 54.20 ± 1.47 |
| T2 | L2 | fixed250_ordered | 95.32 ± 8.11 | 58.36 ± 2.42 | 60.91 ± 2.16 |
| T2 | L2 | fixed250_shuffled | 96.32 ± 2.47 | 51.60 ± 2.43 | 49.38 ± 0.28 |
| T2 | L2 | relative10_ordered | 91.77 ± 4.27 | 68.98 ± 2.50 | 71.64 ± 2.26 |
| T2 | L2 | relative10_shuffled | 96.59 ± 2.69 | 46.28 ± 3.06 | 46.39 ± 0.82 |
| T3 | L1 | whole_count | 99.00 ± 1.73 | 46.00 ± 2.79 | 47.49 ± 1.39 |
| T3 | L1 | fixed250_ordered | 100.00 ± 0.00 | 61.59 ± 1.56 | 60.18 ± 2.34 |
| T3 | L1 | fixed250_shuffled | 96.74 ± 0.61 | 37.14 ± 0.05 | 33.79 ± 0.75 |
| T3 | L1 | relative10_ordered | 99.46 ± 0.50 | 68.36 ± 1.29 | 71.83 ± 3.27 |
| T3 | L1 | relative10_shuffled | 89.09 ± 4.77 | 23.59 ± 0.62 | 22.31 ± 0.89 |
| T3 | L2 | whole_count | 87.34 ± 12.31 | 50.52 ± 1.07 | 51.56 ± 6.07 |
| T3 | L2 | fixed250_ordered | 90.41 ± 4.00 | 53.98 ± 2.36 | 57.03 ± 2.11 |
| T3 | L2 | fixed250_shuffled | 90.67 ± 3.59 | 48.10 ± 2.39 | 49.12 ± 0.54 |
| T3 | L2 | relative10_ordered | 94.31 ± 2.84 | 65.11 ± 2.42 | 67.31 ± 1.66 |
| T3 | L2 | relative10_shuffled | 92.64 ± 2.08 | 45.96 ± 3.42 | 47.09 ± 2.73 |
| T4 | L1 | whole_count | 90.84 ± 7.20 | 53.09 ± 1.15 | 51.28 ± 3.16 |
| T4 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.12 ± 1.84 | 60.23 ± 0.59 |
| T4 | L1 | fixed250_shuffled | 98.16 ± 0.86 | 39.42 ± 0.37 | 34.63 ± 0.41 |
| T4 | L1 | relative10_ordered | 100.00 ± 0.00 | 70.90 ± 1.21 | 70.70 ± 2.92 |
| T4 | L1 | relative10_shuffled | 92.58 ± 0.72 | 25.34 ± 0.97 | 23.84 ± 0.43 |
| T4 | L2 | whole_count | 94.80 ± 3.25 | 53.81 ± 4.54 | 52.55 ± 3.10 |
| T4 | L2 | fixed250_ordered | 98.40 ± 1.54 | 56.20 ± 5.97 | 53.88 ± 7.36 |
| T4 | L2 | fixed250_shuffled | 94.42 ± 2.90 | 51.32 ± 2.45 | 50.31 ± 2.08 |
| T4 | L2 | relative10_ordered | 93.49 ± 1.08 | 63.88 ± 6.95 | 67.07 ± 6.17 |
| T4 | L2 | relative10_shuffled | 96.02 ± 0.44 | 50.72 ± 3.05 | 49.54 ± 2.79 |
| D1 | L1 | whole_count | 87.14 ± 8.57 | 51.60 ± 4.94 | 49.12 ± 2.94 |
| D1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 63.53 ± 2.25 | 60.86 ± 2.55 |
| D1 | L1 | fixed250_shuffled | 98.22 ± 0.71 | 39.51 ± 1.41 | 34.57 ± 1.06 |
| D1 | L1 | relative10_ordered | 99.83 ± 0.29 | 71.65 ± 1.41 | 71.36 ± 2.07 |
| D1 | L1 | relative10_shuffled | 94.86 ± 3.17 | 25.18 ± 1.70 | 24.64 ± 1.59 |
| D1 | L2 | whole_count | 94.99 ± 3.65 | 46.27 ± 1.63 | 45.46 ± 6.29 |
| D1 | L2 | fixed250_ordered | 94.22 ± 3.76 | 54.78 ± 3.61 | 54.62 ± 4.22 |
| D1 | L2 | fixed250_shuffled | 92.33 ± 0.72 | 40.72 ± 4.12 | 38.41 ± 1.98 |
| D1 | L2 | relative10_ordered | 95.20 ± 2.51 | 64.67 ± 2.59 | 65.15 ± 5.47 |
| D1 | L2 | relative10_shuffled | 95.37 ± 3.52 | 35.78 ± 3.09 | 34.19 ± 3.35 |
| D1 | L3 | whole_count | 93.62 ± 3.82 | 50.13 ± 7.38 | 47.27 ± 4.02 |
| D1 | L3 | fixed250_ordered | 95.46 ± 1.30 | 49.20 ± 4.36 | 51.61 ± 6.45 |
| D1 | L3 | fixed250_shuffled | 92.57 ± 4.60 | 46.86 ± 3.80 | 46.44 ± 2.86 |
| D1 | L3 | relative10_ordered | 96.91 ± 1.15 | 54.24 ± 2.95 | 50.74 ± 5.77 |
| D1 | L3 | relative10_shuffled | 91.67 ± 7.09 | 45.67 ± 5.50 | 46.46 ± 2.95 |

### State: pre_reset; decoder: affine

| Case | Layer | Aggregation | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 93.84 ± 5.35 | 56.71 ± 0.15 | 54.79 ± 2.35 |
| O0 | L1 | fixed250_ordered | 100.00 ± 0.00 | 63.95 ± 0.18 | 60.77 ± 3.13 |
| O0 | L1 | fixed250_shuffled | 99.74 ± 0.46 | 40.12 ± 2.32 | 34.91 ± 1.45 |
| O0 | L1 | relative10_ordered | 100.00 ± 0.00 | 70.74 ± 0.28 | 70.09 ± 4.00 |
| O0 | L1 | relative10_shuffled | 95.06 ± 3.88 | 31.01 ± 0.95 | 29.20 ± 1.12 |
| O0 | L2 | whole_count | 92.68 ± 6.81 | 55.29 ± 0.75 | 55.81 ± 3.17 |
| O0 | L2 | fixed250_ordered | 99.95 ± 0.09 | 58.52 ± 3.60 | 55.67 ± 5.59 |
| O0 | L2 | fixed250_shuffled | 98.70 ± 1.51 | 52.49 ± 1.35 | 49.21 ± 2.40 |
| O0 | L2 | relative10_ordered | 99.16 ± 0.29 | 66.85 ± 2.83 | 65.89 ± 2.00 |
| O0 | L2 | relative10_shuffled | 97.01 ± 1.30 | 53.11 ± 1.03 | 50.80 ± 1.31 |
| O1 | L1 | whole_count | 96.63 ± 5.13 | 56.93 ± 0.38 | 51.32 ± 3.48 |
| O1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 64.04 ± 0.26 | 60.46 ± 0.68 |
| O1 | L1 | fixed250_shuffled | 99.76 ± 0.42 | 40.03 ± 0.56 | 35.92 ± 1.59 |
| O1 | L1 | relative10_ordered | 100.00 ± 0.00 | 71.21 ± 0.88 | 72.78 ± 4.49 |
| O1 | L1 | relative10_shuffled | 92.65 ± 3.33 | 32.05 ± 0.51 | 29.86 ± 0.56 |
| O1 | L2 | whole_count | 86.20 ± 8.24 | 51.94 ± 0.45 | 56.34 ± 2.57 |
| O1 | L2 | fixed250_ordered | 99.42 ± 0.72 | 57.28 ± 1.98 | 56.35 ± 6.95 |
| O1 | L2 | fixed250_shuffled | 98.43 ± 1.22 | 49.35 ± 0.56 | 49.24 ± 1.60 |
| O1 | L2 | relative10_ordered | 96.76 ± 2.68 | 62.81 ± 1.75 | 66.49 ± 4.36 |
| O1 | L2 | relative10_shuffled | 94.46 ± 2.16 | 47.04 ± 1.30 | 50.19 ± 0.94 |
| O2 | L1 | whole_count | 96.97 ± 5.09 | 59.39 ± 1.57 | 53.88 ± 0.53 |
| O2 | L1 | fixed250_ordered | 100.00 ± 0.00 | 63.33 ± 1.53 | 58.75 ± 0.93 |
| O2 | L1 | fixed250_shuffled | 100.00 ± 0.00 | 40.10 ± 1.70 | 35.10 ± 0.94 |
| O2 | L1 | relative10_ordered | 100.00 ± 0.00 | 70.03 ± 0.28 | 72.59 ± 5.88 |
| O2 | L1 | relative10_shuffled | 94.46 ± 3.04 | 30.53 ± 1.10 | 29.39 ± 0.95 |
| O2 | L2 | whole_count | 97.39 ± 1.72 | 57.58 ± 3.87 | 53.83 ± 2.18 |
| O2 | L2 | fixed250_ordered | 99.57 ± 0.10 | 60.64 ± 2.42 | 59.94 ± 5.32 |
| O2 | L2 | fixed250_shuffled | 98.39 ± 1.55 | 55.06 ± 1.36 | 49.39 ± 0.57 |
| O2 | L2 | relative10_ordered | 99.83 ± 0.29 | 65.26 ± 1.84 | 64.09 ± 5.60 |
| O2 | L2 | relative10_shuffled | 96.69 ± 1.58 | 55.00 ± 1.99 | 50.23 ± 1.77 |
| O3 | L1 | whole_count | 99.83 ± 0.29 | 58.57 ± 1.95 | 55.01 ± 4.64 |
| O3 | L1 | fixed250_ordered | 100.00 ± 0.00 | 63.65 ± 1.42 | 60.75 ± 0.26 |
| O3 | L1 | fixed250_shuffled | 99.73 ± 0.47 | 40.55 ± 1.24 | 35.52 ± 1.06 |
| O3 | L1 | relative10_ordered | 100.00 ± 0.00 | 69.70 ± 0.31 | 71.07 ± 3.52 |
| O3 | L1 | relative10_shuffled | 97.07 ± 3.02 | 31.64 ± 0.66 | 28.77 ± 0.49 |
| O3 | L2 | whole_count | 93.36 ± 5.04 | 57.68 ± 3.25 | 55.43 ± 5.03 |
| O3 | L2 | fixed250_ordered | 99.70 ± 0.27 | 60.95 ± 4.16 | 57.05 ± 6.38 |
| O3 | L2 | fixed250_shuffled | 99.36 ± 0.71 | 54.03 ± 1.35 | 49.99 ± 2.11 |
| O3 | L2 | relative10_ordered | 98.64 ± 1.93 | 65.44 ± 3.16 | 65.20 ± 1.10 |
| O3 | L2 | relative10_shuffled | 96.89 ± 2.88 | 54.44 ± 3.20 | 50.01 ± 3.27 |
| T1 | L1 | whole_count | 97.97 ± 0.49 | 54.93 ± 3.56 | 53.69 ± 3.13 |
| T1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 60.27 ± 1.40 | 60.52 ± 2.08 |
| T1 | L1 | fixed250_shuffled | 100.00 ± 0.00 | 36.58 ± 0.75 | 34.26 ± 1.80 |
| T1 | L1 | relative10_ordered | 100.00 ± 0.00 | 69.51 ± 1.51 | 69.83 ± 4.90 |
| T1 | L1 | relative10_shuffled | 95.17 ± 3.36 | 28.35 ± 0.35 | 28.85 ± 1.09 |
| T1 | L2 | whole_count | 83.17 ± 2.70 | 55.02 ± 1.80 | 55.18 ± 0.60 |
| T1 | L2 | fixed250_ordered | 97.96 ± 2.32 | 56.72 ± 3.88 | 57.03 ± 4.72 |
| T1 | L2 | fixed250_shuffled | 98.96 ± 0.77 | 51.63 ± 2.77 | 47.37 ± 3.31 |
| T1 | L2 | relative10_ordered | 95.46 ± 0.66 | 65.38 ± 0.67 | 69.86 ± 0.92 |
| T1 | L2 | relative10_shuffled | 89.14 ± 5.70 | 50.28 ± 2.57 | 50.75 ± 1.01 |
| T2 | L1 | whole_count | 95.16 ± 5.99 | 53.28 ± 1.24 | 55.02 ± 3.38 |
| T2 | L1 | fixed250_ordered | 100.00 ± 0.00 | 60.42 ± 0.24 | 59.49 ± 0.83 |
| T2 | L1 | fixed250_shuffled | 100.00 ± 0.00 | 36.40 ± 0.39 | 34.98 ± 0.27 |
| T2 | L1 | relative10_ordered | 99.84 ± 0.28 | 68.44 ± 0.62 | 70.16 ± 5.66 |
| T2 | L1 | relative10_shuffled | 93.83 ± 1.23 | 28.77 ± 0.94 | 28.91 ± 1.24 |
| T2 | L2 | whole_count | 85.69 ± 1.58 | 54.78 ± 2.32 | 59.81 ± 0.66 |
| T2 | L2 | fixed250_ordered | 99.55 ± 0.39 | 59.59 ± 2.95 | 58.21 ± 0.85 |
| T2 | L2 | fixed250_shuffled | 99.45 ± 0.66 | 52.09 ± 1.96 | 49.49 ± 1.13 |
| T2 | L2 | relative10_ordered | 98.58 ± 2.03 | 67.89 ± 3.02 | 67.11 ± 3.26 |
| T2 | L2 | relative10_shuffled | 94.92 ± 0.74 | 48.80 ± 2.64 | 50.40 ± 1.56 |
| T3 | L1 | whole_count | 92.93 ± 6.23 | 51.77 ± 0.64 | 52.00 ± 1.33 |
| T3 | L1 | fixed250_ordered | 100.00 ± 0.00 | 60.77 ± 2.11 | 59.27 ± 3.55 |
| T3 | L1 | fixed250_shuffled | 99.26 ± 0.64 | 36.51 ± 1.04 | 35.40 ± 0.31 |
| T3 | L1 | relative10_ordered | 99.51 ± 0.49 | 69.74 ± 0.79 | 74.78 ± 0.66 |
| T3 | L1 | relative10_shuffled | 98.78 ± 0.15 | 28.18 ± 1.02 | 29.17 ± 0.92 |
| T3 | L2 | whole_count | 87.87 ± 12.02 | 54.44 ± 2.07 | 51.35 ± 5.71 |
| T3 | L2 | fixed250_ordered | 96.03 ± 6.44 | 52.62 ± 1.73 | 51.46 ± 5.74 |
| T3 | L2 | fixed250_shuffled | 94.72 ± 6.99 | 48.19 ± 2.32 | 48.05 ± 3.16 |
| T3 | L2 | relative10_ordered | 93.35 ± 2.41 | 62.19 ± 2.12 | 66.90 ± 3.07 |
| T3 | L2 | relative10_shuffled | 88.72 ± 7.25 | 48.86 ± 3.16 | 50.80 ± 0.79 |
| T4 | L1 | whole_count | 96.53 ± 5.87 | 56.08 ± 1.10 | 55.36 ± 1.03 |
| T4 | L1 | fixed250_ordered | 100.00 ± 0.00 | 62.33 ± 3.40 | 59.90 ± 1.10 |
| T4 | L1 | fixed250_shuffled | 100.00 ± 0.00 | 39.47 ± 0.77 | 35.29 ± 0.80 |
| T4 | L1 | relative10_ordered | 100.00 ± 0.00 | 71.58 ± 1.40 | 70.23 ± 2.61 |
| T4 | L1 | relative10_shuffled | 99.01 ± 0.67 | 31.53 ± 0.20 | 29.27 ± 0.94 |
| T4 | L2 | whole_count | 93.42 ± 0.30 | 55.63 ± 3.26 | 56.36 ± 5.51 |
| T4 | L2 | fixed250_ordered | 99.37 ± 0.81 | 56.63 ± 5.18 | 51.70 ± 4.76 |
| T4 | L2 | fixed250_shuffled | 98.51 ± 0.98 | 49.96 ± 1.01 | 47.19 ± 1.04 |
| T4 | L2 | relative10_ordered | 99.37 ± 0.82 | 62.22 ± 2.61 | 62.98 ± 3.47 |
| T4 | L2 | relative10_shuffled | 96.96 ± 1.10 | 50.58 ± 2.44 | 49.63 ± 2.81 |
| D1 | L1 | whole_count | 98.39 ± 1.59 | 53.24 ± 2.62 | 49.81 ± 2.28 |
| D1 | L1 | fixed250_ordered | 100.00 ± 0.00 | 62.22 ± 2.01 | 61.26 ± 2.29 |
| D1 | L1 | fixed250_shuffled | 99.30 ± 0.73 | 39.33 ± 0.85 | 34.66 ± 1.83 |
| D1 | L1 | relative10_ordered | 100.00 ± 0.00 | 72.00 ± 1.12 | 70.52 ± 2.41 |
| D1 | L1 | relative10_shuffled | 98.45 ± 0.05 | 31.08 ± 0.82 | 29.07 ± 2.48 |
| D1 | L2 | whole_count | 93.39 ± 6.04 | 50.47 ± 1.88 | 49.00 ± 5.44 |
| D1 | L2 | fixed250_ordered | 94.49 ± 3.33 | 52.73 ± 1.81 | 56.97 ± 4.58 |
| D1 | L2 | fixed250_shuffled | 93.31 ± 8.07 | 41.64 ± 3.60 | 41.14 ± 3.15 |
| D1 | L2 | relative10_ordered | 95.34 ± 3.84 | 62.89 ± 3.49 | 64.75 ± 1.58 |
| D1 | L2 | relative10_shuffled | 89.12 ± 7.09 | 39.34 ± 3.83 | 39.01 ± 4.79 |
| D1 | L3 | whole_count | 94.43 ± 3.29 | 49.03 ± 6.39 | 49.63 ± 7.18 |
| D1 | L3 | fixed250_ordered | 93.54 ± 3.76 | 50.57 ± 4.34 | 52.26 ± 4.80 |
| D1 | L3 | fixed250_shuffled | 93.65 ± 6.84 | 45.07 ± 3.63 | 46.69 ± 5.62 |
| D1 | L3 | relative10_ordered | 95.37 ± 2.88 | 53.37 ± 2.85 | 54.72 ± 5.40 |
| D1 | L3 | relative10_shuffled | 94.72 ± 3.42 | 45.12 ± 5.86 | 48.14 ± 6.01 |

## Appendix B. Complete Per-Seed Probe Matrix

All 1,140 coordinates after within-seed shuffle averaging, from original Unity `main/aggregate/probe_seed_means.csv`. Values are BA percentages. This appendix preserves the complete per-seed result matrix without publishing the separate metric/provenance bundle.

### State: spike; decoder: no_bias

| Case | Layer | Aggregation | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 11 | 96.46 | 51.81 | 52.54 |
| O0 | L1 | fixed250_ordered | 11 | 96.34 | 52.18 | 56.50 |
| O0 | L1 | fixed250_shuffled | 11 | 93.51 | 40.58 | 42.40 |
| O0 | L1 | relative10_ordered | 11 | 96.60 | 68.23 | 69.65 |
| O0 | L1 | relative10_shuffled | 11 | 98.48 | 34.12 | 34.94 |
| O0 | L2 | whole_count | 11 | 93.14 | 55.88 | 54.44 |
| O0 | L2 | fixed250_ordered | 11 | 98.26 | 60.76 | 60.38 |
| O0 | L2 | fixed250_shuffled | 11 | 99.56 | 55.13 | 50.39 |
| O0 | L2 | relative10_ordered | 11 | 94.95 | 67.82 | 67.37 |
| O0 | L2 | relative10_shuffled | 11 | 97.26 | 51.26 | 50.43 |
| O1 | L1 | whole_count | 11 | 93.99 | 49.58 | 53.47 |
| O1 | L1 | fixed250_ordered | 11 | 95.36 | 49.28 | 54.50 |
| O1 | L1 | fixed250_shuffled | 11 | 95.85 | 43.68 | 42.76 |
| O1 | L1 | relative10_ordered | 11 | 95.24 | 65.03 | 70.19 |
| O1 | L1 | relative10_shuffled | 11 | 97.84 | 34.52 | 35.05 |
| O1 | L2 | whole_count | 11 | 91.43 | 47.44 | 54.61 |
| O1 | L2 | fixed250_ordered | 11 | 97.05 | 55.07 | 63.09 |
| O1 | L2 | fixed250_shuffled | 11 | 96.96 | 50.04 | 55.19 |
| O1 | L2 | relative10_ordered | 11 | 99.84 | 60.58 | 61.41 |
| O1 | L2 | relative10_shuffled | 11 | 98.76 | 48.43 | 50.68 |
| O2 | L1 | whole_count | 11 | 89.51 | 52.42 | 58.18 |
| O2 | L1 | fixed250_ordered | 11 | 99.53 | 56.78 | 54.73 |
| O2 | L1 | fixed250_shuffled | 11 | 97.93 | 43.91 | 44.95 |
| O2 | L1 | relative10_ordered | 11 | 97.76 | 70.03 | 72.71 |
| O2 | L1 | relative10_shuffled | 11 | 99.04 | 37.87 | 36.65 |
| O2 | L2 | whole_count | 11 | 99.84 | 58.85 | 49.73 |
| O2 | L2 | fixed250_ordered | 11 | 99.04 | 64.75 | 56.76 |
| O2 | L2 | fixed250_shuffled | 11 | 99.52 | 57.74 | 50.39 |
| O2 | L2 | relative10_ordered | 11 | 95.64 | 63.09 | 65.49 |
| O2 | L2 | relative10_shuffled | 11 | 99.68 | 55.95 | 51.41 |
| O3 | L1 | whole_count | 11 | 90.36 | 51.78 | 55.93 |
| O3 | L1 | fixed250_ordered | 11 | 96.48 | 52.55 | 57.40 |
| O3 | L1 | fixed250_shuffled | 11 | 88.51 | 44.46 | 43.01 |
| O3 | L1 | relative10_ordered | 11 | 97.32 | 66.98 | 70.77 |
| O3 | L1 | relative10_shuffled | 11 | 97.93 | 34.97 | 38.69 |
| O3 | L2 | whole_count | 11 | 92.55 | 60.22 | 56.95 |
| O3 | L2 | fixed250_ordered | 11 | 99.84 | 61.01 | 55.01 |
| O3 | L2 | fixed250_shuffled | 11 | 99.49 | 59.19 | 53.77 |
| O3 | L2 | relative10_ordered | 11 | 94.93 | 67.06 | 66.99 |
| O3 | L2 | relative10_shuffled | 11 | 99.71 | 56.31 | 51.88 |
| T1 | L1 | whole_count | 11 | 98.68 | 44.88 | 48.84 |
| T1 | L1 | fixed250_ordered | 11 | 94.27 | 49.66 | 53.85 |
| T1 | L1 | fixed250_shuffled | 11 | 90.97 | 39.85 | 39.15 |
| T1 | L1 | relative10_ordered | 11 | 96.32 | 69.33 | 69.70 |
| T1 | L1 | relative10_shuffled | 11 | 97.47 | 32.40 | 31.17 |
| T1 | L2 | whole_count | 11 | 91.17 | 52.57 | 58.50 |
| T1 | L2 | fixed250_ordered | 11 | 99.84 | 59.90 | 59.61 |
| T1 | L2 | fixed250_shuffled | 11 | 98.26 | 57.13 | 51.94 |
| T1 | L2 | relative10_ordered | 11 | 95.29 | 67.30 | 70.59 |
| T1 | L2 | relative10_shuffled | 11 | 98.26 | 50.54 | 52.54 |
| T2 | L1 | whole_count | 11 | 97.92 | 52.00 | 43.14 |
| T2 | L1 | fixed250_ordered | 11 | 100.00 | 52.17 | 50.54 |
| T2 | L1 | fixed250_shuffled | 11 | 96.04 | 39.69 | 41.08 |
| T2 | L1 | relative10_ordered | 11 | 99.41 | 64.73 | 70.62 |
| T2 | L1 | relative10_shuffled | 11 | 97.03 | 32.79 | 32.99 |
| T2 | L2 | whole_count | 11 | 86.05 | 53.69 | 55.03 |
| T2 | L2 | fixed250_ordered | 11 | 99.65 | 60.97 | 58.68 |
| T2 | L2 | fixed250_shuffled | 11 | 99.45 | 56.73 | 51.05 |
| T2 | L2 | relative10_ordered | 11 | 93.31 | 68.78 | 73.54 |
| T2 | L2 | relative10_shuffled | 11 | 97.09 | 50.47 | 49.41 |
| T3 | L1 | whole_count | 11 | 93.45 | 47.05 | 47.35 |
| T3 | L1 | fixed250_ordered | 11 | 98.69 | 50.99 | 52.23 |
| T3 | L1 | fixed250_shuffled | 11 | 94.73 | 41.13 | 39.76 |
| T3 | L1 | relative10_ordered | 11 | 98.68 | 65.10 | 67.94 |
| T3 | L1 | relative10_shuffled | 11 | 94.78 | 29.05 | 32.89 |
| T3 | L2 | whole_count | 11 | 93.19 | 54.19 | 52.17 |
| T3 | L2 | fixed250_ordered | 11 | 99.65 | 56.13 | 55.75 |
| T3 | L2 | fixed250_shuffled | 11 | 98.72 | 54.26 | 50.09 |
| T3 | L2 | relative10_ordered | 11 | 95.38 | 68.02 | 64.67 |
| T3 | L2 | relative10_shuffled | 11 | 96.89 | 50.41 | 47.89 |
| T4 | L1 | whole_count | 11 | 95.92 | 53.09 | 50.44 |
| T4 | L1 | fixed250_ordered | 11 | 99.02 | 52.51 | 50.37 |
| T4 | L1 | fixed250_shuffled | 11 | 97.11 | 40.45 | 43.87 |
| T4 | L1 | relative10_ordered | 11 | 96.33 | 68.67 | 69.31 |
| T4 | L1 | relative10_shuffled | 11 | 97.64 | 34.20 | 35.52 |
| T4 | L2 | whole_count | 11 | 95.79 | 53.98 | 53.29 |
| T4 | L2 | fixed250_ordered | 11 | 95.56 | 54.98 | 55.44 |
| T4 | L2 | fixed250_shuffled | 11 | 98.37 | 52.61 | 49.33 |
| T4 | L2 | relative10_ordered | 11 | 96.10 | 59.12 | 62.55 |
| T4 | L2 | relative10_shuffled | 11 | 96.82 | 51.92 | 51.31 |
| D1 | L1 | whole_count | 11 | 78.34 | 43.16 | 45.00 |
| D1 | L1 | fixed250_ordered | 11 | 91.44 | 49.77 | 47.82 |
| D1 | L1 | fixed250_shuffled | 11 | 92.07 | 32.83 | 36.60 |
| D1 | L1 | relative10_ordered | 11 | 94.49 | 65.29 | 66.41 |
| D1 | L1 | relative10_shuffled | 11 | 93.62 | 28.13 | 25.92 |
| D1 | L2 | whole_count | 11 | 88.55 | 48.18 | 53.23 |
| D1 | L2 | fixed250_ordered | 11 | 98.37 | 53.82 | 52.22 |
| D1 | L2 | fixed250_shuffled | 11 | 95.66 | 44.62 | 46.34 |
| D1 | L2 | relative10_ordered | 11 | 97.39 | 60.55 | 67.12 |
| D1 | L2 | relative10_shuffled | 11 | 95.53 | 42.62 | 44.68 |
| D1 | L3 | whole_count | 11 | 95.69 | 51.60 | 47.54 |
| D1 | L3 | fixed250_ordered | 11 | 98.36 | 52.94 | 45.28 |
| D1 | L3 | fixed250_shuffled | 11 | 98.18 | 47.63 | 42.54 |
| D1 | L3 | relative10_ordered | 11 | 98.05 | 52.98 | 54.04 |
| D1 | L3 | relative10_shuffled | 11 | 96.93 | 49.63 | 45.42 |
| O0 | L1 | whole_count | 23 | 91.81 | 52.09 | 56.00 |
| O0 | L1 | fixed250_ordered | 23 | 95.41 | 54.84 | 57.63 |
| O0 | L1 | fixed250_shuffled | 23 | 96.99 | 43.75 | 46.98 |
| O0 | L1 | relative10_ordered | 23 | 96.42 | 68.96 | 73.03 |
| O0 | L1 | relative10_shuffled | 23 | 98.65 | 36.16 | 35.25 |
| O0 | L2 | whole_count | 23 | 98.77 | 60.45 | 55.89 |
| O0 | L2 | fixed250_ordered | 23 | 99.84 | 59.09 | 55.52 |
| O0 | L2 | fixed250_shuffled | 23 | 99.38 | 55.15 | 56.25 |
| O0 | L2 | relative10_ordered | 23 | 99.84 | 68.50 | 66.20 |
| O0 | L2 | relative10_shuffled | 23 | 98.62 | 52.48 | 54.57 |
| O1 | L1 | whole_count | 23 | 87.39 | 47.86 | 59.59 |
| O1 | L1 | fixed250_ordered | 23 | 93.40 | 56.81 | 55.84 |
| O1 | L1 | fixed250_shuffled | 23 | 95.27 | 44.97 | 47.77 |
| O1 | L1 | relative10_ordered | 23 | 99.68 | 68.64 | 68.71 |
| O1 | L1 | relative10_shuffled | 23 | 95.98 | 35.36 | 39.63 |
| O1 | L2 | whole_count | 23 | 94.68 | 52.56 | 57.90 |
| O1 | L2 | fixed250_ordered | 23 | 96.64 | 57.93 | 62.81 |
| O1 | L2 | fixed250_shuffled | 23 | 96.22 | 53.65 | 56.84 |
| O1 | L2 | relative10_ordered | 23 | 94.36 | 64.15 | 75.06 |
| O1 | L2 | relative10_shuffled | 23 | 97.18 | 48.06 | 53.34 |
| O2 | L1 | whole_count | 23 | 97.68 | 54.85 | 60.19 |
| O2 | L1 | fixed250_ordered | 23 | 95.37 | 58.80 | 56.97 |
| O2 | L1 | fixed250_shuffled | 23 | 97.00 | 45.71 | 49.63 |
| O2 | L1 | relative10_ordered | 23 | 99.51 | 69.06 | 71.29 |
| O2 | L1 | relative10_shuffled | 23 | 97.99 | 39.53 | 39.76 |
| O2 | L2 | whole_count | 23 | 96.71 | 59.07 | 58.20 |
| O2 | L2 | fixed250_ordered | 23 | 98.36 | 63.89 | 59.04 |
| O2 | L2 | fixed250_shuffled | 23 | 97.11 | 55.65 | 52.62 |
| O2 | L2 | relative10_ordered | 23 | 99.84 | 67.81 | 69.88 |
| O2 | L2 | relative10_shuffled | 23 | 99.45 | 56.09 | 53.56 |
| O3 | L1 | whole_count | 23 | 89.76 | 52.19 | 60.78 |
| O3 | L1 | fixed250_ordered | 23 | 99.34 | 55.66 | 58.18 |
| O3 | L1 | fixed250_shuffled | 23 | 96.73 | 44.74 | 47.87 |
| O3 | L1 | relative10_ordered | 23 | 99.52 | 65.38 | 70.60 |
| O3 | L1 | relative10_shuffled | 23 | 97.73 | 38.40 | 39.77 |
| O3 | L2 | whole_count | 23 | 96.03 | 51.72 | 59.50 |
| O3 | L2 | fixed250_ordered | 23 | 98.70 | 60.50 | 57.95 |
| O3 | L2 | fixed250_shuffled | 23 | 97.71 | 53.40 | 56.29 |
| O3 | L2 | relative10_ordered | 23 | 99.51 | 70.71 | 72.30 |
| O3 | L2 | relative10_shuffled | 23 | 99.53 | 52.04 | 53.24 |
| T1 | L1 | whole_count | 23 | 93.47 | 46.88 | 48.37 |
| T1 | L1 | fixed250_ordered | 23 | 92.37 | 51.59 | 55.20 |
| T1 | L1 | fixed250_shuffled | 23 | 95.28 | 40.63 | 41.51 |
| T1 | L1 | relative10_ordered | 23 | 94.81 | 64.96 | 73.44 |
| T1 | L1 | relative10_shuffled | 23 | 97.21 | 32.39 | 30.53 |
| T1 | L2 | whole_count | 23 | 93.29 | 59.83 | 57.77 |
| T1 | L2 | fixed250_ordered | 23 | 97.73 | 61.99 | 66.46 |
| T1 | L2 | fixed250_shuffled | 23 | 93.53 | 54.94 | 52.15 |
| T1 | L2 | relative10_ordered | 23 | 97.59 | 73.66 | 71.81 |
| T1 | L2 | relative10_shuffled | 23 | 98.21 | 50.43 | 50.16 |
| T2 | L1 | whole_count | 23 | 84.17 | 51.90 | 53.81 |
| T2 | L1 | fixed250_ordered | 23 | 94.17 | 50.76 | 51.87 |
| T2 | L1 | fixed250_shuffled | 23 | 96.33 | 39.61 | 41.13 |
| T2 | L1 | relative10_ordered | 23 | 95.80 | 66.74 | 70.89 |
| T2 | L1 | relative10_shuffled | 23 | 94.66 | 33.10 | 33.66 |
| T2 | L2 | whole_count | 23 | 92.00 | 55.14 | 58.44 |
| T2 | L2 | fixed250_ordered | 23 | 99.84 | 61.49 | 66.22 |
| T2 | L2 | fixed250_shuffled | 23 | 98.95 | 56.13 | 55.34 |
| T2 | L2 | relative10_ordered | 23 | 98.24 | 71.67 | 70.96 |
| T2 | L2 | relative10_shuffled | 23 | 99.13 | 50.82 | 50.64 |
| T3 | L1 | whole_count | 23 | 85.21 | 45.77 | 46.04 |
| T3 | L1 | fixed250_ordered | 23 | 86.26 | 44.18 | 48.40 |
| T3 | L1 | fixed250_shuffled | 23 | 92.86 | 34.47 | 37.25 |
| T3 | L1 | relative10_ordered | 23 | 92.09 | 61.91 | 70.05 |
| T3 | L1 | relative10_shuffled | 23 | 97.23 | 28.51 | 30.30 |
| T3 | L2 | whole_count | 23 | 73.14 | 47.55 | 56.54 |
| T3 | L2 | fixed250_ordered | 23 | 93.71 | 56.49 | 61.88 |
| T3 | L2 | fixed250_shuffled | 23 | 92.73 | 51.46 | 51.76 |
| T3 | L2 | relative10_ordered | 23 | 97.86 | 66.64 | 71.95 |
| T3 | L2 | relative10_shuffled | 23 | 95.70 | 44.42 | 45.96 |
| T4 | L1 | whole_count | 23 | 86.70 | 49.39 | 56.87 |
| T4 | L1 | fixed250_ordered | 23 | 93.68 | 59.45 | 56.65 |
| T4 | L1 | fixed250_shuffled | 23 | 95.42 | 48.04 | 46.50 |
| T4 | L1 | relative10_ordered | 23 | 96.41 | 66.81 | 71.10 |
| T4 | L1 | relative10_shuffled | 23 | 97.10 | 35.96 | 37.38 |
| T4 | L2 | whole_count | 23 | 98.19 | 52.67 | 54.40 |
| T4 | L2 | fixed250_ordered | 23 | 97.89 | 56.90 | 61.42 |
| T4 | L2 | fixed250_shuffled | 23 | 94.27 | 52.45 | 51.06 |
| T4 | L2 | relative10_ordered | 23 | 98.84 | 68.24 | 67.07 |
| T4 | L2 | relative10_shuffled | 23 | 96.52 | 50.31 | 54.49 |
| D1 | L1 | whole_count | 23 | 91.38 | 40.54 | 43.69 |
| D1 | L1 | fixed250_ordered | 23 | 98.38 | 47.00 | 43.57 |
| D1 | L1 | fixed250_shuffled | 23 | 93.74 | 35.44 | 34.51 |
| D1 | L1 | relative10_ordered | 23 | 97.20 | 64.87 | 66.77 |
| D1 | L1 | relative10_shuffled | 23 | 95.22 | 27.73 | 27.99 |
| D1 | L2 | whole_count | 23 | 97.27 | 49.57 | 55.35 |
| D1 | L2 | fixed250_ordered | 23 | 98.52 | 55.13 | 55.03 |
| D1 | L2 | fixed250_shuffled | 23 | 94.55 | 45.40 | 48.60 |
| D1 | L2 | relative10_ordered | 23 | 99.00 | 62.01 | 64.06 |
| D1 | L2 | relative10_shuffled | 23 | 97.71 | 43.38 | 44.70 |
| D1 | L3 | whole_count | 23 | 97.26 | 50.41 | 48.72 |
| D1 | L3 | fixed250_ordered | 23 | 99.00 | 49.01 | 49.67 |
| D1 | L3 | fixed250_shuffled | 23 | 98.20 | 45.55 | 47.96 |
| D1 | L3 | relative10_ordered | 23 | 97.57 | 52.56 | 58.84 |
| D1 | L3 | relative10_shuffled | 23 | 97.30 | 47.85 | 50.40 |
| O0 | L1 | whole_count | 37 | 97.20 | 53.26 | 56.70 |
| O0 | L1 | fixed250_ordered | 37 | 99.17 | 55.99 | 54.72 |
| O0 | L1 | fixed250_shuffled | 37 | 96.71 | 40.05 | 45.19 |
| O0 | L1 | relative10_ordered | 37 | 90.22 | 72.30 | 72.52 |
| O0 | L1 | relative10_shuffled | 37 | 97.80 | 34.11 | 36.30 |
| O0 | L2 | whole_count | 37 | 96.76 | 55.49 | 62.40 |
| O0 | L2 | fixed250_ordered | 37 | 99.68 | 56.62 | 59.95 |
| O0 | L2 | fixed250_shuffled | 37 | 97.87 | 49.42 | 54.73 |
| O0 | L2 | relative10_ordered | 37 | 99.68 | 63.06 | 67.54 |
| O0 | L2 | relative10_shuffled | 37 | 97.19 | 48.53 | 57.73 |
| O1 | L1 | whole_count | 37 | 95.08 | 50.08 | 52.71 |
| O1 | L1 | fixed250_ordered | 37 | 94.46 | 55.96 | 56.76 |
| O1 | L1 | fixed250_shuffled | 37 | 95.18 | 42.57 | 45.06 |
| O1 | L1 | relative10_ordered | 37 | 88.51 | 65.61 | 73.33 |
| O1 | L1 | relative10_shuffled | 37 | 95.15 | 32.96 | 36.61 |
| O1 | L2 | whole_count | 37 | 93.75 | 47.44 | 63.14 |
| O1 | L2 | fixed250_ordered | 37 | 98.85 | 59.64 | 65.31 |
| O1 | L2 | fixed250_shuffled | 37 | 94.74 | 51.50 | 55.92 |
| O1 | L2 | relative10_ordered | 37 | 94.61 | 65.43 | 72.90 |
| O1 | L2 | relative10_shuffled | 37 | 96.91 | 46.70 | 54.74 |
| O2 | L1 | whole_count | 37 | 87.13 | 49.34 | 60.51 |
| O2 | L1 | fixed250_ordered | 37 | 96.81 | 55.85 | 57.07 |
| O2 | L1 | fixed250_shuffled | 37 | 92.94 | 43.63 | 43.72 |
| O2 | L1 | relative10_ordered | 37 | 97.80 | 71.14 | 72.66 |
| O2 | L1 | relative10_shuffled | 37 | 96.95 | 35.18 | 39.83 |
| O2 | L2 | whole_count | 37 | 93.27 | 52.16 | 54.25 |
| O2 | L2 | fixed250_ordered | 37 | 99.68 | 59.30 | 58.08 |
| O2 | L2 | fixed250_shuffled | 37 | 99.39 | 53.68 | 51.69 |
| O2 | L2 | relative10_ordered | 37 | 95.99 | 62.88 | 68.01 |
| O2 | L2 | relative10_shuffled | 37 | 96.57 | 49.98 | 52.10 |
| O3 | L1 | whole_count | 37 | 97.36 | 53.09 | 54.81 |
| O3 | L1 | fixed250_ordered | 37 | 96.27 | 56.61 | 58.20 |
| O3 | L1 | fixed250_shuffled | 37 | 92.98 | 41.19 | 43.43 |
| O3 | L1 | relative10_ordered | 37 | 88.78 | 70.08 | 73.04 |
| O3 | L1 | relative10_shuffled | 37 | 98.74 | 35.29 | 36.06 |
| O3 | L2 | whole_count | 37 | 99.03 | 53.20 | 59.00 |
| O3 | L2 | fixed250_ordered | 37 | 99.68 | 55.29 | 61.15 |
| O3 | L2 | fixed250_shuffled | 37 | 99.52 | 52.00 | 51.55 |
| O3 | L2 | relative10_ordered | 37 | 96.64 | 61.37 | 70.06 |
| O3 | L2 | relative10_shuffled | 37 | 99.32 | 50.22 | 52.66 |
| T1 | L1 | whole_count | 37 | 87.25 | 46.14 | 49.04 |
| T1 | L1 | fixed250_ordered | 37 | 94.15 | 47.92 | 52.64 |
| T1 | L1 | fixed250_shuffled | 37 | 95.83 | 38.30 | 39.83 |
| T1 | L1 | relative10_ordered | 37 | 94.90 | 63.90 | 72.44 |
| T1 | L1 | relative10_shuffled | 37 | 96.41 | 32.81 | 30.78 |
| T1 | L2 | whole_count | 37 | 93.76 | 47.87 | 58.60 |
| T1 | L2 | fixed250_ordered | 37 | 99.49 | 55.95 | 58.37 |
| T1 | L2 | fixed250_shuffled | 37 | 93.65 | 51.42 | 54.42 |
| T1 | L2 | relative10_ordered | 37 | 93.24 | 67.51 | 69.82 |
| T1 | L2 | relative10_shuffled | 37 | 97.06 | 46.82 | 52.07 |
| T2 | L1 | whole_count | 37 | 95.99 | 47.61 | 53.90 |
| T2 | L1 | fixed250_ordered | 37 | 93.97 | 49.17 | 56.93 |
| T2 | L1 | fixed250_shuffled | 37 | 85.95 | 39.04 | 39.48 |
| T2 | L1 | relative10_ordered | 37 | 96.02 | 66.55 | 71.06 |
| T2 | L1 | relative10_shuffled | 37 | 95.66 | 31.97 | 33.67 |
| T2 | L2 | whole_count | 37 | 99.68 | 56.06 | 56.70 |
| T2 | L2 | fixed250_ordered | 37 | 99.84 | 56.76 | 55.97 |
| T2 | L2 | fixed250_shuffled | 37 | 98.97 | 49.92 | 52.90 |
| T2 | L2 | relative10_ordered | 37 | 99.52 | 67.53 | 68.86 |
| T2 | L2 | relative10_shuffled | 37 | 94.97 | 45.66 | 51.56 |
| T3 | L1 | whole_count | 37 | 83.73 | 46.47 | 45.00 |
| T3 | L1 | fixed250_ordered | 37 | 92.67 | 47.72 | 49.75 |
| T3 | L1 | fixed250_shuffled | 37 | 95.13 | 38.08 | 41.18 |
| T3 | L1 | relative10_ordered | 37 | 94.80 | 60.11 | 70.73 |
| T3 | L1 | relative10_shuffled | 37 | 94.96 | 30.97 | 29.41 |
| T3 | L2 | whole_count | 37 | 98.66 | 48.36 | 54.73 |
| T3 | L2 | fixed250_ordered | 37 | 89.37 | 50.34 | 57.11 |
| T3 | L2 | fixed250_shuffled | 37 | 91.22 | 46.50 | 51.75 |
| T3 | L2 | relative10_ordered | 37 | 98.85 | 64.68 | 67.81 |
| T3 | L2 | relative10_shuffled | 37 | 98.35 | 45.79 | 53.06 |
| T4 | L1 | whole_count | 37 | 96.90 | 47.03 | 53.51 |
| T4 | L1 | fixed250_ordered | 37 | 95.11 | 52.44 | 51.46 |
| T4 | L1 | fixed250_shuffled | 37 | 97.08 | 37.56 | 44.02 |
| T4 | L1 | relative10_ordered | 37 | 96.52 | 70.77 | 68.51 |
| T4 | L1 | relative10_shuffled | 37 | 98.21 | 31.11 | 34.87 |
| T4 | L2 | whole_count | 37 | 99.00 | 51.80 | 57.71 |
| T4 | L2 | fixed250_ordered | 37 | 99.16 | 54.14 | 62.34 |
| T4 | L2 | fixed250_shuffled | 37 | 99.09 | 51.50 | 56.82 |
| T4 | L2 | relative10_ordered | 37 | 99.20 | 60.29 | 68.63 |
| T4 | L2 | relative10_shuffled | 37 | 98.30 | 47.49 | 56.89 |
| D1 | L1 | whole_count | 37 | 71.33 | 38.38 | 42.85 |
| D1 | L1 | fixed250_ordered | 37 | 88.04 | 48.88 | 49.32 |
| D1 | L1 | fixed250_shuffled | 37 | 88.84 | 32.10 | 34.85 |
| D1 | L1 | relative10_ordered | 37 | 93.19 | 59.20 | 63.20 |
| D1 | L1 | relative10_shuffled | 37 | 93.04 | 27.49 | 24.69 |
| D1 | L2 | whole_count | 37 | 90.69 | 40.71 | 43.51 |
| D1 | L2 | fixed250_ordered | 37 | 91.30 | 47.61 | 55.73 |
| D1 | L2 | fixed250_shuffled | 37 | 92.57 | 40.85 | 42.22 |
| D1 | L2 | relative10_ordered | 37 | 94.60 | 59.53 | 65.25 |
| D1 | L2 | relative10_shuffled | 37 | 93.38 | 37.13 | 37.49 |
| D1 | L3 | whole_count | 37 | 88.98 | 46.26 | 47.97 |
| D1 | L3 | fixed250_ordered | 37 | 92.64 | 46.75 | 49.03 |
| D1 | L3 | fixed250_shuffled | 37 | 86.20 | 41.53 | 44.32 |
| D1 | L3 | relative10_ordered | 37 | 96.13 | 50.87 | 53.92 |
| D1 | L3 | relative10_shuffled | 37 | 88.37 | 40.01 | 43.43 |

### State: spike; decoder: affine

| Case | Layer | Aggregation | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 11 | 96.28 | 55.14 | 56.11 |
| O0 | L1 | fixed250_ordered | 11 | 98.29 | 51.41 | 55.97 |
| O0 | L1 | fixed250_shuffled | 11 | 98.13 | 38.72 | 43.96 |
| O0 | L1 | relative10_ordered | 11 | 98.58 | 63.53 | 67.68 |
| O0 | L1 | relative10_shuffled | 11 | 97.17 | 38.43 | 39.22 |
| O0 | L2 | whole_count | 11 | 96.22 | 59.02 | 56.18 |
| O0 | L2 | fixed250_ordered | 11 | 99.65 | 61.25 | 57.22 |
| O0 | L2 | fixed250_shuffled | 11 | 99.90 | 54.30 | 52.40 |
| O0 | L2 | relative10_ordered | 11 | 98.58 | 65.07 | 66.60 |
| O0 | L2 | relative10_shuffled | 11 | 96.48 | 52.18 | 54.41 |
| O1 | L1 | whole_count | 11 | 94.33 | 51.96 | 54.68 |
| O1 | L1 | fixed250_ordered | 11 | 97.79 | 47.18 | 57.76 |
| O1 | L1 | fixed250_shuffled | 11 | 97.46 | 39.50 | 40.84 |
| O1 | L1 | relative10_ordered | 11 | 97.61 | 62.55 | 65.41 |
| O1 | L1 | relative10_shuffled | 11 | 97.00 | 38.68 | 38.38 |
| O1 | L2 | whole_count | 11 | 89.25 | 48.36 | 58.61 |
| O1 | L2 | fixed250_ordered | 11 | 99.01 | 54.44 | 61.62 |
| O1 | L2 | fixed250_shuffled | 11 | 96.14 | 48.46 | 54.58 |
| O1 | L2 | relative10_ordered | 11 | 97.47 | 59.96 | 69.17 |
| O1 | L2 | relative10_shuffled | 11 | 94.98 | 48.11 | 54.65 |
| O2 | L1 | whole_count | 11 | 89.31 | 59.94 | 61.07 |
| O2 | L1 | fixed250_ordered | 11 | 99.20 | 55.42 | 56.22 |
| O2 | L1 | fixed250_shuffled | 11 | 99.23 | 42.93 | 43.91 |
| O2 | L1 | relative10_ordered | 11 | 99.17 | 68.27 | 70.97 |
| O2 | L1 | relative10_shuffled | 11 | 98.41 | 41.60 | 40.54 |
| O2 | L2 | whole_count | 11 | 96.64 | 58.85 | 54.61 |
| O2 | L2 | fixed250_ordered | 11 | 98.58 | 64.61 | 56.17 |
| O2 | L2 | fixed250_shuffled | 11 | 100.00 | 56.43 | 50.08 |
| O2 | L2 | relative10_ordered | 11 | 94.86 | 63.75 | 63.92 |
| O2 | L2 | relative10_shuffled | 11 | 98.14 | 56.73 | 50.64 |
| O3 | L1 | whole_count | 11 | 90.51 | 56.16 | 56.42 |
| O3 | L1 | fixed250_ordered | 11 | 98.93 | 51.61 | 56.75 |
| O3 | L1 | fixed250_shuffled | 11 | 98.52 | 41.83 | 44.92 |
| O3 | L1 | relative10_ordered | 11 | 98.58 | 66.68 | 67.39 |
| O3 | L1 | relative10_shuffled | 11 | 95.82 | 39.53 | 43.52 |
| O3 | L2 | whole_count | 11 | 92.81 | 61.41 | 59.51 |
| O3 | L2 | fixed250_ordered | 11 | 98.43 | 65.10 | 56.34 |
| O3 | L2 | fixed250_shuffled | 11 | 99.62 | 59.21 | 51.72 |
| O3 | L2 | relative10_ordered | 11 | 98.58 | 65.27 | 62.94 |
| O3 | L2 | relative10_shuffled | 11 | 98.32 | 56.97 | 55.68 |
| T1 | L1 | whole_count | 11 | 95.22 | 49.21 | 52.92 |
| T1 | L1 | fixed250_ordered | 11 | 100.00 | 50.08 | 53.69 |
| T1 | L1 | fixed250_shuffled | 11 | 98.81 | 37.04 | 38.85 |
| T1 | L1 | relative10_ordered | 11 | 98.26 | 66.06 | 65.15 |
| T1 | L1 | relative10_shuffled | 11 | 95.12 | 34.93 | 37.06 |
| T1 | L2 | whole_count | 11 | 90.06 | 56.17 | 56.57 |
| T1 | L2 | fixed250_ordered | 11 | 98.04 | 60.25 | 63.26 |
| T1 | L2 | fixed250_shuffled | 11 | 99.76 | 57.14 | 50.86 |
| T1 | L2 | relative10_ordered | 11 | 100.00 | 65.63 | 63.25 |
| T1 | L2 | relative10_shuffled | 11 | 95.24 | 51.78 | 53.43 |
| T2 | L1 | whole_count | 11 | 87.56 | 55.65 | 53.71 |
| T2 | L1 | fixed250_ordered | 11 | 98.57 | 52.16 | 54.23 |
| T2 | L1 | fixed250_shuffled | 11 | 96.27 | 38.84 | 42.04 |
| T2 | L1 | relative10_ordered | 11 | 98.26 | 66.38 | 69.70 |
| T2 | L1 | relative10_shuffled | 11 | 98.74 | 36.21 | 35.41 |
| T2 | L2 | whole_count | 11 | 86.79 | 58.50 | 58.38 |
| T2 | L2 | fixed250_ordered | 11 | 100.00 | 60.56 | 56.93 |
| T2 | L2 | fixed250_shuffled | 11 | 99.10 | 57.67 | 50.74 |
| T2 | L2 | relative10_ordered | 11 | 100.00 | 67.49 | 67.68 |
| T2 | L2 | relative10_shuffled | 11 | 91.99 | 53.22 | 51.76 |
| T3 | L1 | whole_count | 11 | 93.54 | 52.15 | 54.04 |
| T3 | L1 | fixed250_ordered | 11 | 97.64 | 50.72 | 53.24 |
| T3 | L1 | fixed250_shuffled | 11 | 97.88 | 37.88 | 39.74 |
| T3 | L1 | relative10_ordered | 11 | 99.65 | 62.42 | 65.50 |
| T3 | L1 | relative10_shuffled | 11 | 93.01 | 33.20 | 38.18 |
| T3 | L2 | whole_count | 11 | 91.86 | 52.44 | 55.96 |
| T3 | L2 | fixed250_ordered | 11 | 99.65 | 57.58 | 58.09 |
| T3 | L2 | fixed250_shuffled | 11 | 99.41 | 54.00 | 51.50 |
| T3 | L2 | relative10_ordered | 11 | 97.59 | 63.29 | 64.09 |
| T3 | L2 | relative10_shuffled | 11 | 94.36 | 50.58 | 54.00 |
| T4 | L1 | whole_count | 11 | 96.20 | 51.26 | 53.69 |
| T4 | L1 | fixed250_ordered | 11 | 99.65 | 53.77 | 49.71 |
| T4 | L1 | fixed250_shuffled | 11 | 98.03 | 36.91 | 41.66 |
| T4 | L1 | relative10_ordered | 11 | 98.27 | 66.35 | 67.20 |
| T4 | L1 | relative10_shuffled | 11 | 93.87 | 36.86 | 42.24 |
| T4 | L2 | whole_count | 11 | 97.31 | 55.30 | 50.51 |
| T4 | L2 | fixed250_ordered | 11 | 98.11 | 56.36 | 58.59 |
| T4 | L2 | fixed250_shuffled | 11 | 99.52 | 53.05 | 49.48 |
| T4 | L2 | relative10_ordered | 11 | 95.41 | 59.43 | 59.34 |
| T4 | L2 | relative10_shuffled | 11 | 97.28 | 50.98 | 53.55 |
| D1 | L1 | whole_count | 11 | 92.13 | 45.59 | 43.45 |
| D1 | L1 | fixed250_ordered | 11 | 95.90 | 44.83 | 52.57 |
| D1 | L1 | fixed250_shuffled | 11 | 95.69 | 28.55 | 35.49 |
| D1 | L1 | relative10_ordered | 11 | 97.44 | 61.65 | 65.97 |
| D1 | L1 | relative10_shuffled | 11 | 84.92 | 30.47 | 31.65 |
| D1 | L2 | whole_count | 11 | 94.40 | 50.10 | 51.50 |
| D1 | L2 | fixed250_ordered | 11 | 95.99 | 51.10 | 54.64 |
| D1 | L2 | fixed250_shuffled | 11 | 96.29 | 43.22 | 46.21 |
| D1 | L2 | relative10_ordered | 11 | 97.11 | 59.02 | 65.77 |
| D1 | L2 | relative10_shuffled | 11 | 94.07 | 44.16 | 48.95 |
| D1 | L3 | whole_count | 11 | 96.47 | 52.15 | 48.77 |
| D1 | L3 | fixed250_ordered | 11 | 98.84 | 53.04 | 45.40 |
| D1 | L3 | fixed250_shuffled | 11 | 98.27 | 48.72 | 44.80 |
| D1 | L3 | relative10_ordered | 11 | 96.28 | 53.18 | 50.39 |
| D1 | L3 | relative10_shuffled | 11 | 98.12 | 47.83 | 44.63 |
| O0 | L1 | whole_count | 23 | 89.84 | 57.50 | 64.64 |
| O0 | L1 | fixed250_ordered | 23 | 98.27 | 52.44 | 55.88 |
| O0 | L1 | fixed250_shuffled | 23 | 98.94 | 41.82 | 46.62 |
| O0 | L1 | relative10_ordered | 23 | 98.58 | 66.31 | 71.18 |
| O0 | L1 | relative10_shuffled | 23 | 96.60 | 37.59 | 41.64 |
| O0 | L2 | whole_count | 23 | 98.59 | 58.83 | 58.55 |
| O0 | L2 | fixed250_ordered | 23 | 100.00 | 61.31 | 53.64 |
| O0 | L2 | fixed250_shuffled | 23 | 99.59 | 55.65 | 52.42 |
| O0 | L2 | relative10_ordered | 23 | 99.06 | 67.12 | 69.17 |
| O0 | L2 | relative10_shuffled | 23 | 98.49 | 53.93 | 55.75 |
| O1 | L1 | whole_count | 23 | 94.78 | 51.55 | 58.15 |
| O1 | L1 | fixed250_ordered | 23 | 99.16 | 52.16 | 47.82 |
| O1 | L1 | fixed250_shuffled | 23 | 99.41 | 37.45 | 38.58 |
| O1 | L1 | relative10_ordered | 23 | 97.58 | 64.07 | 70.73 |
| O1 | L1 | relative10_shuffled | 23 | 93.84 | 35.73 | 43.49 |
| O1 | L2 | whole_count | 23 | 94.40 | 56.01 | 62.86 |
| O1 | L2 | fixed250_ordered | 23 | 95.99 | 54.86 | 63.56 |
| O1 | L2 | fixed250_shuffled | 23 | 98.16 | 51.59 | 53.96 |
| O1 | L2 | relative10_ordered | 23 | 98.53 | 60.18 | 73.19 |
| O1 | L2 | relative10_shuffled | 23 | 94.71 | 48.67 | 57.50 |
| O2 | L1 | whole_count | 23 | 97.04 | 56.52 | 63.51 |
| O2 | L1 | fixed250_ordered | 23 | 94.34 | 53.51 | 57.23 |
| O2 | L1 | fixed250_shuffled | 23 | 98.16 | 42.85 | 48.19 |
| O2 | L1 | relative10_ordered | 23 | 98.85 | 66.88 | 72.48 |
| O2 | L1 | relative10_shuffled | 23 | 96.45 | 41.08 | 43.27 |
| O2 | L2 | whole_count | 23 | 99.84 | 59.64 | 54.18 |
| O2 | L2 | fixed250_ordered | 23 | 99.84 | 65.79 | 57.75 |
| O2 | L2 | fixed250_shuffled | 23 | 99.60 | 55.99 | 55.53 |
| O2 | L2 | relative10_ordered | 23 | 98.90 | 66.62 | 74.47 |
| O2 | L2 | relative10_shuffled | 23 | 98.29 | 56.34 | 54.80 |
| O3 | L1 | whole_count | 23 | 97.83 | 55.70 | 59.56 |
| O3 | L1 | fixed250_ordered | 23 | 98.43 | 52.51 | 58.07 |
| O3 | L1 | fixed250_shuffled | 23 | 99.35 | 42.12 | 47.43 |
| O3 | L1 | relative10_ordered | 23 | 99.01 | 63.14 | 72.43 |
| O3 | L1 | relative10_shuffled | 23 | 95.76 | 39.61 | 42.64 |
| O3 | L2 | whole_count | 23 | 95.66 | 51.92 | 61.64 |
| O3 | L2 | fixed250_ordered | 23 | 99.84 | 61.63 | 57.51 |
| O3 | L2 | fixed250_shuffled | 23 | 98.90 | 53.46 | 55.46 |
| O3 | L2 | relative10_ordered | 23 | 99.84 | 67.53 | 72.94 |
| O3 | L2 | relative10_shuffled | 23 | 98.27 | 52.41 | 53.61 |
| T1 | L1 | whole_count | 23 | 83.84 | 51.20 | 52.57 |
| T1 | L1 | fixed250_ordered | 23 | 89.54 | 48.90 | 55.20 |
| T1 | L1 | fixed250_shuffled | 23 | 97.86 | 38.91 | 40.79 |
| T1 | L1 | relative10_ordered | 23 | 97.76 | 63.26 | 71.94 |
| T1 | L1 | relative10_shuffled | 23 | 91.42 | 34.41 | 35.09 |
| T1 | L2 | whole_count | 23 | 93.53 | 59.81 | 58.69 |
| T1 | L2 | fixed250_ordered | 23 | 97.12 | 60.99 | 67.50 |
| T1 | L2 | fixed250_shuffled | 23 | 98.25 | 53.85 | 54.90 |
| T1 | L2 | relative10_ordered | 23 | 99.81 | 70.18 | 65.11 |
| T1 | L2 | relative10_shuffled | 23 | 93.86 | 53.10 | 53.44 |
| T2 | L1 | whole_count | 23 | 94.02 | 56.21 | 56.32 |
| T2 | L1 | fixed250_ordered | 23 | 97.88 | 49.59 | 51.67 |
| T2 | L1 | fixed250_shuffled | 23 | 99.04 | 39.01 | 40.05 |
| T2 | L1 | relative10_ordered | 23 | 93.98 | 64.10 | 71.77 |
| T2 | L1 | relative10_shuffled | 23 | 95.34 | 35.21 | 37.98 |
| T2 | L2 | whole_count | 23 | 83.87 | 58.45 | 64.70 |
| T2 | L2 | fixed250_ordered | 23 | 99.84 | 64.61 | 65.48 |
| T2 | L2 | fixed250_shuffled | 23 | 99.87 | 57.10 | 53.00 |
| T2 | L2 | relative10_ordered | 23 | 98.24 | 72.80 | 72.32 |
| T2 | L2 | relative10_shuffled | 23 | 96.14 | 54.18 | 54.66 |
| T3 | L1 | whole_count | 23 | 72.57 | 49.39 | 47.96 |
| T3 | L1 | fixed250_ordered | 23 | 94.71 | 44.25 | 50.28 |
| T3 | L1 | fixed250_shuffled | 23 | 94.48 | 33.02 | 35.07 |
| T3 | L1 | relative10_ordered | 23 | 96.04 | 62.91 | 72.09 |
| T3 | L1 | relative10_shuffled | 23 | 92.75 | 30.19 | 32.53 |
| T3 | L2 | whole_count | 23 | 74.88 | 51.10 | 58.63 |
| T3 | L2 | fixed250_ordered | 23 | 97.56 | 58.83 | 57.33 |
| T3 | L2 | fixed250_shuffled | 23 | 94.73 | 50.99 | 53.99 |
| T3 | L2 | relative10_ordered | 23 | 98.53 | 64.70 | 70.00 |
| T3 | L2 | relative10_shuffled | 23 | 88.44 | 49.07 | 51.67 |
| T4 | L1 | whole_count | 23 | 84.48 | 47.45 | 61.29 |
| T4 | L1 | fixed250_ordered | 23 | 98.68 | 52.78 | 51.37 |
| T4 | L1 | fixed250_shuffled | 23 | 96.96 | 40.13 | 41.97 |
| T4 | L1 | relative10_ordered | 23 | 97.85 | 66.74 | 71.69 |
| T4 | L1 | relative10_shuffled | 23 | 93.32 | 35.32 | 41.85 |
| T4 | L2 | whole_count | 23 | 98.07 | 50.92 | 55.15 |
| T4 | L2 | fixed250_ordered | 23 | 97.29 | 56.67 | 64.41 |
| T4 | L2 | fixed250_shuffled | 23 | 98.22 | 49.62 | 51.56 |
| T4 | L2 | relative10_ordered | 23 | 98.69 | 65.01 | 68.19 |
| T4 | L2 | relative10_shuffled | 23 | 97.58 | 49.45 | 55.28 |
| D1 | L1 | whole_count | 23 | 92.65 | 50.25 | 50.27 |
| D1 | L1 | fixed250_ordered | 23 | 96.50 | 47.45 | 48.94 |
| D1 | L1 | fixed250_shuffled | 23 | 95.86 | 34.50 | 36.37 |
| D1 | L1 | relative10_ordered | 23 | 96.58 | 63.49 | 67.75 |
| D1 | L1 | relative10_shuffled | 23 | 93.34 | 32.51 | 35.66 |
| D1 | L2 | whole_count | 23 | 94.94 | 52.48 | 61.49 |
| D1 | L2 | fixed250_ordered | 23 | 97.94 | 55.90 | 57.65 |
| D1 | L2 | fixed250_shuffled | 23 | 98.43 | 45.06 | 48.92 |
| D1 | L2 | relative10_ordered | 23 | 96.06 | 60.64 | 69.70 |
| D1 | L2 | relative10_shuffled | 23 | 93.42 | 44.89 | 49.53 |
| D1 | L3 | whole_count | 23 | 97.43 | 49.19 | 53.18 |
| D1 | L3 | fixed250_ordered | 23 | 99.17 | 50.30 | 48.83 |
| D1 | L3 | fixed250_shuffled | 23 | 99.03 | 47.28 | 48.33 |
| D1 | L3 | relative10_ordered | 23 | 99.17 | 53.26 | 59.95 |
| D1 | L3 | relative10_shuffled | 23 | 98.26 | 47.41 | 52.02 |
| O0 | L1 | whole_count | 37 | 96.55 | 54.32 | 58.48 |
| O0 | L1 | fixed250_ordered | 37 | 98.10 | 52.24 | 56.65 |
| O0 | L1 | fixed250_shuffled | 37 | 99.22 | 37.38 | 44.12 |
| O0 | L1 | relative10_ordered | 37 | 98.58 | 65.61 | 65.59 |
| O0 | L1 | relative10_shuffled | 37 | 96.36 | 35.98 | 39.08 |
| O0 | L2 | whole_count | 37 | 98.58 | 52.91 | 64.75 |
| O0 | L2 | fixed250_ordered | 37 | 92.42 | 54.25 | 58.74 |
| O0 | L2 | fixed250_shuffled | 37 | 98.15 | 47.47 | 51.58 |
| O0 | L2 | relative10_ordered | 37 | 99.84 | 62.99 | 66.68 |
| O0 | L2 | relative10_shuffled | 37 | 96.59 | 49.61 | 56.60 |
| O1 | L1 | whole_count | 37 | 94.88 | 49.26 | 54.93 |
| O1 | L1 | fixed250_ordered | 37 | 97.13 | 50.09 | 53.83 |
| O1 | L1 | fixed250_shuffled | 37 | 97.10 | 36.36 | 42.09 |
| O1 | L1 | relative10_ordered | 37 | 97.66 | 59.90 | 67.46 |
| O1 | L1 | relative10_shuffled | 37 | 85.44 | 33.07 | 40.61 |
| O1 | L2 | whole_count | 37 | 93.88 | 51.92 | 63.41 |
| O1 | L2 | fixed250_ordered | 37 | 99.18 | 56.93 | 63.56 |
| O1 | L2 | fixed250_shuffled | 37 | 97.65 | 48.93 | 54.91 |
| O1 | L2 | relative10_ordered | 37 | 98.69 | 62.87 | 70.32 |
| O1 | L2 | relative10_shuffled | 37 | 92.62 | 49.28 | 56.86 |
| O2 | L1 | whole_count | 37 | 87.64 | 48.02 | 63.63 |
| O2 | L1 | fixed250_ordered | 37 | 94.32 | 51.42 | 58.73 |
| O2 | L1 | fixed250_shuffled | 37 | 98.14 | 41.87 | 42.70 |
| O2 | L1 | relative10_ordered | 37 | 95.66 | 65.79 | 74.82 |
| O2 | L1 | relative10_shuffled | 37 | 99.10 | 36.87 | 39.96 |
| O2 | L2 | whole_count | 37 | 96.35 | 51.53 | 59.72 |
| O2 | L2 | fixed250_ordered | 37 | 92.20 | 57.29 | 59.81 |
| O2 | L2 | fixed250_shuffled | 37 | 98.14 | 53.70 | 49.91 |
| O2 | L2 | relative10_ordered | 37 | 99.84 | 61.37 | 67.65 |
| O2 | L2 | relative10_shuffled | 37 | 96.52 | 51.32 | 53.30 |
| O3 | L1 | whole_count | 37 | 96.87 | 55.19 | 58.58 |
| O3 | L1 | fixed250_ordered | 37 | 98.58 | 50.42 | 56.83 |
| O3 | L1 | fixed250_shuffled | 37 | 97.13 | 39.25 | 43.44 |
| O3 | L1 | relative10_ordered | 37 | 98.58 | 64.89 | 69.62 |
| O3 | L1 | relative10_shuffled | 37 | 98.35 | 37.13 | 41.62 |
| O3 | L2 | whole_count | 37 | 97.24 | 52.03 | 59.17 |
| O3 | L2 | fixed250_ordered | 37 | 99.84 | 55.81 | 63.50 |
| O3 | L2 | fixed250_shuffled | 37 | 98.52 | 52.02 | 50.55 |
| O3 | L2 | relative10_ordered | 37 | 98.90 | 63.45 | 70.40 |
| O3 | L2 | relative10_shuffled | 37 | 98.92 | 49.92 | 54.57 |
| T1 | L1 | whole_count | 37 | 93.96 | 45.31 | 47.63 |
| T1 | L1 | fixed250_ordered | 37 | 91.06 | 45.30 | 52.91 |
| T1 | L1 | fixed250_shuffled | 37 | 97.01 | 34.90 | 39.66 |
| T1 | L1 | relative10_ordered | 37 | 97.37 | 65.62 | 67.33 |
| T1 | L1 | relative10_shuffled | 37 | 93.09 | 32.37 | 35.17 |
| T1 | L2 | whole_count | 37 | 88.31 | 50.58 | 58.41 |
| T1 | L2 | fixed250_ordered | 37 | 99.17 | 58.53 | 60.84 |
| T1 | L2 | fixed250_shuffled | 37 | 96.12 | 50.40 | 54.23 |
| T1 | L2 | relative10_ordered | 37 | 99.65 | 64.53 | 68.70 |
| T1 | L2 | relative10_shuffled | 37 | 90.45 | 49.95 | 54.31 |
| T2 | L1 | whole_count | 37 | 95.32 | 52.31 | 55.94 |
| T2 | L1 | fixed250_ordered | 37 | 99.65 | 50.13 | 54.88 |
| T2 | L1 | fixed250_shuffled | 37 | 97.88 | 37.48 | 40.78 |
| T2 | L1 | relative10_ordered | 37 | 100.00 | 62.29 | 62.38 |
| T2 | L1 | relative10_shuffled | 37 | 94.47 | 33.60 | 40.01 |
| T2 | L2 | whole_count | 37 | 98.17 | 54.94 | 59.47 |
| T2 | L2 | fixed250_ordered | 37 | 100.00 | 60.82 | 58.90 |
| T2 | L2 | fixed250_shuffled | 37 | 99.97 | 50.59 | 51.64 |
| T2 | L2 | relative10_ordered | 37 | 97.81 | 65.49 | 73.49 |
| T2 | L2 | relative10_shuffled | 37 | 96.01 | 47.21 | 54.71 |
| T3 | L1 | whole_count | 37 | 81.88 | 43.13 | 49.00 |
| T3 | L1 | fixed250_ordered | 37 | 98.85 | 42.80 | 45.68 |
| T3 | L1 | fixed250_shuffled | 37 | 96.28 | 34.49 | 41.48 |
| T3 | L1 | relative10_ordered | 37 | 97.27 | 60.31 | 66.90 |
| T3 | L1 | relative10_shuffled | 37 | 94.44 | 29.97 | 31.07 |
| T3 | L2 | whole_count | 37 | 94.11 | 50.57 | 56.41 |
| T3 | L2 | fixed250_ordered | 37 | 88.44 | 51.67 | 56.77 |
| T3 | L2 | fixed250_shuffled | 37 | 95.83 | 46.51 | 52.20 |
| T3 | L2 | relative10_ordered | 37 | 97.44 | 66.89 | 69.28 |
| T3 | L2 | relative10_shuffled | 37 | 95.59 | 47.37 | 54.43 |
| T4 | L1 | whole_count | 37 | 95.79 | 47.39 | 57.62 |
| T4 | L1 | fixed250_ordered | 37 | 98.06 | 47.36 | 51.80 |
| T4 | L1 | fixed250_shuffled | 37 | 99.30 | 35.21 | 40.96 |
| T4 | L1 | relative10_ordered | 37 | 99.49 | 64.09 | 61.83 |
| T4 | L1 | relative10_shuffled | 37 | 94.78 | 31.94 | 37.17 |
| T4 | L2 | whole_count | 37 | 94.04 | 51.56 | 60.12 |
| T4 | L2 | fixed250_ordered | 37 | 99.65 | 55.10 | 65.08 |
| T4 | L2 | fixed250_shuffled | 37 | 98.90 | 51.65 | 56.83 |
| T4 | L2 | relative10_ordered | 37 | 99.17 | 60.49 | 67.85 |
| T4 | L2 | relative10_shuffled | 37 | 96.97 | 48.17 | 55.78 |
| D1 | L1 | whole_count | 37 | 93.13 | 37.59 | 39.38 |
| D1 | L1 | fixed250_ordered | 37 | 93.93 | 45.30 | 51.99 |
| D1 | L1 | fixed250_shuffled | 37 | 94.80 | 31.11 | 34.96 |
| D1 | L1 | relative10_ordered | 37 | 95.16 | 56.62 | 60.15 |
| D1 | L1 | relative10_shuffled | 37 | 88.96 | 29.76 | 31.59 |
| D1 | L2 | whole_count | 37 | 72.21 | 41.63 | 45.74 |
| D1 | L2 | fixed250_ordered | 37 | 95.72 | 45.93 | 52.84 |
| D1 | L2 | fixed250_shuffled | 37 | 91.63 | 39.24 | 44.40 |
| D1 | L2 | relative10_ordered | 37 | 95.71 | 56.53 | 60.69 |
| D1 | L2 | relative10_shuffled | 37 | 89.52 | 38.40 | 41.17 |
| D1 | L3 | whole_count | 37 | 95.97 | 43.32 | 47.66 |
| D1 | L3 | fixed250_ordered | 37 | 94.35 | 46.36 | 50.22 |
| D1 | L3 | fixed250_shuffled | 37 | 88.86 | 39.98 | 45.89 |
| D1 | L3 | relative10_ordered | 37 | 95.22 | 49.76 | 54.52 |
| D1 | L3 | relative10_shuffled | 37 | 92.16 | 39.87 | 45.03 |

### State: pre_reset; decoder: no_bias

| Case | Layer | Aggregation | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 11 | 99.17 | 53.37 | 50.79 |
| O0 | L1 | fixed250_ordered | 11 | 100.00 | 63.61 | 58.91 |
| O0 | L1 | fixed250_shuffled | 11 | 99.07 | 40.64 | 36.06 |
| O0 | L1 | relative10_ordered | 11 | 100.00 | 70.84 | 67.73 |
| O0 | L1 | relative10_shuffled | 11 | 91.72 | 24.67 | 23.77 |
| O0 | L2 | whole_count | 11 | 84.31 | 54.68 | 54.09 |
| O0 | L2 | fixed250_ordered | 11 | 97.15 | 57.21 | 55.44 |
| O0 | L2 | fixed250_shuffled | 11 | 93.92 | 53.77 | 48.86 |
| O0 | L2 | relative10_ordered | 11 | 99.40 | 67.85 | 59.23 |
| O0 | L2 | relative10_shuffled | 11 | 96.69 | 50.08 | 46.79 |
| O1 | L1 | whole_count | 11 | 87.56 | 53.88 | 51.61 |
| O1 | L1 | fixed250_ordered | 11 | 100.00 | 63.21 | 60.14 |
| O1 | L1 | fixed250_shuffled | 11 | 99.19 | 39.58 | 36.86 |
| O1 | L1 | relative10_ordered | 11 | 100.00 | 71.24 | 69.23 |
| O1 | L1 | relative10_shuffled | 11 | 86.13 | 25.34 | 23.88 |
| O1 | L2 | whole_count | 11 | 86.01 | 50.88 | 49.79 |
| O1 | L2 | fixed250_ordered | 11 | 86.43 | 56.68 | 59.38 |
| O1 | L2 | fixed250_shuffled | 11 | 89.54 | 52.30 | 50.62 |
| O1 | L2 | relative10_ordered | 11 | 95.95 | 63.91 | 68.00 |
| O1 | L2 | relative10_shuffled | 11 | 95.03 | 44.96 | 45.55 |
| O2 | L1 | whole_count | 11 | 88.43 | 56.58 | 53.40 |
| O2 | L1 | fixed250_ordered | 11 | 100.00 | 64.04 | 57.92 |
| O2 | L1 | fixed250_shuffled | 11 | 98.91 | 41.21 | 35.33 |
| O2 | L1 | relative10_ordered | 11 | 99.34 | 70.54 | 74.49 |
| O2 | L1 | relative10_shuffled | 11 | 85.09 | 24.68 | 23.47 |
| O2 | L2 | whole_count | 11 | 99.84 | 56.92 | 49.78 |
| O2 | L2 | fixed250_ordered | 11 | 97.50 | 62.95 | 57.42 |
| O2 | L2 | fixed250_shuffled | 11 | 94.53 | 58.53 | 50.20 |
| O2 | L2 | relative10_ordered | 11 | 100.00 | 65.13 | 61.00 |
| O2 | L2 | relative10_shuffled | 11 | 95.18 | 52.63 | 48.65 |
| O3 | L1 | whole_count | 11 | 99.34 | 52.66 | 51.93 |
| O3 | L1 | fixed250_ordered | 11 | 100.00 | 64.60 | 59.71 |
| O3 | L1 | fixed250_shuffled | 11 | 100.00 | 41.41 | 35.94 |
| O3 | L1 | relative10_ordered | 11 | 99.49 | 69.21 | 73.65 |
| O3 | L1 | relative10_shuffled | 11 | 84.27 | 25.00 | 24.09 |
| O3 | L2 | whole_count | 11 | 90.76 | 59.34 | 55.61 |
| O3 | L2 | fixed250_ordered | 11 | 97.82 | 63.79 | 55.25 |
| O3 | L2 | fixed250_shuffled | 11 | 92.14 | 59.02 | 51.26 |
| O3 | L2 | relative10_ordered | 11 | 99.57 | 70.58 | 59.65 |
| O3 | L2 | relative10_shuffled | 11 | 95.88 | 52.91 | 48.51 |
| T1 | L1 | whole_count | 11 | 98.11 | 47.31 | 53.33 |
| T1 | L1 | fixed250_ordered | 11 | 100.00 | 59.70 | 61.11 |
| T1 | L1 | fixed250_shuffled | 11 | 93.69 | 36.40 | 33.62 |
| T1 | L1 | relative10_ordered | 11 | 100.00 | 69.77 | 67.14 |
| T1 | L1 | relative10_shuffled | 11 | 91.03 | 23.13 | 21.37 |
| T1 | L2 | whole_count | 11 | 83.00 | 52.23 | 55.37 |
| T1 | L2 | fixed250_ordered | 11 | 87.12 | 57.15 | 57.41 |
| T1 | L2 | fixed250_shuffled | 11 | 95.72 | 50.26 | 47.38 |
| T1 | L2 | relative10_ordered | 11 | 90.70 | 64.34 | 70.71 |
| T1 | L2 | relative10_shuffled | 11 | 96.64 | 48.16 | 45.28 |
| T2 | L1 | whole_count | 11 | 97.30 | 47.73 | 53.38 |
| T2 | L1 | fixed250_ordered | 11 | 100.00 | 59.57 | 58.90 |
| T2 | L1 | fixed250_shuffled | 11 | 90.33 | 37.16 | 31.27 |
| T2 | L1 | relative10_ordered | 11 | 100.00 | 68.44 | 66.36 |
| T2 | L1 | relative10_shuffled | 11 | 91.44 | 23.38 | 22.31 |
| T2 | L2 | whole_count | 11 | 85.25 | 48.49 | 52.51 |
| T2 | L2 | fixed250_ordered | 11 | 100.00 | 58.86 | 58.55 |
| T2 | L2 | fixed250_shuffled | 11 | 95.98 | 50.85 | 49.55 |
| T2 | L2 | relative10_ordered | 11 | 88.98 | 66.12 | 74.10 |
| T2 | L2 | relative10_shuffled | 11 | 95.43 | 45.81 | 45.62 |
| T3 | L1 | whole_count | 11 | 97.00 | 48.83 | 48.22 |
| T3 | L1 | fixed250_ordered | 11 | 100.00 | 61.88 | 60.09 |
| T3 | L1 | fixed250_shuffled | 11 | 96.53 | 37.09 | 34.58 |
| T3 | L1 | relative10_ordered | 11 | 99.35 | 67.13 | 73.88 |
| T3 | L1 | relative10_shuffled | 11 | 92.49 | 23.88 | 21.96 |
| T3 | L2 | whole_count | 11 | 90.10 | 50.61 | 51.54 |
| T3 | L2 | fixed250_ordered | 11 | 94.89 | 53.40 | 54.89 |
| T3 | L2 | fixed250_shuffled | 11 | 90.97 | 50.57 | 49.74 |
| T3 | L2 | relative10_ordered | 11 | 95.53 | 67.19 | 65.73 |
| T3 | L2 | relative10_shuffled | 11 | 93.98 | 49.53 | 47.34 |
| T4 | L1 | whole_count | 11 | 87.93 | 53.03 | 54.81 |
| T4 | L1 | fixed250_ordered | 11 | 100.00 | 62.45 | 59.55 |
| T4 | L1 | fixed250_shuffled | 11 | 99.12 | 39.25 | 34.91 |
| T4 | L1 | relative10_ordered | 11 | 100.00 | 70.20 | 71.02 |
| T4 | L1 | relative10_shuffled | 11 | 93.14 | 24.79 | 23.84 |
| T4 | L2 | whole_count | 11 | 92.73 | 56.35 | 49.57 |
| T4 | L2 | fixed250_ordered | 11 | 99.04 | 55.64 | 53.51 |
| T4 | L2 | fixed250_shuffled | 11 | 97.05 | 52.47 | 48.61 |
| T4 | L2 | relative10_ordered | 11 | 93.35 | 61.33 | 60.02 |
| T4 | L2 | relative10_shuffled | 11 | 96.53 | 51.47 | 47.93 |
| D1 | L1 | whole_count | 11 | 81.20 | 50.60 | 45.75 |
| D1 | L1 | fixed250_ordered | 11 | 100.00 | 61.09 | 58.63 |
| D1 | L1 | fixed250_shuffled | 11 | 99.00 | 38.43 | 35.77 |
| D1 | L1 | relative10_ordered | 11 | 99.50 | 70.60 | 71.65 |
| D1 | L1 | relative10_shuffled | 11 | 93.34 | 23.89 | 22.91 |
| D1 | L2 | whole_count | 11 | 90.89 | 46.29 | 44.11 |
| D1 | L2 | fixed250_ordered | 11 | 98.40 | 52.90 | 49.74 |
| D1 | L2 | fixed250_shuffled | 11 | 92.23 | 40.89 | 37.19 |
| D1 | L2 | relative10_ordered | 11 | 98.10 | 66.17 | 63.88 |
| D1 | L2 | relative10_shuffled | 11 | 97.04 | 38.73 | 34.31 |
| D1 | L3 | whole_count | 11 | 93.66 | 53.47 | 47.44 |
| D1 | L3 | fixed250_ordered | 11 | 96.03 | 53.59 | 51.72 |
| D1 | L3 | fixed250_shuffled | 11 | 95.91 | 48.77 | 45.44 |
| D1 | L3 | relative10_ordered | 11 | 98.05 | 53.26 | 48.76 |
| D1 | L3 | relative10_shuffled | 11 | 96.23 | 49.23 | 45.98 |
| O0 | L1 | whole_count | 23 | 99.66 | 52.09 | 56.20 |
| O0 | L1 | fixed250_ordered | 23 | 100.00 | 65.26 | 61.26 |
| O0 | L1 | fixed250_shuffled | 23 | 100.00 | 38.05 | 34.87 |
| O0 | L1 | relative10_ordered | 23 | 100.00 | 70.34 | 67.99 |
| O0 | L1 | relative10_shuffled | 23 | 91.94 | 26.21 | 23.72 |
| O0 | L2 | whole_count | 23 | 99.24 | 56.65 | 57.45 |
| O0 | L2 | fixed250_ordered | 23 | 99.68 | 62.62 | 55.50 |
| O0 | L2 | fixed250_shuffled | 23 | 96.65 | 54.12 | 49.73 |
| O0 | L2 | relative10_ordered | 23 | 92.60 | 70.86 | 73.04 |
| O0 | L2 | relative10_shuffled | 23 | 94.66 | 53.10 | 50.08 |
| O1 | L1 | whole_count | 23 | 100.00 | 51.48 | 50.94 |
| O1 | L1 | fixed250_ordered | 23 | 100.00 | 65.83 | 61.16 |
| O1 | L1 | fixed250_shuffled | 23 | 93.22 | 40.48 | 34.01 |
| O1 | L1 | relative10_ordered | 23 | 100.00 | 73.68 | 68.42 |
| O1 | L1 | relative10_shuffled | 23 | 84.72 | 27.14 | 22.88 |
| O1 | L2 | whole_count | 23 | 93.29 | 52.94 | 59.38 |
| O1 | L2 | fixed250_ordered | 23 | 99.36 | 61.23 | 56.55 |
| O1 | L2 | fixed250_shuffled | 23 | 94.26 | 52.16 | 52.81 |
| O1 | L2 | relative10_ordered | 23 | 89.25 | 66.88 | 72.21 |
| O1 | L2 | relative10_shuffled | 23 | 98.34 | 46.87 | 44.03 |
| O2 | L1 | whole_count | 23 | 87.82 | 52.97 | 53.01 |
| O2 | L1 | fixed250_ordered | 23 | 100.00 | 65.20 | 60.61 |
| O2 | L1 | fixed250_shuffled | 23 | 98.46 | 38.97 | 33.89 |
| O2 | L1 | relative10_ordered | 23 | 99.01 | 69.15 | 71.90 |
| O2 | L1 | relative10_shuffled | 23 | 91.26 | 26.34 | 25.45 |
| O2 | L2 | whole_count | 23 | 90.46 | 58.84 | 57.43 |
| O2 | L2 | fixed250_ordered | 23 | 99.51 | 64.54 | 59.92 |
| O2 | L2 | fixed250_shuffled | 23 | 97.38 | 56.14 | 51.51 |
| O2 | L2 | relative10_ordered | 23 | 89.75 | 69.41 | 71.46 |
| O2 | L2 | relative10_shuffled | 23 | 99.05 | 56.15 | 50.57 |
| O3 | L1 | whole_count | 23 | 99.50 | 53.57 | 55.87 |
| O3 | L1 | fixed250_ordered | 23 | 100.00 | 64.50 | 60.45 |
| O3 | L1 | fixed250_shuffled | 23 | 96.02 | 38.97 | 33.61 |
| O3 | L1 | relative10_ordered | 23 | 100.00 | 71.47 | 66.47 |
| O3 | L1 | relative10_shuffled | 23 | 83.84 | 25.47 | 23.99 |
| O3 | L2 | whole_count | 23 | 89.98 | 55.99 | 57.36 |
| O3 | L2 | fixed250_ordered | 23 | 96.77 | 66.68 | 56.98 |
| O3 | L2 | fixed250_shuffled | 23 | 93.90 | 54.62 | 55.07 |
| O3 | L2 | relative10_ordered | 23 | 97.23 | 72.56 | 68.64 |
| O3 | L2 | relative10_shuffled | 23 | 98.97 | 51.67 | 48.89 |
| T1 | L1 | whole_count | 23 | 96.55 | 48.73 | 51.66 |
| T1 | L1 | fixed250_ordered | 23 | 100.00 | 62.08 | 59.75 |
| T1 | L1 | fixed250_shuffled | 23 | 98.63 | 37.98 | 33.71 |
| T1 | L1 | relative10_ordered | 23 | 100.00 | 68.32 | 67.67 |
| T1 | L1 | relative10_shuffled | 23 | 85.17 | 23.98 | 21.27 |
| T1 | L2 | whole_count | 23 | 86.18 | 55.42 | 51.85 |
| T1 | L2 | fixed250_ordered | 23 | 95.62 | 60.63 | 61.38 |
| T1 | L2 | fixed250_shuffled | 23 | 95.05 | 53.69 | 51.42 |
| T1 | L2 | relative10_ordered | 23 | 96.52 | 68.10 | 69.49 |
| T1 | L2 | relative10_shuffled | 23 | 95.57 | 48.49 | 48.10 |
| T2 | L1 | whole_count | 23 | 84.87 | 46.68 | 55.11 |
| T2 | L1 | fixed250_ordered | 23 | 100.00 | 60.46 | 60.94 |
| T2 | L1 | fixed250_shuffled | 23 | 97.42 | 37.70 | 34.25 |
| T2 | L1 | relative10_ordered | 23 | 100.00 | 67.82 | 67.46 |
| T2 | L1 | relative10_shuffled | 23 | 84.92 | 24.05 | 22.27 |
| T2 | L2 | whole_count | 23 | 85.14 | 53.25 | 55.13 |
| T2 | L2 | fixed250_ordered | 23 | 100.00 | 60.48 | 61.39 |
| T2 | L2 | fixed250_shuffled | 23 | 98.94 | 54.32 | 49.06 |
| T2 | L2 | relative10_ordered | 23 | 96.69 | 70.74 | 69.65 |
| T2 | L2 | relative10_shuffled | 23 | 99.67 | 49.55 | 46.28 |
| T3 | L1 | whole_count | 23 | 100.00 | 43.26 | 48.37 |
| T3 | L1 | fixed250_ordered | 23 | 100.00 | 59.90 | 57.88 |
| T3 | L1 | fixed250_shuffled | 23 | 97.43 | 37.16 | 33.70 |
| T3 | L1 | relative10_ordered | 23 | 100.00 | 68.25 | 68.06 |
| T3 | L1 | relative10_shuffled | 23 | 91.14 | 24.00 | 23.32 |
| T3 | L2 | whole_count | 23 | 73.88 | 51.55 | 57.65 |
| T3 | L2 | fixed250_ordered | 23 | 89.13 | 56.58 | 57.11 |
| T3 | L2 | fixed250_shuffled | 23 | 86.94 | 47.95 | 48.87 |
| T3 | L2 | relative10_ordered | 23 | 91.07 | 65.69 | 69.03 |
| T3 | L2 | relative10_shuffled | 23 | 93.70 | 42.70 | 44.24 |
| T4 | L1 | whole_count | 23 | 85.55 | 54.26 | 50.31 |
| T4 | L1 | fixed250_ordered | 23 | 100.00 | 63.81 | 60.61 |
| T4 | L1 | fixed250_shuffled | 23 | 97.87 | 39.16 | 34.82 |
| T4 | L1 | relative10_ordered | 23 | 100.00 | 70.20 | 67.63 |
| T4 | L1 | relative10_shuffled | 23 | 92.83 | 26.46 | 24.27 |
| T4 | L2 | whole_count | 23 | 93.11 | 56.52 | 55.76 |
| T4 | L2 | fixed250_ordered | 23 | 99.51 | 62.43 | 46.71 |
| T4 | L2 | fixed250_shuffled | 23 | 91.31 | 52.99 | 49.70 |
| T4 | L2 | relative10_ordered | 23 | 94.64 | 71.74 | 69.69 |
| T4 | L2 | relative10_shuffled | 23 | 95.73 | 53.32 | 47.93 |
| D1 | L1 | whole_count | 23 | 83.25 | 47.24 | 50.52 |
| D1 | L1 | fixed250_ordered | 23 | 100.00 | 63.98 | 60.30 |
| D1 | L1 | fixed250_shuffled | 23 | 98.08 | 38.99 | 34.19 |
| D1 | L1 | relative10_ordered | 23 | 100.00 | 71.10 | 69.16 |
| D1 | L1 | relative10_shuffled | 23 | 92.75 | 27.10 | 24.97 |
| D1 | L2 | whole_count | 23 | 96.23 | 47.88 | 52.32 |
| D1 | L2 | fixed250_ordered | 23 | 93.14 | 58.93 | 57.17 |
| D1 | L2 | fixed250_shuffled | 23 | 93.09 | 44.76 | 40.69 |
| D1 | L2 | relative10_ordered | 23 | 93.91 | 66.17 | 71.14 |
| D1 | L2 | relative10_shuffled | 23 | 97.74 | 36.04 | 37.48 |
| D1 | L3 | whole_count | 23 | 97.43 | 55.26 | 51.20 |
| D1 | L3 | fixed250_ordered | 23 | 96.38 | 49.14 | 57.99 |
| D1 | L3 | fixed250_shuffled | 23 | 94.47 | 49.32 | 49.67 |
| D1 | L3 | relative10_ordered | 23 | 95.75 | 57.55 | 57.23 |
| D1 | L3 | relative10_shuffled | 23 | 95.28 | 48.46 | 49.62 |
| O0 | L1 | whole_count | 37 | 100.00 | 45.97 | 47.31 |
| O0 | L1 | fixed250_ordered | 37 | 100.00 | 65.52 | 63.27 |
| O0 | L1 | fixed250_shuffled | 37 | 97.69 | 40.60 | 35.90 |
| O0 | L1 | relative10_ordered | 37 | 100.00 | 69.86 | 70.05 |
| O0 | L1 | relative10_shuffled | 37 | 91.53 | 26.30 | 25.62 |
| O0 | L2 | whole_count | 37 | 91.46 | 53.17 | 57.87 |
| O0 | L2 | fixed250_ordered | 37 | 99.68 | 56.12 | 63.32 |
| O0 | L2 | fixed250_shuffled | 37 | 98.67 | 51.08 | 50.88 |
| O0 | L2 | relative10_ordered | 37 | 91.87 | 64.97 | 71.04 |
| O0 | L2 | relative10_shuffled | 37 | 92.77 | 50.62 | 51.46 |
| O1 | L1 | whole_count | 37 | 88.53 | 51.75 | 47.50 |
| O1 | L1 | fixed250_ordered | 37 | 100.00 | 65.03 | 62.93 |
| O1 | L1 | fixed250_shuffled | 37 | 98.55 | 40.62 | 35.32 |
| O1 | L1 | relative10_ordered | 37 | 100.00 | 71.33 | 72.77 |
| O1 | L1 | relative10_shuffled | 37 | 86.55 | 25.07 | 26.01 |
| O1 | L2 | whole_count | 37 | 86.83 | 52.81 | 53.52 |
| O1 | L2 | fixed250_ordered | 37 | 99.17 | 58.48 | 61.99 |
| O1 | L2 | fixed250_shuffled | 37 | 94.45 | 50.54 | 51.47 |
| O1 | L2 | relative10_ordered | 37 | 96.63 | 64.52 | 71.16 |
| O1 | L2 | relative10_shuffled | 37 | 96.45 | 42.72 | 44.31 |
| O2 | L1 | whole_count | 37 | 100.00 | 54.77 | 47.64 |
| O2 | L1 | fixed250_ordered | 37 | 100.00 | 63.11 | 62.78 |
| O2 | L1 | fixed250_shuffled | 37 | 98.11 | 41.96 | 35.55 |
| O2 | L1 | relative10_ordered | 37 | 100.00 | 69.88 | 69.75 |
| O2 | L1 | relative10_shuffled | 37 | 84.20 | 27.03 | 25.16 |
| O2 | L2 | whole_count | 37 | 99.68 | 55.32 | 48.86 |
| O2 | L2 | fixed250_ordered | 37 | 99.51 | 59.61 | 64.00 |
| O2 | L2 | fixed250_shuffled | 37 | 98.65 | 56.01 | 48.67 |
| O2 | L2 | relative10_ordered | 37 | 97.59 | 64.91 | 72.48 |
| O2 | L2 | relative10_shuffled | 37 | 96.76 | 53.04 | 44.56 |
| O3 | L1 | whole_count | 37 | 99.84 | 55.13 | 45.17 |
| O3 | L1 | fixed250_ordered | 37 | 100.00 | 65.20 | 62.51 |
| O3 | L1 | fixed250_shuffled | 37 | 98.25 | 41.74 | 36.09 |
| O3 | L1 | relative10_ordered | 37 | 100.00 | 69.44 | 70.99 |
| O3 | L1 | relative10_shuffled | 37 | 90.63 | 25.49 | 25.07 |
| O3 | L2 | whole_count | 37 | 96.84 | 57.03 | 53.70 |
| O3 | L2 | fixed250_ordered | 37 | 99.84 | 56.09 | 61.88 |
| O3 | L2 | fixed250_shuffled | 37 | 98.79 | 53.72 | 50.20 |
| O3 | L2 | relative10_ordered | 37 | 97.76 | 66.45 | 68.14 |
| O3 | L2 | relative10_shuffled | 37 | 98.33 | 50.69 | 47.89 |
| T1 | L1 | whole_count | 37 | 96.71 | 50.42 | 48.00 |
| T1 | L1 | fixed250_ordered | 37 | 100.00 | 61.79 | 61.63 |
| T1 | L1 | fixed250_shuffled | 37 | 97.56 | 37.46 | 34.09 |
| T1 | L1 | relative10_ordered | 37 | 99.51 | 69.64 | 72.40 |
| T1 | L1 | relative10_shuffled | 37 | 91.45 | 23.07 | 22.71 |
| T1 | L2 | whole_count | 37 | 87.52 | 51.01 | 52.56 |
| T1 | L2 | fixed250_ordered | 37 | 88.28 | 56.75 | 59.47 |
| T1 | L2 | fixed250_shuffled | 37 | 91.15 | 47.78 | 48.84 |
| T1 | L2 | relative10_ordered | 37 | 96.81 | 63.71 | 71.33 |
| T1 | L2 | relative10_shuffled | 37 | 90.26 | 44.84 | 47.37 |
| T2 | L1 | whole_count | 37 | 83.95 | 47.81 | 50.93 |
| T2 | L1 | fixed250_ordered | 37 | 98.97 | 59.18 | 59.69 |
| T2 | L1 | fixed250_shuffled | 37 | 91.63 | 36.76 | 31.90 |
| T2 | L1 | relative10_ordered | 37 | 100.00 | 67.56 | 66.86 |
| T2 | L1 | relative10_shuffled | 37 | 89.98 | 24.01 | 23.45 |
| T2 | L2 | whole_count | 37 | 74.62 | 51.34 | 54.95 |
| T2 | L2 | fixed250_ordered | 37 | 85.95 | 55.73 | 62.79 |
| T2 | L2 | fixed250_shuffled | 37 | 94.04 | 49.63 | 49.55 |
| T2 | L2 | relative10_ordered | 37 | 89.63 | 70.09 | 71.18 |
| T2 | L2 | relative10_shuffled | 37 | 94.68 | 43.48 | 47.25 |
| T3 | L1 | whole_count | 37 | 100.00 | 45.91 | 45.89 |
| T3 | L1 | fixed250_ordered | 37 | 100.00 | 62.97 | 62.56 |
| T3 | L1 | fixed250_shuffled | 37 | 96.27 | 37.18 | 33.08 |
| T3 | L1 | relative10_ordered | 37 | 99.03 | 69.70 | 73.54 |
| T3 | L1 | relative10_shuffled | 37 | 83.64 | 22.87 | 21.64 |
| T3 | L2 | whole_count | 37 | 98.02 | 49.42 | 45.50 |
| T3 | L2 | fixed250_ordered | 37 | 87.22 | 51.97 | 59.10 |
| T3 | L2 | fixed250_shuffled | 37 | 94.09 | 45.79 | 48.75 |
| T3 | L2 | relative10_ordered | 37 | 96.35 | 62.45 | 67.16 |
| T3 | L2 | relative10_shuffled | 37 | 90.24 | 45.66 | 49.68 |
| T4 | L1 | whole_count | 37 | 99.04 | 51.97 | 48.72 |
| T4 | L1 | fixed250_ordered | 37 | 100.00 | 66.09 | 60.52 |
| T4 | L1 | fixed250_shuffled | 37 | 97.47 | 39.85 | 34.16 |
| T4 | L1 | relative10_ordered | 37 | 100.00 | 72.30 | 73.45 |
| T4 | L1 | relative10_shuffled | 37 | 91.76 | 24.78 | 23.42 |
| T4 | L2 | whole_count | 37 | 98.55 | 48.57 | 52.31 |
| T4 | L2 | fixed250_ordered | 37 | 96.64 | 50.53 | 61.43 |
| T4 | L2 | fixed250_shuffled | 37 | 94.90 | 48.50 | 52.62 |
| T4 | L2 | relative10_ordered | 37 | 92.49 | 58.56 | 71.51 |
| T4 | L2 | relative10_shuffled | 37 | 95.79 | 47.37 | 52.77 |
| D1 | L1 | whole_count | 37 | 96.96 | 56.96 | 51.10 |
| D1 | L1 | fixed250_ordered | 37 | 100.00 | 65.52 | 63.64 |
| D1 | L1 | fixed250_shuffled | 37 | 97.60 | 41.11 | 33.74 |
| D1 | L1 | relative10_ordered | 37 | 100.00 | 73.26 | 73.27 |
| D1 | L1 | relative10_shuffled | 37 | 98.51 | 24.56 | 26.04 |
| D1 | L2 | whole_count | 37 | 97.86 | 44.63 | 39.96 |
| D1 | L2 | fixed250_ordered | 37 | 91.11 | 52.49 | 56.93 |
| D1 | L2 | fixed250_shuffled | 37 | 91.67 | 36.52 | 37.34 |
| D1 | L2 | relative10_ordered | 37 | 93.60 | 61.68 | 60.43 |
| D1 | L2 | relative10_shuffled | 37 | 91.32 | 32.57 | 30.79 |
| D1 | L3 | whole_count | 37 | 89.78 | 41.67 | 43.16 |
| D1 | L3 | fixed250_ordered | 37 | 93.98 | 44.87 | 45.10 |
| D1 | L3 | fixed250_shuffled | 37 | 87.32 | 42.49 | 44.20 |
| D1 | L3 | relative10_ordered | 37 | 96.94 | 51.89 | 46.22 |
| D1 | L3 | relative10_shuffled | 37 | 83.51 | 39.33 | 43.79 |

### State: pre_reset; decoder: affine

| Case | Layer | Aggregation | Seed | Train BA | Validation BA | Test BA |
| --- | --- | --- | --- | --- | --- | --- |
| O0 | L1 | whole_count | 11 | 100.00 | 56.68 | 52.27 |
| O0 | L1 | fixed250_ordered | 11 | 100.00 | 64.04 | 58.31 |
| O0 | L1 | fixed250_shuffled | 11 | 100.00 | 41.69 | 36.01 |
| O0 | L1 | relative10_ordered | 11 | 100.00 | 70.47 | 74.39 |
| O0 | L1 | relative10_shuffled | 11 | 99.45 | 31.96 | 28.94 |
| O0 | L2 | whole_count | 11 | 84.83 | 55.77 | 53.57 |
| O0 | L2 | fixed250_ordered | 11 | 100.00 | 55.85 | 50.31 |
| O0 | L2 | fixed250_shuffled | 11 | 99.58 | 52.75 | 46.48 |
| O0 | L2 | relative10_ordered | 11 | 98.93 | 67.91 | 63.58 |
| O0 | L2 | relative10_shuffled | 11 | 95.63 | 54.30 | 50.29 |
| O1 | L1 | whole_count | 11 | 90.72 | 57.15 | 52.49 |
| O1 | L1 | fixed250_ordered | 11 | 100.00 | 63.84 | 59.71 |
| O1 | L1 | fixed250_shuffled | 11 | 100.00 | 39.82 | 37.68 |
| O1 | L1 | relative10_ordered | 11 | 100.00 | 70.68 | 76.07 |
| O1 | L1 | relative10_shuffled | 11 | 95.34 | 31.70 | 29.21 |
| O1 | L2 | whole_count | 11 | 77.56 | 52.35 | 55.47 |
| O1 | L2 | fixed250_ordered | 11 | 99.84 | 57.37 | 55.04 |
| O1 | L2 | fixed250_shuffled | 11 | 99.44 | 49.56 | 47.43 |
| O1 | L2 | relative10_ordered | 11 | 95.48 | 60.79 | 66.51 |
| O1 | L2 | relative10_shuffled | 11 | 95.20 | 45.82 | 51.27 |
| O2 | L1 | whole_count | 11 | 99.83 | 61.12 | 54.30 |
| O2 | L1 | fixed250_ordered | 11 | 100.00 | 63.90 | 57.92 |
| O2 | L1 | fixed250_shuffled | 11 | 100.00 | 40.18 | 36.04 |
| O2 | L1 | relative10_ordered | 11 | 100.00 | 69.98 | 75.03 |
| O2 | L1 | relative10_shuffled | 11 | 92.70 | 29.67 | 30.42 |
| O2 | L2 | whole_count | 11 | 96.78 | 59.76 | 55.36 |
| O2 | L2 | fixed250_ordered | 11 | 99.68 | 60.64 | 53.95 |
| O2 | L2 | fixed250_shuffled | 11 | 98.62 | 56.45 | 48.86 |
| O2 | L2 | relative10_ordered | 11 | 100.00 | 65.07 | 58.29 |
| O2 | L2 | relative10_shuffled | 11 | 96.57 | 57.12 | 48.58 |
| O3 | L1 | whole_count | 11 | 99.50 | 60.41 | 54.28 |
| O3 | L1 | fixed250_ordered | 11 | 100.00 | 65.29 | 60.90 |
| O3 | L1 | fixed250_shuffled | 11 | 100.00 | 40.80 | 36.71 |
| O3 | L1 | relative10_ordered | 11 | 100.00 | 69.70 | 74.15 |
| O3 | L1 | relative10_shuffled | 11 | 98.81 | 31.28 | 28.99 |
| O3 | L2 | whole_count | 11 | 90.78 | 61.41 | 53.92 |
| O3 | L2 | fixed250_ordered | 11 | 99.57 | 61.89 | 52.12 |
| O3 | L2 | fixed250_shuffled | 11 | 98.54 | 55.57 | 48.71 |
| O3 | L2 | relative10_ordered | 11 | 96.41 | 68.65 | 66.00 |
| O3 | L2 | relative10_shuffled | 11 | 93.56 | 58.13 | 52.69 |
| T1 | L1 | whole_count | 11 | 98.29 | 52.14 | 52.33 |
| T1 | L1 | fixed250_ordered | 11 | 100.00 | 59.04 | 59.37 |
| T1 | L1 | fixed250_shuffled | 11 | 100.00 | 35.88 | 35.77 |
| T1 | L1 | relative10_ordered | 11 | 100.00 | 69.01 | 66.38 |
| T1 | L1 | relative10_shuffled | 11 | 98.93 | 28.55 | 27.60 |
| T1 | L2 | whole_count | 11 | 82.96 | 53.53 | 55.69 |
| T1 | L2 | fixed250_ordered | 11 | 99.25 | 57.38 | 54.84 |
| T1 | L2 | fixed250_shuffled | 11 | 99.82 | 51.99 | 43.83 |
| T1 | L2 | relative10_ordered | 11 | 96.10 | 64.62 | 68.89 |
| T1 | L2 | relative10_shuffled | 11 | 93.48 | 48.60 | 51.09 |
| T2 | L1 | whole_count | 11 | 99.00 | 52.34 | 58.00 |
| T2 | L1 | fixed250_ordered | 11 | 100.00 | 60.63 | 60.15 |
| T2 | L1 | fixed250_shuffled | 11 | 100.00 | 35.97 | 35.22 |
| T2 | L1 | relative10_ordered | 11 | 100.00 | 68.44 | 66.36 |
| T2 | L1 | relative10_shuffled | 11 | 95.21 | 29.50 | 27.80 |
| T2 | L2 | whole_count | 11 | 87.13 | 55.85 | 60.57 |
| T2 | L2 | fixed250_ordered | 11 | 100.00 | 58.17 | 57.31 |
| T2 | L2 | fixed250_shuffled | 11 | 99.85 | 52.10 | 49.41 |
| T2 | L2 | relative10_ordered | 11 | 100.00 | 65.46 | 63.78 |
| T2 | L2 | relative10_shuffled | 11 | 95.03 | 48.98 | 48.84 |
| T3 | L1 | whole_count | 11 | 86.19 | 51.42 | 51.47 |
| T3 | L1 | fixed250_ordered | 11 | 100.00 | 61.59 | 57.81 |
| T3 | L1 | fixed250_shuffled | 11 | 98.94 | 36.53 | 35.37 |
| T3 | L1 | relative10_ordered | 11 | 99.51 | 69.98 | 74.54 |
| T3 | L1 | relative10_shuffled | 11 | 98.72 | 28.31 | 28.71 |
| T3 | L2 | whole_count | 11 | 90.76 | 55.74 | 51.90 |
| T3 | L2 | fixed250_ordered | 11 | 99.84 | 53.00 | 49.84 |
| T3 | L2 | fixed250_shuffled | 11 | 98.65 | 50.53 | 46.51 |
| T3 | L2 | relative10_ordered | 11 | 94.85 | 63.46 | 64.40 |
| T3 | L2 | relative10_shuffled | 11 | 94.92 | 52.47 | 50.97 |
| T4 | L1 | whole_count | 11 | 99.84 | 54.80 | 55.86 |
| T4 | L1 | fixed250_ordered | 11 | 100.00 | 58.67 | 59.11 |
| T4 | L1 | fixed250_shuffled | 11 | 100.00 | 39.13 | 35.85 |
| T4 | L1 | relative10_ordered | 11 | 100.00 | 71.53 | 70.22 |
| T4 | L1 | relative10_shuffled | 11 | 99.60 | 31.49 | 28.29 |
| T4 | L2 | whole_count | 11 | 93.37 | 57.18 | 51.39 |
| T4 | L2 | fixed250_ordered | 11 | 98.44 | 57.96 | 51.86 |
| T4 | L2 | fixed250_shuffled | 11 | 99.44 | 50.92 | 46.47 |
| T4 | L2 | relative10_ordered | 11 | 98.43 | 61.21 | 62.97 |
| T4 | L2 | relative10_shuffled | 11 | 97.24 | 53.38 | 50.51 |
| D1 | L1 | whole_count | 11 | 100.00 | 51.91 | 50.41 |
| D1 | L1 | fixed250_ordered | 11 | 100.00 | 60.00 | 59.61 |
| D1 | L1 | fixed250_shuffled | 11 | 98.54 | 39.04 | 36.72 |
| D1 | L1 | relative10_ordered | 11 | 100.00 | 71.65 | 69.44 |
| D1 | L1 | relative10_shuffled | 11 | 98.44 | 30.73 | 27.57 |
| D1 | L2 | whole_count | 11 | 97.05 | 52.55 | 45.48 |
| D1 | L2 | fixed250_ordered | 11 | 98.29 | 51.92 | 51.79 |
| D1 | L2 | fixed250_shuffled | 11 | 98.89 | 41.68 | 38.52 |
| D1 | L2 | relative10_ordered | 11 | 94.06 | 65.26 | 66.33 |
| D1 | L2 | relative10_shuffled | 11 | 95.95 | 40.63 | 36.27 |
| D1 | L3 | whole_count | 11 | 96.63 | 49.87 | 47.77 |
| D1 | L3 | fixed250_ordered | 11 | 95.72 | 54.12 | 51.80 |
| D1 | L3 | fixed250_shuffled | 11 | 97.58 | 46.99 | 42.27 |
| D1 | L3 | relative10_ordered | 11 | 92.34 | 54.51 | 51.36 |
| D1 | L3 | relative10_shuffled | 11 | 96.27 | 48.94 | 45.82 |
| O0 | L1 | whole_count | 23 | 90.38 | 56.87 | 56.93 |
| O0 | L1 | fixed250_ordered | 23 | 100.00 | 63.75 | 59.71 |
| O0 | L1 | fixed250_shuffled | 23 | 100.00 | 37.46 | 33.26 |
| O0 | L1 | relative10_ordered | 23 | 100.00 | 71.03 | 66.47 |
| O0 | L1 | relative10_shuffled | 23 | 93.60 | 30.07 | 28.22 |
| O0 | L2 | whole_count | 23 | 96.16 | 55.67 | 59.44 |
| O0 | L2 | fixed250_ordered | 23 | 100.00 | 62.62 | 55.22 |
| O0 | L2 | fixed250_shuffled | 23 | 99.57 | 53.69 | 51.02 |
| O0 | L2 | relative10_ordered | 23 | 99.06 | 69.00 | 67.00 |
| O0 | L2 | relative10_shuffled | 23 | 98.20 | 52.44 | 49.82 |
| O1 | L1 | whole_count | 23 | 100.00 | 56.49 | 54.06 |
| O1 | L1 | fixed250_ordered | 23 | 100.00 | 63.94 | 60.61 |
| O1 | L1 | fixed250_shuffled | 23 | 99.28 | 39.61 | 34.59 |
| O1 | L1 | relative10_ordered | 23 | 100.00 | 70.74 | 67.66 |
| O1 | L1 | relative10_shuffled | 23 | 93.67 | 31.81 | 30.25 |
| O1 | L2 | whole_count | 23 | 93.97 | 52.03 | 59.24 |
| O1 | L2 | fixed250_ordered | 23 | 99.84 | 59.21 | 50.15 |
| O1 | L2 | fixed250_shuffled | 23 | 98.77 | 49.77 | 50.47 |
| O1 | L2 | relative10_ordered | 23 | 99.84 | 63.73 | 62.12 |
| O1 | L2 | relative10_shuffled | 23 | 96.15 | 48.40 | 49.71 |
| O2 | L1 | whole_count | 23 | 100.00 | 59.00 | 53.29 |
| O2 | L1 | fixed250_ordered | 23 | 100.00 | 64.50 | 58.56 |
| O2 | L1 | fixed250_shuffled | 23 | 100.00 | 38.36 | 34.16 |
| O2 | L1 | relative10_ordered | 23 | 100.00 | 70.34 | 65.88 |
| O2 | L1 | relative10_shuffled | 23 | 97.98 | 30.15 | 29.19 |
| O2 | L2 | whole_count | 23 | 99.33 | 59.87 | 54.80 |
| O2 | L2 | fixed250_ordered | 23 | 99.51 | 63.05 | 61.75 |
| O2 | L2 | fixed250_shuffled | 23 | 99.82 | 55.01 | 49.99 |
| O2 | L2 | relative10_ordered | 23 | 100.00 | 67.19 | 64.50 |
| O2 | L2 | relative10_shuffled | 23 | 98.32 | 54.70 | 52.10 |
| O3 | L1 | whole_count | 23 | 100.00 | 58.78 | 59.96 |
| O3 | L1 | fixed250_ordered | 23 | 100.00 | 62.99 | 60.45 |
| O3 | L1 | fixed250_shuffled | 23 | 100.00 | 39.20 | 34.67 |
| O3 | L1 | relative10_ordered | 23 | 100.00 | 70.00 | 67.23 |
| O3 | L1 | relative10_shuffled | 23 | 98.82 | 31.23 | 28.20 |
| O3 | L2 | whole_count | 23 | 90.11 | 55.44 | 61.05 |
| O3 | L2 | fixed250_ordered | 23 | 100.00 | 64.55 | 54.76 |
| O3 | L2 | fixed250_shuffled | 23 | 99.78 | 53.48 | 52.42 |
| O3 | L2 | relative10_ordered | 23 | 99.84 | 65.35 | 63.95 |
| O3 | L2 | relative10_shuffled | 23 | 98.52 | 52.67 | 50.97 |
| T1 | L1 | whole_count | 23 | 97.41 | 53.70 | 51.46 |
| T1 | L1 | fixed250_ordered | 23 | 100.00 | 60.00 | 59.28 |
| T1 | L1 | fixed250_shuffled | 23 | 100.00 | 36.49 | 32.26 |
| T1 | L1 | relative10_ordered | 23 | 100.00 | 68.32 | 67.67 |
| T1 | L1 | relative10_shuffled | 23 | 94.09 | 28.55 | 29.61 |
| T1 | L2 | whole_count | 23 | 85.98 | 57.02 | 55.34 |
| T1 | L2 | fixed250_ordered | 23 | 95.28 | 60.24 | 62.45 |
| T1 | L2 | fixed250_shuffled | 23 | 98.73 | 54.20 | 50.39 |
| T1 | L2 | relative10_ordered | 23 | 94.78 | 65.67 | 70.73 |
| T1 | L2 | relative10_shuffled | 23 | 91.26 | 53.24 | 49.62 |
| T2 | L1 | whole_count | 23 | 98.21 | 52.81 | 51.34 |
| T2 | L1 | fixed250_ordered | 23 | 100.00 | 60.46 | 59.75 |
| T2 | L1 | fixed250_shuffled | 23 | 100.00 | 36.52 | 34.68 |
| T2 | L1 | relative10_ordered | 23 | 100.00 | 67.82 | 67.46 |
| T2 | L1 | relative10_shuffled | 23 | 92.88 | 27.71 | 30.24 |
| T2 | L2 | whole_count | 23 | 85.93 | 56.38 | 59.43 |
| T2 | L2 | fixed250_ordered | 23 | 99.33 | 62.97 | 58.33 |
| T2 | L2 | fixed250_shuffled | 23 | 99.82 | 54.05 | 50.65 |
| T2 | L2 | relative10_ordered | 23 | 96.25 | 71.28 | 70.29 |
| T2 | L2 | relative10_shuffled | 23 | 94.14 | 51.35 | 51.95 |
| T3 | L1 | whole_count | 23 | 94.12 | 52.50 | 51.02 |
| T3 | L1 | fixed250_ordered | 23 | 100.00 | 58.38 | 56.69 |
| T3 | L1 | fixed250_shuffled | 23 | 98.85 | 35.47 | 35.11 |
| T3 | L1 | relative10_ordered | 23 | 99.03 | 68.85 | 74.28 |
| T3 | L1 | relative10_shuffled | 23 | 98.67 | 29.14 | 30.23 |
| T3 | L2 | whole_count | 23 | 74.66 | 55.52 | 56.77 |
| T3 | L2 | fixed250_ordered | 23 | 88.59 | 54.13 | 57.84 |
| T3 | L2 | fixed250_shuffled | 23 | 86.66 | 48.14 | 51.68 |
| T3 | L2 | relative10_ordered | 23 | 90.57 | 63.35 | 70.33 |
| T3 | L2 | relative10_shuffled | 23 | 80.75 | 46.61 | 51.48 |
| T4 | L1 | whole_count | 23 | 100.00 | 56.69 | 54.18 |
| T4 | L1 | fixed250_ordered | 23 | 100.00 | 62.91 | 59.42 |
| T4 | L1 | fixed250_shuffled | 23 | 100.00 | 38.93 | 34.38 |
| T4 | L1 | relative10_ordered | 23 | 100.00 | 70.20 | 67.63 |
| T4 | L1 | relative10_shuffled | 23 | 98.27 | 31.74 | 29.35 |
| T4 | L2 | whole_count | 23 | 93.75 | 57.82 | 55.41 |
| T4 | L2 | fixed250_ordered | 23 | 99.84 | 61.01 | 46.86 |
| T4 | L2 | fixed250_shuffled | 23 | 97.48 | 50.04 | 46.71 |
| T4 | L2 | relative10_ordered | 23 | 99.84 | 65.18 | 66.46 |
| T4 | L2 | relative10_shuffled | 23 | 97.90 | 49.48 | 46.48 |
| D1 | L1 | whole_count | 23 | 98.35 | 51.55 | 51.74 |
| D1 | L1 | fixed250_ordered | 23 | 100.00 | 63.91 | 60.30 |
| D1 | L1 | fixed250_shuffled | 23 | 100.00 | 38.66 | 34.03 |
| D1 | L1 | relative10_ordered | 23 | 100.00 | 71.10 | 68.83 |
| D1 | L1 | relative10_shuffled | 23 | 98.41 | 30.50 | 27.71 |
| D1 | L2 | whole_count | 23 | 96.71 | 49.97 | 55.27 |
| D1 | L2 | fixed250_ordered | 23 | 93.07 | 54.81 | 60.50 |
| D1 | L2 | fixed250_shuffled | 23 | 96.99 | 45.22 | 44.63 |
| D1 | L2 | relative10_ordered | 23 | 99.65 | 64.54 | 63.17 |
| D1 | L2 | relative10_shuffled | 23 | 81.81 | 42.34 | 44.54 |
| D1 | L3 | whole_count | 23 | 96.01 | 54.97 | 57.55 |
| D1 | L3 | fixed250_ordered | 23 | 95.70 | 51.87 | 57.28 |
| D1 | L3 | fixed250_shuffled | 23 | 97.62 | 47.35 | 53.02 |
| D1 | L3 | relative10_ordered | 23 | 98.08 | 55.46 | 60.95 |
| D1 | L3 | relative10_shuffled | 23 | 97.08 | 48.05 | 54.96 |
| O0 | L1 | whole_count | 37 | 91.14 | 56.58 | 55.16 |
| O0 | L1 | fixed250_ordered | 37 | 100.00 | 64.07 | 64.30 |
| O0 | L1 | fixed250_shuffled | 37 | 99.21 | 41.22 | 35.45 |
| O0 | L1 | relative10_ordered | 37 | 100.00 | 70.70 | 69.40 |
| O0 | L1 | relative10_shuffled | 37 | 92.12 | 31.00 | 30.42 |
| O0 | L2 | whole_count | 37 | 97.04 | 54.42 | 54.44 |
| O0 | L2 | fixed250_ordered | 37 | 99.84 | 57.08 | 61.47 |
| O0 | L2 | fixed250_shuffled | 37 | 96.96 | 51.03 | 50.12 |
| O0 | L2 | relative10_ordered | 37 | 99.49 | 63.64 | 67.09 |
| O0 | L2 | relative10_shuffled | 37 | 97.18 | 52.60 | 52.28 |
| O1 | L1 | whole_count | 37 | 99.15 | 57.15 | 47.40 |
| O1 | L1 | fixed250_ordered | 37 | 100.00 | 64.33 | 61.05 |
| O1 | L1 | fixed250_shuffled | 37 | 100.00 | 40.66 | 35.49 |
| O1 | L1 | relative10_ordered | 37 | 100.00 | 72.22 | 74.62 |
| O1 | L1 | relative10_shuffled | 37 | 88.93 | 32.64 | 30.12 |
| O1 | L2 | whole_count | 37 | 87.06 | 51.45 | 54.32 |
| O1 | L2 | fixed250_ordered | 37 | 98.58 | 55.25 | 63.87 |
| O1 | L2 | fixed250_shuffled | 37 | 97.07 | 48.71 | 49.80 |
| O1 | L2 | relative10_ordered | 37 | 94.96 | 63.90 | 70.84 |
| O1 | L2 | relative10_shuffled | 37 | 92.02 | 46.91 | 49.58 |
| O2 | L1 | whole_count | 37 | 91.10 | 58.04 | 54.06 |
| O2 | L1 | fixed250_ordered | 37 | 100.00 | 61.60 | 59.75 |
| O2 | L1 | fixed250_shuffled | 37 | 100.00 | 41.76 | 35.11 |
| O2 | L1 | relative10_ordered | 37 | 100.00 | 69.78 | 76.86 |
| O2 | L1 | relative10_shuffled | 37 | 92.71 | 31.77 | 28.55 |
| O2 | L2 | whole_count | 37 | 96.05 | 53.12 | 51.33 |
| O2 | L2 | fixed250_ordered | 37 | 99.51 | 58.22 | 64.12 |
| O2 | L2 | fixed250_shuffled | 37 | 96.74 | 53.73 | 49.33 |
| O2 | L2 | relative10_ordered | 37 | 99.49 | 63.51 | 69.47 |
| O2 | L2 | relative10_shuffled | 37 | 95.17 | 53.17 | 50.01 |
| O3 | L1 | whole_count | 37 | 100.00 | 56.52 | 50.78 |
| O3 | L1 | fixed250_ordered | 37 | 100.00 | 62.68 | 60.90 |
| O3 | L1 | fixed250_shuffled | 37 | 99.18 | 41.64 | 35.18 |
| O3 | L1 | relative10_ordered | 37 | 100.00 | 69.38 | 71.84 |
| O3 | L1 | relative10_shuffled | 37 | 93.57 | 32.40 | 29.11 |
| O3 | L2 | whole_count | 37 | 99.17 | 56.20 | 51.33 |
| O3 | L2 | fixed250_ordered | 37 | 99.51 | 56.40 | 64.26 |
| O3 | L2 | fixed250_shuffled | 37 | 99.75 | 53.03 | 48.83 |
| O3 | L2 | relative10_ordered | 37 | 99.68 | 62.32 | 65.66 |
| O3 | L2 | relative10_shuffled | 37 | 98.58 | 52.51 | 46.37 |
| T1 | L1 | whole_count | 37 | 98.21 | 58.94 | 57.27 |
| T1 | L1 | fixed250_ordered | 37 | 100.00 | 61.79 | 62.92 |
| T1 | L1 | fixed250_shuffled | 37 | 100.00 | 37.38 | 34.76 |
| T1 | L1 | relative10_ordered | 37 | 100.00 | 71.22 | 75.43 |
| T1 | L1 | relative10_shuffled | 37 | 92.48 | 27.95 | 29.34 |
| T1 | L2 | whole_count | 37 | 80.58 | 54.52 | 54.51 |
| T1 | L2 | fixed250_ordered | 37 | 99.35 | 52.55 | 53.81 |
| T1 | L2 | fixed250_shuffled | 37 | 98.32 | 48.70 | 47.89 |
| T1 | L2 | relative10_ordered | 37 | 95.49 | 65.86 | 69.94 |
| T1 | L2 | relative10_shuffled | 37 | 82.69 | 49.00 | 51.55 |
| T2 | L1 | whole_count | 37 | 88.26 | 54.69 | 55.73 |
| T2 | L1 | fixed250_ordered | 37 | 100.00 | 60.16 | 58.56 |
| T2 | L1 | fixed250_shuffled | 37 | 100.00 | 36.72 | 35.03 |
| T2 | L1 | relative10_ordered | 37 | 99.51 | 69.06 | 76.67 |
| T2 | L1 | relative10_shuffled | 37 | 93.40 | 29.10 | 28.69 |
| T2 | L2 | whole_count | 37 | 84.00 | 52.12 | 59.42 |
| T2 | L2 | fixed250_ordered | 37 | 99.33 | 57.62 | 59.00 |
| T2 | L2 | fixed250_shuffled | 37 | 98.68 | 50.13 | 48.41 |
| T2 | L2 | relative10_ordered | 37 | 99.49 | 66.94 | 67.27 |
| T2 | L2 | relative10_shuffled | 37 | 95.60 | 46.08 | 50.40 |
| T3 | L1 | whole_count | 37 | 98.47 | 51.37 | 53.51 |
| T3 | L1 | fixed250_ordered | 37 | 100.00 | 62.34 | 63.32 |
| T3 | L1 | fixed250_shuffled | 37 | 100.00 | 37.54 | 35.72 |
| T3 | L1 | relative10_ordered | 37 | 100.00 | 70.38 | 75.52 |
| T3 | L1 | relative10_shuffled | 37 | 98.96 | 27.11 | 28.57 |
| T3 | L2 | whole_count | 37 | 98.18 | 52.06 | 45.38 |
| T3 | L2 | fixed250_ordered | 37 | 99.65 | 50.73 | 46.71 |
| T3 | L2 | fixed250_shuffled | 37 | 98.86 | 45.90 | 45.96 |
| T3 | L2 | relative10_ordered | 37 | 94.63 | 59.74 | 65.97 |
| T3 | L2 | relative10_shuffled | 37 | 90.49 | 47.49 | 49.94 |
| T4 | L1 | whole_count | 37 | 89.76 | 56.74 | 56.04 |
| T4 | L1 | fixed250_ordered | 37 | 100.00 | 65.40 | 61.16 |
| T4 | L1 | fixed250_shuffled | 37 | 100.00 | 40.36 | 35.66 |
| T4 | L1 | relative10_ordered | 37 | 100.00 | 72.99 | 72.85 |
| T4 | L1 | relative10_shuffled | 37 | 99.15 | 31.35 | 30.18 |
| T4 | L2 | whole_count | 37 | 93.15 | 51.88 | 62.28 |
| T4 | L2 | fixed250_ordered | 37 | 99.84 | 50.92 | 56.38 |
| T4 | L2 | fixed250_shuffled | 37 | 98.62 | 48.91 | 48.38 |
| T4 | L2 | relative10_ordered | 37 | 99.84 | 60.27 | 59.52 |
| T4 | L2 | relative10_shuffled | 37 | 95.75 | 48.87 | 51.89 |
| D1 | L1 | whole_count | 37 | 96.81 | 56.26 | 47.29 |
| D1 | L1 | fixed250_ordered | 37 | 100.00 | 62.75 | 63.87 |
| D1 | L1 | fixed250_shuffled | 37 | 99.35 | 40.29 | 33.23 |
| D1 | L1 | relative10_ordered | 37 | 100.00 | 73.26 | 73.27 |
| D1 | L1 | relative10_shuffled | 37 | 98.50 | 32.02 | 31.94 |
| D1 | L2 | whole_count | 37 | 86.42 | 48.89 | 46.27 |
| D1 | L2 | fixed250_ordered | 37 | 92.10 | 51.47 | 58.61 |
| D1 | L2 | fixed250_shuffled | 37 | 84.06 | 38.02 | 40.27 |
| D1 | L2 | relative10_ordered | 37 | 92.31 | 58.88 | 64.75 |
| D1 | L2 | relative10_shuffled | 37 | 89.61 | 35.03 | 36.22 |
| D1 | L3 | whole_count | 37 | 90.65 | 42.27 | 43.57 |
| D1 | L3 | fixed250_ordered | 37 | 89.21 | 45.74 | 47.70 |
| D1 | L3 | fixed250_shuffled | 37 | 85.75 | 40.89 | 44.78 |
| D1 | L3 | relative10_ordered | 37 | 95.71 | 50.13 | 51.86 |
| D1 | L3 | relative10_shuffled | 37 | 90.80 | 38.37 | 43.63 |

