# Exp16 — Prefix Credit, Matched-Budget Writing, and Persistent-Spike Reward

## Report Scope and Evidence Cutoff

This report reconstructs the **Exp16 family**: `experiment_16_prefix_supervised_selective_memory`, Exp16.1 z-only Prefix interaction, Exp16.2 matched-budget selective writing and its two persistent-firing analyses, and the actual Exp16.3 run-reward experiment. It records design intent, implemented contracts, executed runs, negative results, corrections, and the reasoning that led to successor experiments. The earlier counterfactual write-utility proposal called “Exp16.3” and the proposed “Exp16.4” are explicitly distinguished from executed cases.

Evidence was inspected on **2026-10-02**. GitHub source snapshot: `a858ce635d024181c0ccf8cebca1ca62b8d07c40`. Unity tracked HEAD: `77fdad527ef82c79a35930b4822b5ac8a81928b3`, at `/home/zhaolongwei_umass_edu/projects/writingRing`. The source snapshot and Unity HEAD are different and are not treated as the same run-time revision. The inspected Unity checkout had no tracked modifications; Exp16.2 and Exp16.3 result directories were untracked, whereas the main Exp16 and Exp16.1 artifacts were also represented in GitHub. Their Unity results remain the quantitative authority. This report does not imply that all result files are published with it. Fifty implementation, support, README, test and batch-script files were compared between the inspected GitHub snapshot and Unity checkout; their contents matched after removing terminal newline differences introduced by the local text capture. This agreement describes the current inspected source, not an undocumented reconstruction of each job's historical checkout.

Source, README revisions, validation histories, all **111 family checkpoint files**, per-run JSON/CSV, aggregate tables, Slurm accounting, and targeted Project-history searches were inspected. Project retrieval supplied dated user decisions and prior assistant summaries, rather than a complete exported conversation archive. Those retrievals establish the reasoning trace, not numerical truth. Original failed-job stdout/stderr were not recovered; scheduler outcomes and correction commits are available. Appendix C identifies these evidence boundaries.

BA denotes balanced accuracy. BA tables use percent; differences use percentage points (pp). Unless stated otherwise, **mean ± sample SD** is computed over model seeds **11, 23, 37, n=3**, with equal seed weight. Shuffle replicates are averaged within seed first. Neurons, τ groups, batches, and samples are not extra independent model seeds. The tables are descriptive paired-seed evidence, without a significance claim. Project and commit timestamps are UTC. Scheduler timestamps were checked against explicit `TZ=UTC` accounting for the four preparation jobs and matched the retained records.

## 1. Experiment Identification

| Alias | Canonical ID / protocol | Actual executed scope |
| --- | --- | --- |
| Exp16, Prefix selective memory | `experiment_16_prefix_supervised_selective_memory` / `prefix_selective_memory_v1` | Phase0A selector audit; Phase0B one-epoch Prefix calibration; 12 Phase1 runs; conditional Phase1.5 skipped; functional gate interventions |
| Exp16.1, z-only × Prefix | `experiment_16_1_z_only_prefix_interaction` / `z_only_prefix_interaction_v1` | 36 fresh models crossing gate mode with λP; 24 gated-checkpoint intervention tasks; gradient and history diagnostics |
| Exp16.2, matched budget | `experiment_16_2_matched_budget_selective_write` / `matched_budget_selective_write_v1` | 12 seed 101 calibrations; 24 fresh models in 12 paired tasks; 12 replay tasks; persistent-neuron and sustained-firing posthoc analyses |
| Actual Exp16.3, run reward | `experiment_16_3_run_reward` / `run_reward_v1` | 21 fresh LIN/CAP/SUB/EP models; 24 reused Exp16.2 scale diagnostics; six reused Core O0/O1 objective diagnostics |
| Earlier proposed “Exp16.3” | Sampled counterfactual write utility, CF90 / R10-90 / oracle controls | Retrieved design proposal, subsequently replaced; not an executed case in `run_reward_v1` |
| Proposed Exp16.4 | Native-head/high-low-random persistent-neuron ablations | Successor proposal; no canonical Exp16.4 implementation in the inspected tree |

Implementations: [main Exp16](../../scripts/experiment_16_prefix_supervised_selective_memory.py), [Exp16.1](../../scripts/experiment_16_1_z_only_prefix_interaction.py), [Exp16.2](../../scripts/experiment_16_2_matched_budget_selective_write.py), [Exp16.3](../../scripts/experiment_16_3_run_reward.py). The two genuine Exp16.2 analyses are [persistent-neuron utility](../../scripts/analyze_experiment_16_2_persistent_neurons.py) and [sustained-firing mechanism](../../scripts/analyze_experiment_16_2_sustained_firing_mechanism.py). Slurm stage names are execution stages, not additional experiment families.

All four roots follow `notebooks/artifacts/<canonical ID>/<protocol>/`. This shorthand is used below. “C0” is local to its family: it is not one immutable baseline shared across Exp15, main Exp16, Exp16.1, and Exp16.2. Exp16.3 LIN is another fresh baseline. Reusing the same seed labels does not make these separately trained checkpoints interchangeable.

## 2. Scientific Context

Exp13 separated available history from transferable consolidation. Exp14 found that better history representations or stronger early discriminability did not reliably make a shared native evidence accumulator use that information. Exp14.1's corrected Prefix grid admitted no validation-supported Prefix candidate. Exp15 then asked whether a contextual L2 write gate could select useful memory content while preserving WCCE. Its gate read current L1 spikes and previous L2 spikes, and used **g/0.9**, allowing amplification. Joint training produced little gate-specific native transfer benefit: the preceding report records GJ−C0 validation +1.25 pp and test −0.32 pp.

The retrieved discussions therefore did not start from “memory is useless” or “WCCE is wrong.” The unresolved question was whether classification credit reaches a useful **write policy**, or can instead be satisfied by stronger gain, representation adaptation, and readout shortcuts. The user retained one whole-character label and sequence-level classification as the appropriate task contract.

On October 1 at 02:40:26 UTC, the user accepted main Exp16's selector/calibration/factorial design and a truly suppressive scalar gate. After calibration removed Prefix, Exp16.1 directly tested positive Prefix coefficients in gated and ungated models. After these gates remained nearly open, the user at 05:45:35 UTC stopped further Prefix/λ/gate-complexity tuning and required matched static-versus-dynamic write budgets. The subsequent raster question shifted attention from gate statistics to systematic persistent L2 firing and the information carried by repeated spikes.

## 3. Motivation and Open Question

The family addresses a sequence of increasingly specific questions.

1. Can intermediate accumulated-evidence CE recruit useful selective writing, rather than improve only early prediction or the readout? A factorial interaction and inference interventions are required.
2. If WCCE prefers an open gate, does forcing the same mean budget expose a benefit of **dynamic allocation** over a static multiplier? Budget compliance and matched initialization are required.
3. If dynamic training improves BA, does its learned temporal variation remain necessary at inference? Learned-versus-mean/shuffle/shift replay separates this from training-time changes.
4. Does a mean gate budget actually constrain current, state, or firing, or can learned weights compensate for the multiplier?
5. Does linear repeated-spike accumulation reward redundant persistent firing? If so, diminishing reward within each firing episode should reduce persistence while preserving useful whole-character classification.

The last question changes the aggregator before the native head. It does not add an activity penalty, turn into TSCE, or train the proposed counterfactual gate. These distinctions are necessary for interpreting the final negative result.

## 4. Hypotheses

| Hypothesis | Expected evidence | Discriminating test |
| --- | --- | --- |
| Selector choice hides useful learning | Minimum validation CE retains BA and yields a defensible selector | Same-trajectory Phase0A, validation safeguard |
| Prefix specifically helps gated writing | Positive Gate×Prefix validation interaction, count-accessible improvement, non-open gate | Main Exp16 / direct positive-λ Exp16.1 factorial controls |
| Explicit previous-L2 gate input is necessary | z+history improves matched outcomes and useful intervention dependence | GZH versus GZ at each λ, with stateful L1 retained |
| Dynamic allocation helps under matched budget | Gρ−Sρ native and representation gains | Exp16.2's cloned S/G pairs and budget-compliant checkpoints |
| Temporal variation is causally used | Learned > per-sample Mean | Fixed-checkpoint record-and-replay |
| Temporal placement is causally used | Learned > shuffled/shifted schedules | Distribution-preserving schedule interventions |
| Mean gate is a capacity/activity budget | Lower ρ reliably constrains effective write/state/activity | u, gu, W2, I, V, firing and τ-aware drive diagnostics |
| High firing is entirely nuisance | Removal consistently improves held-out classification without losing unique information | Frozen versus refitted probes; duration/class/user controls |
| Linear episode reward drives redundant persistence | Diminishing reward lowers occupancy/run length and preserves/improves BA | Fresh LIN/CAP4/8/16/24/SUB8/EP1 under sequence-level CE |

These are alternative explanations with distinct endpoints. A smaller Relative10−WholeCount gap, a lower gate mean, or a successful Slurm exit alone cannot establish selective memory.

## 5. Experimental Design

### 5.1 Main Exp16: selector audit and conditional Prefix interaction

Phase0A trains **three fresh ungated WCCE trajectories**, one per formal seed. BA-first and minimum-CE checkpoints are selected from the same trajectory. The CE selector is adopted only if its validation BA is within 0.5 pp of BA-first on at least two seeds. Preserved epoch 10/20/30/50 snapshots support Phase0B; they are not separate trained models.

Phase0B crosses three seeds, four snapshot epochs, and λP={0,.01,.03,.05,.10}: **60 one-epoch lookaheads**. Each positive coefficient is compared with the λ=0 one-epoch continuation from the same seed/snapshot. Eligibility requires validation CE improvement ≥.005 and BA decrease no worse than .5 pp. A coefficient needs support on at least two seeds; the smallest supported coefficient is chosen. There is a legal λP=0 outcome. Test metrics do not select the coefficient or selector.

Phase1 freshly trains C0, P0, G0, GP_J for each seed. It does **not** continue Exp15 or the Phase0A selected checkpoint. Planned Prefix gradient-routing attribution cases GP_stopR, GP_gate, GP_noGate require stable positive validation interaction. The observed λP=0 removes Prefix from every Phase1 run, and attribution is skipped. Thus Phase1's P0/GP_J names do not establish that a positive Prefix treatment happened.

### 5.2 Exp16.1: direct positive-λ factorial test

Each seed has C0; P at .01/.03/.05; GZ0 and GPZ at those coefficients; GZH0 and GPZH at those coefficients. These are **12 cases × three seeds = 36 fresh models**. Z means current L1 spikes; ZH adds previous L2 spikes. Both are suppressive gates initialized at .9. Positive coefficients are tested directly, without main Exp16's ungated calibration veto.

For each gate mode, the primary interaction is

\[
\Delta_g(\lambda)=[M(GP_g^\lambda)-M(G_g^0)]-[M(P^\lambda)-M(C0)].
\]

The same contrast is calculated for validation/test native BA, L2 WholeCount, Relative10, and collapse gap. All 24 gated checkpoints receive learned/one/mean/shuffle interventions. No best-looking test coefficient is promoted to a confirmed treatment.

The initial single-node design was replaced before the completed run set by independent 1-CPU arrays. The observed ten-node set and `single_node_verified=false` are recorded, despite the README's stale “on one node” title and motivation line. Pairing is by seed and deterministic parameter initialization; a same-node control was not achieved here.

### 5.3 Exp16.2: calibration, matched formal models, mandatory replay

Calibration uses **seed 101 only**, ρ={.9,.7,.5}, and λB={.1,.3,1,3}, for 12 runs. The late-five-epoch validation mean absolute per-sample budget error must be ≤.02. Selection first seeks the smallest coefficient compliant for all three ρ values; otherwise it uses per-ρ smallest compliant values. BA does not choose λB.

Formal cases are C0, GZ0, S90/G90, S70/G70, S50/G50 for each seed, **24 models**. A common parameter state is explicitly cloned. Static Sρ and initially uniform dynamic Gρ have identical first-forward evidence, L2 spikes, current and membrane; C0≡S100 is checked. Epoch-specific deterministic sample permutations and hashes enforce matched batches. Each pair runs sequentially in one 1-CPU task, with S/G order counterbalanced by seed/ρ. Common-epoch hashes and checkpoint initialization hashes agree in the retained records.

Constrained Gρ checkpoints must satisfy validation budget MAE≤.02 **before** BA/CE selection. If none qualify, the run is `BUDGET_INVALID`; closest-budget output is diagnostic only. All nine constrained formal models have qualifying checkpoints. C0, GZ0 and Sρ use ordinary BA-first selection. Unlike main Exp16/16.1, Exp16.2's stored formal selection histories start at **epoch 1**, without an eligible epoch 0 checkpoint.

Phase2 records the learned gate schedule, then replays learned, exact per-sample mean, three within-valid-range shuffles, and circular shifts .25T/.50T/.75T at fixed selected weights. This is **12 dynamic checkpoints × eight schedules = 96 native rows**, with matched probes. Gate schedules are interventions on writes, not retrained gate policies.

### 5.4 Actual Exp16.3: episode reward and checkpoint-only diagnostics

All seven formal cases use a fresh, exactly shared initialization per seed, deterministic epoch batches, the same backbone/head geometry, and **one sequence-level CE**. LIN is ordinary valid-mean spike accumulation. CAP4/8/16/24 cap reward per consecutive firing run. SUB8 applies a saturating sublinear reward. EP1 counts episode onsets. No gate, Prefix, TSCE, activity penalty, utility loss, or additional regularizer is used in formal training.

Phase0 diagnostics reuse **24 Exp16.2 checkpoints** for drive/weight scale analysis and **six historical Core O0/WCCE and O1/TSCE checkpoints** for activity and run-feature probes. They do not add independent seeds, retrain SNNs, or select formal cases from test results. Formal checkpoints are selected on validation BA under their own reward aggregator, then CE, then earliest tie; the original head is also evaluated with a linear aggregator at the same learned weights.

## 6. Implementation

### 6.1 Shared data, dynamics, and optimization

The locked Core cache is `core_benchmark_v1.0`, identity `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`, dataset SHA-256 `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a`. Loaders adapt the protocol object to the current version; this does not imply regeneration of the historical dataset. Actions 0+1 use 64 Hz, 256 padded steps, valid lengths, and `custom_wavelet_polarity_split_abs_events_v1`: **30 unsigned event input channels**, drawn from the 36-channel producer. Continuous producer channels are not additional SNN inputs.

There are 581/126/146 train/validation/test sequences. Classes are A, B, C, D, E, G, H, I, J, K, L, X. Train users: 0,1,2,5,7,8,11,12,13,14,15,18,19,20; validation: 4,9,16; test: 3,6,10. Split seed is 12345; formal model seeds are 11/23/37. Seed 101 is a calibration seed.

The backbone has two 128-neuron layers, bias-free signed feedforward matrices and a bias-free signed analog class head. In each layer shifts 2/3/4 allocate 43/43/42 neurons, α=.75/.875/.9375. At Δt=15.625 ms, synaptic e-folding times are approximately 54.31/117.01/242.10 ms. Membrane τ=22.54 ms; threshold=.5; subtractive reset; binary forward spikes. There is temporal state in I/V, without a trainable recurrent matrix.

\[
I_t=\alpha I_{t-1}+Wx_t,\qquad
V_t=\beta V_{t-1}+I_t-\theta s_t.
\]

The new drive is **unnormalized**, without a (1−α) factor. The native baseline head computes eₜ=Rsᴸ²ₜ and WCCE is CE of valid-mean logits. Native argmax is unchanged by using their valid sum. Adam uses LR=.001, weight decay=0, batch 128, max 100 epochs, min 20, patience 30. Checkpoints use validation native BA, then validation CE, then earliest tie, with 1e−12 comparison tolerance. Tests never enter the selectors. Main Exp16/16.1 and Exp16.3 permit epoch 0; Exp16.2's implementation starts checkpoint eligibility at epoch 1.

### 6.2 Suppressive writing and Prefix credit

\[
g_t=\sigma(w_z^\top s^{L1}_t+w_h^\top s^{L2}_{t-1}+b),\qquad
I^{L2}_t=\alpha I^{L2}_{t-1}+g_tW_2s^{L1}_t.
\]

The history term is absent in z-only models. Zero gate vectors and b=logit(.9) initialize g=.9; Exp16.2 constrained gates use b=logit(ρ). The range is **(0,1)** with no g/.9 amplification. The scalar gates the whole newly written message; it does not directly gate decay, membrane retention, or outgoing evidence. Z-only removes explicit L2 feedback, while current L1 spikes still contain L1 temporal context.

Prefix uses accumulated **mean evidence** through ceil(.5T) and ceil(.75T):

\[
L_P=.5\,CE(\bar e_{1:\lceil .5T\rceil},y)+.5\,CE(\bar e_{1:\lceil .75T\rceil},y),
\qquad L=L_{WCCE}+\lambda_P L_P.
\]

It is not a Relative10 teacher, timestep CE, spike penalty, or direct desired gate target. Main Exp16's unexecuted attribution routes would exclude R, target only gate parameters, or exclude the gate from Prefix credit. They are implemented conditional branches, not measured outcomes.

Exp16.2 adds

\[
\bar g_i=T_i^{-1}\sum_{t<T_i}g_{it},\qquad
L_B=N^{-1}\sum_i(\bar g_i-\rho)^2,\qquad
L=L_{WCCE}+\lambda_B L_B.
\]

The squared training penalty is distinct from the **absolute-error** validation compliance criterion. Each sequence receives equal budget weight. Matching mean g does not match \(\|g u\|\), I, V, spike count, weight scale, or memory capacity.

### 6.3 Run-reward forward and gradient contract

For neuron j with consecutive firing-run lengths ℓⱼᵦ,

\[
A_j=T^{-1}\sum_b\phi(\ell_{jb}),\qquad L=CE(RA,y).
\]

LIN: φ(ℓ)=ℓ. CAPK: min(ℓ,K). SUB8: 8(1−exp(−ℓ/8)). EP1: one per nonempty episode. Duration normalization uses each sequence's valid T in every case. LIN is exactly WCCE for the bias-free linear head.

Run position is computed from **detached binary forward spikes**. Surrogate gradient through the current spike is weighted by the hypothetical marginal reward φ(previous run length+1)−φ(previous run length). A silent timestep after a gap receives first-spike credit; a CAP continuation beyond K gets zero direct reward credit; EP1 gives direct credit only at potential onsets. The method does not differentiate through discrete run boundaries. Earlier spikes can still affect later SNN states through dynamics. Consequently, the intervention changes both forward evidence and temporal gradient allocation; a bad CAP result does not isolate an exact differentiable objective over all possible run-boundary changes.

## 7. Executed Runs and Validity Audit

Main Exp16: prepare `65103451`; Phase0A `65103452` (3); selector finalizer `65103453`; Phase0B `65103454` (60); coefficient selector `65103455`; Phase1 `65103456` (12); decision finalizer `65103457`; conditional attribution `65103458` (9 scheduled tasks, **zero attribution models**); interventions `65103459` (15 scheduled tasks, only six available G0/GP_J checkpoints); finalizer `65103460`. All retained top-level task records are `COMPLETED`, exit 0:0. Conditional skip returns success, so scheduler completion must not be counted as nine attribution results.

Exp16.1's original `65106790` was `CANCELLED by 3883`, exit 0:0, no start, elapsed 0. It is excluded from results. Completed arrays: prepare `65107098`, training `65107099` (36), ablation `65107100` (24), finalizer `65107101`. The manifest reports PASS, 36 training runs, 24 gated runs, 18 interaction rows, independent arrays, and ten hostnames. All completed array tasks have exit 0:0.

Exp16.2's initial prepare `65111341` **FAILED, 127:0** after three seconds. Its dependency stages `65111342`–`65111346` were cancelled before starting, including three 12-task arrays. The corrected pipeline completed: prepare `65111359`, calibration `65111360` (12), calibration finalizer `65111361`, paired training `65111362` (12 tasks / 24 models), replay `65111363` (12), finalizer `65111364`. The summary reports **24/24 valid**, with no budget-invalid model. The failed submission contributes no formal model or extra seed.

Exp16.3 completed prepare `65131206`, scale diagnostics `65131207` (24), objective diagnostics `65131208` (6), formal training `65131209` (21), finalizer `65131210`, all exit 0:0. Its PASS summary contains 21/21 formal runs.

The family has **93 formal fresh models**: 12+36+24+21. In addition, three full Phase0A trajectories and twelve seed 101 calibration trajectories were trained; 60 one-epoch lookaheads are separate short continuations. The 111 retained family `.pt` files comprise 93 formal selected checkpoints plus main Exp16's 12 snapshot and six selector files. They are not 111 independently trained models. The 24+6 Exp16.3 reused diagnostic models likewise do not enlarge n.

The audit matched formal CSV identities to per-run native JSON, checkpoint experiment/case/seed, selected/last epochs and exact validation-selector histories. All 93 selectors reproduced when their 1e−12 tolerance was applied. Nine Gρ selected histories satisfy MAE≤.02. Within each seed, all eight Exp16.2 and seven Exp16.3 cases share their documented initialization hash and matching common-epoch sampler hashes. Phase2 learned native BA matches Phase1 for all 12 dynamic checkpoints exactly. Appendix A records the selected epochs, individual BA and checksums.

## 8. Evaluation and Diagnostics

Primary representation comparisons use **L2 spike probes, no-bias decoder**. WholeCount sums valid spikes into 128 features. Fixed250 concatenates sixteen 16-step bins, 2048 features; Relative10 concatenates ten normalized-progress bins, 1280 features. Ordered probes retain bin positions. Fixed250 shuffle permutes only complete valid bins, preserving a partial last bin and padding; Relative10 permutes all ten bins. Stable sample-ID-keyed permutations are applied to train/validation/test, and a readout is refitted for each transform.

Core probes use scale-only `StandardScaler(with_mean=False)`, both for no-bias and affine. No-bias sets `fit_intercept=False`; affine adds an intercept. C={.001,.01,.1,1,10,100} is selected from converged candidates by validation BA, retaining smaller C on a tie. **Temporal probe shuffles use five seeds 101/211/307/401/503. Gate-schedule shuffles use three seeds 101/211/307.** The two replication counts are different and neither changes model-seed n. Affine and L1 results are retained but not substituted for the primary L2/no-bias comparison. A native head and a refitted probe are different classifiers.

Collapse gap is BA(Relative10 ordered)−BA(WholeCount), in pp. A decrease caused by Relative10 deteriorating is not full consolidation success. Ordered−shuffled gaps quantify order-accessible information under matched refitting; they do not prove the deployed native head uses that order.

Main Exp16/16.1 also retain trajectory prototype retrieval and final-state history diagnostics. Retrieval compares resampled test trajectories against training class prototypes. Final L2 pre-reset-state readouts compare windows 50/100/250/500/1000 ms/full, with a centered, affine balanced logistic decoder (C=1), training-user-group three-fold CV and all-train→held-out-test transfer. That diagnostic has different preprocessing, feature geometry and validation procedure from Core temporal probes; its CV score is not native seen-user training BA.

Gate interventions report intervention−learned BA. Mean replacement preserves each sequence's recorded mean; shuffle preserves its gate distribution; circular shifts preserve distribution but alter alignment. In Exp16.2 these replay recorded schedules rather than recomputing a closed-loop altered policy. Per-intervention probes are refitted on the corresponding trajectories.

Persistent-neuron utility is measured in a separately fitted WholeCount no-bias probe, with the SNN frozen. Single-neuron and top-k frozen ablations replace features by their **training mean**, not zero. Top-k ranks training occupancy and removes ceil(fraction×128): 7/13/26/39 neurons at 5/10/20/30%. Refit controls delete the feature dimensions and retrain only the probe. The stored `delta_ba_pp_vs_full` is **full BA−ablated BA**, a positive loss; `delta_ce` is ablated CE−full CE. These interventions do not remove a neuron from the recurrent temporal dynamics or retrain the SNN.

Occupancy is valid spike count/T, per sequence and neuron, then averaged. A longest run is the longest consecutive binary-spike episode within a sequence. Run lengths in this report are **steps**, with one step 15.625 ms. Activity summaries weight the 43/43/42 τ groups by neuron count before computing seed SD. This slightly differs from averaging three τ-group means equally in some conversation summaries. User/class η² and occupancy correlations are descriptive associations, not causal information decompositions.

## 9. Quantitative Results

### 9.1 Main Exp16: calibration rejects Prefix, gates are almost open

Same-trajectory selector audit (BA percent, CE unscaled):

| seed | selector | selected_epoch | val_ba | val_mean_logit_ce | test_ba |
| --- | --- | --- | --- | --- | --- |
| 11 | ba_first | 91 | 55.2561 | 1.3501 | 61.7489 |
| 11 | ce_min | 99 | 53.1097 | 1.3134 | 59.683 |
| 23 | ba_first | 63 | 53.4566 | 1.4222 | 50.8524 |
| 23 | ce_min | 92 | 53.165 | 1.2625 | 51.3115 |
| 37 | ba_first | 89 | 59.1444 | 1.2485 | 56.0168 |
| 37 | ce_min | 99 | 58.2816 | 1.2367 | 58.9533 |

The CE selector retains validation BA within .5 pp on only **one of three** seeds. BA-first remains the rule. Every positive Prefix coefficient has **zero eligible lookahead points and zero seed support**; λP*=0. The rejection is stronger than “not two seeds”: no positive coefficient passed a single snapshot's joint CE/BA safeguard. It is specific to this one-epoch ungated calibration, not a theorem that Prefix cannot help a gate.

Formal native results:

| Case | train_ba | val_ba | test_ba |
| --- | --- | --- | --- |
| C0 | 88.40 ± 6.29 | 56.58 ± 2.89 | 55.83 ± 4.88 |
| P0 | 88.86 ± 6.77 | 56.34 ± 3.55 | 56.27 ± 5.45 |
| G0 | 91.68 ± 1.62 | 59.07 ± 2.00 | 56.76 ± 1.32 |
| GP_J | 91.57 ± 1.83 | 59.32 ± 2.06 | 57.78 ± 4.20 |

P0−C0 and GP_J−G0 cannot be attributed to Prefix because both actual coefficients are zero. The validation difference-in-differences is +.499 pp, with seed interactions +4.907/0/−3.410 pp: only one positive seed, so attribution is disabled. The named λ=0 duplicates exhibit numerical trajectory divergence. Initial common-parameter RNG streams are paired by parameter name and seed, but the main design did not enforce one physical CPU/node family or store explicit cloned-state/epoch hashes. The original Exp16.1 README recorded cross-node-family divergence as the motivation for a proposed single-node control. The available evidence does not quantify its sole cause.

Primary L2 probes:

| Case | whole_count | fixed250_ordered | fixed250_shuffled | relative10_ordered | relative10_shuffled | R10−WC pp |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 54.25 ± 7.19 | 58.71 ± 5.93 | 50.03 ± 5.14 | 67.94 ± 4.89 | 50.61 ± 6.36 | 13.69 ± 2.75 |
| G0 | 57.24 ± 2.80 | 60.50 ± 4.07 | 51.08 ± 1.57 | 65.37 ± 2.85 | 51.74 ± 2.19 | 8.13 ± 5.27 |
| GP_J | 56.20 ± 3.12 | 61.04 ± 3.74 | 53.20 ± 3.08 | 66.32 ± 3.47 | 53.43 ± 3.99 | 10.13 ± 2.69 |
| P0 | 54.61 ± 7.52 | 58.19 ± 5.64 | 50.60 ± 5.75 | 66.81 ± 4.97 | 50.38 ± 5.91 | 12.20 ± 4.23 |

G0 versus C0 descriptively raises WholeCount while lowering Relative10; the gap narrows from 13.69 to 8.13 pp. This is not a measured Gate×Prefix effect. Selected gates:

| Case | Test mean g | P(g>.9) | Mean g(1−g) |
| --- | --- | --- | --- |
| G0 | 0.9870 ± 0.0004 | 1.000 ± 0.000 | 0.0120 ± 0.0003 |
| GP_J | 0.9871 ± 0.0003 | 1.000 ± 0.000 | 0.0119 ± 0.0003 |

Native intervention−learned differences, three gate shuffles averaged within seed:

| Case | gate_one | gate_time_mean | gate_time_shuffle | learned |
| --- | --- | --- | --- | --- |
| G0 | -0.20 ± 0.34 | 0.00 ± 0.00 | -0.07 ± 0.11 | 0.00 ± 0.00 |
| GP_J | -0.20 ± 0.34 | 0.08 ± 0.14 | -0.05 ± 0.13 | 0.00 ± 0.00 |

The learned schedule is not materially necessary in these checkpoints. Phase1 names and attractive probe values therefore do not rescue a selective-Prefix claim.

### 9.2 Exp16.1: positive Prefix is tested, but no stable interaction emerges

| Case | train_ba | val_ba | test_ba |
| --- | --- | --- | --- |
| C0 | 88.36 ± 6.26 | 55.95 ± 2.91 | 56.21 ± 5.45 |
| P_lp0p01 | 89.17 ± 5.60 | 56.40 ± 2.93 | 57.89 ± 5.08 |
| P_lp0p03 | 89.32 ± 5.31 | 55.53 ± 3.06 | 55.40 ± 2.34 |
| P_lp0p05 | 90.17 ± 3.13 | 55.73 ± 1.35 | 57.35 ± 5.46 |
| GZ0 | 88.73 ± 6.86 | 57.41 ± 0.79 | 55.69 ± 4.81 |
| GPZ_lp0p01 | 88.87 ± 6.17 | 57.31 ± 1.46 | 56.44 ± 5.66 |
| GPZ_lp0p03 | 92.32 ± 1.45 | 56.19 ± 1.52 | 55.79 ± 5.84 |
| GPZ_lp0p05 | 91.65 ± 2.55 | 56.44 ± 2.41 | 55.55 ± 1.35 |
| GZH0 | 92.89 ± 0.97 | 59.29 ± 3.77 | 56.57 ± 3.35 |
| GPZH_lp0p01 | 92.77 ± 1.27 | 56.76 ± 3.29 | 56.88 ± 3.57 |
| GPZH_lp0p03 | 88.69 ± 5.91 | 56.56 ± 2.37 | 55.53 ± 4.91 |
| GPZH_lp0p05 | 91.34 ± 2.31 | 57.43 ± 1.86 | 57.01 ± 5.05 |

Paired interaction estimates:

| Gate | λP | val_ba_interaction_pp | test_ba_interaction_pp | whole_count_interaction_pp | relative10_ordered_interaction_pp |
| --- | --- | --- | --- | --- | --- |
| z_history | 0.01 | -2.98 ± 3.65 | -1.37 ± 1.28 | -1.32 ± 1.45 | 0.70 ± 2.58 |
| z_history | 0.03 | -2.31 ± 4.48 | -0.24 ± 5.38 | -1.60 ± 5.60 | 0.09 ± 4.25 |
| z_history | 0.05 | -1.64 ± 1.89 | -0.71 ± 4.60 | -2.52 ± 4.02 | 0.80 ± 8.74 |
| z_only | 0.01 | -0.54 ± 4.32 | -0.94 ± 2.18 | -2.34 ± 3.68 | 1.91 ± 11.84 |
| z_only | 0.03 | -0.79 ± 3.77 | 0.90 ± 4.48 | -1.33 ± 6.94 | -0.22 ± 5.01 |
| z_only | 0.05 | -0.74 ± 4.68 | -1.29 ± 3.86 | -1.19 ± 2.59 | -2.07 ± 9.65 |

All six **mean validation interactions are negative**. Z-only λ=.03 is the one positive mean test interaction (+.90 pp), but its seed values are **+6.08/−1.65/−1.71 pp**. It is seed 11-driven and not supported by validation. WholeCount interaction means are negative for every gate/λ combination. This is a direct negative result for the tested Prefix recruitment mechanism, unlike main Exp16 where Prefix was never active.

| Case | Test mean g | P(g>.9) | Mean g(1−g) |
| --- | --- | --- | --- |
| GPZH_lp0p01 | 0.9874 ± 0.0007 | 1.000 ± 0.000 | 0.0117 ± 0.0006 |
| GPZH_lp0p03 | 0.9865 ± 0.0005 | 1.000 ± 0.000 | 0.0125 ± 0.0005 |
| GPZH_lp0p05 | 0.9870 ± 0.0002 | 1.000 ± 0.000 | 0.0121 ± 0.0002 |
| GPZ_lp0p01 | 0.9762 ± 0.0035 | 1.000 ± 0.000 | 0.0226 ± 0.0033 |
| GPZ_lp0p03 | 0.9777 ± 0.0010 | 1.000 ± 0.000 | 0.0212 ± 0.0010 |
| GPZ_lp0p05 | 0.9770 ± 0.0018 | 1.000 ± 0.000 | 0.0218 ± 0.0017 |
| GZ0 | 0.9763 ± 0.0036 | 1.000 ± 0.000 | 0.0225 ± 0.0034 |
| GZH0 | 0.9875 ± 0.0002 | 1.000 ± 0.000 | 0.0116 ± 0.0002 |

Z-only is less saturated than z+history, but every retained case/seed has test P(g>.9)=1. Positive Prefix does not recruit a substantially closed policy. Native intervention−learned results:

| Case | gate_one | gate_time_mean | gate_time_shuffle | learned |
| --- | --- | --- | --- | --- |
| GPZH_lp0p01 | 0.00 ± 0.00 | -0.33 ± 1.17 | -0.07 ± 0.66 | 0.00 ± 0.00 |
| GPZH_lp0p03 | 0.20 ± 0.34 | 0.83 ± 0.18 | 0.36 ± 0.34 | 0.00 ± 0.00 |
| GPZH_lp0p05 | 0.20 ± 0.34 | -0.20 ± 0.34 | -0.16 ± 0.51 | 0.00 ± 0.00 |
| GPZ_lp0p01 | -0.53 ± 0.46 | -0.55 ± 0.52 | -0.26 ± 0.63 | 0.00 ± 0.00 |
| GPZ_lp0p03 | -0.76 ± 0.85 | -0.14 ± 0.40 | -0.54 ± 0.87 | 0.00 ± 0.00 |
| GPZ_lp0p05 | 0.00 ± 0.60 | -0.20 ± 0.34 | -0.08 ± 0.83 | 0.00 ± 0.00 |
| GZ0 | 0.68 ± 1.18 | 0.20 ± 0.83 | 0.23 ± 1.03 | 0.00 ± 0.00 |
| GZH0 | 0.00 ± 0.00 | 0.20 ± 0.34 | 0.16 ± 0.44 | 0.00 ± 0.00 |

Some individual small effects occur, including GPZ .01 mean replacement −.55 pp and GPZ .03 always-open −.76 pp. They do not form a stable positive factorial validation result or a robust timing-dependence pattern. The z+history .03 mean intervention actually improves BA by .83 pp. Removing the explicit controller history is therefore not shown to solve the mechanism or to establish that L2 history itself is unnecessary.

### 9.3 Exp16.2: a G90 training benefit, without native timing necessity

Calibration error and coverage:

| rho | lambda_budget | late_budget_mae | late_coverage_within_0p05 | compliant |
| --- | --- | --- | --- | --- |
| 0.9 | 0.1 | 0.054122 | 0.333333 | False |
| 0.9 | 0.3 | 0.019096 | 1.0 | True |
| 0.9 | 1.0 | 0.007012 | 1.0 | True |
| 0.9 | 3.0 | 0.004012 | 1.0 | True |
| 0.7 | 0.1 | 0.074191 | 0.069841 | False |
| 0.7 | 0.3 | 0.028623 | 0.960317 | False |
| 0.7 | 1.0 | 0.011282 | 1.0 | True |
| 0.7 | 3.0 | 0.006438 | 1.0 | True |
| 0.5 | 0.1 | 0.106946 | 0.0 | False |
| 0.5 | 0.3 | 0.04466 | 0.638095 | False |
| 0.5 | 1.0 | 0.016032 | 0.996825 | True |
| 0.5 | 3.0 | 0.009197 | 1.0 | True |

λB=.3 is compliant atρ=.9 only; **1.0 is the smallest common compliant coefficient** and is used at all three budgets. Formal native results:

| Case | train_ba | val_ba | test_ba |
| --- | --- | --- | --- |
| C0 | 91.73 ± 2.59 | 56.44 ± 3.47 | 55.93 ± 3.82 |
| GZ0 | 81.20 ± 16.70 | 54.66 ± 1.42 | 52.41 ± 7.91 |
| S90 | 90.94 ± 4.96 | 56.59 ± 2.43 | 55.42 ± 1.77 |
| G90 | 92.59 ± 1.42 | 56.67 ± 3.61 | 57.97 ± 3.59 |
| S70 | 92.29 ± 0.87 | 56.29 ± 0.65 | 57.46 ± 3.20 |
| G70 | 87.96 ± 2.56 | 55.95 ± 2.60 | 55.23 ± 5.29 |
| S50 | 91.14 ± 1.46 | 56.67 ± 1.45 | 56.70 ± 4.03 |
| G50 | 88.09 ± 5.27 | 54.87 ± 3.85 | 56.04 ± 3.81 |

The primary **Gρ−Sρ** contrasts are:

| ρ | delta_native_pp | delta_wholecount_pp | delta_fix250_pp | delta_relative10_pp | delta_collapse_pp | Native seed 11 / 23 / 37 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 2.55 ± 1.83 | 2.53 ± 1.71 | -1.57 ± 3.55 | -3.14 ± 1.92 | -5.67 ± 0.22 | +4.65 / +1.34 / +1.64 |
| 0.7 | -2.24 ± 3.79 | 0.20 ± 2.99 | -0.35 ± 0.60 | -0.29 ± 3.30 | -0.49 ± 2.75 | +0.60 / -0.76 / -6.55 |
| 0.5 | -0.66 ± 4.51 | -1.75 ± 2.95 | -1.99 ± 2.14 | -0.78 ± 1.58 | 0.97 ± 2.53 | -3.50 / +4.54 / -3.01 |

G90 improves native BA by **2.55±1.83 pp**, positive on all three seeds, and WholeCount by 2.53±1.71 pp. G70 and G50 do not show a matched static-budget native win. G90's −5.67 pp collapse-gap change combines a WholeCount gain with a **−3.14 pp Relative10 loss**; it is partial count accessibility, not preservation of all temporal information.

For replay, `delta_variation=Learned−Mean`, `delta_shuffle=Learned−mean(shuffles)`, and `delta_alignment=Learned−mean(shifts)`:

| Case | delta_variation_pp | delta_shuffle_pp | delta_alignment_pp |
| --- | --- | --- | --- |
| G50 | 1.20 ± 0.73 | 0.39 ± 1.37 | 0.70 ± 1.29 |
| G70 | 0.04 ± 0.78 | 0.22 ± 1.76 | -0.29 ± 1.89 |
| G90 | 0.00 ± 0.60 | -0.04 ± 0.26 | 0.02 ± 0.34 |
| GZ0 | -0.28 ± 0.48 | -0.34 ± 0.60 | -0.25 ± 0.24 |

G90 Learned−Mean averages **0.00±.60 pp**; its individual differences are −.595/0/+.595 pp. Learned−Shuffle and Learned−Shift are also near zero. Thus the G90 between-trained-model gain is supported, while functional necessity of its learned native timing is not. The schedules are not bitwise identical and this is not an equivalence proof. It is a negative outcome under the planned native-BA intervention test.

G50 has Learned−Mean +1.20±.73 pp, positive on all seeds, yet G50−S50 is −.66±4.51 pp. Temporal variation can matter to a particular trained checkpoint without outperforming the static training control. Replay probe changes additionally concern decodability, not a rescue of the native endpoint.

Absolute formal probes:

| Case | whole_count | fixed250_ordered | fixed250_shuffled | relative10_ordered | relative10_shuffled | R10−WC pp |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 55.46 ± 4.50 | 59.01 ± 1.66 | 50.11 ± 3.90 | 67.47 ± 1.35 | 50.46 ± 4.78 | 12.01 ± 4.38 |
| G50 | 55.02 ± 2.58 | 56.25 ± 1.04 | 52.22 ± 1.99 | 67.16 ± 6.25 | 50.81 ± 1.99 | 12.14 ± 3.69 |
| G70 | 55.07 ± 6.34 | 59.08 ± 3.07 | 49.91 ± 3.08 | 65.92 ± 2.86 | 49.49 ± 4.99 | 10.85 ± 6.57 |
| G90 | 57.35 ± 2.63 | 57.96 ± 4.72 | 51.18 ± 4.81 | 66.13 ± 2.55 | 52.46 ± 3.85 | 8.78 ± 1.42 |
| GZ0 | 54.83 ± 5.07 | 58.58 ± 4.99 | 47.45 ± 6.99 | 63.51 ± 1.67 | 46.81 ± 7.33 | 8.67 ± 4.31 |
| S50 | 56.76 ± 4.14 | 58.24 ± 2.84 | 51.24 ± 3.74 | 67.94 ± 4.97 | 50.12 ± 2.87 | 11.17 ± 3.33 |
| S70 | 54.87 ± 4.08 | 59.43 ± 3.65 | 52.04 ± 2.12 | 66.21 ± 3.17 | 51.46 ± 3.07 | 11.34 ± 6.47 |
| S90 | 54.82 ± 3.23 | 59.52 ± 2.60 | 50.98 ± 3.58 | 69.27 ± 2.34 | 51.43 ± 4.85 | 14.45 ± 1.22 |

### 9.4 Exp16.3: diminishing episode reward harms prediction and does not suppress persistence

Reward-native BA and the same selected head evaluated with linear aggregation:

| Case | train_reward_ba | val_reward_ba | test_reward_ba | test_linear_same_head_ba |
| --- | --- | --- | --- | --- |
| LIN | 93.28 ± 0.29 | 58.64 ± 2.31 | 56.81 ± 2.98 | 56.81 ± 2.98 |
| CAP4 | 42.40 ± 16.05 | 31.75 ± 6.94 | 31.57 ± 5.43 | 30.72 ± 1.40 |
| CAP8 | 61.52 ± 19.59 | 40.34 ± 4.72 | 37.50 ± 5.61 | 41.38 ± 3.15 |
| CAP16 | 86.43 ± 1.14 | 50.18 ± 2.89 | 53.05 ± 4.65 | 52.72 ± 3.40 |
| CAP24 | 89.93 ± 1.76 | 52.74 ± 1.96 | 55.03 ± 3.17 | 53.75 ± 1.13 |
| SUB8 | 70.15 ± 4.91 | 42.37 ± 5.97 | 40.78 ± 2.71 | 43.03 ± 2.62 |
| EP1 | 35.11 ± 2.32 | 26.08 ± 3.09 | 18.39 ± 2.84 | 23.71 ± 1.64 |

All six non-LIN cases have lower mean reward-native train, validation and test BA than LIN. CAP24 is the least damaging cap, at 55.03% versus 56.81%; EP1 drops to 18.39%. Strong caps reduce training BA too, so the result includes optimization/representation loss, rather than only excessive overfitting or held-out-user transfer failure. Linear-same-head evaluation does not restore the LIN baseline.

Neuron-weighted selected-checkpoint test L2 activity:

| Case | L2 occupancy | Mean longest run (steps) |
| --- | --- | --- |
| CAP16 | 0.4019 ± 0.0088 | 24.21 ± 0.26 |
| CAP24 | 0.3881 ± 0.0089 | 22.91 ± 0.30 |
| CAP4 | 0.4113 ± 0.0517 | 26.33 ± 3.43 |
| CAP8 | 0.4126 ± 0.0390 | 25.84 ± 2.03 |
| EP1 | 0.4286 ± 0.0131 | 27.32 ± 1.52 |
| LIN | 0.3726 ± 0.0053 | 21.62 ± 0.28 |
| SUB8 | 0.4349 ± 0.0054 | 27.31 ± 0.07 |

The desired joint outcome—less persistence without losing classification—does not occur. LIN occupancy .3726 and mean longest run 21.62 steps rise under every diminishing reward case; EP1 has .4286 and 27.32 steps. Equal-τ averages in earlier summaries gave approximately 21.65 and 27.35; those are different weighting conventions, not new runs. Mean longest runs correspond to roughly 338 ms for LIN and 427 ms for EP1.

Refitted raw-spike L2 probes retain substantial information even when the reward-native head fails:

| Case | whole_count | fixed250_ordered | fixed250_shuffled | relative10_ordered | relative10_shuffled | R10−WC pp |
| --- | --- | --- | --- | --- | --- | --- |
| CAP16 | 51.91 ± 5.98 | 60.38 ± 3.77 | 50.47 ± 3.16 | 65.50 ± 4.13 | 49.06 ± 3.30 | 13.59 ± 4.09 |
| CAP24 | 54.88 ± 4.57 | 60.95 ± 4.05 | 51.19 ± 2.70 | 63.62 ± 2.69 | 49.10 ± 2.12 | 8.74 ± 2.22 |
| CAP4 | 47.64 ± 2.17 | 53.79 ± 2.57 | 44.90 ± 5.92 | 59.51 ± 2.15 | 40.27 ± 6.39 | 11.87 ± 3.15 |
| CAP8 | 49.69 ± 1.39 | 57.39 ± 3.06 | 47.90 ± 3.68 | 64.15 ± 0.96 | 42.60 ± 4.35 | 14.46 ± 2.33 |
| EP1 | 48.07 ± 3.39 | 55.33 ± 2.60 | 45.80 ± 2.77 | 56.46 ± 4.18 | 39.54 ± 1.42 | 8.39 ± 0.94 |
| LIN | 56.50 ± 3.20 | 61.45 ± 2.17 | 53.29 ± 2.63 | 65.78 ± 4.42 | 51.88 ± 3.04 | 9.28 ± 5.31 |
| SUB8 | 49.24 ± 0.66 | 59.28 ± 1.98 | 49.27 ± 1.52 | 63.64 ± 3.55 | 44.77 ± 0.91 | 14.39 ± 2.89 |

For example EP1 native is 18.39%, while its WholeCount probe is 48.07% and Relative10 probe 56.46%. This separates poor reward-native usage/learning from complete loss of decodable class information. It does not make the trained EP1 classifier successful.

## 10. Training / Mechanistic Diagnostics

### 10.1 Prefix credit reaches the gate but mostly reinforces WCCE direction

Main Exp16 stores empty Prefix gradient rows because λP=0. This is an inactive auxiliary treatment, not a measured vanishing-gradient failure. Exp16.1 computes weighted Prefix/WCCE norm ratios and cosines on fixed diagnostic batches. At epoch 20:

| Case | Parameter | Epoch 20 λ /  / ∇P /  / / /  / ∇W /  /  | Epoch 20 gradient cosine |
| --- | --- | --- | --- |
| GPZH_lp0p01 | G_h | 0.0086 ± 0.0010 | 0.9501 ± 0.0149 |
| GPZH_lp0p01 | G_z | 0.0115 ± 0.0028 | 0.7049 ± 0.2146 |
| GPZH_lp0p03 | G_h | 0.0275 ± 0.0044 | 0.9533 ± 0.0144 |
| GPZH_lp0p03 | G_z | 0.0355 ± 0.0110 | 0.7811 ± 0.1707 |
| GPZH_lp0p05 | G_h | 0.0453 ± 0.0081 | 0.9533 ± 0.0174 |
| GPZH_lp0p05 | G_z | 0.0627 ± 0.0204 | 0.7533 ± 0.1798 |
| GPZ_lp0p01 | G_z | 0.0117 ± 0.0016 | 0.9455 ± 0.0330 |
| GPZ_lp0p03 | G_z | 0.0356 ± 0.0069 | 0.9407 ± 0.0328 |
| GPZ_lp0p05 | G_z | 0.0591 ± 0.0118 | 0.9452 ± 0.0230 |

For z-only Gz, weighted ratios scale from about.012 to.059 and cosines are roughly.94–.95. ZH history gradients are similarly positively aligned. Prefix does reach gate parameters in the implemented positive-λ experiment; the evidence does not support “no gate gradient.” It supplies small, generally aligned credit at this diagnostic epoch rather than a strong competing incentive for suppression. These local batches do not prove a global optimizer cause. First-step relative update norms for zero-initialized gate vectors have an almost-zero denominator and can be enormous; they must not be read as physically enormous parameter steps.

Main Exp16's supplementary history diagnostics are:

| Case | L2 trajectory retrieval BA % | Final L2 full-history CV BA % | Final L2 full-history test BA % |
| --- | --- | --- | --- |
| C0 | 65.86 ± 4.62 | 49.65 ± 1.39 | 48.14 ± 3.43 |
| G0 | 65.73 ± 3.25 | 51.73 ± 3.46 | 51.28 ± 2.66 |
| GP_J | 66.66 ± 5.14 | 52.10 ± 0.27 | 55.54 ± 1.64 |
| P0 | 66.17 ± 4.83 | 50.25 ± 3.74 | 50.30 ± 6.74 |

Improved full-history diagnostics in GP_J are descriptive λ=0 model differences. Retrieval and final-state readouts are neither the native accumulator nor evidence of a Prefix-selected memory policy. Exp16.1's corresponding supplementary outputs are included in Appendix B.

### 10.2 A gate budget is not an effective-write or firing budget

Selected Exp16.2 test diagnostics use u=W2sᴸ¹ and gu as newly written drive. Mean gate and within-sequence variation are sequence-weighted; current/state norm averages and firing rates follow their recorded diagnostic contract:

| Case | mean_sample_gate | mean_within_sample_gate_std | w2_frobenius | mean_u_l2 | mean_gu_l2 | l2_firing_rate | mean_l2_syn_l2 | mean_l2_mem_l2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G50 | 0.5174 ± 0.0009 | 0.0716 ± 0.0081 | 15.6314 ± 1.4271 | 2.6166 ± 0.0931 | 1.3854 ± 0.0510 | 0.3643 ± 0.0121 | 6.1198 ± 0.1806 | 7.4013 ± 0.3770 |
| G70 | 0.7107 ± 0.0029 | 0.0544 ± 0.0048 | 14.0187 ± 1.2076 | 2.2030 ± 0.0555 | 1.5849 ± 0.0383 | 0.3609 ± 0.0108 | 6.7071 ± 0.1614 | 8.6588 ± 0.5012 |
| G90 | 0.9055 ± 0.0016 | 0.0274 ± 0.0034 | 14.9830 ± 0.5565 | 2.0599 ± 0.0730 | 1.8747 ± 0.0636 | 0.3727 ± 0.0048 | 7.5574 ± 0.2906 | 10.2556 ± 0.6484 |
| GZ0 | 0.9695 ± 0.0115 | 0.0229 ± 0.0007 | 12.6083 ± 3.2981 | 1.8541 ± 0.2382 | 1.8271 ± 0.2516 | 0.3566 ± 0.0353 | 7.5409 ± 0.6083 | 10.4194 ± 0.8409 |
| S50 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 16.3320 ± 0.5224 | 2.7222 ± 0.0740 | 1.3611 ± 0.0370 | 0.3694 ± 0.0017 | 5.8484 ± 0.0900 | 6.7772 ± 0.1964 |
| S70 | 0.7000 ± 0.0000 | 0.0000 ± 0.0000 | 15.5915 ± 0.5743 | 2.3508 ± 0.0292 | 1.6455 ± 0.0204 | 0.3722 ± 0.0048 | 6.6797 ± 0.1170 | 8.4402 ± 0.3078 |
| S90 | 0.9000 ± 0.0000 | 0.0000 ± 0.0000 | 14.4737 ± 1.2742 | 2.0366 ± 0.1105 | 1.8329 ± 0.0995 | 0.3679 ± 0.0074 | 7.3551 ± 0.3151 | 9.9035 ± 0.6152 |

Lowering staticρ from.9 to.5 increases W2 Frobenius norm from 14.47 to 16.33 and raw input norm from 2.04 to 2.72. Effective input norm falls from 1.83 to 1.36, but L2 firing stays near.368–.372 instead of tracking the gate reduction proportionally. Dynamic G50 has larger raw drive than G90, yet both maintain dense L2 firing. Therefore matching mean g is a valid gate-budget control, not evidence of equal memory capacity or equal transmitted magnitude. Current and membrane norms do change; compensation is substantial but not exact invariance of every measured quantity.

Exp16.3's reused Exp16.2 scale audit includes absolute means/RMS and W2 row norms to avoid interpreting a small **signed** mean as small input energy:

| Case | τ group | w2_row_norm | raw_mean_abs | raw_rms | actual_rms | actual_rms_over_one_minus_alpha |
| --- | --- | --- | --- | --- | --- | --- |
| G50 | long | 1.3415 ± 0.1169 | 0.1469 ± 0.0073 | 0.2155 ± 0.0112 | 0.1251 ± 0.0092 | 2.0020 ± 0.1468 |
| G50 | medium | 1.3790 ± 0.1431 | 0.1794 ± 0.0120 | 0.2564 ± 0.0167 | 0.1454 ± 0.0109 | 1.1630 ± 0.0874 |
| G50 | short | 1.4120 ± 0.1218 | 0.2231 ± 0.0049 | 0.3113 ± 0.0039 | 0.1720 ± 0.0024 | 0.6880 ± 0.0095 |
| G90 | long | 1.3087 ± 0.0497 | 0.1228 ± 0.0067 | 0.1849 ± 0.0107 | 0.1705 ± 0.0099 | 2.7283 ± 0.1578 |
| G90 | medium | 1.3172 ± 0.0452 | 0.1435 ± 0.0094 | 0.2112 ± 0.0124 | 0.1943 ± 0.0107 | 1.5544 ± 0.0855 |
| G90 | short | 1.3360 ± 0.0530 | 0.1679 ± 0.0020 | 0.2418 ± 0.0040 | 0.2217 ± 0.0039 | 0.8867 ± 0.0156 |
| S50 | long | 1.3982 ± 0.0501 | 0.1522 ± 0.0076 | 0.2253 ± 0.0120 | 0.1126 ± 0.0060 | 1.8022 ± 0.0960 |
| S50 | medium | 1.4408 ± 0.0344 | 0.1861 ± 0.0066 | 0.2681 ± 0.0080 | 0.1340 ± 0.0040 | 1.0724 ± 0.0320 |
| S50 | short | 1.4774 ± 0.0656 | 0.2339 ± 0.0084 | 0.3266 ± 0.0141 | 0.1633 ± 0.0070 | 0.6533 ± 0.0281 |
| S90 | long | 1.2599 ± 0.1256 | 0.1225 ± 0.0079 | 0.1841 ± 0.0132 | 0.1657 ± 0.0118 | 2.6516 ± 0.1894 |
| S90 | medium | 1.2756 ± 0.1002 | 0.1427 ± 0.0102 | 0.2100 ± 0.0141 | 0.1890 ± 0.0127 | 1.5119 ± 0.1013 |
| S90 | short | 1.2907 ± 0.1121 | 0.1648 ± 0.0072 | 0.2376 ± 0.0110 | 0.2139 ± 0.0099 | 0.8554 ± 0.0398 |

For S90, actual RMS is .2139/.1890/.1657 in short/medium/long groups; dividing by 1−α gives .8554/1.5119/2.6516. Learned signed/DC balance partly offsets τ-dependent gain, while RMS and absolute drive remain nonzero. Long-τ firing cannot be explained solely by an exactly fixed positive DC input or by signed cancellation. These per-neuron statistics are seed summaries, not extra n.

The static-gate scale degeneracy follows algebraically: ρW2 can be absorbed into W2. Likewise normalized \(I_t=\alpha I_{t-1}+(1-\alpha)W'x_t\) reproduces unnormalized drive for \(W'=W/(1-\alpha)\), row-wise. This is a representational reparameterization, not proof that optimizer trajectories, constraints or regularization are equivalent. The user at 15:24:09 UTC therefore downgraded simple normalization as a clean structural solution and required scale-aware diagnostics.

### 10.3 Persistence comes from state and drive; high firing has mixed utility

The 16.2 activity audit finds L1 test occupancy about.059–.080, whereas L2 is about.357–.373 with average longest runs 20–22 steps. A neuron can have a long episode without firing for the entire sequence. Approximately 12–15% of L2 neurons have mean occupancy>.5; “all L2 neurons are always firing” is inaccurate.

L2 counterfactual replay fixes the learned L1-derived drive and gate schedule, possible because these gates are z-only. Original, hard-reset, normalized-drive, and normalized+hard-reset conditions rerun L2 from zero. Selected results across all cases:

| Case | Condition | Test occupancy | Mean longest run |
| --- | --- | --- | --- |
| C0 | hard_reset | 0.3530 ± 0.0060 | 19.04 ± 0.15 |
| C0 | normalized_hard_reset | 0.0144 ± 0.0015 | 0.35 ± 0.03 |
| C0 | normalized_synapse | 0.0154 ± 0.0017 | 0.43 ± 0.04 |
| C0 | original | 0.3726 ± 0.0064 | 21.63 ± 0.18 |
| G50 | hard_reset | 0.3446 ± 0.0109 | 17.92 ± 0.42 |
| G50 | normalized_hard_reset | 0.0068 ± 0.0002 | 0.19 ± 0.01 |
| G50 | normalized_synapse | 0.0073 ± 0.0002 | 0.22 ± 0.01 |
| G50 | original | 0.3643 ± 0.0121 | 20.22 ± 0.55 |
| G70 | hard_reset | 0.3418 ± 0.0098 | 18.28 ± 0.16 |
| G70 | normalized_hard_reset | 0.0089 ± 0.0003 | 0.24 ± 0.01 |
| G70 | normalized_synapse | 0.0095 ± 0.0004 | 0.28 ± 0.02 |
| G70 | original | 0.3609 ± 0.0108 | 20.69 ± 0.19 |
| G90 | hard_reset | 0.3531 ± 0.0042 | 19.02 ± 0.19 |
| G90 | normalized_hard_reset | 0.0133 ± 0.0012 | 0.35 ± 0.05 |
| G90 | normalized_synapse | 0.0142 ± 0.0014 | 0.42 ± 0.06 |
| G90 | original | 0.3727 ± 0.0048 | 21.60 ± 0.14 |
| GZ0 | hard_reset | 0.3384 ± 0.0330 | 18.60 ± 1.33 |
| GZ0 | normalized_hard_reset | 0.0134 ± 0.0033 | 0.33 ± 0.10 |
| GZ0 | normalized_synapse | 0.0143 ± 0.0036 | 0.40 ± 0.13 |
| GZ0 | original | 0.3566 ± 0.0353 | 20.99 ± 1.61 |
| S50 | hard_reset | 0.3490 ± 0.0014 | 17.91 ± 0.27 |
| S50 | normalized_hard_reset | 0.0053 ± 0.0005 | 0.13 ± 0.01 |
| S50 | normalized_synapse | 0.0056 ± 0.0006 | 0.15 ± 0.01 |
| S50 | original | 0.3694 ± 0.0017 | 20.30 ± 0.31 |
| S70 | hard_reset | 0.3522 ± 0.0043 | 18.61 ± 0.13 |
| S70 | normalized_hard_reset | 0.0092 ± 0.0009 | 0.23 ± 0.02 |
| S70 | normalized_synapse | 0.0098 ± 0.0010 | 0.27 ± 0.02 |
| S70 | original | 0.3722 ± 0.0048 | 21.17 ± 0.12 |
| S90 | hard_reset | 0.3484 ± 0.0069 | 18.75 ± 0.35 |
| S90 | normalized_hard_reset | 0.0117 ± 0.0006 | 0.29 ± 0.02 |
| S90 | normalized_synapse | 0.0124 ± 0.0006 | 0.34 ± 0.02 |
| S90 | original | 0.3679 ± 0.0074 | 21.30 ± 0.39 |

For C0, hard reset reduces occupancy .3726→.3530 and mean longest run 21.63→19.04; fixed-weight normalized drive reduces them to.0154 and.43. This supports a strong drive-scale/current-accumulation contribution under existing weights, with a smaller reset contribution. It does **not** show that a retrained normalized model must stay sparse or classify well; normalization can be absorbed by trainable weights.

The 1000-ms free-tail diagnostic distinguishes zero **sensor input** while both layers continue from exactly zero **L2 external drive** while only L2 state decays. Across cases/seeds, long-τ tail occupancy averages.1844 for network-zero-input versus.1067 for L2-zero-drive; corresponding fractions with a spike at/after 250 ms are.5162 versus.2831. Thus residual upstream drive prolongs activity, and L2's own retained state can sustain it too. These summaries pool cases descriptively; they are not an n=24 causal treatment test.

The persistent-neuron removal controls are especially important. **Positive entries below are BA losses, full−removed**, from a fitted WholeCount probe:

| Case | Method | Test BA loss (full−removed) pp | Train BA loss (full−removed) pp |
| --- | --- | --- | --- |
| C0 | frozen_mean_replacement | 5.09 ± 5.14 | 5.60 ± 0.81 |
| C0 | retrained_without_neurons | 2.24 ± 2.42 | -1.55 ± 5.25 |
| G50 | frozen_mean_replacement | 4.25 ± 3.07 | 10.49 ± 3.12 |
| G50 | retrained_without_neurons | 1.10 ± 1.56 | 1.04 ± 5.11 |
| G70 | frozen_mean_replacement | 4.30 ± 5.04 | 13.50 ± 9.97 |
| G70 | retrained_without_neurons | 1.00 ± 1.54 | 7.70 ± 7.16 |
| G90 | frozen_mean_replacement | 3.22 ± 2.52 | 6.99 ± 3.52 |
| G90 | retrained_without_neurons | 1.42 ± 2.64 | 1.08 ± 5.60 |
| GZ0 | frozen_mean_replacement | 7.63 ± 1.61 | 9.37 ± 3.91 |
| GZ0 | retrained_without_neurons | 1.46 ± 1.74 | 0.84 ± 3.13 |
| S50 | frozen_mean_replacement | 4.08 ± 0.51 | 6.27 ± 4.05 |
| S50 | retrained_without_neurons | 5.44 ± 3.85 | -1.98 ± 5.01 |
| S70 | frozen_mean_replacement | 2.09 ± 2.49 | 8.09 ± 2.68 |
| S70 | retrained_without_neurons | -3.08 ± 5.43 | 0.89 ± 3.17 |
| S90 | frozen_mean_replacement | 7.05 ± 2.52 | 8.56 ± 4.72 |
| S90 | retrained_without_neurons | -2.32 ± 1.47 | 3.39 ± 3.21 |

Frozen replacement of the top 39 neurons loses 2.09–7.63 pp on average across cases. Refit deletion improves S90 by 2.32 pp and S70 by 3.08 pp, but harms S50 by 5.44 pp; other means mostly recover part of the frozen loss. This supports dependence and redundancy that vary by case. It does not show every persistent neuron is nuisance or that a retrained SNN can recover, because only the probe was refitted.

| Case | Train corr(occupancy,class η²) | Train corr(occupancy,user η²) | High32 train ablation ΔCE | High32 test ablation ΔCE |
| --- | --- | --- | --- | --- |
| C0 | -0.452 ± 0.030 | 0.243 ± 0.050 | 0.0030 ± 0.0001 | -0.0025 ± 0.0020 |
| G50 | -0.458 ± 0.153 | 0.191 ± 0.086 | 0.0148 ± 0.0115 | 0.0025 ± 0.0147 |
| G70 | -0.453 ± 0.066 | 0.194 ± 0.109 | 0.0156 ± 0.0220 | 0.0184 ± 0.0378 |
| G90 | -0.486 ± 0.022 | 0.225 ± 0.072 | 0.0063 ± 0.0070 | -0.0008 ± 0.0013 |
| GZ0 | -0.321 ± 0.240 | 0.163 ± 0.135 | 0.0049 ± 0.0023 | -0.0013 ± 0.0010 |
| S50 | -0.517 ± 0.022 | 0.255 ± 0.033 | 0.0034 ± 0.0019 | -0.0015 ± 0.0003 |
| S70 | -0.457 ± 0.017 | 0.241 ± 0.060 | 0.0032 ± 0.0015 | -0.0044 ± 0.0012 |
| S90 | -0.413 ± 0.061 | 0.180 ± 0.098 | 0.0048 ± 0.0023 | -0.0015 ± 0.0029 |

Training occupancy is negatively associated with class η² and positively associated with user η² in these eight cases. High32 single-neuron ablation ΔCE is positive on train but near zero/negative on test in several cases; G70 remains positive on test and G50 is weakly positive. Single-feature CE utility can be small or negative while joint 39-feature removal still harms BA: interactions, redundancy and the retained decoder geometry matter. The table uses a train-ranked fixed high32 group; it is distinct from the source `case_summary.csv`'s test-ranked highest occupancy quartile.

Duration-only affine BA is 11.31%, while rate-normalized and count probes retain roughly 54–59%. Duration is a possible correlate, not a sufficient explanation of all persistent class information. Independently refitted posthoc WholeCount baselines differ from formal WholeCount for G50 (54.46 versus 55.02%) and G70 (54.82 versus 55.07%); other case means agree to reported precision. The report preserves within-analysis ablation contrasts and does not replace formal probe numbers with these refits. The exact origin of those solver/refit differences was not isolated from retained outputs.

### 10.4 Persistent tails carry information; capped native training is not a posthoc probe

Core O0/O1 selected-checkpoint L2 activity:

| Objective | τ group | Occupancy | Mean longest run |
| --- | --- | --- | --- |
| O0 | long | 0.3657 ± 0.0216 | 24.38 ± 1.60 |
| O0 | medium | 0.3712 ± 0.0124 | 21.78 ± 0.39 |
| O0 | short | 0.3607 ± 0.0170 | 17.32 ± 1.38 |
| O1 | long | 0.3329 ± 0.0097 | 22.20 ± 0.52 |
| O1 | medium | 0.2594 ± 0.0123 | 14.47 ± 0.63 |
| O1 | short | 0.1777 ± 0.0116 | 7.69 ± 0.60 |

O1/TSCE has less activity and shorter runs, especially in short/medium groups. This is descriptive evidence that the objective changes the regime, not isolation of repeated-count causality. Those historical models differ from fresh Exp16.3 LIN.

Run-feature probes on **frozen Core spikes**, no-bias decoder:

| Objective | Feature | Test BA % |
| --- | --- | --- |
| O0 | cap1 | 39.00 ± 3.38 |
| O0 | cap16 | 53.15 ± 4.03 |
| O0 | cap2 | 39.60 ± 6.40 |
| O0 | cap24 | 57.96 ± 3.01 |
| O0 | cap4 | 43.96 ± 5.05 |
| O0 | cap8 | 49.35 ± 3.52 |
| O0 | linear | 57.74 ± 5.16 |
| O0 | sub8 | 50.69 ± 5.37 |
| O1 | cap1 | 39.41 ± 1.94 |
| O1 | cap16 | 53.91 ± 4.19 |
| O1 | cap2 | 41.20 ± 1.35 |
| O1 | cap24 | 56.76 ± 4.52 |
| O1 | cap4 | 48.05 ± 1.61 |
| O1 | cap8 | 50.42 ± 2.56 |
| O1 | linear | 58.64 ± 3.08 |
| O1 | sub8 | 52.37 ± 2.25 |

For O0, cap24 is 57.96% versus linear 57.74%; cap16 is 53.15%, cap8 is 49.35%, cap1 is 39.00%. Episodes alone retain information but lose substantial count-accessible evidence. A cap24 probe retaining BA suggests very long repeats can be redundant for this fitted readout. It is not a general statement that every late spike or internal state is unnecessary.

The early/tail decomposition defines early=min(run length,K), tail=linear−early, and concatenates both as 256 features. All are fitted with the same validation-selected probe contract:

| Objective | K | Early BA % | Tail BA % | Early+tail BA % | Increment pp |
| --- | --- | --- | --- | --- | --- |
| O0 | 4 | 43.96 ± 5.05 | 59.53 ± 2.33 | 58.29 ± 3.76 | 14.32 ± 4.20 |
| O0 | 8 | 49.35 ± 3.52 | 57.37 ± 1.99 | 57.01 ± 4.87 | 7.66 ± 4.25 |
| O0 | 16 | 53.15 ± 4.03 | 50.62 ± 2.65 | 60.27 ± 3.96 | 7.12 ± 0.64 |
| O1 | 4 | 48.05 ± 1.61 | 54.28 ± 0.75 | 57.57 ± 3.09 | 9.52 ± 1.82 |
| O1 | 8 | 50.42 ± 2.56 | 54.28 ± 1.40 | 56.98 ± 1.32 | 6.57 ± 1.24 |
| O1 | 16 | 53.91 ± 4.19 | 47.22 ± 7.42 | 58.25 ± 2.24 | 4.34 ± 2.63 |

O0 early+tail gains **14.32 pp atK4, 7.66 pp atK8 and 7.12 pp atK16** over early alone. Tail after 16 spikes is therefore still complementary in this diagnostic. There was **no K24 early+tail decomposition** in the retained grid, and concatenation changes feature dimension. These facts limit the earlier conversational shorthand “late tail mostly redundant.” The precise supported statement is that cap24 loses little mean probe BA here, while earlier sustained tail remains informative. CAP24 trained-native 55.03% and frozen Core cap24-probe 57.96% are different model/decoder populations and must not be conflated.

## 11. Negative and Null Results

| Stage | Observed negative/null result | Valid implication |
| --- | --- | --- |
| Main Phase0A | CE selector preserves validation BA on 1/3 seeds | Keep BA-first under the predefined safeguard |
| Main Phase0B | No positive λ has any eligible point | Legal λP=0; positive Prefix interaction not executed |
| Main Phase1.5 | Zero attribution checkpoints | Skip decision, not a failed attribution result |
| Main gates | Nearly open; mean/shuffle effects near zero | No convincing selective timing dependence at these selected weights |
| Exp16.1 | All mean validation interactions negative; test .03 z-only seed 11-driven | Tested Prefix variants do not establish useful gate recruitment |
| Exp16.2 G70/G50 | No mean native gain over matched static controls | Dynamic budgets do not uniformly help |
| Exp16.2 G90 replay | Learned−Mean 0.00 pp, shuffle/shift near zero | Between-model training gain does not establish native timing necessity |
| Exp16.2 activity | Dense firing survives reduced mean gate | Gate-budget compliance does not imply an activity/capacity budget |
| Persistent removal | Frozen loss, mixed refit effects | High firing is neither wholly dispensable nor uniformly useful |
| Exp16.3 | Every diminishing reward lowers mean BA and increases persistence | These implemented rewards fail the intended joint mechanism test |

These scientific outcomes are separate from the cancelled single-node submission, environment startup failure, parser correction, and completed skip tasks. They remain part of the experiment record rather than being removed by selecting a favorable case or changing thresholds.

## 12. Interpretation

The main Exp16 result is principally a **calibration negative**: the gate×positive-Prefix hypothesis was not actually tested after λP=0. Exp16.1 supplies that missing direct test and finds no stable validation interaction, despite measurable Prefix gradients. The evidence favors an available nearly-open solution over the tested expectation that Prefix creates a useful selective write policy.

Exp16.2 is more informative than an undifferentiated “gating failed.” Atρ=.9, dynamic constrained training improves native and count-probe BA relative to static constrained training. Its recorded temporal schedule is not necessary for mean native performance in the planned replay tests. A plausible explanation is training-time representation/optimization change or effective-gain adaptation. This is an inference from the between-model and within-model contrast; no unique optimizer cause is established. G50 additionally shows that variation dependence and a static-control win are logically distinct.

Persistent activity has two roles that cannot be collapsed into one rate number: it can provide state/history and class evidence, while some high-occupancy dimensions are more user-associated or redundant for transfer. Fixed-weight normalization strongly reduces activity, but reparameterization and observed compensation prevent treating it as a guaranteed learned solution. Probe deletion shows some recoverable dependence, not a universal nuisance channel.

Exp16.3 rejects the **tested joint prediction** that diminishing episode rewards would reduce persistence while preserving classification. Its caps also reduce direct continuation credit and total feature magnitude, so diminished gradients/changed readout learning are available explanations. The result does not prove linear WCCE causes no persistence, nor that all alternative repeat-aware objectives must fail. What it does show is that removing repetition reward alone, in this surrogate-gradient contract, damages useful classification without achieving the intended firing reduction.

The coherent next question is how to preserve useful internal context while controlling which outgoing messages support transferable evidence. The evidence does not justify equating sparse spikes with better memory or substituting a fitted high-dimensional probe for native success.

## 13. Bugs, Corrections, and Reruns

| Revision / execution event | Direct evidence | Effect on final analysis |
| --- | --- | --- |
| Main implementation, [9ffcbd83](https://github.com/hellowPluto78700/writingRing/commit/9ffcbd83e891f1e56b0d053a6cabe767a34eae6c), Oct 1 02:53 UTC | Phase0A/B, suppressive gate, conditional routes and validation decisions | Main protocol, including legal λ0, retained |
| Exp16.1 initial [40e32e49](https://github.com/hellowPluto78700/writingRing/commit/40e32e4976f789f68afc63d1c61b9e46513d0c5d), 04:15 UTC | Single-node implementation; cancelled 65106790 | No numerical result from original allocation |
| [90ddb746](https://github.com/hellowPluto78700/writingRing/commit/90ddb746f09d40a1985f7f84d884397866517500), 04:32 UTC | Deletes single-node script/assertion; adds 1-CPU arrays; manifest reports host set | Completed results are multi-node; stale README title is not execution authority |
| Exp16.2 parser [2ba02a01](https://github.com/hellowPluto78700/writingRing/commit/2ba02a01145a0dc5389cace2d59825db228ddc06), 06:36 UTC | Splits two parser statements accidentally placed on one line | Pre-run syntax correction; separate from scheduler exit 127 |
| Failed prepare 65111341, 06:45 UTC | Slurm FAILED 127:0; downstream cancellations | Entire failed pipeline excluded; no invented calibration results |
| Environment [3b1828e4](https://github.com/hellowPluto78700/writingRing/commit/3b1828e4e11b2d35ab965bc5c640d2faa1f3009c), 06:47 UTC | Adds `source /etc/profile`, environment/thread setup to prepare script | Rerun 65111359–64 completes; recovered prior assistant diagnosis says `module: command not found` |
| Persistent utility [0ed0df17](https://github.com/hellowPluto78700/writingRing/commit/0ed0df1716f62025c08513a443f8dfdf919061e5), 14:36 UTC | New artifact-only analysis | Posthoc diagnostic, no extra SNN training |
| Sustained replay [286fc231](https://github.com/hellowPluto78700/writingRing/commit/286fc2313d06a8abbe5791d3c060abc7a6f931a4), 14:59 UTC; [fc2c3f94](https://github.com/hellowPluto78700/writingRing/commit/fc2c3f947f591bde522d79177828cd2ae98bc964), 15:00 UTC | Fixed-drive dynamics analysis; τ-group alignment hardened | Final aligned per-neuron/τ tables used |
| Actual 16.3 [04214659](https://github.com/hellowPluto78700/writingRing/commit/042146593dc8dbab77982c36fcbd8ffdb3fe51df), 15:36 UTC | Seven run-reward cases and diagnostics | Canonical executed 16.3, not CF90 |
| [50e073df](https://github.com/hellowPluto78700/writingRing/commit/50e073df06aef7b8ee4c5c0040bef774e7abca31), 15:38 UTC | Removes a redundant reference-model construction from shared-init preparation | Code cleanup before successful prepare; no observed failed Slurm 16.3 run attributed to it |
|16.3 aggregate [7eca349a](https://github.com/hellowPluto78700/writingRing/commit/7eca349aa2bff998ab14581cb24e1528525aa6fa), cleanup [544be6d4](https://github.com/hellowPluto78700/writingRing/commit/544be6d42331cabbd7943d68b4bb1e4054fcda85), identifiers [0bacdd48](https://github.com/hellowPluto78700/writingRing/commit/0bacdd48e1650b0962dbf517f1542879aa790f00), 15:42–45 UTC | Adds summaries; avoids averaging seed/neuron identifiers | Report recomputes seed statistics from per-seed rows; numeric ID means have no scientific meaning |

The exact original failed-job error line was not independently reread: `scontrol` no longer retained the job, and a bounded check of the known working directory and script directory found no matching log. Accounting proves failure 127; the dated prior diagnosis and environment patch support the Modules startup explanation. This is recorded as a provenance gap, rather than fabricated stdout.

The Exp16.2 cached `adaptive_gating_gain_mean.csv` still contains a numeric seed average 23.6667. This report ignores that field and reconstructs mean/SD from seeds 11/23/37. Main λ0 duplicate divergence and small G50/G70 independent probe-refit differences are retained, not silently replaced by presumed identical controls. No thresholds, checkpoint selections, training code or Unity files were changed for this report.

## 14. Relationship to Previous Experiments

| Earlier finding | Exp16 response | What the family resolves |
| --- | --- | --- |
| Exp13: more available history need not improve transferable consolidation | Test selective writing and native/probe alignment | History access and native utility remain distinct |
| Exp14/14.1: Prefix can favor early discriminability without useful accumulation; no eligible corrected Prefix | Main calibration safeguards, then direct gated factorial grid | Main calibration rejects Prefix;16.1 directly fails the tested recruitment prediction |
| Exp15: contextual g/.9 can amplify and joint gains need continuation null | True suppressive g, fresh local C0/P/G controls | Gate amplification removed; open-policy shortcut persists |
| Exp15.1: reducing width did not establish robust selective-gate transfer | Keep H128 fixed while changing objective/controller constraints | Exp16 is not a hidden width/depth study |
| Core O0/WCCE versus O1/TSCE | Reuse for descriptive activity/tail diagnostics only | Formal 16.3 preserves sequence-level label contract |

Historical Core O0 native 57.93% and O1 native 53.72%, the preceding Exp15 continuation controls, main Exp16 C0 55.83%, Exp16.1 C0 56.21%, Exp16.2 C0 55.93%, and fresh 16.3 LIN 56.81% refer to different training/checkpoint populations. They are contextual references, not pooled baselines. Fresh 16.3 LIN's equivalence to the **WCCE formula** does not make its selected result identical to historical Core O0.

## 15. How This Led to the Next Experiment

The chronology matters because “Exp16.3” was reused for a changed question. After 16.2's timing-null result, the user at 14:14:38 UTC preferred sampled counterfactual write utility atρ=.9. The 14:15:28 UTC proposal included C0/S90/G90, CF90, a Relative10 utility positive control, and an oracle diagnostic. Its question was whether useful write timing exists and can be learned. It was a proposal, not the subsequently executed canonical 16.3.

Raster/persistent-activity analysis then exposed a prior issue: the signal being controlled could be useful retained context and repeated class evidence, rather than uniformly harmful writes. At 15:24:09 UTC the user required scale-aware RMS/absolute-drive/weight diagnostics and a descriptive WCCE/TSCE comparison, noting that fixed normalization/gate factors can be absorbed into W. At 15:29:35 UTC the user specified the **actual seven run-reward cases** and preserved whole-character CE. The implemented 15:36 UTC code and successful 15:56–16:19 UTC pipeline match that decision.

After the reward results, the user at 21:30:47 UTC recognized that persistent spikes contain classification information; at 22:04:47 UTC noted frozen ablation loss and partial refit recovery; at 22:35:14 UTC emphasized that caps did not suppress persistence. The proposed 16.4 high/low/random native-head and probe analyses sought to distinguish dependency, unique information and transfer. It is not promoted here to an executed subexperiment.

Successor Exp17's persistent-pathway/epoch-trajectory study and Exp17.1's gradient-rescue study pursue when persistent versus lower-firing dimensions acquire class/user selectivity. Later routed-context/evidence and edge-level proposals separate retaining internal state from emitting evidence. Those motivations belong in this chronology, but their quantitative results are outside the present Exp16 evidence cohort. Exp16 does not itself establish a specific routed architecture or an exact gradient-starvation cause.

## 16. Final Conclusions

1. **Main Exp16 selected λP=0.** It supplies a selector/calibration negative and nearly-open gate controls, not a positive-Prefix interaction test. Conditional attribution was correctly skipped.
2. **Exp16.1 directly tests positive Prefix and finds no stable validation interaction or selective-gate recruitment.** Z-only is less saturated, but the tested gates remain almost open; Prefix credit is measurable.
3. **Exp16.2 G90 improves matched static-budget native BA by 2.55 pp across all three seeds.** Its learned schedule's mean native timing necessity is absent in Mean/Shuffle/Shift replay. Tighter budgets do not uniformly improve classification, and mean g does not define a state/activity budget.
4. **Persistent activity has useful, redundant and less-transferable components.** Frozen removal harms fitted readouts, refit effects vary, and tail beyond 16 spikes retains complementary information. Occupancy alone does not classify a neuron or spike as nuisance.
5. **Actual Exp16.3's diminishing episode rewards fail the intended joint test:** classification declines while selected-checkpoint persistence increases. The next defensible question concerns classification-aware communication that preserves useful context, rather than blanket firing suppression.

## Appendix A. Run Inventory

All formal rows below are included in the primary n=3 aggregates. Val/test BA uses each branch's selected native head/aggregator. `Selected / last epoch` distinguishes the evaluated checkpoint from the last optimizer epoch. Each checkpoint lives at its canonical root's `runs/<case>__seed<seed>/checkpoint.pt`. Checksums were read directly on Unity on 2026-10-02, without downloading or rewriting weights.

### A.1 Main Exp16 formal 12

| Case | Seed | Selected / last epoch | Val BA % | Test BA % | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- |
| C0 | 11 | 88 / 100 | 57.15 | 60.61 | 0e38edba66511734570b94c89b92cb561822717a52215a4503cd89e43a59f628 |
| C0 | 23 | 63 / 93 | 53.46 | 50.85 | 2e7ac1e2f0ed0f9a2d0f3768e6db7381efb2daac27d57ac3d5f8096166fb9a00 |
| C0 | 37 | 89 / 100 | 59.14 | 56.02 | 80860efe7b38df8b40ae20af16890b100577f60b7e06d1fbabf1236ee9f2fce4 |
| P0 | 11 | 91 / 100 | 55.26 | 61.75 | f675d333eea64e0f42348600f3d4ea50ef05b408bc163bfeb5a20ec7e28b9737 |
| P0 | 23 | 63 / 93 | 53.46 | 50.85 | 467a999211551455dd3d8f1fca4176a8a92a3ababa3f8e40fc48d304b6ec8639 |
| P0 | 37 | 97 / 100 | 60.30 | 56.22 | e64cb11ea2866e9760479379b521111a098f8dadfd27d9d2ecc00fe3aff10729 |
| G0 | 11 | 95 / 100 | 56.98 | 58.25 | 6515a555710b51f6ad3f09e7579a6aacc1503025c7cff4c1dd0d9cb498eaa02c |
| G0 | 23 | 87 / 100 | 60.96 | 55.79 | b6f2f33d774b33997788b19dd407b5e57d1dc5aa863391934f62214c87bd1c83 |
| G0 | 37 | 91 / 100 | 59.27 | 56.22 | fa870c5c5e23466e75fe3ff48682644be4221573df79508910c108d721c91c61 |
| GP_J | 11 | 97 / 100 | 59.99 | 62.61 | c24d7d3bba5b7d5d07c4ca3bff5b945f276d291175cd10889b3a60a3b1556af1 |
| GP_J | 23 | 87 / 100 | 60.96 | 55.79 | b9b894372739847dcf64993e02701d4f3b5c13804441d7bf983aafe6632a1dc4 |
| GP_J | 37 | 91 / 100 | 57.02 | 54.94 | c14ab572667d173f4c68ae98c5824f1477f5092ad53012f0bd2a3b77a867e263 |

λP=0 in all 12. Array `65103456` uses case-major C0/P0/G0/GP_J and seed 11/23/37: task index=3×case index+seed index. All completed. Phase1.5 has no selected checkpoint and is not listed as a trained case.

### A.2 Exp16.1 formal 36

| Case | Seed | Selected / last epoch | Val BA % | Test BA % | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- |
| C0 | 11 | 91 / 100 | 55.26 | 61.75 | 8cc00805e5cafcbc0e81ea67eb1e410808edc9af48fdc753f3b046940951fb81 |
| C0 | 23 | 63 / 93 | 53.46 | 50.85 | a731754e6da7c0f1235e25945fff849121d424c8c77f252b5932ea51f1535cd7 |
| C0 | 37 | 89 / 100 | 59.14 | 56.02 | 836b1320d6d557807d7fbbad1bffe790d967823a41b78a33f0d5cddcd2de34f0 |
| P_lp0p01 | 11 | 100 / 100 | 53.02 | 62.11 | 69e301a6e03303ce67a2ed6ce5dffd4940dd58769939737d9cda04051b5285de |
| P_lp0p01 | 23 | 68 / 98 | 58.12 | 52.25 | f16214e9cff17d1f077124d9b7c144ab998b94a8926d35f7a1b441518c1602c0 |
| P_lp0p01 | 37 | 91 / 100 | 58.06 | 59.30 | 7c21a1eb9c57d7ab79aff0113a4fd95f8cb5b019c59d4a377cb4ed1af0a2537b |
| P_lp0p03 | 11 | 94 / 100 | 52.19 | 57.84 | 2019d89b73e205d4a97ae0f856a91858a33e55e047024e4361a70f89eb61be4a |
| P_lp0p03 | 23 | 68 / 98 | 58.20 | 53.17 | be55f072dae11601dcf7b61f86b80886b4ec9154e8d396202a6a3b8d8dd2a25b |
| P_lp0p03 | 37 | 93 / 100 | 56.20 | 55.19 | 9b16c6dd229533ca52360673917b51694af2af8eaf1d99fb5094d10d49b1f2c6 |
| P_lp0p05 | 11 | 99 / 100 | 54.17 | 63.53 | d68047b1dc5603736aadaa422c32dc8d21c703aee5671dccdf0d1aecffc17768 |
| P_lp0p05 | 23 | 83 / 100 | 56.62 | 53.15 | 675c5cc50c0426a4733db4c47c324ed7bc1f9507a4ddf5f679f36753d0306989 |
| P_lp0p05 | 37 | 71 / 100 | 56.38 | 55.37 | aeba300e687789c955147647e21c0b31721548c535d9c2d38fccc58b5a1af5df |
| GZ0 | 11 | 97 / 100 | 56.72 | 60.22 | 4eaffcf9f5a488ae247585c98b507b8e5ced31c4852f4fbe26d7182548026c52 |
| GZ0 | 23 | 68 / 98 | 58.28 | 50.64 | 3c1a2e6c35366aadb1e3775f5bf635b879704c3dcbb27695e53dbc63584b7df2 |
| GZ0 | 37 | 95 / 100 | 57.22 | 56.21 | f37bb663612c718c93c9517b5139a71d62efd11e83a13e4910c95403cfd2d4d5 |
| GPZ_lp0p01 | 11 | 99 / 100 | 58.25 | 62.00 | 052c8693fee21e5f20130697b8b02a5e8c20e979384c86e177f239205c929d19 |
| GPZ_lp0p01 | 23 | 68 / 98 | 58.06 | 50.69 | 2107facb18b9851d224c2220f7ab7a4eab32d95683f36c31260a49cf6148783c |
| GPZ_lp0p01 | 37 | 87 / 100 | 55.64 | 56.62 | 082ca25ad12551f838c7d06a697c80a3fed2df67aaf03b3a3a0b6c7160a74a94 |
| GPZ_lp0p03 | 11 | 95 / 100 | 55.59 | 62.39 | e1b9867b10f4da0388b826528890866f2917588e1f7dbf63b81fd4c60a5a4596 |
| GPZ_lp0p03 | 23 | 86 / 100 | 57.92 | 51.30 | 50d139af58f8ffd688cea5ce183f2b37f81096756aa47f845f6cce83ec255940 |
| GPZ_lp0p03 | 37 | 95 / 100 | 55.07 | 53.67 | ae1a6929d55a5ceb86fc9b04a05d53e4860137d9a4e46a4f251d0f9a93de67a3 |
| GPZ_lp0p05 | 11 | 100 / 100 | 53.97 | 56.26 | 57d379d414d948afe4a069aa842d58e2575eae6d2141ee7fc8c946b9ac3634a1 |
| GPZ_lp0p05 | 23 | 90 / 100 | 56.56 | 54.00 | 116b5a3235b81c9661aa1e72a78f6473529e2e18cebb5b8f80dd24fd838f2c70 |
| GPZ_lp0p05 | 37 | 80 / 100 | 58.79 | 56.39 | 86b83ef53c8a5c5ccf2eeb0e086508b41e3807f5551158aac3ed497e9f46a7d8 |
| GZH0 | 11 | 100 / 100 | 55.89 | 60.42 | 2b03df2128764c866f824912584af1b36f13427d8ede23d1895d9ef476871548 |
| GZH0 | 23 | 95 / 100 | 58.64 | 54.41 | aa07fa485129f297959e6c840c7485bbdd013842ad8df5d9faa17c5e60671065 |
| GZH0 | 37 | 99 / 100 | 63.34 | 54.87 | 21f175b1471d5701d5d36ab38fdeed7f50ec8135b9d9685afcc7424227fbf8dd |
| GPZH_lp0p01 | 11 | 100 / 100 | 54.25 | 59.99 | 18adc8fee1936fcead97a4b75db6a0b18b834810209e1644f6086b6463a26aed |
| GPZH_lp0p01 | 23 | 100 / 100 | 60.48 | 52.98 | ac68032bd62e211e779b1da64f5aaa8ef99f541597f56551b9eed5252bb7f056 |
| GPZH_lp0p01 | 37 | 90 / 100 | 55.56 | 57.68 | 32a044a8253878a003b888e09ecd78151c42892d87363cb774a6da98fe14b52c |
| GPZH_lp0p03 | 11 | 100 / 100 | 54.40 | 60.11 | c733aaf518ac1edd7948ab8f2268aace1383dfd049eb94acb5748f5d711771c7 |
| GPZH_lp0p03 | 23 | 63 / 93 | 56.18 | 50.34 | daaecbc66afd4bca2e027fb65cf73afdeb86c47c62db0bd67319d5e9dfd686bc |
| GPZH_lp0p03 | 37 | 86 / 100 | 59.10 | 56.13 | fd1ef364b64dc379e2f53b0dfec0531b110ce630bd7df4fd2cfb2ac049c4ca92 |
| GPZH_lp0p05 | 11 | 93 / 100 | 55.33 | 61.37 | 8e6b1889157d70b47d2df79616b3da583557c9dbfeee359bd831d4ff234796cb |
| GPZH_lp0p05 | 23 | 81 / 100 | 58.85 | 51.47 | 023299b7e0063d2979c6fb8d36a765d89bf6c1cfcd200561d8d5f718a2760302 |
| GPZH_lp0p05 | 37 | 97 / 100 | 58.11 | 58.18 | e8020f0f82307e451348c6491e8f2a4289a61b702525bf21431117e2e63854d6 |

Array `65107099` is seed-major,12 cases per seed in the table order: task index=12×seed index+case index. Each selected checkpoint records its actual hostname; the manifest's ten-node set is cpu032/033/034/035/036/043/045/072, gypsum-gpu054/070. All36 completed; original single-node65106790 produced no row.

### A.3 Exp16.2 formal 24

| Case | Seed | Selected / last epoch | Val BA % | Test BA % | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- |
| C0 | 11 | 100 / 100 | 52.48 | 59.81 | 09d991848a4a6c30692267929258f8524eaace7efb22e14c35a3150883e562cf |
| C0 | 23 | 80 / 100 | 58.94 | 52.17 | 19487e03ba0c8de8cf9c07b7ebbfefef3ccb117d367cdd5fb0876b668c76bebf |
| C0 | 37 | 91 / 100 | 57.91 | 55.81 | 24a545a6cfeb7ec9e9956775e2f2cc88adcfa9ef3d416b3f40b21982b745f59a |
| GZ0 | 11 | 99 / 100 | 54.79 | 60.16 | 94844867d2b72352b0a1593e0f57a1c7ce47fc265ec5407b4c2319297cbaabdf |
| GZ0 | 23 | 75 / 100 | 56.00 | 52.72 | 388f9a5bd060b1722cae5ac4c73d0c906340b1509ad355685f4eec894c68cd20 |
| GZ0 | 37 | 37 / 67 | 53.17 | 44.35 | 7956432438b0d2721fca739a25107b03356dd298cbaae57f8b2432216f1fffc7 |
| S90 | 11 | 98 / 100 | 53.80 | 57.39 | ed7458bebe74796312cbfd7937d03388753bae53bee351683d2de831d609de25 |
| S90 | 23 | 72 / 100 | 57.69 | 53.95 | 1f414b692734232e2f1d98c246f2ea5cd31181a97f0f0676adb0db777d7001da |
| S90 | 37 | 98 / 100 | 58.28 | 54.92 | 1bc40749113aec4cb0591a4ac78aed8cc9f244e8d600e927aa8c07daa4af0a0b |
| G90 | 11 | 98 / 100 | 52.62 | 62.04 | 60a03565069a38a092ddf53bbb235b402c197ec8f950ea9299bd814cb172c906 |
| G90 | 23 | 86 / 100 | 59.54 | 55.29 | 24cf1f985f7996fcd00467fcb03a765140ab930ab51a4a2be33438a571c749d4 |
| G90 | 37 | 100 / 100 | 57.85 | 56.56 | 81c6ab5e8458113ad070ec8c80ac1c11b0ab8400682a4e5400593e21dbbb3745 |
| S70 | 11 | 96 / 100 | 55.72 | 60.58 | e51172d6577308af3de7644fc9ebb0d47c839c9b9ee38dd33fe077d401379d0c |
| S70 | 23 | 97 / 100 | 56.99 | 54.19 | 1e294352689aa2326ebc6794f66c53d22ff77200f2ace16bc7474e9414fe6fa3 |
| S70 | 37 | 94 / 100 | 56.17 | 57.62 | 4069f798fa0fd4f4129e4e360e12ff703d2177ac60df0114358a699c63669875 |
| G70 | 11 | 86 / 100 | 54.17 | 61.18 | b080a983e406d83e6c3cef91f3618223f862fc1a362997591dbfd88ed1c86a01 |
| G70 | 23 | 82 / 100 | 58.94 | 53.43 | 24ec1afca280ea151691fa275f97aac04409755757f6f0d5cb46c03ffe35877a |
| G70 | 37 | 70 / 100 | 54.75 | 51.07 | 78efd64663777ed857a8c3d3ecc209b7387620fd61125d3541d12cdb7b9862ca |
| S50 | 11 | 97 / 100 | 56.91 | 61.33 | f9fbab90957c1ba20609c909a36a0f5044b305a77a92df5993ad94cdbe0e6d5d |
| S50 | 23 | 87 / 100 | 55.11 | 54.09 | 0946825bf5e78d53c1c7999964a79eff87fff70148d91455ed4f8e001b220216 |
| S50 | 37 | 98 / 100 | 57.99 | 54.67 | 17c01e6f98d14f7d5920ba607e3f6e6366836010c8dd8143939bbe804eca26e1 |
| G50 | 11 | 71 / 100 | 51.47 | 57.83 | 20f3590a0b5528f6e7962b8d975fed411b2ce00fabd895b616df9295b709948d |
| G50 | 23 | 96 / 100 | 54.08 | 58.62 | d31cc1ed4be2adb6bf001ac0203fefeb634352af28d58d86b5b89abdadfc780e |
| G50 | 37 | 99 / 100 | 59.05 | 51.66 | ca1a7f2e46e2499b8065276a41a9a0b654fd40846d0b520e52efa20bc363ee32 |

Array `65111362` is four tasks per seed: C0→GZ0,ρ=.9,.7,.5. S/G order: seed 11 SG/GS/SG; seed 23 GS/SG/GS; seed 37 SG/GS/SG. Task index=4×seed index+pair index. Each task has two independently optimized models from the same cloned initial state; the pair shares a host. All24 are valid. Constrained selected-checkpoint compliance:

| Case | Seed | Epoch | Val budget MAE | Coverage ±.05 |
| --- | --- | --- | --- | --- |
| G90 | 11 | 98 | 0.005725 | 1.0 |
| G70 | 11 | 86 | 0.008053 | 1.0 |
| G50 | 11 | 71 | 0.014285 | 1.0 |
| G90 | 23 | 86 | 0.005584 | 1.0 |
| G70 | 23 | 82 | 0.009762 | 1.0 |
| G50 | 23 | 96 | 0.01706 | 1.0 |
| G90 | 37 | 100 | 0.006261 | 1.0 |
| G70 | 37 | 70 | 0.010843 | 1.0 |
| G50 | 37 | 99 | 0.015348 | 0.992063 |

No invalid diagnostic checkpoint is used in a primary mean. Calibration seed 101 is excluded from this table and from formal means; its12 cells and compliance are enumerated in Section9.3.

### A.4 Exp16.3 formal 21

| Case | Seed | Selected / last epoch | Val BA % | Test BA % | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- |
| LIN | 11 | 95 / 100 | 56.33 | 60.10 | 635a18662d6d0bb257b0af5abcc31f34d7a2dd0608030082df69fef480800ab2 |
| LIN | 23 | 98 / 100 | 60.96 | 56.06 | 6db72400666c66fb52e5d36522b7cc0defdda4d9cb2bd812d2a42233657f5747 |
| LIN | 37 | 95 / 100 | 58.64 | 54.28 | 2b2198087798e1fdbebf54b753aeabb9d4997490d19198ea1431a31b9522b2e8 |
| CAP4 | 11 | 31 / 61 | 24.24 | 28.84 | 3be3f544a6d63de63fe7b9087f672ce30e1e8cd38c30e0e89892425c08bbb0ca |
| CAP4 | 23 | 96 / 100 | 37.94 | 37.83 | d7542bfdfca4da8db3c13ab4b7b890302269a3d93413d47922c5e07691e5389f |
| CAP4 | 37 | 80 / 100 | 33.08 | 28.05 | 8e18821279d9bb809e0293b58e49c4a02e1e75fbb23c78f67ca4d033abfcf4af |
| CAP8 | 11 | 100 / 100 | 37.13 | 37.93 | d4ce0ab37d7b62ce383cf5a9c55ac09f650ff09868a88f86a62fb064d14f8180 |
| CAP8 | 23 | 98 / 100 | 45.75 | 42.88 | f3c7a8d60cf2991cd76c02d46e391b76531f41c547f9df19b14fff82c95ec63e |
| CAP8 | 37 | 40 / 70 | 38.13 | 31.69 | 30533afc4f9276522e3951ae6c48ab3c76097c8c5a6809a4bccacd0301201886 |
| CAP16 | 11 | 97 / 100 | 52.30 | 58.28 | b7cf2ba01665aaaf762d16650b7f5ebd2f9d239d3754eebe00899bd1306ae675 |
| CAP16 | 23 | 99 / 100 | 46.88 | 49.41 | 40553cbe3b9bfe92343930e7e66334f0143aece3e61b6842d338e33b6c5ac47c |
| CAP16 | 37 | 91 / 100 | 51.34 | 51.45 | 6040104f8268ad347723d34cd33de32ba236609a4d32569fa63d76ebb6189ced |
| CAP24 | 11 | 100 / 100 | 51.76 | 56.28 | 1d27a832d994c051109256e326eacf4347569ba119146f043b87060daed341d7 |
| CAP24 | 23 | 89 / 100 | 55.00 | 51.42 | eff9f1e01280c309a1ec045ca414668e0012d81060178c263fce28d76a2e1dbc |
| CAP24 | 37 | 97 / 100 | 51.47 | 57.38 | 0f3ba32d72b4466ca8bdbd1dc66b9a28e504c046c889b7767b1660f53dbc09a0 |
| SUB8 | 11 | 96 / 100 | 35.63 | 40.12 | ac25528b86608aa3b22651a8ea831bca6cfd9166fa5f59f714f1a55beb1f4216 |
| SUB8 | 23 | 98 / 100 | 47.00 | 43.76 | 0793f4a47986c6babd11016c49ba2e4ea62470f81f7af49ca318946efac31b8b |
| SUB8 | 37 | 98 / 100 | 44.48 | 38.46 | 6c85268e9b61d7397ad1ffbf52bb8f3c237f47545b956fe520d2f095ba9b4a1f |
| EP1 | 11 | 87 / 100 | 22.57 | 16.52 | b6fbde978f072339befd05f6b69bf0dfe280f01d06c899b9bfdd3b7f98e1afb6 |
| EP1 | 23 | 83 / 100 | 27.25 | 16.98 | c7a3d2301475fec9a40e52fe72068e28cc4842ce1c1ea2e1bf797be0c33d667a |
| EP1 | 37 | 93 / 100 | 28.41 | 21.65 | 0193172cc2a79da887f9abceea7cdccde21b74b42924679b9abd3a2d2d6023a6 |

Array `65131209` is seed-major, seven cases in implementation order LIN/CAP4/CAP8/CAP16/CAP24/SUB8/EP1. Task index=7×seed index+case index. These are21 fresh models, not cap interventions on one trained LIN checkpoint.

### A.5 Main Phase0A retained snapshots/selectors and Phase0B cells

| Phase0A path (relative to Exp16 root) | Epoch | SHA-256 |
| --- | --- | --- |
| phase0a/seed 11/epoch 10.pt | 10 | e609adbb591dcbe904c2b32c32f18c6d1658e4ede21e837dbfafdc385f970fd5 |
| phase0a/seed 11/epoch 20.pt | 20 | c0ee28a319670f667ed7f2e69ff76c56c1686814856ffdd350b71572474eb98e |
| phase0a/seed 11/epoch 30.pt | 30 | 492c82dfffbb9a89b5aa938c4ff19d6c085d4ecf34923a31b088009ec17b2cef |
| phase0a/seed 11/epoch 50.pt | 50 | 7b2f93a497b23bc7916697b6671a7630647cab28a268bc12ad78e7e55223a31f |
| phase0a/seed 11/selected__ba_first.pt | 91 | 189612354c55ab6627139af1f6ca7b624edd0b5dc49c0849e13b4abfbafef7d7 |
| phase0a/seed 11/selected__ce_min.pt | 99 | f6627858b5da95460f797d231b766ecb30598979da44c8556beac11a94fa6cf8 |
| phase0a/seed 23/epoch 10.pt | 10 | 7369d0104ec5fb468b313dfa9a57f9b526fc1c9c0350061ee3853c2c92a1b29d |
| phase0a/seed 23/epoch 20.pt | 20 | 0c0e0aba976fd0df7b955115875dde3b3cda99807154bc9275405eb89f34de02 |
| phase0a/seed 23/epoch 30.pt | 30 | d2021cdf638036d144a37a8eaa01932654217c736b065427168910c1b3b4ef3a |
| phase0a/seed 23/epoch 50.pt | 50 | 09d1a736f1d97653fa169a1a02b7d040e64d086cb2a97a2abb591741d0371176 |
| phase0a/seed 23/selected__ba_first.pt | 63 | 5f082e90de7cf800fee8c338422355c20a88e1c910b95fef5b2f7f66909ef8f6 |
| phase0a/seed 23/selected__ce_min.pt | 92 | 0e7fb4b97e65b85c744da18ed22e1de167b10273078d16f7b5d7975dc0fda863 |
| phase0a/seed 37/epoch 10.pt | 10 | 903739f0b6450c3aeb9c185b86c89cb4a51be82ec16a4b57ed5e53d330b7446d |
| phase0a/seed 37/epoch 20.pt | 20 | 25862ccd7209e9bc1f4a17b08ba58d4d79df4270d8c1de996bf63b023810d166 |
| phase0a/seed 37/epoch 30.pt | 30 | 7cbfccba7ba8c8ea3e3c00ce020c12df9ce8bb9aa88268377ed6600eee2ae915 |
| phase0a/seed 37/epoch 50.pt | 50 | a8050fd354b3e5399123a1fb432d9f2d550799121189c2108e7248a85b4da0c1 |
| phase0a/seed 37/selected__ba_first.pt | 89 | 4992811bde11ee7b8a104aa194ab4279fc8b1d090e8e3af64ddfcc3ab543751e |
| phase0a/seed 37/selected__ce_min.pt | 99 | c9f4f655c26d650b1385e987d0a000686be9bea34c5a095bcc237f4a0652c83c |

For each seed 11/23/37, Phase0B executed all λ={0,.01,.03,.05,.10} at each base epoch 10/20/30/50:20 cells per seed,60 total. Task index=20×seed index+5×epoch index+λ index. All60 validation records are present; every positive λ has zero eligible points. They load a preserved snapshot and run one additional epoch; a resulting selected formal model is not preserved from each lookahead. Phase0A's six selector files reuse three trajectories and must not be counted as six independent models.

### A.6 Scheduler inventory

Times below are2026-10-01 UTC; an array's first start/last end is a stage range, not one task's duration.

| Stage | Job / array parent | Accounting rows | State | First start | Last end |
| --- | --- | --- | --- | --- | --- |
| exp16_prepare | 65103451 | 1 | COMPLETED | 2026-10-01T02:55:32 | 2026-10-01T02:55:48 |
| exp16_p0a_fin | 65103453 | 1 | COMPLETED | 2026-10-01T02:59:19 | 2026-10-01T02:59:47 |
| exp16_p0b_sel | 65103455 | 1 | COMPLETED | 2026-10-01T03:05:49 | 2026-10-01T03:06:16 |
| exp16_p1_fin | 65103457 | 1 | COMPLETED | 2026-10-01T03:25:20 | 2026-10-01T03:25:47 |
| exp16_final | 65103460 | 1 | COMPLETED | 2026-10-01T03:50:16 | 2026-10-01T03:50:34 |
| exp16_p0a | 65103452 | 3 | COMPLETED | 2026-10-01T02:56:04 | 2026-10-01T02:58:59 |
| exp16_p0b | 65103454 | 60 | COMPLETED | 2026-10-01T03:00:57 | 2026-10-01T03:05:23 |
| exp16_p1 | 65103456 | 12 | COMPLETED | 2026-10-01T03:06:22 | 2026-10-01T03:23:40 |
| exp16_p15 | 65103458 | 9 | COMPLETED | 2026-10-01T03:26:25 | 2026-10-01T03:27:46 |
| exp16_ablate | 65103459 | 15 | COMPLETED | 2026-10-01T03:28:02 | 2026-10-01T03:50:13 |
| exp16_1 | 65106790 | 1 | CANCELLED by 3883 | None | 2026-10-01T04:17:14 |
| exp16_1_prepare | 65107098 | 1 | COMPLETED | 2026-10-01T04:35:14 | 2026-10-01T04:36:39 |
| exp16_1_final | 65107101 | 1 | COMPLETED | 2026-10-01T05:33:46 | 2026-10-01T05:34:10 |
| exp16_1_train | 65107099 | 36 | COMPLETED | 2026-10-01T04:37:24 | 2026-10-01T04:56:16 |
| exp16_1_ablate | 65107100 | 24 | COMPLETED | 2026-10-01T04:56:55 | 2026-10-01T05:31:05 |
| exp16_2_prepare | 65111341,65111359 | 2 | COMPLETED,FAILED | 2026-10-01T06:45:51 | 2026-10-01T06:49:44 |
| exp16_2_p0fin | 65111361 | 1 | COMPLETED | 2026-10-01T07:05:21 | 2026-10-01T07:05:55 |
| exp16_2_final | 65111364 | 1 | COMPLETED | 2026-10-01T08:20:09 | 2026-10-01T08:20:35 |
| exp16_2_p0 | 65111360 | 12 | COMPLETED | 2026-10-01T06:54:31 | 2026-10-01T07:03:30 |
| exp16_2_p1 | 65111362 | 12 | COMPLETED | 2026-10-01T07:11:51 | 2026-10-01T07:37:43 |
| exp16_2_p2 | 65111363 | 12 | COMPLETED | 2026-10-01T07:43:50 | 2026-10-01T08:19:28 |
| exp16_3_prepare | 65131206 | 1 | COMPLETED | 2026-10-01T15:56:06 | 2026-10-01T15:56:54 |
| exp16_3_final | 65131210 | 1 | COMPLETED | 2026-10-01T16:19:06 | 2026-10-01T16:19:43 |
| exp16_3_scale | 65131207 | 24 | COMPLETED | 2026-10-01T15:57:15 | 2026-10-01T15:58:55 |
| exp16_3_obj | 65131208 | 6 | COMPLETED | 2026-10-01T15:58:20 | 2026-10-01T16:00:49 |
| exp16_3_train | 65131209 | 21 | COMPLETED | 2026-10-01T15:58:53 | 2026-10-01T16:18:54 |

The cancelled initial16.2 dependencies, independently queried by job IDs, were65111342_[0-11%12],65111343,65111344_[0-11%12],65111345_[0-11%12],65111346: CANCELLED, exit 0:0, no start, end06:45:56. They add no executed model. Posthoc analysis manifests establish24 saved-checkpoint coverage for each of the two16.2 analyses; their standalone process/job IDs were not recovered and are not invented.

### A.7 Reused Core O0/O1 diagnostic checkpoints

| Core case | Seed | Selected epoch | SHA-256 |
| --- | --- | --- | --- |
| O0 | 11 | 93 | 5845f3ac7c707f111e98084f4e644fac990839b5d9792bb5236fe263e3cc203e |
| O0 | 23 | 99 | c34ac6d8f1fe2e982ab1b904732f61251befa6e91370897c8eb9ed5a53529284 |
| O0 | 37 | 99 | 382b3e1a34a7190c7c6dddbaf60353966a747d955ee299bdd5b026ef9fc226c0 |
| O1 | 11 | 88 | 508fab70961c3493c86d7d97bb0a46d21eb871f3eb302803c1d7534737eeb580 |
| O1 | 23 | 94 | de9c7daa7d196b520eaad22eb9e3480c3ae5824c7a696e159c847452352aff1a |
| O1 | 37 | 99 | cafc44f77d9ff70d595bbe0d50a66cd7a65e63ae6b6aeae3d3e3cb8c2a8771f5 |

These six checkpoints are `training_complete=true`, at `core_benchmark_v1/results/main/runs/<case>__seed<seed>/checkpoint.pt`. They are diagnostic inputs to Exp16.3, without fresh SNN optimization. Exp16.3's24 scale diagnostics reuse the Exp16.2 checkpoints in A.3.

## Appendix B. Configuration / Case Definitions

| Branch / case | Trainable experiment change | Initialization and selection |
| --- | --- | --- |
| Main C0 | Ungated WCCE | Fresh Core-compatible parameter streams; selected BA-first |
| Main P0 | Planned WCCE+Prefix | Actual λP=0, ungated WCCE |
| Main G0 | z+previous-L2 suppressive write gate + WCCE | g0=.9, fresh joint training |
| Main GP_J | Planned gate+joint Prefix | Actual λP=0, same objective asG0 |
| Main GP_stopR / GP_gate / GP_noGate | Planned Prefix gradient-routing attribution | Conditional branch not executed |
|16.1 P / GPZ / GPZH | λP=.01/.03/.05, jointly trained | Direct grid, BA-first; no test-selected λ |
|16.1 GZ0 / GZH0 | z-only / z+history, λP=0 | g0=.9; suppressive, no amplification |
|16.2 C0 / GZ0 | Ungated / unconstrained z-only WCCE | Exact cloned backbone/head; ordinary BA-first |
|16.2 S90/S70/S50 | Constantρ=.9/.7/.5 | Same clone and epoch batches as pairedG; ordinary BA-first |
|16.2 G90/G70/G50 | z-only dynamic, WCCE+λB budget penalty | Initialg=ρ; λB=1; budget eligibility before BA-first |
|16.3 LIN | Linear spike sum/T | Fresh shared state; reward-native validation selection |
|16.3 CAP4/8/16/24 | min(run,K) reward/T | Same formal model/optimizer geometry; marginal-credit rule |
|16.3 SUB8 |8(1−exp(−run/8))/T | Same formal contract |
|16.3 EP1 | Episode onsets/T | Same formal contract; direct continuation credit zero |

Supplementary Exp16.1 primary probe table:

| Case | whole_count | fixed250_ordered | fixed250_shuffled | relative10_ordered | relative10_shuffled | R10−WC pp |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 54.46 ± 7.48 | 58.67 ± 5.90 | 50.44 ± 5.81 | 67.75 ± 4.58 | 50.31 ± 5.91 | 13.28 ± 3.12 |
| GPZH_lp0p01 | 56.01 ± 4.78 | 59.12 ± 2.44 | 52.15 ± 2.69 | 64.74 ± 0.26 | 51.21 ± 2.40 | 8.73 ± 4.96 |
| GPZH_lp0p03 | 55.45 ± 5.91 | 57.94 ± 4.68 | 51.13 ± 4.51 | 67.01 ± 3.36 | 50.65 ± 4.64 | 11.56 ± 2.60 |
| GPZH_lp0p05 | 55.33 ± 3.69 | 59.28 ± 3.01 | 51.60 ± 2.96 | 66.96 ± 2.72 | 51.45 ± 4.82 | 11.63 ± 1.82 |
| GPZ_lp0p01 | 55.30 ± 3.85 | 58.22 ± 6.77 | 50.35 ± 5.49 | 68.19 ± 7.06 | 50.02 ± 4.27 | 12.89 ± 3.45 |
| GPZ_lp0p03 | 56.02 ± 4.82 | 59.02 ± 2.86 | 52.52 ± 2.85 | 68.93 ± 2.90 | 51.33 ± 5.23 | 12.91 ± 7.10 |
| GPZ_lp0p05 | 56.97 ± 2.36 | 58.48 ± 1.95 | 51.70 ± 1.93 | 66.32 ± 4.63 | 51.22 ± 4.22 | 9.35 ± 2.52 |
| GZ0 | 57.22 ± 3.46 | 58.22 ± 1.75 | 51.40 ± 3.51 | 68.72 ± 2.58 | 50.47 ± 4.94 | 11.50 ± 2.26 |
| GZH0 | 56.91 ± 5.92 | 60.09 ± 2.51 | 52.09 ± 1.71 | 66.48 ± 4.31 | 51.95 ± 3.39 | 9.57 ± 4.37 |
| P_lp0p01 | 54.88 ± 5.02 | 58.50 ± 4.65 | 50.86 ± 5.14 | 65.30 ± 1.13 | 49.47 ± 5.09 | 10.42 ± 4.78 |
| P_lp0p03 | 54.59 ± 2.73 | 58.55 ± 0.70 | 51.07 ± 4.16 | 68.18 ± 1.60 | 50.46 ± 4.49 | 13.58 ± 1.94 |
| P_lp0p05 | 55.40 ± 5.95 | 58.21 ± 4.80 | 51.77 ± 5.93 | 67.42 ± 2.34 | 50.63 ± 5.91 | 12.02 ± 7.87 |

Supplementary Exp16.1 history diagnostics, with the distinct CV/retrieval definitions from Section8:

| Case | L2 trajectory retrieval BA % | Final L2 full-history CV BA % | Final L2 full-history test BA % |
| --- | --- | --- | --- |
| C0 | 66.04 ± 4.93 | 49.18 ± 2.15 | 48.27 ± 3.27 |
| GPZH_lp0p01 | 68.61 ± 3.95 | 52.97 ± 0.82 | 51.72 ± 3.05 |
| GPZH_lp0p03 | 65.52 ± 5.10 | 51.41 ± 2.10 | 48.04 ± 4.27 |
| GPZH_lp0p05 | 66.71 ± 4.43 | 51.03 ± 0.77 | 50.68 ± 2.49 |
| GPZ_lp0p01 | 67.30 ± 6.29 | 50.80 ± 3.23 | 50.06 ± 3.59 |
| GPZ_lp0p03 | 68.11 ± 3.15 | 51.36 ± 0.57 | 49.70 ± 4.48 |
| GPZ_lp0p05 | 67.38 ± 2.90 | 49.43 ± 1.61 | 49.54 ± 2.59 |
| GZ0 | 66.28 ± 2.99 | 51.41 ± 0.51 | 50.35 ± 2.45 |
| GZH0 | 69.67 ± 3.22 | 51.71 ± 0.83 | 50.67 ± 2.20 |
| P_lp0p01 | 65.16 ± 5.53 | 49.41 ± 2.53 | 51.37 ± 7.41 |
| P_lp0p03 | 65.83 ± 2.82 | 49.59 ± 3.38 | 49.90 ± 3.71 |
| P_lp0p05 | 65.70 ± 4.21 | 48.88 ± 1.19 | 49.68 ± 4.30 |

Exp16.2 per-layer persistent activity:

| Case | Layer | Test occupancy | Test mean longest run | Fraction neurons occ>.5 |
| --- | --- | --- | --- | --- |
| C0 | L1 | 0.0607 ± 0.0046 | 1.87 ± 0.11 | 0.0260 ± 0.0045 |
| C0 | L2 | 0.3726 ± 0.0064 | 21.63 ± 0.18 | 0.1198 ± 0.0316 |
| G50 | L1 | 0.0781 ± 0.0104 | 2.62 ± 0.28 | 0.0391 ± 0.0135 |
| G50 | L2 | 0.3643 ± 0.0121 | 20.22 ± 0.55 | 0.1432 ± 0.0045 |
| G70 | L1 | 0.0688 ± 0.0056 | 2.23 ± 0.17 | 0.0312 ± 0.0078 |
| G70 | L2 | 0.3609 ± 0.0108 | 20.69 ± 0.19 | 0.1276 ± 0.0197 |
| G90 | L1 | 0.0636 ± 0.0058 | 2.04 ± 0.13 | 0.0234 ± 0.0078 |
| G90 | L2 | 0.3727 ± 0.0048 | 21.60 ± 0.14 | 0.1328 ± 0.0207 |
| GZ0 | L1 | 0.0588 ± 0.0059 | 1.75 ± 0.39 | 0.0130 ± 0.0119 |
| GZ0 | L2 | 0.3566 ± 0.0353 | 20.99 ± 1.61 | 0.1172 ± 0.0488 |
| S50 | L1 | 0.0801 ± 0.0102 | 2.78 ± 0.25 | 0.0365 ± 0.0119 |
| S50 | L2 | 0.3694 ± 0.0017 | 20.30 ± 0.31 | 0.1458 ± 0.0401 |
| S70 | L1 | 0.0724 ± 0.0069 | 2.39 ± 0.12 | 0.0312 ± 0.0078 |
| S70 | L2 | 0.3722 ± 0.0048 | 21.17 ± 0.12 | 0.1302 ± 0.0325 |
| S90 | L1 | 0.0636 ± 0.0063 | 2.01 ± 0.20 | 0.0234 ± 0.0078 |
| S90 | L2 | 0.3679 ± 0.0074 | 21.30 ± 0.39 | 0.1276 ± 0.0119 |

Exp16.2 separately fitted duration controls:

| Case | duration_only | rate_normalized | whole_count |
| --- | --- | --- | --- |
| C0 | 11.31 ± 0.00 | 54.50 ± 3.84 | 55.46 ± 4.50 |
| G50 | 11.31 ± 0.00 | 55.89 ± 3.04 | 54.46 ± 2.14 |
| G70 | 11.31 ± 0.00 | 55.46 ± 4.21 | 54.82 ± 6.56 |
| G90 | 11.31 ± 0.00 | 58.57 ± 4.88 | 57.35 ± 2.63 |
| GZ0 | 11.31 ± 0.00 | 55.72 ± 4.54 | 54.83 ± 5.07 |
| S50 | 11.31 ± 0.00 | 56.08 ± 4.31 | 56.76 ± 4.14 |
| S70 | 11.31 ± 0.00 | 57.43 ± 2.91 | 54.87 ± 4.08 |
| S90 | 11.31 ± 0.00 | 55.50 ± 3.67 | 54.82 ± 3.23 |

Shared initialization hashes are equal within each seed/family. The two families serialize/hash states differently, so cross-family hash inequality is not itself evidence of different parameter values.

| Branch | Seed | Shared initialization hash |
| --- | --- | --- |
| 16.2 | 11 | f960e8063db7a430cee3f826ed80a02ef18d058fed6c242f9bb86b20d230167a |
| 16.2 | 23 | a8c243534bee6dd74e1ebaea45c9f6f4a2b5c58cd031c3ae773a7648f77e17c5 |
| 16.2 | 37 | fedb75829e06ff63f73dea9d13d7f167e3953ed4e5d60115e6bdd2143bb2da04 |
| 16.3 | 11 | 496d93cc03424283bd3c147306f3e432be37a089b592aba6a67a2f1fcf94ca20 |
| 16.3 | 23 | 7736cbbdd68b177c9a5c27f5f58b86cf3ff330e287ba37ff73de98ed12a88b8a |
| 16.3 | 37 | 537300a72de5835dadda8dfe5a1a08bb9584b356120fe3e1454455f6fee82696 |

Slurm scripts request one CPU per independent completed task, force numerical libraries to one thread, and clear CUDA visibility. Some hostnames contain “gpu”; that does not make these GPU training runs. Exp16.1 training max concurrency30 and ablation24; Exp16.2 arrays12; Exp16.3 training21, scale 24, objective 6. [Submission scripts](../../scripts/bash_script/SNN_Bash/submit_exp_16_cpu.bash) and the family-specific wrappers record dependencies; array entries and complete case definitions are in their corresponding implementations.

## Appendix C. Evidence Map

| ID | Evidence inspected | Role / coverage / boundary |
| --- | --- | --- |
| E16-S1 | [main source](../../scripts/experiment_16_prefix_supervised_selective_memory.py), [README](../../scripts/experiment_16_prefix_supervised_selective_memory/README.md), [contract tests](../../tests/test_experiment_16_prefix_supervised_selective_memory_contract.py) | Selector, lookahead, conditional λ/routes, gate and diagnostic definitions |
| E16-S2 | [16.1 source](../../scripts/experiment_16_1_z_only_prefix_interaction.py), [README](../../scripts/experiment_16_1_z_only_prefix_interaction/README.md), [contract tests](../../tests/test_experiment_16_1_z_only_prefix_interaction_contract.py), initial/final scheduling diff | Direct factorial and actual multi-node execution; README title conflicts with final contract |
| E16-S3 | [16.2 source](../../scripts/experiment_16_2_matched_budget_selective_write.py), [README](../../scripts/experiment_16_2_matched_budget_selective_write/README.md), [contract tests](../../tests/test_experiment_16_2_matched_budget_selective_write_contract.py) | Calibration, clone/batch pairing, per-sample budget, constrained selector and replay |
| E16-S4 | [16.3 source](../../scripts/experiment_16_3_run_reward.py), [README](../../scripts/experiment_16_3_run_reward/README.md), [contract tests](../../tests/test_experiment_16_3_run_reward_contract.py) | Reward forward/gradient contract, formal 21 and reused 24+6 diagnostics |
| E16-S5 | Two16.2 analysis scripts linked in Section1; [Core data](../../core_benchmark_v1/data.py), [model](../../core_benchmark_v1/model.py), [training](../../core_benchmark_v1/training.py), [probes](../../core_benchmark_v1/probes.py); Exp14 helper source | Dynamics, feature replacements, utility signs, occupancy, scaling, retrieval and history definitions |
| E16-U0 | Unity `core_benchmark_v1/results/main/protocol.lock.json` and matched four experiment protocols | Frozen identity/cache/split/geometry; historical lockv1.0 distinguished from loader adapter |
| E16-U1 | Main root checkpoint/prefix/phase1 decision JSON; aggregate selector/lookahead/eligibility/summary/native/probe/ablation tables; per-run histories/gates/history outputs;30 `.pt` metadata/checksums | Primary calibration and conditional-skip authority;12 formal models, not30 independent runs |
| E16-U2 |16.1 `allocation.json`, protocol, PASS manifest; native/probe/interaction/gate/gradient/update/ablation CSV; per-run JSON;36 `.pt` metadata/checksums | Actual hostname set, coefficient treatments and all seed outcomes |
| E16-U3 |16.2 protocol, λB selection, calibration, formal native/probes, matched gains, replay/causal tables, per-run write/state/gate/history JSON,24 `.pt` metadata/checksums |24/24 valid; paired hashes; selected budget eligibility; replay learned identity |
| E16-U4 |16.2 `persistent_neuron_diagnostic/` manifest and sixCSV; `sustained_firing_mechanism/` manifest and sixCSV; raster manifest |24 checkpoints covered by each analysis; frozen/refitted distinction; fixed-drive normalization/reset/free decay |
| E16-U5 |16.3 protocol/PASS summary, formal native/probes/activity/deltas, scale neuron/τ, objective activity, cap/tail CSV and per-run histories;21 `.pt` metadata/checksums | Formal reward result and separate reused diagnostic cohorts; neuron-weighted activity recomputed |
| E16-J | Unity user-scoped Slurm accounting for listed job IDs, plus failed pipeline direct-ID query; explicitUTC preparation checks | Completion/cancellation/failure and dependency outcomes; not proof of a non-skipped scientific treatment |
| E16-H | Targeted Project-history searches by family IDs, selector/Prefix/gate/budget/replay terms, raster/persistence/reward, execution failures, and successor terms | Dated intent/interpretation trace; returned summaries/user statements, not a full archive export; conversation titles not recovered |
| E16-P | [preceding Exp15 report](../exp15/report.md), source/commit continuity | Cross-family baseline and gate-equation consistency; predecessor quantitative values treated as contextual references |

Quantitative tables in Sections9–10 and Appendices A/B derive from E16-U1–U5. Scientific decisions and interpretation changes in Sections2–5 and 15 derive from E16-H, cross-checked against source revisions and executed cohorts. Bugs/reruns use E16-J plus inspected commit patches. Source tests were read to identify contracts; this documentation-only reconstruction did not rerun training or claim to rerun the historical test suite.

Important boundaries: full conversation transcripts/titles are unavailable; failed-job stdout/stderr were not recovered; no unique cause was isolated for λ0 cross-node divergence or the two independent probe-refit discrepancies. The inspected tree contains no actual CF90/R10-90 or canonical 16.4 run to support a numerical claim.16.2/16.3 raw artifacts were present on Unity but untracked at cutoff. All93 formal rows and hashes are included here so the report can be audited without silently depending on cached means.

Aggregate CSV/JSON checksums at the evidence cutoff (paths are relative to each canonical root's `aggregate/`):

| Branch | Aggregate file | SHA-256 |
| --- | --- | --- |
| 16 | functional_ablation_native_deltas.csv | 31b4b666b9fe54b4e73cfe36c650fe5788bdba83ac3a8a0208148fee65116794 |
| 16 | functional_ablation_native_runs.csv | 9026f35b0da282f8a82018af964c091ee94490b415a79423ad2284fa9d8c9ee7 |
| 16 | functional_ablation_probe_runs.csv | 770e7d0ab3a6ad7b28611d0d1a01e14c84c5edd4ca75557075a2c3c2facf3f81 |
| 16 | manifest.json | db651b4ff28250c944bc40c202f9fedb49bd846b21beefb63731f305509a58f4 |
| 16 | phase0a_checkpoint_selection.csv | 4944d5dbfb6595a2e4b1babee9b62edd75e16709dbc451d3ae8b3c189b69a0cb |
| 16 | phase0b_prefix_eligibility.csv | 19326565fe73769a7e2bdf83637a2a55a69b91e9538399a0423bc6ba163ded92 |
| 16 | phase0b_prefix_lookahead.csv | 6e9dcab2d97795067c65e766181cb0d43dd4a21d864b5428bae43e38eb880a88 |
| 16 | phase0b_prefix_summary.csv | 67bcf3c16c311c38ddc9fd95f0d479d13439c85fde4fde121614599020b45362 |
| 16 | phase1_collapse_gap.csv | fc9011325f243bcd9aefb1c7fbee12c098b8abd3e226873c097d6b728f48b208 |
| 16 | phase1_native_runs.csv | 22e44087eb9fe9fef8dd7b51bda46a4fe3281182f8f42b901f26f4e1569b0de5 |
| 16 | phase1_probe_runs.csv | 5f231397068bdb7bdaf0b2d6a053ec4b0ca74297f9b07223e3592da36e72407e |
| 16.1 | collapse_gap.csv | 416ad3f83523237ddb7673d076c1c6747875ae311ad6df05a6897448478b6ef2 |
| 16.1 | functional_ablation_native_deltas.csv | 34cb009bb5347285349b52d8bce6e849b61c1abbdc29534e9fe8aadf6b31e605 |
| 16.1 | functional_ablation_native_runs.csv | c1b2d9089de07e22ab8b5401a17bfbd16ea8426d36f535d12bc16d271b3ea026 |
| 16.1 | functional_ablation_probe_runs.csv | 44c24d2933bd53117821ab5addecc66a87eae7ef4e314682169e7865ab1ff9f9 |
| 16.1 | gate_phase10.csv | 949fc2975d684daf1960af3901a61819d5f5603a506b83d764762afef8687a0e |
| 16.1 | gate_summary.csv | 49eae58546fd09825381b2333129dca560f98caab1cf218b338ea4cfedcea5de |
| 16.1 | gradient_geometry.csv | 004a18e23757e8c7fd393d4ae4079ad06d7d331e9afddc60d5ca28c9ad35bf5b |
| 16.1 | interaction_summary.csv | afbb6e14e70405b84f6165c6b266f64b9880ec85e0e30ad53c7d0de74b0e3d89 |
| 16.1 | interactions.csv | f1711310f6f77ce561acf2b1837ffdb10978d4c11283ee543a4529aa5eb9dfc3 |
| 16.1 | manifest.json | 84496e4ae94f26413b1599554b589435ee4572243ae6ed19800a620d4f702a02 |
| 16.1 | native_runs.csv | 2af4731bde73429abce26a05a196c6afd6bddca73f6d7f3800c0e243d09d58c3 |
| 16.1 | optimizer_update_diagnostics.csv | 194737c9817f2d6885b053635f452978a508600e2894abef643bfa0037f6393b |
| 16.1 | probe_runs.csv | 2bb6b79c8191d3d143904bb0233c69c16558424d84901550d4c6441ed13548fd |
| 16.2 | adaptive_gating_gain_mean.csv | 8dc4e0b748d50034a86615c07ab44615267faf2fee934178a06f68b604f4dc2c |
| 16.2 | adaptive_gating_gain_per_seed.csv | 67678dfdd1cc4363f890448dd8ebaefb65b2ce08a0efb8e5975ccab15e2fcd9e |
| 16.2 | formal_native_runs.csv | b78ce286641f2e56b5b9af3542aceaab9b7740f06ce2cd02ac92bc6b434ea38e |
| 16.2 | formal_probe_runs.csv | f05e74500546d426892e78a4182057dc5d9ca2c8125fbfcce7c173f6445bcb71 |
| 16.2 | phase0_budget_calibration.csv | c5a6867ae0fa3023168b053b7d662e731098fd94147232f6d35f1a43103126b5 |
| 16.2 | phase2_causal_deltas.csv | 8291187e6b6d3d0452bd035b5611445fbe97148e98f9f9bdb00aeb44471b6dc1 |
| 16.2 | phase2_replay_native.csv | a180da3dbf197c3f0452822225cf56540478ccf2693222626b729b8e739617be |
| 16.2 | summary.json | 08daa10ddb0bde45184b37ac746b9f116518fa43028897e746da2556ab69e016 |
| 16.3 | early_tail_incremental.csv | 48fd08273fe58739fbb9449a30f206cd757cb2e54136f51f31b32285ded9b890 |
| 16.3 | early_tail_incremental_mean.csv | 59b1219594b647ed414517eb9112a363fc7d993c9016cb57c300dd3bd1e6f5e3 |
| 16.3 | formal_activity.csv | b3e07f824893ecd210caa806af8250e231a63f75a08df51720367204ef10c40d |
| 16.3 | formal_activity_mean.csv | 7c837d1e90d36ce93f426949bd365e45c2f27cc034b0b03c5c745bbba7691fee |
| 16.3 | formal_deltas_vs_linear.csv | 3751ea92dcfe749c4ad24da3c9b5b10bdf105862bf709584e05bfad0f6e39565 |
| 16.3 | formal_deltas_vs_linear_mean.csv | bef9b08a2e8905adebf540b55360b8a1315f6da938bce3d4b6e75fd5829c3cfe |
| 16.3 | formal_native.csv | 968d6e766f6013b829d5731796d6d46ff4788b131f435dea8e3d13315fe49e6b |
| 16.3 | formal_probes.csv | dd16c87b4275f00831b0d36e375568d72f92052eb49ad13217a17c60844d92c3 |
| 16.3 | objective_activity.csv | 5d90d4b274df53d412f24151db5868ba0866426194213351a464e384ee2beaea |
| 16.3 | objective_activity_mean.csv | fceaa4898ed967592a7a7951d892cbb5f89dacb81460768939244205d893a143 |
| 16.3 | run_cap_curve_mean.csv | 5830a39096a53ea719579eed02ec8d3c28a44c5130965b51a362af80aee17dce |
| 16.3 | run_tail_probe_diagnostic.csv | 7c120d26a51e0a1db43f5b2acd9b92abb93741bc96b4ea4e1f13d414b4e36249 |
| 16.3 | scale_compensation_neurons.csv | 924daa8c4e9a1ce5dbbdc266e73a78a44a19cf06041fa912c0787c5b23c2fe5d |
| 16.3 | scale_compensation_tau.csv | f3a05b69c0906eb6192793021fbdad3ae4af5fe231904e59e789e5020de1a108 |
| 16.3 | scale_compensation_tau_mean.csv | 5f97a5e673908ad8186b6f5644fd16053efd9ce2af8c282689e6b52cbee6b297 |
| 16.3 | summary.json | 2096f494e8b794b3946c703217d556fc7ac7d5ed5818aeb555151e6408f8cb4f |

## Appendix D. Chronological Reasoning Trace

Conversation titles were not exposed by retrieval; entries below use topic labels, not invented titles. User decisions and prior assistant interpretations are distinguished. Commits and accounting are independent execution evidence. Times areUTC,2026.

| Time | Source / topic | Decision, observation or proposal | Consequence / evidence boundary |
| --- | --- | --- | --- |
| Sep 30 19:18 | User, history/transfer bottleneck | Memory-content selection/organization remained a hypothesis | Motivated writing control; not a proven capacity diagnosis |
| Sep 30 19:57 | User, Exp15 controls | Required frozen/joint plus native and temporal probes | Preceding contextual gate study; no Prefix in Exp15 |
| Sep 30 21:39;22:14 | User / prior assistant, gate outcome | Small native gate-specific held-out effect; strong seen-training fit | Selective writing remained unestablished |
| Oct 1 00:46 | Prior assistant, WCCE versus TSCE | Proposed keeping sequence-level WCCE and testing Prefix/gated controls | TSCE discussion is motivation; not a formal main16 treatment |
| Oct 1 01:39–01:47 | Prior assistant, Prefix credit | Prefix could reach gate but also use readout/backbone shortcuts; propose routing and gradients | Motivated factorial interaction and conditional attribution |
| Oct 1 02:40:26 | User, accepted main16 | Phase0A/B, C0/P0/G0/GP_J, conditional attribution, g∈(0,1) | Implemented02:53; completed 02:55–03:50 |
| Oct 1 03:55:29 | Prior assistant, main16 outcome | λP=0; positive Prefix interaction not tested; nearly-open gates | Direct positive-λ follow-up required; Unity decisions confirm |
| Oct 1 04:05:43 | User, accepted16.1 initial plan | Single-node30CPU, z-only/ZH, λ=.01/.03/.05 | Initial job cancelled; final scheduling differs |
| Oct 1 04:15–04:32 | Git commits,16.1 execution revision | Initial one-node code replaced by 1CPU arrays and host reporting | Completed36/24 on ten nodes; single-node assertion absent |
| Oct 1 05:39:22 | Prior assistant,16.1 interpretation | Prefix fails recruitment; .03 z-only test interaction seed 11-driven | Motivated a budget constraint, not more λ tuning |
| Oct 1 05:45:35 | User,16.2 design | Stop Prefix/complexity tuning; static/dynamicρ=.9/.7/.5 | Defines primary Gρ−Sρ |
| Oct 1 06:12:00 | User, final16.2 amendments | seed 101 calibration, exact clone, deterministic batches, record/replay; exclude Prefix/ZH/entropy/hardgate/utility | Final implemented scientific contract |
| Oct 1 06:36;06:45–06:48 | Commit/accounting/history,16.2 bugs | Parser fixed; prepare then fails127; profile initialization fixed and resubmitted | Successful rerun retained; initial dependencies cancelled |
| Oct 1 08:20;13:58:43 | Accounting / prior assistant,16.2 results | Formal pipeline complete; G90+2.55pp but Learned−Mean≈0; tighter budgets not a win | Training benefit separated from inference timing necessity |
| Oct 1 14:14:38;14:15:28 | User / prior assistant, proposed CF16.3 | Sampled write utility atρ=.9; CF90/R10/oracle design | Retrieved proposal, later replaced; no executed numbers |
| Oct 1 14:36;14:59–15:00 | Git, persistent analyses | Artifact-only neuron utility and fixed-drive mechanism/τ alignment | Reused24 checkpoints; no SNN retraining |
| Oct 1 15:24:09 | User, scale-identifiability amendment | Fixed normalization/gate can be absorbed by W; require RMS/abs/weight and run-tail/objective diagnostics | Prevents treating fixed-weight sparse replay as a learned structural fix |
| Oct 1 15:29:35 | User, actual16.3 decision | LIN/CAP4/8/16/24/SUB8/EP1, one sequence-level CE | Replaces earlier proposal; source15:36 matches |
| Oct 1 15:36–15:45;15:56–16:19 | Git / accounting,16.3 implementation and execution | Preparation cleanup, aggregate identifier fixes;21 formal+24scale+6objective tasks complete | Final per-seed outputs retained |
| Oct 1 16:43:07;16:58:22 | Prior assistant / user,16.3 results | Caps hurt BA and fail to lower persistence; tail carries information | Numerical claims verified here; K16-tail complementarity retained |
| Oct 1 21:30:47 | User, persistence interpretation | Persistent firing systematically occurs and contains class information | Shift from blanket suppression to useful communication |
| Oct 1 22:04:47;22:07:02 | User / prior assistant, ablation/refit | Frozen loss and refit recovery vary; S90/S70 improve, S50 harms | Probe refit distinguished from native/SNN retraining |
| Oct 1 22:21:41 | Prior assistant, proposed 16.4 | High/low/random/native-head and transfer controls | Proposal only at this cutoff |
| Oct 1 22:35:14 | User, reward outcome | Strong caps do not suppress persistence; architecture/objective interaction remains | Leads to persistent-pathway trajectory analysis |
| Oct 2 00:53;04:14:11;05:15:37 | Successor discussion / user decisions | Exp17 temporal pathway analysis; routed context/evidence pivot; Exp17.1 rescue implementation | Explains successor questions; no successor numerical cohort imported |
| Oct 2 report reconstruction | Source + Unity + bounded history | Separate legal calibration negatives, executed factorials, training gains, timing nulls, and persistent information | This report preserves the scientific reasoning and its evidence limits |

