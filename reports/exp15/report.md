# Exp15 — Context-Dependent Write Gate and Width Capacity Ablation

## Report Scope and Evidence Cutoff

This report covers `experiment_15_context_dependent_write_gate` (`context_write_gate_v1`) and `experiment_15_1_width_capacity_ablation` (`width_capacity_v1`). It includes the pretrained-baseline controls, gate-only/joint continuations, inference interventions, and matched 64/96/128 width comparison. Prefix supervision, suppress-only gating, z-only gates, budget constraints, and Exp16/16.1/16.2 are successor designs, not Exp15 cases. Their design is discussed only to explain the transition.

Evidence was inspected on 2026-10-02. Implementation was read from GitHub snapshot `3f0e349e48f46f0c14b536e1923488c27381dcbc`; Unity results were read from `/home/zhaolongwei_umass_edu/projects/writingRing` at tracked HEAD `ef12b4e22de64693f3af317e4757ba913bf8ab12`. The two snapshots are recorded separately: current source explains the final contract; commits and run manifests establish revisions. Unity files and checkpoint metadata determine quantitative outcomes. No training or cluster files were changed for this reconstruction.

Project history was actively searched by experiment IDs and mechanism terms, including discussions before design, during implementation, after results, and in successor experiments. The accessible retrievals contain dated user statements and prior assistant summaries, rather than a complete exported Project archive. They support the reasoning chronology but do not establish run completion or numerical truth. Titles are reported only where recovered; other entries are labeled by topic with title unavailable. Missing full transcripts and original failed-job stdout/stderr are evidence gaps, not reconstructed quotations.

All BA values are balanced accuracy, reported as percentages; differences are percentage points (pp). Unless explicitly labeled otherwise, `mean ± SD` is an equally weighted mean and sample SD (`ddof=1`) over model seeds 11, 23, 37, **n=3**. Shuffle replicates, batches, neurons, strokes, and users are not additional model seeds. No significance claim is made from these three paired seeds.

## 1. Experiment Identification

| Alias | Canonical repository ID / protocol | Executed substructure |
| --- | --- | --- |
| Exp15, remember gate, context write gate | `experiment_15_context_dependent_write_gate` / `context_write_gate_v1` | Phase1 C0/GF/GJ; immutable B0; phase1.5 A0/A1/A2/A3/A4/A4b interventions |
| Exp15.1, width/capacity | `experiment_15_1_width_capacity_ablation` / `width_capacity_v1` | Fresh B0 at H64/H96, then same C0/GF/GJ and interventions; H128 immutable reuse |
| B0 | Historical Core O0 native WCCE checkpoint | No Exp15 training; paired seed reference |
| C0 | Ungated WCCE continuation | Controls additional optimization of pretrained native model |
| GF | Gate-only continuation | Original backbone and native head frozen |
| GJ | Joint gate + backbone/head continuation | Extra parameters and representation adaptation both possible |

Implementations are [Exp15](../../scripts/experiment_15_context_dependent_write_gate.py) and [Exp15.1](../../scripts/experiment_15_1_width_capacity_ablation.py). Artifacts use `notebooks/artifacts/<canonical ID>/<protocol>/runs/`, with H64/H96 subdirectories in 15.1. Slurm `exp15_p1/p15` and `exp15_1_b0/p1/p15` names are stages. Exp14.1 is a motivation source, not the checkpoint source: Exp15 loads historical Core O0, not the fresh Exp14.1 C0.

## 2. Scientific Context

Exp13 separated longer remembered context from transferable consolidation. Exp14 improved some representation geometry but did not reliably close the Relative10–WholeCount/native gap; its corrected Prefix grid failed validation safeguards. One explanation remained that all incoming L1 messages were written indiscriminately into L2, mixing useful and nuisance history before the native head accumulated evidence.

At 2026-09-30 19:57 UTC the user required frozen-backbone and joint training in parallel, with diagnostics beyond native BA. Prefix and further temporal-loss changes were reserved for unresolved cases. The implemented first test therefore changes a write multiplier while keeping the WCCE objective. This is important: the prompt's discovery hints about “z-only” or “Prefix” refer to later directions and are not Exp15's executed mechanism.

## 3. Motivation and Open Question

Could a small state-dependent gate decide how much current communicated input to write into L2's synaptic memory? If learned timing and history conditioning are useful, gate-only training should improve its paired immutable baseline, and joint training should improve a matched ungated continuation. Replacing the learned gate by uniform or mean gain, shuffling its timing, or removing its history input should then harm performance.

Alternative explanations were extra WCCE optimization, adaptation of the backbone/readout, a nearly constant global gain, or an easy all-open solution. Gate statistics alone cannot distinguish them. The controls and interventions were designed to do so. Later width reduction tested whether a high-capacity backbone made a useful gate unnecessary, rather than silently changing gate design or loss at the same time.

## 4. Hypotheses

| Hypothesis | Prediction | Discriminating comparison |
| --- | --- | --- |
| Useful selective writing | Learned context/time variation improves held-out BA and is necessary at inference | GF−B0, GJ−C0, A0 versus mean/shuffle/history removal |
| Additional training explains changes | Ungated continuation provides similar gains | C0 versus B0; GJ compared with C0 rather than B0 alone |
| Gate acts mainly as gain | Nearly open multiplier; per-sequence mean or m=1 preserves test BA | Statistics, A1/A2/A3 |
| Previous L2 spikes provide useful context | Removing or averaging gate history degrades matched inference | A4/A4b; underlying L2 dynamics remain intact |
| Capacity suppresses a need for gating | H64/H96 reduce generalization gap and increase gate-specific benefit | Width-matched baselines and paired contrasts |
| Smaller models only need more epochs | Extra ungated continuation substantially closes held-out deficit | C0 versus B0 and checkpoint/training history, bounded by max epochs |

## 5. Experimental Design

Exp15 begins with the same validation-selected Core O0 checkpoint for each seed. B0 is immutable evaluation of that checkpoint. C0 continues all native parameters without a gate. GF adds the gate but freezes every original SNN parameter and the output head. GJ trains both gate and original parameters. The matched causal contrasts are GF−B0 and GJ−C0; GF−C0 is an unmatched contrast because C0 adapts the original network.

Gate initialization preserves the baseline function to floating-point tolerance. Epoch0 is eligible, so a run whose training does not improve validation can retain the identity gate. Each continuation has a newly initialized Adam optimizer and the same WCCE training contract. Evaluation restores the validation-selected checkpoint, not the last epoch.

Phase1.5 applies six interventions to each selected GF/GJ checkpoint without changing learned weights. For each intervention, the SNN is replayed from zero state; native output is evaluated directly and diagnostic probes are refitted on matching intervention train/validation/test trajectories. A3 has five sample-keyed shuffle replicates, averaged within seed.

Exp15.1 changes only **both hidden widths**, H=64 or 96, while keeping input, task data, τ groups, gate equation/init, WCCE, optimizer, selection, probes, and interventions. It trains six fresh smaller B0 models, then 18 continuations and 12 intervention tasks. H128 reference outcomes are reused from Exp15 through a hashed manifest, not retrained, copied into a new independent sample, or counted as extra seeds.

## 6. Implementation

The inherited Core cache uses action 0 and action 1 handwriting, 64 Hz, 256 padded steps, and valid lengths. The actual schema is `custom_wavelet_polarity_split_abs_events_v1`: 30 unsigned event channels enter the SNN; six additional continuous channels from the 36-channel producer are not SNN inputs. The class order is A, B, C, D, E, G, H, I, J, K, L, X. Train users are 0, 1, 2, 5, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20; validation users 4, 9, 16; test users 3, 6, 10. There are 581/126/146 train/validation/test sequences. Split seed is 12345; model seeds are 11/23/37.

The inspected lock identifies `core_benchmark_v1.0`, identity `85ccaddc963c4502624f02b11ba420bb5c1ac9037abd2715b11b0f52ca175d99`, dataset SHA-256 `d57813f9cb16f3624dfed10e294ab6c4a9005f0384f025145fcc37f4634ffd0a`. Experiment loaders accept v1.0/v1.1 locks and adapt the protocol object; this does not mean the historical cache was regenerated as v1.1. Data roots are `outputs/action{0,1}_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded`.

The two-layer baseline has width 128 per layer and shifts 234×234: α = 1−2^(−shift) gives 0.75, 0.875, 0.9375, allocated 43/43/42 neurons per layer. At Δt=15.625 ms these imply synaptic e-folding times approximately 54.31, 117.01, 242.10 ms. Membrane τ is 22.54 ms, threshold 0.5, spike cap 1, subtractive reset. Signed bias-free feedforward matrices communicate spikes; there is no trainable recurrent matrix. Synaptic current and membrane carry temporal state.

The update is **unnormalized**: Iₜ = αIₜ₋₁ + Wxₜ, without a (1−α) drive factor. The output is signed bias-free evidence eₜ=Wout sᴸ²ₜ with an accumulator, no extra output synaptic filtering. Whole-sequence cross-entropy (WCCE) is CE of **valid-mean logits**; native prediction is argmax of valid-summed logits, which has the same argmax as their mean. Timestep CE (TSCE) instead computes CE at each valid timestep; Core averages within each sequence, then across sequences so longer sequences do not receive extra weight. Exp14/15 use WCCE, not TSCE. Adam uses learning rate 0.001, weight decay 0, ordinary task batch size 128 where preserved, maximum 100 epochs, minimum 20, patience 30. Epoch 0 is eligible. Checkpoints maximize held-out-user validation native BA, then minimize validation valid-mean-logit CE, then prefer the earliest exact tie. Training gradients minimize the training objective; neither test BA nor a fitted test probe selects checkpoints.

The gate reads **current L1 spikes sᴸ¹ₜ and previous L2 spikes sᴸ²ₜ₋₁**:

\[
a_t=w_z^\top s^{L1}_t+w_h^\top s^{L2}_{t-1}+b_g,\qquad
g_t=\sigma(a_t),\qquad m_t=g_t/0.9,
\]
\[
I^{L2}_t=\alpha I^{L2}_{t-1}+m_t\,W_2s^{L1}_t.
\]

The scalar multiplier applies to the whole current input message, not separately to each neuron. It gates the new write, not the αI decay, membrane retention, or final evidence directly. Previous L2 spikes are communicated output, not direct access to complete L2 I/U. Current L1 spikes already contain L1 history, so even a future “z-only” gate would not necessarily mean history-free raw input. Exp15 itself uses **both terms** and retains closed-loop L2 dynamics.

Both gate vectors initialize to zero; b=logit(0.9), so g≈0.9 and m≈1. The multiplier range is **(0,1/0.9)≈(0,1.111…)**, allowing amplification as well as suppression. Thus g near one means an amplified write relative to B0, not merely a baseline all-open write. The gate has 2H+1 trainable parameters, including a scalar bias; the native matrices/head remain bias-free. Adam uses the same 0.001 rate and zero weight decay; gate bias is in an explicitly zero-decay group.

No Prefix loss, gate penalty, sparsity budget, reward, or CU projection exists in this family. WCCE offers differentiable task credit through surrogate spikes and the gate. It does not explicitly penalize writing or require selective timing, so increasing gain and adapting weights are available solutions. This is an interpretation of the optimization incentives, not a proof that WCCE can never learn selection.

## 7. Executed Runs and Validity Audit

Slurm `65092452` prepared Exp15; `65092453` completed nine C0/GF/GJ continuation tasks; `65092454` completed six GF/GJ intervention tasks; `65092455` finalized successfully. B0's three historical Core checkpoints were reused. All formal task records show `COMPLETED`, exit0:0. No failed task is included in primary means.

Exp15.1 `65095880` prepared, `65095881` trained six B0 runs, `65095882` completed 18 continuation runs, `65095883` completed 12 intervention tasks, and `65095884` finalized, all successfully. Its `reference_manifest.json` fixes Core identity/cache hash and hashes of Exp15 reference metrics. H128 is reference reuse, so the combined comparison has three seeds per width/case and 24 new smaller-width models, not 36 freshly trained models.

Repository revisions strengthened accounting and diagnostic retention: B0 provenance, optimizer handling, causal deltas, and compact credit diagnostics are recorded in Section13. Current source and final manifests agree on gate equation, selected model identities, case counts, and intervention scope. No evidence supports relabeling a later z-only or suppress-only run as an Exp15 replacement.

## 8. Evaluation and Diagnostics

Core temporal probes freeze the SNN and train a separate multiclass logistic readout. WholeCount sums valid spikes (dimension H); Fixed250 concatenates 16 fixed 16-step/250-ms bins (dimension 16H); Relative10 concatenates 10 normalized-progress bins (dimension 10H, integer endpoints from `linspace`). Ordered probes preserve bin positions. Fixed250 shuffle permutes only complete valid bins, leaving a partial final bin and padding fixed; Relative10 shuffle permutes all ten bins. Each sample uses a stable sample-ID-keyed permutation. Train, validation, and test are all transformed under the same matched shuffle protocol and the decoder is refitted for each replicate. This measures access to order-specific information, not corruption of test data under an ordered-only decoder.

The C grid is 0.001, 0.01, 0.1, 1, 10, 100, selected on validation BA among converged candidates, favoring smaller C on ties. Both Core decoders use scale-only `StandardScaler(with_mean=False)`; `no_bias` sets `fit_intercept=False`, `affine` sets it true. Thus no-bias is origin-preserving; affine adds a constant class offset. A fitted probe is not the trained native head. Five shuffle seeds 101, 211, 307, 401, 503 are averaged within each model seed before the n=3 aggregate. Primary comparisons below are L2 spikes with no bias. Affine outputs are supplementary and are not silently substituted for this contract.

Native train/validation/test BA uses the selected **original signed head and accumulator**. Probe gains are differences between separately fitted feature readouts; they do not deploy a new native classifier. Width changes feature dimension and capacity, so cross-width probe differences include both changes. Primary gate usefulness requires the matched native contrast and interventions, not a favorable high-dimensional probe alone.

Gate summaries exclude padded timesteps. They record mean/SD/quantiles of g/m, fractions near saturation or away from identity, within-sequence variance, variance of sequence means, and ten normalized-progress bins. Input/history contributions are RMS of wᶻ·sᴸ¹ and wʰ·sᴸ²prev; their ratio measures logit contribution size, not task usefulness, causal information, or a full-memory decomposition. Pooled timestep SD is first computed within each selected run; the table then reports its mean±seed SD.

The intervention definitions are [E15-4]:

| Case | Exact inference manipulation | What remains / what it tests |
| --- | --- | --- |
| A0 | Selected learned gate | Reference closed-loop trajectory |
| A1 | Force mₜ=1 | Remove all gate modulation at fixed learned native weights |
| A2 | First run A0, compute each sequence's valid mean m; replay with constant m | Preserve sequence-level gain, remove gate timing |
| A3 | First run A0, permute valid m values per sample; replay | Preserve multiplier distribution, destroy temporal alignment; five replicates |
| A4 | Set wʰ·sᴸ²prev contribution to zero in the gate | Remove gate history conditioning while native L2 I/U history remains |
| A4b | First run A0, average gate history logit within each sequence; replay using that constant | Preserve mean history contribution, remove its timing |

A2/A3/A4b are two-pass trajectory interventions, not a continuously adaptive policy after intervention. The second run changes downstream states, so the preserved values belong to the original trajectory. A4 does not erase memory itself. Refit-probe BA can show accessible information after intervention but cannot establish that the original native gate is necessary for that decoder.

## 9. Quantitative Results

### 9.1 H128 native performance and matched effects

Unity phase1 native rows, three seeds [E15-1]:

| case | Train BA (%) | Validation BA (%) | Test BA (%) |
| --- | --- | --- | --- |
| B0 | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 |
| C0 | 95.28 ± 1.57 | 55.51 ± 1.91 | 57.87 ± 2.09 |
| GF | 92.98 ± 0.13 | 54.18 ± 2.00 | 57.72 ± 2.52 |
| GJ | 95.81 ± 2.21 | 56.76 ± 1.83 | 57.55 ± 3.29 |

GJ raises train BA by ≈0.54 pp and validation BA by ≈1.25 pp over C0, but test BA falls ≈0.32 pp. GF is essentially unchanged on train and loses≈0.21 pp test versus B0. GJ's stronger validation does not demonstrate held-out benefit; test users remain independent and were not used for selection.

### 9.2 H128 representation probes

No-bias L2 spike test BA (%), same three selected models per case [E15-2]:

| case | WholeCount | Fixed250 order | Fixed250 shuffle | Relative10 order | Relative10 shuffle |
| --- | --- | --- | --- | --- | --- |
| B0 | 57.57 ± 4.24 | 58.62 ± 2.69 | 53.79 ± 3.04 | 67.04 ± 0.73 | 54.24 ± 3.66 |
| C0 | 58.88 ± 3.33 | 59.36 ± 2.73 | 53.28 ± 1.62 | 68.07 ± 6.33 | 54.35 ± 1.96 |
| GF | 57.37 ± 4.47 | 58.19 ± 2.35 | 52.16 ± 2.09 | 66.20 ± 1.34 | 53.20 ± 4.32 |
| GJ | 59.02 ± 3.90 | 58.37 ± 2.16 | 52.84 ± 3.05 | 69.07 ± 5.41 | 54.14 ± 3.08 |

GJ WholeCount exceeds C0 by only≈0.14 pp and Relative10 by≈1.00 pp with large between-seed variability; Fixed250 ordered is lower. Relative10−WholeCount remains≈9.46 pp for B0, 9.19 for C0, 8.83 for GF, 10.05 for GJ. There is no general temporal-to-count consolidation effect. The probe table records class access, not proof of selective writing.

### 9.3 Capacity reduction and native generalization (Exp15.1)

Combined width table with immutable H128 references [E15-5]:

| width | case | Train BA (%) | Validation BA (%) | Test BA (%) |
| --- | --- | --- | --- | --- |
| 64 | B0 | 76.01 ± 0.77 | 47.58 ± 4.03 | 51.40 ± 1.11 |
| 64 | C0 | 84.55 ± 7.42 | 51.90 ± 1.80 | 54.21 ± 2.80 |
| 64 | GF | 76.05 ± 0.75 | 47.82 ± 4.42 | 51.18 ± 0.84 |
| 64 | GJ | 84.23 ± 7.62 | 52.00 ± 0.71 | 53.79 ± 2.28 |
| 96 | B0 | 88.05 ± 1.37 | 56.77 ± 5.76 | 53.54 ± 3.76 |
| 96 | C0 | 93.79 ± 4.38 | 59.98 ± 5.10 | 53.69 ± 4.15 |
| 96 | GF | 88.00 ± 1.43 | 57.08 ± 5.90 | 53.93 ± 4.43 |
| 96 | GJ | 90.88 ± 3.65 | 60.34 ± 3.87 | 53.87 ± 6.18 |
| 128 | B0 | 92.97 ± 0.12 | 53.70 ± 1.18 | 57.93 ± 2.23 |
| 128 | C0 | 95.28 ± 1.57 | 55.51 ± 1.91 | 57.87 ± 2.09 |
| 128 | GF | 92.98 ± 0.13 | 54.18 ± 2.00 | 57.72 ± 2.52 |
| 128 | GJ | 95.81 ± 2.21 | 56.76 ± 1.83 | 57.55 ± 3.29 |

Smaller width substantially reduces training fit (B0≈76.01% at H64,88.05% at H96,92.97% at H128), but held-out native BA is also lower. H64 continuation C0 improves its own B0 test by ≈2.82 pp; H96 by≈0.16 pp; H128≈−0.06 pp. Thus extra training can matter at H64, while it is not a sufficient explanation for the entire smaller-model generalization deficit, especially H96.

Matched gate effects, pp (three paired seeds), separate from cross-width differences:

| Width | Matched contrast | Train Δ (pp) | Validation Δ (pp) | Test Δ (pp) |
| --- | --- | --- | --- | --- |
| 64 | GF_minus_B0 | 0.05 ± 0.08 | 0.23 ± 0.40 | -0.21 ± 0.37 |
| 64 | GJ_minus_C0 | -0.32 ± 0.20 | 0.10 ± 1.41 | -0.42 ± 1.52 |
| 96 | GF_minus_B0 | -0.06 ± 0.19 | 0.31 ± 0.35 | 0.40 ± 0.69 |
| 96 | GJ_minus_C0 | -2.91 ± 6.14 | 0.36 ± 2.44 | 0.18 ± 2.04 |
| 128 | GF_minus_B0 | 0.01 ± 0.01 | 0.47 ± 0.82 | -0.21 ± 0.37 |
| 128 | GJ_minus_C0 | 0.54 ± 0.66 | 1.25 ± 0.07 | -0.32 ± 1.70 |

GF−B0 is≈−0.21 pp at H64, +0.40 at H96, −0.21 at H128. GJ−C0 is≈−0.42, +0.18, −0.32 pp. These small, variable effects do not establish that lower capacity makes the unchanged gate usefully selective. Reduced train-test gap alone is insufficient: H64 can reduce the gap by fitting less while also reducing held-out BA.

No-bias L2 spike probe results for every width/case:

| width | case | WholeCount | Fixed250 order | Fixed250 shuffle | Relative10 order | Relative10 shuffle |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | B0 | 55.03 ± 1.53 | 58.31 ± 0.19 | 49.54 ± 1.92 | 65.29 ± 3.15 | 46.35 ± 1.35 |
| 64 | C0 | 55.55 ± 4.86 | 57.64 ± 5.27 | 51.14 ± 2.21 | 64.06 ± 4.28 | 50.77 ± 3.04 |
| 64 | GF | 54.78 ± 1.38 | 58.44 ± 0.40 | 49.38 ± 1.73 | 65.29 ± 3.15 | 46.20 ± 1.30 |
| 64 | GJ | 55.40 ± 4.51 | 56.31 ± 4.80 | 51.17 ± 3.17 | 64.98 ± 3.43 | 49.41 ± 3.42 |
| 96 | B0 | 51.67 ± 3.57 | 59.34 ± 2.27 | 50.12 ± 1.86 | 63.70 ± 1.42 | 47.88 ± 0.92 |
| 96 | C0 | 53.74 ± 3.36 | 58.83 ± 3.31 | 51.74 ± 3.70 | 64.52 ± 2.37 | 50.28 ± 4.67 |
| 96 | GF | 52.67 ± 2.25 | 59.10 ± 3.03 | 50.18 ± 1.61 | 63.96 ± 2.42 | 46.59 ± 1.02 |
| 96 | GJ | 52.47 ± 6.13 | 58.27 ± 4.54 | 50.59 ± 2.70 | 63.76 ± 3.28 | 48.53 ± 3.68 |
| 128 | B0 | 57.57 ± 4.24 | 58.62 ± 2.69 | 53.79 ± 3.04 | 67.04 ± 0.73 | 54.24 ± 3.66 |
| 128 | C0 | 58.88 ± 3.33 | 59.36 ± 2.73 | 53.28 ± 1.62 | 68.07 ± 6.33 | 54.35 ± 1.96 |
| 128 | GF | 57.37 ± 4.47 | 58.19 ± 2.35 | 52.16 ± 2.09 | 66.20 ± 1.34 | 53.20 ± 4.32 |
| 128 | GJ | 59.02 ± 3.90 | 58.37 ± 2.16 | 52.84 ± 3.05 | 69.07 ± 5.41 | 54.14 ± 3.08 |

Relative10 remains appreciably above WholeCount across widths. Lower width changes diagnostic feature capacity but does not consistently eliminate the accumulated-readout gap or uncover a stable gate-specific transfer advantage.

### 9.4 The approximately70% observation

The approximately70% figure in follow-up discussion is **H96 user_3 BA**, not aggregate native test BA. Unity per-user values establish the distinction [E15-6]:

| H96 case | user_3 BA (%) | user_6 BA (%) | user_10 BA (%) |
| --- | --- | --- | --- |
| B0 | 68.89 ± 3.89 | 46.30 ± 13.80 | 37.50 ± 4.81 |
| C0 | 68.33 ± 1.00 | 47.69 ± 14.12 | 37.96 ± 3.21 |
| GF | 69.81 ± 5.04 | 46.30 ± 13.80 | 37.50 ± 4.81 |
| GJ | 70.23 ± 4.51 | 46.11 ± 14.36 | 37.04 ± 3.50 |

User_6 and user_10 are much lower. Overall BA is computed across class recall on the whole146-sequence test set, not an unweighted average of per-user BA. The headline H96 native mean remains≈53.54–53.93% across cases, not70%. This corrected the capacity interpretation in the October1 discussion.

## 10. Training / Mechanistic Diagnostics

### 10.1 Did the gate receive gradients?

On the fixed first training diagnostic batch, gate parameter gradients are nonzero at initialization and selected checkpoints [E15-3]. The table reports absolute gradient norms, mean±SD across seeds; GF's frozen L2/head gradients are zero as expected.

| Case | Diagnostic stage | Input gate gradient | History gate gradient | Gate bias gradient | L2 W gradient | Head gradient |
| --- | --- | --- | --- | --- | --- | --- |
| GF | Initialization | 0.00282 ± 0.00035 | 0.00660 ± 0.00175 | 0.00246 ± 0.00052 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| GF | Validation-selected checkpoint | 0.00236 ± 0.00107 | 0.00541 ± 0.00350 | 0.00211 ± 0.00107 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| GJ | Initialization | 0.00282 ± 0.00035 | 0.00660 ± 0.00175 | 0.00246 ± 0.00052 | 0.09727 ± 0.01057 | 0.14932 ± 0.02218 |
| GJ | Validation-selected checkpoint | 0.00066 ± 0.00026 | 0.00103 ± 0.00080 | 0.00082 ± 0.00033 | 0.08505 ± 0.02292 | 0.10094 ± 0.03853 |

The autograd node diagnostic also records absolute ∂L/∂aₜ in ten valid progress bins, showing time-resolved task credit. For example, GJ seed11 initialization ranges approximately4.2×10⁻⁷–2.0×10⁻⁶ over phase bins on this batch. These magnitudes depend on averaging, scale, and batch; they are not “zero gradient.” Conversely, a gradient's existence does not show that it identifies useful past content. Relative gradients divide by parameter norm plusε; zero-initialized gate vectors can yield enormous initial ratios, so those ratios are not evidence of unusually strong useful learning.

### 10.2 Gate behavior: selective suppression or amplification?

Test summaries for H64/H96/H128, including within-run timestep SD and history/input logit RMS ratio [E15-3/E15-5]:

| Width | Case | Mean g | Mean multiplier m | Within-run SD of m | History/input RMS ratio |
| --- | --- | --- | --- | --- | --- |
| 64 | GF | 0.9057 ± 0.0098 | 1.0063 ± 0.0109 | 0.0031 ± 0.0054 | 1.8967 ± 3.2852 |
| 64 | GJ | 0.9573 ± 0.0413 | 1.0636 ± 0.0459 | 0.0235 ± 0.0157 | 4.9343 ± 1.5288 |
| 96 | GF | 0.9248 ± 0.0145 | 1.0276 ± 0.0161 | 0.0141 ± 0.0079 | 4.7263 ± 0.7731 |
| 96 | GJ | 0.9650 ± 0.0166 | 1.0722 ± 0.0185 | 0.0315 ± 0.0050 | 4.8752 ± 0.4346 |
| 128 | GF | 0.9139 ± 0.0241 | 1.0154 ± 0.0267 | 0.0074 ± 0.0128 | 1.4613 ± 2.5310 |
| 128 | GJ | 0.9767 ± 0.0099 | 1.0852 ± 0.0110 | 0.0349 ± 0.0014 | 5.3459 ± 0.5689 |

H128 GJ mean g≈0.9767, mean m≈1.0852: the gate mostly **amplifies** writes relative to identity. Across selected GJ seeds, test g>.9 fraction is1.0 at every width. GF H128 g>.9 fraction averages1/3 because two seeds retain epoch0 identity; this is not selective closure. GJ H128 phase-bin mean m rises from≈1.015 in the first tenth to≈1.10 in later bins, approaching the allowed1.111 ceiling. History/input RMS ratio≈5.35 means a large history contribution to the gate logit, but it can drive a nearly open solution instead of useful selection.

### 10.3 Inference intervention results

The table reports **A0 BA minus intervention BA**, pp, positive meaning learned modulation outperforms that ablation. A3 first averages its five shuffles within seed; all cells then use three seeds [E15-4/E15-5].

| Width | Case | A0−A1 (pp) | A0−A2 | A0−A3 | A0−A4 | A0−A4b |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | GF | -0.21 ± 0.37 | -0.20 ± 0.34 | -0.08 ± 0.14 | -0.41 ± 0.71 | -0.20 ± 0.34 |
| 64 | GJ | -0.36 ± 0.87 | 0.00 ± 0.60 | 0.13 ± 0.09 | -0.02 ± 0.60 | 0.00 ± 0.60 |
| 96 | GF | 0.40 ± 0.69 | 0.20 ± 0.34 | 0.11 ± 0.51 | 0.20 ± 0.34 | 0.20 ± 0.34 |
| 96 | GJ | -0.23 ± 0.72 | 0.20 ± 0.34 | 0.07 ± 0.61 | -0.23 ± 0.72 | 0.40 ± 0.69 |
| 128 | GF | -0.21 ± 0.37 | 0.00 ± 0.00 | -0.09 ± 0.15 | -0.21 ± 0.37 | 0.00 ± 0.00 |
| 128 | GJ | 0.00 ± 0.14 | -0.38 ± 0.79 | -0.06 ± 0.17 | 0.28 ± 0.37 | -0.38 ± 0.94 |

AtH128 GJ, forcing identity differs by only≈+0.003 pp; sequence-mean gain outperforms learned timing by≈0.38 pp; temporal shuffling does not lower test BA; removing the gate history term loses≈0.28 pp, while averaging it improves≈0.38 pp. GF H128 often slightly improves when modulation is removed. Small H64/H96 effects are not stable across the different controls. Native interventions provide no robust evidence that precise gate timing is necessary for cross-user performance.

GJ H128 validation A0−A1 is≈+1.72 pp while the test difference is near zero. That disagreement reinforces the need to separate selected validation behavior from transfer. Joint-trained backbone/head adaptation remains part of the final model; A1 shows what those fixed adapted weights can do without modulation. It does not reconstruct the ungated training trajectory, which is why C0 is also required.

### 10.4 Epoch selection and the undertraining question

Historical H128 B0 best epochs are93/99/99, stopped100. C0 best/stopped continuation epochs are17/47,34/64,7/37; GF5/35,0/30,0/30; GJ22/52,68/98,8/38. The two GF epoch0 selections are valid outcomes, not failed learning tasks.

Smaller B0 best epochs areH64 97/95/97, H96 94/100/99, all stopped100. H64 C0 best/stopped are1/31,68/98,43/73; H96 C0 93/100,6/36,33/63. These late B0 selections prevent claiming complete convergence from a100-epoch cap. However, additional matched continuation is already available: H96's train/validation gains are much larger than its ≈0.16-pp test change, weakening “just train longer” as a sufficient transfer explanation. H64's≈2.82-pp continuation gain keeps partial optimization limitation plausible.

Continuation epoch numbers count from restored best B0 weights with a new optimizer. They cannot be interpreted as100 uninterrupted training epochs plus the reported index: a B0 selected at94 resumes from94's weights, and optimizer state is reset. On October1 00:29:59UTC the user chose not to pursue longer training. No unexecuted extra-duration run is presented as evidence against undertraining.

## 11. Negative and Null Results

Gate-only H128 training does not improve held-out native BA; two seeds choose the initial identity. Joint training improves train/validation but not test over ungated continuation. Gate values saturate toward amplification, and identity/mean/shuffle interventions preserve native test performance. History-logit magnitude does not establish useful historical selection. Width reduction lowers overfit indicators but also lowers native performance and does not consistently improve matched gate usefulness. No Prefix supervision was tested here, so its absence cannot be used as proof that Prefix would solve the problem.

## 12. Interpretation

### 12.1 Direct observations

All formal Exp15/15.1 runs complete. Matched gate-specific native effects are small and variable. Selected GJ writes are predominantly amplified/open; gate gradients exist, but timing/history interventions have weak or inconsistent effects on held-out BA. Smaller networks fit less and generalize less well in absolute native BA. Additional H64 training helps, whereas H96 continuation largely improves train/validation rather than test.

### 12.2 Mechanistic interpretation

The results are consistent with the gate acting mainly as a modest context-dependent gain in a jointly adaptable network, rather than learning a necessary policy for retaining only useful past input. Backbone/readout adaptation and ordinary continuation are viable explanations for apparent improvements. The gate's architecture allows selective suppression, but its q-normalization also permits easy gain increases, and WCCE does not charge for excessive writing.

The first September30 expectation was that a context gate could directly solve indiscriminate history writing. By21:34–22:14, the interpretation narrowed to history-dependent gain with weak evidence of useful timing. Width reduction at22:19 tested a competing capacity explanation. October1 capacity/per-user/epoch review further narrowed the claim: reduced capacity did not reveal a successful selective-memory mechanism, and the approximately70% value was not aggregate generalization. The hypothesis that selective writing could help remains open; this implementation/objective did not establish it.

### 12.3 What the experiment does not establish

It does not show that memory is useless, that gates cannot help, or that WCCE can never supply selective credit. Nonzero gradients are not useful-information labels. A saturated scalar is not evidence of selective storage. Ablation at a trained joint checkpoint isolates inference dependence, not the full causal contribution of gating during optimization. Smaller widths are not automatically better regularizers, and epoch caps do not prove convergence. Three seeds cannot resolve sub-pp effects with high confidence. Later suppress-only/z-only/budget results cannot be imported into Exp15.

## 13. Bugs, Corrections, and Reruns

| Issue / audit revision | Affected evidence | Repository correction | Scientific consequence |
| --- | --- | --- | --- |
| Gate task-credit output size | Per-timestep diagnostics | `72b69da` retains compact phase credit | Enables gradient audit; not an objective change |
| Explicit immutable B0 provenance | Phase1 baseline interpretation | `6d2ae66` records B0 checkpoint identity and evaluation | Prevents mistaking B0 for an extra continuation |
| Gate-bias optimizer grouping | Gated optimizer contract | `1c38153` gives bias explicit zero decay | Final optimizer aligns with native zero-decay contract; no inferred failed-run mixture |
| Matched causal delta accounting | GF/GJ comparison and inference interventions | `b579408` records paired contrasts and A0-minus-ablation deltas | Avoids attributing extra native training to gate benefit |
| H128 provenance in width study | Reference artifacts could be double-counted or drift | `0d0496e` hashes immutable Exp15 references | H128 remains exact reused result, not independent replication |
| Memory/chart hints about z-only and Prefix | Scientific narrative | Source gate inputs and WCCE override remembered aliases | Neither is an executed Exp15 objective/input restriction |
| Approximately70% generalization claim | Follow-up capacity interpretation | Per-user artifact and chat correction | It is user_3, while aggregate native BA remains≈51–58% |

Implementation began at `9247df7`, final aggregate integration at `ec341e7`; width implementation `852ed82`, artifact completion `967df8b`. These revisions do not establish failed Slurm training: accounting shows all formal tasks completed. No pre-fix and post-fix model runs are averaged without identity; final selected checkpoint hashes and immutable reference hashes determine the analyzed population.

## 14. Relationship to Previous Experiments

| Component | Inherited | Modified |
| --- | --- | --- |
| Data/users/seeds | Locked Core cache, exact hash and split | None in15/15.1 |
| H128 backbone/head | Historical Core O0 best checkpoint | Gate added; frozen or joint continuation |
| H64/H96 backbone | Same architecture/dynamics contract | Both widths reduced; new matched B0 training |
| Objective | WCCE | No Prefix, CU, budget, or reward |
| Readout | Signed bias-free native accumulated evidence | Unchanged; diagnostic probes trained separately |
| Checkpoint selection | Native validation BA then mean-logit CE, epoch0 allowed | New optimizer for continuation; exact best/stopped metadata |
| Gate | None in Core | Scalar currentL1/previousL2 gate, multiplier g/.9 |
| Reference identity | Exp15 H128 metrics and checkpoints | Reused in 15.1 through hashed manifest |

Exp14.1's fresh C0 **54.60 ± 5.69%** is not Exp15 B0 **57.93 ± 2.23%**. Exp13's source A2 **56.45 ± 3.28%** is also distinct. Shared historical Core O0 numbers agree across the three reports where they actually refer to the same checkpoint family.

## 15. How This Led to the Next Experiment

Two ambiguities remained: whether normalized amplification provided an easy substitute for selective writing, and whether WCCE offered the right learning incentive at the right training stage. Exp16, canonically [prefix-supervised selective memory](../../scripts/experiment_16_prefix_supervised_selective_memory.py), changes to a **true suppressive multiplier g∈(0,1)**, reconsiders fresh baseline/continuation stage selection, and screens equal50%/75% Prefix supervision under native and Relative10 preservation constraints. It includes matched no-gate/gate and gradient-routing cases. This follows the Exp15 negative evidence without claiming that simply adding Prefix guarantees success.

Retrospective October1 03:55 discussion noted that the first Exp16 Prefix selection could beλ=0, in which case that execution does not test a positive Prefix×gate interaction. A later [Exp16.1 z-only Prefix interaction](../../scripts/experiment_16_1_z_only_prefix_interaction.py) separately isolates currentL1-only versus additional history gate input; that restriction must not be backdated to Exp15. Subsequent “almost all-open” discussion motivated [matched-budget selective write](../../scripts/experiment_16_2_matched_budget_selective_write.py): compare learned timing against uniform writing at a matched budget so a favorable all-open/gain solution cannot explain an apparent selection benefit.

These are successor design facts and recovered motivation, not numerical Exp15 outcomes. The exact unresolved mechanism is **useful selection beyond uniform gain and native adaptation**. The next experiment must change incentives or restrict write freedom while preserving matched controls, rather than merely infer successful selection from a gate's presence.

## 16. Final Conclusions

Exp15/15.1 do not demonstrate robust useful selective memory writing. The gate is differentiable and history-conditioned, but predominantly open/amplifying, and matched native/intervention evidence does not show a stable transferable timing benefit. Width reduction reduces training fit without improving absolute held-out native BA or reliably strengthening gate-specific effects. The justified revision is from “a context gate may solve history selection” to “this gate/objective mainly permits gain and adaptation; selection requires a more discriminating test.”

## Appendix A. Run Inventory

### A.1 Every analyzed selected checkpoint

Each linked directory contains `checkpoint.pt`. Full SHA-256 and best/stopped epochs were read from Unity. Core rows are the three immutable H128 B0 sources. Exp15 rows are nine new continuations; Exp15.1 rows are six fresh smaller B0 plus18 continuations. H128 reuse is represented by its original rows, not duplicated as new15.1 training.

| Run directory / family | Seed | Best epoch | Stopped epoch | Final valid task or source | Scope / validity | Checkpoint SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| [15 / H128 / C0__seed11](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/C0__seed11) | 11 | 17 | 47 | 65092453_0 | Valid train/val/test | `1bb9c8dce42faca626c900ddbc178f6f12dd9ee0f49c3846fe129cf7d6afcdd6` |
| [15 / H128 / C0__seed23](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/C0__seed23) | 23 | 34 | 64 | 65092453_1 | Valid train/val/test | `11b08d12cd4de446c87ccf7ca2d0d659e43a5bb2aa1a64b0b92d64aa7cbae820` |
| [15 / H128 / C0__seed37](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/C0__seed37) | 37 | 7 | 37 | 65092453_2 | Valid train/val/test | `41dbfaa3821460e83f7e02e0ce0a037744a8bb45e51f95bf341917c7924da3d1` |
| [15 / H128 / GF__seed11](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GF__seed11) | 11 | 5 | 35 | 65092453_3 | Valid train/val/test | `7690c1120ca59f171683294a137b897cccc5062d9d69812dab2cf59437f3a1c5` |
| [15 / H128 / GF__seed23](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GF__seed23) | 23 | 0 | 30 | 65092453_4 | Valid train/val/test | `f1c30a962e950b42dd2022ce6476c7d85d62ec3d25fd8a40be46ea36dd42bc04` |
| [15 / H128 / GF__seed37](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GF__seed37) | 37 | 0 | 30 | 65092453_5 | Valid train/val/test | `f4e36290d7f55447fc8ad7d42c57c777444f7ca0822ad28c792b1a1fd54aa2d4` |
| [15 / H128 / GJ__seed11](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GJ__seed11) | 11 | 22 | 52 | 65092453_6 | Valid train/val/test | `584db5eefd38414ca64ee3053fe84a31936959c0237463a25bf91c2083cacb11` |
| [15 / H128 / GJ__seed23](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GJ__seed23) | 23 | 68 | 98 | 65092453_7 | Valid train/val/test | `cebd4a481bb9327277dfd21327086b706c938892aa99c4224c5c06189d5060c2` |
| [15 / H128 / GJ__seed37](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/runs/GJ__seed37) | 37 | 8 | 38 | 65092453_8 | Valid train/val/test | `3bb17e1d346524c45a828bdbdb021d821bd8fb557c1ad6c8d4b2f7bd35d12276` |
| [15.1 / H64 / B0__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/B0__seed11) | 11 | 97 | 100 | 65095881_0 | Valid train/val/test | `039342ff6a21cd7b0fb6eb16cd53da2d7b0aeab7b2d2dd4d0c6fd800b1deddea` |
| [15.1 / H64 / B0__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/B0__seed23) | 23 | 95 | 100 | 65095881_1 | Valid train/val/test | `6931689229fe58e62a908413257481ac7b2a11cf26fce37315f827c5a5f2b6b5` |
| [15.1 / H64 / B0__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/B0__seed37) | 37 | 97 | 100 | 65095881_2 | Valid train/val/test | `606b3c0b1a898748e4330ca96b09657986401cb1773ca26e2dd2cea452245831` |
| [15.1 / H64 / C0__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/C0__seed11) | 11 | 1 | 31 | 65095882_0 | Valid train/val/test | `ef936a84178ddd657292ca29fcd29eabf72160ec1ef54ebad7a0d920228e9f6e` |
| [15.1 / H64 / C0__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/C0__seed23) | 23 | 68 | 98 | 65095882_1 | Valid train/val/test | `a4a85d3362f282bdb1485c0e5fdb47fec8fab3e237a2c66cf57c7f86ea301568` |
| [15.1 / H64 / C0__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/C0__seed37) | 37 | 43 | 73 | 65095882_2 | Valid train/val/test | `28ba5f1900db8823d8815c32e5145def4d6b7495fd38f4e8dc0217f7d8ec499d` |
| [15.1 / H64 / GF__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GF__seed11) | 11 | 2 | 32 | 65095882_3 | Valid train/val/test | `69c4ebeea9d6c399cd18b30f730c6a302becb5b7a7baefbf37a1f801a1420063` |
| [15.1 / H64 / GF__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GF__seed23) | 23 | 0 | 30 | 65095882_4 | Valid train/val/test | `2a149ca82daae184c97af314514c7ebb08a0ed4226cb9990683953da5dbc6a84` |
| [15.1 / H64 / GF__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GF__seed37) | 37 | 0 | 30 | 65095882_5 | Valid train/val/test | `5b2022888b3e9b66314e6024404dd7b3bc8174bb62f50b4b874b958fa40f4a3b` |
| [15.1 / H64 / GJ__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GJ__seed11) | 11 | 1 | 31 | 65095882_6 | Valid train/val/test | `a34871022815d84fd1fe05c6d547d8ef4cf9a8d874d660a452d98bbed0694216` |
| [15.1 / H64 / GJ__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GJ__seed23) | 23 | 66 | 96 | 65095882_7 | Valid train/val/test | `e41a53099255b2f2e332e80c374227a58ec04a0ea1df98a8c03bf16e1f5d43cf` |
| [15.1 / H64 / GJ__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H64/GJ__seed37) | 37 | 40 | 70 | 65095882_8 | Valid train/val/test | `f4dbe1dcec970c505fbc8b4d330e7a6d7f1b48d80e82be646ae06b34aa35653b` |
| [15.1 / H96 / B0__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/B0__seed11) | 11 | 94 | 100 | 65095881_3 | Valid train/val/test | `ebbd9152bfbb92111bc9a3d16b9ddba227b90ad97f6a7b9d833cae0fd2649e39` |
| [15.1 / H96 / B0__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/B0__seed23) | 23 | 100 | 100 | 65095881_4 | Valid train/val/test | `a50c5444924625dcde30cc4efb81e1738f7aaf8d79cc25263f48ff0ae3a18155` |
| [15.1 / H96 / B0__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/B0__seed37) | 37 | 99 | 100 | 65095881_5 | Valid train/val/test | `33ad4a9f94f029ec332939ef9a6b7510e4410633b058f8a95f8e781f7116b92a` |
| [15.1 / H96 / C0__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/C0__seed11) | 11 | 93 | 100 | 65095882_9 | Valid train/val/test | `6f0d7f635e0f9bd5f629faf93d7ae850065aec16fc58753d185419d39fcd1b4c` |
| [15.1 / H96 / C0__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/C0__seed23) | 23 | 6 | 36 | 65095882_10 | Valid train/val/test | `77d74c5e9a93f58a0c5c1d71fffeae393bf9864d2570ce636b5acde323767080` |
| [15.1 / H96 / C0__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/C0__seed37) | 37 | 33 | 63 | 65095882_11 | Valid train/val/test | `04f64d3dff5e3fe23a681b39852bcf8d23c7895b839635c978bf063fa1ce47c4` |
| [15.1 / H96 / GF__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GF__seed11) | 11 | 4 | 34 | 65095882_12 | Valid train/val/test | `5f102e8bdf4e27ded79f65220160ddb62763320799d9ee21e46778e4222f726b` |
| [15.1 / H96 / GF__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GF__seed23) | 23 | 1 | 31 | 65095882_13 | Valid train/val/test | `ae5d1ee966067afe4661d5a91e67371fda0290eae2cbf0601629c9f3ad2483ae` |
| [15.1 / H96 / GF__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GF__seed37) | 37 | 1 | 31 | 65095882_14 | Valid train/val/test | `4fba55948aeed47d2e2879b365e550c6c7a840522fa6fe2cde3b44afc8dc0582` |
| [15.1 / H96 / GJ__seed11](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GJ__seed11) | 11 | 5 | 35 | 65095882_15 | Valid train/val/test | `ec876f0a032faae702daef8a96095f9117ea2ed676492443f13e0f399713955a` |
| [15.1 / H96 / GJ__seed23](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GJ__seed23) | 23 | 11 | 41 | 65095882_16 | Valid train/val/test | `34f5bb2b5a86c4b012dcc7bf8be6fb3b45101c9b390a516ea1d9539830bf2af5` |
| [15.1 / H96 / GJ__seed37](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/runs/H96/GJ__seed37) | 37 | 33 | 63 | 65095882_17 | Valid train/val/test | `85b537afea5dfdad68070071637ff7712672bc3f7d2566d94109cf14179da1ec` |
| Core O0 / H128 / O0__seed11: `core_benchmark_v1/results/main/runs/O0__seed11` (Unity) | 11 | 93 | 100 | Inherited; training job not recovered | Immutable B0; H128 reused | `5845f3ac7c707f111e98084f4e644fac990839b5d9792bb5236fe263e3cc203e` |
| Core O0 / H128 / O0__seed23: `core_benchmark_v1/results/main/runs/O0__seed23` (Unity) | 23 | 99 | 100 | Inherited; training job not recovered | Immutable B0; H128 reused | `c34ac6d8f1fe2e982ab1b904732f61251befa6e91370897c8eb9ed5a53529284` |
| Core O0 / H128 / O0__seed37: `core_benchmark_v1/results/main/runs/O0__seed37` (Unity) | 37 | 99 | 100 | Inherited; training job not recovered | Immutable B0; H128 reused | `382b3e1a34a7190c7c6dddbaf60353966a747d955ee299bdd5b026ef9fc226c0` |

### A.2 Scheduler accounting

| Job / array base | Name | Recorded tasks | States | Exit codes | First start (Unity local timestamp) | Last end |
| --- | --- | --- | --- | --- | --- | --- |
| 65092452 | exp15_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-30T20:49:01 | 2026-09-30T20:49:27 |
| 65092453 | exp15_p1 | 9 | COMPLETED: 9 | 0:0 | 2026-09-30T20:50:06 | 2026-09-30T20:58:24 |
| 65092454 | exp15_p15 | 6 | COMPLETED: 6 | 0:0 | 2026-09-30T20:58:46 | 2026-09-30T21:26:31 |
| 65092455 | exp15_finalize | 1 | COMPLETED: 1 | 0:0 | 2026-09-30T21:27:01 | 2026-09-30T21:27:25 |
| 65095880 | exp15_1_prepare | 1 | COMPLETED: 1 | 0:0 | 2026-09-30T22:30:57 | 2026-09-30T22:31:19 |
| 65095881 | exp15_1_b0 | 6 | COMPLETED: 6 | 0:0 | 2026-09-30T22:31:30 | 2026-09-30T22:39:01 |
| 65095882 | exp15_1_p1 | 18 | COMPLETED: 18 | 0:0 | 2026-09-30T22:39:37 | 2026-09-30T22:47:55 |
| 65095883 | exp15_1_p15 | 12 | COMPLETED: 12 | 0:0 | 2026-09-30T22:48:17 | 2026-09-30T23:09:30 |
| 65095884 | exp15_1_finalize | 1 | COMPLETED: 1 | 0:0 | 2026-09-30T23:09:59 | 2026-09-30T23:10:22 |

### A.3 Every task / intervention mapping

Let s index seeds[11,23,37], c index cases[C0,GF,GJ], g index gated cases[GF,GJ], w index newly trained widths[64,96]. Exp15 phase1 index=3c+s in job65092453; phase1.5 index=3g+s in65092454. Every phase1.5 task emits A0/A1/A2/A3/A4/A4b, with A3's five shuffle seeds101/211/307/401/503. Exp15.1 B0 index=3w+s in65095881, continuation index=9w+3c+s in65095882, intervention index=6w+3g+s in65095883. Parent run key is `<case>__seed<seed>` under `runs/H<width>/` for smaller models.

Intervention outputs are under `phase1_5/<parent key>/<intervention tag>/` in Exp15 and `phase1_5/H<width>/<parent key>/<intervention tag>/` in 15.1, as defined by `run_phase1_5`. Each manipulation uses the selected parent checkpoint and all train/val/test splits; A3 tag includes its shuffle seed. Evaluation/intervention tasks introduce no new model-training epochs or independent seeds. Original native checkpoints remain unchanged.

### A.4 Immutable reference and aggregate audit

Core identity and dataset SHA-256 are given in Section6. The width reference manifest hashes Exp15 `phase1_native_runs.csv` as `dd9654bc0a9e6a28dba09e42f9fd0b9fd5fb9eec4a8af2d06a5c3d22787a3e5b`, and `phase1_gate_summary.csv` as `fd3e25c0c9468f917d3542efb3c462cab2ddec7cbdd3c9fe74a89237412846ac`. Its Exp15 aggregate manifest hash is `f5978b9b575f36ff6ad5ecb703cd7ec29348f4fed7258da580d639f5060c2362`. These are artifact identity checks, not new statistics.

Native groups require exactly three model seeds. Width/case/intervention identities remain separate. Per-user BA is never substituted for pooled test BA. Probe shuffle/intervention replicates are reduced within seed before SD; H128 reuse does not increase n. Epoch0 selections stay valid and are retained. All formal Slurm tasks completed; no failed or synthetic replacement contributes to primary means.

## Appendix B. Configuration / Case Definitions

| Width | Native parameters | Gate parameters | Gated total | Fraction of H128 native |
| --- | --- | --- | --- | --- |
| 64 | 6784 | 129 | 6913 | 0.312 |
| 96 | 13248 | 193 | 13441 | 0.609 |
| 128 | 21760 | 257 | 22017 | 1.000 |

Native parameter count is P(H)=30H+H²+12H=H²+42H; the scalar gate adds 2H+1. H64/H96/H128 gated totals are 6913/13441/22017. τ buffers and state variables are not trainable parameters. GF trains only the 129/193/257 gate parameters; GJ trains the gated total. α groups retain the balanced split at each width (64:22/21/21;96:32/32/32;128:43/43/42).

| Diagnostic | Definition / interpretive limit |
| --- | --- |
| GF−B0 | Gate-only matched contrast; original weights fixed |
| GJ−C0 | Joint gate matched against ungated continuation; avoids extra-training confound |
| WholeCount / Fixed250 / Relative10 | Valid spike-sum features of dimensionsH/16H/10H, independent fitted readouts |
| Temporal order gap | Ordered BA minus within-seed averaged shuffled BA, pp; matched decoders refitted |
| Collapse gap | Ordered Relative10−WholeCount BA, pp; information accessible to probe, not native accuracy |
| Mean m | Mean over valid timesteps; m=g/.9 and values>1 amplify writes |
| History/input RMS ratio | RMS(wʰ·previousL2 spikes)/[RMS(wᶻ·currentL1 spikes)+ε]; magnitude not causal usefulness |
| Gate gradient | Absolute parameter-gradient norm on a fixed train batch; node credit is meanabs ∂L/∂aₜ in phase bins |
| A0−ablation | Matched selected-parent BA difference; no retraining of native weights |
| User-specific BA | Average class recall within one user; cannot replace whole-test BA |
| Continued epoch | Relative epoch after restoring selected B0 weights and resetting optimizer, not uninterrupted total training |

## Appendix C. Evidence Map

| ID / claim | Repository implementation / revision | Unity evidence | Project reasoning |
| --- | --- | --- | --- |
| E15-1: native and matched effects | Exp15 `train_one`, `_load_baseline`, optimizer and finalizer | [phase1_native_runs.csv](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/aggregate/phase1_native_runs.csv), `phase1_paired_contrasts.csv`, Core checkpoint metadata | Sep30 19:57 frozen/joint request;21:30–22:14 result analysis |
| E15-2: probe access / native distinction | [Core probes](../../core_benchmark_v1/probes.py), Exp15 `_run_l2_probes` | [phase1_probe_seed_means.csv](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/aggregate/phase1_probe_seed_means.csv), `phase1_probe_gains.csv` | Sep30 20:18 probe semantics |
| E15-3: actual gate / saturation / gradient credit | Exp15 `ContextWriteGateNet.forward`, `_gradient_diagnostic`, `_gate_summary` | `runs/GF or GJ__seed*/gate_gradient_diagnostics.json`, [gate summary](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/aggregate/phase1_gate_summary.csv), `phase1_gate_phase10.csv` | Sep30 20:06 gate protocol;21:34 history gain interpretation |
| E15-4: inference necessity weak | Exp15 `_intervention_forward`, phase1.5 finalizer | [phase1_5_native_deltas.csv](../../notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/aggregate/phase1_5_native_deltas.csv), native/probe seedmeans | Sep30 22:14 matched inference versus trained adaptation |
| E15-5: width does not rescue gate usefulness | Exp15.1 baseline/continuation specs; unchanged contract | [width_native_summary.csv](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/aggregate/width_native_summary.csv), `phase1_native_runs.csv`, `width_probe_summary.csv`, gate/intervention aggregates, reference manifest | Sep30 22:19 width-only decision;Oct1 capacity review |
| E15-6: approximately70% is per-user | Exp15.1 per-user evaluator; BA from class recall | [phase1_native_per_user.csv](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/aggregate/phase1_native_per_user.csv), native aggregate | Oct1 00:22–00:29 parameters/epochs/per-user correction |
| Core and immutable references | [Core model](../../core_benchmark_v1/model.py), locked protocol | `protocol.lock.json`, dataset hash, full checkpoint hashes AppendixA; [reference_manifest.json](../../notebooks/artifacts/experiment_15_1_width_capacity_ablation/width_capacity_v1/reference_manifest.json) | Matched comparisons required before interpretation |
| Implementation lineage | [9247df7](https://github.com/hellowPluto78700/writingRing/commit/9247df75b9f0e63c9d62e48a94d0612cb98d8484), [credit72b69da](https://github.com/hellowPluto78700/writingRing/commit/72b69da1d5aaf748e117667bb421a1b10c8bb55d7f), [aggregateec341e7](https://github.com/hellowPluto78700/writingRing/commit/ec341e7a503dcee394de5a167292b40cf2aeeff2), [B0 manifest6d2ae66](https://github.com/hellowPluto78700/writingRing/commit/6d2ae6689f9c546bde17f5deed630150eb222019), [optimizer1c38153](https://github.com/hellowPluto78700/writingRing/commit/1c381532c7ff05d7442b02c8334a61ac629c43ac), [causal deltab579408](https://github.com/hellowPluto78700/writingRing/commit/b5794081f540e1cb1134c2e9e0b0e7c11c640a35) | `sacct`, final manifests and selected checkpoints | Implementation details cross-checked rather than inferred from chat |
| Width lineage | [852ed82](https://github.com/hellowPluto78700/writingRing/commit/852ed824cfd19873f4df427d2678b9abe0888027), [967df8b](https://github.com/hellowPluto78700/writingRing/commit/967df8bf7b663aff3e3405f7a6df70fa4007da2e), [immutableH1280d0496e](https://github.com/hellowPluto78700/writingRing/commit/0d0496eef8770b3ba5872364548de72acc7b99b3) | Width reference hashes and 24 smaller checkpoint records | Capacity and duration considered separately |
| Successor design, no imported outcome | [Exp16](../../scripts/experiment_16_prefix_supervised_selective_memory.py), [Exp16.1](../../scripts/experiment_16_1_z_only_prefix_interaction.py), [Exp16.2](../../scripts/experiment_16_2_matched_budget_selective_write.py) | Source establishes suppress-only / prefix / z-only / budget boundaries; no successor metric enters Exp15 tables | Oct1 03:55,05:53–06:10 retrospective selection/incentive discussion |

Chat coverage audit: motivation before Exp15 (Sep30 history-content discussion), design/implementation (Sep30 frozen/joint and gate contract), result analysis (Sep30–Oct1), and transition (Oct1 Exp16/16.1/budget discussion) were all actively searched. Recovered topic titles include “历史特征机制解释”, “分析有用历史”, “exp15 related”, and “selective memory report”; individual timed search entries did not always expose a reliable title. Full Project transcripts and a complete historical log archive remain unavailable. Quantitative conclusions rely on Unity artifacts, not remembered assistant numbers.

## Appendix D. Chronological Reasoning Trace

| UTC date/time | Working hypothesis / decision | Evidence status and revision |
| --- | --- | --- |
| Sep30 19:49 | Prior stage through14.1 suggests content organization/selection, not lack of any memory | Motivation synthesized from predecessor evidence |
| Sep30 19:57 | User requires frozen and joint training, metrics beyond native; prefix deferred if needed | Implemented GF/GJ with paired B0/C0, WCCE only |
| Sep30 20:06–20:18 | Context gate uses currentL1/previousL2, q=.9 identity init; clarify probe semantics | Actual source/Unity protocol confirm both inputs and amplification-capable multiplier |
| Sep30 21:30–21:34 | User asks result analysis; first revision: gate mostly history-dependent gain, not useful selective memory | Native/probe/statistics evidence narrows initial hope |
| Sep30 22:14 | Evaluate matched causal effects and inference removals, avoid crediting all joint adaptation to gate | Intervention evidence is weak and test differs from validation |
| Sep30 22:19 | Accept width-only capacity test | Implemented15.1 with exact128 reuse, no new loss/gate confound |
| Oct1 00:22–00:25 | Review parameter counts, near-end selected epochs, and approximately70% value | Per-user evidence corrects70%; undertraining considered but not assumed |
| Oct1 00:29:59 | User declines longer training | No subsequent extra-duration experiment attributed to 15.1 |
| Oct1 03:55 | Retrospective Exp16 discussion: λ=0 selection cannot establish Prefix interaction; suppressive gate can remain open | Successor concern, not Exp15 outcome |
| Oct1 05:53–06:10 | Gate nearly open leads to budget and uniform-write controls | New test asks usefulness beyond all-open gain; later parameter proposals remain outside this report |

The revised scientific position preserves the negative gate/capacity evidence without concluding that selective writing is impossible. It specifies the missing discriminating comparison and why successor interventions follow.
