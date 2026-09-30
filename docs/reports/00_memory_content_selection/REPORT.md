# Report 00 — From Temporal Memory Capacity to Memory-Content Selection in the Multi-\tau SNN

**Stage:** intermediate research report  
**Evidence cutoff:** all results finalized through **Experiment 14.1**  
**Cutoff experiment:** experiment_14_1_dual_loader_accumulation, protocol dual_loader_accumulation_v1  
**Report date:** 2026-09-30  
**Status:** hypothesis-forming report; not a final paper claim

---

## 0. Scope and evidence cutoff

This report summarizes the current evidence about how the multi-\tau SNN represents and uses temporal history for cross-user handwriting classification. The report is intentionally frozen at the end of Experiment 14.1. Results from experiments implemented or executed after Exp14.1 must not be treated as evidence for any conclusion in this document unless this report is explicitly revised.

The central question is no longer simply whether the SNN has enough temporal memory. The accumulated evidence instead motivates a narrower question:

> **What historical information is written into the recurrent state, how is it organized, and how much of that information transfers to unseen users?**

The report distinguishes three levels of claim:

1. **Direct result:** measured by a finalized experiment artifact.
2. **Interpretation:** an explanation consistent with several direct results.
3. **Working hypothesis:** a mechanism that remains to be tested experimentally.

The proposed context-dependent write gate at the end of this report is therefore a **working hypothesis and next-step mechanism**, not a result included in the Exp14.1 evidence cutoff.

### 0.1 Direct quantitative experiments included

The following experiments and benchmark blocks contribute direct quantitative evidence to Report 00.

#### A. CoreBenchmark v1.1

Repository root:

- core_benchmark_v1/
- finalized production results under core_benchmark_v1/results/main/

The report uses the following CoreBenchmark evidence:

- **01_objective / O0 reference**
  - standardized two-layer WCCE backbone;
  - native train/validation/test Balanced Accuracy;
  - fixed cross-user split and seeds 11, 23, and 37.
- **02_tau**
  - systematic changes to the layer-wise multi-\tau synaptic shift assignments;
  - native BA and temporal-probe behavior;
  - used to test whether changing the memory horizon is sufficient.
- **03_depth**
  - two-layer versus three-layer depth control;
  - used to test whether adding a deeper historical layer improves generalization.
- **05_probes**
  - WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled;
  - spike and pre-reset states;
  - no-bias and affine decoder variants.
- **06_temporal_diagnostics**
  - lag similarity;
  - history reset with matched recent suffix;
  - prediction/state sensitivity to prior recurrent state.
- **07_membrane**, where needed for interpretation of state choice.
  - used only as supporting context; the main representation in this report remains communicated spike state unless stated otherwise.

CoreBenchmark is especially important because it reruns several older ideas under one locked user split, one data contract, and the same three model seeds. Older experiment numbers are not substituted for CoreBenchmark numbers when a standardized CoreBenchmark result exists.

#### B. Experiment 13 — Hierarchical Temporal Representation

Experiment identity:

- experiment_13_hierarchical_temporal_representation
- protocol hierarchical_temporal_representation_v2

Primary artifacts:

- notebooks/artifacts/experiment_13_hierarchical_temporal_representation/hierarchical_temporal_representation_v2/

This report uses Exp13 for the following questions:

- how representation similarity changes with temporal lag at L1/L2/L3;
- whether deeper representations become temporally stable or instead remain rapidly changing;
- whether matched recent input can produce different representation/prediction when earlier history is changed;
- whether L2/L3 behavior should be described as simple local feature extraction, temporal contextualization, or a stronger abstraction.

The key role of Exp13 in this report is to establish that deeper SNN states are strongly history-dependent and that deeper does not automatically mean slower, more stable, or more abstract.

#### C. Experiment 13.1 — Abstraction Generalization

Experiment identity:

- experiment_13_1_abstraction_generalization
- protocol abstraction_generalization_v1

Primary artifacts:

- F_cross_user_geometry/F_pair_summary.csv
- F_cross_user_geometry/F_retrieval_summary.csv
- G_user_leakage/G_user_leakage_summary.csv
- H_history_generalization/H_history_summary.csv

This report uses Exp13.1 as the main experiment for the distinction between **history usefulness** and **history transferability**.

The included analyses are:

- cross-user geometry;
- cross-user retrieval;
- user leakage;
- history-length generalization from 50 ms through longer windows and full history;
- L1/L2/L3 comparisons;
- C1 frozen-depth control and C2 end-to-end three-layer replication where relevant.

Exp13.1 supplies the strongest evidence that additional historical context can improve seen-user performance much more than unseen-user performance, particularly in L3.

#### D. Experiment 14 — History Organization

Experiment identity:

- experiment_14_history_organization
- protocol history_organization_v1

Primary aggregate artifacts:

- native_summary.csv
- cross_user_geometry.csv
- cross_user_retrieval.csv
- history_generalization.csv

Exp14 directly tests whether adding explicit auxiliary supervision for cross-user class/phase/history organization is sufficient to improve the WCCE backbone.

The report includes:

- C0 WCCE baseline;
- C1 whole-gesture cross-user supervision;
- C2 phase-aware cross-user supervision;
- C3 history-delta supervision;
- combined controls;
- shuffled-phase control;
- native BA, cross-user geometry/retrieval, and history-generalization diagnostics.

Exp14 is important because it tests a representation-supervision explanation before changing the recurrent update mechanism itself.

#### E. Experiment 14.1 — Dual-Loader Accumulation

Experiment identity:

- experiment_14_1_dual_loader_accumulation
- protocol dual_loader_accumulation_v1

Primary aggregate artifacts:

- native_runs.csv
- native_summary.csv
- probe_summary.csv
- collapse_gap.csv
- collapse_gap_summary.csv
- cross_user_retrieval.csv
- cross_user_geometry.csv
- history_generalization.csv
- evidence_organization_diagnostics.csv
- gradient_diagnostics.csv
- paired_delta_vs_c0.csv

Exp14.1 is the **final experiment included in this report**.

It isolates task supervision from structured auxiliary supervision with separate loaders, so that the WCCE task batch remains consistent with the CoreBenchmark training distribution while cross-user/phase structure is imposed through an auxiliary batch.

The report uses Exp14.1 to ask:

- does cleaner phase/cross-user supervision improve native WCCE accumulation?
- does it reduce the WholeCount-to-Relative10 collapse gap?
- does it reliably improve cross-user retrieval or geometry?
- does it change how useful full history is for ID versus OOD classification?
- can a representation-level auxiliary objective solve the accumulation problem without changing the state-update rule?

### 0.2 Background experiments referenced but not re-aggregated as primary evidence

Earlier experiments are used only to explain how the current questions arose. Examples include the Exp7.x WCCE/A2 family, earlier Fixed250/Relative10 probes, and depth/readout studies that motivated the standardized CoreBenchmark.

When an older result conflicts numerically with CoreBenchmark because of different user splits, seeds, affine probe choices, or implementation versions, this report uses the standardized CoreBenchmark result for quantitative comparison.

In particular:

- earlier reports that a single weight matrix is weak;
- earlier observations that Relative10 strongly benefits from phase-specific decoding;
- earlier A2/WCCE improvements;
- earlier L1/L2 probe results;

are treated as historical motivation, not as interchangeable estimates of the current locked protocol.

### 0.3 Explicitly excluded from Report 00

The following are not evidence in this report:

- any experiment after Exp14.1;
- any future context-gating experiment;
- any future LSTM/GRU comparison added after this cutoff;
- any later revised user split or probe protocol;
- any unfinalized Slurm run;
- any unpublished local result that is not represented by the finalized artifacts above.

The context-dependent write gate proposed in Section 10 is therefore a **prediction-generating mechanism**, not a completed result.

---

# 1. Executive summary

The current evidence supports a different diagnosis from the earlier question of whether the SNN simply needs a longer time constant.

Three results are now clear.

### Finding 1 — The SNN already uses substantial temporal history

Reset experiments show that the same recent sensory suffix can lead to different internal states and predictions depending on the preceding recurrent history. Therefore the current network cannot be accurately described as a purely local feature extractor.

A more appropriate description is:

\[
\boxed{
z_t^{(l)} = f_l(x_t, h_{t-1}^{(l)}, z_t^{(l-1)})
}
\]

with a substantial dependence on historical state.

### Finding 2 — Additional history is not uniformly transferable

Exp13.1 shows that increasing available history can help both seen-user and unseen-user classification over part of the history range, demonstrating that history does contain transferable class context.

However, at longer history and especially at L3, the seen-user gain can continue increasing while the OOD gain saturates or grows much less.

One representative C1 pre-reset result at phase 0.75 is:

\[
L2,\ full:
\quad
BA_{ID} \approx 69.3\%,\qquad
BA_{OOD} \approx 56.5\%
\]

versus

\[
L3,\ full:
\quad
BA_{ID} \approx 78.6\%,\qquad
BA_{OOD} \approx 57.5\%.
\]

Thus the L2-to-L3 change adds roughly 9.3 percentage points to ID BA but only about 1.0 point to OOD BA.

This supports:

\[
\boxed{
\text{more historical information}
\neq
\text{more transferable information}.
}
\]

### Finding 3 — Changing memory horizon or representation supervision alone is insufficient

The CoreBenchmark \tau sweep changes native test BA but does not produce a monotonic or fundamental improvement with a longer history profile. The standardized test BA remains roughly in the low-to-high 50% range across the tested configurations.

Exp14 and Exp14.1 further show that auxiliary cross-user/phase/history supervision can change representation geometry, but these changes do not reliably translate into improved native accumulation. In Exp14.1, the selected phase-CU condition does not consistently improve native BA across seeds, and the WholeCount-to-Relative10 collapse gap is not reduced.

The combined evidence therefore motivates the following working hypothesis:

\[
\boxed{
\begin{aligned}
&\text{The multi-}\tau\text{ SNN has substantial temporal memory,}\\
&\text{but the fixed recurrent update weakly constrains which}\\
&\text{historical content is retained for transferable classification.}
\end{aligned}
}
\]

The candidate bottleneck is therefore **memory-content selection and organization**, not simply memory capacity or memory horizon.

---

# 2. The current SNN as a dynamical memory system

A useful starting point is to describe the current backbone as a state-space system rather than a sequence of static feature mappings.

For a generic layer \(l\), a simplified synaptic-current and membrane update is:

\[
I_t^{(l)}
=
\alpha^{(l)} I_{t-1}^{(l)}
+
W^{(l)} s_t^{(l-1)}
\]

\[
V_t^{(l)}
=
\beta^{(l)} V_{t-1}^{(l)}
+
I_t^{(l)}
-
V_{\mathrm{th}} s_{t-1}^{(l)}
\]

\[
s_t^{(l)}
=
H\left(V_t^{(l)}-V_{\mathrm{th}}\right).
\]

The exact implementation contains multiple synaptic time constants inside a layer, but the central point is already visible from the linearized current recurrence.

Expanding the current state gives:

\[
I_t
=
W s_t
+
\alpha W s_{t-1}
+
\alpha^2 W s_{t-2}
+\cdots
=
\sum_{k=0}^{t}\alpha^k W s_{t-k}.
\]

Therefore each present state is a weighted superposition of current and past representations.

For the discrete decay used by the shift-based multi-\tau mechanism:

\[
\alpha = 1-2^{-q},
\]

and the equivalent continuous time constant is approximately

\[
\tau
=
-\frac{\Delta t}{\log \alpha}.
\]

At 64 Hz, commonly used shifts correspond approximately to:

- shift 2: 54.3 ms;
- shift 3: 117.0 ms;
- shift 4: 242.1 ms;

with longer shifts extending the history horizon further.

The important structural observation is:

\[
\boxed{
\tau
\text{ controls how strongly information of different ages persists.}
}
\]

It does not itself define which semantic content should be retained.

If the incoming representation decomposes conceptually as

\[
z_t
=
z_t^{class}
+
z_t^{user}
+
z_t^{timing}
+
z_t^{trajectory}
+
z_t^{noise},
\]

then a fixed leaky update retains all components according to the same temporal rule:

\[
h_t
=
\sum_j
\beta^{t-j}
W z_j^{class}
+
\sum_j
\beta^{t-j}
W z_j^{user}
+
\sum_j
\beta^{t-j}
W z_j^{timing}
+\cdots.
\]

Increasing \(\tau\) can therefore increase useful history and nuisance history simultaneously.

This distinction between **memory horizon** and **memory content policy** is the central mathematical distinction used throughout this report.

---

# 3. Evidence I — The network already uses substantial history

## 3.1 Why the reset intervention is stronger than a local probe

A conventional probe asks whether a classifier can decode information from a state. A reset intervention asks a more causal question.

Consider two executions that share the same recent sensory suffix:

\[
x_{t-D:t}^{full}
=
x_{t-D:t}^{reset}.
\]

In the full execution, recurrent state has accumulated from the beginning of the gesture. In the reset execution, earlier state is erased before the matched suffix is replayed.

If

\[
z_t^{full}
\neq
z_t^{reset}
\]

or

\[
\hat y_t^{full}
\neq
\hat y_t^{reset},
\]

the difference cannot be attributed to the recent input suffix. It must be mediated by the earlier recurrent state.

This does not directly measure the Jacobian

\[
\frac{\partial \hat y_t}{\partial h_t},
\]

so this report does not claim a numerical derivative from the reset result. The defensible statement is:

\[
\boxed{
\text{the prediction has substantial causal dependence on prior recurrent state.}
}
\]

## 3.2 CoreBenchmark reset evidence

CoreBenchmark history_reset.csv measures the effect of resetting individual layers or all layers while preserving a recent suffix.

A representative O0 seed-23 test example at a 500 ms suffix gives:

\[
BA_{\mathrm{suffix,full}}
\approx 47.9\%
\]

and, after resetting all recurrent layers:

\[
BA_{\mathrm{suffix,reset}}
\approx 25.4\%.
\]

The suffix-level drop is therefore about:

\[
22.5\text{ percentage points}.
\]

The same intervention also produces a large change in the class-score vector.

The exact magnitude varies with history length, seed, reset layer, and split, but the qualitative conclusion is robust: the current classifier is not making predictions from the local suffix alone.

## 3.3 Implication for the description of L2

This result changes how L2 should be described.

It is misleading to write:

\[
z_t^{L2}
\approx
\text{local motion feature only}.
\]

A more faithful description is:

\[
\boxed{
z_t^{L2}
=
\text{current motion representation conditioned on accumulated history}.
}
\]

This is temporal **contextualization**.

Whether that contextualization is a useful, user-invariant abstraction is a separate question, addressed by Exp13.1.

---

# 4. Evidence II — Explicit temporal order is partly internalized by L2

The standardized CoreBenchmark temporal probes provide a second route to the same conclusion.

Define:

\[
G_{order}
=
BA_{\mathrm{Fixed250,ordered}}
-
BA_{\mathrm{Fixed250,shuffled}}.
\]

A large \(G_{order}\) means an external decoder strongly benefits from preserving the explicit temporal order of fixed-duration bins.

For the Core O0 spike-state no-bias probe:

\[
G_{order}^{L1}
\approx
11.4\text{ pp}
\]

while

\[
G_{order}^{L2}
\approx
4.8\text{ pp}.
\]

The Relative10 ordered-versus-shuffled gap also falls strongly:

\[
G_{\mathrm{relative\ order}}^{L1}
\approx
36.2\text{ pp}
\]

versus

\[
G_{\mathrm{relative\ order}}^{L2}
\approx
12.8\text{ pp}.
\]

The important point is not that order stops mattering. It still matters. The point is that an external decoder needs **less additional explicit ordering information** at L2 than at L1.

One consistent interpretation is:

\[
\boxed{
\text{part of the temporal context that was externally supplied at L1
has already been folded into the L2 state.}
}
\]

Thus WCCE does not simply ignore sequence history.

Instead, deeper recurrent states already encode a history-conditioned representation.

This provides an important bridge to the generalization results:

\[
\boxed{
\text{temporal contextualization}
\neq
\text{transferable abstraction}.
}
\]

The network may successfully internalize temporal context while still organizing that context in a way that is overly user-, timing-, or trajectory-specific.

---

# 5. Evidence III — More history is not equivalent to more transferable information

Exp13.1 directly tests this distinction.

## 5.1 History-length generalization

For a representation \(z_t\), Exp13.1 varies the amount of accessible history and evaluates both seen-user and unseen-user classification.

Define:

\[
G_{ID}(H)
=
BA_{ID}(H)
-
BA_{ID}(50\text{ ms})
\]

and

\[
G_{OOD}(H)
=
BA_{OOD}(H)
-
BA_{OOD}(50\text{ ms}).
\]

The experiment also reports:

\[
G_{excess}(H)
=
G_{ID}(H)-G_{OOD}(H).
\]

This last quantity is useful because it asks whether additional history disproportionately helps the seen-user domain.

The broad pattern is:

1. increasing history from a very short context initially improves both ID and OOD performance;
2. therefore history does contain transferable class information;
3. at longer context, especially in deeper layers, ID gain can continue increasing more than OOD gain;
4. therefore not all newly accumulated history is equally transferable.

This is more informative than saying simply that the model overfits.

The relevant conclusion is:

\[
\boxed{
\text{the marginal utility of history depends on domain and depth.}
}
\]

## 5.2 L2 is a strong transfer point

Cross-user retrieval provides an independent view.

For the C1 spike state under fixed alignment:

\[
L1:
44.97\%
\]

\[
L2:
57.11\%
\]

\[
L3:
52.38\%.
\]

Under DTW alignment:

\[
L1:
37.55\%
\]

\[
L2:
56.00\%
\]

\[
L3:
51.13\%.
\]

The C2 replication shows the same broad pattern: a large L1-to-L2 improvement and no corresponding L2-to-L3 improvement.

Therefore:

\[
\boxed{
L2
\text{ is the strongest transferable representation among these tested depths.}
}
\]

This does not mean L2 is perfectly invariant. It means that the additional transformation from L1 to L2 improves cross-user class accessibility, whereas the additional transformation from L2 to L3 does not preserve the same transfer efficiency.

## 5.3 L3 demonstrates that more usable information can still transfer poorly

A representative C1 pre-reset, phase-0.75 history result is:

\[
L2,\ full:
\quad
BA_{ID}
=
69.26\%,
\quad
BA_{OOD}
=
56.49\%
\]

versus

\[
L3,\ full:
\quad
BA_{ID}
=
78.55\%,
\quad
BA_{OOD}
=
57.45\%.
\]

The L2-to-L3 gain is therefore approximately:

\[
\Delta BA_{ID}
=
+9.29\text{ pp}
\]

but only:

\[
\Delta BA_{OOD}
=
+0.96\text{ pp}.
\]

This is a particularly useful result because it rules out the trivial statement that L3 simply failed to learn anything.

L3 clearly learns information useful on the seen-user domain.

The problem is that the marginal information transfers inefficiently.

A useful descriptive quantity is therefore:

\[
\eta_{transfer}
=
\frac{\Delta BA_{OOD}}
{\Delta BA_{ID}}.
\]

For this representative comparison:

\[
\eta_{transfer}
\approx 0.10.
\]

This metric is not yet an official benchmark metric, but it captures the scientific point:

\[
\boxed{
\text{deeper / more historical}
\neq
\text{more transferable}.
}
\]

## 5.4 What the current evidence does not prove

These results do **not** prove that every piece of L3-only information is nuisance information.

Possible components include:

- class-relevant but poorly aligned history;
- user-specific motor trajectory;
- writing speed;
- amplitude;
- phase timing;
- user-specific transition statistics;
- useful information represented in a geometry poorly matched to the current readout.

The defensible claim is therefore:

\[
\boxed{
\text{the additional information in deeper/longer-history states
has lower cross-user transfer efficiency.}
}
\]

The stronger statement that all of it is nuisance remains a hypothesis.

---

# 6. Evidence IV — Changing the memory horizon is not sufficient

CoreBenchmark Block 02 tests multiple layer-wise multi-\tau assignments under one locked protocol.

The standardized native results are:

| Case | Train BA | Test BA | Train-test gap |
| --- | ---: | ---: | ---: |
| O0 reference | 92.97% | 57.93% | 35.04 pp |
| T1 | 89.09% | 56.22% | 32.87 pp |
| T2 | 84.14% | 58.17% | 25.97 pp |
| T3 | 84.47% | 52.93% | 31.54 pp |
| T4 | 93.19% | 54.83% | 38.36 pp |

The protocol defines the synaptic shift assignments as:

\[
T1:
((1,2,3),(2,3,4))
\]

\[
T2:
((1,2,3),(1,2,3))
\]

\[
T3:
((1,2,3),(3,4,5))
\]

\[
T4:
((2,3,4),(3,4,5)).
\]

The important empirical observation is not that \(\tau\) is irrelevant. Changing the time constants does change performance.

The important observation is:

\[
\boxed{
\text{there is no simple monotonic relation in which extending the
history profile fundamentally solves cross-user classification.}
}
\]

The best tested tau case is close to the reference, not a dramatic improvement.

This is consistent with the state equation.

If:

\[
z_j=z_j^{class}+z_j^{nuisance},
\]

then:

\[
h_t
=
\sum_j\beta^{t-j}Wz_j^{class}
+
\sum_j\beta^{t-j}Wz_j^{nuisance}.
\]

Changing \(\tau\) changes \(\beta\), and therefore changes the age weighting of **both** terms.

It does not directly implement:

\[
z^{class}\rightarrow retain
\]

and

\[
z^{nuisance}\rightarrow suppress.
\]

Thus:

\[
\boxed{
\text{memory horizon is a meaningful control parameter,
but the current evidence does not support it as the dominant bottleneck.}
}
\]

---

# 7. Depth is not equivalent to abstraction

CoreBenchmark depth and Exp13/13.1 together provide an important negative result.

If the hierarchy automatically behaved like a classical abstraction stack, one might expect:

\[
L1
\rightarrow
L2
\rightarrow
L3
\]

to become progressively:

- slower-changing;
- more class-centered;
- less user-specific;
- more transferable.

The measured pattern is not that simple.

Exp13 temporal diagnostics show that deeper states can change rapidly and remain strongly sensitive to historical context.

Exp13.1 retrieval shows:

\[
L1\rightarrow L2
\]

substantially improves cross-user class retrieval,

while:

\[
L2\rightarrow L3
\]

does not continue the same improvement.

The three-layer CoreBenchmark depth control also does not improve native test BA:

\[
D0\text{ (two-layer reference)}
\approx57.9\%
\]

versus

\[
D1\text{ (three-layer)}
\approx46.9\%.
\]

The exact three-layer control has its own training and optimization behavior, so it should not be interpreted as proof that a third layer is inherently harmful.

However, in combination with Exp13.1, it rejects the assumption:

\[
\boxed{
\text{more depth automatically produces more transferable abstraction.}
}
\]

A better description is:

\[
\boxed{
\text{depth increases representational transformation and historical context,
but transferability must be measured rather than assumed.}
}
\]

---

# 8. Evidence V — Representation supervision alone does not reliably solve native accumulation

## 8.1 Exp14

Exp14 introduces explicit auxiliary losses for history organization and cross-user structure while retaining the same underlying recurrent update.

The native test BA summary includes:

\[
C0_{\mathrm{WCCE}}
=
50.48\%\pm4.02\%.
\]

Moderate cross-user supervision produces small improvements in some conditions:

\[
C1_{\mathrm{whole-CU}},\ \lambda=0.03
\approx52.78\%
\]

and

\[
C2_{\mathrm{phase-CU}},\ \lambda=0.10
\approx52.49\%.
\]

However, stronger auxiliary weights can substantially reduce native performance. History-delta and combined variants are similarly mixed.

Therefore Exp14 does not support the simple statement:

> impose cross-user geometry and native WCCE performance will automatically improve.

It instead shows that auxiliary representation constraints interact nontrivially with the task objective.

## 8.2 Why Exp14.1 was needed

One confound in a structured auxiliary sampler is that changing the training batch composition may change the task optimization itself.

Exp14.1 therefore separates:

\[
B_{\mathrm{task}}
\]

for the standard WCCE objective from:

\[
B_{\mathrm{aux}}
\]

for structured phase/cross-user supervision.

This is a cleaner test of whether the auxiliary representation objective itself improves accumulation.

## 8.3 Exp14.1 native result

For C0 dual-null, the three test BAs are approximately:

\[
61.12\%,\ 52.11\%,\ 50.58\%.
\]

For the selected phase-CU condition:

\[
59.03\%,\ 53.05\%,\ 58.01\%.
\]

The effect is seed-dependent rather than uniformly positive.

The standardized collapse metrics are:

\[
C0:
\quad
BA_{\mathrm{WholeCount}}
=
55.39\%,
\quad
BA_{\mathrm{Relative10}}
=
66.49\%
\]

with a collapse gap of:

\[
11.11\text{ pp}.
\]

For the selected phase-CU condition:

\[
BA_{\mathrm{WholeCount}}
=
55.07\%,
\quad
BA_{\mathrm{Relative10}}
=
66.34\%
\]

with a collapse gap of:

\[
11.27\text{ pp}.
\]

Thus the auxiliary phase-CU objective does not reduce the key gap between temporally structured decoding and native accumulation.

## 8.4 Cross-user retrieval is also mixed

Exp14.1 does change retrieval in some seeds and states, but not monotonically.

For example, L2 spike retrieval increases for some seeds and decreases for others. Pre-reset retrieval shows the same general instability.

Therefore the correct summary is:

\[
\boxed{
\text{cross-user/phase auxiliary supervision modifies representation geometry,
but the improvement is neither uniform nor sufficient to improve native accumulation.}
}
\]

This is more informative than either extreme claim:

- “the auxiliary loss does nothing,” or
- “the representation is fixed and only the readout is wrong.”

The actual result is mixed and points toward a coupling problem between representation, historical state, and accumulation.

---

# 9. Reframing the WCCE result

The cumulative evidence changes how WCCE should be described.

A weak formulation would be:

\[
\text{WCCE does not use history well}.
\]

The reset and probe results contradict this.

A better formulation is:

\[
\boxed{
\text{WCCE successfully encourages the network to use history,
but weakly constrains what history should be stored and how that
history should be organized for cross-user classification.}
}
\]

WCCE optimizes:

\[
L_{\mathrm{WCCE}}
=
CE\left(
y,
\frac{1}{T}\sum_t Rz_t
\right).
\]

This directly constrains the final average class evidence.

It does not uniquely constrain the decomposition of recurrent state.

Many internal solutions can reduce the same training loss.

An idealized transferable state might emphasize:

\[
h_t
=
h_t^{class}
+
h_t^{discriminative\ transition}.
\]

A different state could also reduce training loss:

\[
h_t
=
h_t^{class}
+
h_t^{user}
+
h_t^{timing}
+
h_t^{velocity}
+
h_t^{amplitude}
+
h_t^{trajectory}.
\]

If user-specific variables are predictive within the training users, the optimizer has no reason under WCCE alone to reject them.

This gives a coherent explanation for the coexistence of:

\[
BA_{\mathrm{train}}
\gg
BA_{\mathrm{OOD}}
\]

and

\[
\text{strong history dependence}.
\]

The two observations are not contradictory.

They may share the same cause:

\[
\boxed{
\text{the model can learn substantial history without learning
a sufficiently transferable organization of that history.}
}
\]

The stronger statement “the network learned too much history” should be used cautiously. The issue is not necessarily the scalar amount of information. The issue is the **composition and organization** of the stored information.

---

# 10. Unified working hypothesis — memory-content selection

The current evidence motivates the following hypothesis.

\[
\boxed{
\begin{aligned}
&\textbf{The current multi-}\tau\textbf{ SNN has substantial temporal memory,}\\
&\textbf{but its fixed state-update rule mixes transferable class context}\\
&\textbf{with history whose cross-user transfer efficiency is lower.}
\end{aligned}
}
\]

This hypothesis distinguishes three questions.

### 10.1 Memory capacity

Can the network carry information from the past?

Current evidence:

\[
\boxed{\text{yes, clearly nontrivial}.}
\]

Reset experiments strongly support this.

### 10.2 Memory horizon

Does the network retain information long enough?

Current evidence:

\[
\boxed{\text{history horizon matters, but changing it alone is insufficient}.}
\]

The tau sweep does not produce a fundamental improvement.

### 10.3 Memory-content policy

Does the network selectively retain the historically useful information under the current context?

Current evidence:

\[
\boxed{\text{not directly controlled by the current fixed update}.}
\]

This is the remaining candidate bottleneck.

It is not yet proven to be the only bottleneck. Readout geometry, optimization, and state discretization can still contribute. But among the hypotheses tested through Exp14.1, memory-content selection is a direct unresolved mechanism.

---

# 11. Why a context-dependent write gate directly tests the hypothesis

The current simplified update is:

\[
h_t
=
\beta h_{t-1}
+
Wz_t.
\]

Every incoming feature is written according to the same fixed transformation and decay structure.

A context-dependent write gate would instead introduce:

\[
g_t
=
\sigma
\left(
W_g z_t
+
U_g h_{t-1}
+
b_g
\right)
\]

and

\[
h_t
=
\beta h_{t-1}
+
g_t\odot Wz_t.
\]

The scientific difference is not merely additional capacity.

The new model can ask:

> Given what is already stored, is this incoming information worth writing into memory?

The gate is therefore conditional on both:

\[
\text{current representation}
\]

and

\[
\text{existing context}.
\]

This matters because the same local pattern may have different discriminative value in different phases of a gesture.

For example, suppose a local primitive-like feature \(P_3\) appears.

Under one history:

\[
h_{t-1}
\approx
\text{“P1 and P2 have already occurred”}.
\]

Then:

\[
g_t(P_3,h_{t-1})
\approx1
\]

may be useful because the transition:

\[
P_1\rightarrow P_2\rightarrow P_3
\]

is discriminative.

Under another history:

\[
h_{s-1}
\approx
\text{repositioning / unrelated context},
\]

the same local feature could receive:

\[
g_s(P_3,h_{s-1})
\approx0.
\]

A gate based only on \(z_t\),

\[
g_t=\sigma(Gz_t),
\]

cannot express this difference as directly because it mainly learns whether a feature is generally worth storing.

The context-dependent gate instead implements a candidate **write policy**.

This is why it is more closely matched to the Exp13/14 evidence than simply increasing \(\tau\).

---

# 12. Falsifiable predictions for the next stage

A successful gate should not be judged only by native BA.

If the memory-content-selection hypothesis is correct, a successful mechanism should produce a recognizable pattern.

## 12.1 OOD classification

Desired:

\[
BA_{\mathrm{OOD}}
\uparrow.
\]

An increase in train BA alone would not support the hypothesis.

## 12.2 Generalization gap

Desired:

\[
BA_{\mathrm{train}}
-
BA_{\mathrm{OOD}}
\downarrow
\]

or, more conservatively, OOD should improve without a disproportionate train-only gain.

## 12.3 Exp13-style excess seen-history gain

Define:

\[
G_{excess}
=
G_{ID}-G_{OOD}.
\]

If the gate is suppressing poorly transferable history, the excessive full-history seen-user gain should decrease:

\[
G_{excess}(full)
\downarrow.
\]

This is one of the strongest mechanism-specific predictions.

## 12.4 Cross-user retrieval

Desired:

\[
\text{cross-user retrieval}
\uparrow
\]

or at minimum remain stable while native OOD improves.

A large native gain accompanied by severe loss of cross-user organization would suggest a different mechanism.

## 12.5 Native accumulation and collapse gap

The most direct deployment-facing target is:

\[
BA_{\mathrm{WholeCount}}
\uparrow.
\]

If Relative10 remains similar while WholeCount improves, then:

\[
\text{collapse gap}
=
BA_{\mathrm{Relative10}}
-
BA_{\mathrm{WholeCount}}
\]

should decrease.

This would directly address the failure mode that Exp14.1 did not solve.

## 12.6 Gate learning diagnostics

A negative BA result is not enough to conclude that the hypothesis is wrong. It is possible for the mechanism to be appropriate but poorly optimized.

At minimum, record:

\[
E[g_t]
\]

\[
Std(g_t)
\]

\[
P(g_t<0.1)
\]

\[
P(g_t>0.9)
\]

and the gradient magnitude to the pre-sigmoid gate activation \(a_t\):

\[
\left\|
\frac{\partial L}{\partial a_t}
\right\|.
\]

These quantities should also be examined against normalized gesture time:

\[
t/T.
\]

If:

\[
g_t\approx\text{constant}
\]

or:

\[
\nabla a_t\approx0,
\]

then a failure to improve BA would not constitute a clean test of the memory-selection hypothesis.

---

# 13. Scientific narrative emerging from Exp13 through Exp14.1

The current project stage can be summarized as the following evidence chain.

\[
\boxed{
\begin{array}{c}
\text{The SNN already uses substantial history}\\
\downarrow\\
\text{L2 partly internalizes temporal ordering/context}\\
\downarrow\\
\text{But contextualization is not the same as transferable abstraction}\\
\downarrow\\
\text{Additional/deeper history has lower marginal transfer efficiency}\\
\downarrow\\
\text{Changing }\tau\text{ changes horizon but does not solve the problem}\\
\downarrow\\
\text{Adding representation-level cross-user/phase supervision alone}\\
\text{does not reliably solve native accumulation}\\
\downarrow\\
\boxed{
\text{memory-content selection / organization remains a candidate bottleneck}
}
\end{array}
}
\]

This framing unifies several results that initially looked unrelated.

The tau sweeps, depth studies, order/shuffle probes, reset interventions, Exp13.1 history generalization, and Exp14/14.1 auxiliary losses can all be understood as testing different pieces of the same state-space question:

\[
\boxed{
\textbf{What should be written into recurrent memory under the current context?}
}
\]

The next stage should therefore test mechanisms that alter the **state-update policy itself**, rather than only extending temporal persistence or reshaping a downstream representation.

---

# 14. Claims supported at the Exp14.1 cutoff

The following claims are considered supported by the current evidence.

### Supported

1. The current multi-\tau SNN uses substantial history.
2. L2 representations contain more temporal context than a purely local-feature interpretation would imply.
3. Explicit temporal ordering becomes less necessary for an external decoder from L1 to L2, consistent with partial temporal-context internalization.
4. History contains transferable class information.
5. More history and more depth do not guarantee more transferable information.
6. L2 is a stronger cross-user retrieval representation than L1 and, in the tested Exp13.1 configurations, also stronger than L3.
7. Changing multi-\tau horizon alone does not produce a fundamental generalization improvement.
8. Three-layer/deeper representations can gain substantial seen-user utility without commensurate OOD gain.
9. Exp14/14.1 representation-level auxiliary supervision does not reliably close the native WholeCount versus temporally structured decoding gap.
10. Memory-content selection is therefore a well-motivated remaining mechanism to test.

### Not yet supported

The following statements should not yet be presented as established results.

1. All additional L3 history is nuisance.
2. User identity is the only source of non-transferable history.
3. A write gate will improve OOD performance.
4. WCCE is incapable of learning a useful write policy in principle.
5. Longer time constants are useless.
6. L3 is inherently harmful.
7. Cross-user auxiliary supervision is useless.
8. The current problem is exclusively a recurrent-state problem rather than partly a readout/optimization problem.

These remain open questions.

---

# 15. Source map for future updates

When Report 00 is revised or superseded, the following files should be rechecked first.

## CoreBenchmark

- core_benchmark_v1/protocol.py
- core_benchmark_v1/results/main/aggregate/02_tau/native_summary.csv
- core_benchmark_v1/results/main/aggregate/02_tau/probe_gains_summary.csv
- core_benchmark_v1/results/main/aggregate/03_depth/native_summary.csv
- core_benchmark_v1/results/main/aggregate/03_depth/probe_gains_summary.csv
- core_benchmark_v1/results/main/aggregate/probe_summary.csv
- core_benchmark_v1/results/main/aggregate/probe_gains_summary.csv
- core_benchmark_v1/results/main/aggregate/history_reset.csv
- core_benchmark_v1/results/main/aggregate/lag_similarity.csv

## Experiment 13

- notebooks/artifacts/experiment_13_hierarchical_temporal_representation/hierarchical_temporal_representation_v2/manifest.json
- notebooks/experiment_13_hierarchical_temporal_representation.ipynb
- scripts/experiment_13_hierarchical_temporal_representation.py

## Experiment 13.1

- notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/F_cross_user_geometry/F_pair_summary.csv
- notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/F_cross_user_geometry/F_retrieval_summary.csv
- notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/G_user_leakage/G_user_leakage_summary.csv
- notebooks/artifacts/experiment_13_1_abstraction_generalization/abstraction_generalization_v1/H_history_generalization/H_history_summary.csv

## Experiment 14

- notebooks/artifacts/experiment_14_history_organization/history_organization_v1/aggregate/native_summary.csv
- notebooks/artifacts/experiment_14_history_organization/history_organization_v1/aggregate/cross_user_geometry.csv
- notebooks/artifacts/experiment_14_history_organization/history_organization_v1/aggregate/cross_user_retrieval.csv
- notebooks/artifacts/experiment_14_history_organization/history_organization_v1/aggregate/history_generalization.csv

## Experiment 14.1

- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/native_runs.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/native_summary.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/probe_summary.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/collapse_gap_summary.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/cross_user_retrieval.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/cross_user_geometry.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/history_generalization.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/evidence_organization_diagnostics.csv
- notebooks/artifacts/experiment_14_1_dual_loader_accumulation/dual_loader_accumulation_v1/aggregate/gradient_diagnostics.csv

---

## Final stage conclusion

At the Exp14.1 cutoff, the evidence no longer points primarily to a lack of temporal memory.

The SNN already remembers, and it already transforms local input into a strongly history-conditioned state.

The unresolved issue is that the **utility of additional history decreases when evaluated by cross-user transfer**, and neither longer fixed decay nor representation-level auxiliary supervision has reliably corrected that behavior.

The working research question for the next stage is therefore:

\[
\boxed{
\textbf{Can the SNN learn a context-dependent policy for deciding
what information should be written into recurrent memory?}
}
\]

Report 00 ends at this hypothesis. Any experimental answer to that question belongs to a later report.
