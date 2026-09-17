# Exp8.0 Series Summary — Local Representation, Layer Complementarity, Phase-Conditioned Evidence, and RSNN Motivation

## 1. Purpose

The Exp8.0 series was designed to answer a sequence of increasingly specific questions about the two-layer local SNN backbone:

1. **What temporal constants should L1 and L2 use?**
2. **Do L1 and L2 contain complementary information?**
3. **Can direct supervision make that complementarity more useful?**
4. **Does temporal phase change the class meaning of local evidence?**
5. **Is any apparent phase gain caused by extra parameters, end-to-end co-adaptation, or by correct temporal alignment itself?**
6. **What does this imply for the next recurrent/context SNN stage?**

The current high-level conclusion is:

> **The 234x234 local SNN is a strong local backbone. L2 is more class-oriented and sequence-integrated, while L1 preserves richer temporally localized information. A large fraction of L1's useful information is not well exposed by a time-shared whole-sequence linear head, but becomes useful when the readout knows the correct temporal phase. This motivates a recurrent/context module whose role is to infer history/phase and reinterpret local L1 evidence, while L2 can remain a more integrated evidence pathway.**

The most important distinction throughout the series is:

```text
representation quality
        !=
readout alignment
        !=
phase/context access
        !=
output-LIF realization quality
```

---

## 2. Common setup and notation

The relevant Exp8.0 experiments use the WritingRing 64 Hz spike representation and a two-layer feed-forward SNN backbone:

```text
30 spike-event channels
    -> L1: 128 binary spiking neurons
    -> L2: 128 binary spiking neurons
    -> Linear / phase-conditioned readout
    -> 12 classes
```

The standard seeds are:

```text
11, 23, 37
```

At 64 Hz, a 250 ms temporal bin is 16 timesteps. For a padded 256-step sequence, Fixed250 therefore produces up to 16 ordered bins.

Let

\[
z_t^{(1)}, z_t^{(2)} \in \{0,1\}^{128}
\]

be L1 and L2 spike vectors.

Useful representations are:

\[
\mathrm{Whole}(Lk)=\sum_{t<T}z_t^{(k)},
\]

and

\[
\mathrm{Fixed250}_b(Lk)
=\sum_{t\in b,\,t<T}z_t^{(k)}.
\]

For readout training, Exp8.0.4 and Exp8.0.5 use valid-length normalization:

\[
\bar z = \frac{1}{T}\sum_{t<T} z_t.
\]

A terminology caution used below:

- **native Linear / native BA** means the analog Linear/readout path used by that experiment;
- **LIF BA** means the same evidence is passed through the output LIF realization and classified by output spike count;
- in **Exp8.0.5**, the backbone is frozen, so the reported 60.79% is the frozen-backbone + newly trained Linear readout BA, not a newly end-to-end-trained SNN backbone.

---

## 3. Exp8.0 — Local-backbone tau sweep

### 3.1 Question

The first question was whether local representation should be improved by changing the time constants of L1 and L2.

Architectures were denoted by shift groups:

```text
234x234
123x234
123x123
123x345
234x345
```

where the first group is L1 and the second group is L2.

### 3.2 Main results

| Architecture | Native Linear test BA | LIF test BA | L1 whole probe | L1 Fixed250 probe | L2 whole probe | L2 Fixed250 probe |
|---|---:|---:|---:|---:|---:|---:|
| `234x234` | **56.25%** | **52.30%** | **58.65%** | 54.21% | 56.93% | 58.69% |
| `123x234` | 55.40% | 51.52% | 53.78% | 51.81% | 56.58% | **59.40%** |
| `123x123` | 55.77% | 48.54% | 55.13% | 52.79% | 56.43% | 58.98% |
| `123x345` | 52.52% | 48.01% | 50.92% | 51.32% | 51.56% | 55.20% |
| `234x345` | 52.56% | 47.48% | 52.55% | 53.67% | 55.79% | 54.57% |

### 3.3 Interpretation

The main result is that simply making deeper layers slower did not improve the local backbone.

In particular:

\[
\boxed{234\times234}
\]

remained the strongest overall architecture.

Moving L2 toward the longer `345` group degraded native classification and generally weakened the representation.

The very short shift-1 population was often weakly active, but this result is confounded by the neuron's gain/DC response and should not be interpreted as definitive proof that the shortest physical timescale is intrinsically useless.

The practical conclusion was:

> **Do not try to solve the long-context problem by continuously increasing local L1/L2 tau. Keep the local backbone local; handle longer context separately.**

This result becomes important later when motivating a dedicated context/RSNN stage.

---

## 4. Exp8.0.1 — Frozen L1/L2 fusion probes

### 4.1 Question

Exp8.0.1 froze the strongest `234x234` backbone from Exp8.0 and asked:

> Are L1 and L2 redundant, or do they contain complementary information that a downstream classifier can exploit?

Eight frozen representations were tested.

### 4.2 Probe results

| Frozen feature | Test BA |
|---|---:|
| `l1_fixed250` | 54.01% |
| `l2_fixed250` | 58.94% |
| `l1_l2_fixed250` | 58.65% |
| `l1_whole` | 58.65% |
| `l2_whole` | 57.33% |
| `l1_l2_whole` | **60.28%** |
| `l1whole_l2fixed250` | 58.89% |
| `l1fixed250_l2whole` | **59.70%** |

### 4.3 Interpretation

Several important points emerged.

First, L1 and L2 prediction errors were not identical. Their information was therefore not fully redundant.

Second, naive concatenation did not automatically improve performance. Complementarity existed, but it was not always linearly accessible under every aggregation.

Third, the mixed feature

\[
\mathrm{Fixed250}(L1)+\mathrm{Whole}(L2)
\]

was already competitive and consistently suggested a useful asymmetry:

- L1 may preserve more local/temporally resolved detail;
- L2 may be more useful as an integrated/global feature.

This motivated direct supervision of both layers instead of treating L1 only as an intermediate computational state.

---

## 5. Exp8.0.2 — Joint L1/L2 supervision

### 5.1 Question

Exp8.0.2 asked whether L1/L2 complementarity could be made more useful during end-to-end training.

Three objectives were compared on the same `234x234` backbone.

### A. L2-only

\[
\mathcal L
=
CE(W_2\bar z_2,y).
\]

### B. Cooperative joint L1+L2

\[
\mathcal L
=
CE(W_1\bar z_1+W_2\bar z_2,y).
\]

### C. L2 main + independent L1 auxiliary

\[
\mathcal L
=
CE(W_2\bar z_2,y)
+0.1\,CE(W_1\bar z_1,y).
\]

The important conceptual distinction is that the joint objective asks L1 and L2 to **cooperate on one decision**, whereas the auxiliary objective asks L1 to become an independently correct classifier.

### 5.2 Native results

| Method | Native test BA | LIF test BA |
|---|---:|---:|
| L2-only | 56.19% | 51.84% |
| **L1+L2 joint** | **57.80%** | 50.46% |
| L2 main + L1 aux | 52.90% | 49.07% |

The native gain from cooperative joint supervision was modest:

\[
57.80-56.19
\approx
\boxed{+1.61\text{ pp}}.
\]

### 5.3 Frozen probe results

The representation improvement was much larger than the native-head improvement.

For the joint model:

| Feature | Test BA |
|---|---:|
| L1 Fixed250 | 59.74% |
| L2 Fixed250 | 60.94% |
| L1+L2 Fixed250 | 63.49% |
| L1 whole | 61.01% |
| L2 whole | 58.55% |
| **L1 Fixed250 + L2 whole** | **64.21%** |

Compared with L2-only training, joint supervision improved L1 Fixed250 by about 7.79 pp and the reverse mixed representation by about 6.20 pp.

### 5.4 Interpretation

This was one of the most important representation results in the series.

The conclusion is not merely that "adding an L1 loss helps." The stronger interpretation is:

> **Cooperative joint supervision changes the geometry of both L1 and L2 so that their complementary information becomes much more linearly accessible.**

The independent auxiliary objective did not reproduce this behavior and was substantially weaker.

Thus:

```text
cooperative evidence sharing
        !=
forcing every layer to classify independently
```

The gap between native joint BA (~57.8%) and the best frozen probe (~64.2%) also showed that readout design, not just backbone capacity, remained a major bottleneck.

---

## 6. Exp8.0.3 — Phase-aware hierarchical readout, first implementation

### 6.1 Motivation

The strongest frozen probe from Exp8.0.2 was approximately:

\[
\boxed{
\mathrm{Fixed250}(L1)+\mathrm{Whole}(L2)
}
\]

which suggested a hierarchy:

```text
L1 -> local / temporally resolved information
L2 -> whole-sequence / more integrated information
```

Exp8.0.3 tried to make this structure native and end-to-end trainable.

The phase-aware score can be written as

\[
s
=
\sum_b W_{1,b}c_b^{(1)}
+
W_2 Z_2,
\]

or per timestep,

\[
e_t
=
W_{1,b(t)}z_t^{(1)}
+
W_2z_t^{(2)}.
\]

### 6.2 Methods

Four methods were compared:

1. `l2_only_count`
2. `l1_l2_timeshared_count`
3. `l1_fixed250_l2_whole_count` — true phase-aware readout
4. `l1_capacity_no_phase_l2_whole_count` — parameter-count control without true phase access

### 6.3 Initial result

Native test BA was poor across all methods:

| Method | Test BA |
|---|---:|
| L2-only | 37.81% |
| L1+L2 time-shared | 46.77% |
| Phase-aware | **48.53%** |
| No-phase capacity control | 43.21% |

The phase-aware model exceeded the no-phase control by about 5.31 pp, with the paired difference positive in all seeds, so phase conditioning appeared promising.

However, the absolute accuracy collapse exposed a more fundamental problem.

### 6.4 Critical methodological finding: raw count CE is not equivalent to mean CE during training

Exp8.0.3 trained with

\[
CE\left(\sum_t e_t,y\right),
\]

whereas the previous strong backbone used a valid-length mean:

\[
CE\left(\frac1T\sum_t e_t,y\right).
\]

For frozen inference,

\[
\arg\max_k \sum_t e_{t,k}
=
\arg\max_k \frac1T\sum_t e_{t,k},
\]

because the two logits differ only by a positive scalar for one sample.

But cross-entropy is not scale invariant:

\[
CE(T\bar e,y)
\neq
CE(\bar e,y).
\]

The raw-count form effectively changes the softmax temperature and scales gradients with sequence length.

This produced a severe backbone-training pathology. The clearest control was the L2-only case:

```text
Exp8.0.2 / mean-style baseline: ~56.19% test BA
Exp8.0.3 / raw-count CE:       ~37.81% test BA
```

Thus Exp8.0.3 could not be used as the final causal test for the phase-aware architecture until the normalization issue was fixed.

The important lesson is:

\[
\boxed{
\text{count and mean may be argmax-equivalent at frozen inference,}
\\
\text{but they are not equivalent training objectives under CE.}
}
\]

---

## 7. Exp8.0.4 — Mean-normalized phase-aware hierarchical readout

### 7.1 Question

Exp8.0.4 changed one main variable from Exp8.0.3:

\[
CE\left(\sum_t e_t,y\right)
\rightarrow
CE\left(\frac1T\sum_t e_t,y\right).
\]

Everything else was kept as close as possible.

### 7.2 Recovery of the backbone

The normalization fix restored performance:

| Method | Exp8.0.3 | Exp8.0.4 | Recovery |
|---|---:|---:|---:|
| L2-only | 37.81% | 56.45% | +18.64 pp |
| L1+L2 time-shared | 46.77% | 57.53% | +10.76 pp |
| Phase-aware | 48.53% | **58.63%** | +10.10 pp |
| No-phase capacity control | 43.21% | 57.91% | +14.69 pp |

This confirmed that the raw-count objective had been the dominant pathology in Exp8.0.3.

### 7.3 Phase-aware native gain

With valid-mean CE:

\[
\text{Phase-aware}=58.63\%,
\]

\[
\text{No-phase}=57.91\%,
\]

so the paired phase-aware gain was only about:

\[
\boxed{+0.72\text{ pp}}.
\]

It was positive for all three seeds, but small.

Phase-aware also exceeded time-shared by about 1.10 pp on average, but that difference was not stable across all seeds.

### 7.4 Phase-bank structure

The learned phase bank was clearly not collapsing to one shared matrix.

For the phase-aware method, mean pairwise cosine similarity between phase matrices was approximately 0.21 and adjacent-bin cosine similarity approximately 0.45.

For the no-phase control, the bank matrices were almost identical, with pairwise cosine around 0.99.

Thus the phase-aware model really learned distinct phase-dependent directions.

### 7.5 Native classifier quality and reusable representation diverged

This was a crucial result.

The phase-aware model had the strongest native classifier, but its frozen representation was not the strongest reusable representation.

For example, the frozen target probe

\[
\mathrm{Fixed250}(L1)+\mathrm{Whole}(L2)
\]

reached approximately:

```text
phase-aware C: 60.61%
time-shared B: 63.91%
no-phase D:    64.52%
```

Therefore:

\[
\boxed{
\text{best native classifier} \neq \text{best reusable frozen representation}
}
\]

The phase-aware end-to-end head was partially co-adapting the backbone to itself.

This motivated the next experiment: freeze a strong backbone first, then test phase access only at the readout.

### 7.6 Separate observation: output-LIF compatibility became worse

The output-LIF penalty was especially large for the phase-aware readout:

```text
L2-only:      56.45 -> 52.47  (-3.98 pp)
time-shared:  57.53 -> 51.33  (-6.21 pp)
no-phase:     57.91 -> 49.01  (-8.90 pp)
phase-aware:  58.63 -> 41.93  (-16.70 pp)
```

The likely mechanism is that a phase-aware analog Linear readout can use signed, small-magnitude, phase-specific evidence that a leaky binary output LIF cannot faithfully preserve under thresholding, reset, nonnegative spike counts, and `beta=0.5` leakage.

This is a separate realization problem. It should not be confused with whether the hidden representation contains useful phase-conditioned information.

---

## 8. Exp8.0.5 — Frozen-backbone phase readout

### 8.1 Goal

Exp8.0.5 was designed to remove backbone/head co-adaptation completely.

For each seed, it reuses the strong Exp8.0.4 `l1_l2_timeshared_count` `234x234` checkpoint and freezes L1 and L2.

Only a new bias-free Linear readout is trained.

The question is:

\[
\boxed{
\text{Does an already-good frozen local SNN contain information whose class meaning depends on temporal phase?}
}
\]

### 8.2 Readouts

#### A. L2 whole

\[
s=W_2\frac{\mathrm{Whole}(L2)}T.
\]

#### B. L1+L2 time-shared

\[
s=
W_1\frac{\mathrm{Whole}(L1)}T
+
W_2\frac{\mathrm{Whole}(L2)}T.
\]

#### C. Correct phase-aware L1 + L2 whole

\[
s=
\sum_b
W_{1,b}
\frac{\mathrm{Fixed250}_b(L1)}T
+
W_2\frac{\mathrm{Whole}(L2)}T.
\]

#### D. Destroyed-phase control

D uses the same `16 x 128` L1 phase feature bank and the same total head dimensionality as C, but every sample receives a deterministic non-zero cyclic phase offset:

\[
b'(t)=(b(t)+r_i)\bmod B.
\]

Thus C and D have the same effective input dimension and readout parameter count; the main difference is whether absolute temporal phase is aligned correctly across sequences.

### 8.3 Main result

| Frozen-backbone readout | Test BA |
|---|---:|
| L2 whole | 55.57% |
| L1+L2 time-shared | 56.42% |
| **L1 Fixed250 true phase + L2 whole** | **60.79%** |
| Destroyed phase + L2 whole | 53.46% |

The three most important paired differences are:

\[
\text{Time-shared}-L2
=
\boxed{+0.85\text{ pp}},
\]

\[
\text{True phase}-\text{Time-shared}
=
\boxed{+4.38\text{ pp}},
\]

\[
\text{True phase}-\text{Destroyed phase}
=
\boxed{+7.33\text{ pp}}.
\]

All three seeds were positive for both of the phase comparisons.

The true-phase test BA per seed was approximately:

```text
seed 11: 57.17%
seed 23: 64.18%
seed 37: 61.03%
mean:    60.79% +/- 3.51%
```

### 8.4 Why this is stronger evidence than Exp8.0.4

The result cannot be explained by end-to-end co-adaptation because the backbone is frozen.

It also cannot be explained simply by "more phase-bank parameters" because the destroyed-phase control has the same 2176-D feature input and the same 26112 Linear parameters.

The destroyed-phase model can fit the training set well, but its test BA is much lower.

Therefore the strongest interpretation is:

> **The local SNN already contains temporally localized information, and its class meaning generalizes only when that information is aligned with the correct sequence phase.**

### 8.5 Branch removal diagnostics

For the time-shared model:

```text
full:        56.42%
L2-only:     53.94%
L1-only:     20.32%
remove L1:   -2.48 pp
```

For the true-phase model:

```text
full:        60.79%
L2-only:     54.78%
L1-only:     31.71%
remove L1:   -6.01 pp
```

For the destroyed-phase model:

```text
full:        53.46%
L2-only:     55.11%
L1-only:     16.83%
remove L1:   +1.65 pp
```

The last case is especially informative: incorrectly aligned L1 evidence is not merely useless; on average it harms the joint classifier.

These removal values are diagnostic drops, not additive attribution percentages. L1/L2 interactions mean they should not be interpreted as a strict decomposition of accuracy.

### 8.6 Circular phase-shift diagnostic

The trained true-phase head was frozen and the test-time L1 phase bins were circularly shifted without retraining.

Mean test BA was approximately:

| Shift | Test BA |
|---|---:|
| 0 ms | **60.79%** |
| 250 ms | 59.76% |
| 500 ms | 54.73% |
| 750 ms | 42.97% |
| 1000 ms | 37.01% |
| 1250 ms | 36.95% |
| 1500 ms | 35.94% |
| 1750 ms | 36.46% |

The large collapse under wrong alignment demonstrates that the learned phase weights are functionally phase-specific, not merely extra classifier capacity.

One nuance is that a one-bin shift is relatively well tolerated, and one seed even slightly improved at +250 ms. Therefore the current result supports a **coarse temporal phase/context** interpretation more strongly than an exact 250 ms clock interpretation.

Because the diagnostic is circular and sequence valid lengths vary, later large shifts should not be interpreted as a strictly monotonic distance curve.

---

## 9. Integrated interpretation of L1 and L2

### 9.1 What can currently be said about L2

Under whole-sequence CE/WCCE-style supervision, L2 is directly optimized so that its aggregate representation is class-discriminative:

\[
\mathcal L
=
CE(W\bar z_2,y)
\]

or, in the joint case,

\[
\mathcal L
=
CE(W_1\bar z_1+W_2\bar z_2,y).
\]

Because L2 also receives the temporally filtered output of L1, its state already contains some temporal integration.

A useful working description is:

\[
\boxed{
L2:\ \text{more temporally integrated and class-oriented representation}
}
\]

However, WCCE itself is an order-insensitive final aggregation. Therefore it is too strong to claim that WCCE explicitly teaches L2 to encode temporal order. A safer statement is that L2 is shaped into a sequence-level representation while its internal SNN dynamics can carry some history into that representation.

### 9.2 What can currently be said about L1

Exp8.0.5 shows that collapsing L1 into one whole-sequence vector exposes little extra information over L2:

\[
55.57\%\rightarrow56.42\%.
\]

But preserving L1's ordered Fixed250 structure and giving the classifier correct phase access gives:

\[
56.42\%\rightarrow60.79\%.
\]

Thus a useful working description is:

\[
\boxed{
L1:\ \text{richer temporally localized / phase-dependent information}
}
\]

This does not mean L1 is a better standalone classifier. It means L1 retains information whose interpretation depends more strongly on when it occurs.

### 9.3 Current hierarchical picture

The strongest interpretation of the series is:

```text
input spike train
    -> L1: local motion / stroke / motif / phase-dependent detail
    -> L2: transformed + more integrated class evidence
```

or conceptually:

\[
\boxed{
L1:\ WHAT\ happened\ locally
}
\]

with richer temporal detail, followed by

\[
\boxed{
L2:\ more\ integrated\ evidence\ for\ the\ whole\ gesture
}
\]

This is a working hypothesis supported by the current probes; it is not yet a formal information-theoretic decomposition.

---

## 10. RSNN/context-layer motivation from Exp8.0.5

### 10.1 Explicit phase is currently an oracle-like context variable

Exp8.0.5 uses an externally known phase index:

\[
b(t).
\]

The phase-aware classifier implements:

\[
e_t=W_{b(t)}z_t^{(1)}.
\]

A future recurrent module should replace the externally supplied phase with a learned state:

\[
h_t=f(h_{t-1},z_t).
\]

The intended role is not simply "remember the past" in an abstract sense.

A more precise target is:

\[
\boxed{
\text{use history to reinterpret current local evidence}
}
\]

or

\[
(z_t,h_{t-1})\rightarrow\tilde z_t
\rightarrow\text{class evidence}.
\]

### 10.2 Why L1 may be a better RSNN input

The current evidence suggests that L1 retains information that is lost or weakened when time is collapsed.

Therefore a strong hypothesis is:

\[
\boxed{
RSNN(L1)\ \text{may be more useful than}\ RSNN(L2)
}
\]

because a context learner benefits from temporally localized information that has not yet been fully transformed into a sequence-level class representation.

However, this is still a hypothesis and should be tested directly.

### 10.3 A likely better architecture than simply `L2 -> RSNN`

The current results naturally suggest a two-pathway architecture:

```text
L1_t -> recurrent/context branch
L2_t -> integrated evidence branch
          \        /
            fusion
              -> class
```

For example:

\[
h_t=RSNN(z_t^{(1)},h_{t-1}),
\]

\[
e_t=W_2 z_t^{(2)} + W_c[h_t,z_t^{(1)}].
\]

This matches the empirical picture from Exp8.0.5:

- L1 benefits strongly from phase/context;
- L2 already provides useful whole-sequence evidence.

### 10.4 Controlled RSNN input ablation

The next recurrent experiment should keep the recurrent module fixed and compare:

```text
A: RSNN(L1)
B: RSNN(L2)
C: RSNN([L1;L2])
D: no recurrent context baseline
```

The hidden size, parameter budget, tau, objective, seeds, and frozen local backbone should be matched as closely as possible.

This will directly test which representation is the best substrate for learned temporal context.

---

## 11. What the Exp8.0 series has not proven

Several claims should still be avoided.

### 11.1 Phase dependence does not prove recurrence is necessary

Exp8.0.5 proves that correct temporal context/phase is useful.

It does **not** prove that only an RSNN can provide it.

A positional encoding, elapsed-time counter, temporal convolution, RNN, or another context mechanism could potentially reproduce some of the same gain.

### 11.2 Phase dependence does not yet prove long-term memory

The experiment uses explicit 250 ms bins. It shows that temporal position/context matters, but it does not establish the minimum history horizon required to infer that context.

The next recurrent experiments need history-destruction or sequence-order controls to establish that the model truly uses past information.

### 11.3 L1 is not yet proven to be the best RSNN input

The L1 hypothesis is well motivated, but the required L1-vs-L2-vs-L1+L2 recurrent ablation has not yet been run.

### 11.4 Score magnitude is not accuracy contribution

Branch score RMS can describe how heavily a trained head uses one branch, but it is not a direct attribution of accuracy.

Use same-model removal tests, paired accuracy changes, and when needed a carefully defined attribution method rather than interpreting RMS fraction as "percent contribution."

---

## 12. Implications for further local-representation optimization

The Exp8.0 tau sweep suggests that the next local-backbone improvement should focus more on **training objective and representation organization** than on simply extending tau.

A promising direction is layer-specific temporal supervision.

### 12.1 L1 local supervision

For 250 ms bins:

\[
c_b^{(1)}=\sum_{t\in b} z_t^{(1)},
\]

with one shared classifier across all bins:

\[
\mathcal L_{L1,250}
=
\frac1{B_{valid}}
\sum_b CE(W_{local}c_b^{(1)},y).
\]

Because the same \(W_{local}\) is shared across bins, this encourages local discriminative information without explicitly supplying absolute phase identity.

### 12.2 L2 should not automatically receive the identical local objective

A reasonable hierarchy is:

```text
L1 -> short/local supervision
L2 -> medium/whole-sequence integration supervision
```

For example:

\[
\mathcal L
=
\mathcal L_{whole}
+
\lambda_1\mathcal L_{L1,250}
+
\lambda_2\mathcal L_{L2,500},
\]

or initially the simpler:

\[
\mathcal L
=
\mathcal L_{whole}
+0.1\mathcal L_{L1,250}.
\]

The reason not to blindly apply the same 250 ms classification loss to both layers is that this may make L1 and L2 learn redundant local classifiers rather than a hierarchy.

### 12.3 Useful evaluation set for future local objectives

Every new local-objective experiment should report at least:

```text
L1 whole
L1 Fixed250
L2 whole
L2 Fixed250
L1+L2 whole/time-shared
L1 Fixed250 + L2 whole true-phase
```

A useful diagnostic is the **phase-utilization gap**:

\[
\Delta_{phase}
=
BA_{phase}
-
BA_{timeshared}.
\]

For Exp8.0.5:

\[
\Delta_{phase}
=
60.79\%-56.42\%
=
\boxed{4.38\text{ pp}}.
\]

This can be interpreted as the amount of performance currently unlocked by explicit context/phase beyond a time-shared readout on the same frozen representation.

A strong local-representation improvement would ideally increase the absolute BA while preserving useful context-dependent structure rather than merely collapsing the phase gap by removing temporal specificity.

---

## 13. Current consolidated conclusions

The Exp8.0 series currently supports the following chain of conclusions:

1. **`234x234` is the strongest tested local tau organization.** Extending L2 to much slower `345` dynamics hurts rather than helps.
2. **L1 and L2 are not redundant.** Frozen fusion probes show complementary information.
3. **Cooperative joint L1/L2 supervision is substantially better for representation quality than treating L1 as an independent auxiliary classifier.**
4. **A large part of L1's useful information is temporally localized.** Whole-sequence L1 adds little over L2, while ordered/phase-aware L1 adds much more.
5. **Raw count CE was a training-objective pathology.** Mean normalization is necessary to make fair comparisons with the strong Exp8 backbone.
6. **End-to-end phase-aware training gives only a small native gain and can specialize the backbone to its head.** Best native accuracy and best reusable representation are not the same thing.
7. **Frozen-backbone Exp8.0.5 provides the cleanest phase result.** Correct phase-aware readout reaches **60.79% test BA**, versus 56.42% time-shared and 53.46% destroyed-phase.
8. **Correct phase alignment matters causally.** A matched-capacity destroyed-phase control and circular phase-shift test both strongly reduce performance.
9. **L1 is best described as retaining richer temporally localized/phase-dependent information; L2 is best described as more temporally integrated and class-oriented.**
10. **The next context model should be evaluated as a mechanism for reinterpreting current local evidence using history, not merely as an extra classifier layer.**

A compact architectural hypothesis for the next stage is therefore:

```text
local spike input
    -> L1: rich local temporal detail
       -> RSNN/context state
    -> L2: integrated class evidence
       -> context-conditioned fusion
    -> final class decision
```

The immediate experimental question is no longer simply:

> "Can recurrence increase accuracy?"

It is:

> **Can a learned recurrent state recover the explicit-phase advantage seen in Exp8.0.5 without being given the phase index, and can history-destruction controls verify that the gain truly comes from temporal memory/context?**
