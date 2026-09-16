# Exp7.3 Series Summary — Representation Quality, Linear Readout Alignment, and LIF Realization

## 1. Purpose

The Exp7.3 series was designed to separate three questions that were previously mixed together:

1. **What information is present in the L2 SNN representation?**
2. **How well is that information aligned with the final linear readout that we are allowed to deploy?**
3. **How much additional performance is lost when the same continuous evidence must be realized through an output LIF neuron and spike-count readout?**

The central conclusion is:

> **TSCE and WCCE shape different L2 geometries. TSCE can produce a highly linearly decodable representation, but that information is not necessarily arranged in a form that a constrained bias-free whole-sequence head can use directly. WCCE produces a representation that is better aligned with that constrained sequence-level readout. A flexible affine probe can recover information from either representation that the deployable head cannot directly access. The output LIF then adds a second, distinct realization bottleneck.**

This distinction is important throughout the series:

```text
representation quality
        !=
readout alignment
        !=
LIF realization quality
```

---

## 2. Common notation

Let the frozen L2 SNN output be

\[
z_t \in \{0,1\}^{128},
\]

with valid sequence length \(T\).

Define valid whole count

\[
Z = \sum_{t=1}^{T} z_t,
\]

and valid temporal mean

\[
\bar z = \frac{1}{T}\sum_{t=1}^{T} z_t.
\]

A bias-free deployment-compatible linear head is

\[
e_t = W z_t,
\]

with sequence-level analog score

\[
s_{\rm linear}
= \frac{1}{T}\sum_t Wz_t
= W\bar z.
\]

Because multiplying all class logits for one sample by a positive scalar does not change `argmax`, whole-count and mean are classification-equivalent for a fixed sequence length:

\[
WZ = T W\bar z.
\]

However, the exact training objective, feature normalization, intercept, and deployment interface still matter.

---

## 3. Exp7.3: decomposition of backbone training, W training, and final deployment

Exp7.3 explicitly separated:

1. how L1/L2 are trained;
2. how the final \(128\to12\) matrix \(W\) is trained;
3. how the trained \(W\) is finally deployed.

The common final LIF interface was

```text
L2 spikes -> W -> LIF(beta=0.5, threshold=0.5, cap=1)
          -> valid spike count -> argmax
```

### 3.1 Group A: end-to-end training

Important cases:

- `A1_e2e_linear_tsce`: Linear path + TSCE;
- `A2_e2e_linear_wcce`: Linear path + WCCE.

Their selected L2 backbones later became the frozen TSCE and WCCE representations used in Stage 2.

Across three seeds, the main Exp7.3 summary was approximately:

| Method | Backbone objective | Head objective | Native Analog test BA | Final LIF test BA | L2 whole-count probe test BA |
|---|---:|---:|---:|---:|---:|
| A1 | TSCE | TSCE | 47.19% | 45.86% | 59.61% |
| A2 | WCCE | WCCE | 56.45% | 52.47% | 58.63% |

The key observation is already visible here:

- the **TSCE backbone has very strong linear accessibility under the flexible L2 probe**;
- the original TSCE-trained head is much weaker as a whole-sequence classifier;
- the WCCE backbone is much better aligned with its native whole-sequence head.

### 3.2 Group B: freeze L1/L2 and retrain only W

Stage 2 discarded the Stage-1 head and retrained a fresh bias-free \(W\) on frozen L2 trajectories.

The most important matched comparison is:

- `B2_tsbackbone_linear_wcce`: TSCE-trained backbone + fresh Linear/WCCE head;
- `B6_wcbackbone_linear_wcce`: WCCE-trained backbone + fresh Linear/WCCE head.

Results:

\[
\text{B2 native Analog BA} \approx 51.11\%,
\]

\[
\text{B6 native Analog BA} \approx 55.27\%.
\]

The corresponding final same-W LIF results were approximately:

\[
\text{B2 final LIF BA} \approx 49.73\%,
\]

\[
\text{B6 final LIF BA} \approx 52.28\%.
\]

This means that retraining a whole-sequence head on the TSCE representation helps substantially relative to using the original TSCE head, but it still does not fully match a representation that was trained with WCCE from the beginning.

---

## 4. Why the original TSCE head must be discarded for whole-sequence decoding

TSCE optimizes

\[
\mathcal L_{\rm TS}
=
\frac{1}{T}\sum_t CE(W z_t, y).
\]

The original TSCE head therefore learns a matrix appropriate for

\[
z_t \rightarrow \text{class}
\]

at individual timesteps.

There is no requirement that the same \(W\) must also be optimal for

\[
\bar z \rightarrow \text{class}.
\]

Therefore, when the final desired decoder is a whole-sequence readout, it is reasonable to discard the original TSCE head and retrain a sequence-level head:

\[
\mathcal L_{\rm WC}
=
CE(W_{\rm new}\bar z,y).
\]

This is exactly what B2 does.

The improvement from A1-style whole-sequence use of the TSCE head to B2 confirms that the TSCE backbone contains useful sequence-level information that the original timestep-trained head does not optimally extract.

However, B2 still remains below B6. Therefore the remaining gap is not only a head mismatch; it also reflects a difference in the geometry learned by the backbone.

---

## 5. TSCE and WCCE create different L2 geometry

### 5.1 TSCE gradient

For TSCE,

\[
\mathcal L_{\rm TS}
=
\frac{1}{T}\sum_t CE(Wz_t,y),
\]

and the gradient on one L2 state is approximately

\[
\frac{\partial \mathcal L_{\rm TS}}{\partial z_t}
\propto
W^T(p_t-y),
\]

where

\[
p_t=\operatorname{softmax}(Wz_t).
\]

Each timestep receives its own local classification error.

Thus TSCE strongly encourages every local L2 state to contain class-discriminative evidence.

A useful conceptual form is

\[
z_t = c_t + \delta_{y,t} + \epsilon_t,
\]

where

- \(c_t\) is common/background activity;
- \(\delta_{y,t}\) is class-specific local evidence;
- \(\epsilon_t\) is local noise.

TSCE strongly encourages \(\delta_{y,t}\) to exist, but does not explicitly require the sequence aggregate to be centered around the origin or directly separable by one constrained bias-free sequence-level head.

### 5.2 WCCE gradient

WCCE optimizes

\[
\mathcal L_{\rm WC}
=
CE(W\bar z,y),
\]

with

\[
\bar z=\frac1T\sum_t z_t.
\]

Its gradient is

\[
\frac{\partial \mathcal L_{\rm WC}}{\partial z_t}
=
\frac1T W^T(p_{\rm seq}-y),
\]

where

\[
p_{\rm seq}=\operatorname{softmax}(W\bar z).
\]

Every valid timestep receives the same sequence-level classification error.

Therefore WCCE directly shapes the **aggregate representation** so that

\[
W\bar z
\]

is discriminative.

Because the Exp7.3 Linear head is bias-free, the backbone is implicitly encouraged to create a geometry that works with pairwise boundaries

\[
(w_i-w_j)^T\bar z = 0,
\]

which all pass through the origin.

This makes the WCCE backbone naturally compatible with B6.

---

## 6. The central crossover: TSCE wins with the flexible probe, WCCE wins with the constrained head

The most important combined observation is:

### Flexible whole-count probe / P7

\[
\text{TSCE backbone} \approx 59.61\%,
\]

\[
\text{WCCE backbone} \approx 58.63\%.
\]

So under the repository-standard affine probe, the TSCE backbone is at least as linearly accessible and is slightly higher on the three-seed mean.

### Constrained bias-free WCCE head

\[
\text{B2: TSCE backbone + WCCE head} \approx 51.11\%,
\]

\[
\text{B6: WCCE backbone + WCCE head} \approx 55.27\%.
\]

So under the deployment-compatible bias-free sequence-level head, the WCCE backbone is clearly better aligned.

The correct interpretation is not that one backbone simply contains more information in every sense.

A more precise interpretation is:

> **TSCE appears to produce a representation with strong linear decodability, but some of that information requires affine recalibration to become easy to extract. WCCE more directly organizes the aggregate representation around the geometry required by a simple bias-free sequence-level readout.**

A useful shorthand is:

```text
TSCE -> strong local class evidence / high decodability
WCCE -> stronger sequence-readout alignment
```

or

```text
TSCE: information is present
WCCE: information is present and better arranged for the constrained head
```

---

## 7. The repository L2 linear probe is not the same model as B6

This distinction is essential.

### 7.1 Legacy L2 whole-count probe / Exp7.3.2 P7

The repository-standard probe first forms

\[
Z=\sum_t z_t.
\]

Then it applies train-only `StandardScaler(with_mean=True, with_std=True)`:

\[
\tilde Z = D^{-1}(Z-\mu),
\]

and fits multinomial logistic regression with an explicit intercept:

\[
s = A\tilde Z+b.
\]

In raw L2 coordinates this is exactly

\[
s = W_{\rm raw}Z+b_{\rm eff},
\]

where

\[
W_{\rm raw}=AD^{-1},
\]

\[
b_{\rm eff}=b-W_{\rm raw}\mu.
\]

Thus P7 is an **affine classifier**.

Its pairwise decision boundaries are

\[
(w_i-w_j)^TZ+(b_i-b_j)=0,
\]

so they do not need to pass through the origin.

### 7.2 B6

B6 trains only

\[
s=W\bar z
\]

with

```text
bias = False
no centering
no runtime scaler
```

and WCCE:

\[
\mathcal L=CE(W\bar z,y).
\]

Its pairwise boundaries are

\[
(w_i-w_j)^T\bar z=0.
\]

This is a significantly more constrained classifier family.

Therefore

\[
\boxed{\text{high P7 probe BA} \not\Rightarrow \text{high B6-style BA}}
\]

because P7 is allowed to recenter the feature space and use class-specific affine offsets.

---

## 8. Exp7.3.1 and Exp7.3.2: why optimization alone did not close the probe gap

Exp7.3.1 asked whether the gap between the Exp7.3 Stage-2 Linear heads and the L2 probe was merely caused by weak optimization.

It tested longer Adam training, LBFGS, L2 regularization, feature scaling, and converged deployment-compatible bias-free heads while keeping L1/L2 frozen.

The important result was that better optimization did **not** fully reproduce the legacy probe.

This motivated Exp7.3.2, which explicitly crossed:

- mean vs whole-count aggregation;
- centered vs non-centered scaling;
- intercept off vs on.

P7 was defined as the exact legacy probe:

```text
whole-count + scaling + centering + intercept
```

The bridge experiment showed that the missing affine freedom is a major part of the difference.

For the TSCE backbone, for example:

- P4: whole-count, no center, no bias: ~55.10%;
- P5: whole-count, no center, bias: ~59.36%;
- P6: whole-count, center, no bias: ~59.69%;
- P7: whole-count, center, bias: ~59.61%.

For the WCCE backbone:

- P4: whole-count, no center, no bias: ~54.10%;
- P5: whole-count, no center, bias: ~58.63%;
- P6: whole-count, center, no bias: ~57.25%;
- P7: whole-count, center, bias: ~58.63%.

These results show that the probe advantage is not simply "a better optimizer found a better W." A large part comes from changing the classifier family from a homogeneous bias-free map to an affine decision function.

---

## 9. Why P7 is especially effective for the TSCE backbone

Suppose TSCE creates a sequence aggregate of the form

\[
Z = c + \Delta_y,
\]

where \(c\) is a large common firing baseline and \(\Delta_y\) is class-specific information.

A bias-free classifier must separate classes using

\[
W(c+\Delta_y).
\]

If \(c\) is large or the class clusters occupy a region far from the origin, the origin constraint can make separation difficult.

P7 instead uses

\[
Z-\mu,
\]

and then an intercept.

This can remove much of the shared baseline:

\[
Z-\mu \approx \Delta_y + \text{residual variation}.
\]

Therefore P7 can expose class information that is already present in the TSCE representation but is poorly aligned with an origin-referenced bias-free head.

This is why the TSCE backbone can look very strong under P7 while remaining weaker under B2.

The correct wording is therefore:

> **TSCE produces information that is highly linearly accessible after flexible affine calibration; WCCE produces information that is more directly accessible to the constrained sequence-level head used for deployment.**

---

## 10. Exp7.3.3: separating affine information from output-neuron realization

Exp7.3.3 started from P7 and wrote the exact source classifier as

\[
s_{\rm affine}=WZ+b.
\]

It then asked what happens when the same affine evidence is implemented with spiking output neurons.

The strongest control used a non-leaky \(\beta=1\) IF neuron and the charge-preserving readout

\[
\theta N + U_T.
\]

For a non-leaky integrate-and-fire output,

\[
\theta N + U_T
=
\sum_t Wz_t+b
=
WZ+b,
\]

up to numerical precision.

The experiment confirmed that both start- and end-bias injection with this charge-preserving readout reproduce the P7 Analog classifier exactly.

This establishes an important conservation result:

> **The affine information itself is not inherently incompatible with an IF state. The large loss appears when the analog state must be compressed into spike count.**

A spike-count readout discards or distorts information through several mechanisms:

1. threshold quantization;
2. finite output-spike bandwidth;
3. unresolved membrane residual;
4. negative evidence cannot be represented as negative spike count;
5. previously emitted positive spikes cannot later be canceled by negative evidence.

Thus the Linear-to-LIF gap should be viewed as a **realization/readout bottleneck**, not merely as evidence that L2 lacks information.

A practical caution from Exp7.3.3 is that P7's folded raw-space weight scale is optimized for affine classification, not for the fixed LIF threshold. Therefore directly passing P7 weights through the beta=0.5 output LIF without calibration can produce near-silent/chance behavior. That result reflects a severe scale/dynamics mismatch and should not be interpreted as the intrinsic quality of the P7 classifier.

---

## 11. Why a Linear-trained W can outperform a W trained directly through the LIF

Exp7.3 and related Exp7.2.6 experiments showed an important phenomenon:

> A weight matrix learned through a continuous Linear path can perform better after direct substitution into the same output LIF than a matrix learned directly through the LIF surrogate objective.

The main explanation is that the two optimizations solve different problems.

### 11.1 Linear training

For frozen L2,

\[
e_t=Wz_t.
\]

The continuous objective has direct access to:

- signed evidence;
- evidence magnitude;
- dense gradients;
- class margins.

It can focus mainly on learning class-discriminative directions.

### 11.2 LIF training

With output dynamics

\[
I_t=Wz_t,
\]

\[
U_t^- = \beta U_{t-1}+I_t,
\]

\[
s_t=H(U_t^- - \theta),
\]

and spike-count classification, the same \(W\) must simultaneously solve:

```text
class geometry
+ current amplitude calibration
+ threshold crossing
+ leakage compensation
+ temporal alignment
+ spike-count quantization
```

and training relies on a surrogate derivative around threshold.

Therefore LIF training can be pulled toward weights that make neurons fire more effectively without preserving the best class-discriminative geometry.

This gives the working hypothesis:

```text
Linear objective:
    first finds a good class geometry.

LIF objective:
    must jointly learn class geometry and neuron-compatible firing dynamics.
```

This is why a Linear-pretrained solution can provide a better starting basin for later LIF adaptation.

---

## 12. Exp7.3.4: test whether the remaining problem is initialization or the LIF objective

Exp7.3.4 directly tests this hypothesis on the WCCE backbone.

The SNN backbone is frozen and only the final \(128\to12\) matrix is trainable:

```text
Frozen L1/L2 -> W -> LIF(beta=0.5, threshold=0.5, cap=1)
                         -> spike-count WCCE
```

Three initialization conditions are compared:

```text
random W
A2 pretrained Linear W
B6 pretrained Linear W
```

For the pretrained cases,

\[
W_{\rm epoch\ 0}=W_{\rm Linear}.
\]

Epoch 0 is the unchanged direct-transfer reference. From epoch 1 onward, only \(W\) is optimized through the LIF spike-count WCCE objective.

The key contrasts are:

\[
\Delta_{\rm adapt}
=
BA_{\rm pretrained\ fine\mbox{-}tune}
-
BA_{\rm direct\ transfer},
\]

and

\[
\Delta_{\rm init}
=
BA_{\rm pretrained\ fine\mbox{-}tune}
-
BA_{\rm random\ LIF\ train}.
\]

Interpretation:

- `fine-tune > direct`: Linear pretraining finds useful class geometry and LIF training then performs beneficial dynamics adaptation;
- `direct > fine-tune`: the LIF objective actively moves a good Linear solution toward a worse region;
- `pretrained fine-tune > random`: initialization/optimization basin is an important part of the problem;
- `pretrained fine-tune ~= random` while direct remains better: the dominant limitation is likely the LIF objective/realization rather than initialization alone.

Exp7.3.4 also tracks the same current \(W\) through both:

```text
LIF readout
Linear bypass readout
```

plus weight drift, row cosine similarity, spike counts, silent-sample fraction, and margins. This allows us to determine whether LIF adaptation improves firing while preserving class geometry or improves firing by destroying the original Linear solution.

---

## 13. Working model of the entire Exp7.3 series

The current evidence supports the following decomposition.

### Stage A: representation formation

TSCE:

```text
strong per-timestep supervision
-> local class-discriminative L2 activity
-> high flexible-probe accessibility
```

WCCE:

```text
sequence-level supervision
-> aggregate L2 geometry aligned with W * mean(z)
-> stronger constrained whole-sequence readout
```

### Stage B: continuous readout

A flexible affine probe can use

```text
whole-count
+ per-feature scaling
+ centering
+ intercept
```

to recover information that a bias-free deployment-compatible \(W\) cannot fully exploit.

### Stage C: spiking realization

Replacing the continuous score with

```text
W z_t -> LIF -> spike count
```

introduces another bottleneck:

```text
thresholding
+ leakage
+ quantization
+ finite firing bandwidth
+ loss of signed evidence
```

Therefore a high L2 probe score is an upper-level diagnostic of accessible information, not a guarantee that the final LIF output can realize the same classifier.

---

## 14. Main conclusions

### Conclusion 1 — Probe quality and deployable-head quality measure different things

The L2 probe asks approximately:

> How much class information can a flexible affine decoder recover from the frozen L2 representation?

B2/B6-style training asks:

> Is that information already arranged so that a simple bias-free whole-sequence matrix can use it?

These are not equivalent.

### Conclusion 2 — TSCE and WCCE should not be ranked by only one readout

TSCE shows stronger or comparable flexible affine decodability, while WCCE is better aligned with the constrained whole-sequence head.

A concise description is:

```text
TSCE improves decodability.
WCCE improves readout alignment.
```

This should be treated as a working interpretation rather than a claim that TSCE universally contains more information.

### Conclusion 3 — For a TSCE backbone, the original TSCE head is not the right whole-sequence decoder

If the final inference rule is whole count / whole mean, discard the original timestep-trained head and retrain a sequence-level head on the frozen L2 representation.

This recovers part of the gap, but not all of it.

### Conclusion 4 — The remaining TSCE-vs-WCCE gap under a bias-free sequence head is a representation-geometry mismatch

The WCCE backbone was explicitly trained so that \(W\bar z\) works. The TSCE backbone was not.

### Conclusion 5 — P7 recovers information by adding affine freedom

Centering and intercept are major contributors to the gap between the constrained Linear head and the legacy L2 probe.

### Conclusion 6 — Output LIF introduces a separate realization bottleneck

Even with a strong continuous classifier, converting analog signed evidence into thresholded spike count can lose information.

### Conclusion 7 — Linear pretraining may be a useful initialization strategy for LIF output training

Because the Linear objective can first identify a clean discriminative geometry, using its \(W\) as the initial LIF connection matrix may separate

```text
class learning
```

from

```text
LIF dynamics adaptation.
```

Exp7.3.4 is the direct test of this hypothesis.

---

## 15. Practical implication for future SNN architecture design

When evaluating a future SNN readout, always separate at least three diagnostics:

1. **L2 affine probe** — how much information exists and is linearly accessible with flexible calibration;
2. **deployment-compatible continuous head** — how well the representation is aligned with the actual allowed readout;
3. **same-W LIF realization** — how much additional loss is introduced only by output-neuron dynamics.

A single final BA cannot identify which of these three stages is failing.

The useful diagnostic chain is therefore:

```text
L2 representation
    |
    +-> flexible affine probe
    |
    +-> constrained continuous W
                |
                +-> same W through LIF
```

This decomposition is the main methodological lesson from the Exp7.3 series.

---

## 16. Relevant files

Main experiment and results:

- `scripts/experiment_7_3_training_strategy_decomposition.py`
- `scripts/experiment_7_3_training_strategy_decomposition/README.md`
- `notebooks/experiment_7_3_training_strategy_decomposition.ipynb`
- `notebooks/artifacts/experiment_7_3_training_strategy_decomposition/training_strategy_decomposition_v1/`

Frozen-L2 Linear optimization:

- `docs/plans/EXP7_3_1_FROZEN_L2_LINEAR_OPTIMIZATION.md`
- `scripts/experiment_7_3_1_frozen_l2_linear_optimization.py`
- `notebooks/artifacts/experiment_7_3_1_frozen_l2_linear_optimization/frozen_l2_linear_optimization_v1/`

Affine-probe bridge:

- `docs/notes/EXP7_3_2_AFFINE_PROBE_BRIDGE.md`
- `scripts/experiment_7_3_2_affine_probe_bridge.py`
- `notebooks/artifacts/experiment_7_3_2_affine_probe_bridge/affine_probe_bridge_v1/`

Affine-to-LIF substitution:

- `docs/notes/EXP7_3_3_AFFINE_LIF_SUBSTITUTION.md`
- `scripts/experiment_7_3_3_affine_lif_substitution.py`
- `notebooks/artifacts/experiment_7_3_3_affine_lif_substitution/affine_lif_substitution_v1/`

Linear-pretrained LIF fine-tuning:

- `scripts/experiment_7_3_4_linear_pretrained_lif_finetuning.py`
- `scripts/experiment_7_3_4_linear_pretrained_lif_finetuning/README.md`
- `notebooks/experiment_7_3_4_linear_pretrained_lif_finetuning.ipynb`
- `notebooks/artifacts/experiment_7_3_4_linear_pretrained_lif_finetuning/linear_pretrained_lif_finetuning_v1/`
