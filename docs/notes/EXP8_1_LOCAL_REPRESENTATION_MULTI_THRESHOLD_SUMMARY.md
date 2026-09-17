# Exp8.1 / Exp8.1.1 — Local representation, hidden quantization, and multi-threshold findings

## Scope

This note summarizes the current local-representation investigation that led from Exp7.3.5 to Exp8.1 and Exp8.1.1. The goal is to separate three questions that can otherwise be conflated:

1. How much local class/temporal information exists in a hidden analog state before thresholding?
2. How much of that information is preserved, destroyed, or reshaped by hidden spike communication?
3. If a local representation improves, does the downstream hidden layer preserve the improvement?

This note intentionally does **not** use RSNN/long-range memory as an explanation. The focus is the two local SNN layers and the representation they communicate.

---

## 1. Locked reference: Exp7.3 A2

The common reference architecture is:

```text
30-channel raw spike train
    -> L1: 128 neurons, shifts (2,3,4)
    -> L2: 128 neurons, shifts (2,3,4)
    -> bias-free Linear(128 -> 12)
```

Training follows Exp7.3 A2:

```text
end-to-end L1 + L2 + Linear
valid-length temporal-mean evidence
cross entropy (A2 WCCE)
no activity regularizer
seeds 11 / 23 / 37
```

For a hidden layer, the useful states are:

```math
I_t = \alpha I_{t-1} + W x_t
```

```math
U_t^- = \beta U_{t-1} + I_t
```

```math
S_t = H(U_t^- - \theta)
```

```math
U_t = U_t^- - \theta S_t.
```

The key distinction throughout this investigation is:

```text
pre_reset       = U^-       continuous internal membrane state
communication   = S or n    value actually sent to the next layer
```

For binary neurons, `communication = S in {0,1}`. For the weighted/multi-spike condition in Exp8.1, `communication = n in {0,...,31}`.

---

## 2. How representation quality is measured

Two probe families are important and answer different questions.

### Whole-mean probe

```math
\bar z = \frac{1}{T}\sum_t z_t.
```

A linear probe on `bar z` asks whether class information is available after temporal position has been removed.

### Ordered Fixed250 probe

At 64 Hz, each 250 ms bin contains 16 timesteps. For each ordered bin:

```math
h_b = \mathrm{mean}_{t\in b} z_t,
```

and the probe sees:

```math
H = [h_1; h_2; \ldots; h_B].
```

This lets the classifier use different weights at different temporal positions:

```math
score = W_1 h_1 + W_2 h_2 + \cdots + W_B h_B.
```

Therefore `Fixed250 BA` is a diagnostic of **linearly accessible local-temporal / phase information**, not the native A2 final accuracy.

This distinction becomes critical in Exp8.1.1: a hidden representation can have a strong Fixed250 probe while the native whole-mean A2 head remains much lower.

---

## 3. Exp7.3.5: the original hidden-state diagnostic

Exp7.3.5 froze the selected A2 backbone and probed `I`, `U^-`, binary spikes, and post-reset membrane without retraining the SNN.

### L1 Fixed250 result

The strongest initial signal was at L1:

| L1 state | Fixed250 test BA |
| --- | ---: |
| pre-reset `U^-` | 62.31% |
| binary spike `S` | 55.14% |

Thus:

```math
62.31 - 55.14 = 7.17\ \text{pp}.
```

This is strong evidence that the L1 membrane contained local-temporal class information that was not linearly accessible after binary communication.

However, this should **not** be generalized to every hidden layer or every aggregation.

### L1 whole-mean result

For whole-mean features:

| L1 state | Whole-mean test BA |
| --- | ---: |
| pre-reset `U^-` | 53.53% |
| binary spike `S` | 56.48% |

Binary thresholding did not show a whole-sequence information loss here. The main L1 issue was specifically visible in the ordered Fixed250 representation.

### L2 result is different

For L2 Fixed250:

| L2 state | Fixed250 test BA |
| --- | ---: |
| pre-reset `U^-` | 58.70% |
| binary spike `S` | 61.84% |

So L2 thresholding actually made the representation **more linearly accessible** for this probe:

```math
61.84 - 58.70 = +3.14\ \text{pp}.
```

Therefore the durable conclusion from Exp7.3.5 is not “binary hidden spikes are always bad.” It is:

> L1 shows a substantial phase-aware `U^- -> binary spike` communication gap, while L2 thresholding can itself act as a useful nonlinear representation transform.

That observation motivated richer L1 communication experiments rather than abandoning spike communication in general.

---

## 4. Why weighted/multi-spike and multi-threshold were considered

### Binary communication

A binary hidden neuron communicates only:

```math
S_t \in \{0,1\}.
```

A strong pre-reset state can only emit one event in that timestep, and subtractive reset removes one threshold:

```math
U_t = U_t^- - \theta S_t.
```

This can both quantize activation magnitude and temporally redistribute strong activation across later timesteps.

### Weighted / multi-spike communication

Exp8.1 tested:

```math
n_t = \mathrm{clip}\left(
\left\lfloor\frac{\max(U_t^-,0)}{\theta}\right\rfloor,
0,31
\right)
```

with:

```math
U_t = U_t^- - n_t\theta.
```

One neuron can communicate `0..31` events in one timestep. Cap 31 is the hidden hardware capacity limit, not a tuned software hyperparameter.

### Fixed heterogeneous thresholds

The alternative was to keep every neuron binary but diversify excitability across the population:

```math
\theta_i \in \{0.5\theta_0,\theta_0,1.5\theta_0\}.
```

These are **fixed heterogeneous thresholds**, not adaptive thresholds. A neuron's threshold never changes with time or firing history.

For the current baseline:

```text
theta0 = 0.5
```

so the actual thresholds are:

```text
0.25 / 0.50 / 0.75
```

The initial purpose was a clean mechanism ablation centered on the baseline threshold, not an optimal threshold search.

---

# 5. Exp8.1: L1 communication-code ablation

## Question

Exp8.1 changed L1 communication while keeping L2 as the original 128-neuron uniform-binary `(2,3,4)` layer.

The main methods were:

| Method | L1 | L2 |
| --- | --- | --- |
| `binary128` | 128 uniform-binary | 128 binary |
| `weighted31_128` | 128 multi-spike, cap 31 | 128 binary |
| `mt3_128` | 128 fixed-threshold heterogeneous binary | 128 binary |
| `binary384` | 384 uniform-binary | 128 binary |
| `mt3_384` | 384 MT3 | 128 binary |

The 384-neuron binary control was necessary because otherwise `MT384 > MT128` could simply be a width/capacity effect.

## Key results

| Method | Final A2 BA | L1 pre-reset Fixed250 | L1 communication Fixed250 | L1 `U^- - comm` gap | L2 communication Fixed250 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `binary128` | 56.45% | 62.31% | 53.41% | 8.90 pp | 61.84% |
| `weighted31_128` | 54.16% | 61.32% | 53.77% | 7.56 pp | 59.50% |
| `mt3_128` | 53.47% | 62.46% | 56.75% | 5.71 pp | 59.36% |
| `binary384` | 57.23% | 63.11% | 52.90% | 10.21 pp | 64.58% |
| `mt3_384` | 57.39% | 64.95% | 58.66% | 6.29 pp | 63.93% |

### Finding 1: MT improves the communicated L1 representation

At width 128:

```math
53.41\% \rightarrow 56.75\%
```

for L1 Fixed250 communication, a gain of about:

```math
+3.34\ \text{pp}.
```

At the same time, L1 pre-reset remained almost unchanged:

```math
62.31\% \rightarrow 62.46\%.
```

So the MT improvement was not primarily caused by a stronger latent analog state. It improved how information was converted into the communicated binary population code.

The L1 Fixed250 quantization gap shrank from:

```math
8.90\ \text{pp} \rightarrow 5.71\ \text{pp}.
```

This supports the original Exp7.3.5 hypothesis that L1 binary communication was leaving local-temporal information inaccessible.

### Finding 2: cap-31 weighted communication was rarely actually multi-event

The weighted neuron theoretically allowed `0..31`, but the trained model almost never used values above 1:

```text
shift 2: P(n > 1) ~= 0.0019%
shift 3: P(n > 1) ~= 0.0098%
shift 4: P(n > 1) ~= 0.5065%
```

Therefore the hardware capacity of 31 was not the limiting factor. The operating regime almost always remained effectively binary.

This explains why weighted communication only weakly changed L1 Fixed250 accessibility:

```math
53.41\% \rightarrow 53.77\%.
```

The current evidence therefore favors **population excitability heterogeneity** over simply giving one neuron a larger event-count alphabet.

### Finding 3: width alone does not explain MT behavior

Increasing binary L1 width from 128 to 384 changed final BA only modestly on average:

```math
56.45\% \rightarrow 57.23\%.
```

MT benefited much more from additional width:

```math
53.47\% \rightarrow 57.39\%.
```

This is consistent with the fact that MT128 divides only 128 neurons over:

```text
3 tau groups x 3 threshold groups = 9 neuron types
```

or roughly 14 neurons/type. Some high-threshold / short-tau groups are very sparse, so heterogeneity trades some feature-detector capacity for excitability diversity.

### Finding 4: a richer L1 representation did not automatically improve L2

The most important negative result in Exp8.1 was:

```text
L1 MT representation improves
        but
L2 remains uniform binary
        and
L2/final improvement mostly disappears.
```

For 128 neurons:

```math
L1\ communication: 53.41\% \rightarrow 56.75\%  \quad (+3.34\ pp)
```

but:

```math
L2\ communication: 61.84\% \rightarrow 59.36\%  \quad (-2.47\ pp).
```

For the matched-width 384 comparison, MT384 improved L1 communication by about +5.76 pp over binary384, but L2 communication was essentially not improved.

This led to the next hypothesis:

> The richer MT L1 code may require heterogeneous downstream threshold coding to remain useful. A uniform-binary L2 may act as a new bottleneck or may fail to recode the new L1 population structure appropriately.

---

# 6. Exp8.1.1: two-layer MT factorial

## Design

Exp8.1.1 isolated L1 and L2 threshold coding with a matched 2 x 2 factorial. All four cases used the same `128 -> 128`, `(234)(234)`, A2 WCCE architecture and identical trainable parameter count.

| Method | L1 | L2 |
| --- | --- | --- |
| `BB` | binary | binary |
| `MB` | MT3 | binary |
| `BM` | binary | MT3 |
| `MM` | MT3 | MT3 |

The primary local-representation endpoint was:

```text
L2 communication Fixed250 BA
```

because this directly measures the representation actually leaving L2 while retaining ordered local temporal structure.

## Main results

| Case | Final A2 BA | L1 comm Fixed250 | L2 pre-reset Fixed250 | L2 comm Fixed250 |
| --- | ---: | ---: | ---: | ---: |
| `BB` | 56.45% | 53.41% | 58.97% | 61.84% |
| `MB` | 53.47% | 56.75% | 58.06% | 59.36% |
| `BM` | 56.07% | 56.04% | 59.19% | 60.16% |
| `MM` | 56.42% | 57.27% | 57.56% | **63.84%** |

## Result A: MB reproduces the Exp8.1 failure mode

With MT only in L1:

```math
L1\ comm: 53.41\% \rightarrow 56.75\%
```

but:

```math
L2\ comm: 61.84\% \rightarrow 59.36\%.
```

So the richer L1 representation still did not survive through the original uniform-binary L2.

## Result B: L2 MT alone is not a general improvement

Compare `BM - BB`:

```math
Final\ BA: 56.45\% \rightarrow 56.07\%  \quad (-0.38\ pp)
```

```math
L2\ comm\ Fixed250: 61.84\% \rightarrow 60.16\% \quad (-1.68\ pp).
```

Therefore the result is **not** “MT L2 is universally better than binary L2.”

## Result C: L2 MT specifically helps when L1 is already MT

Compare `MM - MB`, where the main change is L2 `binary -> MT`:

```math
L1\ comm: 56.75\% \rightarrow 57.27\% \quad (+0.52\ pp)
```

```math
L2\ pre-reset: 58.06\% \rightarrow 57.56\% \quad (-0.50\ pp)
```

but:

```math
L2\ communication: 59.36\% \rightarrow 63.84\%
```

which is:

```math
\boxed{+4.48\ \text{pp}}
```

and final A2 BA recovered from:

```math
53.47\% \rightarrow 56.42\% \quad (+2.95\ pp).
```

The important localization is that the improvement does **not** appear in the L2 pre-reset probe. It appears mainly during:

```text
L2 pre-reset membrane -> L2 communicated spike population
```

For MB:

```math
58.06\% \rightarrow 59.36\% \quad (+1.31\ pp)
```

whereas MM gives:

```math
57.56\% \rightarrow 63.84\% \quad (+6.28\ pp).
```

Thus MT-L2 is acting as a useful nonlinear population recoder for the representation coming from MT-L1.

## Result D: the factorial interaction is the central evidence

The formal interaction is:

```math
interaction = MM - MB - BM + BB.
```

For L2 communication Fixed250 BA:

```math
\boxed{+6.16\ \text{pp mean interaction}}
```

and for native final BA:

```math
\boxed{+3.34\ \text{pp mean interaction}}.
```

The more intuitive form is:

```text
L2 MT effect when L1 is binary:
    BM - BB = -1.68 pp on L2 Fixed250

L2 MT effect when L1 is MT:
    MM - MB = +4.48 pp on L2 Fixed250
```

Therefore:

> L2 MT is valuable mainly in the presence of an MT L1. The two layers are not independent substitutions; their threshold-coding effects interact.

This supports the hypothesis that heterogeneous L1 communication benefits from heterogeneous downstream population coding.

### Seed consistency

For `MM - MB`, L2 communication Fixed250 improved for all three seeds, by roughly +3 to +6 pp. This reduces concern that the +4.48 pp mean gain comes from a single anomalous seed.

---

# 7. What does the 63.84% MM result actually mean?

The `63.84%` value is **not** the native A2 final classifier BA.

It is obtained after freezing the selected MM backbone, extracting ordered L2 spike features in 250 ms bins, and fitting a separate linear diagnostic probe:

```math
score = W_1 h_1 + W_2 h_2 + \cdots + W_B h_B.
```

The native A2 head instead uses one time-shared matrix and then temporal mean:

```math
e_t = W z_t
```

```math
score = \frac{1}{T}\sum_t e_t
      = W\left(\frac{1}{T}\sum_t z_t\right).
```

So the native head cannot assign different class meaning to the same L2 feature depending on its temporal bin.

For MM:

```text
L2 communication Fixed250 probe = 63.84%
L2 communication whole-mean probe = 57.98%
native A2 final BA = 56.42%
```

This gives two conceptually separate gaps:

```math
63.84 - 57.98 = 5.86\ \text{pp}
```

from retaining vs removing ordered local temporal position, and then:

```math
57.98 - 56.42 = 1.56\ \text{pp}
```

between a post-hoc frozen whole-mean probe and the native jointly trained A2 head.

Therefore:

> A low native final BA does not imply that the hidden local representation itself is only a 56% representation.

For the current local-representation question, the relevant comparison is primarily the fixed diagnostic protocol:

```math
BB\ L2\ Fixed250 = 61.84\%
```

versus:

```math
MM\ L2\ Fixed250 = 63.84\%.
```

Thus MM currently provides about +2.0 pp more linearly accessible ordered local-temporal information than the original BB hidden stack, even though the native A2 final BA is nearly unchanged.

---

# 8. Threshold choice and its current limitation

The current MT thresholds are:

```text
0.5 x theta0 / 1.0 x theta0 / 1.5 x theta0
```

with baseline:

```text
theta0 = 0.5
```

so:

```text
0.25 / 0.50 / 0.75.
```

These values were selected as a simple symmetric mechanism test:

- keep one population exactly at the original A2 threshold;
- add one more excitable population;
- add one less excitable population;
- avoid mixing the architecture question with an extensive threshold hyperparameter search.

They were **not** selected from an optimal sweep or from activation quantiles.

## L1: the current threshold range creates distinct activity bands

For MT L1, the three thresholds produce visibly different firing regimes. For example, around shift 4 the activity is approximately:

```text
low threshold  (0.25): ~25%
base threshold (0.50): ~9%
high threshold (0.75): ~3%
```

This is consistent with the intended low/mid/high excitability interpretation.

## L2: the same threshold range is poorly separated

For MM L2, firing rates are much closer across the three threshold groups, roughly in the 35-44% range depending on tau. For example, shift 4 is approximately:

```text
0.25 threshold: 43.7%
0.50 threshold: 38.8%
0.75 threshold: 39.4%
```

Therefore L2 operates in a very different current/membrane regime from L1. Simply copying the L1 threshold multipliers into L2 does not produce clearly separated excitability bands.

This means Exp8.1.1 demonstrates the **value of two-layer heterogeneous threshold coding**, but it does **not** establish that `{0.25,0.50,0.75}` is a good final L2 threshold set.

A future local-representation refinement should treat threshold selection as layer-specific, and potentially tau-specific, rather than assuming one global multiplier set.

One principled calibration approach is to use the empirical pre-reset membrane distribution and choose thresholds for target firing probabilities. For example, if desired firing regimes are approximately 30%, 10%, and 3%, thresholds can be selected from corresponding upper-tail quantiles of `U^-` rather than chosen by hand.

---

# 9. Current conclusions, separated by confidence

## Strongly supported by the experiments

### 1. L1 has a real ordered-local communication bottleneck under uniform binary coding

Exp7.3.5 showed a large L1 Fixed250 gap between `U^-` and binary spikes. Exp8.1 then showed that fixed threshold heterogeneity can reduce this gap and increase the linearly accessible communicated L1 representation.

### 2. Weighted cap-31 communication is not useful in the current operating regime simply because cap 31 is available

The trained weighted neurons almost never emit more than one event per timestep. The network does not currently exploit the large event-count alphabet.

### 3. Fixed threshold heterogeneity is more effective than weighted event multiplicity for improving L1 local communication in the tested setup

MT128 improves L1 Fixed250 communication by several percentage points while weighted31 does not.

### 4. A richer L1 MT representation does not automatically survive a uniform-binary L2

This is reproduced in both Exp8.1 and Exp8.1.1 MB.

### 5. L2 MT specifically complements L1 MT

The positive factorial interaction, and particularly `MM - MB`, shows that an MT L2 can recover/reshape the representation from an MT L1 in a way that it does not do for a binary L1.

### 6. Hidden thresholding should be viewed as a representation transform, not only as destructive quantization

L2 frequently shows `spike probe > pre-reset probe`. In MM, the L2 MT threshold stage increases Fixed250 linear accessibility from about 57.56% to 63.84%.

### 7. MM is currently the stronger local hidden representation, even though native final BA is unchanged

At L2 communication Fixed250:

```text
BB: 61.84%
MM: 63.84%
```

but native A2 final BA is:

```text
BB: 56.45%
MM: 56.42%
```

These measure different properties and should not be conflated.

---

## Not established yet

The current experiments do **not** establish that:

- MT improves native end-task classification under the existing A2 readout;
- `{0.25,0.50,0.75}` is an optimal threshold set;
- the same threshold distribution should be used in L1 and L2;
- more threshold groups are necessarily better;
- 384 neurons are required for MT;
- hidden analog membrane should replace spike communication;
- every information gap should be interpreted as destructive quantization.

---

# 10. Current local-representation working model

The most consistent interpretation of the experiments so far is:

```text
raw event train
    |
    v
L1 analog dynamics U1-
    |
    | uniform binary: loses substantial ordered-local accessibility
    | MT population: preserves / reshapes more of it
    v
L1 communicated population
    |
    v
L2 analog dynamics U2-
    |
    | uniform binary L2:
    |     does not make good use of MT-L1 improvement
    |
    | MT L2:
    |     produces a substantially stronger L2 local spike representation
    v
L2 communicated population
```

The best current conceptual statement is:

> The local SNN does not only need multiple synaptic time constants. It also benefits from heterogeneity in neuronal excitability, and that heterogeneous code appears to work best when it is propagated through multiple hidden layers rather than inserted into only one layer.

At the same time, the current L2 threshold groups are not well calibrated to the L2 activity regime, so the full potential of this architecture has not yet been tested.

---

# 11. Source experiments and artifacts

Design / implementation:

```text
scripts/experiment_7_3_5_hidden_state_information_loss/README.md
scripts/experiment_8_1_hidden_quantization_ablation/README.md
scripts/experiment_8_1_1_two_layer_mt_factorial/README.md
```

Exp7.3.5 results:

```text
notebooks/artifacts/experiment_7_3_5_hidden_state_information_loss/
  hidden_state_information_loss_v1/
```

Exp8.1 results:

```text
notebooks/artifacts/experiment_8_1_hidden_quantization_ablation/
  hidden_quantization_ablation_v1/
```

Exp8.1.1 results:

```text
notebooks/artifacts/experiment_8_1_1_two_layer_mt_factorial/
  two_layer_mt_factorial_v1/
```

Primary finalized tables used in this note:

```text
method_summary.csv
contrast_summary.csv
factorial_effect_summary.csv
activity_summary.csv
```

Analysis notebooks:

```text
notebooks/experiment_7_3_5_hidden_state_information_loss.ipynb
notebooks/experiment_8_1_hidden_quantization_ablation.ipynb
notebooks/experiment_8_1_1_two_layer_mt_factorial.ipynb
```
