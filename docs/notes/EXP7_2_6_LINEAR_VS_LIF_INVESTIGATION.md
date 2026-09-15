# Exp7.2.6 series: why a Linear head and LIF output neurons behave differently

This note records the full reasoning chain behind the Exp7.2.6 series. The goal is not only to preserve final numbers, but also to document **why each follow-up experiment was introduced, what hypothesis it tested, what confound was discovered, and how the interpretation changed after each result**.

The central question came from Exp7.2.4:

> Given the same two-layer SNN backbone and the same L2 spike representation, why can a simple Linear readout classify much better than an output layer composed of LIF neurons, and why does training through the output neuron sometimes appear to damage the representation learned before it?

The investigation gradually separated this question into three mechanisms:

1. **Training-loss construction:** does the CE objective itself distort optimization because sequence sums and sequence means have different logit scales?
2. **Readout dynamics:** for a fixed good L2 and a fixed good projection matrix `W`, how much information is lost when continuous evidence is converted into output spikes?
3. **Projection learning:** when `W` is trained through LIF neurons and surrogate gradients, does it become intrinsically less class-discriminative even before the LIF readout is applied?

The current evidence points strongly to a combination of (1) and (3). Direct LIF inference on a good Linear projection causes a moderate loss, but the larger and more stable loss comes from the **projection matrix learned through the LIF training path**.

---

## 1. Starting point: what Exp7.2.4 suggested

The relevant backbone is the two-layer local SNN:

```text
input
  -> L1, shifts (2,3,4), width 128
  -> L2, shifts (2,3,4), width 128
  -> readout
```

For a valid sequence of L2 spikes,

\[
z_t \in \{0,1\}^{128}, \qquad t=1,\ldots,T,
\]

a Linear head produces per-timestep evidence

\[
e_t = W z_t.
\]

An analog whole-sequence classifier can accumulate this evidence as

\[
E = \sum_{t=1}^{T} e_t.
\]

Equivalently, for classification by `argmax`, it can use the mean

\[
\bar E = \frac{1}{T}\sum_{t=1}^{T}e_t,
\]

because multiplication by the positive scalar `T` does not change the predicted class:

\[
\arg\max_k E_k = \arg\max_k \bar E_k.
\]

Exp7.2.4 showed that a whole-sequence Linear readout on L2 can perform very well. For the exact `234x234` analog A2 configuration, the observed test balanced accuracy was about **57.6%**. In contrast, related end-to-end SNN configurations with output neurons were clearly worse.

This created an apparent puzzle:

\[
\boxed{
\text{same L2 information} + \text{simple Linear readout}
\gg
\text{output LIF neuron readout}
}
\]

The initial interpretation was that the output neuron might be discarding useful evidence through thresholding, reset, leakage, or spike-count compression, and that the corresponding gradients might also damage L2 during end-to-end training.

That interpretation was plausible, but the Exp7.2.6 series showed that the situation was more complicated.

---

## 2. The three quantities that must be separated

A useful way to write the problem is

\[
L2 \xrightarrow{W} e_t \xrightarrow{\text{output dynamics}} \text{readout}.
\]

There are therefore three distinct objects that can become worse:

### 2.1 L2 representation quality

Does training through the output layer change the upstream L2 representation so that less class information remains linearly accessible?

### 2.2 Projection quality

Given a fixed L2, does training through LIF produce a worse projection matrix

\[
W_{\rm lif}
\]

than direct Linear training produces

\[
W_{\rm lin}?
\]

### 2.3 Output-dynamics loss

Given the **same** L2 and the **same** `W`, how much accuracy is lost by replacing an analog accumulator with thresholded spiking dynamics?

The purpose of Exp7.2.6 was to separate these terms instead of measuring only the final end-to-end BA.

---

# 3. Exp7.2.6: first decomposition of readout loss and representation shaping

## 3.1 Initial hypothesis

The first Exp7.2.6 design assumed that the Linear-vs-LIF difference could be decomposed into:

- a direct output-neuron readout loss;
- additional loss caused by membrane leakage;
- possible degradation of L2 during end-to-end training through the output neuron.

The design therefore contained two complementary parts.

### Frozen-L2 mechanism analysis

Freeze L2, then train matched `128 -> 12` heads and compare:

```text
W_linear -> Analog
W_linear -> LIF
W_lif    -> Analog
W_lif    -> LIF
```

This 2x2 comparison was intended to separate projection quality from output dynamics.

A mechanism ladder also compared:

```text
Analog
  -> beta=1 IF with charge-preserving readout
  -> beta=1 IF spike count
  -> beta=.5 LIF spike count
```

with extra controls for `cap=31`, bipolar spikes, and a beta sweep.

### End-to-end representation shaping

The original E2E comparison used:

```text
C0: Analog output
C1: IF output, beta=1
C2: LIF output, beta=.5
```

and then evaluated frozen L2 with common probes.

The expectation was that this would reveal whether LIF causes both a readout bottleneck and an upstream representation bottleneck.

---

## 3.2 Original Exp7.2.6 result

For `234x234`, `task_only`, three seeds, the original E2E test balanced accuracies were approximately:

| condition | test BA |
| --- | ---: |
| C0 Analog | 40.39% |
| C1 IF, beta=1 | 22.81% |
| C2 LIF, beta=.5, reused from Exp7.2.5 | 43.21% |

The original frozen 2x2 comparison also showed a smaller but visible Analog-to-LIF drop. For `task_only`:

| frozen head / readout | test BA |
| --- | ---: |
| `Wlin -> Analog` | 37.98% |
| `Wlin -> LIF beta=.5` | 32.67% |
| `Wlif -> Analog` | 34.32% |
| `Wlif -> LIF beta=.5` | 33.48% |

At face value, these results seemed to support the idea that the output neuron causes direct information loss and that the E2E LIF/IF path may also produce poorer upstream representations.

However, the Analog baseline itself was unexpectedly weak compared with Exp7.2.4 A2. That discrepancy forced a re-examination of the training objective.

---

## 3.3 Critical confound discovered after Exp7.2.6

The key issue was that **the training objective in Exp7.2.6 was not equivalent to the objective used by the stronger Exp7.2.4 Analog baseline**.

Exp7.2.4 A2 effectively trained with a valid-time mean before CE:

\[
\mathcal L_{\rm mean}
=
CE\left(
\frac{1}{T}\sum_t l_t,
y
\right).
\]

The original Exp7.2.6 Analog path instead trained on a raw sum:

\[
\mathcal L_{\rm sum}
=
CE\left(
\sum_t l_t,
y
\right).
\]

For a given sample,

\[
\sum_t l_t = T\bar l,
\]

so the two objectives are related by a sample-dependent scale inside softmax:

\[
CE(T\bar l,y) \neq CE(\bar l,y).
\]

This is important because valid sequence length varies across samples. The raw-sum objective therefore injects sequence length into the logit scale / softmax temperature and also changes gradient magnitude.

At inference,

\[
\arg\max \sum_t l_t
=
\arg\max \frac{1}{T}\sum_t l_t,
\]

but **during CE training the two are not equivalent**.

This became the main motivation for Exp7.2.6.1.

---

# 4. Exp7.2.6.1: Sum-CE vs Mean-CE

## 4.1 Question

Was the poor Analog result in original Exp7.2.6 actually caused by the output architecture, or was it mostly caused by training on `sum` logits instead of `mean` logits?

## 4.2 Controlled change

Everything was paired and held fixed:

- same `234x234` backbone;
- `task_only`;
- seeds `11, 23, 37`;
- same model initialization;
- same loader order;
- same optimizer and LR;
- same checkpoint selection;
- same bias-free `128 -> 12` Linear head.

The only training change was:

\[
CE\left(\sum_t Wz_t,y\right)
\quad\rightarrow\quad
CE\left(\frac{1}{T}\sum_tWz_t,y\right).
\]

The CE logit gain for the Analog path remained `1`.

---

## 4.3 Result: Mean-CE almost completely repaired the Analog baseline

Test BA changed from:

\[
40.39\% \rightarrow 54.54\%.
\]

The gain was therefore:

\[
\boxed{+14.14\text{ percentage points}}.
\]

The important sanity check also held for every evaluated sample:

\[
\arg\max \sum_t l_t
=
\arg\max \frac1T\sum_tl_t.
\]

So the improvement was **not** caused by changing the deployment decision rule. It came from changing the training geometry seen by CE.

The original Sum-CE model also showed the characteristic pattern of a poorly conditioned objective: high train BA but weak validation/test generalization. Mean-CE reduced this discrepancy substantially.

---

## 4.4 Mean-CE also repaired L2, not only the final Linear head

The L2 probe results were equally important.

Test BA changed approximately as follows:

| L2 probe | Sum-CE model | Mean-CE model |
| --- | ---: | ---: |
| standardized WholeCount Linear | 42.25% | 55.76% |
| Fixed250 Linear | 47.52% | 59.60% |
| matched bias-free WholeCount | 40.49% | 47.13% |

Therefore the Sum-CE problem was not merely a bad final `W`.

Its gradients propagated into the backbone and changed the representation learned by L2:

\[
\boxed{
\text{loss scaling at the output can reshape and degrade upstream L2}
}
\]

This was the first major correction to the original Exp7.2.6 interpretation.

The canonical Analog baseline for later experiments therefore became:

\[
\boxed{\text{Mean-CE + bias=False}}.
\]

---

# 5. Exp7.2.6.2: does bias explain the remaining gap?

## 5.1 Motivation

After fixing Sum-CE, the Mean-CE/no-bias Analog model reached about 54.54% test BA, while the earlier Exp7.2.4 A2 result for the same `234x234` architecture was still a few percentage points higher.

One obvious implementation difference was output bias:

- Exp7.2.4 A2 used a Linear head with bias;
- Exp7.2.6.1 deliberately kept the matched head bias-free.

The hypothesis was therefore:

\[
\boxed{\text{perhaps bias explains the remaining A2 advantage}.}
\]

---

## 5.2 Experiment

Train the same Mean-CE model with a zero-initialized output bias:

\[
l = W\bar z+b.
\]

Evaluate three conditions:

1. Mean-CE, bias-free reference;
2. bias-trained model with the learned bias active;
3. the same bias-trained model with the bias set to zero at test time.

This allows the total bias effect to be decomposed into:

\[
\Delta_{\rm total}
=
\Delta_{\rm direct}
+
\Delta_{\rm training}.
\]

The direct term asks whether the final offset itself helps predictions. The training term asks whether allowing bias changes the optimization trajectory of `W`, L1, and L2.

---

## 5.3 Result: bias did not explain the gap

Test BA:

| condition | test BA |
| --- | ---: |
| Mean-CE, bias=False | 54.54% |
| Mean-CE, bias=True | 51.69% |
| bias-trained, bias forced to zero at test | 51.35% |

The mean decomposition was:

\[
\Delta_{\rm total}\approx -2.85\text{ pp},
\]

\[
\Delta_{\rm direct}\approx +0.34\text{ pp},
\]

\[
\Delta_{\rm training}\approx -3.18\text{ pp}.
\]

Thus the learned bias itself had almost no direct test benefit. The larger effect came from the fact that **allowing a bias changed the training trajectory and slightly weakened the learned representation/projection**.

The L2 probes moved in the same direction:

| probe | bias=False | bias-trained |
| --- | ---: | ---: |
| standardized WholeCount | 55.76% | 55.18% |
| Fixed250 | 59.60% | 58.00% |
| matched bias-free WholeCount | 47.13% | 44.38% |

Therefore bias was ruled out as the explanation for the major Linear-vs-LIF difference.

At this point the clean reference remained:

\[
\boxed{\text{Mean-CE + bias=False}}.
\]

---

# 6. Exp7.2.6.3: isolate Linear vs LIF on exactly the same L2

## 6.1 Why this experiment was necessary

After Exp7.2.6.1 and .2, two large confounds had been removed:

- raw Sum-CE was known to be bad;
- bias was known not to explain the Linear advantage.

The remaining question could finally be asked directly:

\[
\boxed{
\text{Given exactly the same L2 spike sequence, why does a Linear head outperform a LIF output head?}
}
\]

The key design decision was to **freeze L2 completely**. No end-to-end backbone retraining was allowed in Exp7.2.6.3.

The frozen source was the Exp7.2.6.1 Mean-CE, bias-free `234x234` model.

For every seed, train/val/test L2 spike sequences were cached once and reused by every downstream condition.

---

## 6.2 Important gain convention

A second source of confusion had emerged from the repository's older SNN objectives: some SNN `whole_count_ce` implementations multiply mean firing rate by a fixed `LOGIT_GAIN=5` before CE.

Exp7.2.6.1 and .2 Analog training did **not** use that gain; their effective CE logit gain was `1`.

To avoid introducing another variable, Exp7.2.6.3 forced:

\[
\boxed{G_{CE}=1\text{ for both Linear and LIF head training}.}
\]

The primary LIF input current scale was also fixed at:

\[
\boxed{G_{input}=1}. 
\]

Input-gain calibration, `cap=31`, and bipolar output were kept only as secondary mechanism diagnostics.

---

# 7. Exp7.2.6.3 Part A: same L2, same W, only change output dynamics

The first part uses the already-trained good Linear projection from Exp7.2.6.1:

\[
W = W_{\rm source}.
\]

Both L2 and `W` are frozen. No CE training occurs in this part.

This gives the cleanest possible readout-dynamics test.

---

## 7.1 A0: Analog accumulator

\[
e_t=Wz_t,
\]

\[
E=\sum_t e_t.
\]

Test BA:

\[
\boxed{54.54\%}.
\]

---

## 7.2 A1: beta=1 IF with charge-preserving readout

For a no-leak IF neuron with subtractive reset,

\[
U_t^- = U_{t-1}+e_t,
\]

\[
U_t = U_t^- - \theta S_t.
\]

The final charge identity is

\[
\sum_t e_t = \theta\sum_tS_t + U_T.
\]

Therefore the readout

\[
E_{charge}=\theta N+U_T
\]

should be equivalent to the Analog accumulator.

It was.

Test BA:

\[
\boxed{54.54\%}.
\]

The prediction equality and numerical charge identity held to floating-point precision.

This proves an important point:

\[
\boxed{
\text{the IF state equation itself does not destroy the Linear evidence.}
}
\]

If both spike count and final membrane are available, the Analog evidence is recoverable.

---

## 7.3 A2: beta=1 IF, spike count only

Now discard `U_T` and classify only from

\[
N=\sum_tS_t.
\]

Test BA collapsed to:

\[
\boxed{32.19\%}.
\]

Relative to charge-preserving readout, the apparent spike-conversion loss was about:

\[
\boxed{22.35\text{ pp}}.
\]

At first this looked like strong evidence that converting continuous evidence into spikes destroys class information.

However, the next result changed that interpretation.

---

## 7.4 A3: beta=.5 LIF, spike count only

With the same frozen L2 and the same frozen `W`, only change the membrane factor from

\[
\beta=1
\]

to

\[
\beta=0.5.
\]

The spike-count test BA became:

\[
\boxed{52.78\%}.
\]

This is only about

\[
1.76\text{ pp}
\]

below the Analog reference.

Therefore leakage was not acting as a simple information-loss term. In this regime, leakage actually **rescued the spike-count readout**.

---

## 7.5 Beta sweep: less persistence gave better spike-count classification

Mean test BA across three seeds:

| beta | test BA |
| ---: | ---: |
| 1.00 | 32.19% |
| 0.95 | 40.09% |
| 0.90 | 45.60% |
| 0.80 | 50.33% |
| 0.70 | 51.71% |
| 0.60 | 52.51% |
| 0.50 | 52.78% |

The trend is clear:

\[
\beta \downarrow \Rightarrow BA_{spike-count} \uparrow.
\]

For this output readout, more persistent membrane state was **not** more useful memory.

---

## 7.6 Why beta=1 performs badly with spike count

The firing diagnostics show that beta=1 leaves a large amount of evidence in the residual membrane.

For example, the mean absolute final membrane for the source Linear `W` was on the order of roughly `144-194` for beta=1 in the three seeds, while beta=.5 reduced it to roughly `6-8`.

The fraction of silent output neurons was also much higher for beta=1.

This explains the apparent contradiction:

- `beta=1` preserves total charge very well;
- but the deployment readout observes only spike count;
- a large portion of the class evidence remains in `U_T` rather than becoming observable output spikes.

Thus:

\[
\boxed{
\text{persistent membrane state can preserve information internally while making spike-count decoding worse.}
}
\]

Leakage reduces this residual-state problem and keeps the output dynamics in a regime where more of the useful evidence is expressed through countable spikes.

---

## 7.7 Secondary controls

The secondary diagnostics support the conclusion that the beta=1 IF regime is poorly matched to the amplitude and bandwidth of the Linear evidence.

Mean test BA:

| control | test BA |
| --- | ---: |
| IF beta=1, cap=1, gain=1 | 32.19% |
| IF beta=1, cap=31 | 45.46% |
| IF beta=1, validation-calibrated input gain | 46.12% |
| IF beta=1, bipolar | 32.89% |
| LIF beta=.5, cap=31 | 49.62% |
| LIF beta=.5, bipolar | 52.29% |
| LIF beta=.5, validation-calibrated input gain | 45.49% |

Allowing multiple threshold crossings or reducing current amplitude repairs much of the beta=1 IF failure, which is consistent with a dynamic-range / residual-state problem rather than irreversible loss at the moment evidence enters the neuron.

The bipolar control did not solve the beta=1 problem, so loss of negative evidence alone is not the main explanation.

---

# 8. Exp7.2.6.3 Part B: same frozen L2, let Linear and LIF learn their own W

Part A answered:

> What happens if an already-good Linear projection is passed through output-neuron dynamics?

Part B asks the more important training question:

> If Linear and LIF see exactly the same frozen L2 and start from the same `W_0`, what projection matrix does each training path learn?

L2 remained frozen.

Two paired heads were trained:

### Linear head

\[
\mathcal L_{lin}
=
CE\left(
\frac1T\sum_t W_{lin}z_t,
y
\right).
\]

### LIF head

\[
W_{lif}z_t \rightarrow LIF_{\beta=.5} \rightarrow S_t,
\]

\[
\mathcal L_{lif}
=
CE\left(
\frac1T\sum_t S_t,
y
\right).
\]

Both used:

```text
CE logit gain = 1
bias = False
same initial W
same loader order
same optimizer / LR / weight decay
same checkpoint rule
```

---

## 8.1 Native head performance

Test BA:

| trained head | native test BA |
| --- | ---: |
| Linear | 54.98% |
| LIF | 47.27% |

The native gap is therefore about:

\[
\boxed{7.72\text{ pp}}.
\]

This reproduces the original qualitative observation: even with exactly the same frozen L2 input, training a LIF head yields substantially lower accuracy than training a Linear head.

The 2x2 cross-evaluation explains where that gap comes from.

---

# 9. The critical 2x2 result

After training both heads, evaluate each learned `W` through both Analog and LIF readouts.

Mean test BA:

| learned projection | Analog readout | LIF beta=.5 readout |
| --- | ---: | ---: |
| `W_lin` | **54.98%** | **49.98%** |
| `W_lif` | **47.88%** | **47.27%** |

This is the key result of the entire Exp7.2.6.3 experiment.

---

## 9.1 Direct LIF dynamics loss for a Linear-trained W

For the good Linear projection:

\[
54.98\% \rightarrow 49.98\%.
\]

So applying LIF dynamics directly causes about:

\[
\boxed{5.00\text{ pp}}
\]

of loss in this paired-head experiment.

This is a real readout bottleneck, but it does not explain the entire Linear-vs-LIF training gap.

---

## 9.2 Projection quality of the LIF-trained W

Now compare both projections under the **same Analog readout**:

\[
W_{lin}: 54.98\%,
\]

\[
W_{lif}: 47.88\%.
\]

The projection-quality gap is therefore:

\[
\boxed{7.11\text{ pp}}.
\]

More importantly, this effect is highly consistent across the three seeds:

```text
seed 11: 7.34 pp
seed 23: 6.98 pp
seed 37: 7.00 pp
```

This is much more stable than many of the raw SNN accuracy comparisons.

It means that the LIF training path does not merely produce a good class projection that is later damaged by output spiking.

Instead:

\[
\boxed{
W_{lif}\text{ itself is a less linearly discriminative projection of the same L2 representation.}
}
\]

---

## 9.3 W_lif is already adapted to its LIF dynamics

For `W_lif`:

\[
47.88\% \rightarrow 47.27\%
\]

when switching from Analog to LIF readout.

The additional dynamics loss is only about:

\[
\boxed{0.61\text{ pp}}.
\]

So `W_lif` has learned a projection that is already highly compatible with the LIF threshold/reset dynamics.

This suggests a trade-off:

\[
\boxed{
\text{LIF compatibility improves while continuous class separability decreases.}
}
\]

---

## 9.4 Firing statistics support the adaptation interpretation

When the Linear-trained projection is passed through LIF, the output is relatively aggressive:

- roughly `255` spikes/sample on average;
- mean `|U_T|` around `4.3` across seeds;
- potential multi-crossing fraction around `0.22`.

For the LIF-trained projection:

- roughly `181` spikes/sample;
- mean `|U_T|` around `0.94`;
- potential multi-crossing fraction around `0.07`.

Thus `W_lif` appears to move the evidence into a smaller, more threshold-compatible operating range.

This is consistent with the following mechanism:

\[
\boxed{
\text{surrogate-gradient training through LIF encourages W to adapt to neuron dynamics,}
}
\]

but that adaptation may sacrifice some of the continuous discriminative geometry available to an unconstrained Linear accumulator.

---

# 10. What the Exp7.2.6 series has established so far

The sequence of experiments changed the interpretation substantially.

## 10.1 Result 1: the original Analog failure was mostly a loss-construction bug/confound

Original Exp7.2.6 suggested that Analog E2E itself performed poorly.

Exp7.2.6.1 showed that the main reason was:

\[
\boxed{CE(sum) \neq CE(mean)}.
\]

The deployment argmax was equivalent, but the optimization problem was not.

Fixing this alone increased Analog test BA from about 40.4% to 54.5% and also repaired L2 probe performance.

---

## 10.2 Result 2: bias is not the reason Linear works better

Exp7.2.6.2 showed that adding bias did not recover the remaining gap. Its direct prediction effect was tiny, and allowing bias during training slightly weakened the learned representation/projection.

Therefore:

\[
\boxed{\text{bias is not the important Linear-vs-LIF mechanism}.}
\]

---

## 10.3 Result 3: beta=1 IF can preserve evidence internally while making spike-count decoding terrible

Exp7.2.6.3 Part A showed:

\[
Analog = IF\ charge
\]

but

\[
IF\ spike\ count \ll Analog.
\]

The missing information is largely stored in residual membrane state rather than visible spike count.

Reducing beta toward .5 greatly improves count-based decoding.

Therefore:

\[
\boxed{\text{longer persistence is not automatically useful memory for a spike-count classifier}.}
\]

---

## 10.4 Result 4: a good Linear W survives LIF better than initially expected

Using the fixed Exp7.2.6.1 source `W`, Analog BA was 54.54% and beta=.5 LIF spike-count BA was 52.78%.

Thus in that direct frozen-W setting, the LIF readout itself lost only about 1.8 pp.

In the separately paired head-training experiment, `W_lin -> LIF` lost about 5 pp.

The exact direct dynamics penalty therefore depends on which Linear `W` is being evaluated, but it is **not large enough by itself to explain the full trained Linear-vs-LIF gap**.

---

## 10.5 Result 5: the strongest stable effect is projection degradation under LIF training

The paired-head 2x2 comparison showed:

\[
BA(W_{lin},Analog) \approx 54.98\%,
\]

\[
BA(W_{lif},Analog) \approx 47.88\%.
\]

This approximately 7.1 pp projection-quality gap is highly consistent across seeds.

At the same time,

\[
BA(W_{lif},Analog) \approx BA(W_{lif},LIF),
\]

so the LIF-trained projection is already adapted to the LIF readout.

The current best interpretation is therefore:

\[
\boxed{
\text{LIF training changes the optimization target seen by W and learns a more neuron-compatible but less linearly separable projection.}
}
\]

This is currently a stronger explanation than the simple statement that "the output neuron throws away L2 information at inference."

---

# 11. Important unresolved issue: CE gain = 1 may still bias the LIF optimization geometry

Exp7.2.6.3 deliberately set the CE logit gain to `1` for both Linear and LIF training to remove the repository's previous `LOGIT_GAIN=5` difference.

However, equal numerical gain does **not** imply equal effective softmax scale.

For Linear:

\[
\bar e = \frac1T\sum_tWz_t
\]

is an unbounded continuous logit vector.

For binary LIF output:

\[
\bar S = \frac1T\sum_tS_t
\]

is bounded:

\[
0 \le \bar S_k \le 1.
\]

With 12 classes and gain `1`, even the extreme firing-rate logit vector

\[
[1,0,\ldots,0]
\]

can only assign the top class a softmax probability of roughly 0.20.

Thus the LIF training objective at gain `1` remains in a relatively flat softmax regime even though the Linear and LIF code now multiply their logits by the same scalar.

This creates the next question:

\[
\boxed{
\text{Is the poorer }W_{lif}\text{ caused mainly by CE logit-scale conditioning, or by the LIF/surrogate-gradient path itself?}
}
\]

This is the motivation for Exp7.2.6.4.

---

# 12. Planned Exp7.2.6.4: CE-scale vs LIF-gradient geometry

Exp7.2.6.4 should continue using the same frozen Exp7.2.6.1 L2 cache and keep neuron dynamics fixed:

```text
architecture: 234x234
regularization: task_only
seeds: 11, 23, 37
bias: False
beta: .5
cap: 1
LIF input gain: 1
```

The only major sweep should be CE logit gain:

\[
G_{CE}\in\{1,2,5,10\}.
\]

For Linear:

\[
\mathcal L_{lin}^{(G)}
=
CE\left(G\frac1T\sum_tWz_t,y\right).
\]

For LIF:

\[
\mathcal L_{lif}^{(G)}
=
CE\left(G\frac1T\sum_tS_t,y\right).
\]

All conditions should share the same initial `W` and loader order for a given seed.

Every trained projection should then be cross-evaluated through:

1. Analog accumulation;
2. LIF beta=.5 spike-count readout.

The primary quantity is not simply native LIF BA. It is:

\[
\boxed{BA_{Analog}(W_{lif}^{(G)})}.
\]

If increasing CE gain causes this value to recover from roughly 48% toward the roughly 55% Linear level, then the current projection degradation is mainly a CE-temperature / conditioning problem.

If Analog separability remains around 48% across gain despite better-calibrated CE, then the stronger interpretation is that thresholding and surrogate-gradient dynamics change the **direction**, not just the magnitude, of the useful gradient for `W`.

A particularly useful diagnostic is therefore the gradient cosine similarity at identical `W_0`:

\[
\cos\left(
\nabla_W\mathcal L_{lin},
\nabla_W\mathcal L_{lif}
\right).
\]

Together with gradient norms, this can distinguish:

- a gradient-scale problem;
- a fundamentally different gradient direction induced by the LIF path.

Exp7.2.6.4 is currently a planned experiment; no result should be inferred until it is run.

---

# 13. Current causal picture

The current evidence can be summarized as the following chain.

```text
Exp7.2.4:
Linear on L2 works much better than output-neuron variants.
        |
        v
Exp7.2.6:
Try to decompose readout loss, leakage, W quality, and L2 shaping.
        |
        +--> discover Analog baseline itself is unexpectedly weak
        |
        v
Exp7.2.6.1:
Raw Sum-CE -> Mean-CE
40.39% -> 54.54%
=> sequence-dependent CE scaling was a major confound and also damaged L2.
        |
        v
Exp7.2.6.2:
Add output bias
54.54% -> 51.69%
=> bias does not explain the Linear advantage.
        |
        v
Exp7.2.6.3 Part A:
Same L2, same good W
Analog 54.54
IF charge 54.54
IF count 32.19
LIF beta=.5 count 52.78
=> beta=1 residual membrane hides evidence from spike-count readout;
   leakage can improve observability rather than simply destroy information.
        |
        v
Exp7.2.6.3 Part B:
Same frozen L2, paired W training
Wlin + Analog = 54.98
Wlin + LIF    = 49.98
Wlif + Analog = 47.88
Wlif + LIF    = 47.27
=> largest stable effect is that LIF training learns a poorer Analog projection.
        |
        v
Exp7.2.6.4 planned:
Is W_lif degradation caused by bounded firing-rate CE scale,
or by LIF/surrogate-gradient geometry itself?
```

---

# 14. Practical conclusions for later experiments

Until Exp7.2.6.4 resolves the remaining CE-gain question, the following rules should be treated as the cleanest working protocol.

### Do not use raw sequence sums directly inside CE when valid length varies

Use a valid-time mean / firing rate before CE unless a deliberate length-dependent temperature is part of the experiment.

The deployment readout can still use sums/counts because positive normalization does not change `argmax`.

### Do not attribute Linear-vs-LIF accuracy gaps directly to leakage

For the current output layer, beta=.5 performs **better** than beta=1 under spike-count readout because beta=1 leaves too much class evidence in unobserved residual membrane state.

### Separate projection quality from neuron readout quality

Whenever comparing Linear and LIF heads, use the 2x2 diagnostic:

```text
W_linear x {Analog, LIF}
W_lif    x {Analog, LIF}
```

Without this cross-check, a lower native LIF accuracy cannot tell whether the problem is:

- the learned projection;
- output dynamics;
- or both.

### Preserve frozen-L2 experiments before returning to E2E

The frozen-L2 series has made the causal interpretation much cleaner. End-to-end experiments should only be revisited after the head-level mechanism is resolved, otherwise upstream representation changes will again be mixed with readout effects.

---

# 15. Reproducibility map

## Exp7.2.6

Main script:

```text
scripts/experiment_7_2_6_output_readout_loss_shaping.py
```

Notebook:

```text
notebooks/experiment_7_2_6_output_readout_loss_shaping.ipynb
```

Artifacts:

```text
notebooks/artifacts/experiment_7_2_6_output_readout_loss_shaping/
    output_readout_loss_shaping_v1/
```

Important aggregate files include:

```text
e2e_performance_summary.csv
crosscheck_2x2_summary.csv
beta_sweep_summary.csv
e2e_l2_probe_summary.csv
```

## Exp7.2.6.1

Script:

```text
scripts/experiment_7_2_6_1_sum_vs_mean_ce.py
```

Artifacts:

```text
notebooks/artifacts/experiment_7_2_6_1_sum_vs_mean_ce/
    sum_vs_mean_ce_v1/
```

Important aggregate files:

```text
native_performance_summary.csv
paired_delta_summary.csv
l2_probe_summary.csv
```

## Exp7.2.6.2

Script:

```text
scripts/experiment_7_2_6_2_mean_ce_bias.py
```

Artifacts:

```text
notebooks/artifacts/experiment_7_2_6_2_mean_ce_bias/
    mean_ce_bias_v1/
```

Important aggregate files:

```text
native_performance_summary.csv
bias_effect_decomposition_summary.csv
l2_probe_summary.csv
```

## Exp7.2.6.3

Script:

```text
scripts/experiment_7_2_6_3_linear_vs_lif_frozen_l2.py
```

Notebook:

```text
notebooks/experiment_7_2_6_3_linear_vs_lif_frozen_l2.ipynb
```

Artifacts:

```text
notebooks/artifacts/experiment_7_2_6_3_linear_vs_lif_frozen_l2/
    linear_vs_lif_frozen_l2_v1/
```

Important aggregate files:

```text
mechanism_summary.csv
mechanism_decomposition_runs.csv
beta_sweep_summary.csv
secondary_controls_summary.csv
head_summary.csv
crosscheck_summary.csv
crosscheck_decomposition_runs.csv
crosscheck_firing_runs.csv
```

---

# 16. One-sentence status

The current Exp7.2.6-series conclusion is:

\[
\boxed{
\textbf{The large Linear-vs-LIF gap is not explained by one simple LIF inference loss.}
\textbf{After correcting CE normalization, the strongest remaining effect is that LIF-based training learns a projection }W_{lif}\textbf{ that is about 7 pp less linearly separable than }W_{lin},\textbf{ while being well adapted to LIF dynamics.}
}
\]

The next unresolved question is whether that projection loss comes primarily from **bounded firing-rate CE scale** or from the **threshold/surrogate-gradient optimization geometry** itself.
