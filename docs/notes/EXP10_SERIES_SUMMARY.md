# Exp10 Series Summary — Airborne Nuisance, Binary Communication, Supervision, and Membrane-Memory Bottlenecks

## 1. Purpose

The Exp10 series was designed to move beyond the question:

> What A2 configuration gives the highest test BA?

and instead diagnose **where classification information is lost or misused** along the current two-layer SNN pipeline.

The series progressively isolates three factors:

1. **Exp10.0 — input nuisance / preprocessing**
   - Does airborne / repositioning motion help classification, or is it mostly nuisance?
   - Does the encoder still benefit from the full motion history even if airborne spikes are removed before the SNN?

2. **Exp10.1 — binary coding and supervision**
   - Does multi-threshold coding improve hidden binary communication?
   - Does cooperative L1/L2 supervision organize the backbone more usefully?
   - Does a weak L1 timestep-CE auxiliary objective improve local representation?

3. **Exp10.2 — L1 membrane memory**
   - Is the current \(\tau_{mem}\approx22.5\) ms too short?
   - Can longer membrane integration preserve subthreshold evidence long enough to convert it into useful spikes?
   - Does the benefit survive end-to-end retraining and the current time-shared readout?

The integrated working picture is:

~~~text
input nuisance
    ↓
L1 analog representation
    ↓ threshold / binary communication
L1 spikes
    ↓ L1→L2 transformation
L2 temporally structured representation
    ↓ time collapse / time-shared readout
native Linear decision
    ↓ output-LIF realization
pure-SNN output
~~~

The main distinction throughout this note is:

~~~text
input quality
    !=
hidden representation quality
    !=
binary transmission quality
    !=
temporal-readout alignment
    !=
output-LIF realization quality
~~~

---

## 2. Common backbone and notation

The Exp10 series is centered on the Exp7.3/Exp8 A2 backbone:

~~~text
30 event channels
    -> L1: 128 spiking neurons
    -> L2: 128 spiking neurons
    -> bias-free Linear readout
    -> 12 classes
~~~

The standard local synaptic shift groups are:

\[
L1:\ (2,3,4),
\]

\[
L2:\ (2,3,4).
\]

The original membrane constant is approximately:

\[
\tau_{mem}=22.54\text{ ms},
\]

which at 64 Hz corresponds to:

\[
\beta\approx0.49997.
\]

Unless explicitly changed in Exp10.2, the native classification objective uses a valid-length time-shared readout:

\[
\bar z_2
=
\frac1T\sum_{t<T}z_t^{(2)},
\]

\[
s=W_2\bar z_2,
\]

\[
\mathcal L=CE(s,y).
\]

### 2.1 Hidden-state probes

The series repeatedly uses frozen post-hoc Linear probes.

For example, the L1 pre-reset trajectory is:

\[
U_{1,t}^- \in \mathbb R^{128},
\]

where \(U_t^-\) is the membrane value **after membrane integration but before threshold/reset**.

A Fixed250 probe keeps ordered 250 ms bins:

\[
r=
[
\bar U^-_1;
\bar U^-_2;
\ldots;
\bar U^-_B
].
\]

A separate Linear classifier is trained on this frozen feature.

Therefore probe BA answers:

> **How much class information is linearly decodable from this representation under this aggregation?**

It does **not** mean the native SNN can already achieve that BA.

In particular:

\[
BA(L1\ pre\text{-}reset)
-
BA(L1\ spike)
\]

is a useful diagnostic of threshold/binary-communication loss, but it is not an information-theoretic decomposition.

---

# 3. Exp10.0 — Airborne-motion ablation

## 3.1 Question

The first question was whether motion outside the writing interval is useful evidence or nuisance.

Let:

- \(a\): full acceleration trajectory;
- \(m\): press-to-lift writing mask;
- \(Encoder(\cdot)\): the event encoder.

Three dataset variants were defined.

### D0 — original

\[
\boxed{
D0=Encoder(a)
}
\]

The SNN sees all encoded motion, including airborne/repositioning periods.

### D1 — post-encode mask

\[
\boxed{
D1=m\cdot Encoder(a)
}
\]

The encoder still sees the full acceleration history, but explicit spike events outside the writing interval are zeroed before entering the SNN.

### D2 — mask acceleration before re-encoding

\[
\boxed{
D2=m\cdot Encoder(m\cdot a)
}
\]

Airborne acceleration is removed before encoding, so it cannot affect either explicit airborne events or the encoder's temporal state/history.

This gives three useful contrasts:

\[
D0-D1:
\]

effect of **explicit airborne/repositioning spikes seen by the SNN**;

\[
D1-D2:
\]

effect of airborne acceleration on the **writing-period representation through encoder temporal context**;

\[
D0-D2:
\]

total effect of removing airborne acceleration from the representation pipeline.

## 3.2 Protocol

Exp10.0 is the most statistically broad experiment in the current Exp10 series:

- combined actions: 0 and 1;
- A2 234x234;
- 5 cross-user rotations;
- seeds: 11, 23, 37;
- 45 total training runs;
- the three seeds are averaged within each rotation;
- the five rotations are the primary statistical units.

This statistical scope is stronger than Exp10.1 and Exp10.2, which use one locked split.

## 3.3 Native results

| Variant | Native test BA | Output-LIF test BA |
|---|---:|---:|
| D0 original | 45.44% | 41.12% |
| **D1 post-encode mask** | **47.81%** | **41.32%** |
| D2 pre-encode mask + re-encode | 44.01% | 37.65% |

Paired native contrasts across the five rotations:

\[
D1-D0
\approx
\boxed{+2.37\text{ pp}},
\]

\[
D1-D2
\approx
\boxed{+3.79\text{ pp}}.
\]

Thus D1 was the best of the three preprocessing variants.

## 3.4 Input and hidden probes

### Input Fixed250

| Variant | Input Fixed250 test BA |
|---|---:|
| D0 | 59.59% |
| **D1** | **62.77%** |
| D2 | 56.41% |

D1 improves over D0 by about:

\[
\boxed{+3.18\text{ pp}}.
\]

D1 improves over D2 by about:

\[
\boxed{+6.36\text{ pp}}.
\]

### L1 pre-reset Fixed250

| Variant | Test BA |
|---|---:|
| D0 | 64.05% |
| **D1** | **66.78%** |
| D2 | 63.95% |

### L1 spike Fixed250

| Variant | Test BA |
|---|---:|
| D0 | 56.31% |
| **D1** | **57.35%** |
| D2 | 56.87% |

### L2 spike Fixed250

| Variant | Test BA |
|---|---:|
| D0 | 57.61% |
| **D1** | **59.39%** |
| D2 | 57.69% |

### L2 spike whole-count

| Variant | Test BA |
|---|---:|
| D0 | 50.60% |
| **D1** | **52.55%** |
| D2 | 50.03% |

## 3.5 Interpretation

The strongest interpretation is **not**:

> airborne motion contains no useful information.

The data support a more specific statement:

> **Explicit airborne/repositioning event spikes presented to the SNN are, on average, more nuisance-like than useful for cross-user classification, but removing airborne acceleration before the encoder is too aggressive.**

D1 is better than D0 because it removes explicit airborne-period spikes from the SNN input.

But D2 is worse than D1, which suggests that the encoder can still use the full acceleration history to construct a better representation during the writing period.

Conceptually:

~~~text
full acceleration history
        ↓
event encoder
        ↓
keep only writing-period events
        ↓
SNN
~~~

works better than:

~~~text
erase airborne acceleration first
        ↓
event encoder with truncated context
        ↓
SNN
~~~

Therefore D1 became the default candidate input for later Exp10 experiments.

---

# 4. Exp10.1 — Coding × supervision factorial

## 4.1 Motivation

Exp10.0 exposed two additional gaps:

1. L1 pre-reset analog state was substantially more decodable than L1 binary spikes.
2. L2 Fixed250 probes were stronger than L2 whole/native readouts.

Exp10.1 focused on the first part while deliberately keeping the final readout **time-shared**.

The experiment asks whether two mechanisms help:

- **multi-threshold coding**: improve binary population coding;
- **stronger supervision**: organize hidden representation more effectively.

## 4.2 Protocol

Exp10.1 uses:

- datasets: D0 and D1;
- one locked cross-user split: rotation 0;
- coding:
  - BB: binary L1 + binary L2;
  - MM: MT3 L1 + MT3 L2;
- objectives:
  - L2 WCCE baseline;
  - cooperative L1+L2 Joint;
  - L2 WCCE + \(0.1\) L1 TSCE;
- seeds: 11, 23, 37;
- 36 total runs;
- time-shared readout only.

Important statistical caution:

> Exp10.1 uses one user split. The three seeds are paired optimization replicates, not independent user-split replicates.

## 4.3 Objective definitions

### O0 — L2 WCCE baseline

\[
\boxed{
\mathcal L_{base}
=
CE(W_2\bar z_2,y)
}
\]

### O1 — cooperative Joint

\[
\boxed{
\mathcal L_{joint}
=
CE(W_1\bar z_1+W_2\bar z_2,y)
}
\]

The key point is that L1 and L2 cooperate on **one** decision.

This is not the same as forcing both layers to independently classify the sample.

### O2 — L2 WCCE + 0.1 L1 TSCE

\[
\boxed{
\mathcal L
=
CE(W_2\bar z_2,y)
+
0.1
\frac1T
\sum_{t<T}
CE(W_1z_{1,t},y)
}
\]

The L1 TSCE head is training-only; native inference still uses the L2 time-shared branch.

## 4.4 Coding definitions

### BB

Both hidden layers use the normal binary threshold:

\[
s_t=\mathbf1[U_t^-\ge\theta].
\]

### MM

Both hidden layers use MT3 threshold heterogeneity:

\[
\theta_i
\in
\{0.5\theta,\ 1.0\theta,\ 1.5\theta\}.
\]

Each neuron is still binary. MT changes population coding, not spike precision.

## 4.5 Native results

| Dataset | Coding | Objective | Test BA |
|---|---|---|---:|
| D0 | BB | WCCE | 47.38 ± 3.69% |
| D0 | BB | Joint | 48.65 ± 2.55% |
| D0 | BB | +0.1 L1 TSCE | 46.85 ± 3.06% |
| D0 | MM | WCCE | 48.32 ± 5.29% |
| D0 | MM | Joint | **50.33 ± 1.20%** |
| D0 | MM | +0.1 L1 TSCE | 47.59 ± 0.64% |
| D1 | BB | WCCE | **52.09 ± 1.08%** |
| D1 | BB | Joint | 51.99 ± 2.53% |
| D1 | BB | +0.1 L1 TSCE | 50.45 ± 1.73% |
| D1 | MM | WCCE | 49.84 ± 1.99% |
| D1 | MM | Joint | **53.10 ± 2.94%** |
| D1 | MM | +0.1 L1 TSCE | **51.63 ± 0.73%** |

The strongest native condition was:

\[
\boxed{
D1+MM+Joint
=
53.10\%
}
\]

on this locked split.

## 4.6 D1 remains beneficial

Across the matched coding/objective conditions, D1 generally outperformed D0.

For example, under BB+WCCE:

\[
D1-D0
\approx
\boxed{+4.71\text{ pp}}.
\]

This independently supports the Exp10.0 conclusion that D1 is a useful preprocessing choice.

Because this is only one user split, Exp10.0 remains the stronger evidence for the dataset-level conclusion.

## 4.7 MT alone does not improve native A2

On D1 with baseline WCCE:

\[
BB:
52.09\%,
\]

\[
MM:
49.84\%.
\]

Thus:

\[
\boxed{
MM-BB\approx-2.25\text{ pp}
}
\]

for native test BA.

Therefore MT cannot be described as a direct accuracy improvement.

However, its hidden-representation effect is different.

D1 baseline L2 Fixed250:

\[
BB:
59.20\%,
\]

\[
MM:
62.02\%.
\]

So MT improved L2 temporally resolved decodability by about:

\[
\boxed{+2.82\text{ pp}}
\]

while native BA decreased.

This is an important separation:

\[
\boxed{
\text{better temporal hidden representation}
\neq
\text{better current time-shared classifier}
}
\]

## 4.8 Joint and TSCE interact with MT

On D1:

### Joint

BB:

\[
52.09\rightarrow51.99\%
\]

is essentially neutral.

MM:

\[
49.84\rightarrow53.10\%
\]

gains about:

\[
\boxed{+3.26\text{ pp}}.
\]

The MT × Joint interaction is approximately:

\[
\boxed{+3.36\text{ pp}}.
\]

### L1 TSCE

BB:

\[
52.09\rightarrow50.45\%
\]

decreases.

MM:

\[
49.84\rightarrow51.63\%
\]

gains about:

\[
\boxed{+1.79\text{ pp}}.
\]

The MT × L1-TSCE interaction is approximately:

\[
\boxed{+3.43\text{ pp}}.
\]

Thus the useful question is not simply:

> Is MT good?

but rather:

\[
\boxed{
\text{Does MT make a representation that a stronger supervision objective can exploit?}
}
\]

The current evidence says yes.

## 4.9 Information-path view

For the D1 BB WCCE baseline:

\[
L1\ pre\text{-}reset:
70.12\%,
\]

\[
L1\ spike:
63.62\%,
\]

\[
L2\ spike\ Fixed250:
59.20\%,
\]

\[
L2\ spike\ whole:
53.95\%,
\]

\[
Native:
52.09\%.
\]

The L1 pre-reset → spike gap is:

\[
\boxed{-6.50\text{ pp}}.
\]

This observation directly motivated Exp10.2.

### D1 MM + Joint

\[
L1\ pre:
70.87\%,
\]

\[
L1\ spike:
64.17\%,
\]

\[
L2\ Fixed250:
63.64\%,
\]

\[
L2\ whole:
55.53\%,
\]

\[
Native:
53.10\%.
\]

### D1 MM + L1 TSCE

\[
L1\ pre:
70.97\%,
\]

\[
L1\ spike:
64.80\%,
\]

\[
L2\ Fixed250:
\boxed{64.10\%},
\]

\[
L2\ whole:
53.93\%,
\]

\[
Native:
51.63\%.
\]

The TSCE case has a particularly important interpretation:

> **It produces the strongest L2 temporally resolved representation in Exp10.1, but the current time-shared head does not convert that gain into the strongest native BA.**

## 4.10 Joint branch removal

For D1 + MM + Joint, the approximate mean branch-removal result was:

\[
Full:
53.10\%,
\]

\[
L2\ only:
49.50\%,
\]

\[
L1\ only:
14.5\%.
\]

Thus removing the L1 branch costs approximately:

\[
\boxed{3.6\text{ pp}}.
\]

The important point is that L1 does not need to be a strong standalone classifier.

Its direct branch can provide **complementary / residual evidence** that corrects L2 errors.

Therefore:

\[
\boxed{
\text{low standalone L1 accuracy does not imply zero joint utility}
}
\]

## 4.11 Exp10.1 conclusion

The strongest conclusions are:

1. D1 remains the better input.
2. MT alone improves some hidden temporal probes but does not improve native BA.
3. Joint and L1-TSCE become more useful when paired with MT.
4. Hidden-representation gains can remain invisible to a time-shared readout.
5. The L1 analog→binary gap remains substantial even with MT.

This led directly to the next question:

> **Can longer membrane integration convert weak subthreshold L1 evidence into more useful spikes?**

---

# 5. Exp10.2 — L1 membrane-memory sweep

## 5.1 Motivation

At 64 Hz, the baseline L1 membrane constant is only:

\[
\tau_{mem}=22.54\text{ ms}.
\]

With:

\[
\beta\approx0.5,
\]

subthreshold membrane evidence decays very quickly.

A possible failure mode is:

~~~text
useful analog membrane amplitude
        ↓ fast decay before threshold crossing
no spike
        ↓
information never reaches L2
~~~

Exp10.2 tests whether longer L1 membrane memory can convert part of this analog evidence into spike timing/count evidence.

## 5.2 Controlled sweep

Only the L1 membrane decay changes.

Everything else is fixed:

- D1 only;
- BB coding;
- WCCE;
- time-shared readout;
- L1/L2 synaptic shifts (2,3,4);
- L2 membrane shift fixed at 1;
- one locked cross-user split;
- seeds 11, 23, 37.

The tested L1 membrane constants are:

| shift_mem | \(\beta\) | \(\tau_{mem}\) |
|---:|---:|---:|
| 1 | ~0.499968 | 22.54 ms |
| 2 | 0.75 | 54.31 ms |
| 3 | 0.875 | 117.01 ms |
| 4 | 0.9375 | 242.10 ms |

---

# 6. Exp10.2 Stage A — Frozen-weight dynamics replay

## 6.1 Purpose

Stage A isolates neuron dynamics.

The shift-1 network is trained normally.

Then **all learned weights are frozen**, and the same checkpoint is replayed with only:

\[
\beta_{L1}
\]

changed.

This asks:

> Can longer membrane memory itself make L1 spike representation more decodable, without learning new weights?

## 6.2 Frozen replay results

| L1 shift | \(\tau_{mem}\) | Native BA | L1 pre-reset | L1 spike | Spike − pre gap | L2 Fixed250 | L2 whole |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.54 ms | 52.09% | 70.12% | 63.62% | -6.50 pp | 59.20% | 53.95% |
| 2 | 54.31 ms | 37.73% | 72.60% | 67.03% | -5.57 pp | 64.03% | 53.79% |
| **3** | **117.01 ms** | 26.84% | **73.23%** | **69.21%** | **-4.02 pp** | 64.25% | 56.29% |
| 4 | 242.10 ms | 21.61% | 72.42% | 67.24% | -5.18 pp | **67.01%** | 54.66% |

## 6.3 Stage-A interpretation

The original subthreshold-recovery hypothesis is supported.

At shift 3:

\[
L1\ spike:
63.62\rightarrow69.21\%,
\]

a gain of approximately:

\[
\boxed{+5.60\text{ pp}}.
\]

The gap changes from:

\[
-6.50
\]

to:

\[
-4.02\text{ pp},
\]

an improvement of approximately:

\[
\boxed{+2.48\text{ pp}}.
\]

Therefore longer membrane integration can indeed convert part of the previously hidden analog state into a more linearly decodable spike representation.

However, native accuracy collapses when \(\tau_{mem}\) is changed without retraining.

For example:

\[
52.09\%
\rightarrow
26.84\%
\]

at shift 3.

This is not evidence that the representation becomes worse.

At the same time:

\[
L2\ Fixed250:
59.20\%\rightarrow64.25\%.
\]

The correct interpretation is:

\[
\boxed{
\text{changing neuron dynamics changes the spike code, so the old learned weights/readout become misaligned}
}
\]

The network therefore needs end-to-end adaptation to the new dynamics.

---

# 7. Exp10.2 Stage B — End-to-end retraining

## 7.1 Main results

| L1 shift | \(\tau_{mem}\) | Native BA | Output-LIF BA | L1 pre | L1 spike | Spike − pre gap | L2 Fixed250 | L2 whole |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.54 ms | 52.09% | 44.64% | 70.12% | 63.62% | -6.50 pp | 59.20% | 53.95% |
| **2** | **54.31 ms** | **55.69%** | **46.79%** | 74.18% | **68.49%** | -5.70 pp | 63.40% | **59.75%** |
| 3 | 117.01 ms | 54.55% | 45.21% | **74.87%** | 68.27% | -6.60 pp | **64.54%** | 57.80% |
| 4 | 242.10 ms | 51.96% | 43.62% | 72.79% | 67.28% | -5.51 pp | 61.93% | 56.56% |

Relative to shift 1, shift 2 improves native test BA by:

\[
\boxed{+3.60\text{ pp}}
\]

with paired-seed standard deviation of approximately:

\[
0.83\text{ pp}.
\]

All three seeds improve.

## 7.2 What longer membrane memory is actually doing

The E2E result shows that the gain is **not only** a reduced analog→spike gap.

For shift 2:

\[
L1\ pre:
70.12\rightarrow74.18,
\]

\[
L1\ spike:
63.62\rightarrow68.49.
\]

So both the analog representation and transmitted spike representation become stronger.

The gap shrinks only modestly:

\[
6.50\rightarrow5.70\text{ pp}.
\]

Therefore the broader mechanism is better described as:

\[
\boxed{
\text{longer temporal integration}
\rightarrow
\text{richer L1 dynamics}
\rightarrow
\text{stronger binary spike representation}
\rightarrow
\text{stronger downstream L2 representation}
}
\]

rather than simply:

\[
\text{quantization gap fixed}.
\]

## 7.3 Best time-shared backbone vs best temporal backbone

The result is especially informative when comparing shift 2 and shift 3.

### Shift 2

\[
L2\ Fixed250=63.40\%,
\]

\[
L2\ whole=59.75\%,
\]

\[
Native=55.69\%.
\]

### Shift 3

\[
L2\ Fixed250=\boxed{64.54\%},
\]

\[
L2\ whole=57.80\%,
\]

\[
Native=54.55\%.
\]

Thus:

\[
\boxed{
shift2
=
\text{best current time-shared backbone}
}
\]

while:

\[
\boxed{
shift3
=
\text{strongest E2E temporally resolved L2 representation in Exp10.2}
}
\]

This is another example of:

\[
\boxed{
\text{best hidden temporal representation}
\neq
\text{best current native readout}
}
\]

## 7.4 Shift 4 is too long for the current local backbone

At:

\[
\tau_{mem}\approx242\text{ ms},
\]

native BA returns to approximately baseline:

\[
51.96\%.
\]

L2 Fixed250 also falls from the shift-3 peak:

\[
64.54\rightarrow61.93\%.
\]

A plausible working interpretation is **temporal smearing**: local phases become less distinct when the membrane carries too much state across time.

This is an interpretation rather than a directly proven causal mechanism.

## 7.5 This is not the old long-\(\tau_{syn}\) sustained-firing failure

The activity diagnostics show no near-total firing saturation in the E2E sweep:

\[
\boxed{
fraction\ of\ neurons\ firing\ on\ge95\%\ valid\ timesteps
=
0
}
\]

for all tested membrane shifts.

Longer membrane constants also reduce dead-neuron fractions in several L1 synaptic-timescale groups.

For the shortest L1 synaptic group, dead-neuron fraction approximately changes:

\[
12.4\%
\rightarrow
4.7\%
\rightarrow
3.1\%
\rightarrow
0.8\%.
\]

Therefore increasing \(\tau_{mem}\) is behaving differently from the previously observed long-\(\tau_{syn}\) persistent-current failure.

A useful current distinction is:

\[
\boxed{
\text{moderately longer }\tau_{mem}
\text{ can provide temporal integration without the same sustained-current pathology}
}
\]

under this architecture and sweep.

---

# 8. Integrated Exp10 interpretation

The three experiments form one causal chain.

## 8.1 Exp10.0: clean the evidence source

D1 says:

\[
\boxed{
\text{remove explicit airborne spike evidence,
but retain full motion history inside the encoder}
}
\]

This improves both input probes and native classification.

## 8.2 Exp10.1: organize and encode hidden evidence

Exp10.1 shows:

\[
\boxed{
\text{coding and supervision interact}
}
\]

MT alone is not sufficient.

Joint/TSCE alone are not universally beneficial.

But MT + stronger supervision can create substantially better temporally resolved L2 representations.

## 8.3 Exp10.2: give weak local evidence enough time to become spikes

Exp10.2 shows:

\[
\boxed{
22.5\text{ ms membrane memory is probably too short for the current L1}
}
\]

and that:

\[
\boxed{
\tau_{mem}^{L1}\approx54\text{ ms}
}
\]

is a better operating point for the current D1 + BB + WCCE + time-shared system.

---

# 9. Representative information path on the locked Exp10.1/10.2 split

For the D1 BB baseline:

~~~text
L1 pre-reset Fixed250     70.12
        ↓
L1 binary spike          63.62
        ↓
L2 Fixed250              59.20
        ↓ time collapse
L2 whole                 53.95
        ↓
Native                    52.09
        ↓ output LIF
Output-LIF                44.64
~~~

After changing only the L1 membrane design and retraining at shift 2:

~~~text
L1 pre-reset Fixed250     74.18
        ↓
L1 binary spike          68.49
        ↓
L2 Fixed250              63.40
        ↓ time collapse
L2 whole                 59.75
        ↓
Native                    55.69
        ↓ output LIF
Output-LIF                46.79
~~~

This makes the current bottlenecks easier to separate.

### Binary communication

Still approximately:

\[
74.18-68.49
\approx
\boxed{5.7\text{ pp}}.
\]

### Temporal collapse

At shift 3:

\[
64.54-57.80
\approx
\boxed{6.75\text{ pp}}.
\]

### Output realization

At the best shift-2 native condition:

\[
55.69-46.79
\approx
\boxed{8.9\text{ pp}}
\]

remain between analog Linear classification and output-LIF realization.

These values are **diagnostic BA differences**, not additive information-theoretic attribution percentages.

---

# 10. What the Exp10 series supports

## 10.1 D1 should be the default preprocessing candidate

\[
\boxed{
D1=m\cdot Encoder(a)
}
\]

is currently better supported than either original D0 or the more aggressive D2.

## 10.2 The A2 backbone is not limited to ~50–55% decodable information

Hidden probes repeatedly reach:

\[
60\%-70\%+
\]

depending on layer/state/readout.

Therefore low native BA should not automatically be interpreted as:

> the hidden layers failed to learn class information.

A more accurate decomposition includes coding and readout bottlenecks.

## 10.3 Binary threshold communication is a real bottleneck

L1 pre-reset representations can be substantially more decodable than L1 binary spikes.

Longer \(\tau_{mem}\) helps, but does not eliminate the gap.

## 10.4 MT is an interaction mechanism, not a standalone accuracy trick

MT can improve temporally resolved hidden representation while leaving or hurting native BA.

Its value becomes clearer when paired with Joint or TSCE supervision.

## 10.5 Current time-shared readout discards useful temporal structure

Across Exp10.1 and Exp10.2:

\[
L2\ Fixed250
>
L2\ whole
\]

by meaningful margins.

This aligns with the earlier Exp8.0.5 evidence that correct phase/temporal alignment can unlock information that time-shared aggregation cannot use.

## 10.6 Moderate membrane memory is better than the original very-fast membrane

For the current time-shared system:

\[
\boxed{
54\text{ ms}
}
\]

is clearly stronger than:

\[
22.5\text{ ms}
\]

on the locked split.

Longer is not automatically better.

---

# 11. What the Exp10 series does not yet prove

## 11.1 Exp10.1/10.2 are not population-level cross-user estimates

They use one locked split.

The three seeds are optimization replicates.

Any new "best model" should be rerun across the five cross-user rotations before being treated as a robust generalization result.

## 11.2 Probe BA is not native achievable BA

A Fixed250 post-hoc Linear probe gets access to an ordered high-dimensional representation and is trained after the backbone is frozen.

It answers whether information is accessible, not whether the current native SNN readout can already use it.

## 11.3 D1 does not mean all airborne motion is useless

The D1 > D2 result suggests the opposite nuance:

> full motion history may help the encoder, while explicit airborne spikes can hurt the downstream classifier.

## 11.4 MT has not been shown to remove the analog→binary gap

Exp10.1 does not show a consistent collapse of the L1 pre-reset→spike gap under MT.

Its strongest effect appears in downstream representation and its interaction with supervision.

## 11.5 Shift-4 temporal smearing remains a working explanation

The degradation at ~242 ms is consistent with temporal smearing, but Exp10.2 did not directly manipulate phase overlap to prove that mechanism.

---

# 12. Recommended current baselines

Two baselines are now useful for different questions.

## 12.1 Best current time-shared baseline candidate

\[
\boxed{
D1
+
BB
+
L1\ \tau_{mem}\approx54\text{ ms}
+
L2\ \tau_{mem}=22.54\text{ ms}
+
WCCE
}
\]

On the locked split:

\[
Native\ BA
=
\boxed{55.69\%}.
\]

This should be validated across the 5 cross-user rotations before replacing older baselines globally.

## 12.2 Best temporal-backbone candidate from Exp10.2

\[
\boxed{
D1
+
BB
+
L1\ \tau_{mem}\approx117\text{ ms}
}
\]

produces:

\[
L2\ Fixed250
=
\boxed{64.54\%}
\]

after E2E training.

This is not the best current time-shared native classifier, but it is a strong candidate for future temporal/phase-aware readouts.

---

# 13. Future directions

The following directions are motivated by the experiments plus the design discussion that produced Exp10.1/10.2.

## 13.1 First priority — validate the membrane result across user splits

Before adding more mechanisms, rerun the key membrane settings across the full 5-rotation protocol.

At minimum:

~~~text
D1 + BB + L1 shift_mem=1
D1 + BB + L1 shift_mem=2
D1 + BB + L1 shift_mem=3
~~~

The goal is to determine whether the shift-2 gain is stable across held-out-user compositions.

## 13.2 Fine-grained membrane sweep around the optimum

The coarse sweep gives:

~~~text
22 ms  -> 52.09 native BA
54 ms  -> 55.69
117 ms -> 54.55
242 ms -> 51.96
~~~

The likely useful region is roughly:

\[
40\text{--}100\text{ ms}.
\]

A finer sweep could use approximately:

\[
\beta
\in
\{0.70,\ 0.75,\ 0.80,\ 0.85\},
\]

corresponding roughly to:

\[
\tau_{mem}
\approx
44,\ 54,\ 70,\ 96\text{ ms}.
\]

This should be treated as refinement after multi-split validation, not as a substitute for it.

## 13.3 Combine membrane memory with MT and supervision

Exp10.1 and Exp10.2 changed different mechanisms.

### Longer membrane

Retains evidence across time within a neuron:

\[
\boxed{
amplitude
\rightarrow
temporal integration / spike timing and count
}
\]

### MT

Uses population threshold heterogeneity:

\[
\boxed{
amplitude
\rightarrow
population activity pattern
}
\]

### Joint / TSCE

Changes what representation the network is encouraged to organize.

These mechanisms may therefore be complementary.

A targeted next factorial does not need to be large:

~~~text
D1 + shift2 + BB + WCCE
D1 + shift2 + MM + WCCE
D1 + shift2 + MM + Joint
D1 + shift2 + MM + 0.1 L1 TSCE
~~~

A separate temporal-backbone branch could test:

~~~text
D1 + shift3 + MM + 0.1 L1 TSCE
~~~

because Exp10.1 TSCE and Exp10.2 shift3 both favor richer temporally resolved L2 representations.

The central question should remain:

\[
\boxed{
\text{Does one mechanism preserve information created by another?}
}
\]

rather than only comparing final BA.

## 13.4 Phase-aware / relative temporal readout

This is the largest unresolved representation-to-native gap.

Exp10.2 shift3:

\[
L2\ Fixed250=64.54\%,
\]

but:

\[
L2\ whole=57.80\%,
\]

and:

\[
Native=54.55\%.
\]

A clean future readout experiment should freeze a selected Exp10 backbone and compare parameter-controlled readouts such as:

~~~text
WholeMean
WholeCount
Relative5bin
Relative10bin
Fixed250
~~~

This should be interpreted together with Exp8.0.5, which already showed that correct phase alignment can outperform time-shared or destroyed-phase controls.

## 13.5 Continue attacking the L1 binary bottleneck

Even the shift-2 model retains approximately:

\[
5.7\text{ pp}
\]

between L1 pre-reset and L1 spike Fixed250 probes.

Possible mechanisms include:

- membrane memory + MT together;
- adaptive / learned thresholds;
- weighted or multi-bit spike communication, if deployment constraints allow;
- multi-threshold population codes with controlled width;
- alternative reset/coding schemes.

These should be tested against the same pre-reset→spike diagnostic, not only final native BA.

## 13.6 Representation-geometry regularization

A supervised-contrastive / metric-learning branch remains plausible after simpler dynamics are stabilized.

A safer version is not timestep-by-timestep contrastive loss, because handwriting samples are not perfectly temporally aligned.

Instead use a phase-aware relative-bin embedding:

\[
r=
[
\bar z^{(1)};
\ldots;
\bar z^{(K)}
]
\]

and apply a supervised contrastive objective to \(r\).

Useful positive sets can separately include:

- same class, same user, different segment;
- same class, different user.

This would test whether representation geometry can be made invariant to within-user execution variation and cross-user variation without throwing away temporal structure.

This is currently secondary to establishing the dynamics/readout baseline.

## 13.7 Output-LIF realization remains a separate problem

The best Exp10.2 native Linear condition is:

\[
55.69\%.
\]

The same evidence through current output LIF reaches only:

\[
46.79\%.
\]

So approximately:

\[
\boxed{8.9\text{ pp}}
\]

remain in the analog-Linear → output-LIF conversion.

Do not interpret a stronger hidden representation as automatically solving the output-neuron problem.

## 13.8 More detailed motion-source attribution

Exp10.0 establishes that a binary writing-vs-airborne mask is useful, but it does not fully separate:

- stroke-related motion;
- pen repositioning;
- transition motion;
- press/lift transients;
- non-writing wrist motion.

A future motion-attribution experiment could create controlled masks for these components and quantify their incremental contribution under the same paired split/seed protocol.

This directly extends the original question of how much classification comes from writing strokes versus repositioning motion.

---

# 14. Current working architecture picture

The Exp10 series currently supports the following working model:

~~~text
D1 post-encode masking
        ↓
cleaner writing-period event stream
        ↓
L1 with moderate membrane integration (~50–100 ms)
        ↓
strong analog local state
        ↓
binary communication still loses some information
        ↓
L2 builds useful temporally structured evidence
        ↓
current whole/time-shared readout discards part of that structure
        ↓
native Linear prediction
        ↓
output LIF introduces an additional realization penalty
~~~

This suggests the project should no longer treat the problem as one scalar question such as:

> How do we make the SNN more accurate?

A more productive decomposition is:

\[
\boxed{
\text{input nuisance}
}
\]

\[
\boxed{
\text{local temporal integration}
}
\]

\[
\boxed{
\text{binary communication}
}
\]

\[
\boxed{
\text{representation supervision}
}
\]

\[
\boxed{
\text{temporal readout}
}
\]

\[
\boxed{
\text{output-neuron realization}
\]

and experiments should continue to isolate these factors before combining them.

---

# 15. Practical decision table

| Question | Current answer |
|---|---|
| Keep explicit airborne spikes? | Generally no; D1 is better supported. |
| Remove airborne acceleration before encoder? | No; D2 is worse than D1. |
| Is MT alone enough? | No. |
| Can MT help with stronger supervision? | Yes, there is evidence of positive interaction. |
| Does weak L1 TSCE always help? | No; BB can get worse, while MM can benefit. |
| Is 22.5 ms L1 membrane likely too short? | Yes, current data support this. |
| Best current time-shared L1 membrane candidate? | ~54 ms on rotation0. |
| Best Exp10.2 temporally resolved L2 candidate? | ~117 ms. |
| Does longer membrane eliminate L1 binary loss? | No; it improves representation but a sizable gap remains. |
| Is very long ~242 ms membrane clearly better? | No; performance falls back toward baseline. |
| Did long membrane produce near-total firing saturation? | No in the current diagnostics. |
| Is time-shared readout still a bottleneck? | Yes; Fixed250 > whole/native remains common. |
| Is output LIF solved? | No; a substantial penalty remains. |

---

# 16. Statistical and interpretation cautions

### Exp10.0

Uses five cross-user rotations and is the strongest source for dataset-level D0/D1/D2 comparisons.

### Exp10.1 and Exp10.2

Use one locked cross-user split.

Therefore:

\[
\boxed{
\text{their seed means measure optimization stability on one split,
not population-level user generalization}
}
\]

The next major claims about a new best A2 should be confirmed under multiple user rotations.

### Probe comparisons

Probe BA differences are diagnostic.

Do not sum:

~~~text
pre-reset → spike loss
+ temporal-collapse loss
+ native-head loss
~~~

and call the result a strict percentage decomposition of information loss.

The probes use different feature spaces and separately trained post-hoc classifiers.

---

# 17. Reproducibility references

## Exp10.0

Implementation/results:

~~~text
scripts/experiment_10_0_airborne_motion_ablation.py
scripts/experiment_10_0_airborne_motion_ablation/README.md
notebooks/artifacts/experiment_10_0_airborne_motion_ablation/
  a2_cross_user_5fold_v1/
~~~

Result commit:

~~~text
a2c53e006073cef6270a7d8e1fdb661a59dba0f1
~~~

## Exp10.1

Implementation/results:

~~~text
scripts/experiment_10_1_a2_backbone_coding_supervision.py
scripts/experiment_10_1_a2_backbone_coding_supervision/README.md
notebooks/artifacts/experiment_10_1_a2_backbone_coding_supervision/
  d0_d1_bb_mm_supervision_single_split_v1/
~~~

Result commit:

~~~text
591c3843615d370aad114a9cc5dc66d3d1d91519
~~~

## Exp10.2

Implementation/results:

~~~text
scripts/experiment_10_2_l1_membrane_memory.py
scripts/experiment_10_2_l1_membrane_memory/README.md
notebooks/artifacts/experiment_10_2_l1_membrane_memory/
  d1_bb_l1_mem_shift_sweep_v1/
~~~

Result commit:

~~~text
8ff26d2f2ec57b7dd70be4ec0a669d48b09bfaaf
~~~

---

# 18. Bottom line

The Exp10 series changes the interpretation of the current A2 bottleneck.

The evidence no longer supports a simple story such as:

> The two-layer SNN cannot extract enough local information.

A better summary is:

\[
\boxed{
\text{The backbone can form strong class-decodable states,
but information is progressively limited by input nuisance,
short membrane integration, binary communication,
time collapse, and output-neuron realization.}
}
\]

The most actionable current findings are:

\[
\boxed{
D1\ post\text{-}encode\ masking
}
\]

for preprocessing,

\[
\boxed{
\tau_{mem}^{L1}\approx54\text{ ms}
}
\]

as the strongest current time-shared membrane candidate,

and the persistent observation that:

\[
\boxed{
L2\ temporal\ probes
>
L2\ whole/native
}
\]

which keeps temporal/phase-aware readout as a major future direction.

The next phase should therefore prioritize **multi-split validation of the membrane result**, then test carefully controlled combinations of membrane memory, MT/supervision, and temporal readout rather than continuing to increase hidden-layer capacity blindly.
