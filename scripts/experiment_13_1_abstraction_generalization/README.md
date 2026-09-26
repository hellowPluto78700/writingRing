# Experiment 13.1 — Abstraction and Cross-user Generalization

## Motivation

Exp13 A1/A2/A3/B/D already establish that deeper states use more temporal
context, that L2 dynamics are faster rather than simply low-pass filtered, and
that older context is expressed in the current communicated representation.
Exp13.1 therefore does not rerun those questions.

The remaining question is whether additional context becomes a cross-user
invariant task representation or remains increasingly entangled with
user/trajectory-specific information.

## Controls

Two three-layer checkpoints from Exp7.3.9 are analyzed.

- **C1 primary depth control**: A2 L1/L2/output are frozen and only the inserted
  L3 is trained. Thus L1/L2 remain exactly the A2 representation.
- **C2 end-to-end replication**: the selected C1 checkpoint is unfrozen and
  L1/L2/L3/output are jointly adapted.

C1 isolates the effect of adding a deeper temporal transformation. C2 tests
whether the same pattern remains after end-to-end adaptation.

## F — Cross-user representation geometry

F caches full L1/L2/L3 trajectories for C1 and C2. The primary communicated
representation is `spike50`; `syn_current` and `pre_reset` are secondary.

Trajectories are resampled to 64 normalized-time points. Two distances are
reported:

1. fixed normalized-time alignment;
2. bounded DTW with a +/-20% Sakoe-Chiba band.

Local distance is cosine distance. DTW cost is normalized by path length.

Balanced pair strata are generated once and reused for all seeds:

- SC_SU: same character, same user;
- SC_CU: same character, cross user;
- DC_CU: different character, cross user. Negative target classes are selected
  with globally balanced class usage first, then duration-matched within the
  selected class.

The principal geometry statistic is:

[
R_l = D_{DC,CU}^l / D_{SC,CU}^l.
]

For `spike50`, F also performs train-user gallery -> held-out-user query
nearest-medoid retrieval and reports balanced accuracy.

The experiment does not assume an inverted-U result. Monotonic increase,
monotonic decrease, flat geometry, or an L2 peak are all valid outcomes.

## G — Character-conditioned user leakage

G asks whether user identity remains decodable after character structure is
removed. At 25%, 50%, 75%, and 100% phase, each outer training fold computes
character centroids using that fold only and residualizes:

[
r_i = z_i - mu_{y_i}.
]

A balanced logistic user-ID probe is then fit on residuals. C is selected only
inside the outer training data. A within-character user-label permutation
control is repeated 100 times by default.

Primary metric:

[
UserLeakage = BA_{real} - BA_{perm}.
]

User leakage is not interpreted as harmful by itself; it must be interpreted
jointly with F.

## H — History benefit versus cross-user generalization

H asks whether extra history disproportionately helps held-out sequences from
seen users compared with held-out users.

C2 reuses existing Exp13 A2 history-truncation features. C1 adds only the
missing 3 seeds x 6 history durations = 18 feature tasks.

History comparison uses a phase-specific common eligibility mask so every
reported condition at a given phase is evaluated on the same samples while
retaining all 12 classes:

- 50% phase: 50, 100, 250, 500 ms (common mask requires 500 ms);
- 75% phase: 50, 100, 250, 500, 750 ms (common mask requires 750 ms);
- 100% phase: 50, 100, 250, 500, 750, 1000 ms (common mask requires 1000 ms).

The implementation hard-fails if either ID/OOD evaluation or any ID-CV fold
loses a class. 75% and 100% remain primary; 50% is secondary.

Within the training-user population, deterministic user x character folds
produce held-out sequences from seen users (ID-CV). The same frozen SNN
representation is also evaluated on the existing held-out-user test split
(OOD). Probe C is selected once from full-history ID-CV and then held fixed
across the history sweep.

For each layer:

[
Gain_{ID}(D) = BA_{ID}(D) - BA_{ID}(50ms)
]

[
Gain_{OOD}(D) = BA_{OOD}(D) - BA_{OOD}(50ms)
]

[
E_l(D) = Gain_{ID,l}(D) - Gain_{OOD,l}(D).
]

Positive E means the extra history is more distribution-specific; it does not
by itself mean that history is harmful to OOD performance.

## Execution

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_13_1_cpu.bash
```

Optional controls:

```bash
SLURM_MAX_CONCURRENCY=20 USER_PERMUTATIONS=100 \
  bash scripts/bash_script/SNN_Bash/submit_exp_13_1_cpu.bash
```

Task order is serialized to keep the repository-wide experiment concurrency
ceiling enforceable:

```text
prepare
  -> F cache (6)
  -> F metric (18)
  -> G user leakage (18)
  -> H C1 history features (18)
  -> H ID/OOD probes (18)
  -> finalizer
```

The finalizer aggregates existing per-run artifacts only and never recomputes
missing runs.

## Validation

```bash
python -m pytest -q tests/test_experiment_13_1_abstraction_generalization_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```

### Re-running only the protocol-fixed stages

If trajectory caches, G probes, and C1 history features already exist, the F/H
protocol fix can be recomputed without rerunning those expensive stages:

```bash
bash scripts/bash_script/SNN_Bash/rerun_exp_13_1_protocol_fix_cpu.bash
```

This regenerates the pair/source manifests, force-reruns F geometry and H
ID/OOD probes, and then rebuilds the final summaries. Existing F trajectory
caches, G results, and C1/C2 history features are reused.
