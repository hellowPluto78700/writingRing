# Experiment 17.1 — Persistent-route gradient starvation and low-rate rescue

## Core hypothesis

Exp17 showed a trajectory in which high-occupancy L2 dimensions are more class-selective early, while later class selectivity shifts toward lower-occupancy dimensions even as persistent activity remains abundant. Exp17.1 tests a specific optimization explanation:

early persistent route raises training margin
→ CE residual shrinks
→ R_S^T delta shrinks
→ the late lower-rate route is under-trained.

The experiment does not assume that persistent activity is intrinsically harmful. The question is whether early success of the persistent route reduces optimization pressure for an alternative discriminative route.

## Locked backbone and protocol

All formal conditions inherit the Exp17 / Exp16.3 LIN-WCCE contract:

- two-layer 128-neuron CoreBenchmark backbone;
- shifts ((2,3,4),(2,3,4));
- exact CoreBenchmark split;
- formal seeds 11, 23, 37;
- bias-free accumulator / WholeCount readout;
- same optimizer, batch size, deterministic epoch permutation, and training horizon;
- native checkpoint selection: validation BA, then validation CE, then earliest epoch;
- test is diagnostic only.

No firing-rate regularizer, gate, phase-CU, relative10 objective, contrastive loss, or dynamic neuron reassignment is introduced.

## Phase 0A — Independent auxiliary-gradient calibration

Calibration uses seed 101, which is not a formal seed.

1. Train the exact LIN-WCCE model to epoch 20.
2. Rank L2 neurons using training-split mean occupancy only.
3. Freeze the bottom occupancy quartile as the calibration low-rate group.
4. On up to eight deterministic epoch-21 training batches, measure the main-WCCE and auxiliary-CE gradient norms on the same L2 rows, plus gradient cosine.
5. Set one global auxiliary coefficient to match the local low-group gradient magnitude, clipped to [1e-4, 100].

No BA, validation metric, or test metric is used to choose lambda. The selected lambda is locked for all C1/C2/C3 formal runs.

## Phase 0B — Shared epoch-20 bootstrap checkpoints

For each formal seed, train one baseline LIN-WCCE model from epoch 0 to the fixed branch epoch e_b = 20.

At epoch 20, compute per-neuron L2 mean occupancy on the training split and freeze three dimension-matched groups:

- low: bottom 25%;
- high: top 25%;
- random: deterministic random 25%.

For width 128 this gives 32 neurons per group. Membership is never updated after epoch 20.

Every C0-C3 condition for a seed starts from the same serialized epoch-20 checkpoint.

## Phase 1 — Matched continuation cases

### C0 — Baseline

Continue ordinary LIN-WCCE training.

### C1 — Low-rate rescue

Add an independent bias-free WholeCount classifier that receives only the frozen epoch-20 bottom quartile.

This is the hypothesis condition.

### C2 — High-rate auxiliary control

Use the same auxiliary architecture and locked lambda on the frozen top occupancy quartile.

### C3 — Random matched control

Use the same auxiliary architecture and locked lambda on a deterministic random quartile.

The auxiliary head is never used for native evaluation.

## Diagnostics

At epoch 21 and every five epochs thereafter, record:

- selected-group main-gradient norm;
- selected-group auxiliary-gradient norm;
- main/aux gradient cosine;
- fixed low, high, and random group mean occupancy;
- fixed-group mean class eta-squared;
- fixed-group native-head Frobenius norm;
- fixed-group mean native-logit contribution norm;
- correlation between neuron occupancy and class eta-squared.

Always retain train / validation / test native BA at the selected main checkpoint.

A crucial failure mode is explicitly visible: if C1 improves only because the frozen epoch-20 low-rate neurons themselves become high-occupancy, then the optimizer has reconstructed another persistent route rather than rescuing a genuinely lower-rate discriminative route.

## Phase 1.5 decision rules

### A. Gradient-starvation support

Support is strongest if C1, relative to C0/C2/C3, shows larger effective optimization of the fixed low group, faster growth of low-group class eta-squared, increased main-head usage of the low group, low-group occupancy remaining clearly below the high group, and improved OOD/native test BA or a reduced train-test gap.

### B. Representation rescue without native gain

If C1 increases low-group class selectivity and main-head use but native test BA is unchanged, the representation bottleneck is partially rescued while direct evidence routing remains a likely bottleneck. This is the condition that motivates a later persistent-readout attenuation experiment.

### C. Generic auxiliary effect

If C1 is similar to C2 and C3, attribute gains to generic auxiliary supervision rather than low-rate gradient starvation.

### D. New persistent route

If C1 drives the frozen low group to high occupancy, interpret the result as evidence that the current SNN/WCCE landscape attracts solutions toward persistence, not as successful low-rate rescue.

### E. No rescue despite nonzero auxiliary gradient

If the auxiliary gradient is present but low-group class selectivity and native performance do not improve, gradient starvation alone is insufficient; the lower-rate subspace may lack suitable representational capacity under the current dynamics.

## Phase 2 is intentionally not implemented yet

Persistent-readout freezing/attenuation is gated on the Phase-1.5 outcome. Implementing it now would mix causal diagnosis with a second intervention and weaken attribution.

## Slurm DAG

The CPU launcher runs:

1. one seed-101 calibration job;
2. three epoch-20 bootstrap tasks;
3. twelve continuation tasks (4 cases x 3 seeds);
4. one aggregate/finalizer job.

Artifacts are written under:

notebooks/artifacts/experiment_17_1_gradient_starvation/gradient_starvation_v1/

## Post-hoc validation — routing bottleneck versus representation bottleneck

Phase 1 produced a mixed outcome: C1 strongly increased class selectivity in the frozen epoch-20 low-occupancy population, but native OOD BA did not improve consistently, and that population also became more active. Before implementing persistent-readout attenuation, run an artifact-only validation to distinguish two remaining explanations.

### Question

Did C1 actually make the fixed epoch-20 low group more transferable to unseen users, while the native main head failed to exploit it?

If yes, the remaining limitation is consistent with an evidence-routing bottleneck. If no, then higher train-side class eta-squared is not sufficient evidence that the rescued representation itself transfers.

### Locked probe geometry

For every selected C0/C1/C2/C3 checkpoint and seed 11/23/37:

- reuse the exact epoch-20 `low` and `high` groups stored by the original experiment;
- cross-check checkpoint group indices against the epoch-20 bootstrap `groups.json`;
- do not redefine groups using the selected checkpoint, validation data, or test data;
- extract L2 spike WholeCount features by summing valid spike activity across time;
- fit the standard CoreBenchmark `no_bias` logistic probe;
- scale using train data only;
- select probe C using validation BA only;
- report train, validation, and test BA plus train-test gap;
- never retrain or modify the SNN.

The primary contrast is:

[
Delta_{mathrm{low}} =
BA_{mathrm{test}}^{mathrm{C1,low}}
-
BA_{mathrm{test}}^{mathrm{C0,low}}.
]

The high-group probe is a matched contextual control.

### Interpretation contract

**Routing-bottleneck support:** C1 substantially and consistently improves fixed-low-group test BA relative to C0, while native C1 test BA remains approximately unchanged. This means transferable information exists in the rescued low group but is not effectively converted into native accumulated evidence.

**Representation-bottleneck support:** C1 strongly improves train-side low-group selectivity but fixed-low-group test BA does not improve consistently. In this case, the apparent representation rescue is not itself OOD-transferable, so persistent-readout attenuation is not yet justified as the main next step.

**Mixed result:** low-group test BA improves only in a subset of seeds or only alongside a comparable train-test gap increase. Treat this as insufficient evidence for a clean routing claim.

### Execution

This block is artifact-only and uses the completed Exp17.1 checkpoints:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_17_1_posthoc_probe_cpu.bash
```

It runs 12 independent probe tasks followed by one aggregate job.

Artifacts are written under:

`notebooks/artifacts/experiment_17_1_gradient_starvation/gradient_starvation_v1/posthoc_group_probes/`

