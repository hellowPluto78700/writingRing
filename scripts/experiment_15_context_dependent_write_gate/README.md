# Experiment 15 — Context-Dependent Scalar Write Gating

## Scientific question

Exp15 tests whether an explicit L2-history-conditioned scalar controller can improve how the existing two-layer CoreBenchmark SNN accumulates history. The Phase-1 mechanism controls **when/how strongly** the complete current L1→L2 message is written; it does not select individual feature dimensions.

Baseline:

```text
I_L2[t] = alpha * I_L2[t-1] + W2 s_L1[t]
```

Gated:

```text
a[t] = wz^T s_L1[t] + wh^T s_L2[t-1] + bg
g[t] = sigmoid(a[t])
m[t] = g[t] / 0.9
I_L2[t] = alpha * I_L2[t-1] + m[t] W2 s_L1[t]
```

The baseline may already carry context through L1 recurrent state. Exp15 therefore tests the added value of an **explicit multiplicative controller conditioned on previous L2 communication**, not the stronger claim that the baseline has no context dependence.

## Locked CoreBenchmark contract

Exp15 reuses the finalized CoreBenchmark O0 data, split, two-layer 234/234 architecture, WCCE task loss, optimizer, batch order, and checkpoint-selection rule. Seeds are 11/23/37. Epoch 0 is a checkpoint candidate. Test data and diagnostic probes never select checkpoints.

Phase 1 changes no tau, width, readout, dataset, user split, or task loss. It adds no Prefix-WCCE, Relative10 training objective, contrastive loss, cross-user auxiliary loss, adaptive tau, forget gate, output gate, neuron-wise gate, or per-tau gate.

## Function-preserving initialization

The gate is initialized with

```text
wz = 0
wh = 0
bg = logit(0.9)
m[t] = sigmoid(a[t]) / 0.9
```

so the initial effective multiplier is one and the gated forward path reproduces the ungated write rule up to floating-point tolerance.

## Phase 1 cases

- **B0-ref**: immutable CoreBenchmark O0 checkpoint; no retraining.
- **C0**: load B0-ref and continue training with the unchanged network.
- **GF**: load B0-ref, insert the gate, freeze all original parameters, train only `gate_input.weight`, `gate_history.weight`, and `gate_bias`.
- **GJ**: load B0-ref, insert the same gate, and jointly train gate + original trainable backbone/readout.

Primary matched comparisons are **GF − B0-ref** for post-hoc routing value and **GJ − C0** for co-adaptive gating value beyond additional optimization. GF − C0 is reported as a secondary practical contrast.

## Native and representation evaluation

Every selected checkpoint reports train/validation/test BA, accuracy, macro-F1, train-test BA gap, and per-user metrics.

L2 spike probes reuse the CoreBenchmark implementations of:

- `whole_count`
- `fixed250_ordered`
- `fixed250_shuffled`
- `relative10_ordered`
- `relative10_shuffled`

Both no-bias and affine decoders are retained; no-bias remains primary.

The shuffled probes exactly follow CoreBenchmark semantics. Complete valid bins are permuted independently per sample. Fixed250 holds the final partial bin and padding fixed. Relative10 shuffles all ten normalized phase bins. Train/validation/test are shuffled under the same deterministic sample-ID rule, and a **separate matched decoder is fit for each shuffled representation and shuffle replicate**. Ordered decoder weights are not reused on shuffled test features.

Reported contrasts include:

```text
G_resolved       = fixed250_ordered - whole_count
G_order          = fixed250_ordered - fixed250_shuffled
G_relative_order = relative10_ordered - relative10_shuffled
Gap_rel10_whole  = relative10_ordered - whole_count
```

These measure decoder-accessible temporal alignment/accessibility, not mutual information.

## Gate diagnostics

GF/GJ record gate-gradient diagnostics at initialization and predefined epochs. Selected checkpoints additionally report:

- mean/std/quantiles of raw `g[t]` and effective `m[t]`;
- mean absolute and RMS deviation from baseline `m=1`;
- fraction with `|m-1| > 0.05` and `> 0.10`;
- within-sequence gate variance;
- between-sequence variance of sequence-mean gate;
- input-term and history-term RMS and their ratio;
- ten-bin relative-progress profiles.

The gradient diagnostic records gate parameter gradient norms, normalized gradient pressure, and the ten-bin mean absolute `dL/da[t]` profile on a fixed deterministic Core-distribution batch.

## Phase 1.5 causal interventions

Phase 1.5 performs no retraining. Each selected GF/GJ checkpoint is fully rerun from zero state under:

- **A0** learned gate.
- **A1** baseline-write override `m[t]=1`.
- **A2** two-pass sequence-mean gate: preserve each gesture's average write strength, remove within-gesture temporal selectivity.
- **A3** two-pass within-sequence shuffle: preserve the learned gate-value multiset and distribution, destroy temporal alignment. Five deterministic shuffle seeds are averaged within model seed.
- **A4** remove the gate's L2-history input while leaving the SNN recurrent dynamics intact.
- **A4b** two-pass mean-history control: preserve the first-pass sequence-average history contribution but remove its time variation.

All interventions rerun the complete SNN trajectory and then call the same CoreBenchmark-style L2 probe pipeline.

## Interpretation levels

- GJ > C0 plus A0 > A1 supports useful adaptive write scaling.
- A0 > A2 and/or A0 > A3 supports functionally important temporal selectivity.
- A0 > A4b provides stronger evidence that time-varying L2 history context contributes to the write decision.

A negative native-BA result alone does not reject the gating hypothesis; interpretation jointly uses native/OOD behavior, probe accessibility, gate engagement/gradients, and Phase-1.5 interventions.

## Multi-CPU execution

The repository default is task-level CPU parallelism:

```text
prepare
  -> Phase 1: 9 independent array tasks
       C0/GF/GJ x seeds 11/23/37
       each task: train -> select checkpoint -> native eval -> probes -> diagnostics
  -> Phase 1.5: 6 independent array tasks
       GF/GJ x seeds 11/23/37
       each task: A0/A1/A2/A3/A4/A4b -> native eval -> probes
  -> finalizer
```

Each array task requests one CPU core and explicitly sets OMP/MKL/OpenBLAS/NumExpr threads to one. Default submission concurrency is 20, capped by the array size and never above 50.

Run:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_15_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_15_context_dependent_write_gate/context_write_gate_v1/
```

## Validation

Focused checks:

```bash
python -m pytest -q tests/test_repository_source_syntax.py
python -m pytest -q tests/test_experiment_15_context_dependent_write_gate_contract.py
```
