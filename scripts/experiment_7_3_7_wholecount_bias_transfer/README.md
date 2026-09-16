# Exp7.3.7 — Whole-count Linear bias transfer to LIF count space

## Question

Exp7.3.6 showed that a fixed affine bias on **mean-pooled** L2 features does not explain the P5/P7 whole-count probe advantage. Exp7.3.7 therefore uses a truly causal accumulator formulation and asks two separate questions:

1. With the Exp7.3 A2/WCCE L1/L2 backbone frozen, how much does a fixed 12-class bias help a Linear whole-count accumulator?
2. If the affine Linear head's weight matrix is transferred **unchanged** to the standard output LIF, how much of the Linear-to-LIF loss can be recovered by only adding a fixed class-specific output-count offset?

No L1/L2 retraining and no LIF weight training are allowed in this experiment.

## Frozen source

For each seed 11/23/37, load the existing Exp7.3 Stage-2 cache sourced from:

```text
A2_e2e_linear_wcce
```

The cached L2 trajectory is binary:

\[
z_t\in\{0,1\}^{128}.
\]

Only valid timesteps are used.

## Stage A — train the Linear accumulator head only

Define valid whole count

\[
Z=\sum_{t=1}^{T}z_t.
\]

Two frozen-backbone controls are fit independently:

- `L0_linear_no_bias`: \(s=WZ\)
- `L1_linear_affine`: \(s=WZ+b\)

Both use the Exp7.3.2 P4/P5 recipe: train-only per-feature standard-deviation scaling, **no centering**, multinomial `LogisticRegression(lbfgs)` with L2 regularization, and `C` selected only by validation balanced accuracy. The scale is folded back into the raw weight matrix before deployment.

`L1_linear_affine` has the causal online interpretation

\[
S_0=b,\qquad S_t=S_{t-1}+Wz_t.
\]

Thus neither final sequence length nor a stored trajectory is required during accumulation.

## Stage B — transfer the same W directly to LIF

Take the raw weight matrix from `L1_linear_affine` and set

\[
W_{\mathrm{LIF}}=W_{\mathrm{Linear}}.
\]

Nothing is retrained. The existing Exp7.3 output neuron is used unchanged:

- \(\beta=0.5\)
- threshold \(=0.5\)
- at most one output spike per neuron per timestep
- no input bias is injected into the LIF dynamics

The valid output spike count is

\[
C=\sum_t s_t^{\mathrm{out}}.
\]

`L2_same_w_lif_count` reports `argmax(C)`.

## Stage C — fit only the output-count bias

Freeze everything from Stage B. On training-set LIF counts, fit only a 12D count-space offset \(q\) plus one positive scalar gain used only to make CE calibration well-conditioned:

\[
\mathcal L=CE\big(g(C+q),y\big),\qquad g>0.
\]

The gauge is fixed with `mean(q)=0`. Because a positive scalar does not change `argmax`, deployment discards `g` and uses

\[
\hat y=\arg\max(C+q).
\]

The calibration checkpoint is selected by validation BA, with validation CE as tiebreak. Test data are untouched until reporting.

Two deployment readouts are reported:

- `L3_same_w_lif_continuous_bias`: \(C+q\)
- `L4_same_w_lif_integer_bias`: shift \(q\) so its minimum is zero, round to non-negative integer virtual spike counts, then add those fixed counts to the output counters.

The integer gauge is therefore

\[
q^{+}=q-\min(q),\qquad q^{\mathrm{int}}=\operatorname{round}(q^{+}).
\]

These are fixed output-counter offsets, not runtime-generated spikes and not per-timestep LIF input bias.

## Primary comparisons

The finalizer reports paired three-seed contrasts:

- Linear affine benefit: `L1 - L0`
- Linear-to-LIF realization loss: `L1 - L2`
- Continuous count-bias recovery: `L3 - L2`
- Integer virtual-spike recovery: `L4 - L2`
- Remaining gap after continuous/integer bias

It also compares Stage-A L0/L1 against Exp7.3.2 P4/P5 as a reproduction check.

## Multi-CPU execution

One independent task is used per seed. Each task performs Stage A -> Stage B -> Stage C -> evaluation and writes its own artifacts. The Slurm array is therefore three one-CPU tasks, and the `afterok` finalizer only aggregates existing outputs.

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_7_cpu.bash
```

## Artifacts

Finalized artifacts are written to:

```text
notebooks/artifacts/experiment_7_3_7_wholecount_bias_transfer/wholecount_bias_transfer_v1/
```

Important files:

- `method_runs.csv`, `method_summary.csv`
- `gap_runs.csv`, `gap_summary.csv`
- `bias_vectors.csv`, `bias_vector_summary.csv`
- `calibration_history_runs.csv`, `calibration_history_summary.csv`
- `source_reproduction_checks.json`
- `manifest.json`

The notebook `notebooks/experiment_7_3_7_wholecount_bias_transfer.ipynb` is analysis-only and shows method-level aggregate comparisons rather than every individual training run.
