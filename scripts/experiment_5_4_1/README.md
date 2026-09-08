# Experiment 5.4.1 — Constrained WHAT×WHEN residual conjunction

## Question

Exp5.4 established two facts that must be separated:

1. the jointly trained Fusion-LIF is sensitive to WHAT/WHEN temporal alignment;
2. that alignment sensitivity did **not** produce a consistent final-BA improvement over WHAT-only, elapsed-time, reset-WHEN, or the matched linear control.

In addition, WHEN-only retained substantial letter identity. Therefore Exp5.4.1 does not add capacity to the old fusion. It asks a narrower question:

> **Can frozen WHEN spikes provide incremental class information when they are structurally prevented from becoming an independent classifier and may only correct an already learned WHAT-only decision through a spike-domain conjunction?**

## Fixed sources

For every seed `(11, 23, 37, 53, 71)`:

- frozen WHAT trajectory: the same 128D Local-SNN L2 spikes cached by Exp5.4;
- frozen WHEN trajectory: Exp5.3.2.3 `ffsnn128_rsnn64` 64D output spikes cached by Exp5.4;
- frozen base classifier: Exp5.4 `what_only_lif` checkpoint for the same seed;
- user-disjoint split and labels are unchanged.

The Exp5.4 WHAT-only model is:

```text
WHAT128 spikes
    -> frozen W_what
    -> short-memory Fusion-LIF128
    -> frozen Fusion spikes s_base[t]
    -> frozen W_out
    -> whole-sequence base logits
```

All base parameters are frozen in Exp5.4.1.

## Residual path

The new path **does not read raw WHAT**. It receives only the already learned frozen base Fusion spikes plus frozen WHEN spikes:

```text
frozen base Fusion128 spikes -> Base selector128 spikes ---\
                                                        AND -> Conjunction128 spikes -> residual W_out -> delta evidence
frozen WHEN64 spikes         -> WHEN selector128 spikes --/
```

The selector layers are bias-free fixed-synapse projections followed by a spike threshold.

The conjunction neuron receives two binary selector spikes. Each side contributes `0.75` and the conjunction threshold is `1.0`:

- base selector only: `0.75 < 1.0` -> no conjunction spike;
- WHEN selector only: `0.75 < 1.0` -> no conjunction spike;
- both selectors: `1.50 > 1.0` -> conjunction spike.

Therefore the residual path is structurally constrained:

\[
C_t \approx B_t \land H_t.
\]

It cannot emit residual evidence when either input side is absent.

## Final prediction

The frozen base logits are preserved:

\[
L_{base}=\sum_t W_{base}^{out}s_t^{base}+b_{base}.
\]

The context path produces only a correction:

\[
\Delta L=\sum_t W_{res}s_t^{conj}.
\]

Final logits are:

\[
\boxed{L=L_{base}+\Delta L}.
\]

There is no second class bias.

`W_res` is initialized to exactly zero. Thus at epoch 0:

\[
\boxed{L=L_{base}}
\]

exactly. Epoch 0 is a legal checkpoint candidate, so the optimizer is not required to replace the frozen base if validation BA does not improve.

## What is trainable

Only:

- `base_selector: 128 -> 128`, bias-free;
- `when_selector: 64 -> 128`, bias-free;
- `residual_output: 128 -> 12`, bias-free.

The frozen Exp5.4 base classifier and frozen WHAT/WHEN source representations are never updated.

The context path has **no temporal recurrence and no membrane history**. It is deliberately instantaneous. Any long/order-dependent context must therefore arrive through the frozen WHEN spikes.

## Objective and causality

Training uses final whole-sequence CE only:

\[
\mathcal L=CE(L,y).
\]

No Fixed250, Relative10, flattening, attention, or recurrent classifier is introduced.

Final duration `T_i` is never a model input. Valid length is used only to mask padded timesteps and select the whole-sequence endpoint.

## Primary evaluation

Five independent runs are trained, one per seed.

Primary quantity:

\[
\Delta BA_{WHEN}=BA(\text{base + constrained residual})-BA(\text{frozen WHAT-only base}).
\]

The finalizer also compares the new model against the committed Exp5.4 controls:

- `what_only_lif`;
- `what_elapsed_lif`;
- `what_resetwhen_lif`;
- `what_when_linear`;
- `what_when_fusion_lif`.

These sources are reused, not retrained.

## Test-time attribution

The selected residual checkpoint is evaluated under:

1. `ordered` — normal aligned WHEN;
2. `reset_when` — Exp5.3.2.3 FF/RSNN state reset every timestep;
3. `when_zero` — WHEN spikes removed;
4. `when_shuffle` — valid WHEN timesteps shuffled, five deterministic replicates;
5. `when_circular_shift` — same gesture-specific WHEN trajectory shifted by roughly one third of valid length;
6. `base_context_zero` — the residual path receives zero base-Fusion spikes while the frozen base classifier itself remains intact.

Structural expectations:

\[
\boxed{\Delta L(\text{base spikes},0)=0}
\]

and

\[
\boxed{\Delta L(0,\text{WHEN})=0}.
\]

Therefore `when_zero` and `base_context_zero` must collapse exactly to the frozen WHAT-only base prediction. This is a hard architecture contract, not merely a desired empirical trend.

The informative attribution comparisons are then:

\[
BA_{ordered}-BA_{reset}
\]

\[
BA_{ordered}-BA_{shuffle}
\]

\[
BA_{ordered}-BA_{shift}.
\]

A useful WHEN correction should improve over the frozen base and lose that incremental gain when history/alignment is broken.

## Interpretation

Strong support requires all of the following:

- positive paired BA gain vs the exact frozen WHAT-only base;
- the gain is reasonably consistent across seeds;
- ordered WHEN beats reset/shuffled/shifted WHEN;
- zeroing either conjunction side gives exactly the frozen base;
- the residual path remains small enough that it is clearly a correction rather than a replacement classifier.

If the best checkpoint remains epoch 0 for most seeds, the correct conclusion is that the frozen WHEN representation did not provide reliable incremental classification value under the constrained routing rule.

## Multi-CPU execution

The experiment follows `AGENTS.md`:

```text
5 source-validation tasks (one seed each)
        -> afterok
5 train/evaluate tasks (one seed each, one CPU each)
        -> afterok
1 aggregation-only finalizer
```

Run from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_1_cpu.bash
```

Artifacts are written to:

```text
notebooks/artifacts/experiment_5_4_1_constrained_conjunction_residual/
    constrained_conjunction_residual_v1/
```

Final files:

- `runs.csv`
- `histories.csv`
- `ablation_runs.csv`
- `activity_runs.csv`
- `source_runs.csv`
- `paired_deltas.csv`
- `manifest.json`

The notebook `notebooks/experiment_5_4_1_constrained_conjunction_residual.ipynb` is analysis-only.
