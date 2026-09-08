# Experiment 5.4.1 — Direct WHAT base + constrained WHAT×WHEN residual conjunction

## Question

Exp5.4 showed that WHAT/WHEN temporal alignment affects the trained Fusion-LIF, but the jointly trained Fusion-LIF did not consistently beat WHAT-only or matched linear controls. The earlier Exp5.4.1 draft still reused Exp5.4 `what_only_lif`, which inserted another 128-neuron LIF between the original WHAT branch and the classifier.

That extra layer is removed here.

The cleaner question is:

> **Starting from the original frozen WHAT L2 spikes themselves, can frozen WHEN spikes provide incremental classification value only through a structurally constrained WHAT×WHEN conjunction residual?**

## Frozen source representations

For each seed `(11, 23, 37, 53, 71)`:

- WHAT is the same frozen 128D binary Local-SNN L2 spike trajectory used throughout Exp5.x;
- WHEN is the frozen Exp5.3.2.3 `ffsnn128_rsnn64` 64D output-spike trajectory cached by Exp5.4;
- ordered and state-reset WHEN trajectories come from the same frozen source checkpoint;
- the user-disjoint train/val/test split is unchanged.

No Exp5.4 Fusion-LIF checkpoint is used as the base classifier.

## Stage A — direct WHAT base

The base classifier reads the original WHAT spikes directly:

```text
frozen WHAT128 spikes
        -> bias-free Linear(128 -> 12)
        -> per-timestep class evidence
        -> sum over valid timesteps
        -> one final 12D class bias
```

For timestep `t`:

\[
e_t^{base}=W_{base}s_t^{WHAT}.
\]

The whole-sequence base logits are

\[
L_{base}=\sum_{t\in valid}e_t^{base}+b_{base}.
\]

Because the projection is linear,

\[
L_{base}=W_{base}\sum_{t\in valid}s_t^{WHAT}+b_{base}.
\]

So this is deliberately a **direct whole-count readout of the original WHAT representation**. There is no additional Fusion-LIF, no extra membrane state, and no temporal pooling before the final valid-time sum.

Stage A is trained with final whole-sequence CE and checkpointed by validation balanced accuracy.

## Stage B — freeze the base and train only contextual correction

After Stage A, `W_base` and `b_base` are frozen.

The residual path receives the **same original WHAT spikes** plus frozen WHEN spikes:

```text
WHAT128 spikes -> WHAT selector128 spikes ---\
                                           AND -> conjunction128 spikes -> residual Linear(128 -> 12) -> delta evidence
WHEN64 spikes -> WHEN selector128 spikes ----/
```

The selector projections are bias-free. Their outputs are thresholded into binary selector spikes.

The conjunction population is structurally constrained. Each side contributes `0.75`; the conjunction threshold is `1.0`:

- WHAT selector only: `0.75 < 1.0` -> no conjunction spike;
- WHEN selector only: `0.75 < 1.0` -> no conjunction spike;
- both selector spikes: `1.50 > 1.0` -> conjunction spike.

Thus

\[
C_t \approx Q_t^{WHAT}\land Q_t^{WHEN}.
\]

The residual branch cannot emit class evidence from WHEN alone and cannot emit class evidence from WHAT alone.

## Final prediction

Residual class evidence is

\[
\Delta L=\sum_{t\in valid}W_{res}C_t.
\]

Final logits are

\[
\boxed{L=L_{base}+\Delta L}.
\]

There is no second class bias.

`W_res` is initialized to exactly zero, so before residual training:

\[
\boxed{L=L_{base}}
\]

exactly. Epoch 0 is a legal residual checkpoint candidate. If validation BA never improves, the experiment is allowed to retain the direct WHAT base unchanged.

## What is trainable

Stage A trains only:

- `base_output: 128 -> 12`, bias-free;
- one final 12D class bias.

Stage B freezes Stage A and trains only:

- `what_selector: 128 -> 128`, bias-free;
- `when_selector: 64 -> 128`, bias-free;
- `residual_output: 128 -> 12`, bias-free.

The original WHAT and WHEN source networks remain frozen throughout.

The residual path has **no recurrence, no synaptic state, and no membrane memory across timesteps**. Any long/history-dependent information must arrive through the frozen WHEN spikes.

## Objective and causality

Both stages use final whole-sequence CE only:

\[
\mathcal L=CE(L,y).
\]

There is no Fixed250, Relative10, flattening, attention, RNN classifier, or extra Fusion-LIF.

Final duration `T_i` is never a model input. Valid length is used only to mask padded timesteps and determine where accumulation stops.

## Primary test

Five independent seeds are used.

The primary paired quantity is

\[
\boxed{
\Delta BA_{WHEN}=BA(WHAT+WHEN\ residual)-BA(direct\ WHAT\ base)
}
\]

where both terms use the same frozen WHAT trajectory for that seed.

If this is consistently positive, the gain cannot be attributed to retraining a new WHAT classifier: the direct WHAT base is frozen before the residual branch is trained.

## Test-time attribution

The selected residual checkpoint is evaluated with:

1. `ordered` — correctly aligned causal WHEN;
2. `reset_when` — WHEN FF/RSNN state reset every timestep;
3. `when_zero` — remove WHEN spikes;
4. `when_shuffle` — shuffle valid WHEN timesteps, five deterministic replicates;
5. `when_circular_shift` — shift the same gesture-specific WHEN trajectory by about one third of valid length;
6. `residual_what_zero` — zero WHAT only inside the residual branch while keeping the frozen direct WHAT base intact.

Two architecture-level invariants must hold:

\[
\boxed{\Delta L(WHAT,0)=0}
\]

and

\[
\boxed{\Delta L(0,WHEN)=0}.
\]

Therefore `when_zero` and `residual_what_zero` must recover the direct WHAT base exactly.

The useful temporal-context comparisons are

\[
BA_{ordered}-BA_{reset},
\]

\[
BA_{ordered}-BA_{shuffle},
\]

and

\[
BA_{ordered}-BA_{shift}.
\]

## Interpretation

Strong support for the factorization requires:

- residual model > direct WHAT base on paired test BA;
- ordered WHEN > reset/shuffled/shifted WHEN;
- `when_zero` and `residual_what_zero` produce exactly zero residual logits and recover the base classifier;
- the selected checkpoint is not simply epoch 0 for most seeds.

If most seeds choose epoch 0, the correct conclusion is that the frozen WHEN representation did not provide reliable incremental classification value under this constrained routing rule.

## Multi-CPU execution

The experiment follows `AGENTS.md`:

```text
5 input-preparation tasks
        -> afterok
5 direct-WHAT-base training tasks
        -> afterok
5 residual train/evaluate tasks
        -> afterok
1 aggregation-only finalizer
```

Every array task uses one CPU core.

Run from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_4_1_cpu.bash
```

Artifacts are written to:

```text
notebooks/artifacts/experiment_5_4_1_constrained_conjunction_residual/
    direct_what_conjunction_residual_v2/
```

Final files:

- `base_runs.csv`
- `base_histories.csv`
- `runs.csv`
- `histories.csv`
- `ablation_runs.csv`
- `activity_runs.csv`
- `paired_deltas.csv`
- `manifest.json`

The notebook `notebooks/experiment_5_4_1_constrained_conjunction_residual.ipynb` is analysis-only.
