# Experiment 4.1 — Causal positional information for RSNN

## Question

Experiment 4.0 showed that the same Fixed250 representation supports a much stronger flattened Linear decoder than the stateful SNNs. Experiment 4.1 tests whether the missing ingredient is explicit **causal absolute time position**, which Linear receives implicitly from its separate temporal slots.

The raw representation remains unchanged:

```text
30-channel weighted event train
  -> 250 ms channel-wise aggregation
  -> train-only per-channel zero-preserving scaling
  -> ordered Fixed250 vectors
```

Only the RSNN input is extended with causal position.

## Causality

For padded macro-bin index `b = 1..Bmax`, position is defined from elapsed time since gesture start, never from the unknown final gesture duration.

### Scalar

```text
p_b = b / Bmax
input_b = [z_b, p_b]
```

### One-hot absolute slot

```text
q_b = one_hot(b, Bmax)
input_b = [z_b, q_b]
```

Both are causal because `b` is known online from elapsed time. Neither uses `b / B_i` or any Relative10/final-duration normalization.

For bins after the valid endpoint, **all positional channels are zeroed**. Therefore post-end dynamics still receive zero external input and remain directly comparable with Experiment 4.0 tail diagnostics.

## Anchor configurations

Experiment 4.0 already identified two informative RSNN operating points:

```text
H=64,  tau_mem=1000 ms
H=128, tau_mem=250 ms
```

The no-position controls are reused from Experiment 4.0 rather than retrained.

New training grid:

```text
2 anchor configs
x 2 position encodings (scalar, onehot)
x 5 seeds (11, 23, 37, 53, 71)
= 20 new runs
```

The finalizer also loads 10 matching no-position Experiment 4.0 controls, producing a paired 30-run comparison table.

## Fixed contracts

Unchanged from Experiment 4.0:

- same user-disjoint split (`split_seed=12345` through the shared Exp4.0 loader);
- same 30-channel Fixed250 vectors and train-only zero-preserving channel scales;
- same threshold, reset, output dynamics, optimizer, epoch count and batch size;
- same valid WholeCount CE training objective;
- all padded timesteps execute; state is never frozen;
- `valid_count`, `full_count`, valid final membrane and full final membrane come from the same trajectory.

The only experimental variable is the causal position input.

## Multi-CPU execution

Following `AGENTS.md`:

```text
20 independent new runs
  -> Slurm array 0-19%20
  -> one CPU core per task
  -> train -> select best checkpoint -> evaluate -> save JSON
  -> afterok finalizer
```

The finalizer aggregates existing artifacts only and fails if either the 20 new runs or the required Exp4.0 controls are missing.

Submit Experiment 4.1 alone:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_1_cpu.bash
```

Submit Experiments 4.1 and 4.2 together while respecting the repository-wide 50-CPU default cap:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_1_4_2_cpu.bash
```

## Artifacts

```text
notebooks/artifacts/
  experiment_4_1_causal_position_rsnn/
    causal_position_v1/
      checkpoints/
      evaluations/
      runs.csv
      summary.csv
      paired_position_effects.csv
      provenance.json
```

`paired_position_effects.csv` reports, for each anchor and seed:

```text
scalar_minus_none
onehot_minus_none
```

on test balanced accuracy.

## Interpretation

- large position gain: the Exp4.0 bottleneck was primarily missing explicit temporal position;
- one-hot > scalar: discrete absolute slot identity matters beyond coarse elapsed time;
- little/no gain: the remaining limitation is more likely recurrent/spiking dynamics or readout optimization than temporal position.
