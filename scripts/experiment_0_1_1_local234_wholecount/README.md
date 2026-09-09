# Experiment 0.1.1 — Local `(234) x 2` direct WholeCount

## Question

Experiment 0.1 showed that the timestep-trained local `(234)->(234)` SNN contains substantially more class information once its L2 trajectory is decoded with an ordered Fixed250 Linear probe. Exp0.1.1 adds the missing **direct spiking-output control** under the same Exp0.1 paired seed namespace:

```text
Raw64 30-channel weighted events
  -> 128 multi-tau SNN, shifts (2,3,4)
  -> 128 multi-tau SNN, shifts (2,3,4)
  -> K=12 binary spiking class neurons
  -> valid Output WholeCount
  -> argmax
```

The deployment readout is always Output WholeCount. Only the training objective changes:

- `timestep_ce`
- `whole_count_ce`

Both are run for:

- `binary`: hidden cap = 1, output cap = 1
- `multi_h`: hidden cap = 31, output cap = 1

## Why this is not copied from Exp5.0

Exp5.0 already contains the same `(234)->(234)->K` architecture and the same two objectives, but its paired random namespace is `exp5_0_paired`. Exp0.1 uses `exp0_1_general_comparison`.

Exp0.1.1 therefore retrains this architecture with the **Exp0.1 paired initialization and loader-order namespace**, so architecture/objective/capacity comparisons remain paired with Exp0.1.

## Run matrix

```text
2 objectives x 2 capacities x 5 seeds = 20 runs
```

Seeds:

```text
11, 23, 37, 53, 71
```

The user split remains the Exp0.1 / Exp3 split with split seed `12345`.

## Checkpoint selection

For both objectives, select the checkpoint by:

1. maximum validation Output WholeCount balanced accuracy;
2. tie-break minimum validation normalized WholeCount CE.

This matches the direct-SNN selection rule in Exp0.1.

## Resume / early-exit contract

Every array task checks its artifacts before loading data or training.

A run is considered complete only when all three are valid and identity-matched:

```text
checkpoint
+ non-empty training history with required columns
+ evaluation JSON with train/val/test WholeCount metrics
```

Behavior:

```text
checkpoint + history + evaluation complete
    -> exit the array task immediately

checkpoint + history complete, evaluation missing/incomplete
    -> skip training
    -> evaluate the existing checkpoint

checkpoint/history missing or invalid
    -> train normally
    -> evaluate
```

`--force` disables the resume checks and reruns the selected task.

This means it is safe to resubmit the full array after partial completion.

## Outputs

```text
notebooks/artifacts/
  experiment_0_1_1_local234_wholecount/
    local234_wholecount_v1/
      checkpoints/
      histories/
      evaluations/
      runs.csv
      summary.csv
      paired_objective_effects.csv
      paired_capacity_effects.csv
      comparison_with_exp0_1.csv
      manifest.json
```

`comparison_with_exp0_1.csv` appends the four new local234 direct-SNN summary rows to the finalized Exp0.1 `comparison_summary.csv` without modifying the parent experiment artifacts.

## Multi-CPU execution

Each of the 20 runs is one independent Slurm array task using one CPU core. Maximum concurrency is 20, below the repository-wide cap of 50.

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_1_1_local234_wholecount_cpu.bash
```

The submit script launches the training/evaluation array and then an `afterok` finalizer.

## Validation

```bash
python -m pytest -q tests/test_experiment_0_1_1_local234_wholecount_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
