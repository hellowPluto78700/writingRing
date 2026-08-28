# Experiment 3.0.2 — Hidden multi-tau architecture comparison

## Question

With the 30 -> 128 -> 128 -> 64 feed-forward SNN and the final feature layer fixed to shift 3, does hidden-layer multi-tau organization improve native classification or frozen representation quality?

## Architectures

| ID | L1 shifts | L2 shifts | L3 shifts | Training |
|---|---|---|---|---|
| A | (3) | (3) | (3) | Reuse Experiment 3.0.1 `paired_split_v2`, shift 3 |
| B | (2,3) | (2,3) | (3) | New |
| C | (2,3,4) | (2,3,4) | (3) | New |
| D | (2,3) | (2,3,4,5) | (3) | New |
| E | (2,3) | (2,3,4) | (3) | New |

Multi-tau neurons are distributed as evenly as possible within each layer. L3 stays homogeneous at shift 3 so all architectures expose the same temporal output interface to the classifier.

## Training protocol

Inherited from Experiment 3.0.1 `paired_split_v2`:

- 64 Hz, 30 unsigned event channels.
- User-disjoint split with split seed 12345.
- Widths 128 / 128 / 64.
- `tau_mem_ms=22.54`, threshold 0.5, reset `subtract`.
- Objectives: `timestep_ce`, `relative10_sequence_ce`, `fixed250_sequence_ce`.
- Seeds: 11, 23, 101.
- 100 epochs, batch size 128, Adam lr 1e-3.
- Best checkpoint: maximum validation balanced accuracy, then minimum validation loss.
- Same master seed uses paired f1/f2/f3 initialization across B/C/D/E and objectives.

A contributes 9 reused runs. B/C/D/E add 36 training runs.

## Evaluation protocol

Every B/C/D/E Slurm array task trains exactly one run and then immediately freezes and evaluates that run on the same allocated CPU. This avoids a global train/evaluation barrier and reuses the already loaded dataset and checkpoint context.

Architecture A is not retrained. One separate single-CPU job sequentially evaluates its 9 reused Experiment 3.0.1 checkpoints.

For each frozen SNN:

1. Whole-layer L1/L2/L3 full-count representation + StandardScaler + LogisticRegression.
2. Whole-layer L1/L2/L3 fixed-250-ms representation + StandardScaler + LogisticRegression.
3. Whole-layer firing rate on train/val/test.
4. For genuinely multi-tau layers, each tau subgroup gets its own full-count probe, fixed250 probe, and firing rate.

Probe regularization C is selected on validation BA from `{1e-3, 1e-2, 1e-1, 1, 10}`. Test is evaluated only after validation selection.

Subgroup probes should primarily be compared within the same layer/configuration because subgroup neuron counts may differ between architectures.

The standalone `02_evaluate_one_run.py` and evaluation-array Slurm script are retained as maintenance tools for re-evaluating saved checkpoints without retraining, but they are not part of the formal submission pipeline.

## CPU execution

Each Slurm task uses one CPU core.

Formal pipeline:

- B/C/D/E Train+Eval array: `0-35%50` -> 36 tasks total, at most 36 CPU cores actually used.
- A baseline evaluation: one CPU core, sequentially evaluates 9 checkpoints.
- Finalizer: one CPU core, starts only after both of the above jobs complete successfully.

Therefore the formal pipeline uses at most 37 CPU cores concurrently, below the requested 50-core ceiling. No GPU is used.

The Train+Eval tasks have a 6-hour Slurm walltime to leave room for frozen probe evaluation after training.

## Submit

From the repository root after `git pull`:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_3_0_2_pipeline.bash
```

The submitter creates this dependency structure:

```text
36-task B/C/D/E Train+Eval array ----\
                                      -> finalizer
1-core A baseline evaluation --------/
```

Monitor with:

```bash
squeue -u $USER
```

The submitter prints a `sacct` command containing all three job IDs.

## Artifacts

Finalized files are written to:

```text
notebooks/artifacts/
  experiment_3_0_2_hidden_multitau_architecture_comparison/
    hidden_multitau_v1/
```

including native results/history, layer probes, tau-subgroup probes, firing rates, and mean+-SD summaries.

## Notebook

Open:

```text
notebooks/experiment_3_0_2_hidden_multitau_architecture_comparison.ipynb
```

The notebook is analysis-only. It reads finalized CSV files and produces native-BA, layer-probe, subgroup-probe, firing-rate, validation-trajectory, and ranking plots. It does not train networks, load checkpoints, or fit probes.
