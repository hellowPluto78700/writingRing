# Experiment 4.4.1 — Stage-2 memory architecture sweep

## Question

With the Raw64 local front-end fixed, how do passive Stage-2 membrane memory and dense recurrence interact to determine gesture-level memory quality and spike-accessible classification?

## Fixed protocol

- Input: Raw64, 30-channel scaled weighted event sequence.
- Local128: `shift34`, no local recurrence, unchanged from Exp4.0.6 / Exp4.3.
- Stage-2 width: 128.
- Event capacity: `multi_ho` (`hidden_cap=31`, `output_cap=31`).
- Threshold: 0.5.
- Output membrane time constant: 250 ms.
- Training objective: valid normalized Output WholeCount CE.
- Seeds: 11, 23, 37, 53, 71.
- Same split/scaling protocol inherited from Exp4.0.5 / Exp4.0.6.

## Swept factors

Stage-2 membrane time constant:

- 125 ms
- 250 ms
- 500 ms
- 1000 ms
- 2000 ms

Architecture:

- `ff`: no recurrent contribution, passive membrane state only.
- `rsnn`: dense recurrent `W_rec * S[t-1]` contribution enabled.

Total independent runs: `5 tau × 2 architectures × 5 seeds = 50`.

## Readout matrix

Every run reports the same three readouts:

1. Output WholeCount balanced accuracy — deployment-facing metric.
2. Hidden WholeCount + train-only scaled Linear probe balanced accuracy.
3. Stage-2 endpoint membrane `Uend` + train-only scaled Linear probe balanced accuracy — primary architecture/memory metric.

Also report:

- `Uend - HiddenCount` BA gap.
- `Uend - OutputCount` BA gap.
- Stage-2/output firing rate.
- Stage-2/output tail-event fraction.

The Linear probes never expose temporal phase slots; they are fit on train-user representations only.

## Multi-CPU execution

This follows `AGENTS.md`: one independent run per Slurm array task, one CPU core per task, train -> best checkpoint -> evaluate -> per-run artifact in the same task. The 50-task array is capped at 50 concurrent tasks.

From the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_4_1_cpu.bash
```

The finalizer runs only after all 50 tasks succeed and aggregates existing artifacts; it never retrains missing runs.

## Final artifacts

Written under:

```text
notebooks/artifacts/experiment_4_4_1_stage2_memory_sweep/stage2_tau_x_recurrence_v1/
```

Files:

- `runs.csv`
- `summary.csv`
- `paired_effects.csv`
- `paired_effects_summary.csv`
- `manifest.json`

Use `notebooks/experiment_4_4_1_stage2_memory_sweep.ipynb` for analysis and visualization only.
