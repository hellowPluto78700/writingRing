# Exp5.4.3 — Causal elapsed-time readout capacity

## Goal

Determine whether the remaining gap between frozen-WHAT whole-count readout and frozen-WHAT Fixed250 + full Linear is caused by insufficient capacity in the causal elapsed-time-conditioned readout.

## Required behavior

1. Use the same five frozen WHAT seeds and user split as Exp5.4.1/5.4.2. Do not train a new WHAT or WHEN representation.
2. Build a paired Fixed250 full-Linear reference per seed from train/validation only. Convert the standardized classifier into raw-count-space phase-specific matrices and require offline/streaming logits to match within `1e-6`.
3. Stage A sweeps `K={4,8,16}` and `r={4,8,12}` using exact causal elapsed time, whole-sequence CE, frozen WHAT base, and `alpha=0` initialization. One seed/configuration is one one-core Slurm task.
4. Select on validation only. Quantify recovery of the Fixed250-over-WHAT validation gap. Strong recovery is preregistered as mean recovery >= 90% and mean gap to Fixed250 <= 1 BA point.
5. If Stage A does not close the gap, run one 15-task validation-only diagnostic extension: higher K at full rank when K16 continues to improve over K8 on >=4/5 seeds; otherwise direct full-matrix banks at K={4,8,16}.
6. Lock `selection.json` before constructing/evaluating final test loaders. Final test is five independent seed tasks.
7. Finalizers aggregate existing per-run artifacts only. The notebook is analysis-only and never trains/recomputes missing runs.

## Preserved contracts

- WHAT source, labels, split, and sampling rate are inherited from the existing Exp5.4 cache and Exp5.4.1 base checkpoints.
- Exact elapsed time is causal and uses no final gesture duration.
- Whole-sequence CE is the only train objective.
- Rank never exceeds 12 because a 12 x 128 readout matrix has maximum rank 12.
- Epoch 0 must reproduce frozen WHAT logits exactly because `alpha=0`.
- Test metrics cannot affect Stage-A or Stage-B selection.
- Slurm compute jobs use real Bash scripts, not `sbatch --wrap`, initialize Conda on compute nodes, and use one CPU thread per independent task.

## Acceptance criteria

- Stage-A mapping has exactly 45 unique run specs and no more than 45 concurrent tasks.
- Stage-B mapping, when required, has exactly 15 unique run specs.
- Fixed250 offline/raw-space/streaming validation logits match within `1e-6` and predictions are identical.
- `capacity_selection.json` is validation-only and either locks a strong Stage-A selection or preregisters exactly one extension mode.
- `selection.json` is validation-only and is required before final test evaluation.
- Final artifacts include reference/capacity/recovery tables, selected-test metrics, gate activity, effective-weight distance, effective rank, prefix BA, residual metrics, rescue/harm, and a manifest.
- Focused contract tests and repository source syntax checks pass in CI.
