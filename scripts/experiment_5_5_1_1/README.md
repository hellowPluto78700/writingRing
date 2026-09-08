# Experiment 5.5.1.1 — Routing diagnostic extension

## Question

Exp5.5.1 showed that progress supervision makes the Ordered-GRU history state more temporally organized, but Ordered still does not consistently beat matched Reset. This diagnostic extension asks which routing bottleneck is now dominant:

1. **Diffuse q:** the state head ranks useful experts but mixes too many of them at each timestep.
2. **Weak expert specialization:** the eight full `12 x 128` expert matrices behave too similarly, so sharper q cannot materially change class evidence.

This experiment is post-hoc and performs **no training**.

## Source

For each seed `(11, 23, 37, 53, 71)`, reuse the finalized Exp5.5.1 selected Ordered checkpoint and the same frozen Local-SNN L2 WHAT representation. Source checkpoints are read-only.

## Diagnostics

### q confidence

On validation and test valid timesteps report:

- mean/median and quantiles of `max(q)`;
- mean/median top1-top2 probability margin;
- entropy and normalized entropy;
- effective state count `exp(H(q))`;
- fractions with `max(q) >= 0.3, 0.4, 0.5, 0.6`.

### Post-hoc routing sharpness

Evaluate the unchanged model after replacing q by:

`softmax(log(q) / tau)` for `tau in (1.0, 0.8, 0.6, 0.4, 0.25)`, plus hard top-1 one-hot routing.

Temperature 1.0 must reproduce the finalized Exp5.5.1 Ordered test BA. All temperatures are reported; **Exp5.5.1.1 does not select a temperature**.

### Temporal-alignment sensitivity

For every routing variant, permute q within each gesture's valid prefix for five deterministic replicates. This preserves each transformed q distribution while destroying WHAT/q temporal alignment.

### Expert specialization

For all 28 unordered K=8 expert pairs, report:

- flattened weight cosine similarity;
- absolute and relative weight L2 distance;
- on active WHAT vectors, mean pairwise class-evidence L1/L2 difference;
- relative evidence L2 difference;
- evidence cosine similarity;
- top class-support disagreement rate.

The functional diagnostic is important because different parameters are only useful if they induce meaningfully different `W_k m_t` class evidence.

## Multi-CPU execution

Run from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_5_1_1_cpu.bash
```

Execution is:

```text
5 independent seed diagnostic tasks
        -> afterok
1 artifact-only finalizer
```

Each diagnostic task uses one CPU core, initializes Conda on the compute node, prefers `writingring-gpu`, falls back to `writingring-viz`, and fixes BLAS/OpenMP thread counts to one.

## Artifacts

Finalized outputs live under:

```text
notebooks/artifacts/experiment_5_5_1_1_routing_diagnostics/routing_diagnostics_v1/
```

Key files:

- `confidence_metrics.csv`
- `confidence_summary.csv`
- `temperature_sweep.csv`
- `temperature_summary.csv`
- `temperature_shuffled_q.csv`
- `temperature_alignment.csv`
- `expert_weight_similarity.csv`
- `expert_functional_diversity.csv`
- `expert_summary.csv`
- `manifest.json`

Per-seed JSON artifacts are retained under `per_seed/`.

## Notebook policy

`notebooks/experiment_5_5_1_1_routing_diagnostics.ipynb` is analysis-only. It reads finalized artifacts, makes tables/plots, and never retrains, launches Slurm, reruns diagnostics, or regenerates missing artifacts.
