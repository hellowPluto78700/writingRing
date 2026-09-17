# Exp8.0.3 — Phase-aware hierarchical readout

This experiment trains the fixed `234x234` two-layer local SNN with four readout topologies:

- `l2_only`
- `l1_l2_timeshared`
- `l1_fixed250_l2_whole`
- `l1_capacity_no_phase_l2_whole`

The primary comparison is the phase-aware method versus the parameter-matched no-phase control.

## Submit on Unity

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_0_3_cpu.bash
```

This launches 12 independent CPU tasks (`4 methods x 3 seeds`) and one `afterok` finalizer.

## Run a single configuration

```bash
python -m scripts.experiment_8_0_3_phase_aware_hierarchical_readout \
  --device cpu --threads 1 run \
  --method l1_fixed250_l2_whole --seed 11
```

## Finalize manually

```bash
python -m scripts.experiment_8_0_3_phase_aware_hierarchical_readout \
  --device cpu --threads 1 finalize
```

## Aggregated outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_8_0_3_phase_aware_hierarchical_readout/
  phase_aware_hierarchical_readout_v1/
```

Important files:

- `method_runs.csv`, `method_summary.csv`
- `paired_deltas.csv`, `paired_delta_summary.csv`
- `probe_runs.csv`, `probe_summary.csv`
- `fusion_gain_runs.csv`, `fusion_gain_summary.csv`
- `correctness_overlap_runs.csv`, `correctness_overlap_summary.csv`
- `trained_head_runs.csv`, `trained_head_summary.csv`
- `phase_bin_runs.csv`, `phase_bin_summary.csv`
- `phase_structure_runs.csv`, `phase_structure_summary.csv`
- `history_runs.csv`
- `manifest.json`

The aggregation notebook is:

```text
notebooks/experiment_8_0_3_phase_aware_hierarchical_readout.ipynb
```

It is aggregation-only and does not retrain models or refit probes.
