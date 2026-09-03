# Experiment 4.4.2 — Stage-2 recurrence topology

## Question

At the two useful Stage-2 membrane regimes identified by Exp4.4.1, is neuron-wise self recurrence enough to form the gesture-level endpoint state, or is full population mixing from dense recurrence important?

## Fixed protocol

- Input: Raw64, 30-channel scaled weighted event sequence.
- Local128: `shift34`, unchanged from Exp4.0.6 / Exp4.3 / Exp4.4.1.
- Stage-2 width: 128.
- Event capacity: `multi_ho` (`hidden_cap=31`, `output_cap=31`).
- Threshold: 0.5.
- Output membrane time constant: 250 ms.
- Training objective: valid normalized Output WholeCount CE.
- Seeds: 11, 23, 37, 53, 71.
- Same split/scaling protocol as Exp4.4.1.

## Conditions

Stage-2 tau:

- 250 ms — primary condition, best Exp4.4.1 Dense-RSNN `Uend` regime.
- 500 ms — secondary condition, still strong `Uend`, retained to test topology × tau interaction.

Recurrence topology:

- `ff`: no recurrent contribution. Reused exactly from Exp4.4.1.
- `diagonal`: one trainable recurrent self-weight per Stage-2 neuron; no cross-neuron recurrent mixing.
- `dense`: full 128×128 recurrent matrix. Reused exactly from Exp4.4.1.

Only diagonal models are newly trained: `2 tau × 5 seeds = 10` new runs.

The diagonal recurrent vector is initialized from the diagonal of the paired Dense-RSNN recurrent matrix after all shared feed-forward layers are initialized. This keeps the shared initialization stream aligned with Exp4.4.1.

## Parameter caveat

Recurrent parameter counts are intentionally not matched:

- FF: 0 active recurrent parameters.
- Diagonal: 128 recurrent parameters.
- Dense: 16,384 recurrent parameters.

Therefore `Dense > Diagonal` supports the efficacy of population-level recurrent mixing, but by itself does **not** prove that cross-neuron recurrence is superior at matched parameter capacity. A later low-rank/sparse control would be required for that stronger claim.

## Readout matrix

Every topology is compared with the same diagnostics:

1. Output WholeCount BA.
2. Hidden WholeCount + train-only scaled Linear BA.
3. Stage-2 endpoint membrane `Uend` + train-only scaled Linear BA.
4. `Uend - HiddenCount` and `Uend - OutputCount` gaps.
5. Stage-2/output firing rates and tail-event fractions.

## Multi-CPU execution

Only the 10 new diagonal runs execute. Each Slurm array task owns one `(tau, seed)` run, one CPU core, and performs train -> best checkpoint -> evaluation -> per-run artifact. Baseline FF/Dense rows are read from finalized Exp4.4.1 artifacts by the finalizer.

From repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_4_2_cpu.bash
```

## Final artifacts

Written under:

```text
notebooks/artifacts/experiment_4_4_2_recurrence_topology/stage2_recurrence_topology_v1/
```

Files:

- `runs.csv`
- `summary.csv`
- `paired_effects.csv`
- `paired_effects_summary.csv`
- `manifest.json`

Use `notebooks/experiment_4_4_2_recurrence_topology.ipynb` for aggregation and visualization only.
