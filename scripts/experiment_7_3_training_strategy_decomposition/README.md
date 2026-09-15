# Exp7.3 — training-strategy decomposition with common LIF deployment

## Scientific question

Exp7.2.4 showed that TSCE can shape a strong L2 representation, while Exp7.2.6.4 showed that, for a frozen L2, a weight matrix learned through an Analog/Linear path can outperform a weight matrix learned through the LIF surrogate path when both are finally deployed through the same LIF output neuron.

Exp7.3 separates three choices:

1. how L1/L2 are trained;
2. how the final 128->12 matrix `W` is trained;
3. how the trained model is deployed.

The deployment interface is held fixed for the primary comparison:

```text
L2 spikes -> W -> LIF(beta=0.5, threshold=0.5, cap=1) -> valid spike count -> argmax
```

`final_lif_test_ba` is the primary metric for every method.

## Locked common contract

- architecture: `234x234`;
- L1 shifts `(2,3,4)`, width 128;
- L2 shifts `(2,3,4)`, width 128;
- binary hidden spikes;
- bias-free hidden and output linear matrices;
- task-only loss, no activity regularizer;
- seeds `11,23,37`;
- maximum 100 epochs, minimum 20 epochs, patience 30;
- CE gain fixed to `1.0` for this experiment;
- output LIF: `beta=0.5`, repository threshold, max one output spike/neuron/timestep;
- checkpoint selection uses the readout that is actually being trained: native validation BA primary, native objective loss tiebreak;
- test is evaluated only after checkpoint selection.

For WCCE, the training logits are the valid-length temporal mean. This is classification-equivalent to valid whole-count at inference but avoids changing CE scale with sample duration. TSCE applies CE to every valid timestep.

## Group A — joint end-to-end training

All of `L1`, `L2`, and `W` are trainable.

| Method | Train readout after L2 | Objective | Final deployment |
|---|---|---|---|
| `A1_e2e_linear_tsce` | Linear evidence | TSCE | same W -> LIF count |
| `A2_e2e_linear_wcce` | Linear evidence | WCCE | same W -> LIF count |
| `A3_e2e_lif_tsce` | LIF spikes | TSCE | same LIF count |
| `A4_e2e_lif_wcce` | LIF spikes | WCCE | same LIF count |

Within a seed, all four conditions share the same initial L1/L2/W parameters and training-sample order. Readout type and objective are excluded from the pairing seed.

## Group B — two-stage training

### Stage 1: representation learning

Two backbones are created by the Linear E2E runs from Group A:

```text
TS backbone: A1 Linear + TSCE -> trained L1/L2
WC backbone: A2 Linear + WCCE -> trained L1/L2
```

The temporary Stage-1 W is discarded. L1/L2 are frozen and their complete valid/padded L2 trajectories are cached once per seed/backbone.

### Stage 2: retrain only W

A fresh bias-free `W: 128 -> 12` is initialized and trained on the frozen L2 cache.

| Method | Frozen backbone | W-train readout | W objective | Final deployment |
|---|---|---|---|---|
| `B1_tsbackbone_linear_tsce` | TSCE | Linear | TSCE | same W -> LIF count |
| `B2_tsbackbone_linear_wcce` | TSCE | Linear | WCCE | same W -> LIF count |
| `B3_tsbackbone_lif_tsce` | TSCE | LIF | TSCE | same LIF count |
| `B4_tsbackbone_lif_wcce` | TSCE | LIF | WCCE | same LIF count |
| `B5_wcbackbone_linear_tsce` | WCCE | Linear | TSCE | same W -> LIF count |
| `B6_wcbackbone_linear_wcce` | WCCE | Linear | WCCE | same W -> LIF count |
| `B7_wcbackbone_lif_tsce` | WCCE | LIF | TSCE | same LIF count |
| `B8_wcbackbone_lif_wcce` | WCCE | LIF | WCCE | same LIF count |

All eight Stage-2 conditions for one seed share the same fresh W initialization and sample order. Only W is trainable.

The main hypothesis case is `B2`: TSCE shapes L1/L2, WCCE learns the final continuous evidence projection, and the resulting W is then realized through the common LIF output.

## Diagnostics saved for every method

Every selected checkpoint is evaluated with:

- native train/val/test BA under the readout used during training;
- `same-W Analog` valid-sequence BA;
- `same-W final LIF(beta=0.5)` spike-count BA;
- Analog-to-LIF BA gap;
- frozen-L2 whole-count linear probe;
- frozen-L2 ordered Fixed250 linear probe.

The two L2 probes use the repository-standard train-only `StandardScaler` plus validation-selected `LogisticRegression`. They diagnose whether a training strategy damaged the hidden representation even when the final readout is weak.

## Key controlled comparisons

- `A1 vs A3`: TSCE E2E, Linear path vs through-LIF path.
- `A2 vs A4`: WCCE E2E, Linear path vs through-LIF path.
- `B2 vs B6`: TSCE-trained backbone vs WCCE-trained backbone with identical Stage-2 Linear/WCCE training.
- `B2 vs B1`: WCCE vs TSCE for learning W on the same TSCE backbone.
- `B2 vs B4`: Linear vs through-LIF learning of W on the same TSCE backbone with WCCE.
- `B2 vs A1/A2/A3/A4`: decoupled representation/projection learning against all joint E2E strategies.

Positive `left - right` values in `contrast_summary.csv` mean the left method has higher final-LIF test BA.

## Multi-CPU execution

Exp7.3 follows `AGENTS.md` task-level CPU execution:

```text
12 E2E runs (4 methods x 3 seeds), one CPU each
        |
        +--> 6 Stage-2 L2 cache/probe tasks
                 |
                 +--> 24 Stage-2 W-training runs (8 methods x 3 seeds), one CPU each
                          |
                          +--> finalizer / aggregator
```

Each training array task performs train -> best checkpoint selection -> evaluation -> per-run artifacts atomically. The finalizer never retrains or regenerates missing runs.

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_cpu.bash
```

## Durable aggregate outputs

Finalized artifacts are written under:

```text
notebooks/artifacts/experiment_7_3_training_strategy_decomposition/training_strategy_decomposition_v1/
```

Important files:

- `manifest.json`
- `method_runs.csv` — 36 method/seed rows
- `method_summary.csv` — mean/std by method
- `two_stage_matrix_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`

Per-run histories, checkpoints, E2E evaluations, Stage-2 caches, and Stage-2 evaluations are retained for auditability.

## Notebook contract

`notebooks/experiment_7_3_training_strategy_decomposition.ipynb` is analysis-only. It reads finalized CSV/JSON files and shows method-level comparisons. It does not import Torch, launch subprocesses, invoke Slurm, train models, or perform hidden regeneration.
