# Exp0.1.3 — Doc-faithful SAE-Dense Regularization on Binary Direct SNNs

## Scientific question

Does the source-document SAE-Dense regularizer improve the binary direct-SNN conditions from Exp0.1 when the SNN architecture and every non-regularization training parameter are unchanged?

This experiment is deliberately narrower than Exp0.1.2. It removes the multi-H comparison and removes the Exp0.1.2 regularizer adaptations that used pre-reset membrane, layer/neuron means, all-hidden A1, and cap normalization.

## Frozen Exp0.1 contract

Exp0.1.3 reuses the Exp0.1 direct-SNN definitions without changing:

- Raw64 30-channel unsigned weighted-event input.
- Hidden width = 128.
- Output = 12 binary spiking class neurons.
- The exact three direct architectures and their multi-tau synaptic shift groups.
- `whole_count_ce` and `timestep_ce` objectives.
- Output WholeCount inference/readout.
- Validation Output WholeCount BA checkpoint selection, with normalized WholeCount CE as tie-break.
- Optimizer, learning rate, weight decay, batch size, epoch count, split, and paired seeds.

Only the hidden spike cap is in scope here, and it is fixed to binary (`hidden_cap=1`).

## Run matrix

Three direct architectures × two objectives × five seeds = **30 new regularized runs**.

Seeds: `11, 23, 37, 53, 71`.

The matching 30 Exp0.1 binary direct-SNN runs are frozen `reg_off` controls and are never retrained by Exp0.1.3.

The Exp0.1 SNN+Fixed250 Linear binary probe and raw linear baselines are carried only as frozen comparison context. They are not part of the 30-run training array.

## Doc-faithful SAE-Dense regularization

For each spiking layer, including the output layer, track the **post-reset membrane** returned by the neuron update:

```text
U_l(t) = 0.99 * U_l(t-1) + mem_post_l(t)
```

For variable-length samples, `U_l` is updated only while the timestep is valid and then frozen at that sample's endpoint. This is the only adaptation required because the source formulation assumes fixed `T`.

### P2

At each sample endpoint, use the source-style direct sum:

```text
P2 = sum_l sum_batch sum_neurons U_l(T)^2
```

Tracked P2 layers are **all spiking layers**, including the output class layer.

There is no division by batch size, neuron count, layer count, hidden cap, or temporal accumulation mass.

### A1

SAE-Dense applies activity regularization only to the latent/bottleneck layer. In Exp0.1.3, the latent layer is the **last hidden spiking layer**.

For sample `b` with valid length `T_b`:

```text
A1_b = sum_i sum_{t < T_b} spike[b,t,i] / T_b
A1   = sum_b A1_b
```

Because Exp0.1.3 is binary-only, the spike signal is already in `{0,1}` and no cap normalization is used.

### Total objective

For either frozen Exp0.1 task objective:

```text
L_total = L_task + 0.01 * P2 + 0.1 * A1
```

`P1=0` and the source SAE-Dense L2 term is disabled. Exp0.1.3 otherwise preserves Exp0.1's optimizer and weight-decay setting exactly.

## Why this experiment follows Exp0.1.2

Exp0.1.2 showed severe firing collapse under the adapted regularizer. Exp0.1.3 isolates whether that collapse came from the adaptation or from applying the source SAE-Dense regularizer to this classification SNN at all.

The decisive comparison is paired by architecture, objective, and seed:

```text
delta_test_ba = BA(doc_reg_on) - BA(frozen_Exp0.1_off)
```

Training histories also retain total P2, A1, weighted regularizer-to-task ratio, and per-layer P2 contributions, including `train_p2_output`, so any new collapse can be localized.

## Multi-CPU execution

Each independent run is one Slurm array task and one CPU core:

```text
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=1
```

Each task performs `train -> best-checkpoint selection -> evaluate -> save artifacts` atomically. The finalizer starts only through `afterok` after all 30 runs succeed and aggregates existing artifacts only.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_0_1_3_doc_faithful_regularization_cpu.bash
```

The submit wrapper requires finalized Exp0.1 `runs.csv` and `baseline_results.csv` before launching.

## Finalized artifacts

The finalizer writes under:

```text
notebooks/artifacts/experiment_0_1_3_doc_faithful_regularization/doc_faithful_sae_dense_binary_v1/
```

Key files:

- `runs.csv`: 30 frozen Exp0.1 binary direct controls + 30 paired doc-regularized runs.
- `summary.csv`: 5-seed aggregate by architecture/objective/regularization.
- `paired_regularization_effects.csv`: paired BA and last-hidden firing-rate deltas.
- `regularization_objective_interactions.csv`: whether the regularizer affects timestep CE differently from WholeCount CE.
- `regularization_architecture_interactions.csv`: whether the regularizer changes the long-layer/longer-shift penalties relative to `short_mid`.
- `binary_probe_context.csv`: frozen Exp0.1 binary post-hoc SNN probe context.
- `baseline_results.csv`: frozen raw baselines.
- `comparison_summary.csv`: compact cross-system comparison table.
- `manifest.json`: durable experiment contract.

## Notebook strategy

`notebooks/experiment_0_1_3_doc_faithful_regularization.ipynb` is analysis-only. It does not train, launch Slurm, or regenerate missing runs. It reads the finalized CSV/JSON artifacts, reports paired regularization effects, plots test-BA deltas, and inspects saved training histories for regularizer/task scale and P2 layer contributions.
