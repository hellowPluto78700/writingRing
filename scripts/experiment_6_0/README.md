# Experiment 6.0 — Single-hidden multi-timescale SNN for phase-aware evidence

## Scientific question

Exp6.0 tests whether a causal single-hidden-layer SNN can turn raw local spike evidence into useful history-conditioned / phase-aware class evidence, without giving the network an explicit phase index.

The reference is the existing raw-spike `Fixed250 + Linear` solution, where ordered 250-ms bins make temporal position directly accessible to a linear classifier.

## Data and split contract

Exp6.0 reuses the existing Raw64 experiment contract:

- 64 Hz raw event spike train;
- 30 event channels;
- globally padded length 256 timesteps;
- one fixed user-disjoint split (`split_seed=12345`);
- the same train/validation/test users and class set returned by the existing Exp3 data loader.

Training randomness uses three paired seeds:

```text
11, 23, 37
```

The user split is not regenerated per training seed.

## SNN architecture

```text
Raw30 -> 128 hidden binary-spike neurons -> K binary-spike output neurons
```

The hidden layer is partitioned into three fixed synaptic-timescale groups:

```text
shift_tau_syn = 4 : 43 neurons
shift_tau_syn = 5 : 43 neurons
shift_tau_syn = 6 : 42 neurons
```

The hidden synaptic current uses the normalized low-pass update

```text
I_t = alpha * I_{t-1} + (1 - alpha) * W x_t
alpha = 1 - 2^(-shift_tau_syn)
```

so long synaptic time constants change temporal horizon without the `1/(1-alpha)` DC amplification of the previous unnormalized update.

All 128 hidden neurons in one run share one membrane shift. Exp6.0 sweeps

```text
shift_tau_mem in {1, 2, 3}
```

The output-neuron dynamics are fixed across the sweep. Hidden and output communication are strictly binary (`cap=1`).

## Main readout and task loss

The deployment readout is valid-length output whole count. Padding does not contribute to the class count.

```text
L_WC = CE(valid-length output whole-count logits, y)
```

The best checkpoint is chosen by validation Balanced Accuracy, with lower validation WholeCount CE as the tie-breaker. The test set is evaluated only after reloading that checkpoint.

## Objective 1 — WholeCount only

```text
L = L_WC
```

This is the clean baseline asking whether sequence-level supervision alone makes the heterogeneous hidden dynamics discover useful temporal context.

## Objective 2 — WholeCount + order-HCE

At 64 Hz, 250 ms is exactly 16 timesteps. For each valid sequence, split only the divisible valid prefix into complete 16-step bins:

```text
X1, X2, ..., XB, R
```

where `R` is the incomplete valid remainder. Construct the counterfactual sequence

```text
XB, X(B-1), ..., X1, R
```

with these invariants:

- order inside every 16-step bin is unchanged;
- the incomplete remainder is unchanged and stays at the end;
- padding is unchanged;
- the reversed branch receives no classification CE.

For original and reversed order, sum/average the hidden-to-output pre-output class evidence over the valid region and compute the correct-class margin

```text
m(E,y) = E_y - logsumexp(E_not_y)
```

The auxiliary loss is

```text
L_HCE = softplus(m_reversed - m_original)
L = L_WC + L_HCE
```

Thus the same set of local 250-ms segments should support the correct class more strongly when presented in the true temporal order.

## Objective 3 — WholeCount + Contextual Gain

Partition the output weight matrix by the hidden `s4/s5/s6` neuron groups. For every complete 250-ms bin in the original sequence, compute class-evidence contributions

```text
E4
E45  = E4 + E5
E456 = E4 + E5 + E6
```

and gains

```text
G5 = margin(E45,y)  - margin(E4,y)
G6 = margin(E456,y) - margin(E45,y)
```

The gains are averaged before applying the penalty; Exp6.0 does not require every local bin to benefit from longer context.

```text
L_CG = softplus(-mean(G5)) + softplus(-mean(G6))
L = L_WC + L_CG
```

Both auxiliary weights are fixed to 1.0 in this first protocol; they are not swept.

## Experiment matrix

```text
3 objectives
x 3 hidden membrane shifts
x 3 training seeds
= 27 independent SNN runs
```

Every run trains for exactly 100 epochs.

## Fixed250 + Linear reference

A separate one-core job fits the raw-spike reference using the identical split:

```text
Raw64
-> ordered non-overlapping 250-ms channel counts
-> flatten
-> train-only StandardScaler
-> validation-selected LogisticRegression C
```

This is the explicit-temporal-position reference for the causal SNN comparison.

## Per-epoch artifacts

Each SNN worker writes only run-unique files. Every epoch records at least:

- train / validation total loss;
- train / validation WholeCount CE;
- train / validation accuracy;
- train / validation Balanced Accuracy;
- HCE margins/loss for the HCE condition;
- middle/long contextual gains and CG loss for the Contextual Gain condition.

Each run also writes:

- a two-panel train/validation loss and BA learning curve;
- a component-loss plot;
- the validation-selected checkpoint;
- one final train/validation/test evaluation JSON.

## 256-timestep spike-activity diagnostic

All 27 runs use the same deterministic validation segment: the first validation sample whose valid length is closest to the validation median. The sample is selected without inspecting model results.

Using the best-validation checkpoint, every run performs a 256-timestep inference with zero raw input after the valid endpoint and saves:

```text
input_spikes  [256, 30]
hidden_spikes [256, 128]
output_spikes [256, K]
```

The raster plot uses timestep on x and neuron index on y, marks the valid endpoint, and separates the hidden `s4/s5/s6` groups. It also records valid-region and zero-tail firing fractions for every hidden group and the output layer.

For HCE runs, the same sample is additionally evaluated and plotted with reversed complete 250-ms bins.

## Multi-CPU / Slurm policy

Exp6.0 avoids shared-worker writes:

```text
1 one-core Fixed250 baseline job
27 independent one-core SNN array tasks
1 one-core artifact-only finalizer
```

Every SNN array task owns its checkpoint/history/evaluation/activity paths. No worker appends to a shared CSV. The finalizer runs only after the baseline and all 27 SNN tasks succeed; it reads existing artifacts and produces finalized tables/plots.

CPU math-library thread counts are fixed to one to avoid hidden oversubscription.

## Finalized artifacts

```text
notebooks/artifacts/
  experiment_6_0_multiscale_phase_evidence/
    single_hidden_multiscale_phase_evidence_v1/
      checkpoints/
      histories/
      evaluations/
      learning_curves/
      loss_components/
      activities/
      rasters/
      summary_plots/
      baseline_fixed250_linear.json
      runs.csv
      summary.csv
      history_index.csv
      visualization_sample.json
      manifest.json
```

Summary plots include:

- test BA vs hidden membrane shift, mean +/- SD over the three seeds, with the Fixed250+Linear reference;
- validation BA vs hidden membrane shift;
- `s4/s5/s6` valid-region firing-rate summaries.

## Analysis notebook

```text
notebooks/experiment_6_0_multiscale_phase_evidence.ipynb
```

The notebook is analysis-only. It reads finalized artifacts and never trains models or mutates run outputs.

## Submit on Unity

From the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_6_0_cpu.bash
```

To inspect the 27 array identities without training:

```bash
python -m scripts.experiment_6_0_multiscale_phase_evidence list-runs
```

## Validation

```bash
python -m pytest -q tests/test_experiment_6_0_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
