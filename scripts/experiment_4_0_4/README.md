# Experiment 4.0.4 — Hidden-state memory vs fully-spiking readout

## Question

Experiment 4.0.1 showed a strong interaction between hidden and output event caps. In the recurrent anchor, the `multi_h` condition `(hidden cap=31, output cap=1)` performed much worse than both binary and `multi_ho`. Experiment 4.0.3 then showed that a post-hoc Linear probe on the recurrent endpoint membrane can outperform the fully-spiking WholeCount readout.

Exp4.0.4 asks a narrower diagnostic question:

> When an Exp4.0.1 recurrent model performs poorly, especially Multi-H with a binary output layer, is the recurrent hidden memory itself poor, or is useful information present in the hidden state but lost after the hidden-to-output spiking interface?

This is a frozen-checkpoint diagnostic. No SNN is retrained.

## Fixed source model

Only the Exp4.0.1 recurrent memory anchor is used:

- architecture: `rsnn`
- hidden width: 128
- hidden membrane time constant: 250 ms
- input: exactly the same scaled Fixed250 vectors as Exp4.0/4.0.1
- seeds: `11, 23, 37, 53, 71`

The four existing Exp4.0.1 variants are probed:

| Variant | Hidden cap | Output cap | Role |
| --- | ---: | ---: | --- |
| `binary` | 1 | 1 | binary reference |
| `multi_h` | 31 | 1 | rich hidden communication, binary output bottleneck |
| `multi_o` | 1 | 31 | binary hidden bottleneck, rich output communication |
| `multi_ho` | 31 | 31 | rich hidden and output communication |

Total frozen checkpoint probes:

`4 variants × 5 seeds = 20 tasks`.

## Probe feature

For every gesture, replay the frozen Exp4.0.1 hidden dynamics and collect the hidden LIF post-reset membrane after every Fixed250 macro step. At the causal valid endpoint `B_i`, select exactly one vector:

`h_i = U_hidden[B_i] in R^128`.

The probe therefore receives one 128-D endpoint vector per gesture.

It does **not** receive:

- raw spike trains;
- sub-bin timing;
- a flattened list of Fixed250 phases;
- output-neuron membrane states;
- output spikes or output WholeCount.

Thus the probe cannot reconstruct temporal order by assigning separate weights to different phases. Any history available to the probe must already have been compressed into the recurrent hidden endpoint state by the frozen SNN.

## Linear probe

For each source checkpoint independently:

1. Extract train endpoint hidden states.
2. Fit `LogisticRegression(class_weight="balanced", solver="lbfgs")` on train states only.
3. Evaluate the frozen probe on train, validation, and test endpoint states.
4. Save one per-checkpoint JSON artifact.

The probe is diagnostic only. It is not part of the original Exp4.0.1 training and is not a deployment claim.

## Main comparison

The finalizer merges the new endpoint-state probe metrics with the existing Exp4.0.1 fully-spiking valid WholeCount metrics for the exact same `variant × seed` checkpoint.

The central table is therefore:

| Variant | Fully-spiking WholeCount BA | Endpoint hidden-state + Linear BA | Probe − spiking |
| --- | ---: | ---: | ---: |
| Binary | existing | new | new |
| Multi-H | existing | new | new |
| Multi-O | existing | new | new |
| Multi-HO | existing | new | new |

Interpretation for Multi-H is especially important:

- If `Multi-H endpoint-state probe ≈ Multi-H fully-spiking BA`, then the hidden recurrent state itself is poor under the Multi-H training formulation.
- If `Multi-H endpoint-state probe >> Multi-H fully-spiking BA`, then useful recurrent memory exists but the binary output path fails to expose it.
- If `Multi-H endpoint-state probe` approaches the Multi-HO probe despite much worse fully-spiking BA, that is strong evidence that the output cap/readout is the dominant reason Multi-H appeared poor.

The same comparison is repeated for Binary, Multi-O, and Multi-HO so that hidden-cap and output-cap effects can be separated from the final spiking readout.

## Multi-CPU execution

Exp4.0.4 follows `AGENTS.md` task-level CPU parallelism even though the SNNs are frozen, because the 20 checkpoint probes are independent and the user explicitly requested multi-CPU execution.

Each Slurm array task:

`load one frozen checkpoint -> extract endpoint states -> fit one Linear probe -> evaluate -> write one JSON`.

The array is:

```bash
#SBATCH --array=0-19%20
#SBATCH --cpus-per-task=1
```

Every task initializes Conda locally and sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.

After all 20 tasks succeed, one `afterok` finalizer aggregates artifacts only. It raises on missing tasks rather than silently regenerating them.

Run on Unity from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_4_cpu.bash
```

## Final artifacts

Artifacts are written to:

`notebooks/artifacts/experiment_4_0_4_hidden_state_probe/rsnn_cap_variant_endpoint_state_probe_v1/`

Expected finalized files:

- `probes/*.json` — 20 independent probe artifacts;
- `runs.csv` — probe metrics merged with the matching Exp4.0.1 fully-spiking metrics;
- `summary.csv` — mean/std by cap variant;
- `paired_effects.csv` — seed-paired differences, including probe-vs-spiking and variant-vs-binary effects;
- `provenance.json` — frozen source and probe contract.

## What this experiment does not test

Exp4.0.4 does not:

- retrain Binary/Multi-H/Multi-O/Multi-HO networks;
- add a Linear layer to the deployed fully-spiking output;
- recover within-bin timing;
- change Fixed250;
- test extra SNN depth;
- claim that an analog Linear probe is a hardware deployment solution.

Its only purpose is to locate the information bottleneck between recurrent hidden memory and the existing fully-spiking output/readout.
