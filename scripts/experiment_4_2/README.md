# Experiment 4.2 — Continuous recurrent controls

## Question

Experiment 4.0 leaves two explanations for the gap between Fixed250+Linear and RSNN:

1. shared recurrent sequence modeling itself may be a poor inductive bias for this user-disjoint task; or
2. recurrent modeling may be sufficient, while spiking neuron dynamics / spike-count readout create the remaining loss.

Experiment 4.2 separates these explanations using continuous-state controls on the **exact same scaled Fixed250 sequence**.

## Models

Both controls use `tanh` hidden units and bias-free Linear maps so that FF-ANN versus RNN changes only the recurrent state path.

### FF-ANN

```text
h_b = tanh(W_in z_b)
logits_b = W_out h_b
```

There is no recurrent connection. With padded zero input, the hidden state and logits return exactly to zero.

### Vanilla RNN

```text
h_b = tanh(W_in z_b + W_rec h_(b-1))
logits_b = W_out h_b
```

This has the same dense `H x H` recurrent topology and essentially the same trainable parameter count as the matched RSNN, but uses a continuous hidden state rather than LIF membrane/spikes.

## Readouts and objectives

Each forward pass exposes both readouts.

### Endpoint

```text
endpoint_logits = logits[B_i]
```

### Valid summed logits

```text
valid_sum_logits = sum_{b <= B_i} logits_b
```

The experiment trains both objectives as separate conditions:

```text
endpoint_ce
sum_logits_ce
```

This is deliberate. If endpoint CE succeeds but summed-logit CE does not, then a WholeCount-style accumulated readout is itself part of the bottleneck. If both continuous RNN objectives outperform RSNN, the major remaining loss is attributable to spiking dynamics rather than recurrence alone.

A full-padded summed-logit diagnostic is also recorded from the same trajectory. No state freeze is applied after the endpoint.

## Sweep

```text
model_type: ff_ann, rnn
hidden_width: 64, 128
objective: endpoint_ce, sum_logits_ce
seeds: 11, 23, 37, 53, 71
```

Total:

```text
2 x 2 x 2 x 5 = 40 runs
```

The finalizer additionally reads the existing Exp4.0 Linear baseline and matched FF-SNN/RSNN reference configurations:

```text
H=64,  tau_mem=1000 ms
H=128, tau_mem=250 ms
```

No Exp4.0 models are retrained.

## Shared data contract

All models consume the exact Exp4.0 Fixed250 representation:

- raw first 30 weighted event channels;
- 250 ms channel-wise aggregation;
- final non-empty partial bin retained;
- train-only per-channel standard-deviation scaling using valid bins only;
- no mean subtraction;
- padded bins remain exactly zero;
- same fixed user-disjoint split.

## Parameter matching

Ignoring biases (all maps are bias-free):

```text
FF-ANN: 30H + 12H
RNN:    30H + H^2 + 12H
```

Thus the RNN parameter formula exactly matches the Exp4.0 dense RSNN formula at the same hidden width.

## Multi-CPU execution

Following `AGENTS.md`:

```text
40 independent runs
  -> Slurm array 0-39%30
  -> one CPU core per task
  -> train -> best checkpoint -> evaluate -> per-run artifacts
  -> afterok finalizer
```

The Exp4.2 array is capped at 30 concurrent tasks. Together with Exp4.1's 20-task cap, the combined submit path never requests more than 50 simultaneous experiment CPU tasks.

Submit Experiment 4.2 alone:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_2_cpu.bash
```

Submit Exp4.1 + Exp4.2 together:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_1_4_2_cpu.bash
```

## Artifacts

```text
notebooks/artifacts/
  experiment_4_2_continuous_recurrent_controls/
    continuous_controls_v1/
      checkpoints/
      evaluations/
      runs.csv
      summary.csv
      references.csv
      provenance.json
```

`references.csv` contains the Exp4.0 Fixed250 Linear score and matched Exp4.0 FF-SNN/RSNN test results used only for comparison.

## Interpretation matrix

### RNN ~ Linear, RSNN far lower

Recurrent sequence modeling is sufficient; spiking dynamics / spike-count training are the major bottleneck.

### RNN ~ RSNN, both far below Linear

The main limitation is shared recurrent state compression / missing explicit temporal-slot structure, not spiking specifically.

### RNN between Linear and RSNN

Both effects matter: recurrent state compression costs information and spiking dynamics add a further penalty.

### Endpoint CE >> summed-logit CE

Accumulated WholeCount-like readout is itself a significant source of loss and should be redesigned before increasing SNN capacity.
