# Experiment 4.0.1 — Binary vs multi-spike macro-LIF

## Question

Experiment 4.0 compresses each 250 ms Fixed250 bin into one SNN macro timestep. The input to that timestep is a 30-D weighted event-mass vector, but the original snnTorch LIF communication is capped at one emitted event per neuron per macro timestep.

Experiment 4.0.1 tests whether this one-event-per-250-ms communication constraint is a major source of the current FF-SNN / RSNN performance loss.

This experiment is intentionally inserted before further interpretation of Experiments 4.1 and 4.2. Their continuous ANN/RNN controls remain valid, but their ANN–SNN gap should only be interpreted against the binary macro-step SNN until this event-cap ablation is known.

## Shared Fixed250 representation

The experiment reuses Experiment 4.0 data preparation exactly:

```text
30 weighted event channels
  -> 250 ms channel-wise aggregation
  -> retain final non-empty partial bin
  -> train-valid-bin per-channel std scaling
  -> no mean subtraction
  -> padded bins remain zero
```

The split, labels, scaling, batch size, optimizer, threshold, and endpoint semantics are inherited from Experiment 4.0.

## Controlled macro-LIF implementation

All four variants use the same custom `MacroMultiSpikeLIF`. This is important: the binary control is retrained through the same update and reset code as the multi-spike variants, so the experiment does not confound snnTorch reset timing with event multiplicity.

For every macro timestep:

```text
pre = beta * membrane + current
events = clip(floor(max(pre, 0) / threshold), 0, event_cap)
membrane = pre - events * threshold
```

The backward pass uses a surrogate derivative formed by summing fast-sigmoid derivatives around each reachable threshold crossing.

### Binary cap

```text
event_cap = 1
events in {0, 1}
```

### Multi-spike cap

```text
event_cap = 31
events in {0, 1, ..., 31}
```

## 2 x 2 factorial

Two factors are varied independently:

```text
hidden_cap in {1, 31}
output_cap in {1, 31}
```

| Variant | Hidden max events / dt | Output max events / dt | Role |
| --- | ---: | ---: | --- |
| `binary` | 1 | 1 | custom-binary baseline |
| `multi_h` | 31 | 1 | primary hidden/recurrent multi-event condition |
| `multi_o` | 1 | 31 | output-resolution diagnostic |
| `multi_ho` | 31 | 31 | software upper-bound diagnostic |

`multi_h` is the primary hardware-motivated condition because it tests richer hidden and recurrent event communication while leaving output binary. `multi_o` and `multi_ho` are diagnostic conditions and should not be described as strict output-hardware configurations.

## Output evidence normalization

Changing output cap from 1 to 31 would otherwise change both resolution and raw softmax-logit scale. To isolate resolution, the primary CE readout is:

```text
normalized_output_event = raw_output_event / output_cap
valid_evidence = sum(normalized_output_event over valid bins)
loss = CE(valid_evidence, label)
```

Thus each output neuron contributes within `[0, 1]` per macro timestep in every condition, while cap=31 provides 32 discrete levels instead of two.

Hidden and recurrent communication are **not divided by 31**. Their raw event multiplicity is the quantity under test.

## Architecture anchors

Experiment 4.0 already identified representative FF and RSNN operating points. Exp4.0.1 does not repeat the width/tau sweep.

```text
FF:   H=128, tau_mem=2000 ms
RSNN: H=128, tau_mem=250 ms
```

For RSNN:

```text
current_b = W_in z_b + W_rec events_(b-1)
```

so increasing hidden cap directly increases the amplitude resolution of both hidden-to-output and hidden-to-hidden recurrent communication.

## Run matrix

```text
2 architecture anchors
x 4 event-cap variants
x 5 seeds (11, 23, 37, 53, 71)
= 40 independent runs
```

Every variant, including `binary`, is retrained.

## Multi-CPU execution

Following `AGENTS.md`:

```text
40 independent runs
  -> Slurm array 0-39%40
  -> one CPU core per task
  -> train
  -> select best validation checkpoint
  -> evaluate train/val/test from that checkpoint
  -> save checkpoint + evaluation JSON
all tasks succeed
  -> afterok finalizer
  -> analysis-only notebook
```

Submit:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_4_0_1_cpu.bash
```

The finalizer never retrains missing runs.

## Primary factorial effects

For test balanced accuracy:

```text
H effect at O=1  = BA(multi_h)  - BA(binary)
O effect at H=1  = BA(multi_o)  - BA(binary)
H effect at O=31 = BA(multi_ho) - BA(multi_o)
O effect at H=31 = BA(multi_ho) - BA(multi_h)
interaction      = BA(multi_ho) - BA(multi_h) - BA(multi_o) + BA(binary)
```

These are written per architecture and seed to `factorial_effects.csv`.

## Required firing diagnostics

Classification accuracy alone is insufficient. Each evaluation also records hidden and output valid-period distributions:

```text
mean events / neuron / macro-step
fraction S = 0
fraction S = 1
fraction S > 1
fraction S >= 4
fraction hitting the configured cap
mean pre-reset membrane
max pre-reset membrane
post-end tail-event fraction
```

`fraction S > 1` indicates whether the multi-spike capacity is actually being used. A high `fraction_at_cap` indicates saturation rather than useful extra resolution.

## Binary sanity comparison

Because the binary control uses the new shared custom macro-LIF implementation, the finalizer also compares its test BA with the matching original Exp4.0 snnTorch-binary result when Exp4.0 `runs.csv` is present.

This comparison is written to:

```text
binary_sanity.csv
```

It is a sanity diagnostic, not a reused training result.

## Artifacts

```text
notebooks/artifacts/
  experiment_4_0_1_multispike_macro_lif/
    macro_lif_event_cap_factorial_v1/
      checkpoints/
      evaluations/
      runs.csv
      summary.csv
      factorial_effects.csv
      binary_sanity.csv       # when Exp4.0 reference artifacts are present
      provenance.json
```

## Interpretation

### `multi_h >> binary`

The one-event hidden/recurrent communication cap is a major bottleneck. Reinterpret the Exp4.2 RNN–RSNN gap using multi-hidden RSNN rather than binary RSNN.

### `multi_o >> binary`

Output event-count resolution is the dominant bottleneck. Subsequent work should focus on output coding/readout rather than adding recurrent capacity.

### `multi_ho >> both single-factor variants`

Hidden and output quantization interact; both constraints jointly cause the loss.

### Little improvement from all multi-spike variants

The binary event cap is not the main explanation for the ANN–SNN gap. Return to membrane/analog-head/surrogate-dynamics decomposition rather than increasing event multiplicity further.
