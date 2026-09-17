# Exp8.1 — L1 hidden quantization and threshold-population ablation

## Scientific question

Exp7.3 A2 provides the locked local backbone/training reference:

```text
30 input channels -> L1 -> L2 -> Linear(12)
L1 shifts: (2,3,4)
L2 shifts: (2,3,4)
training: end-to-end Linear + WCCE
```

Exp7.3.5 showed that L1 pre-reset membrane retains substantially more Fixed250-decodable information than the communicated binary spikes. Exp8.1 asks whether that L1 communication bottleneck is better addressed by:

1. allowing one neuron to emit multiple weighted events per timestep, up to the hidden hardware capacity of 31; or
2. retaining binary communication but using a population with fixed heterogeneous thresholds.

The primary experiment changes **L1 only**. L2 is always a 128-neuron binary `(2,3,4)` layer. This keeps the location of any representation improvement interpretable.

## Locked Exp7.3 A2 training contract

All methods inherit the A2 choices:

- shifts `(2,3,4)` in both hidden stages;
- bias-free hidden and output linear matrices;
- task-only training, no activity regularizer;
- all L1, L2, and `W:128->12` parameters train end-to-end;
- Linear evidence readout;
- WCCE implemented as CE on the valid-length temporal mean evidence;
- seeds `11,23,37`;
- maximum 100 epochs, minimum 20 epochs, patience 30;
- checkpoint selection by native validation balanced accuracy, native objective loss as tiebreak;
- test evaluated only after checkpoint selection.

The output-neuron transfer diagnostic also remains the Exp7.3 convention: the same trained `W` is passed through a binary LIF with `beta=0.5`, repository threshold, and valid spike-count readout.

## Five methods

| Method | L1 width | L1 communication | L2 | Purpose |
| --- | ---: | --- | --- | --- |
| `binary128` | 128 | uniform-threshold binary | binary 128 | A2 reference |
| `weighted31_128` | 128 | integer multi-spike, cap 31 | binary 128 | amplitude/event-resolution ablation |
| `mt3_128` | 128 | binary, thresholds `0.5x/1.0x/1.5x` | binary 128 | threshold heterogeneity at original width |
| `binary384` | 384 | uniform-threshold binary | binary 128 | pure L1-width control |
| `mt3_384` | 384 | binary, thresholds `0.5x/1.0x/1.5x` | binary 128 | 3x128 threshold-bank population |

The two 384-neuron cases are both required. `mt3_384 - mt3_128` measures the capacity expansion, while `mt3_384 - binary384` isolates threshold heterogeneity at matched width.

## Weighted / multi-spike L1

The weighted condition reuses the Exp4.0.1 hidden event-cap neuron:

```math
U_t^- = \beta U_{t-1} + I_t
```

```math
n_t = \mathrm{clip}\left(\left\lfloor\frac{\max(U_t^-,0)}{\theta}\right\rfloor,0,31\right)
```

```math
U_t = U_t^- - n_t\theta.
```

Thus a hidden neuron communicates one integer value `n_t in {0,...,31}` to L2. Cap 31 is fixed because it is the target hidden-neuron hardware capacity, not a software hyperparameter sweep.

## Fixed heterogeneous-threshold L1

The multi-threshold conditions are **not adaptive-threshold neurons**. Every neuron receives a fixed threshold at initialization and that threshold never changes:

```math
\theta_i \in \{0.5\theta_0,\theta_0,1.5\theta_0\}.
```

Each neuron remains binary:

```math
s_{i,t}=\mathbb{1}[U^-_{i,t}\ge\theta_i],
```

with threshold-specific subtractive reset:

```math
U_{i,t}=U^-_{i,t}-\theta_i s_{i,t}.
```

Threshold heterogeneity therefore changes neuronal excitability and population coding, but never gives one neuron an analog or integer-valued output.

### MT128 layout

The 128 L1 neurons are split as evenly as possible into three threshold banks (`43/43/42`). Within **each** threshold bank, neurons are again distributed over shifts `(2,3,4)`. This prevents threshold assignment from becoming a proxy for tau assignment.

### MT384 layout

The 384-neuron L1 contains three complete 128-neuron banks:

```text
128 neurons at 0.5x threshold, each bank covering shifts (2,3,4)
128 neurons at 1.0x threshold, each bank covering shifts (2,3,4)
128 neurons at 1.5x threshold, each bank covering shifts (2,3,4)
```

L2 remains 128-neuron binary `(2,3,4)` in every method.

## Representation diagnostics

Final classification accuracy alone cannot localize the mechanism. Every selected checkpoint therefore exposes, for both L1 and L2:

- `pre_reset`: continuous membrane immediately before threshold/reset;
- `communication`: the value actually sent to the next layer (binary or weighted event count).

For each state, train-only linear probes evaluate:

- valid whole-sequence mean;
- ordered 250 ms mean bins.

The main quantization diagnostic is:

```math
\Delta_{quant}=BA(U^-)-BA(communication).
```

A smaller L1 gap together with higher L2/final BA supports the interpretation that the communication code preserved information that binary A2 discarded.

## Activity diagnostics

For every L1 `(threshold,tau)` group and L2 tau group, Exp8.1 records on the test split:

- mean communicated value per neuron per valid timestep;
- fraction nonzero;
- fraction greater than one;
- fraction at configured cap.

Only `weighted31_128` can have communication values above one. The multi-threshold methods remain binary and are differentiated by the fixed threshold group metadata.

## Paired contrasts

The finalizer reports paired per-seed deltas for:

```text
weighted31_128 - binary128       amplitude/event resolution
mt3_128        - binary128       heterogeneity at A2 width
mt3_128        - weighted31_128  population coding vs event multiplicity
binary384      - binary128       pure width effect
mt3_384        - mt3_128         MT capacity effect
mt3_384        - binary384       heterogeneity at matched width
```

## Run matrix and multi-CPU execution

```text
5 methods x 3 seeds = 15 independent runs
        -> Slurm array 0-14%15
        -> one CPU core per task
        -> train -> select best checkpoint -> evaluate/probe -> save artifacts
all 15 runs succeed
        -> afterok finalizer
        -> analysis-only notebook
```

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_8_1_cpu.bash
```

The finalizer only aggregates existing per-run artifacts. It never retrains missing runs.

## Durable artifacts

```text
notebooks/artifacts/
  experiment_8_1_hidden_quantization_ablation/
    hidden_quantization_ablation_v1/
      checkpoints/
      evaluations/
      histories/
      method_runs.csv
      method_summary.csv
      contrast_runs.csv
      contrast_summary.csv
      activity_runs.csv
      activity_summary.csv
      manifest.json
```

`notebooks/experiment_8_1_hidden_quantization_ablation.ipynb` is analysis-only. It reads finalized CSV/JSON files and compares method accuracy, representation probes, quantization gaps, and activity. It never trains models or launches Slurm jobs.
