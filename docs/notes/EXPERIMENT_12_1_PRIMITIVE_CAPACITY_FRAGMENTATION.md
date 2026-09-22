# Experiment 12.1 — Primitive Capacity and Stroke Fragmentation

## Question

Exp12.1 isolates two coupled questions left open by Exp12.0:

1. Does widening the primitive projection from `K=16` toward `K=128`
   recover information lost by the 128-to-K bottleneck?
2. Does extra primitive capacity encode genuinely useful distinctions, or does it
   split different temporal phases of one real stroke into different latent
   primitive slots?

The experiment deliberately does **not** add an RSNN, fixed bin, fixed window,
or sliding window to primitive assignment.

## Fixed backbone and run matrix

The Exp7.3 A2 backbone is identical and frozen in every run:

```text
input -> L1 (2,3,4) -> L2 (2,3,4) -> z_t in R^128
```

Temporal state is one of:

```text
T0:  h_t = z_t
EMA: h_t = beta h_(t-1) + (1-beta) z_t, beta=15/16
```

The primitive capacity sweep is:

```text
K = 16, 32, 64, 128
seed = 11, 23, 37
```

Thus there are exactly:

```text
2 temporal modes x 4 K values x 3 seeds = 24 independent runs
```

Each run trains only the bias-free primitive projection `W_p` and a bias-free
linear gesture classifier:

```text
s_t = W_p h_t
score = sum_t W_y s_t
```

Validation Balanced Accuracy selects the checkpoint.

## Phase I: analog capacity diagnosis

Every selected analog checkpoint is evaluated with matched linear probes on:

- `R0`: direct L2 `z_t^128`;
- `R0a`: integrated `h_t^128`;
- `R0b`: analog primitive state `s_t^K`.

The primary capacity quantity is:

```text
Delta_projection(K) = BA(R0b_K) - BA(R0a)
```

The finalizer reports this by K and temporal mode.

The learned `W_p` is also analyzed with SVD:

- numerical rank;
- stable rank;
- effective rank;
- nonzero condition number;
- full singular-value spectrum in each per-run JSON.

## Phase II: fixed-checkpoint softmax decomposition

No additional primitive encoder is trained for these transforms. The same
analog checkpoint is reused post hoc.

WHAT:

```text
q_raw  = softmax(s_t)
q_norm = softmax(normalize(s_t))
```

HOW MUCH:

```text
m_h = RMS(h_t)
m_s = RMS(s_t)
```

The complete 2x2 representation set is:

```text
m_h q_raw
m_h q_norm
m_s q_raw
m_s q_norm
```

Hard-token controls replace q by one-hot(argmax(q)) while retaining the matched
magnitude. This measures whether the representation is becoming genuinely
discrete or still relies on a distributed mixture.

## Ground-truth stroke-boundary diagnosis

The writing-motion ablation pipeline already exports weak stroke annotations.
For each padded gesture, Exp12.1 maps the padded `output_segment_index` back to
the original `segment_index`, then reads:

```text
original_reference/annotations/<user>/action_<action>/
  <user>_action_<action>_writing_intervals.csv
```

These annotations provide:

- stroke index within the gesture;
- press/start segment-local timestep;
- lift/end segment-local timestep.

They do **not** provide semantic stroke identities. Therefore Exp12.1 never
assumes that two different strokes must use different latent primitive slots.

For each true stroke it reports:

- number of latent slots used;
- dominant-slot purity;
- winner switches per stroke;
- fraction of strokes with any fragmentation;
- sustained fragmentation, requiring both adjacent runs to last at least
  3 samples;
- winner run length;
- within-stroke JS change of the soft distribution;
- within-stroke transition matrices.

A predicted slot change exactly across two true strokes is not counted as
within-stroke fragmentation.

This directly tests the phase-fragmentation hypothesis instead of using global
winner-switch rate as a proxy.

## R4 sparse-event probe

R4 remains the Exp12.0 post-hoc same-primitive confidence-peak operator.

For every 2x2 soft representation, selection still uses the normalized
primitive-space winner margin and the same-primitive local peak rule. No
refractory, selection training, binning, or sliding window is introduced.

Quantiles are:

```text
0.2, 0.4, 0.6, 0.8, 0.9
```

Reported sparse diagnostics include:

- BA;
- events/sample;
- events/true-stroke;
- fraction of selected events inside annotated strokes;
- short-lag cross-primitive event-pair fraction at 16 and 32 timesteps.

## Multi-CPU execution

Per repository policy, one independent `(temporal mode, K, seed)` condition is
one Slurm array task using one CPU core. Training, checkpoint selection,
post-hoc probes, stroke diagnostics, and R4 analysis for that checkpoint all
finish inside the same task.

The array is:

```text
#SBATCH --array=0-23%24
#SBATCH --cpus-per-task=1
```

The finalizer is an `afterok` dependency and aggregates existing artifacts
only.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_12_1_cpu.bash
```

Concurrency can be lowered without editing the batch file:

```bash
SLURM_MAX_CONCURRENCY=12 \
  bash scripts/bash_script/SNN_Bash/submit_exp_12_1_cpu.bash
```

The submit wrapper rejects values above the repository-wide 50-task limit.

To inspect the exact task mapping without submitting:

```bash
python -m scripts.experiment_12_1_primitive_capacity_fragmentation list-runs
```

## Output

Artifacts are written under:

```text
notebooks/artifacts/experiment_12_1_primitive_capacity_fragmentation/
  primitive_capacity_fragmentation_v1/
```

Important finalized tables:

- `runs.csv`
- `capacity_fragmentation_summary.csv`
- `representation_runs.csv`
- `representation_summary.csv`
- `decomposition_summary.csv` (explicit softmax, magnitude, and hard-token deltas)
- `r4_runs.csv`
- `r4_summary.csv`
- `best_softmax_by_k.csv`
- `k_cls_selection.csv`
- `manifest.json`

Per-run transition matrices are saved as compressed NPZ files.

## Selection policy

`K_cls` is selected using validation BA only.

The experiment intentionally does not hard-code `K_sparse`, because the plan
defines it as the lowest-event / lowest-fragmentation K whose BA is "close to"
the best model but does not define a numerical BA tolerance. The finalized
tables expose the complete validation-BA / event-rate / fragmentation Pareto
surface so that threshold can be chosen explicitly rather than silently
invented in code.

## Scope

Exp12.1 is a capacity-and-fragmentation diagnosis. It does not introduce a
recurrent contextualizer or a same-stroke consistency training loss. Those are
appropriate follow-ups only if this experiment shows that increased K recovers
classification information while systematically increasing within-stroke
fragmentation.


## Required checks

```bash
python -m pytest -q tests/test_experiment_12_1_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
