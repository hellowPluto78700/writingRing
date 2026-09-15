# Exp7.2.6.4 — Frozen-L2 output-interface decomposition

## Scientific goal

Exp7.2.6.4 keeps the Exp7.2.6.1 Mean-CE, bias-free `234x234` L1/L2 representation completely frozen and asks four output-level questions without end-to-end backbone retraining:

1. **Part A — output bandwidth:** is the poor beta=1 IF spike-count result mainly caused by finite firing bandwidth / residual membrane that cannot discharge inside the valid window?
2. **Part B — evidence semantics:** for the same good frozen `W_linear`, how much classification accuracy depends on negative/signed evidence, evidence magnitude, and per-timestep softmax voting?
3. **Part C — objective semantics:** on the same frozen L2, how do Raw-WCCE, softmax-vote WCCE, and TSCE train the `128 -> 12` projection matrix differently?
4. **Part D — CE scale vs LIF gradient geometry:** is the poorer `W_lif` mainly caused by bounded firing-rate CE scale, or by the LIF threshold/reset/surrogate-gradient training path?

The experiment boundary is strict:

```text
input -> frozen L1 -> frozen L2 -> [only W/readout is studied]
```

No Part A/B/C/D job updates L1 or L2.

## Fixed contract

- source: Exp7.2.6.1 Mean-CE, `bias=False`, `task_only`;
- architecture: `234x234 = (2,3,4) -> (2,3,4)`;
- seeds: `11, 23, 37`;
- L2 cache contains the full padded trajectory plus valid lengths;
- output alpha: `0`;
- native LIF beta: `0.5`;
- threshold inherited from Exp7.2.6;
- output cap: `1`;
- LIF input gain: `1`;
- head bias: `False`;
- trainable projection size in C/D: `128 * 12`;
- C/D reuse the Exp7.2.6.3 paired initialization and loader seeds, so C0 Raw-WCCE and D gain=1 are directly comparable to the established 7.2.6.3 paired-head baseline.

## Part A — output bandwidth / residual discharge

Uses the Exp7.2.6.1 source `W_linear` without training.

References:

```text
Analog
beta=1 IF charge = theta*N + U_T
beta=1 IF count
beta=.5 LIF count
```

### A1: zero-input output-only flush

After each sample's valid L2 window ends, set all later L2 input to zero and continue only the output neuron for

```text
K = 0, 1, 2, 4, 8, 16, 32, 64, 128
```

for beta=1 and beta=.5.

For beta=1 also compute the theoretical infinite-flush count ceiling

```text
N_inf = N_T + floor(max(U_T, 0) / theta)
```

and report positive/negative residual, extra spikes, discharge ratio, and fireable-discharge ratio.

### A2: padded-tail diagnostic

Compare:

- `zero_l2_tail`: valid L2 followed by forced zeros;
- `native_l2_tail`: the actual cached L2 activity in the padded region.

This is secondary because native tail mixes backbone residual activity with output residual discharge.

## Part B — evidence semantics, no training

Freeze both L2 and the same source `W_linear`.

For

```text
e_t = W_linear z_t
```

compare:

1. `raw_signed`: `sum_t e_t`;
2. `positive_only`: `sum_t max(e_t, 0)`;
3. `softmax_vote`: `sum_t softmax(e_t / tau)`, `tau in {0.25, 0.5, 1, 2}`;
4. `lif_beta05`: actual beta=.5 LIF spike count.

The softmax temperature sweep is reported in full. If a single temperature is highlighted, it is selected on validation BA only and then evaluated on test.

## Part C — objective semantics, frozen L2; only W trainable

Three paired `128 -> 12`, bias-free projection trainings start from the same W initialization and see the same train-loader order.

### C0 Raw-WCCE

```text
e_t = W z_t
loss = CE(mean_t(e_t), y)
```

### C1 Softmax-vote WCCE

```text
p_t = softmax(e_t)
loss = -log(mean_t p_t[y])
```

Different timesteps can compensate each other, but each timestep contributes normalized positive class evidence.

### C2 TSCE

```text
p_t = softmax(e_t)
loss = -mean_t log p_t[y]
```

This strongly penalizes timesteps that individually assign low probability to the final class.

Every learned W is cross-evaluated through the same three readouts:

```text
Analog
Softmax vote (tau=1)
LIF beta=.5 count
```

This distinguishes projection quality from readout compatibility.

## Part D — CE gain vs LIF gradient geometry

Freeze L2 and train only W.

Main paired sweep:

```text
head in {Linear, LIF beta=.5}
G_CE in {1, 2, 5, 10}
```

Linear:

```text
loss = CE(G_CE * mean_t(W z_t), y)
```

LIF:

```text
W z_t -> LIF -> S_t
loss = CE(G_CE * mean_t(S_t), y)
```

`G_CE` is applied only immediately before CE. It never changes the LIF input current or forward spike trajectory for a fixed W.

Every trained W is cross-evaluated with:

```text
Analog
Softmax vote (tau=1)
LIF beta=.5 count
```

The primary quantity is:

```text
BA_Analog(W_lif^(G))
```

Secondary LR control:

```text
LIF, G_CE=5, LR = base_LR / 5
```

This helps separate softmax/CE conditioning from a simple effective gradient-step-size change.

### Gradient diagnostic

For each seed and `G_CE in {1,2,5,10}`, use the exact same initial W and the same diagnostic minibatch to compute:

```text
||g_linear||
||g_lif||
||g_lif|| / ||g_linear||
cos(g_lif, g_linear)
per-class row cosine for all 12 output classes
```

## Multi-CPU execution

Submit the complete dependency graph with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_6_4_cpu.bash
```

Dependency graph:

```text
cache[3]
  |-- Part A inference[3] --------|
  |-- Part B inference[3] --------|
  |-- Part C training[9] ---------|
  |-- Part D training[27] --------|
  `-- gradient diagnostics[3] ----|
                                  |
                              finalizer
```

Part D contains 24 primary jobs plus 3 LIF `G=5, LR/5` controls. The Part D array is concurrency-limited so the cluster is not flooded with small CPU jobs.

The finalizer hard-fails if any required per-run JSON is missing.

## Aggregated outputs

The finalizer writes, among others:

```text
part_a_reference_summary.csv
part_a_flush_summary.csv
part_a_infinite_flush_summary.csv
part_a_padded_tail_summary.csv

part_b_evidence_semantics_summary.csv
part_b_selected_softmax_summary.csv
part_b_evidence_diagnostics_runs.csv

part_c_objective_training_summary.csv

part_d_gain_sweep_summary.csv
part_d_gradient_geometry_summary.csv

manifest.json
```

Per-run training histories for Parts C and D are retained under `part_c_histories/` and `part_d_histories/`, but the notebook does not display every run individually.

## Notebook contract

`notebooks/experiment_7_2_6_4_output_interface_decomposition.ipynb` is analysis-only.

It:

- reads finalized aggregate CSV/JSON outputs only;
- shows method-level mean/std comparisons across seeds;
- plots Part A flush BA vs K;
- compares Part B raw / positive-only / softmax-vote / LIF;
- shows the Part C objective x evaluation-readout matrix;
- plots Part D Analog BA of learned W vs CE gain;
- plots gradient cosine / norm ratio vs CE gain.

It does **not** train models, run inference, submit Slurm jobs, or show every individual training run.
