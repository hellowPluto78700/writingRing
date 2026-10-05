# CoreBenchmark extension: spike-only TSCE output-beta sweep

This extension supplements CoreBenchmark v1 without modifying the canonical Core run matrix. It asks how output-neuron membrane retention changes end-to-end learning when the task loss, checkpoint selection, and native inference all use **positive output spikes only**.

## Scientific question

Core O1 applies sample-balanced TSCE directly to analog instantaneous evidence `e[t] = W_out z_L2[t]`; no output neuron is present. This extension keeps the same sample-balanced temporal reduction but replaces the classifier signal by the cap-1 output spike emitted by an explicit LIF output neuron:

```text
e[t]    = W_out z_L2[t]
Upre[t] = beta U[t-1] + e[t]
s[t]    = 1[Upre[t] >= theta]
U[t]    = Upre[t] - theta s[t]
```

The output membrane is signed and unclamped, `theta=0.5`, output spikes are positive-only and cap-1, and there is no output synaptic filter. `beta` is therefore a real part of the trainable end-to-end computation rather than a post-hoc readout setting.

The experiment has two paired modes.

### `valid`

Every valid sensory timestep is supervised using only the spike emitted at that timestep:

```text
q[t] = s[t]
L_valid = mean_i (1/T_i) sum_t CE(q_i[t], y_i)
```

Checkpoint selection and native inference use valid-window output spike count:

```text
C_valid = sum_t s[t]
prediction = argmax(C_valid)
```

Validation BA is the primary checkpoint criterion. Ties use lower `CE(C_valid / T, y)`, then the earliest epoch.

### `drain`

Valid sensory dynamics are identical. At each sample's final valid timestep the backbone stops and admits no new evidence. The output neuron then serializes positive endpoint backlog with the established `output_spike_drain` contract:

```text
beta_drain = 1
new evidence = 0
D = sum_k s_drain[k]
```

Drain iterations are **not new sensory timesteps**. Their total spike count is attached to the final valid TSCE timestep:

```text
q[t] = s[t]                 for t < T
q[T] = s[T] + D
L_drain = mean_i (1/T_i) sum_t CE(q_i[t], y_i)
```

Thus the denominator remains exactly `T_i`, matching the sample-balanced Core O1 temporal reduction. Training never uses analog endpoint residual `U[T]` as a classifier logit.

Checkpoint selection and native inference use fully-drained spike count:

```text
C_drain = sum_t s[t] + D
prediction = argmax(C_drain)
```

Validation drained BA is primary; ties use lower `CE(C_drain / T, y)`, then the earliest epoch.

## Formal matrix and pairing

- modes: `valid`, `drain`
- beta: `0.0, 0.1, ..., 1.0`
- seeds: `11, 23, 37`
- total: **66 end-to-end training runs**
- backbone: Core `30 -> 128 (2,3,4) -> 128 (2,3,4) -> 12`
- output threshold: Core threshold `0.5`
- output spike cap: one positive spike per class per timestep
- drain beta: `1.0`
- maximum drain serialization depth: `1024`

Core parameter initialization and DataLoader randomness are keyed by the Core protocol version, model seed, and parameter/loader role. Neither mode nor beta participates in those streams. Consequently every `(mode, beta)` run at a fixed seed starts from the same L1/L2/output-W initialization and sees the same shuffled training batches.

## What is and is not Core-compatible

The temporal reduction is intentionally Core-O1-like:

```text
one CE per valid timestep -> mean within sample -> mean across samples
```

The classifier variable is intentionally different:

```text
Core O1:          e[t] = W_out z[t]       (signed analog evidence)
this extension:   q[t] = output spikes    (positive, thresholded, cap-1)
```

Therefore this experiment is a **spike-interface analogue of Core TSCE**, not a bit-for-bit reproduction of O1. Even at `beta=1`, full positive drain does not make per-timestep spike TSCE algebraically equal to analog TSCE.

## Diagnostics

Every selected checkpoint reports train/validation/test metrics for:

- native spike-count BA for its own mode;
- valid-only spike-count BA;
- fully-drained spike-count BA;
- analog `sum_t Wz[t]` BA as a **diagnostic counterfactual only**;
- spike-TSCE objective value;
- native sequence mean-count CE;
- valid/drained prediction disagreement;
- native/analog prediction disagreement;
- valid and drain spike counts;
- drain spike fraction and required drain depth;
- positive and negative endpoint residual magnitude;
- L1/L2 firing-rate distributions and train-test gap.

Analog evidence is never used for training, checkpoint selection, or native prediction.

Per-run traces are retained for later representation analysis. Standard Core logistic-regression probes are intentionally **not a v1 completion requirement**; the causal question here is the beta-dependent spike-output interface, and excluding the heavy probe stage keeps the formal 66-run sweep focused. A later probe-only extension can reuse the saved traces without retraining.

## Completion and provenance contract

A formal run is complete only when all of the following exist and match the extension/core identity:

```text
checkpoint.pt
history.json
native.json
traces.npz
complete.json
```

`complete.json` is the completion gate. A valid matching completion is skipped. A valid matching checkpoint without completion skips training and continues extraction/evaluation. Missing or mismatched extension/core identities fail rather than silently reusing artifacts.

The finalizer requires all expected run completions and writes:

```text
aggregate/native_runs.csv
aggregate/native_summary.csv
aggregate/firing_runs.csv
aggregate/manifest.json
```

`native_summary.csv` groups test metrics by `(mode, beta)` across the three seeds.

## Slurm execution contract

Default result root:

```text
core_benchmark_v1/results/output_spike_tsce_beta_v1/
```

The formal DAG is:

```text
prepare
  -> compute-node preflight smoke
  -> 66-task train/evaluate array
  -> finalize
```

Each training array task performs the full atomic unit:

```text
train -> select checkpoint -> native/diagnostic evaluation -> save traces -> complete.json
```

There is no separate required postprocess stage. This is deliberate: all required diagnostics depend only on the selected run and are lightweight relative to training. No heavy cross-run work occurs before finalization.

Recommended array concurrency is **33**, below the repository-wide ceiling of 50. Each task requests one CPU core. The compute-node jobs source `scripts/bash_script/SNN_Bash/slurm_cpu_env.bash`, which initializes Unity Lmod/Conda without sourcing `/etc/profile` wholesale, activates `writingring-gpu` with `writingring-viz` fallback, and enforces one-core BLAS/OpenMP limits.

### Mandatory preflight

`smoke.bash` uses the exact formal environment bootstrap and the real prepared Core data lock. It checks Python/Torch imports, then exercises both modes at beta endpoints (`0` and `1`) on a real training batch, including forward, spike-TSCE loss, backward propagation to both hidden layers and output W, one optimizer step, and checkpoint save/load round trip. The formal array is submitted only after this job succeeds.

Submit:

```bash
bash core_benchmark_v1/extensions/output_spike_tsce_beta/slurm/submit.bash
```

Manual commands:

```bash
python -m core_benchmark_v1.extensions.output_spike_tsce_beta plan
python -m core_benchmark_v1.extensions.output_spike_tsce_beta prepare
python -m core_benchmark_v1.extensions.output_spike_tsce_beta smoke
python -m core_benchmark_v1.extensions.output_spike_tsce_beta run --task-id 0
python -m core_benchmark_v1.extensions.output_spike_tsce_beta finalize
```
