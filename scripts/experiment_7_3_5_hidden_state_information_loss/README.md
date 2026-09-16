# Exp7.3.5 — Hidden-state information loss across LIF quantization

## Scientific question

Exp7.3.4 showed that a strong Linear-trained evidence matrix can be deployed through the output LIF, while optimizing that matrix through the output-LIF surrogate path does not reliably improve generalization. That isolates an output-interface problem, but it does not answer whether hidden binary spikes themselves are already discarding useful information.

Exp7.3.5 asks:

```text
How much linearly accessible class and temporal information is lost when each
hidden LIF layer converts its internal analog state into binary spikes?
```

The experiment does **not** retrain the SNN. It freezes the selected Exp7.3 A2 backbone and replays its exact hidden dynamics while exposing four states per hidden layer:

```text
I_t      = alpha I_{t-1} + W x_t              syn_current
U_t^-    = beta U_{t-1} + I_t                 pre_reset
S_t      = binary threshold crossing           spike
U_t      = U_t^- - threshold * S_t            post_reset (secondary diagnostic)
```

Forward communication remains exactly the original Exp7.3 binary-spike path:

```text
input -> L1 spike -> L2 spike -> Linear A2 head
```

The internal analog states are diagnostic only.

## Locked source backbone

- source experiment: Exp7.3;
- source method: `A2_e2e_linear_wcce`;
- architecture: `234x234`;
- seeds: `11,23,37`;
- same user split, labels, 64 Hz event input, hidden weights, multi-tau synaptic dynamics, membrane dynamics, threshold, reset, and binary hidden event cap as the selected A2 checkpoint;
- L1/L2 are never optimized in Exp7.3.5;
- output head is not used by the probes.

Each extraction task replays the selected A2 checkpoint and verifies that its reconstructed L2 binary-spike features reproduce the existing Exp7.3 WCCE L2 cache within `1e-6` max absolute error. A mismatch aborts extraction rather than silently probing a different realization.

## Representations

For both `l1` and `l2`, the primary representations are:

```text
syn_current  = I_t
pre_reset    = U_t^-
spike        = S_t
```

`post_reset = U_t` is retained as a secondary diagnostic so reset-induced state loss can be measured without changing the primary question.

## Probe A — whole-sequence mean

For state trajectory `h_t` and valid length `T_valid`:

```math
h_bar = (1 / T_valid) * sum_{t < T_valid} h_t
```

The feature dimension is 128 for every state and layer.

This probe measures whole-gesture class information accessible without absolute temporal position.

## Probe B — ordered Fixed250 mean

At 64 Hz, 250 ms is 16 timesteps. Each state trajectory is partitioned into the same absolute ordered 250 ms bins used elsewhere in the repository. Within each bin only valid timesteps contribute:

```math
h_bar_b = mean_{t in valid bin b} h_t
```

Bins after the valid endpoint are zero. The ordered bin vectors are concatenated before the probe.

This probe measures linearly accessible temporal/phase information while keeping the same underlying frozen SNN trajectory.

## Probe protocol

Every `(seed, layer, state, aggregation)` condition uses the same diagnostic classifier protocol:

```text
train-only StandardScaler()
    -> LogisticRegression(solver="lbfgs", fit_intercept=True, max_iter=5000)
    -> C in {1e-3, 1e-2, 1e-1, 1, 10}
    -> choose highest validation balanced accuracy
    -> first C in ascending grid wins exact ties
    -> evaluate test only after C selection
```

There is no SNN training and no auxiliary loss. The classifier is only a probe of linearly accessible information.

## Primary contrasts

For each layer and aggregation:

### Membrane integration

```math
Delta_integration = BA(U^-) - BA(I)
```

Positive values mean membrane integration makes class information more linearly accessible than the synaptic-current state.

### Spike quantization loss

```math
Delta_quantization = BA(U^-) - BA(S)
```

This is the primary Exp7.3.5 result.

- large positive value: the hidden neuron contains useful analog membrane information that is lost when converted to binary spikes;
- value near zero: the binary hidden spike retains nearly all linearly accessible information measured by the probe;
- negative value: thresholding produces a more linearly accessible representation than the raw pre-reset membrane for this probe.

Secondary diagnostics are:

```math
BA(U^-) - BA(U_post)
BA(U_post) - BA(S)
```

They help separate threshold/reset effects but are not the primary claim.

## Interpretation matrix

### `whole_mean: U^- ~= S`, `fixed250: U^- ~= S`

Hidden binary communication is not the main bottleneck. The evidence should remain focused on the final output-LIF readout/credit-assignment interface.

### `whole_mean: U^- ~= S`, `fixed250: U^- > S`

Binary spikes retain class identity but lose temporal/phase information. This directly motivates a hidden-state temporal-supervision or richer temporal communication experiment.

### `whole_mean: U^- > S`, `fixed250: U^- > S`

Binary hidden quantization is already discarding both whole-gesture and phase-aware information. A membrane auxiliary loss, multi-bit/weighted spike communication, or hybrid spike+state design becomes justified.

The experiment is diagnostic only; it does **not** conclude in advance that L1/L2 should stop using spikes.

## Multi-CPU execution

Exp7.3.5 uses a two-stage Slurm pipeline so the expensive frozen-SNN replay is not repeated for every probe.

```text
Stage 1: 3 extraction tasks
  seed11 / seed23 / seed37
  each task:
    load selected A2 checkpoint
    -> replay train/val/test once
    -> collect I, U^-, S, U_post for L1/L2
    -> build whole-mean and Fixed250 ordered-mean features
    -> verify L2 spike replay against Exp7.3 cache
    -> write feature caches

Stage 2: 48 independent probe tasks
  3 seeds x 2 layers x 4 states x 2 aggregations
  each task:
    load one cached feature set
    -> train-only scaler
    -> validation-select C
    -> report train/val/test metrics

Stage 3: one afterok finalizer
  aggregate existing JSON/NPZ-derived probe results only
```

All compute tasks use one CPU core. Probe concurrency is capped at 48, below the repository default maximum of 50 simultaneous CPU tasks.

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_5_cpu.bash
```

## Finalized outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_7_3_5_hidden_state_information_loss/hidden_state_information_loss_v1/
```

Important finalized files:

- `manifest.json`
- `method_runs.csv`
- `method_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`
- `source_reproduction_checks.csv`
- per-seed `extractions/*.json`
- per-condition `feature_cache/*.npz`
- per-condition `evaluations/*.json`

## Notebook contract

`notebooks/experiment_7_3_5_hidden_state_information_loss.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts and presents whole-mean results, Fixed250 results, the primary information-loss contrasts, secondary reset diagnostics, and source-replay checks. It does not import Torch, invoke Slurm, regenerate feature caches, or fit probes.
