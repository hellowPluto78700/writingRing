# Experiment 7.1 — Long-tau anti-persistence regularization

## Scientific question

Exp7.0 showed that the legacy long-timescale layer (`shift_syn=6`, about 1 s at 64 Hz) can enter a sustained-firing regime: accumulated synaptic current repeatedly recharges the membrane and produces horizontal firing bands that can continue after the valid input interval. Exp7.1 asks whether this pathology can be repaired **without changing the legacy neuron equation** by applying a targeted regularizer to the long hidden layer only.

The hypothesis is:

```text
anti-persistence loss
    -> changes the effective long-layer input weights
    -> reduces pathological accumulated current
    -> shortens continuous spike runs
    -> preserves non-zero useful long-timescale activity
```

This experiment does not compare fusion architectures and does not compare legacy versus normalized synaptic updates.

## Fixed end-to-end backbone

```text
Raw64 30-channel spike train
    -> Short hidden layer: 128 neurons, shift_syn=4 (~242 ms)
    -> Middle hidden layer: 128 neurons, shift_syn=5 (~492 ms)
    -> Long hidden layer: 128 neurons, shift_syn=6 (~992 ms)
    -> 12 short-tau binary output LIF neurons
    -> valid-length WholeCount CE
```

All hidden layers use binary spikes (`cap=1`), `tau_mem=22.54 ms`, threshold `0.5`, and the legacy synaptic update

```math
I_t = \alpha I_{t-1} + W x_t.
```

The output layer reads **only long-layer spikes**. There is no short/middle skip connection, WHAT bypass, or context-gain bypass. This prevents a regularized model from trivially killing the long branch while classifying through an alternate path.

## Conditions

Three paired conditions are run for seeds `11, 23, 37`:

1. `no_reg`
   - WholeCount CE uses valid length.
   - No long-layer regularizer.

2. `all_loss_masked`
   - WholeCount CE uses valid length.
   - Long firing-rate regularizer uses valid timesteps only.
   - Long persistence windows are included only when the whole sliding window lies inside the valid interval.

3. `wholecount_only_masked`
   - WholeCount CE uses valid length.
   - Long firing-rate regularizer uses the full 256-step padded trajectory.
   - Long persistence loss uses every sliding window across the full 256-step trajectory, including windows that cross the valid-to-padding boundary.

Within a seed, all three conditions share the same model-initialization stream and minibatch-order stream.

## Objective

For the two regularized conditions:

```math
\mathcal L(e)
=
\mathcal L_{\rm WC}^{valid}
+
s(e)\left[
\lambda_r \mathcal L_{\rm rate}^{L}
+
\lambda_p\left(
\mathcal L_{3,2}^{L}
+0.5\mathcal L_{8,4}^{L}
\right)
\right]
```

with a 10-epoch linear warmup

```math
s(e)=\min(1,e/10).
```

The regularizer directly observes only the long-layer spike tensor `z^L [B,T,128]`, but gradients are allowed to propagate end-to-end through long, middle, and short layers.

### Long firing-rate term

```math
\mathcal L_{\rm rate}^{L}=E[z^L]
```

under the condition-specific mask policy.

### 3-step anti-persistence term

For each 3-step window, let

```math
c^{(3)}=z_t+z_{t+1}+z_{t+2}.
```

Then

```math
\mathcal L_{3,2}^{L}
=E[\mathrm{ReLU}(c^{(3)}-2)^2].
```

Only `111` is penalized; patterns such as `110` or `101` are allowed.

### 8-step high-duty-cycle term

For each 8-step window,

```math
c^{(8)}=\sum_{k=0}^{7}z_{t+k},
```

and

```math
\mathcal L_{8,4}^{L}
=E[\mathrm{ReLU}(c^{(8)}-4)^2].
```

This catches sustained high-duty-cycle patterns such as `11011011` even when no long exact `111111...` run exists.

## Gradient-strength calibration

Do not compare raw numerical loss scales. For each regularized condition and seed, five independent training batches are used before training to measure gradients with respect to the long input matrix only:

```math
G_{task}=\|\nabla_{W_L}\mathcal L_{WC}\|,
```

```math
G_{rate}=\|\nabla_{W_L}\mathcal L_{rate}\|,
```

```math
G_{persist}=\|\nabla_{W_L}(\mathcal L_{3,2}+0.5\mathcal L_{8,4})\|.
```

`lambda_r` and `lambda_p` are frozen for the run so that the median initial gradient ratios target:

```text
rate / task        = 2.5%
persistence / task = 5.0%
```

`all_loss_masked` and `wholecount_only_masked` are calibrated separately because their mask policies change regularizer gradient magnitude.

## Training

- seeds: `11, 23, 37`
- fixed user split seed: `12345`
- max epochs: `100`
- minimum training gate: `20` epochs
- patience after that gate: `30` epochs
- earliest possible early stop: epoch `50`
- checkpoint: highest validation balanced accuracy; tie-break lower validation WholeCount CE
- final test metrics are computed only from the selected best checkpoint

## Per-run artifacts

Each run writes its own unique files:

```text
checkpoints/
histories/
calibrations/
evaluations/
activities/
rasters/
training_curves/
current_traces/
```

The fixed validation sample uses the same deterministic median-valid-length rule across all runs. Rasters are saved for short, middle, long, and output layers. The long synaptic-current trace stores mean absolute current versus timestep.

Per-run diagnostics include:

- full / valid / padding long firing fraction
- long current RMS over full / valid / padding regions
- mean continuous run length
- mean maximum run length per neuron
- P95 maximum run length per neuron
- global maximum run length
- long weight Frobenius norm
- mean absolute long weight
- positive / negative long-weight magnitude statistics

Padding firing is a **diagnostic only**. There is no separate tail-specific loss.

## Final aggregation

The dependency finalizer creates method-level aggregate artifacts:

```text
summary.csv
dynamics_summary.csv
paired_condition_deltas.csv
calibration_summary.csv
manifest.json
summary_plots/
```

`runs.csv` is also retained as a machine-readable intermediate for debugging, but the analysis notebook must not consume or display individual runs.

The principal paired contrasts are:

```text
all_loss_masked - no_reg
wholecount_only_masked - no_reg
wholecount_only_masked - all_loss_masked
```

The last contrast directly tests whether constraining the **complete neuron trajectory** is more effective than regularizing only the annotated valid interval.

## Notebook policy

`notebooks/experiment_7_1_long_tau_antipersistence.ipynb` is analysis-only. It reads only:

```text
summary.csv
dynamics_summary.csv
paired_condition_deltas.csv
calibration_summary.csv
```

It does not train models and does not load individual histories, evaluations, rasters, checkpoints, or per-seed results.

## Multi-CPU execution

Submit on Unity with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_1_cpu.bash
```

This launches one CPU array with 9 independent workers (`3 conditions x 3 seeds`) and then an `afterok` aggregate finalizer. Every worker is single-threaded at the BLAS/OpenMP level to avoid oversubscription.
