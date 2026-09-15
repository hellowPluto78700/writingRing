# Exp7.2.6 — Output Readout Loss & Representation Shaping Decomposition

Exp7.2.6 is a mechanism follow-up to Exp7.2.5. It fixes the SNN backbone to `234x234` only:

- L1 synaptic shifts `(2,3,4)`;
- L2 synaptic shifts `(2,3,4)`;
- hidden width `128`;
- binary hidden spikes;
- seeds `11,23,37`;
- regularization conditions `task_only` and `task_plus_reg` are reported separately;
- the task objective is fixed to `whole_count_ce`.

The experiment separates two questions:

1. **Frozen-L2 decoding bottleneck:** if class information is already present in L2, where is it lost by the output spiking interface?
2. **End-to-end representation-shaping bottleneck:** does the output neuron also make the learned L2 representation worse during training?

## Prerequisite

Exp7.2.6 reuses the matched Exp7.2.5 C2 checkpoint for each `(regularization, seed)`:

```text
architecture = 234x234
objective = whole_count_ce
output_alpha = 0.0
output_beta = 0.5
```

These six Exp7.2.5 checkpoints must exist before Exp7.2.6 starts. The reused condition is the C2 LIF anchor for both the frozen-L2 source and the paired E2E comparison.

## Part I — Frozen-L2 output mechanism

Each C2 checkpoint is forwarded once and its complete train/val/test L2 binary trajectory is cached as compressed `uint8`. There are `2 regularizers x 3 seeds = 6` cache tasks.

### Paired frozen heads

For every frozen L2 source, Exp7.2.6 trains two fresh parameter-matched heads:

- `analog`: `Linear(128,12,bias=False)` followed by valid-length leakless accumulation;
- `lif`: the same `Linear(128,12,bias=False)` followed by the repository `MacroMultiSpikeLIF` with `alpha_out=0`, `beta=0.5`, threshold unchanged, `cap=1`, and valid-length output spike count.

The two heads share the exact same linear-weight initialization and loader-order seeds. Head mode is excluded from all pairing seeds. Both use WholeCount CE and checkpoint selection by validation valid-length WholeCount balanced accuracy with validation loss as tie-break.

There are `6 sources x 2 heads = 12` frozen-head fits.

### 2x2 cross-check

For each source the mechanism evaluator reports:

| case | projection | temporal readout |
|---|---|---|
| X1 | matched `W_lin` | analog accumulator |
| X2 | calibrated `g* W_lin` | LIF `beta=.5`, cap1 |
| X3 | matched `W_lif` | analog accumulator |
| X4 | `W_lif` | native LIF `beta=.5`, cap1 |

The main projection contrast is `X1 - X3`. The main dynamics contrasts are `X1 - X2` and `X3 - X4`.

### Validation-only global gain

`W_lin` is not trained against a spike threshold, so spiking evaluations allow one scalar gain. The gain is selected **only on validation data** under:

```text
alpha_out = 0
beta = 1
cap = 1
readout = pure output spike count
```

A coarse power-of-two sweep is followed by a local fine sweep. Once selected, the same `g*` is frozen for every beta, cap, polarity, and test evaluation. The test split never participates in calibration.

### Direction B — Linear to LIF mechanism ladder

The primary ladder is:

```text
B0  Wlin analog
B1  beta=1 IF, charge-preserving readout theta*N + U_T
B2  beta=1 IF, pure spike-count readout N
B2a beta=1 IF, cap31
B2b beta=1 bipolar IF, N+ - N-
B3  beta=.5 LIF, cap1
B3a beta=.5 LIF, cap31
B3b beta=.5 bipolar LIF
```

`B0` and `B1` must produce identical predictions because `theta*N + U_T = g* sum_t Wlin z_t` for beta=1 under the repository subtractive-reset neuron. This is an implementation sanity check, not a performance claim.

`B1 - B2` is called the **spike-interface penalty**, not merely quantization loss, because it can contain final-residual discard, threshold discretization, unsigned communication, and cap effects.

### Beta sweep

With `Wlin`, `g*`, threshold, cap1, unipolar output, and pure spike-count readout fixed, Exp7.2.6 evaluates:

```text
beta = 1.0, .95, .9, .8, .7, .6, .5
```

The endpoint difference `BA(beta=1) - BA(beta=.5)` is the clean membrane-leakage penalty.

### Direction A — Reverse validation with Wlif

A smaller reverse decomposition uses the same frozen `W_lif`:

```text
A0 Wlif + beta=.5 LIF spike count
A1 Wlif + beta=1 IF spike count
A2 Wlif + beta=1 charge-preserving readout
A3 Wlif + direct analog accumulation
```

`A2` and `A3` must match. This independently checks the conclusions obtained with `Wlin`.

### Native E2E control

The original Exp7.2.5 C2 output matrix is also evaluated with analog accumulation and its native beta=.5 LIF readout. It is a secondary control and is not mixed into the matched-head causal decomposition.

### Evidence accounting

For the soft subtractive reset

```text
U_t = beta U_{t-1} + e_t - theta S_t
```

the evaluator verifies sample-wise and class-wise:

```text
sum e_t = theta sum S_t + (1-beta) sum_{t<T} U_t + U_T
```

Failure analysis focuses on samples where the analog `Wlin` classifier is correct but beta=.5 LIF is wrong. For a fixed strongest LIF competitor `k*`, it stores correct-vs-competitor margins for input evidence, emitted spike charge, leakage, and final residual.

A representative trajectory is selected automatically from the validation split (never hand-picked from test) and saved under `representative_trajectories/`.

## Part II — End-to-end representation shaping

Three paired conditions are compared:

- `c0_analog`: L1 -> L2 -> bias-free linear output -> valid-length analog sum;
- `c1_if_beta10`: same network, then beta=1 cap1 output IF -> valid-length spike count;
- `c2_lif_beta05_reuse`: reused Exp7.2.5 alpha=0 beta=.5 cap1 LIF checkpoint.

Only C0 and C1 are newly trained (`2 conditions x 2 regularizers x 3 seeds = 12` runs). C2 is not retrained.

### Exact pairing with Exp7.2.5 C2

C0 and C1 deliberately reuse the exact Exp7.2.5 C2 model-initialization and loader-order seed stream. For `task_plus_reg`, they also reuse the regularizer coefficients stored in the matched C2 checkpoint. Therefore output dynamics, rather than initialization, loader order, or regularizer calibration, is the intended systematic difference.

Training uses max 100 epochs, minimum 20 epochs, patience 30, and validation valid-length WholeCount BA checkpoint selection with validation loss tie-break.

## Post-hoc L2 probes

Every C0/C1/C2 checkpoint is frozen and evaluated with three probes:

1. `l2_wholecount_matched_biasfree`: raw valid L2 whole-count -> fresh `Linear(128,12,bias=False)` with no scaler;
2. `l2_wholecount_linear`: the Exp7.2.5 train-only StandardScaler + validation-selected LogisticRegression protocol;
3. `l2_fixed250_linear`: ordered Fixed250 L2 counts with the same standardized probe protocol.

Probe randomness is paired across C0/C1/C2.

The main representation-shaping contrasts are computed per seed and per probe:

```text
Analog - IF
IF - LIF
Analog - LIF
```

WholeCount degradation indicates loss of class-accessible local evidence. Fixed250 degradation additionally tests whether ordered temporal/phase structure is damaged.

## Multi-CPU execution

Submit the full experiment with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_6_cpu.bash
```

The pipeline is intentionally serialized by dependency while each stage is parallelized across CPU tasks:

1. 6-task frozen-L2 cache array;
2. 12-task paired frozen-head training array;
3. 6-task frozen mechanism-evaluation array;
4. 12-task new paired E2E training array;
5. 18-task post-hoc probe array including reused C2;
6. one `afterok` finalizer.

Every array task requests one CPU core and the stage sizes stay below the repository-wide 50-task concurrency convention. The finalizer only aggregates existing artifacts; missing expected runs are an error.

## Finalized outputs

Important finalized files are:

```text
source_manifest.csv
crosscheck_2x2_runs.csv
crosscheck_2x2_summary.csv
mechanism_ladder_runs.csv
mechanism_ladder_summary.csv
beta_sweep_runs.csv
beta_sweep_summary.csv
cap_polarity_controls.csv
firing_diagnostics.csv
margin_decomposition.csv
e2e_performance_runs.csv
e2e_performance_summary.csv
e2e_l2_probe_runs.csv
e2e_l2_probe_summary.csv
representation_shaping_delta.csv
representation_shaping_delta_summary.csv
manifest.json
```

Presentation reports are split by regularization; `task_only` and `task_plus_reg` are never averaged together.

## Notebook contract

`notebooks/experiment_7_2_6_output_readout_loss_shaping.ipynb` is analysis-only. It reads finalized CSV/JSON outputs and shows aggregate method comparisons. It must not train models, load checkpoints, run inference, recalibrate gains, or dispatch Slurm jobs. Per-run artifacts remain on disk for reproducibility but are not dumped into the notebook.
