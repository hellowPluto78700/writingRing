# Core benchmark v1

A standardized measurement framework, separate from the historical Exp7.x-13.x ablations. This directory contains its own immutable protocol, model/training implementation, probes, readout controls, Slurm submission graph, and analysis-only notebooks. Existing experiments, data and checkpoints are not modified or silently reused.

## What runs by default

| Block | Canonical cases | Independent backbone runs |
| --- | --- | ---: |
| `01_objective` | O0 WCCE; O1 TSCE; O2 WCCE + 0.1 L1 WCCE; O3 WCCE + 0.1 L1 TSCE | 12 |
| `02_tau` | T1 123/234; T2 123/123; T3 123/345; T4 234/345 | 12 |
| `03_depth` | D1: 234/234/234, end-to-end | 3 |
| `04_readout` | R_IF beta=1; R_LIF beta=0.5 on O0; R_LIF_E2E full-network beta=0.5 control | 9 readout tasks |

There are **27 primary backbone training runs plus 9 readout tasks**. Six O0-derived readout tasks include same-W evaluation, validation-only threshold calibration, and (by default) one output-W-only training run. Three additional `R_LIF_E2E` tasks train the complete two-layer network through a beta=0.5 LIF output from the same paired initialization as O0. All use seeds **11, 23, 37**. `REF`, `T0`, `D0`, `R0`, and optional `M0` are aliases of the **same O0 checkpoint**, not additional runs.

O0 is the fixed anchor: 30 event channels -> 128 L1 (234) -> 128 L2 (234) -> bias-free accumulator. Objectives, tau configurations and depth are compared against this anchor independently. No winner from one block is used to select the next block. A later selected-factor composition study is intentionally outside this benchmark.

`depth_controls=true` adds D2 (freeze trained O0 L1/L2; train new L3 and head) and D3 (start from D2, unfreeze all), adding 6 backbone runs. `membrane_sweep=true` adds M1/M2/M3 (L1 membrane tau 54/117/242 ms), adding 9. Both are **off by default**. D2/D3 have additional optimization history and are secondary mechanistic controls, not matched-budget primary depth comparisons.

## Locked data and model protocol

`00_protocol/default.json` is the complete configuration. `prepare` materializes `protocol.lock.json` containing the configuration, concrete user assignment, sample manifest, data hashes, source hashes, package versions, aliases, and deduplicated run manifest. Workers refuse mismatched identities, changed source code, modified cached data, or different numerical package versions. Use a **new results directory** after changing the protocol or benchmark Python sources.

The default data are D0, **action0 + action1**, from the existing 64 Hz polarity-split wavelet pipeline:

```text
outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded
```

The producer has 36 channels; the model consumes **only the first 30 unsigned event channels**, not the six appended IMU channels. The 12 labels are `A B C D E G H I J K L X`. The horizon is 256 padded steps, but all objectives, scores and probes use only valid timesteps. Padding is not a post-gesture evidence-accumulation period.

The default split deliberately reuses the historical split-generation convention: shuffle lexicographically sorted users once with split seed `12345`, then assign floor(70%) train, floor(15%) validation, remainder test. For 20 users this is **14/3/3 users**. It is not a new user split for each model seed. Explicit nonoverlapping `train_users`, `val_users`, `test_users` can instead be provided before prepare; they must partition the complete cohort. Every split must contain all 12 classes. The concrete lists and sample IDs are persisted, not inferred from the training seed.

Hidden neuron dynamics are:

```text
I[t] = alpha * I[t-1] + W * x[t]
V_pre[t] = beta * V[t-1] + I[t]
s[t] = 1[V_pre[t] >= threshold]
V[t] = V_pre[t] - threshold * s[t]
```

**There is no `(1-alpha)` multiplier on the input drive.** Alpha is `1 - 2**(-shift)` and is fixed, not learned. Width-128 groups distribute neurons as 43/43/42. Hidden membrane tau is **22.54 ms**, threshold 0.5, cap one spike per timestep, normalized fast-sigmoid surrogate slope 25. Membrane voltage remains signed; there is no zero clamp. Forward and backward behavior are regression-tested against the repository's existing binary `MacroMultiSpikeLIF`. The optional M0 alias denotes this exact 22.54 ms anchor, not an independently rounded 22 ms run.

Longer synaptic tau changes gain and firing activity under this unnormalized equation. Accordingly, firing activity, silent-neuron fraction and pre-reset magnitude accompany the representation results. Tau allocation alone is not evidence of abstraction.

## Objectives and accumulator geometry

For hidden spikes `z[t]`, the deployed native readout is:

```text
e[t] = W_out * z[t]
A[t] = A[t-1] + e[t]
prediction = argmax(A[T_valid])
```

It has no bias, decay, threshold or reset. To preserve the historical A2 training scale, **WCCE trains `CE(A[T_valid] / T_valid, y)`**. Summed and mean logits have the same argmax for this no-bias head, but their cross-entropy losses and gradients are not identical. The normalization is a training-loss convention, not a normalization of synaptic current or of the deployed accumulator.

TSCE is the agreed sample-balanced loss: average CE over valid timesteps **within each sample**, then average samples. Historical Exp7.3 flattened all valid timesteps across the batch, weighting longer samples more. Therefore O1 is a standardized TSCE control, **not a claim of bit-for-bit reproduction of old A1**.

O2/O3 add a separate L1 auxiliary head at weight 0.1; it is used only in training. The final classifier never fuses L1 logits or bypasses L2 through this auxiliary head.

All backbone cases use Adam (learning rate 0.001, weight decay 0), batch 128, at most 100 epochs, minimum 20 epochs and patience 30. The checkpoint rule is common: **validation native accumulator BA**, then **validation mean-logit CE**, then earliest epoch. Epoch zero is a candidate in every case, including staged controls. Probes and test metrics never select a training checkpoint. Each parameter name has its own deterministic initialization stream, so adding an auxiliary head or L3 cannot shift the initial weights of shared layers. DataLoader randomness is paired across compatible cases as well.

## Unified probes

Every selected backbone checkpoint is evaluated at every layer with:

```text
state:        spike (primary), pre_reset (secondary)
aggregation:  whole_count
              fixed250_ordered / fixed250_shuffled
              relative10_ordered / relative10_shuffled
decoder:      true no_bias / affine
```

Both decoders use `StandardScaler(with_mean=False, with_std=True)` fitted on **train only**. Only `fit_intercept` changes. A centered no-bias model is forbidden because it introduces an effective intercept. Logistic regression uses the same C grid `[0.001, 0.01, 0.1, 1, 10, 100]` for every representation, selected by validation BA; ties prefer smaller C. Nonconverged candidates are recorded and excluded; if no candidate converges, the run fails. The fitted decoder is not refit on train+validation before testing. Coefficients, scales, intercepts, validation search records, predictions, sample IDs and labels are saved.

Fixed250 is 16 steps per bin and 16 bins, hence 2048 dimensions at width 128. **The shuffled control independently permutes complete valid bins within each sample, holding the final partial bin and all padding bins in their original positions.** This prevents changed exposure/duration cues from masquerading as changed ordering. The partial bin therefore retains some temporal anchoring; the measured contrast is deliberately limited to order among full bins. Samples with fewer than two full bins have no shuffleable order; eligibility is reported.

Shuffle seeds are `101, 211, 307, 401, 503`. Permutations depend on sample ID and shuffle seed, not model seed, label, layer or state. A separate decoder is fit on each shuffled train/validation/test representation; this is **not** only test-time corruption. Shuffles are averaged within each model seed before seed-level means/SD. A global permutation shared across all samples would merely relabel classifier columns and is not used.

Relative10 requires final valid length. It is an **offline, duration-aware phase-normalized reference**, not a streaming deployment method and not a guaranteed upper bound. Relative10 shuffles permute all ten normalized bins.

Reported contrasts:

```text
G_resolved = BA(Fixed250 ordered) - BA(WholeCount)
G_order    = BA(Fixed250 ordered) - BA(Fixed250 shuffled)
G_geometry = BA(affine) - BA(no_bias)
```

These are changes in decoder-accessible classification under controlled representations, not direct measurements of mutual information, pure classifier capacity, or high-level abstraction. Train/validation/test BA, accuracy, macro-F1, and native/probe train-test gaps are saved separately.

## Accumulator-to-spike controls

`04_readout` always starts from the new O0 checkpoint and its exact cached L2 spikes:

- `R0_accumulator`: original W, analog accumulated evidence.
- `R1_sameW_fixed`: original W, threshold 0.5, IF beta=1 or LIF beta=0.5.
- `R1_sameW_calibrated`: original W; threshold chosen using validation BA from `[0.125, 0.25, 0.5, 1, 2]`, preferring the native/nearest threshold in ties.
- `R2_adaptW`: same frozen hidden spikes; initialize W from O0, train **only W** using CE of valid output spike counts. The calibrated threshold stays fixed. Selection uses validation spike-count BA then spike-count CE, with epoch zero included.
- `R3_e2e_LIF`: independent full-network control with the same 234->234 architecture, paired parameter initialization, loader randomness, optimizer, WCCE valid-mean reduction, training budget, beta=0.5, threshold=0.5, alpha_out=0 and bias-free W as O0 except that the final accumulator is replaced by the output LIF during both training and inference. L1, L2 and W are all trainable. Checkpoint selection uses validation LIF BA, then validation valid-mean output-spike CE, then earliest epoch.
- `R4_e2e_accumulator_swap`: no retraining. Evaluate the selected R3 checkpoint using the exact same hidden layers and W, but bypass the output LIF and accumulate the analog `Wz[t]` evidence. The paired R4-R3 contrast isolates the residual spike-realization loss after the network has already adapted end-to-end to the LIF interface.

Output spike dynamics have **alpha_out=0**, so evidence is injected directly into the membrane; there is no additional output synaptic filter. Both use immediate subtractive reset, signed unclamped membrane, and binary cap one. All-zero and tied outputs are reported. Ties use the first class index consistently. No hidden SNN gradient exists in R2 because training uses cached hidden spikes; R3 explicitly supplies that missing full-network gradient path. A remaining R4-R3 gap is therefore the cleanest controlled estimate here of output-LIF realization loss after E2E adaptation, not a fundamental impossibility result.

## New temporal diagnostics

By default, O0 and D1 additionally produce `06_temporal_diagnostics` artifacts in their own atomic tasks. Lag similarity uses **all valid timestep pairs** at each requested lag, for all samples and layers, not just t=0. Correlation/cosine exclude undefined constant/zero-vector comparisons and report coverage.

History controls compare full replay with a reset of synaptic **and** membrane state immediately before the final H valid timesteps, for each layer separately and all layers together. The input suffix remains identical. Both the full accumulator and suffix-only evidence are evaluated against matched unmodified references on the same eligible cohort. Samples no longer than H are excluded and cohort/class coverage is reported. The H grid is 50/100/250/500/1000 ms rounded to actual 64 Hz timesteps. This is an endpoint common-suffix diagnostic, not a sliding-window study or a full reimplementation of every old Exp13 experiment.

## Running locally and on Slurm

Use **Python 3.11** in `writingring-gpu`, falling back to `writingring-viz` when the former is unavailable. Existing repository dependencies plus scikit-learn are needed for workers; notebook execution additionally needs nbformat, nbclient/nbconvert, and ipykernel.

From the repository root:

```bash
# Inspect the matrix without loading the dataset or training anything.
python -m core_benchmark_v1 plan

# Inspect submissions without activating Conda or submitting jobs.
DRY_RUN=1 bash core_benchmark_v1/slurm/submit.bash

# Submit the default benchmark, one core per task.
SLURM_MAX_CONCURRENCY=50 bash core_benchmark_v1/slurm/submit.bash
```

Submission uses one preparation job, a **30-task phase-1 array** containing all independent Objective/Tau/Depth cases plus the three independent R_LIF_E2E runs, a **6-task phase-2 O0-derived readout array**, and an `afterok` finalizer. Optional D2 joins phase 2; optional D3 uses phase 3; optional membrane runs join phase 1. Phase barriers are conservative checkpoint dependencies and keep total simultaneous benchmark CPU tasks within the requested cap (never above 50). They are not scientific winner-selection stages.

Each array task performs `train -> selected checkpoint -> native evaluation -> probes -> optional diagnostics -> atomic completion marker`. Evaluation functions are separate and reusable. The finalizer **only** consumes completed artifacts; missing runs, mismatched metadata, incomplete probe coordinates and altered artifact hashes fail closed. A rerun can reuse a completed training checkpoint and redo evaluation without retraining. Interrupted unfinished training restarts that run; there is no unvalidated optimizer-resume path.

Every compute job initializes Conda locally through `slurm/common.bash`, then limits OpenMP/MKL/OpenBLAS/NumExpr and PyTorch to one thread. DataLoaders have zero subprocess workers. No submit-shell Python executable path is exported. Slurm exports complete environment values through `--export=ALL`, never comma-containing lists in `--export=...`. Worker defaults are 12 GB and 8 hours **including evaluation**; adjust Slurm resources after profiling real data. No claim is made that the synthetic smoke runtime predicts production runtime.

For a separate configuration/result root:

```bash
CORE_CONFIG=/absolute/path/to/my_protocol.json \
CORE_RESULTS=/absolute/path/to/my_core_benchmark \
SLURM_MAX_CONCURRENCY=20 \
bash core_benchmark_v1/slurm/submit.bash
```

Manual stages / recovery:

```bash
python -m core_benchmark_v1 --results core_benchmark_v1/results/main prepare
python -m core_benchmark_v1 --results core_benchmark_v1/results/main run --run-key O0__seed11
python -m core_benchmark_v1 --results core_benchmark_v1/results/main evaluate --run-key O0__seed11
python -m core_benchmark_v1 --results core_benchmark_v1/results/main finalize
```

`evaluate` refuses to create a missing training checkpoint. `run` skips a fully validated complete run; `run --reevaluate` reuses a completed checkpoint but rewrites its evaluation. Re-evaluation invalidates the old aggregate PASS manifest; rerun the finalizer before viewing results. Per-run locks prevent concurrent writers.

## Results and notebooks

Default result root: `core_benchmark_v1/results/main/` (git-ignored).

```text
protocol.lock.json              # actual users, sample provenance, source/config identity
dataset.npz                     # identical immutable input cache for all workers
runs/<case>__seed<seed>/
  initial.pt / checkpoint.pt    # backbone initialization / selected weights
  history.json                 # optimizer history and validation selection
  traces.npz                   # valid-only spike, pre-reset and evidence trajectories
  native.json                  # aggregate/per-user metrics and activity diagnostics
  probes.json / probe_search.json
  probe_decoders.npz / probe_predictions.npz
  diagnostics.json
  provenance.json / complete.json
aggregate/
  manifest.json                # PASS only after strict full coverage verification
  native_* / probe_* / paired_* / readout_*.csv
  sample_manifest.csv / sample_counts.csv
  activity.csv / native_per_user.csv
  lag_similarity.csv / history_reset.csv
  01_objective/ ... 04_readout/ # per-block views, canonical REF preserved
```

Readout runs instead contain `head.pt`, `readout.json`, and `predictions.npz`, plus history/provenance/completion. The aggregate report does not count duplicated R0 readout views as independent models.

Open **`overview.ipynb`** for the integrated results, then each block's **`analysis.ipynb`**:

```text
01_objective/analysis.ipynb
02_tau/analysis.ipynb
03_depth/analysis.ipynb
04_readout/analysis.ipynb
05_probes/analysis.ipynb
06_temporal_diagnostics/analysis.ipynb
07_membrane/analysis.ipynb       # optional, empty when disabled
```

Notebooks are analysis-only and read finalized CSV/JSON files. They do not train, refit probes, submit jobs, or regenerate missing results. Set `CORE_BENCHMARK_RESULTS` before starting the kernel to read another result root. No finalized manifest means no reported performance; synthetic results are prominently labeled as execution checks. Mean/SD summarize **three optimization seeds on one split**, not broad cross-user-split uncertainty. Further robustness claims require additional independently locked splits, not reinterpreting these seeds as splits.

## Verification

```bash
python -m pytest -q tests/test_repository_source_syntax.py tests/test_core_benchmark_v1_contract.py

# Optional explicit synthetic end-to-end run; never mix with production results.
python -m core_benchmark_v1 --results /tmp/core-benchmark-smoke smoke
```

The focused suite checks case/dependency mapping, current and reset equations, reference-neuron forward/backward parity, paired initialization, loss/padding support, sample-wise shuffling, true no-bias preprocessing, cohort identity, frozen depth controls, readout-only adaptation, immutable checkpoint evaluation, strict aggregation, Slurm dry-run contracts, and execution of all eight notebooks on finalized synthetic artifacts. Synthetic BA values are not research results. Running these checks does not submit production Slurm jobs.

## Optional extensions

Optional studies that reuse the locked CoreBenchmark data/model/probe contract without changing the v1.1 run matrix live under `core_benchmark_v1/extensions/`.

- `extensions/output_residual_leakage/`: 11-point output beta sweep under WCCE/TSCE with endpoint residual correction, full Core probes, firing/persistence diagnostics, high/low-rate group probes, pruning controls, and a separate Slurm pipeline. It is not part of the canonical 36-run CoreBenchmark v1.1 manifest.
- `extensions/output_spike_drain/`: 11-point beta sweep for end-to-end WCCE with positive-only cap-1 output spikes and output-only full endpoint drain (`beta_drain=1`, zero new evidence), reporting drained vs valid-only BA, drain-capacity diagnostics, standard Core probes, and a separate 33-run Slurm pipeline. It is not part of the canonical CoreBenchmark v1.1 manifest.
- `extensions/output_spike_tsce_beta/`: paired 11-point beta sweep for end-to-end spike-only TSCE, with separate valid-only and fully-drained modes. Training loss, checkpoint selection, and native inference use output spikes only; analog `Wz` is diagnostic-only. Drain spikes are attached to the final valid TSCE timestep without increasing the valid-length denominator. The 66-run Slurm pipeline includes a compute-node preflight smoke and is not part of the canonical CoreBenchmark v1.1 manifest.
