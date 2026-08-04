# Neuromorphic-Gravity Project Analysis Progress

## Objective

Analyze the complete repository with emphasis on:

- wavelet encoding implementation
- wavelet parameter provenance
- continuous IMU signal to spike conversion
- SNN architecture and neuron dynamics
- training, evaluation, quantization, and deployment
- supporting scripts and notebooks
- recommended learning path

Final deliverable:

- `docs/PROJECT_SUMMARY_REPORT.md`

## Current status

- [x] Repository structure collected
- [x] Project-summary task created
- [ ] Repository instructions created and verified
- [ ] Investigation plan generated
- [ ] Investigation plan reviewed
- [ ] Wavelet implementation traced
- [ ] Encoder variables and parameter provenance traced
- [ ] SNN implementation traced
- [ ] Training and evaluation pipeline traced
- [ ] Scripts and notebooks classified
- [ ] Learning path prepared
- [ ] Final report written
- [ ] Final report verified

## Investigation log

### Initial setup

Files available:

- `AGENTS.md`
- `docs/PROJECT_SUMMARY_TASK.MD`
- `PROGRESS.md`

Next step:

- Ask Codex to inspect the repository and produce an investigation plan.

### Phase 1 — Repository State and Subsystem Map (2026-07-28; corrected)

- Authorization: explicitly authorized Phase 1 only; Phase 2 not started. The correction set is complete and the Phase 1 completion criterion is now met; evidence is in `docs/PROJECT_SUMMARY_REPORT.md` Appendices A and F.
- Corrected baseline: `main...origin/main [ahead 1]`, commit `47c0289b10b390ae27527810df5dedaefa47006a`. Pre-existing changes preserved: modified/mode-changed `environment.yml` and untracked `environment.original.yml`. Phase 1 created untracked `docs/PROJECT_SUMMARY_REPORT.md` and modified tracked `PROGRESS.md` only.
- Files inspected: required instructions/task/plan/progress; `README.md`; three environment declarations (Git `HEAD`, local, untracked); Git state; filesystem- and Git-aware artifact inventories; candidate-entrypoint/path excerpts in `preprocessing/`, `python-pipeline/`, and `snn/`.
- Corrected findings: environment declarations differ on Python, NumPy, `torchaudio`, and CUDA/backend pins; all listed Rockpool/NeuroBench/Samna/Tonic/Sinabs/snnTorch/XyloSim declarations match. `data/` has two tracked CSVs and `results/` 13 tracked files; `rg --files` omitted ignored files and is not a complete count source. Candidate entrypoints now include `preprocessing/run_metrics.sh`. Hard-coded `/work/...` and `/home/...` paths are a portability risk.
- Commands: prior read-only `sed`/`wc`, `git branch/rev-parse/status/diff/ls-files/log`, `rg`, `nl -ba`, `stat`, and one import-availability check; correction-only read-only `git status -sb/status --short/show HEAD:environment.yml`, `nl -ba`, `find -type f`, and `git ls-files`. No new runtime checks. The prior combined static-search quoting error was rerun successfully.
- Unresolved: authoritative environment ownership/provenance; availability of the absolute external data/model roots; and active-production reachability of every candidate entrypoint.
- Next step: Awaiting explicit user authorization for Phase 2.

### Phase 2 — Active End-to-End Data Flow (2026-07-28; corrected)

- Authorization: explicitly authorized Phase 2 only; Phase 3 not started. The requested Phase 2 correction set is complete; revised evidence is in `docs/PROJECT_SUMMARY_REPORT.md` Appendix F.
- Files inspected: `preprocessing/preprocess.py`, `preprocessing/prepare_data.py`, `snn/har_snn.py`, `snn/utils_datasets.py`, `snn/utils_architectures.py`, `snn/utils_run.py`, `snn/utils_parser.py`, `python-pipeline/modules.py:L19-L31`, and ordered JSON source cells of `notebooks/xylo_sim.ipynb`; static repository searches checked producer-consumer links.
- Corrected classifications: default custom branch is **Strongly inferred intended active production; current executable reachability blocked**. Its intent and `eventsNIMU` consumer match, but `preprocess.py` is first blocked by an unavailable unconditional `spikify` import and would later append an unavailable absolute `python-pipeline` root with no repository-relative fallback; `NeuromorphicIMUPipeline` is defined at `python-pipeline/modules.py:L19-L31`. `--use_xylo` and raw-with-gravity retain active optional intent but their `preprocess.py` execution is likewise blocked before branch selection; standalone FilterBank remains unresolved/likely defective and blocked; `prepareWindows()` remains dead/unexecuted; `xylo_sim.ipynb` remains experimental software simulation.
- Phase 6 reachability amendment: `preprocess.py` imports `spike_encoders.py` before parser construction, and that module unconditionally imports unavailable `spikify`; therefore all `preprocess.py` selectors are static in the current environment. In particular, `--spike_encoder` is statically selectable but not currently executable; its metrics utility is a static consumer with the same import blocker.
- Corrected defect/boundary: raw gravity-removed is **Unresolved / statically defective** because `gr_path` is assigned only under `if not args.raw_data` but referenced under `args.raw_data`, so `_gr/windows` is not statically reachable. `XyloSamna` is imported/referenced but not constructed or executed; physical deployment is an unexecuted hardware-only boundary. No legacy path was established within the Phase 2 inspection scope.
- Persistence: custom `eventsNIMU` and Xylo `spikesXylo` intent flows to `windows/P*_spikes.npy`/`P*_labels.npy` then `BaseHARDataset`; raw `_wg` has a matching consumer, while raw `_gr` is intended only and defective.
- Commands: prior static `wc`, `rg`, `nl -ba`, `sed`, and `git status`; standard-library JSON source inspection of `xylo_sim.ipynb`. Correction-only read-only `nl -ba python-pipeline/modules.py`, `rg`, and `git status`. No new runtime checks or execution occurred.
- Unresolved/blockers: current custom-path import reachability, hard-coded external data/model roots, standalone FilterBank consumer, defective raw `_gr` branch, notebook execution order/reproducibility, and physical-hardware behavior.
- Next step: Awaiting explicit user authorization for Phase 3.

### Phase 3 — Custom Wavelet Mathematics (2026-07-28)

- Authorization: explicitly authorized Phase 3 only. Completion criterion met: source equations, support/normalization, sampled FIR, Prony IIR, and notebook-only alternatives are documented in `docs/PROJECT_SUMMARY_REPORT.md` Appendices A and B.
- Files inspected: `python-pipeline/wavelets.py`, mathematical/filter portions of `python-pipeline/modules.py`, `python-pipeline/utils.py`, and relevant ordered source cells from `cwt_fir_iir.ipynb`, `wavelet_experiments.ipynb`, `reconstruction_error.ipynb`, `module_test.ipynb`, and `test_MCU.ipynb`.
- Findings: all three kernels use `x=(n-(M-1)/2)/s`, strict `(-0.5,0.5)` support, their exact polynomial amplitudes, and `sqrt(1/s)`. Intended custom preprocessing uses acceleration wavelets at `[0.5,1,2,4,8]` with `sampleFreq=64`, widths `[128,64,32,16,8]`; the pipeline selects second-order (`p=q=2`) Prony IIR, while FIR is defined but not selected.
- Distinctions: `signal.cwt` and varied width/response/reconstruction settings are notebook experiments; source FIR is a causal tapped-delay sampled-kernel convolution; source IIR has `b0..b2` and `a=[1,a1,a2]` recurrence. `convm()` is Prony’s causal convolution-matrix helper; `butter()` is separate from wavelet-bank construction.
- Commands: read-only `sed`, `wc`, `rg`, `nl -ba`, `git status`, and Python-standard-library notebook JSON source inspection. No runtime coefficient check, training, preprocessing, notebook execution, GPU, hardware, or installation occurred.
- Unresolved: physical units and calibrated frequency/scale meaning; discrete unit-energy claim; per-band IIR numerical stability and exact delay; current custom-entrypoint reachability remains unresolved from Phase 2. Notebook execution order is uncertain.
- Next step: Awaiting explicit user authorization for Phase 4.

### Phase 4 — Custom Wavelet Implementation and Parameter Provenance (2026-07-28)

- Authorization: Phase 3 is approved; explicitly authorized Phase 4 only. Phase 4 corrections are recorded in `docs/PROJECT_SUMMARY_REPORT.md` Appendices B and C; Phase 5 was not started.
- Files inspected: `python-pipeline/modules.py`, `wavelets.py`, `utils.py`; custom `preprocessing/preprocess.py` branch; `preprocessing/spike_encoder_neurobench_utils_xylo_nimu.py`; `snn/utils_datasets.py`, `snn/utils_run.py`; and ordered construction/forward source cells in `reconstruction_error.ipynb`, `module_test.ipynb`, `test_MCU.ipynb`, `cwt_fir_iir.ipynb`, and `wavelet_experiments.ipynb`.
- Corrected output semantics: with `usePower=False`, `MaxFilterModule` sums independently qualified maximum and minimum terms. It emits an unbounded signed amplitude, is not guaranteed sparse, and doubles a constant nonzero centre value when it equals both extrema; it is not a binary `snn.Leaky` spike or necessarily one centre response.
- Provenance/state: table now separates constructor defaults, hard-coded caller overrides (`usePower=False`, `singleDim=False`, five frequencies, `accelerationWavelet`), CLI `--window_size`, and notebook-only `maxFiltWin=(0.3,0)`. IIR and extrema buffers begin zeroed; one `nimup` instance is reused across all rows without reset, so state is continuous within a run, fresh after reconstruction, and risky/Unresolved across discontinuous sequences.
- Dtype ledger: input tensor, IIR/max outputs, `(T,3,5)`, flattened `(T,15)`, CSV, CSV reload, and `.npy` dtypes are explicitly Unresolved where source gives no dtype; the SNN dataset explicitly casts returned arrays to `np.float32` and `run_epoch()` casts batches to `torch.float`. The CSV/`.npy` upstream dtype remains Unresolved.
- Unexecuted findings: `AbsModule` and local `snn.Leaky` are constructed but omitted from `self.model`; `beta`, `threshold`, and `quantized` do not affect output; FIR, Quantizer, LocalToGlobal, and AccelToPower are unselected alternatives. `AccelToPower.velCurr` buffer is unused. Notebook variants are exploratory; `test_MCU.ipynb` attempts to unpack two outputs from a one-tensor forward call.
- Commands: read-only `sed`, `wc`, `nl -ba`, `rg`, `git status`, `git diff --check`, and Node JSON source-cell inspection. A `jq --version` attempt reported `jq` unavailable, so no notebook was executed or modified. No synthetic forward check was needed; no training, preprocessing, package install, GPU, hardware, or Phase 5 work occurred.
- Unresolved: current absolute-import reachability; physical units/amplitude calibration and IIR stability/phase delay; all custom-producer through `.npy` dtypes before SNN conversion; notebook execution order; and non-selected local-frame/power-path numerical behavior.
- Next step: Awaiting explicit user authorization for Phase 5.

### Phase 5 — Rockpool NIMU / `IMUIFSim` (2026-07-28)

- Authorization: Phase 4 is approved; explicitly authorized Phase 5 only. Phase 5 corrections are applied and its completion status is deferred pending review; Phase 6 was not started. Evidence is in `docs/PROJECT_SUMMARY_REPORT.md` Appendices B, C, and G.
- Files inspected: NIMU/Xylo portions of `preprocessing/preprocess.py` and `preprocessing/spike_encoder_neurobench_utils_xylo_nimu.py`; relevant ordered cells in `module_test.ipynb`, `reconstruction_error.ipynb`, `xylo_sim.ipynb`, and `xylo-imu-intro.ipynb`; installed Rockpool 2.9.1 `Quantizer`, `IMUIFSim`, `RotationRemoval`, `SubSpace`, IMU `FilterBank`, spike encoders, parameters, module shape handling, and IMUIF configuration helpers.
- Call/configuration: the active optional `--use_xylo` path clips `/2`, quantizes three axes with `scale=1`, `num_bits=16`, then calls `IMUIFSim(bypass_jsvd=True, sampling_freq=64, select_iaf_output=True)`. Earlier preprocessing separately applies Rockpool `RotationRemoval(num_avg_bitshift=4, sampling_period=10)` followed by repository high-pass gravity removal. IAF threshold is the installed default 1024; notebook-only `module_test` overrides it to 4098.
- Key correction/ambiguity: both active scripts construct or compute rescaled frequency values for labels/experiments, but neither passes its `FilterBank` list to `IMUIFSim`; the installed 15 default bands remain active. Label-to-fixed-coefficient physical correspondence at 64 Hz is Unresolved.
- Shape/state/events: preprocessing calls the active branch once on the full resampled subject `(T_subject,3)`, which auto-batches to `(1,T_subject,3)` and yields `(1,T_subject,15)` axis-major/band-minor IAF events clipped to 0/1. It reshapes/persists the full `(T_subject,15)` sequence, then reloads and stacks later 10-second `(640,15)` windows into `(N,640,15)`. `FilterBank` reverse-prefix boundary handling is not strictly causal at the subject segment start; its recurrence and IAF integration reset at that subject-level `evolve()` call, not per saved window. Rotation `SubSpace.C_highprec` is stateful within its preprocessing call.
- Software/hardware boundary: `xylo_sim.ipynb` is separate Xylo-core simulation, not `IMUIFSim`; `xylo-imu-intro.ipynb` is experimental/tutorial and includes static-only hardware cells. No hardware discovery, Samna hardware import/object, `XyloSamna`, or `IMUIFSim` evolution was run.
- Commands: read-only `sed`, `wc`, `rg`, Node notebook JSON inspection, `git status`, `nl -ba`, `git diff --check`, and two Python metadata/spec commands recorded verbatim in Appendix G. They returned Rockpool `2.9.1` and its installed source root. No tiny runtime attempt was needed.
- Unresolved: physical label/filter correspondence at 64 Hz; scientific gravity/frame correctness; CSV/window dtype after Boolean persistence; notebook execution order/version compatibility; physical hardware, sensor, Samna transport, and timing equivalence.
- Next step: Await Phase 5 correction review, then explicit user authorization before Phase 6.

### Phase 6 — Active Standalone `FilterBank` Encoding (2026-07-28)

- Authorization: Phase 5 is approved; explicitly authorized Phase 6 only. Phase 6 corrections are applied and its completion status is deferred pending re-review; Phase 7 was not started.
- Files inspected: standalone branch in `preprocessing/preprocess.py`; `preprocessing/spike_encoders.py`; `preprocessing/spike_encoder_neurobench_utils.py`; SNN dataset-root routing; `spike_encoder_neurobench.ipynb` and the relevant Rockpool-FilterBank notebook cells; environment declarations and active site-packages provenance.
- Provenance/configuration: `spike_encoders.py` re-exports `spikify.filtering.FilterBank`, but `preprocess.py` imports it before parsing and is therefore currently blocked. If that blocker were resolved, the static branch would call `FilterBank(fs=64, channels=5, f_min=0.32, f_max=10.24, order=1)` once on full `(T_subject,3)` gravity-removed input, then `decompose().reshape(-1,15)`. Centre-frequency construction, filter mathematics/order convention, state, causality, output dtype/range, and whether it is wavelet-based remain Unresolved.
- Contract/classification: axis-major five-band labels are intended, then a selected encoder would write `spikes_*` CSV columns. The metrics utility is a static CSV consumer but is currently blocked by the same unconditional import. The window branch instead selects `spikenc` columns and `har_snn.py` has no `spikes<encoder>` dataset root, so the intended 15-channel SNN handoff is statically defective; classification remains **Unresolved / likely defective; current executable reachability blocked**.
- Evidence/commands: E1 repository source and static notebook JSON inspection; E3 provenance lookup found package/module/source absence. Multiple Python introspection attempts were made for that one blocked question after the first `PackageNotFoundError`, exceeding the single-attempt policy; the previously omitted exact distribution-search command is now recorded in Appendix G. No further `spikify` check will be made. `git diff --check` passed after these corrections. No `FilterBank` construction, decomposition, downstream encoder, training, preprocessing, install, GPU, hardware, or Phase 7 work occurred.
- Unresolved: exact `spikify` version and implementation; `decompose()` shape/order/dtype/range/state/causality; centre-frequency calculation/units; external encoder output contract; and whether generated datasets exist at the hard-coded external root.
- Next step: Awaiting explicit user authorization for Phase 7.

### Phase 7 — Encoder Comparison, Shape Ledgers, and SNN Input Contracts (2026-07-28)

- Authorization: Phase 6 is approved; explicitly authorized Phase 7 only. Phase 7 corrections are applied and completion status is deferred pending re-review; Phase 8 was not started.
- Files inspected: consolidated Phase 2–6 evidence; `preprocessing/preprocess.py`, `preprocessing/prepare_data.py`; `snn/har_snn.py`, `utils_parser.py`, `utils_datasets.py`, `utils_run.py`, and relevant `utils_architectures.py` constructors.
- Contracts: custom and NIMU each intend `(N,int(64*window_size),15)` windows in `x`-five, `y`-five, `z`-five order. Default `SynNet` uses `inputSize=15`, no default time/channel transform, and `run_epoch()` receives `(B,T,15)`. Custom values are signed amplitudes; NIMU values are Boolean IAF events. Standalone intended 15 columns are selected as zero `spikenc` columns, has no SNN root, and remains blocked by unavailable `spikify`.
- Transform/mismatch findings: polarity maps `(T,C)` to all-positive then all-negated-negative `(T,2C)` unless rectification wins; both flags cause 15 transformed channels but a 30-input model. Time rules/downsampling are SNN-side and padding hard-codes ten seconds. Optional quantization is 5-bit fractional-power/clipping and may be unresolved for negative custom amplitudes. `compress_channels` is stored but unused; `add_gravity` overwrites `[0,5,10]` rather than adding channels. `SNNMLP` is compatible with default untransformed 10-second 15-channel windows (`640*15=9600`), but polarity, time-transform, or non-10-second cases need separate arithmetic; ANN/LSTM/CNN special dimensions remain incompatible with ordinary 15-channel windows.
- Commands: read-only `sed`, `wc`, `nl -ba`, `rg`, and `git status`; `git diff --check` passed. No runtime, synthetic shape check, dependency probe, preprocessing, training, install, GPU, hardware, or Phase 8 work occurred.
- Unresolved: executable preprocessing remains blocked by `spikify` (and custom's absolute import); standalone FilterBank internals/output; producer CSV/`.npy` dtypes; non-default transformed lengths for non-integral factors; and actual artifact availability at external roots.
- Next step: Awaiting explicit user authorization for Phase 8.

### Phase 8 — Primary SNN Architecture and Neuron Dynamics (2026-07-28)

- Authorization: Phase 7 is approved; explicitly authorized Phase 8 only. The Phase 8 correction set is applied and awaits re-review; evidence is recorded in `docs/PROJECT_SUMMARY_REPORT.md` Appendices A, B, C, F, and G. Phase 9 was not started.
- Primary model/call chain: parser default `network_type='SynNet'` → `har_snn.py` hard-codes/flag-derives `inputSize=15` or `30` (it does not inspect loaded data to derive it) and selects output classes → `createModel()` → local snntorch `SynNet`. It uses default hidden sizes `[24,24,24]`, four biased affine layers, and final binary output spikes stacked as `(B,T,C)`; class prediction is normalized time-summed output-spike score, not membrane state.
- Dynamics/state: defaults `shiftSyn=2`, `shiftMem=1` yield the documented per-layer alpha groups and `beta=0.5`; alpha/beta are non-trainable, thresholds start at/trainable 1, reset is delayed subtractive, and the default surrogate is installed snntorch's `atan(alpha=2)`. Each `forward()` zeros all four synaptic/membrane states, then preserves them only within that item’s time axis; no learned recurrent matrix is present. Local nominal tau formulas and `tausFactor` are computed but unused by local `SynNet`.
- Alternatives/classification: `SynNetRP` is active optional/deployment-relevant with actual `dt=timeResolution`, bitshift-derived taus, Rockpool graph export, and caps 31/1; mapping/simulation/deployment are deferred. The repository-local Rockpool LIF duplicate is unimported by this call path. `snn.py` is a **Legacy path — statically defective standalone candidate entrypoint**: its `__main__` constructs/invokes `SNNIMU`, but obsolete/unresolved `from architectures import SynNet` is blocked because only `utils_architectures.py` exists; it has no active artifact consumer. `mnist_snn.py` is an unrelated legacy/tutorial. Physical hardware remains an unexecuted boundary.
- Files inspected: `snn/har_snn.py`, `utils_parser.py`, `utils_architectures.py`, `rockpool_nn_networks_synnet.py`, `rockpool_nn_modules_torch_lif_torch.py`, `utils_run.py`, `utils_losses.py`, `snn.py`, `mnist_snn.py`; relevant ordered cells in `notebooks/har_snn.ipynb` and `notebooks/SynNet_rockpool.ipynb`; exact installed `snntorch 0.9.4` and PyTorch `2.6.0+cpu` source.
- Commands: read-only `sed`, `nl -ba`, `rg`, `rg --files snn | sort`, and `rg --files -g 'architectures.py' || true` (confirmed no repository `architectures.py`), `git status`, `git diff --check` (passed), standard-library JSON notebook-source inspection, and one metadata/spec command recorded verbatim in Appendix G. A correction-only `rg` pattern had a harmless shell-quoting failure (`inputSize: command not found`); it was ignored and is recorded in Appendix G. No synthetic forward/reset check, training, preprocessing, package change, GPU, hardware, or Phase 9 investigation occurred.
- Unresolved: end-to-end default execution remains blocked by the upstream `spikify` import (and the custom absolute-import path); current checkpoint reproducibility and Rockpool graph/Xylo mapping/deployment details are deferred; notebook execution order is uncertain.
- Next step: Awaiting Phase 8 re-review and renewed explicit authorization before Phase 9.

### Phase 9 — Training, Evaluation, Quantization, Simulation, and Deployment (2026-07-28; corrected)

- Authorization: Phase 8 is approved; explicitly authorized Phase 9 only. Corrected static evidence is in `docs/PROJECT_SUMMARY_REPORT.md` Appendices B, F, and G; completion status is deferred pending re-review. Phase 10 was not started.
- Files inspected: `snn/har_snn.py`, `utils_datasets.py`, `utils_losses.py`, `utils_run.py`, `utils_parser.py`, `rockpool_nn_networks_synnet.py`, `utils_architectures.py`, `notebooks/rockpool_transform_quantize_methods.py`; ordered source cells in `neurobench.ipynb`, `har_snn.ipynb`, `xylo_sim.ipynb`, and `xylo-imu-intro.ipynb`; installed Rockpool 2.9.1 mapper/configuration/XyloSim source and NeuroBench 2.1.0 benchmark/metric source.
- Training/evaluation: default `[20,5,5]` is normalized to a reproducible subject split; train/validation are class-balanced and shuffled, test is unbalanced/unshuffled; defaults are batch 256, Adam `5e-4`, 20 epochs, and `CrossEntropySpkReg(alpha=0)`. The alternate FirstWin/regularization paths, score normalization, confusion matrix, AUROC/F1/accuracy/balanced accuracy/MCC/kappa are documented. `confusion_matrix` uses accumulated targets/predictions and `targ_dict.keys()` label order; its result is wrapped unconditionally in `wandb.Table` with `targ_dict.values()` row/column names, although only `wandb.init()`/`wandb.log()` are flag-guarded. No Phase-9 code ran, so no W&B API/network contact occurred. The loss objectives are explicitly separated from the installed snnTorch `atan(alpha=2)` surrogate backward path. Checkpoint loading is optional and not performed; if selected it merges matching non-`'4'` keys and replaces Adam with `fc1/lif1` through `fc4/lif4` layer-specific rates, a layout likely defective for `SynNetRP`. Save-without-W&B remains a likely static defect.
- Metrics: NeuroBench notebook is an experimental `CNNNetSNN`/`TorchModel` workload measuring Footprint, ConnectionSparsity, ActivationSparsity, and SynapticOperations; it is separate from repository encoder/reconstruction metrics and the primary CLI `SynNet` route.
- Quantization/deployment: only optional `SynNetRP` exposes `as_graph()`. The static notebook chain is `as_graph → mapper → global_quantize(fuzzy_scaling=False) → config_from_specification → XyloSim`; `channel_quantize` is a commented/static alternative only. Installed Rockpool factory source (E2; version E3) validates and returns `(config, is_valid, msg)`, but the repository notebook (E1) stores and ignores that result before constructing XyloSim; validity of the repository model remains Unresolved. Reported limits are 8-bit weight/16-bit threshold quantizer defaults, mapper 496 hidden/16 output, one hidden synapse, and factory 128 IEN/OEN truncation behavior. Rockpool and NeuroBench notebook calls are E1; dependency internals are E2 with E3 version metadata; Samna validation is E2 dependency-source evidence, not hardware runtime.
- Boundaries/unresolved: XyloSim is software simulation. `find_xylo_hdks`/`XyloSamna` occur only in tutorial cells and remain physical-hardware-only; no W&B contact, checkpoint load, hardware discovery, Samna/XyloSim construction, model execution, training, preprocessing, package change, GPU, or physical deployment occurred. End-to-end reachability remains blocked upstream by `spikify`; data/model roots, mapped/quantized accuracy, hardware timing/power/transport, and notebook reproducibility remain Unresolved.
- Commands: read-only `sed`, `tail`, `wc`, `rg`, `nl -ba`, and standard-library JSON notebook-source inspection; `/home/ted/miniconda3/envs/xyloIMU/bin/python -c 'import importlib.metadata as m; print("rockpool",m.version("rockpool")); print("neurobench",m.version("neurobench"))'` returned Rockpool `2.9.1` and NeuroBench `2.1.0`. An out-of-range static notebook cell-inspection command printed `IndexError` after the available cells; its prior cell output was used only where present and it supplied no behavioral evidence.
- Correction-only commands: read-only `sed`, `nl -ba snn/har_snn.py`, `rg`, `git diff --check`, and `git status`; no new runtime check, checkpoint load, model operation, W&B contact, simulation, or hardware action occurred.
- Next step: Awaiting Phase 9 correction re-review and explicit user authorization before Phase 10.

### Phase 10 — Supporting Scripts, Notebooks, Data, and Results (2026-07-28)

- Authorization/status: Phase 9 is approved; Phase 10 only is complete. The report additions are in Appendices D, E, and F. Phase 11 was not started.
- Inspected: the complete meaningful `.py`/`.sh` inventory; 13 `.ipynb` files under `notebooks/` plus `xylo-imu/xylo-imu-intro.ipynb` (14 notebooks total); `notebooks/rockpool_transform_quantize_methods.py` as meaningful notebook-adjacent Python source, not a notebook; both included Subject001 left/right CSVs; 13 tracked result files; `notebooks/media/` (119 table JSONs); and `snn/models/` only as a 156-file/approximately-26-MB checkpoint class (no enumeration or load).
- Findings: primary/custom/SNN scripts retain their Phase 2–9 classifications; `run_metrics.sh` is a static, environment-specific metrics entrypoint blocked by the shared `spikify` import; `compatibility.sh` is an unused unsafe installed-package rewrite utility. Sample CSVs have the seven documented time/acc/gyr columns, 512-Hz observed timestamp interval, approximately 1040.7-s coverage, left/right filename identity, and unresolved units/sensor/frame metadata. FIR/IIR and NIMU/Xylo reconstruction figures have matching notebook `savefig` calls; spike figures, historical aggregate CSVs, W&B media, and AUROC figure have only partial or commented/static provenance.
- Commands/results: read-only `sed`/`tail`/`nl -ba` source reads; `rg --files -g '*.py' -g '*.sh' -g '*.ipynb' | sort` listed the script/notebook inventory; `find data results notebooks/media snn/models -maxdepth 3 -type f -printf ... | sort` established artifact classes; `rg -n` searched result names and persistence calls; standard-library `python -c` JSON parsing identified notebook cells (no notebook execution); `head`/`awk` sampled CSV schemas/timestamps; and `wc`/`find` counted result/media/checkpoint classes. Final `git diff --check` emitted no diagnostics; `git diff --no-index --check /dev/null docs/PROJECT_SUMMARY_REPORT.md` likewise emitted no whitespace diagnostics and returned its expected content-difference status 1. `git status --short` confirmed only the permitted report/progress edits plus pre-existing `environment.yml`/`environment.original.yml`. No package import, W&B request, checkpoint load, preprocessing, training, simulation, hardware discovery, or hardware action occurred.
- Unresolved: current end-to-end preprocessing remains blocked by unavailable `spikify` (and custom absolute imports); notebook execution/order/output freshness, physical units/sensor/frame metadata, and exact historical-result/W&B/checkpoint provenance remain unresolved.
- Next step: Awaiting explicit authorization for Phase 11.

### Phase 11 — Minimal Deterministic Verification (2026-07-28)

- Authorization/status: Phase 10 is approved; Phase 11 only is complete. Appendix G records candidate status, rationale, the sole exact runtime command, and observed output. Phase 12 was not started.
- Eligible check: static source left the direct custom-pipeline output dtype unresolved. One CPU-only direct-module check used the active 64-Hz, `globalFrame=True`, `usePower=False`, `singleDim=False`, five-frequency `accelerationWavelet` configuration and the first 20 included left-CSV acceleration rows. It returned `torch.float32`, `(20,3,5)`, eight nonzero values, and nonzero IIR/extrema histories: **PASS**, limited to explicit float32 direct input and not evidence of end-to-end CLI reachability or persisted dtype.
- Exact command/result: `python -c "import csv,sys,torch; sys.path.insert(0,'python-pipeline'); from modules import NeuromorphicIMUPipeline; from wavelets import accelerationWavelet; f=open('data/Subject001_left_global_acc.csv', newline=''); rows=list(csv.DictReader(f)); f.close(); x=torch.tensor([[float(r['acc_x']),float(r['acc_y']),float(r['acc_z'])] for r in rows[:20]], dtype=torch.float32); p=NeuromorphicIMUPipeline(64, globalFrame=True, usePower=False, singleDim=False, frequencies=torch.tensor([0.5,1.0,2.0,4.0,8.0]), waveletFunc=accelerationWavelet); y=torch.stack([p(v) for v in x]); print('input_dtype',x.dtype); print('output_dtype',y.dtype); print('output_shape',tuple(y.shape)); print('output_nonzero',int(torch.count_nonzero(y))); print('iir_state_nonzero',bool(torch.count_nonzero(p.model[2].signalPrev))); print('extrema_state_nonzero',bool(torch.count_nonzero(p.model[3].signalPrev)))"` → `input_dtype torch.float32`; `output_dtype torch.float32`; `output_shape (20, 3, 5)`; `output_nonzero 8`; `iir_state_nonzero True`; `extrema_state_nonzero True`.
- Skipped/blocked: no successful prior check was repeated. `spikify` FilterBank remains **BLOCKED** without a retry because Phase 6 exhausted the safe dependency-provenance attempt; parser help would only repeat its import gate. IMUIFSim, transforms, primary SNN, quantization, mapper/configuration/XyloSim, checkpoint, and sample-data checks are **NOT NEEDED** because static/prior evidence resolves their eligible contracts or a check cannot resolve the remaining actual-model question. Hardware/W&B/history and physical unit/frame questions remain Unresolved and are not locally testable under policy.
- Remaining unresolved: upstream `spikify` and custom absolute-import reachability; standalone FilterBank internals; producer/persistence dtypes; physical units/frame/filter correspondence; actual mapped/checkpoint behavior; notebook/history freshness; and physical-hardware behavior.
- Next step: Awaiting explicit authorization for Phase 12.

### Phase 12 — Synthesis, Learning Path, and Final Report Construction (2026-07-28)

- Authorization/status: Phase 11 is approved; Phase 12 only is complete. `docs/PROJECT_SUMMARY_REPORT.md` now has the Chinese explanatory main narrative for sections 1–15 plus the accumulated Phase 1–11 Appendices A–H. No implementation or protected artifact was modified.
- Synthesis/audit: reconciled the blocked custom/NIMU/standalone producer paths with the 15-channel SNN contracts; preserved custom signed-amplitude versus NIMU Boolean-event semantics; separated static E1, installed-source E2, and minimal-observation E3 claims; retained software `XyloSim` versus unexecuted hardware boundaries; and limited reproduction commands to ones actually recorded as verified. Main-section/appendix heading audit found all 15 required main sections and Appendices A–H; confidence/evidence markers occur throughout the report.
- Commands/results: complete `sed` reads of `AGENTS.md`, task, plan, `PROGRESS.md`, and report evidence; read-only `rg` structure/confidence audits; `git status --short`; `git diff --check` emitted no diagnostics; `git diff --no-index --check /dev/null docs/PROJECT_SUMMARY_REPORT.md` emitted no whitespace diagnostics and returned expected status 1 because the report is untracked. One initial Phase-12 `rg` command had an unmatched-backtick shell-quoting error, was discarded as non-evidentiary, and was immediately replaced by a safe single-quoted audit. No runtime/model/dependency/W&B/simulation/hardware command was run in Phase 12.
- Final limitations: current preprocessing remains blocked by unavailable `spikify` and custom absolute imports; standalone FilterBank internals/SNN handoff, persistence dtypes, physical units/frame/filter correspondence, checkpoint/config validity, notebook/history provenance, and physical-hardware behavior remain Unresolved.
- Next step: Investigation complete; final report delivered.
