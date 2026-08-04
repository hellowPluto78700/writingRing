# Neuromorphic-Gravity Project Summary Execution Plan

## 1. Execution Contract

This plan implements `docs/PROJECT_SUMMARY_TASK.MD`.

The executor must work one phase at a time. Completion of one phase does not authorize another.

At the beginning of every phase:

1. Read `AGENTS.md`.
2. Read `docs/PROJECT_SUMMARY_TASK.MD`.
3. Read this plan.
4. Read `PROGRESS.md`.
5. Confirm the exact phase explicitly authorized by the user.
6. Check the working tree and preserve pre-existing changes.
7. Perform only that phase.

At phase completion:

1. Add detailed evidence to the designated report appendices.
2. Append a concise cumulative entry to `PROGRESS.md`.
3. Report evidence, classifications, checks, blockers, and unresolved questions.
4. State whether the completion criterion was met.
5. Stop.

## 2. Safety Rules

Never modify source code, notebooks, datasets, checkpoints, environment files, existing results, or generated experiment media.

During execution, write only:

- `docs/PROJECT_SUMMARY_REPORT.md`
- `PROGRESS.md`

Do not modify this plan or the task without explicit user instruction.

Do not install packages, update environments, download data, contact W&B, train models, preprocess full datasets, execute notebooks wholesale, use GPU, or discover/access physical hardware.

Use `rg` and `rg --files` for repository searches.

## 3. Evidence Rules

Use `Confirmed`, `Strongly inferred`, and `Unresolved`.

Use citations:

- Python/shell: `path/to/file.py:L10-L25`
- Notebook: `path/to/notebook.ipynb`, cell N

Cite caller and callee for call chains.

### External evidence tiers

- **E1:** repository source and call sites.
- **E2:** installed source at the exact runtime version.
- **E3:** minimal runtime introspection.
- **E4:** official version-relevant documentation.

Keep repository-supplied behavior separate from dependency internals.

### Path classifications

- **Active production:** current CLI/main reachability plus artifacts consumed by active training/evaluation.
- **Active optional:** supported branch/flag plus an identifiable consumer.
- **Experimental:** notebook/ad hoc path without a confirmed active CLI-to-consumer flow.
- **Legacy:** superseded tutorial, obsolete API/import/path, or unused older implementation.
- **Dead or unexecuted:** defined or constructed but unreachable, uncalled, omitted from execution, or caller-less.

Use `Unresolved` where evidence is insufficient.

## 4. Detailed Evidence Storage

During Phases 1–11:

- Keep the main report narrative empty or skeletal.
- Add detailed evidence to:
  - Appendix A: symbols and variables.
  - Appendix B: parameter provenance.
  - Appendix C: shape ledgers.
  - Appendix D: script inventory.
  - Appendix E: notebook inventory.
  - Appendix F: file/function index.
  - Appendix G: verification commands/results.
  - Appendix H: non-wavelet encoder classification.
- Keep `PROGRESS.md` concise.
- Record every exact runtime command and a one-line outcome in `PROGRESS.md`.

Phase 12 completes the main report and audits the appendices.

## 5. Phase 1 — Repository State and Subsystem Map

### Objective

Establish the reproducible baseline and top-level subsystem boundaries without detailed script or notebook analysis.

### Priority

Priority 1 for state and entrypoints; Priority 3 for artifact classification.

### Inspect

- `README.md`
- `environment.yml`
- `environment.original.yml`
- Git branch, commit, status, and diff
- Top-level repository tree
- Candidate CLI and `__main__` entrypoints
- Hard-coded external paths
- Source, generated, data, result, and checkpoint directories at classification level

### Parameters

Branch and commit, tracked/untracked changes, file modes, environment name, declared critical package versions, and hard-coded dataset/model/result/home paths.

### Expected evidence

Repository-state and environment-provenance tables, top-level subsystem diagram, candidate entrypoints, artifact classification, and preliminary path classifications.

### Permitted verification

Read-only Git commands, file listing, environment comparison, and import-availability checks without installation.

### Blockers

Ambiguous environment ownership, undeclared dependencies, minimal documentation, or unavailable external roots.

### Appendix targets

Appendices A and F.

### Completion criterion

Repository state, environment provenance, subsystem boundaries, candidate entrypoints, and hard-coded paths are documented.

### `PROGRESS.md`

Record baseline, files, commands, conclusions, unresolved environment questions, and `Awaiting explicit user authorization for Phase 2.`

### Stopping condition

Stop after the Phase 1 handoff.

---

## 6. Phase 2 — Active End-to-End Data Flow

### Objective

Build the actual raw-input-to-decision call graph and classify every major path.

### Priority

Priority 1 — Exhaustive.

### Inspect

- `preprocessing/preprocess.py`
- `preprocessing/prepare_data.py`
- `snn/har_snn.py`
- `snn/utils_datasets.py`
- `snn/utils_architectures.py`
- `snn/utils_run.py`
- Active conversion/simulation cells in `notebooks/xylo_sim.ipynb`

### Trace

CLI parsing, dataset selection, resampling, gravity handling, encoder dispatch, windowing/persistence, `createLoaders()`, `BaseHARDataset`, active transforms, `createModel()`, `run_epoch()`, and simulation entry cells.

### Parameters

Dataset configuration, sampling, gravity/encoder flags, windowing, persisted paths/columns, subject splits, transforms, batching, model selection, and simulation selectors.

### Expected evidence

A Mermaid graph separating active production, active optional, experimental, legacy, dead/unexecuted, software simulation, and hardware-only paths. Each edge records caller, callee, branch condition, artifact, consumer, and confidence.

### Permitted verification

Static call tracing, producer-consumer matching, and one safe parser `--help` attempt for a material unresolved CLI question.

### Blockers

External datasets, import-time side effects, hard-coded paths, or inconsistent artifact names.

### Appendix targets

Appendices F and H.

### Completion criterion

Every major stage has an entrypoint, branch condition, input/output contract, consumer, and classification.

### `PROGRESS.md`

Record major paths, classifications, commands/blockers, unresolved branches, and `Awaiting explicit user authorization for Phase 3.`

### Stopping condition

Stop after the Phase 2 handoff.

---

## 7. Phase 3 — Custom Wavelet Mathematics

### Objective

Derive repository-specific wavelet equations and mathematical transformations.

### Priority

Priority 1 — Exhaustive.

### Inspect

- `python-pipeline/wavelets.py`
- Mathematical portions of `python-pipeline/modules.py`
- `python-pipeline/utils.py`
- Relevant cells in `cwt_fir_iir.ipynb`, `wavelet_experiments.ipynb`, `reconstruction_error.ipynb`, `module_test.ipynb`, and `test_MCU.ipynb`

### Trace

`powerWavelet()`, `accelerationWavelet()`, `velocityWavelet()`, `convm()`, `prony()`, and `butter()`.

### Parameters

`M`, `s`, support, normalization, sampling, frequency-to-width conversion, kernel length, FIR coefficients, Prony orders, IIR coefficients, and notebook overrides.

### Expected evidence

Exact equations, transform classification, support/normalization, frequency-scale relationship, boundary behavior, delay/causality, IIR state/stability, and active-versus-experimental distinction.

### Permitted verification

One small coefficient, kernel, or impulse check only for an unresolved indexing, delay, or coefficient question.

### Blockers

Unknown units, conceptual frequency labels, conflicting notebook values, or unresolved stability.

### Appendix targets

Appendices A and B.

### Completion criterion

Every custom wavelet expression maps to code variables, operations, timing, and confidence.

### `PROGRESS.md`

Record formulas, derivations, inspected cells, command/result or blocker, unresolved interpretations, and `Awaiting explicit user authorization for Phase 4.`

### Stopping condition

Stop after the Phase 3 handoff.

---

## 8. Phase 4 — Custom Wavelet Implementation and Parameter Provenance

### Objective

Trace the executable custom pipeline through construction, stateful processing, output, persistence, and SNN consumption.

### Priority

Priority 1 — Exhaustive.

### Inspect and trace

`NeuromorphicIMUPipeline`, `QuantizerModule`, `LocalToGlobalModule`, `AccelToPowerModule`, `WaveletFIRFilterModule`, `WaveletIIRFilterModule`, `MaxFilterModule`, `AbsModule`, the custom preprocessing branch, and the custom NIMU metric path.

### Parameters

Sampling/`dt`, coordinate frame, power conversion, dimensions, active frequencies, wavelet selection, FIR/IIR construction, state buffers, extrema window, quantization, thresholds, output labels, axis order, and band order.

### Expected evidence

Constructor-to-operation provenance, executed sequence, state lifecycle, output semantics/range, shape ledger, persisted-output contract, SNN consumer, and dead/unexecuted classification.

### Permitted verification

One short synthetic forward/state attempt only for a material unresolved output-shape or state question.

### Blockers

Hard-coded imports, missing dependency, numerical instability, or misleading names.

### Appendix targets

Appendices B and C.

### Completion criterion

The active custom path has complete construction, provenance, state, timing, output, and SNN-consumer evidence.

### `PROGRESS.md`

Record sequence, key provenance, shape summary, dead/unexecuted findings, command/result or blocker, and `Awaiting explicit user authorization for Phase 5.`

### Stopping condition

Stop after the Phase 4 handoff.

---

## 9. Phase 5 — Rockpool NIMU / `IMUIFSim`

### Objective

Trace repository-supplied NIMU configuration separately from Rockpool-internal behavior.

### Priority

Priority 1 for active NIMU; Priority 2 for hardware context.

### Inspect

The NIMU preprocessing branch, `spike_encoder_neurobench_utils_xylo_nimu.py`, relevant `module_test`, `reconstruction_error`, `xylo_sim`, and `xylo-imu-intro` cells, and exact-version installed Rockpool source where needed.

### Trace

`Quantizer`, `RotationRemoval`, Rockpool IMU-interface `FilterBank`, `IMUIFSim`, and configuration export helpers.

### Parameters

Normalization/clipping, quantizer scale/bits, sampling, `bypass_jsvd`, `select_iaf_output`, threshold/filter overrides, gravity handling, state, output reshape/order, and hardware constraints.

### Expected evidence

E1 repository call chain, E2 installed implementation, E3 observations only when needed, E4 documentation only for unresolved public contracts, NIMU shape ledger, and software/Xylo boundary.

### Permitted verification

One tiny CPU-only `Quantizer`/`IMUIFSim` attempt for a material unresolved default, shape, dtype, range, or state question.

### Blockers

External implementation, hidden defaults, version mismatch, or hardware-only behavior.

### Appendix targets

Appendices B, C, and G.

### Completion criterion

NIMU configuration, dependency behavior, timing, state, events, shapes, and Xylo relevance are established or explicitly unresolved.

### `PROGRESS.md`

Record evidence tiers, parameters, shape summary, runtime result/blocker, boundary, and `Awaiting explicit user authorization for Phase 6.`

### Stopping condition

Stop after the Phase 5 handoff.

---

## 10. Phase 6 — Active Standalone `FilterBank` Encoding

### Objective

Trace the active standalone `FilterBank` without expanding into unrelated downstream encoders.

### Priority

Priority 1 for `FilterBank`; Priority 3 for unrelated encoders.

### Inspect

`FilterBank` import/construction, `decompose()`, reshape/persistence, import provenance in `spike_encoders.py`, relevant consumers/notebook cells, and exact-version installed spikify source where available.

### Parameters

`fs`, bands, frequency bounds, filter order, center frequencies, axes, output order, state, dtype/range, and downstream interface parameters.

### Expected evidence

Import provenance, mathematical category, frequency/filter construction, state/causality, shape ledger, axis/band order, persisted/SNN contracts, and one-sentence downstream classifications.

### Permitted verification

One tiny `FilterBank` construction/decomposition attempt for a material unresolved question. Do not execute downstream unrelated encoders.

### Blockers

Missing spikify, unavailable installed source, column mismatch, or undocumented external construction.

### Appendix targets

Appendices B, C, and H.

### Completion criterion

The active FilterBank path has complete construction, frequency, state, shape, output, and consumer evidence.

### `PROGRESS.md`

Record provenance, frequency/order findings, shape summary, evidence tier, command/result or blocker, and `Awaiting explicit user authorization for Phase 7.`

### Stopping condition

Stop after the Phase 6 handoff.

---

## 11. Phase 7 — Encoder Comparison, Shape Ledgers, and SNN Input Contracts

### Objective

Compare the three detailed encoder families and reconcile each output independently with its SNN consumer.

### Priority

Priority 1 — Exhaustive.

### Inspect

Consolidated encoder evidence, active windowing/transforms, `snn/har_snn.py`, and relevant model constructors.

### Derive independently

Sampling, window duration/stride, time steps, axes, bands, polarity, channel count, order, dtype/range, downsampling/padding, quantization, batching, and `inputSize`.

### Expected evidence

Three encoder ledgers, corresponding SNN-input ledgers, comparison table, producer-consumer contracts, and explicit mismatches.

### Permitted verification

Synthetic shape-only checks for active transforms or reshapes that remain unresolved. No unrelated encoder execution.

### Blockers

Missing data, mismatched columns, undocumented order, or conflicting overrides.

### Appendix target

Appendix C.

### Completion criterion

Every dimension has an independently cited source and matched consumer or explicit unresolved mismatch.

### `PROGRESS.md`

Record shape findings, mismatches, command/result or blocker, unresolved contracts, and `Awaiting explicit user authorization for Phase 8.`

### Stopping condition

Stop after the Phase 7 handoff.

---

## 12. Phase 8 — Primary SNN Architecture and Neuron Dynamics

### Objective

Identify and fully explain the primary SNN while proportionately classifying alternatives.

### Priority

Priority 1 for the primary SNN; Priority 2 for deployment-relevant alternatives; Priority 3 for all remaining models.

### Inspect

`har_snn.py`, `utils_parser.py`, `utils_architectures.py`, `rockpool_nn_networks_synnet.py`, the local LIF implementation only when actually used, relevant training/SynNet notebook cells, and `snn.py`/`mnist_snn.py` for classification.

### Trace

Model dispatch, encoder-derived input, hidden/output sizes, shifts, decays/time constants, `dt`, thresholds/biases, recurrence, reset/state, surrogate gradient, spike limits, and decision rule.

### Expected evidence

Primary construction chain, layer table, Mermaid architecture, neuron equations, state lifecycle, tensor conventions, mode differences, and alternative classifications.

### Permitted verification

One short primary CPU forward/reset attempt for a material unresolved state or shape question.

### Blockers

Broken import, ambiguous primary model, state leakage, or local/installed divergence.

### Appendix targets

Appendices A, B, C, and F.

### Completion criterion

The primary SNN has complete architecture, dynamics, state, shape, input, and deployment evidence.

### `PROGRESS.md`

Record model chain, dynamics, classifications, command/result or blocker, unresolved relationships, and `Awaiting explicit user authorization for Phase 9.`

### Stopping condition

Stop after the Phase 8 handoff.

---

## 13. Phase 9 — Training, Evaluation, Quantization, Simulation, and Deployment

### Objective

Trace the active lifecycle from dataset through optimization/metrics to graph conversion, quantization, `XyloSim`, and hardware boundary.

### Priority

Priority 1 for active paths; Priority 2 for NeuroBench and hardware context.

### Inspect

Training/parser, dataset/loss/epoch utilities, Rockpool graph export, quantization helper, and relevant NeuroBench, training, simulation, and Xylo notebook cells.

### Trace

Splits/balancing, batching/transforms, active loss/regularization, surrogate gradient, optimizer/rates, metrics/decisions, NeuroBench, `as_graph()`, `mapper()`, quantization, configuration, `XyloSim`, and hardware-only interfaces.

### Expected evidence

Training/evaluation graph, quantization equations/bounds, simulation chain, dependency tiers, and software/hardware separation.

### Permitted verification

Only material unresolved questions may receive a synthetic loss, tiny quantization, small mapper/configuration, small `XyloSim`, or one indispensable representative-checkpoint inspection.

### Blockers

Missing data, version incompatibility, incomplete checkpoint configuration, mapper failure, or hardware dependency.

### Appendix targets

Appendices B, F, and G.

### Completion criterion

Active training, evaluation, conversion, simulation, and deployment paths are traceable and classified.

### `PROGRESS.md`

Record lifecycle, parameters, evidence tiers, commands/blockers, checkpoint justification where applicable, and `Awaiting explicit user authorization for Phase 10.`

### Stopping condition

Stop after the Phase 9 handoff.

---

## 14. Phase 10 — Supporting Scripts, Notebooks, Data, and Results

### Objective

Inventory meaningful supporting material without allowing Priority 3 content to dominate.

### Priority

Priority 2 for important support; Priority 3 for all remaining material.

### Inspect

Every meaningful `.py`, `.sh`, and notebook; included CSV data; `results/`; generated media; and the checkpoint directory as one artifact class.

### Meaningful criteria

Include scripts that are entrypoints, active imports, relevant transformations/models/metrics, active artifact producers/consumers, parameter/shape/constraint sources, or necessary classification evidence.

Include notebooks that exercise core code, supply material overrides, generate relevant results, demonstrate simulation/deployment, provide classification evidence, or provide a distinct learning step.

### Expected evidence

Script and notebook inventories, cell references, classifications, data schema/timing/unit findings, result-to-generator map, and checkpoint collection noted without family analysis.

### Permitted verification

Read-only notebook JSON parsing, CSV sampling, result-schema inspection, and static reference search.

### Blockers

Missing units, uncertain notebook order, unavailable paths, or unlinked results.

### Appendix targets

Appendices D, E, and F.

### Completion criterion

Every meaningful item has an evidence-backed role, classification, importance, and reproducibility status.

### `PROGRESS.md`

Record inventory categories, evidence, blockers, unresolved provenance, and `Awaiting explicit user authorization for Phase 11.`

### Stopping condition

Stop after the Phase 10 handoff.

---

## 15. Phase 11 — Resolve Remaining `Unresolved` Items

### Objective

Resolve only material questions still explicitly marked `Unresolved`.

### Eligibility

A check is allowed only when the question remains unresolved, affects Priority 1 or material Priority 2, static/prior checks did not resolve it, and no successful equivalent check already ran.

### Prohibited

Repeating successful checks, confirmation-only reruns, broad exploration, unrelated encoder tests, or retrying a blocker without changed circumstances and authorization.

### Expected evidence

| Unresolved item | Existing evidence | Material impact | One permitted check | Result/blocker | Final confidence |
|---|---|---|---|---|---|

### Permitted verification

One short CPU-only, non-destructive attempt per eligible item.

### Blockers

Missing package/data, unsafe import, incompatible version, network need, or hardware need.

### Appendix target

Appendix G.

### Completion criterion

Every eligible item is resolved once or remains `Unresolved` with its exact blocker. No successful check is repeated.

### `PROGRESS.md`

Record commands/results, final statuses, skipped repeats, and `Awaiting explicit user authorization for Phase 12.`

### Stopping condition

Stop after the Phase 11 handoff.

---

## 16. Phase 12 — Synthesis, Learning Path, and Report Construction

### Objective

Complete the English report using accumulated evidence.

### Priority

Priority 1 for active paths, Priority 2 for learning guidance, and appendix-only treatment for Priority 3.

### Actions

Complete the main narrative; integrate pipeline/SNN Mermaid graphs; finalize the encoder comparison, provenance, shapes, five-stage learning roadmap, and verified reproduction commands; audit citations, confidence labels, evidence tiers, headings, and checklist compliance; run no new implementation checks.

### Permitted verification

Citation-line, notebook-cell, heading, internal-link, and completion-checklist validation.

### Blockers

Missing mandatory evidence or unresolved contradictions preventing an accurate claim.

### Completion criterion

The report follows the task, cites every material claim, keeps Priority 3 compact, and passes the completion checklist.

### `PROGRESS.md`

Record files written, validation performed, remaining limitations, and completed status.

### Stopping condition

Stop after the final report handoff.

## 17. Required Final Report Structure

```markdown
# Neuromorphic-Gravity Project Summary

## 1. Executive Summary, Repository State, and Project Architecture
## 2. End-to-End Active Data Flow
## 3. Custom Wavelet Mathematics
## 4. Custom Wavelet Implementation and Parameter Provenance
## 5. Rockpool NIMU / IMUIFSim
## 6. Active FilterBank Encoding
## 7. Comparison of the Three Priority Encoder Families
## 8. Encoder Shape Ledgers and SNN Input Contracts
## 9. Primary SNN Architecture and Neuron Dynamics
## 10. Training and Evaluation
## 11. Rockpool, NeuroBench, Quantization, XyloSim, and Hardware Integration
## 12. Supporting Scripts, Important Notebooks, Data, and Results
## 13. Recommended Learning Path and Minimal Reproduction Workflow
## 14. Open Questions, Ambiguities, and Technical Risks
## 15. Key Takeaways

## Appendix A. Important Symbols and Variables
## Appendix B. Complete Parameter Provenance
## Appendix C. Encoder and SNN Shape Ledgers
## Appendix D. Scripts Inventory
## Appendix E. Notebooks Inventory
## Appendix F. File and Function Index
## Appendix G. Verification Commands and Results
## Appendix H. Non-Wavelet Encoder Classification
```

## 18. Final Handoff

Return:

1. The report path.
2. A five-to-ten-line findings summary.
3. Files changed.
4. Limitations from missing dependencies, unavailable hardware, incomplete data, or ambiguous code.
