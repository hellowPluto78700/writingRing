# Neuromorphic-Gravity Project Summary Task

## 1. Mission

Perform a repository-wide, evidence-based analysis and produce:

```text
docs/PROJECT_SUMMARY_REPORT.md
```

The report must help a new researcher understand:

1. The project goal and active data-to-decision pipeline.
2. The custom wavelet mathematics and implementation.
3. Rockpool NIMU / `IMUIFSim`.
4. The active standalone `FilterBank` encoding path.
5. Encoder parameter provenance, shapes, timing, state, and SNN input contracts.
6. The primary SNN, training, evaluation, quantization, simulation, and deployment paths.
7. The role of meaningful supporting scripts, notebooks, data, and results.
8. A practical learning and minimal-reproduction path.

Write the plan and report entirely in English. Preserve source identifiers, class names, function names, command-line flags, equations, tensor dimensions, and paths exactly.

---

## 2. Authority and Required Workflow

The approved execution plan is:

```text
docs/PROJECT_SUMMARY_PLAN.md
```

The persistent investigation log is:

```text
PROGRESS.md
```

Required workflow:

1. Read this task completely.
2. Read `docs/PROJECT_SUMMARY_PLAN.md` completely.
3. Read `PROGRESS.md` before every phase.
4. Confirm that the user explicitly authorized the phase being started.
5. Inspect actual imports, callers, branches, parameter propagation, shapes, and consumers.
6. Complete only the authorized phase.
7. Add detailed phase evidence to the designated appendices of `docs/PROJECT_SUMMARY_REPORT.md`.
8. Append a concise cumulative phase entry to `PROGRESS.md`.
9. Report the phase outcome and stop.
10. Begin no later phase without a new explicit user instruction.
11. Complete the main report narrative only in the final synthesis phase.

The task and approved plan must not be modified during investigation without explicit user instruction.

---

## 3. Repository Safety and Write Boundaries

Do not modify:

- source code;
- notebooks;
- datasets;
- checkpoints;
- environment files;
- existing results;
- generated experiment media.

Do not:

- install or upgrade packages;
- modify an environment;
- download data;
- contact W&B;
- run training;
- run full dataset preprocessing;
- execute notebooks wholesale;
- use a GPU;
- discover, configure, or access physical Xylo hardware.

Files permitted to be created or updated during the investigation:

- `docs/PROJECT_SUMMARY_REPORT.md`
- `PROGRESS.md`

Treat `docs/PROJECT_SUMMARY_PLAN.md` as immutable after activation.

During investigation phases, detailed evidence belongs in report appendices. Keep the main report narrative for the final synthesis phase.

Before every write:

1. Confirm the target is permitted.
2. Preserve pre-existing content and changes.
3. Append rather than erase cumulative investigation history.

---

## 4. Analysis Priorities

### Priority 1 — Exhaustive

- Active preprocessing and windowing paths.
- `python-pipeline/wavelets.py`.
- `python-pipeline/modules.py`.
- Custom wavelet encoding.
- Rockpool NIMU / `IMUIFSim`.
- Active standalone `FilterBank` encoding.
- Encoder parameter provenance.
- Encoder shapes and SNN input contracts.
- Primary SNN architecture and neuron dynamics.
- Active training, evaluation, simulation, and deployment paths.

### Priority 2 — Supporting Analysis

- NeuroBench.
- Quantization.
- `XyloSim`.
- Hardware-only code.
- Supporting mathematical and dataset utilities.
- Important experiment notebooks.
- Alternative models relevant to understanding or validating an active path.

### Priority 3 — Classification or Inventory Only

- Unrelated encoders.
- Legacy tutorials.
- MNIST examples.
- Unused model variants.
- Historical results.
- Generated media.
- Checkpoint collections.

Priority 3 material must not consume substantial main-report space.

---

## 5. Required Repository Areas

Prioritize:

```text
README.md
preprocessing/
python-pipeline/
snn/
notebooks/
xylo-imu/
data/
results/
environment.yml
environment.original.yml
```

Highest-priority implementation files include:

```text
preprocessing/preprocess.py
preprocessing/prepare_data.py
preprocessing/spike_encoders.py
preprocessing/spike_encoder_neurobench_utils.py
preprocessing/spike_encoder_neurobench_utils_xylo_nimu.py
python-pipeline/modules.py
python-pipeline/wavelets.py
python-pipeline/utils.py
snn/har_snn.py
snn/utils_architectures.py
snn/utils_datasets.py
snn/utils_losses.py
snn/utils_parser.py
snn/utils_run.py
snn/rockpool_nn_networks_synnet.py
snn/rockpool_nn_modules_torch_lif_torch.py
notebooks/rockpool_transform_quantize_methods.py
```

Do not infer a file’s role from its filename. Establish its role from entrypoints, imports, callers, branches, notebook usage, outputs, and consumers.

---

## 6. Evidence and Confidence Rules

### 6.1 Confidence labels

Use:

- `Confirmed`
- `Strongly inferred`
- `Unresolved`

Do not fabricate missing units, sampling rates, formulas, model sizes, hardware behavior, or experiment intent.

When code, notebooks, installed packages, and documentation disagree, report the discrepancy.

### 6.2 Repository citations

Python and shell evidence:

```text
`path/to/file.py:L10-L25`
```

Notebook evidence:

```text
`path/to/notebook.ipynb`, cell N
```

For a call chain, cite caller and callee.

Clearly state when notebook execution order is uncertain. Saved notebook outputs are historical evidence, not proof of current reproducibility.

### 6.3 External-dependency evidence tiers

Every claim about Rockpool, spikify, NeuroBench, Samna, or XyloSim must identify its evidence tier.

| Tier | Evidence | Appropriate use |
|---|---|---|
| E1 | Repository source and call sites | Repository-supplied arguments, transformations, assumptions, and consumers |
| E2 | Installed source at the exact runtime package version | Exact dependency implementation available in the environment |
| E3 | Minimal runtime introspection | Observed defaults, attributes, shapes, state, dtype, or behavior |
| E4 | Official version-relevant documentation | Intended public behavior not settled by E1–E3 |

Rules:

- Separate repository behavior from dependency-internal behavior.
- Prefer E1 for claims about repository behavior.
- Prefer exact installed source over generic documentation for dependency internals.
- Use runtime introspection only to resolve material static ambiguity.
- Use official documentation only for a material unresolved public contract.
- Documentation cannot establish exact installed behavior when the package is absent.
- Hardware behavior unavailable without hardware remains `Unresolved`.
- Record exact package-version provenance whenever E2–E4 is used.

---

## 7. Execution-Path Classification

Classify every important entrypoint, branch, module, and notebook path.

| Classification | Required evidence |
|---|---|
| Active production path | Reachable from a current CLI or main entrypoint, produces artifacts consumed by active training/evaluation, and needs no notebook-only intervention |
| Active optional path | Reachable through a current supported flag or configuration branch and has an identifiable downstream consumer |
| Experimental path | Notebook-only, ad hoc, exploratory, or hard-coded experiment without a confirmed active CLI-to-consumer flow |
| Legacy path | Superseded tutorial/example, obsolete import/API/path, or implementation no longer used by active callers |
| Dead or unexecuted code | Defined or constructed but unreachable, never called, omitted from the executed module container, or caller-less after repository-wide search |

A filename, comment, or variable name alone is insufficient evidence.

Use `Unresolved` when reachability or usage cannot be established.

---

## 8. Active End-to-End Data Flow

Build a call graph before explaining implementation details:

```text
raw input
-> subject/data selection
-> resampling
-> gravity handling / coordinate transformation
-> wavelet or filter-bank transform
-> event/spike generation
-> windowing
-> dataset loading and transforms
-> batching
-> SNN forward pass
-> loss and optimization
-> evaluation
-> Rockpool graph conversion
-> quantization
-> Xylo simulation
-> hardware-only execution
```

Use a Mermaid flowchart.

Separate:

- active production;
- active optional;
- experimental;
- legacy;
- dead/unexecuted;
- software simulation;
- hardware-only paths.

Identify the producer, intermediate artifact, consumer, branch condition, and confidence for every edge.

---

## 9. Detailed Encoder Scope

Detailed analysis is required only for:

1. The custom wavelet-based pipeline.
2. Rockpool NIMU / `IMUIFSim`.
3. The active standalone `FilterBank`-based pipeline.

For each, trace:

- mathematical basis;
- implementation and call chain;
- parameter provenance;
- frequency and filter construction;
- quantization and thresholds;
- axis and channel ordering;
- tensor shapes;
- timing, causality, and state;
- event or spike output;
- SNN input relationship;
- software and Xylo relevance.

### 9.1 Custom wavelet mathematics

Inspect:

```text
python-pipeline/wavelets.py
python-pipeline/modules.py
python-pipeline/utils.py
preprocessing/preprocess.py
```

Trace:

- `powerWavelet()`
- `accelerationWavelet()`
- `velocityWavelet()`
- `WaveletFIRFilterModule`
- `WaveletIIRFilterModule`
- `convm()`
- `prony()`
- `butter()`
- `MaxFilterModule`
- relevant preprocessing callers

Explain:

1. Exact repository equations.
2. Compact support and normalization.
3. Scale, frequency, sampling rate, and kernel length.
4. FIR and IIR coefficient construction.
5. Boundary behavior, padding, delay, causality, and state.
6. Axis and band processing.
7. Output semantics and value range.
8. Whether the executed path produces filtered values, events, or spikes.

Use relevant notebook cells only as experiment-specific evidence.

### 9.2 Custom wavelet implementation and provenance

Trace:

```text
caller value
-> constructor
-> stored or derived state
-> numerical operation
-> output
-> persisted column/window
-> SNN consumer
```

Inspect:

- `NeuromorphicIMUPipeline`
- `QuantizerModule`
- `LocalToGlobalModule`
- `AccelToPowerModule`
- `WaveletFIRFilterModule`
- `WaveletIIRFilterModule`
- `MaxFilterModule`
- `AbsModule`

Identify constructed but unexecuted modules and parameters.

### 9.3 Rockpool NIMU / `IMUIFSim`

Inspect repository calls to:

- `Quantizer`
- `RotationRemoval`
- Rockpool IMU-interface `FilterBank`
- `IMUIFSim`
- configuration export helpers

Separate:

- values supplied by this repository;
- dependency defaults;
- installed implementation behavior;
- runtime observations;
- official hardware/documentation claims.

Trace input clipping, quantization, sampling, filters, thresholds, state, reshape order, output dtype/range, and Xylo relationship.

### 9.4 Active standalone `FilterBank`

Trace:

- exact import source;
- constructor and arguments;
- frequency-range and center-frequency construction;
- filter order and state;
- `decompose()` input/output;
- axis and band ordering;
- reshape and persistence;
- downstream encoder boundary;
- SNN consumer.

Detailed downstream encoder analysis stops once the FilterBank output contract and downstream interface are established.

### 9.5 Other encoders

For every encoder outside the three detailed families:

1. Locate definition/import.
2. Locate relevant call sites.
3. Determine whether it uses a wavelet-derived operation.
4. Cite decisive evidence.
5. Assign a one-sentence category.
6. Stop.

Do not perform runtime testing, reconstruction analysis, detailed mathematics, or comprehensive parameter tracing for these encoders.

---

## 10. Parameter Provenance

For every important parameter in the three detailed encoder families, create:

| Variable / parameter | Defined in | Supplied by | Actual/default value | Units | Shape/type | Derivation | Used by | Effect | Evidence tier | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|

Cover:

- input axes/channels;
- sampling frequency and `dt`;
- frequency bounds and lists;
- bands/filters;
- center frequencies and scales;
- kernel/support length;
- FIR/IIR coefficients and order;
- quantization scale and bits;
- thresholds;
- reset and state behavior;
- polarity handling;
- downsampling;
- window duration and stride;
- batch size;
- encoder output dimensions;
- SNN input dimensions;
- hardware limits.

Distinguish:

- hard-coded constants;
- function defaults;
- CLI arguments;
- notebook overrides;
- data-derived values;
- learned values;
- checkpoint-loaded values;
- hardware constraints.

Do not label a value as the repository default until its active caller is confirmed.

---

## 11. Shape Ledgers and SNN Input Contracts

Maintain separate ledgers for:

- custom wavelet encoding;
- NIMU / `IMUIFSim`;
- active standalone `FilterBank`;
- each corresponding SNN input path.

Use:

| Stage | Shape | Dimension source | Axis order | Reshape order | Dtype | Value range | Consumer | Evidence tier | Confidence |
|---|---|---|---|---|---|---|---|---|---|

Independently derive:

- sampling frequency;
- window duration;
- time steps;
- axes;
- bands;
- polarity expansion;
- channel count;
- downsampling and padding;
- batch/time/channel order;
- `inputSize`.

Do not assume `64 Hz`, 10 seconds, `640`, five bands, `15`, or `30`.

Include one concrete shape walkthrough for each detailed encoder.

---

## 12. Primary SNN Analysis

Inspect:

```text
snn/har_snn.py
snn/utils_parser.py
snn/utils_architectures.py
snn/utils_datasets.py
snn/utils_losses.py
snn/utils_run.py
snn/rockpool_nn_networks_synnet.py
snn/rockpool_nn_modules_torch_lif_torch.py
```

Determine the primary architecture through call and consumer evidence.

For the primary architecture, report:

| Stage | Class/module | Input shape | Output shape | Neurons | Recurrent? | Parameters | Dynamics | Source |
|---|---|---|---|---|---|---|---|---|

Explain:

- encoder-derived `inputSize`;
- hidden and output sizes;
- LIF/state equations;
- synaptic and membrane time constants;
- threshold, reset, bias, and recurrence;
- surrogate gradient;
- state initialization and persistence;
- batch/time/channel order;
- spike or membrane output;
- class decision rule;
- training, inference, simulation, and hardware differences.

Relevant alternatives receive Priority 2 analysis. Unused, tutorial, or broken alternatives receive Priority 3 classification.

Use a Mermaid architecture diagram and tensor-shape walkthrough.

---

## 13. Training and Evaluation

Trace:

- dataset construction;
- subject splitting;
- class balancing;
- window labels;
- batching;
- transforms;
- forward pass;
- active loss;
- surrogate gradients;
- optimizer and learning rates;
- epoch loop;
- evaluation metrics;
- confusion matrices;
- NeuroBench;
- W&B call sites;
- checkpoint save/load flow only as needed to explain active execution.

Do not contact W&B.

Do not inventory checkpoint families.

Inspect at most one checkpoint, only when indispensable to validate an active SNN, quantization, simulation, or deployment path. Record why it was needed, the safe loading method, relevant keys/shapes, and limitations.

---

## 14. Rockpool, Quantization, XyloSim, and Hardware

Trace:

```text
trained/loaded model
-> as_graph()
-> mapper()
-> quantization
-> config_from_specification()
-> XyloSim
-> hardware-only interfaces
```

Explain:

- graph transformations;
- global or channel quantization;
- integer ranges and clipping;
- thresholds and dashes;
- configuration validation;
- input/neuron/output constraints;
- simulation timing;
- train-time versus deployable-model mismatches;
- Samna and hardware-only boundaries.

Do not call hardware discovery or hardware interfaces.

---

## 15. Meaningful Supporting Material

### 15.1 Meaningful scripts

A script is meaningful when it:

- is a CLI or main entrypoint;
- is imported by an active or optional path;
- defines a relevant transformation, model, loss, metric, quantizer, simulator, or deployment helper;
- produces or consumes an active artifact;
- supplies a relevant parameter, shape, or constraint;
- is necessary to classify a competing, legacy, or dead path.

Empty initializers, generated artifacts, pure data, and unreferenced duplicates without distinct behavior are not meaningful scripts.

Create:

| File | Role | Entrypoint | Inputs | Outputs | Calls/imports | Called by | Hardware? | Path classification | Importance |
|---|---|---|---|---|---|---|---|---|---|

### 15.2 Meaningful notebooks

A notebook is meaningful when it:

- imports or exercises core repository code;
- supplies a parameter override needed to interpret an experiment;
- generates a tracked result discussed in the report;
- demonstrates quantization, simulation, deployment, or hardware behavior;
- supplies path-classification evidence;
- provides a distinct learning or reproduction step.

Create:

| Notebook | Main question | Key cells | Inputs | Outputs | Core reuse | Hardware? | Path classification | Reading order |
|---|---|---|---|---|---|---|---|---|

Inspect notebook JSON source and cell order. Do not rely on filenames or saved output alone.

Treat `notebooks/media/` as generated artifacts. Reclassify an item only when repository call evidence establishes that it is source material.

---

## 16. Data, Results, and Checkpoint Artifacts

For included CSV data, determine where possible:

- columns;
- units;
- sampling rate;
- duration;
- left/right relationship;
- coordinate frame;
- consumers.

For `results/`:

- connect artifacts to generating scripts/notebooks;
- inspect schemas and filenames;
- avoid reverse-engineering every PDF;
- classify historical outputs as Priority 3.

Treat `snn/models/` as one artifact collection. Do not inventory or classify checkpoint families.

---

## 17. Runtime Verification Policy

Use static analysis first.

A runtime check requires:

- a named question still marked `Unresolved`;
- material impact on Priority 1 or significant Priority 2 analysis;
- no earlier successful equivalent check;
- a clear expected observation.

Every check must be:

- short;
- deterministic or explicitly seeded;
- CPU-only;
- non-destructive;
- free of downloads, training, environment mutation, W&B, notebook-wide execution, and hardware access.

For a blocked check:

1. Make one safe attempt.
2. Record the exact command.
3. Record the exact blocker.
4. Keep the conclusion `Unresolved`.
5. Continue unaffected work.

Do not repeat successful checks.

The final verification phase may address only unresolved items remaining after earlier phases.

Record every exact command and a concise result in `PROGRESS.md`. Put detailed results in Appendix G.

---

## 18. Required Report Structure

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

Priority 3 content belongs primarily in appendices.

Do not include a checkpoint-family section or appendix.

---

## 19. Learning Roadmap

Provide five stages:

1. Understand the problem and data.
2. Understand custom wavelets and the three detailed encoders.
3. Understand the primary SNN.
4. Understand training and metrics.
5. Understand simulation and Xylo deployment.

For each stage include:

- goal;
- files to read;
- verified command or notebook;
- expected output;
- common misunderstanding;
- completion criterion.

---

## 20. Minimal Reproduction Workflow

Provide verified commands for:

1. Activating the intended Conda environment.
2. Checking critical imports and versions.
3. Inspecting a sample data file.
4. Running a minimal detailed-encoder example.
5. Running a minimal primary-SNN forward pass.
6. Running one evaluation or metric path without retraining.
7. Running a software simulation alternative where available.

Do not invent commands. Confirm CLI flags from parser source or safe `--help` output.

Keep hardware-only commands separate and clearly labeled as not executed.

---

## 21. Progress Log Requirements

Keep `PROGRESS.md` concise and cumulative.

For every phase record:

- session/date;
- files inspected;
- major call paths confirmed;
- major parameters traced;
- exact verification commands and concise results;
- short conclusions;
- unresolved questions;
- next authorized action.

Do not copy complete provenance tables, shape ledgers, or long command output into `PROGRESS.md`.

Do not erase earlier history.

---

## 22. Phase-Gate Requirement

After every completed phase:

1. Update report appendices.
2. Update `PROGRESS.md`.
3. Report the phase outcome.
4. Stop.

No phase begins without explicit user instruction naming that phase.

---

## 23. Completion Checklist

Before final handoff, verify:

- repository state and commit;
- environment provenance;
- classified end-to-end call graph;
- active/optional/experimental/legacy/dead distinctions;
- custom wavelet equations tied to code;
- complete detailed-encoder provenance;
- three separate encoder shape ledgers;
- SNN input contracts;
- three-encoder comparison;
- primary SNN architecture and dynamics;
- training, evaluation, and surrogate-gradient flow;
- NeuroBench distinction;
- quantization and `XyloSim` path;
- software/hardware separation;
- meaningful script inventory;
- meaningful notebook inventory with cell references;
- data/result overview;
- non-wavelet encoder classification;
- five-stage learning roadmap;
- verified minimal reproduction commands;
- verification appendix;
- open questions and technical risks;
- citations for every material claim.

Final chat handoff must include:

1. Report path.
2. Five-to-ten-line findings summary.
3. Files changed.
4. Limitations caused by missing dependencies, unavailable hardware, incomplete data, or ambiguous code.
