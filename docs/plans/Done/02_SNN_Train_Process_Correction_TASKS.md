# Action0 SNN Training Process Correction — Task DAG and TaskSpecs

## Plan identity

- **Source plan:** `docs/plans/Done/02_SNN_Train_Process_Correction_Plan.md`
- **Baseline commit:** `29da4396b60d9357e0b8edffdb6f02ec4011a987`
- **Primary durable contract:** `docs/notes/ACTION0_SNN_TRAINING.md`

## Dependency DAG

```text
T001 Producer/trainer contract hardening
  ↓
T002 Engine and dry-run correctness
  ↓
T003 Checkpoint contract
  ├── T004 Legacy /snn scope probe
  │     └── T004B Legacy portability (conditional)
  └── T005 Regression/runtime verification
        (waits for T004B if required)
  ↓
T006 Documentation and closure
```

| Task | Goal | Dependencies | State |
| --- | --- | --- | --- |
| T001 | Producer/trainer contract hardening | — | DONE |
| T002 | Engine and dry-run correctness | T001 | DONE |
| T003 | Checkpoint contract | T002 | DONE |
| T004 | Legacy `/snn` scope decision | T003 | DONE — Scope B |
| T004B | Legacy portability | T004 | DONE |
| T005 | Regression/runtime verification | T003, T004B | DONE |
| T006 | Documentation and plan closure | T005 | DONE |

## T001 — FROZEN TaskSpec

- **Goal:** Make `segmentation_padded/padding_dataset_summary.json` an
  explicit Action0 runtime contract without changing producers or SynNet.
- **Source:** plan §14; **dependencies:** none; **validated against:**
  `29da4396b60d9357e0b8edffdb6f02ec4011a987`.
- **Required behavior:** load metadata before class discovery/dataset creation;
  require `input_kind=spike-imu`, canonical feature schema, `channel_count=21`,
  positive integer `target_length`, finite positive `sampling_rate_hz`, and
  `padding_side=right`; require CLI rate equality at `atol=1e-12`, `rtol=0`;
  require every split padded length to equal producer target and one another.
- **Diagnostics:** retain counters only as validated diagnostic/provenance
  fields; do not make Board/label count values blocking or compare relocated
  absolute roots.
- **Contracts to preserve:** 21-source/15-model slice, label boundary,
  producer no-repadding behavior, SynNet dynamics, and producer implementation.
- **Allowed writes:** `snn/action0_dataset.py`, `snn/train_action0.py`,
  `tests/test_snn_action0_dataset.py`, `tests/test_snn_action0_smoke.py`.
- **Forbidden writes:** `src/writingring/**`, `scripts/action0_pipeline/**`,
  `snn/utils_architectures.py`, vendor, sample data, README, notes, configs.
- **Acceptance:** valid/missing/malformed/wrong metadata tests; rate,
  target-length, and cross-split failures; valid existing producer package
  remains trainable; pytest evidence and fresh verifier PASS.
- **Validation command:** `conda run -n writingring-viz python -m pytest -q
  tests/test_snn_action0_dataset.py tests/test_snn_action0_smoke.py`.
- **Replan triggers:** a required producer change, rate-semantic conflict, or
  import architecture cycle.

### T001 execution and verification

Worker implemented producer-metadata validation, sampling-rate provenance
checks, and split-wide target-length validation in the frozen paths. Fresh
verification passed: focused contract, related SNN, and producer/pipeline tests
all passed; producer/runtime behavior and SynNet dynamics remain unchanged.

## T002 — Draft TaskSpec

- **Goal:** Harden exact model output validation, valid-timestep epoch-loss
  reporting, and deterministic same-batch dry-run behavior.
- **Source plan:** §15; **dependencies:** T001 (DONE).
- **Probe scope:** Action0 engine/loss/trainer/parser and focused tests only.
- **Anticipated write paths:** `snn/action0_engine.py`,
  `snn/action0_losses.py`, `snn/train_action0.py`, and focused
  `tests/test_snn_action0_*.py` as confirmed by the probe.
- **Contracts to preserve:** SynNet state advances through padding, masked
  optimization semantics, architecture/shifts, and one optimizer step in dry
  run.
- **Replan triggers:** output/loss semantics require a baseline redesign or
  exact same-batch flow cannot be introduced within this scope.

## T002 — FROZEN TaskSpec

- **Goal:** Enforce exact output classes, report global valid-timestep loss,
  and make dry-run inspect/optimize one batch.
- **Source:** plan §15; **dependencies:** T001 (DONE); **validated against:**
  baseline plus T001 working tree.
- **Required behavior:** engine takes expected class count and rejects any
  nonexact output dimension; epoch loss accumulates scalar batch objective by
  valid mask steps then divides globally; dry-run sends its inspected batch via
  a singleton iterable through the existing training path for exactly one step.
- **Contracts to preserve:** masked CE/regularization formula, per-segment
  spike metrics, padding state evolution, SynNet architecture/dynamics, and
  normal train behavior.
- **Allowed writes:** `snn/action0_engine.py`, `snn/train_action0.py`,
  `tests/test_snn_action0_loss.py`, `tests/test_snn_action0_smoke.py`.
- **Forbidden:** losses, producer, utils_architectures, legacy runners, docs,
  README, vendor, sample data, configs.
- **Acceptance:** wrong class output fails; uneven masks (with regularization)
  yield valid-step weighted epoch loss; dry-run uses one batch/one step; fresh
  verifier PASS.
- **Validation:** focused Action0 loss/smoke tests under `writingring-viz`.
- **Replan triggers:** required loss API redesign or another Action0 caller
  outside the frozen scope.

### T002 execution and verification

Worker implemented the frozen engine/dry-run changes. Fresh verification
passed focused and all Action0 tests, preserving masked dynamics and metrics.

## T003 — Draft TaskSpec

- **Goal:** Define fail-fast checkpoint compatibility and restore semantics
  without claiming unsupported exact resume.
- **Source plan:** §16; **dependencies:** T002 (DONE).
- **Probe scope:** Action0 trainer/parser/checkpoint paths and focused tests.
- **Required decision:** distinguish best-model loading from exact resume;
  identify stable compatibility fields and safe validation ordering.
- **Anticipated write paths:** `snn/train_action0.py`, parser if required, and
  focused Action0 checkpoint tests.
- **Replan triggers:** checkpoint semantics require architecture/product policy
  beyond compatibility validation.

### T003 primary replan decision

The plan's preferred low-risk model is frozen as the design direction to
re-probe: `--model_checkpoint` is a **configuration-compatible model restore**
for a new training run. It loads model weights only; it does not load optimizer
state, restore epoch/best metric, restore RNG or DataLoader state, or claim
trajectory/deterministic resume. `num_epochs` is the requested epoch count for
the new run. Learning rate, spike regularization, and random seed are new-run
controls and may differ. The fixed data/model compatibility fields, including
user splits, must match. Newly written checkpoints carry a schema version;
unversioned or incompatible checkpoint formats fail explicitly rather than
being inferred.

The re-probe must verify that this direction fits all existing Action0 callers
and requires no write outside Action0 trainer/parser/tests.

## T003 — FROZEN TaskSpec

- **Goal:** Enforce schema-v1 configuration-compatible Action0 model restore.
- **Dependencies:** T002 (DONE); **validated against:** baseline plus T001/T002
  working tree.
- **Required behavior:** newly written payloads include
  `checkpoint_schema_version=1`; load rejects missing/unknown schema and
  validates all fixed fields before model weights are applied: variant,
  boundary, class mapping, input slice/counts, topology/shifts, sample rate,
  and ordered train/validation/test user lists. Error messages name the field,
  checkpoint value, and requested value.
- **Restore semantics:** restore model weights only for a new training run.
  Do not load optimizer/epoch/best/RNG/shuffle state. `num_epochs`, learning
  rate, regularization, and seed are new-run settings and may differ. Existing
  optimizer/epoch/best fields are removed from new schema-v1 payloads.
- **Allowed writes:** `snn/train_action0.py`,
  `tests/test_snn_action0_checkpoint.py` (new), and existing focused Action0
  checkpoint tests only if necessary.
- **Forbidden:** parser, producer, architecture, legacy runners, docs, README,
  vendor, sample data, configs.
- **Acceptance:** valid restore; all field/schema/legacy failures; validation
  before state application; changed new-run controls allowed; fresh verifier
  PASS.
- **Validation:** focused checkpoint plus existing Action0 test suite.
- **Replan triggers:** unexpected external checkpoint consumer or requirement
  for optimizer/trajectory compatibility.

### T003 execution and verification

Worker implemented schema-v1 model-only restore in the frozen scope. Fresh
verification passed all checkpoint compatibility/ordering tests and confirmed
that no optimizer or trajectory state is persisted/restored.

## T004 — Draft TaskSpec

- **Goal:** Resolve whether historical absolute paths are an Action0-only or
  whole-`/snn` portability requirement, without changing legacy code before
  that decision.
- **Source plan:** §17; **dependencies:** T003 (DONE).
- **Probe scope:** plan/Check0 acceptance wording, Action0 paths, and legacy
  `/snn` absolute-path inventory only; no code edits.
- **Acceptance:** evidence-backed scope outcome and a determination whether
  conditional T004B is required.
- **Replan trigger:** scope cannot be derived from plan/contracts and needs a
  user product decision.

### T004 probe result

Repository evidence cannot resolve Check0 acceptance [22]. Action0 is portable
and contains no old-machine literal (Scope A), while active legacy HAR code
contains old-machine paths (Scope B requires conditional T004B and its own
frozen portability scope). Await explicit Scope A or Scope B before freezing.

## T004 — FROZEN TaskSpec and execution record

- **Decision authority:** user-selected Scope B.
- **Goal:** Reconcile Check0 [22] as applying to the entire `snn/` tree and
  activate T004B.
- **Dependencies:** T003 (DONE); **basis:** T004 probe plus explicit user
  decision.
- **Required behavior:** no implementation changes in T004 itself; record that
  active `snn/har_snn.py` old-machine paths require separately
  probed/frozen T004B. Commented/generic absolute paths require T004B probe
  classification.
- **Allowed writes:** this task file and WORKBOARD only.
- **Forbidden writes:** all legacy implementation paths until T004B freezes;
  producer, notes, README, vendor, and data paths.
- **Acceptance:** independent verifier confirms Scope B recording and T004B
  activation without unapproved legacy edits.

## T004B — Draft TaskSpec

- **Goal:** Remove old-machine absolute paths from the legacy `snn/` scope
  while preserving HAR training behavior.
- **Source plan:** §17 Scope B; **dependencies:** T004 (DONE).
- **Probe scope:** active paths/imports/CLI/configuration in `har_snn.py`,
  `utils_run.py`, `utils_parser.py`, and relevant tests; classify commented
  and generic absolute paths separately.
- **Required preservation:** legacy data transforms, topology, loss, metrics,
  network selection, and schedule.
- **Replan triggers:** undocumented deployment assumptions, packaging/import
  architecture issue, or old-launch compatibility requirement.

## T004B — FROZEN TaskSpec

- **Goal:** Remove active and textual old-machine paths from the legacy HAR
  path without changing its data/model behavior.
- **Dependencies:** T004 Scope B (DONE); **validated against:** baseline.
- **Required behavior:** add required typed `--data-root` and `--model-root`
  CLI arguments; use `Path` composition for the existing dataset suffix
  branches and checkpoint filenames; retain all suffix selection and training
  semantics. Prefer package-relative imports with direct-script fallback.
- **Textual cleanup:** remove/reword the commented `/home` literal in
  `utils_run.py`. Generic `/tmp/data/mnist` outside this scope remains intact.
- **Contracts to preserve:** HAR transforms, topology, loss, metrics, network
  selection, schedule, W&B behavior, and checkpoint fine-tuning behavior.
- **Allowed writes:** `snn/har_snn.py`, `snn/utils_run.py`,
  `snn/utils_parser.py`, `tests/test_snn_legacy_portability.py` (new).
- **Forbidden:** Action0 modules, utility algorithm modules, docs/README,
  vendor, sample data, configs, and MNIST path handling.
- **Acceptance:** no `/home` or `/work` literal in scoped files; configurable
  root composition preserves suffix/file naming; package import and direct
  script fallback remain testable without optional training dependencies.
- **Validation:** targeted portability tests and textual scan.
- **Replan triggers:** dependencies make import testing impossible within
  scope, direct launch compatibility fails, or root arguments alter legacy
  data semantics.

### T004B execution and verification

Worker implemented the frozen root/import cleanup. Fresh verifier passed
scoped path/import tests and confirmed all legacy HAR behavior remained intact.

## T005 — Draft TaskSpec

- **Goal:** Independently validate completed Action0 and Scope-B legacy work;
  introduce no new behavior.
- **Source plan:** §18; **dependencies:** T003 and T004B (DONE).
- **Probe scope:** runtime availability, targeted/full test commands, real
  producer dry run feasibility, variant resolution, and tiny-overfit
  prerequisites.
- **Anticipated write paths:** task/workboard only unless a verification-only
  fixture is separately frozen.
- **Blocker triggers:** unavailable Python 3.11 `writingring-viz` runtime,
  unavailable real producer artifact, or a failing regression.

### T005 probe result

The required `writingring-viz` runtime reports Python 3.10.20 while
`environment.yml` and repository policy require Python 3.11. That made the
initial probe BLOCKED. The user subsequently authorized a separate,
non-mutating `writingring-test` environment solely to supply the Python 3.11
validation gate; its re-probe is now in progress. `writingring-viz` must not
be changed as part of T005.
Diagnostic-only 3.10 evidence passed: targeted Action0 matrix 26 tests,
supplemental checkpoint/legacy matrix 29 tests, full suite 502 passed/1 skipped,
and a real lowpass padded-producer dry run. The real lowpass artifact and four
synthetic variant resolutions exist; tiny overfit has no current harness and
remains optional absent a separately frozen fixture task.

## T005 — FROZEN TaskSpec

- **Goal:** Independently validate the completed Action0 and Scope-B legacy
  work under the repository-required Python 3.11 runtime, without changing
  implementation behavior or the existing `writingring-viz` environment.
- **Source:** plan §18; **dependencies:** T003 and T004B (DONE);
  **validated against:** baseline plus the current T001–T004B working tree and
  `writingring-test` Python 3.11.15 re-probe.
- **Environment contract:** use the user-authorized isolated
  `writingring-test` Conda environment. It must remain Python 3.11.x, import
  the current editable repository, pass `pip check`, and retain the mirrored
  direct pins: Torch 2.5.1+cpu, snnTorch 1.0.0, Rockpool 2.9.1, NumPy 2.2.6,
  Pandas 2.3.3, Matplotlib 3.10.9, Pytest 9.1.1, SciPy 1.15.3, and
  compress-pickle 2.1.0. Do not modify `writingring-viz`.
- **Required behavior:** record the Python gate; run the plan §18.2 targeted
  Action0 suite; run supplemental checkpoint and legacy-portability suites;
  run the full suite; and run the §18.5 lowpass/label/segmentation_padded
  producer-backed dry run with sample frequency 200, users
  `user_0`/`user_1`/`user_2`, batch size 1, and `--dry_run`.
- **Evidence to record:** passed/failed/skipped counts, Torch and snnTorch
  execution, any known full-suite skip with reason, all dry-run contract
  fields (variant, boundary, root, producer/model dimensions and rate, shifts,
  beta, output, finite loss, spikes, and one-step evidence), and the four
  synthetic variant resolutions. Tiny overfit remains optional; no harness is
  authorized by this TaskSpec.
- **Allowed writes:** task/workboard/plan and durable documentation only.
  **Forbidden writes:** all implementation, test, configuration, vendor,
  sample-data, and `writingring-viz` environment paths.
- **Acceptance:** Python 3.11 gate and imports pass; no Action0 failure or
  skip; supplemental suites pass; full-suite failures attributable to this
  plan are absent (the pre-existing unavailable-real-sample skip is recorded);
  real lowpass dry run demonstrates every required field; fresh verifier PASS.
- **Validation commands:** `conda run --no-capture-output -n writingring-test
  python --version`; the frozen targeted/supplemental/full pytest commands;
  and the frozen `python -m snn.train_action0 ... --dry_run` command.
- **Replan triggers:** any implementation regression, Python 3.11-specific
  incompatibility, dependency-resolution/version-induced behavior change, or
  unavailable real producer artifact.

### T005 execution and verification

PRIMARY ran the frozen matrix in the user-authorized `writingring-test`
environment (Python 3.11.15), leaving `writingring-viz` untouched. Targeted
Action0: 26 passed/0 skipped; checkpoint plus legacy portability: 29 passed/0
skipped; full suite: 502 passed/1 skipped because the real
`user_0/action_0/dataset_0` SpikeIMU artifact is unavailable. The real lowpass
producer dry run validated the label padded root, 21 producer channels to the
15-channel model slice, target length 1024, 200 Hz rate, shifts 2/1, beta 0.5,
52-class output, finite loss, valid spike statistics, and the tested one-step
path. Fresh verifier PASS found no Python 3.11, dependency-resolution, or
implementation regression. Tiny overfit remains optional and was not run.

## T006 — FROZEN TaskSpec

- **Goal:** Reconcile durable documentation with the independently verified
  T001–T005 behavior and close the SNN correction plan.
- **Source:** plan §19; **dependencies:** T005 verifier PASS; **validated
  against:** baseline plus verified T001–T005 working tree and
  `writingring-test` Python 3.11.15 evidence.
- **Required documentation:** update `docs/notes/ACTION0_SNN_TRAINING.md` for
  producer summary/rate/T_pad validation, same-batch one-step dry-run
  semantics, exact class validation, global valid-timestep epoch loss, and
  schema-v1 model-only checkpoint restore. Annotate historical Check0 only to
  redirect stale `segmentation/` training guidance to
  `segmentation_padded/` and record the Scope-B `[22]` result. Update this
  task record, plan status, and WORKBOARD with T005 evidence and closure.
- **Facts and limits:** rate equality is producer provenance/configuration,
  not a universal acquisition-rate claim; dry-run previews then optimizes the
  same batch and is not a single-forward claim; checkpoint restore is not
  optimizer/trajectory/RNG/DataLoader resume; full suite has one known missing
  real-artifact skip. README remains correct and must not change; do not
  document the transient test environment there.
- **Allowed writes:** `docs/notes/ACTION0_SNN_TRAINING.md`,
  `docs/plans/Done/02_SNN_Train_Process_Correction_Plan.md`, this task file,
  `docs/plans/TODO/Check0_Ring_Action_Segment_SNN_Train_Code_Transform_Plan.md`,
  and `docs/plans/WORKBOARD.md`; completed-plan relocation is allowed only if
  all plan/task/workboard pointers stay synchronized.
- **Forbidden writes:** README, all implementation/tests/configuration,
  vendor, sample data, environments, and unrelated plans/notes.
- **Acceptance:** all claims trace to T001–T005 verifier-approved facts;
  stale historical guidance is explicitly labelled; Scope B and required root
  CLI are recorded; final status/pointers are internally consistent; fresh
  verifier PASS.
- **Validation:** `git diff --check`; documentation link/path search; fresh
  verifier independently checks claims against the frozen TaskSpec and T005
  evidence.
- **Replan triggers:** a documentation claim needs unverified behavior, an
  affected user-facing entry point contradicts README stability, or moving the
  plan leaves stale pointers.

### T006 execution and verification

PRIMARY updated only the frozen documentation paths. The first verifier found
stale lifecycle wording only; after PRIMARY synchronized plan/task/workboard
state, a fresh verifier PASS confirmed all contract claims and limitations,
Scope-B reconciliation, unchanged README, and clean documentation checks. The
companion plan and this TaskSpec are closed under `docs/plans/Done/`.
