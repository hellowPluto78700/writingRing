# Fixed User Exclusion for Acceleration Reconstruction — Frozen TaskSpecs

**Lifecycle:** COMPLETE.  The plan in
`10_spike_user_exlude_refine_plan.md` supplied the task ordering. Each
dependency-ready task was probed, frozen by PRIMARY, implemented only within
its allowed scope, and independently verified before dependents started.

## T1 — Config contract

- **Status:** VERIFIED PASS
- **Dependencies:** none.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `snn/accel_reconstruction_eval/config.py` and a new
  focused test module under `tests/` for this acceleration-reconstruction
  config contract. No dataset/split, runner, notebook, checkpoint, or
  documentation changes.
- **Required behavior:** Add `excluded_users` to `UserSplitConfig` (not the
  top-level `ExperimentConfig`), with an empty exclusion as the default. At
  config construction, canonicalize it to an immutable tuple using the
  repository's existing user-name semantics: preserve a value already starting
  with `user_`; convert a decimal-string identifier to `user_<id>`; otherwise
  preserve its text unchanged. Do not trim whitespace or change case. Reject a
  scalar string/non-sequence input and reject duplicate canonical names (so
  `"17"` plus `"user_17"` fails). Preserve the caller's non-duplicate order.
  `ExperimentConfig.to_dict()` must serialize the canonical tuple through the
  existing nested `split` mapping. The default empty value must leave every
  existing preset/config validation behavior unchanged.
- **Acceptance:** The presets validate with `excluded_users == ()`; a
  non-empty configuration serializes canonical names under
  `split.excluded_users`; numeric aliases normalize; canonical duplicates and
  scalar strings fail clearly. This task defines config representation only;
  it must not yet filter a dataset or change an experiment runner.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-test-module>`; if that environment is unavailable, use
  `writingring-viz` and report it. Also run `git diff --check`.
- **Replan triggers:** Implementing canonicalization needs a shared utility or
  a write outside the allowed scope; the existing config serialization cannot
  express the normalized value; or a required compatibility change affects
  checkpoint consumers.

## T2 — Split-layer exclusion

- **Status:** VERIFIED PASS
- **Dependencies:** T1 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `snn/accel_reconstruction_eval/datasets.py` and one new
  focused split-test module under `tests/`. Do not change config, runners,
  checkpoint/provenance I/O, notebooks, or documentation.
- **Required behavior:** Extend `prepare_user_disjoint_splits()` with a
  keyword-only `excluded_users: Sequence[object] = ()` argument without
  changing `SplitAssignment`'s public shape. Normalize exclusions with the
  existing `normalize_user_name` / `normalize_user_list` semantics (including
  alias-duplicate rejection). First normalize every supplied explicit split
  list and fail fast if any canonical excluded user appears in train, val, or
  test. Then remove all excluded-user rows from a copy of the input manifest,
  before automatic splitting or explicit-list eligibility checks. Apply user
  count, non-empty split, missing-user, complete-assignment, class mapping, and
  label-coverage checks to that eligible manifest. Do not error merely because
  an excluded user is absent from the source manifest. The result must contain
  neither excluded users in `train_users`/`val_users`/`test_users` nor excluded
  samples in `sample_manifest`; retain explicit assertions for this. Preserve
  the exact behavior when `excluded_users=()`.
- **Acceptance:** Focused tests cover automatic exclusion with ratios based on
  the eligible cohort; one and multiple exclusions; numeric alias handling;
  explicit overlap failure; absent exclusion; fewer than three eligible users;
  label/class-mapping failure after exclusion; and empty exclusion compatibility.
  No checkpoint/cohort metadata is added in this task.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-split-test-module>`; use `writingring-viz` only if the
  preferred environment is unavailable. Also run `git diff --check`.
- **Replan triggers:** Correct filtering requires a `SplitAssignment` schema
  change, a shared/config utility write outside scope, or makes the documented
  no-exclusion contract incompatible.

## T3 — Experiment A runner integration

- **Status:** VERIFIED PASS
- **Dependencies:** T2 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `scripts/run_experiment_a.py` and one new focused A-runner
  test module under `tests/`. Do not modify config, datasets, checkpoint/provenance
  I/O, notebooks, or documentation.
- **Required behavior:** In A's `_prepare_split()`, forward
  `config.split.excluded_users` verbatim as `excluded_users=` to
  `prepare_user_disjoint_splits()`, preserving every existing argument and
  error-propagation behavior. A must continue to use only the resulting
  `SplitAssignment.sample_manifest` for downstream normalization/loaders and
  embeddings. Do not independently filter packages or samples.
- **Acceptance:** A focused synthetic-manifest test calls `_prepare_split()`
  with a config carrying exclusions and proves excluded samples/users are
  absent and automatic assignment matches the eligible cohort. A no-exclusion
  case remains compatible. No training run is needed.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-A-runner-test-module>` (or `writingring-viz` only when
  needed), then `git diff --check`.
- **Replan triggers:** Correct forwarding needs package filtering, a split or
  artifact schema change, a CLI/notebook control, or a write outside scope.

## T4 — A checkpoint and provenance cohort contract

- **Status:** VERIFIED PASS
- **Dependencies:** T3 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `scripts/run_experiment_a.py` and one new focused
  checkpoint/provenance artifact test module under `tests/`. Do not modify
  shared I/O, config, datasets, B/C/D runners, notebooks, or documentation.
- **Required behavior:** At A-run time, define `excluded_users` as the
  canonical ordered `config.split.excluded_users`. Define `eligible_users` as
  the canonically normalized, naturally sorted unique user names in the loaded
  source sample manifest after removing `excluded_users`, before any explicit
  split lists can leave users unused. Persist both as stable top-level fields in
  the A checkpoint and in `provenance.json`, alongside the existing top-level
  `train_users`, `val_users`, `test_users`, and `class_to_idx`. Create a new
  `cohort.json` sidecar with exactly those audit fields and include it in
  `artifact_paths`; leave existing CSV schemas unchanged. The saved
  `sample_manifest.csv` remains the post-split manifest and therefore contains
  no excluded samples. Do not alter `load_checkpoint()` required keys,
  artifact type/schema handling, or legacy checkpoint readability.
- **Acceptance:** A focused artifact-contract test proves canonical aliases,
  empty exclusions, explicit splits with `require_all_users_assigned=False`,
  and post-split manifest semantics. It must show the same canonical cohort
  payload is supplied to checkpoint, provenance, and the sidecar without an
  end-to-end model-training run. Existing checkpoint fields and CSV output
  schemas remain untouched.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-A-cohort-test-module>` (or `writingring-viz` only when
  needed), then `git diff --check`.
- **Replan triggers:** Enforcing this contract needs a `SplitAssignment` or
  shared I/O schema change, requires changing legacy checkpoint loading, or
  changes existing CSV schemas.

## Pending TaskSpec slots

## T5 — Experiment B cohort inheritance

- **Status:** VERIFIED PASS
- **Dependencies:** T4 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `scripts/run_experiment_b.py` and one new focused B
  cohort/provenance test module under `tests/`. No shared I/O/config/dataset
  changes, no checkpoint writes, no CSV/NPZ schema changes, and no notebooks or
  documentation edits.
- **Required behavior:** A checkpoint is B's sole cohort authority. For a
  modern A checkpoint, require top-level `excluded_users` and `eligible_users`;
  canonicalize and validate both (no duplicates), validate that split users are
  disjoint subsets of eligible users, and validate the source dataset's
  canonical user cohort exactly equals `eligible_users ∪ excluded_users`.
  Apply checkpoint exclusions and checkpoint split/class mapping to the split
  helper. Preserve the A `require_all_users_assigned` setting from the saved
  `experiment_config.split` so eligible-but-unassigned users remain valid when
  A allowed them. Reject class/label mismatches, missing/additional users,
  excluded/split leakage, and malformed checkpoint cohort fields. Local B
  `excluded_users` must be empty and all three local explicit user lists must
  be `None`; any attempt to select a different cohort fails fast.

  A legacy checkpoint lacking both cohort fields remains supported only by an
  explicit fallback: use empty exclusions and require the current dataset user
  set to equal the checkpoint split-user union exactly. Mark this fallback in
  provenance. A partially populated cohort contract is invalid. Record the
  effective `excluded_users`, `eligible_users`, train/val/test users,
  `class_to_idx`, baseline identity, and cohort source (`checkpoint` or
  `legacy_no_exclusion_fallback`) in B provenance. Preserve frozen A weights,
  A normalization, and all B evaluation semantics.
- **Acceptance:** Focused tests cover modern inheritance, local-cohort
  conflicts, added/missing users, excluded rows, split/class mismatch,
  eligible-but-unassigned users, empty exclusion, legacy fallback, and
  provenance. No A checkpoint is modified.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-B-test-module>` (or `writingring-viz` only when needed),
  then `git diff --check`.
- **Replan triggers:** Correct validation requires shared I/O/dataset changes,
  a changed legacy loader policy, or artifact formats beyond provenance.

## T6 — Experiment C cohort inheritance

- **Status:** VERIFIED PASS
- **Dependencies:** T4 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `scripts/run_experiment_c.py` and one new focused C
  inheritance/artifact test module under `tests/`. Do not modify shared
  I/O/config/datasets, B/D, notebooks, or documentation.
- **Required behavior:** When C has a reference A checkpoint, enforce the same
  modern and legacy cohort validation/fallback as T5, including local cohort
  conflict rejection and A's saved complete-assignment policy. Apply A
  exclusions/splits/class mapping before loaders. Persist the effective cohort,
  cohort source, and reference identity in C's checkpoint and provenance. C
  must still train from scratch and fit normalization on reconstruction train;
  it must never reuse A weights or normalization. With `allow_new_split=True`
  and no usable reference checkpoint, retain the existing standalone split
  path, honor local exclusions, and record that no A cohort was inherited.
- **Acceptance:** Focused tests cover strict modern inheritance, conflict and
  dataset/cohort drift rejection, eligible-but-unassigned users, malformed/
  legacy reference behavior, checkpoint/provenance fields, and standalone
  `allow_new_split` truthfulness without training.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-C-test-module>` (or `writingring-viz` only when needed),
  then `git diff --check`.
- **Replan triggers:** Correct behavior needs shared contract changes, alters
  C's from-scratch/reconstruction-normalization protocol, or expands artifacts
  beyond C checkpoint/provenance.

## T7 — Experiment D cohort inheritance

- **Status:** VERIFIED PASS
- **Dependencies:** T4 independently verified.
- **Executor:** `luna_worker` (implementation task).
- **Allowed writes:** `scripts/run_experiment_d.py` and one new focused D
  inheritance/artifact test module under `tests/`. Do not modify shared
  I/O/config/datasets, B/C, notebooks, or documentation.
- **Required behavior:** When D has a reference A checkpoint, enforce the same
  modern and legacy cohort validation/fallback as T5, including local cohort
  conflict rejection and A's saved complete-assignment policy. Apply A
  exclusions/splits/class mapping before loaders. Persist effective cohort,
  cohort source, and reference identity in D's checkpoint and provenance. Keep
  mixed raw/reconstruction training, mixed-train normalization, and one-model
  dual-domain test semantics unchanged. With `allow_new_split=True` and no
  usable reference checkpoint, retain standalone local splitting/exclusions and
  truthful non-inherited provenance.
- **Acceptance:** Focused tests cover strict modern inheritance, conflicts and
  dataset/cohort drift, eligible-but-unassigned users, malformed/legacy
  reference behavior, checkpoint/provenance, and standalone path without model
  training; mixed-domain behavior remains untouched.
- **Validation:** `conda run --no-capture-output -n writingring-gpu python -m
  pytest <new-focused-D-test-module>` (or `writingring-viz` only when needed),
  then `git diff --check`.
- **Replan triggers:** Correct behavior needs shared contract changes, alters
  mixed-domain protocol, or expands artifacts beyond D checkpoint/provenance.

## Pending TaskSpec slots

## T8 — Notebook controls

- **Status:** VERIFIED PASS
- **Dependencies:** T5, T6, and T7 independently verified.
- **Executor:** `luna_worker` (notebook implementation task).
- **Allowed writes:**
  `notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb`,
  `notebooks/experiment_B_reconstruction_frozen_cnn.ipynb`,
  `notebooks/experiment_C_reconstruction_trained.ipynb`, and
  `notebooks/experiment_D_mixed_training.ipynb` only. Do not modify any
  `*.old` notebook, runner, test, documentation, artifact, or shared module.
- **Required behavior:** A's configuration cell defines the sole user-level
  input `EXCLUDED_USERS` as an ordered tuple or list, defaulting to empty. A's
  config construction applies it via `replace(config.split,
  excluded_users=EXCLUDED_USERS)`; it must not call an unsupported preset
  keyword or use a set. B/C/D define no local user-cohort or explicit-split
  control and retain strict `ALLOW_NEW_SPLIT=False` where applicable. After
  their run, B/C/D read `run.output_dir / "provenance.json"` and display the
  inherited `cohort_source`, `excluded_users`, `eligible_users`, split users,
  and `class_to_idx`. Make the same cohort facts visible for A from its
  provenance/cohort artifact. State in notebook prose/code comments that A
  must be regenerated before B/C/D after exclusions change.
- **Acceptance:** Static notebook JSON/AST inspection proves A's only cohort
  selector is an ordered `EXCLUDED_USERS` forwarded through nested config;
  B/C/D have no local exclusion/explicit-split override and visibly render
  inherited provenance, with C/D strict reference mode. Existing runner
  imports/calls and experiment protocols remain intact.
- **Validation:** run a read-only Python/JSON notebook inspection that parses
  every code cell, then `git diff --check`. Do not execute training notebooks.
- **Replan triggers:** Correct controls need an API/artifact change, a runner
  change, a documentation file, or notebook semantics outside these four paths.

## Pending TaskSpec slots

## T9 — Cross-experiment and regression validation

- **Status:** VERIFIED PASS
- **Dependencies:** T8 independently verified.
- **Executor:** `luna_worker` (focused regression test), then PRIMARY
  validation and plan closure.
- **Goal:** Independently validate the complete fixed-user-exclusion workflow
  across configuration, split construction, Experiment A artifacts, and
  Experiment B/C/D cohort inheritance.
- **Required behavior:** The validation matrix must establish that exclusions
  are canonicalized and applied before split assignment; excluded users are
  absent from assignments and manifests; Experiment A persists the complete
  cohort contract; and B/C/D inherit that contract without local cohort
  overrides. It must also exercise empty-exclusion compatibility, explicit
  overlap/insufficient-cohort failures, legacy fallback, and provenance
  reporting. Add one focused synthetic integration regression that creates an
  actual A checkpoint and artifact set through the A runner's existing
  lightweight test seam, then supplies the loaded checkpoint to B, C, and D's
  split preparation paths. The regression must also observe the manifest passed
  to A's loader construction and the sample identities supplied to its embedding
  artifact writer, proving excluded-user samples cannot cross either boundary.
  No production behavior is introduced by this task.
- **Preserved contracts:** User-disjoint split behavior; A as B/C/D's cohort
  authority; B's frozen-A weights and normalization; C's from-scratch,
  reconstruction-normalized protocol; D's mixed-training protocol; existing
  checkpoint and CSV schemas other than the already-verified additive cohort
  fields.
- **Allowed writes:** One new focused regression test module under `tests/`,
  this TaskSpec, `docs/plans/WORKBOARD.md`, and plan lifecycle relocation only
  after the frozen validation passes. Do not modify production code, notebooks,
  data, or vendor content unless a failing result is routed to the applicable
  prior task for repair.
- **Acceptance criteria:** The new integration regression proves the A-produced
  checkpoint is accepted as the same exclusion/split authority by B/C/D and
  that excluded sample IDs are absent from A loader and embedding-artifact
  inputs. The expanded focused pytest matrix passes with no failures;
  `git diff --check` passes; and a fresh independent verifier reports PASS for
  the frozen T9 scope. If the matrix reveals a real defect, route it to the
  owning prior task rather than broadening T9.
- **Validation:**
  `conda run --no-capture-output -n writingring-gpu python -m pytest -q
  tests/test_acceleration_reconstruction_config.py
  tests/test_acceleration_reconstruction_splits.py
  tests/test_acceleration_reconstruction_a_runner.py
  tests/test_acceleration_reconstruction_a_cohort.py
  tests/test_acceleration_reconstruction_b_cohort.py
  tests/test_acceleration_reconstruction_c_cohort.py
  tests/test_acceleration_reconstruction_d_cohort.py <new-focused-integration-
  test-module>`; then `git diff --check`.
  Use `writingring-viz` only if the preferred environment is unavailable.
- **Dependencies:** No external data or model-training artifacts are required;
  the frozen matrix uses synthetic fixtures.
- **Replan triggers:** A failing result requires a change to the established
  cohort, checkpoint, notebook, or experiment-protocol contract; or resolving
  it requires edits outside the owning prior task's allowed scope.

**Execution record:** The new synthetic A→B/C/D integration regression creates
and reloads a real A checkpoint, observes A's loader and embedding-artifact
boundaries, and verifies that B/C/D reuse its cohort authority. The expanded
`tests/test_acceleration_reconstruction_*.py` suite passed with **61 passed**
in the preferred `writingring-gpu` environment. `git diff --check` passed, and
a fresh independent verifier returned **PASS**.
