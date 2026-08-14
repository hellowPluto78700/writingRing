# Experiment Cohort Selection and Downstream Inheritance — Frozen TaskSpecs

**Lifecycle:** COMPLETE. The plan in `11_add_label_select_mode_plan.md`
defined the dependency order. Both tasks received independent PASS.

## T1 — Apply cohort selection before Experiment A splitting

- **Status:** VERIFIED PASS
- **Goal:** Let Experiment A derive its final user-and-label cohort after data
  validation and before splitting, normalization, training, or evaluation.
- **Required behavior:** A supports an optional ordered `included_labels`
  selection in addition to existing user exclusions. `None` means every label;
  a provided selection must be canonicalized to an immutable duplicate-free
  label sequence. After normal dataset/package validation, exclude users first,
  then require every requested label to survive and retain only those labels.
  Build A's split, class mapping, normalization, loaders, embeddings, and
  artifacts from that final manifest. Users without surviving rows must not be
  split. Existing explicit split validation must fail for a filtered-out user.
  The A notebook exposes `EXCLUDED_USERS` and `INCLUDED_LABELS` as the sole
  cohort controls.
- **Preserved contracts:** Existing no-filter behavior, user-disjoint split
  semantics, package/segment identity (`sample_id`, `package_index`, and
  `segment_index`), padded/reconstruction artifacts, CSV/checkpoint schemas,
  model/training/normalization algorithms, and evaluation metrics.
- **Allowed write scope:** `snn/accel_reconstruction_eval/**`,
  `scripts/run_experiment_a.py`, the Experiment A notebook, and focused tests
  for A cohort selection. Documentation updates only for this TaskSpec and the
  workboard.
- **Acceptance criteria:** Exclusion-only, label-only, and combined selection
  produce a final manifest containing exactly the requested surviving labels
  and no excluded-user rows; required missing labels and filtered explicit
  users fail explicitly; class mapping exactly covers final labels; surviving
  identities are unchanged; and normalization uses final train rows only.
- **Validation:** Focused config/dataset/split/A-runner tests covering the
  frozen plan matrix, then `git diff --check`, in `writingring-gpu` (fallback
  only if unavailable). No full-repository pytest.
- **Dependencies:** None.
- **Replan triggers:** Only the three explicit T1 conditions in the frozen
  plan: filtering cannot preserve identity, needs a persisted schema change,
  or existing split APIs cannot express the filtered explicit split without a
  material cross-module contract change.

**Execution record:** T1 passed after one in-scope repair that removed an
unrequested `included_labels` artifact payload. The final implementation keeps
selection in configuration and the in-memory split path only. Independent
verification confirmed the fixed processing order and unchanged artifact
schemas; 29 focused tests passed in `writingring-gpu`.

## T2 — Enforce Experiment A cohort authority in B/C/D

- **Status:** VERIFIED PASS
- **Goal:** Make B/C/D reconstruct A's selected user and label cohort from its
  existing checkpoint split-user fields and `class_to_idx`, without any
  checkpoint migration or local cohort override.
- **Required behavior:** After normal downstream loading/validation and current
  A-checkpoint contract validation, restrict each manifest to A's exact
  train/val/test user union and `class_to_idx` label keys, then rebuild the
  existing explicit split with A's class mapping. A required label missing after
  that restriction, or an A split user with no surviving row, must fail
  explicitly. B/C/D checkpoint-backed configs reject non-`None`
  `included_labels` just as they reject local user exclusions/explicit splits.
  Standalone C/D behavior remains unchanged. Do not add persisted fields.
- **Preserved contracts:** Existing A checkpoint readability and schema; B's
  frozen weights and checkpoint normalization; C's fresh reconstruction
  training/normalization; D's mixed training/normalization; paired raw/recon
  sample membership, ordering, and `package_index`/`segment_index` identity;
  model and evaluation behavior.
- **Allowed write scope:** `scripts/run_experiment_b.py`,
  `scripts/run_experiment_c.py`, `scripts/run_experiment_d.py`, local shared
  cohort helpers under `snn/accel_reconstruction_eval/**` only when necessary,
  B/C/D notebooks only for removing/clarifying prohibited local controls, and
  focused downstream/integration tests. Documentation updates only for this
  TaskSpec and the workboard.
- **Acceptance criteria:** Given one A checkpoint and a dataset containing
  excluded users plus unselected labels, A/B/C/D have identical train/val/test
  users and `class_to_idx`; B/C/D manifests contain only A split users and
  labels; missing required users/labels and local label overrides fail clearly;
  paired identity remains equal where applicable; no checkpoint schema changes
  occur.
- **Validation:** Focused B/C/D cohort tests and one A-to-B/C/D synthetic
  integration fixture, then `git diff --check`, in `writingring-gpu` (fallback
  only if unavailable). Do not run full-repository pytest unless this matrix
  reveals a broader failure.
- **Dependencies:** T1 independently verified PASS.
- **Replan triggers:** Only the three explicit T2 conditions in the frozen
  plan: existing A split/mapping fields are insufficient, exact cohort identity
  needs unavailable persisted data, or enforcement requires a checkpoint/schema
  migration or other durable public contract change.

**Execution record:** T2 passed after extending the real A-to-B/C/D integration
fixture to include a source-only unselected label. Independent verification
confirmed exact split-user and class-mapping inheritance, explicit missing
cohort failures, preserved identities and protocols, and unchanged checkpoint
schemas. The final focused acceleration-reconstruction regression suite passed
with **76 passed** in `writingring-gpu`; `git diff --check` passed.
