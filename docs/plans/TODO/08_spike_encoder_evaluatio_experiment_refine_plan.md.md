# Plan: Acceleration Reconstruction Evaluation Contract Hardening

## 0. Plan Metadata

```text
Plan:
    acceleration_reconstruction_eval_contract_hardening

Status:
    DRAFT

Workflow class:
    HIGH_RISK

Repository:
    hellowPluto78700/writingRing

Validated baseline:
    c58f3b0c134d6c3507c85d5b0c0f0a3694df2ab8

Primary durable contract:
    docs/notes/acceleration_reconstruction_helper_modules_guide.md

Secondary contract / user-facing entry point:
    README.md

Primary implementation surface:
    snn/accel_reconstruction_eval/**
    scripts/run_experiment_a.py
    scripts/run_experiment_b.py
    scripts/run_experiment_c.py
    scripts/run_experiment_d.py
    tests/**
    pyproject.toml
    README.md
```

---

# 1. Purpose

The acceleration reconstruction evaluation path now has reusable A/B/C/D helpers and dedicated experiment runners. The next hardening pass must make the experiment protocol reproducible, regression-tested, package-importable, and explicitly documented.

This plan addresses two coupled goals:

1. make `ExperimentConfig.random_seed` the single authoritative experiment seed controlling every stochastic component of A/B/C/D;
2. add regression coverage and README/package integration checks for the protocol invariants that must remain stable across future refactors.

This work is HIGH_RISK because it touches public helper configuration semantics, legacy checkpoint compatibility, cross-domain dataset contracts, multi-experiment split invariants, packaging/import behavior, and experiment-runner orchestration.

The work must harden the existing design. It must not redesign the acceleration CNN experiments or change the scientific meaning of Experiments A/B/C/D2.

---

# 2. Repository Workflow Contract

Follow `AGENTS.md`.

Knowledge priority for this work:

```text
1. docs/notes/**
2. README.md
3. current code and tests
4. docs/plans/**
5. FROZEN TaskSpec for the active task
```

Use the repository STANDARD/HIGH_RISK workflow:

```text
PRIMARY
  -> luna_probe
  -> freeze TaskSpec
  -> luna_worker
  -> luna_verifier
```

Rules:

- Probe is read-only and validates one DRAFT TaskSpec against the minimum relevant implementation surface.
- PRIMARY freezes only behavior, contracts, allowed paths, acceptance criteria, validation, and replan triggers.
- Worker owns the implementation/debug/test/repair loop inside the frozen contract.
- Verifier independently checks the actual diff and acceptance criteria; worker summary is not proof.
- `vendor/**` and `data_sample/**` are always forbidden writes.
- Run validation from cheap/focused to broad/expensive.
- Do not run full pytest after every small task; reserve broad regression for integration/final closure.
- Continue through runnable tasks until the plan is complete or a genuine BLOCKED/REPLAN condition occurs.

---

# 3. Current Baseline Facts to Reconfirm During Probe

The following observations define the DRAFT plan and must be revalidated by the relevant probe before freezing each TaskSpec:

```text
Configuration:
    ExperimentConfig.random_seed exists.
    RepresentationEvaluationConfig.random_seed also exists independently.

Split / training:
    user split uses ExperimentConfig.random_seed.
    train DataLoader shuffle uses the experiment seed.
    runners seed Python / NumPy / Torch / CUDA from the experiment seed.

Representation evaluation:
    linear probe uses RepresentationEvaluationConfig.random_seed
    plus LinearProbeConfig.seed_offset.
    KMeans uses RepresentationEvaluationConfig.random_seed.
    silhouette balanced sampling uses RepresentationEvaluationConfig.random_seed.

Runner CLI state:
    A and B expose --seed and explicitly synchronize evaluation.random_seed.
    C and D currently construct configs with random_seed=12345 and do not
    expose an equivalent --seed path.

Dataset protocol:
    single-domain raw and reconstruction datasets preserve logical sample_id,
    label, valid_length, and valid_mask alignment.
    D2 mixed dataset duplicates each logical sample once as ::raw and once as
    ::reconstruction.
    normalization uses valid time points only and restores normalized padding
    to exact zero.

Experiment contracts:
    C and D can reuse Experiment A train/val/test users and class_to_idx.
    D trains one mixed-domain model and evaluates the same best checkpoint on
    raw test and reconstruction test.

Checkpoint compatibility:
    MaskAwareAccelerationCNN layer names and architecture configuration are
    intended to preserve strict Experiment A legacy checkpoint restoration.

Packaging / docs:
    snn.accel_reconstruction_eval is imported as a public helper package.
    pyproject.toml packaging must include the helper subpackage in installed
    environments.
    README currently needs complete A/B/C/D runner and C/D/final notebook
    entry points.
```

If probe evidence contradicts a material assumption above, return REVISE and update the DRAFT TaskSpec before freezing.

---

# 4. Non-Goals

Do not use this work to:

- change the A/B/C/D2 scientific protocols;
- change the acceleration CNN architecture or layer names except if required to restore the documented legacy compatibility contract;
- change reconstruction algorithms or producer artifact formats;
- change raw acceleration channel semantics;
- alter user split policy beyond seed propagation and checkpoint-reuse contracts;
- introduce a new experiment E or new evaluation metric;
- replace NumPy/PyTorch random APIs solely for style;
- redesign the repository packaging layout beyond what is needed to make the existing public helper package install correctly;
- rewrite notebooks beyond changes strictly required for the documented runner/seed contract;
- modify `vendor/**` or `data_sample/**`.

---

# 5. Dependency DAG

```text
T001 Unified experiment random-seed contract
  |\
  | \
  |  +--> T003 Package / CLI smoke contract
  |
  +-----> T002 Protocol regression suite
             |
             +--> T004 README / entry-point contract

T001 + T002 + T003 + T004
              |
              v
        T005 Integration closure
```

| Task | Goal | Dependencies | Initial state |
| --- | --- | --- | --- |
| T001 | Make `ExperimentConfig.random_seed` the single experiment seed | — | DRAFT |
| T002 | Lock A/B/C/D protocol invariants with focused regression tests | T001 | DRAFT |
| T003 | Lock package import and A/B/C/D CLI behavior, repair packaging if required | T001 | DRAFT |
| T004 | Update README A/B/C/D/final entry points and execution contract | T002, T003 | DRAFT |
| T005 | Run integration/final regression and close the plan | T001-T004 | DRAFT |

Parallelism guidance:

- T002 and T003 may run in parallel only after T001 is verified PASS and only if their frozen write scopes do not overlap.
- Do not run concurrent workers that both modify the same test file, runner, `pyproject.toml`, or README.
- T004 should wait until final runner/package paths are known.

---

# 6. T001 — DRAFT TaskSpec: Unified Experiment Random Seed

## Goal

Make `ExperimentConfig.random_seed` the single source of truth for all stochastic behavior owned by an A/B/C/D experiment.

## Probe scope

Inspect only the minimum relevant code in:

```text
snn/accel_reconstruction_eval/config.py
snn/accel_reconstruction_eval/evaluation.py
snn/accel_reconstruction_eval/metrics.py
snn/accel_reconstruction_eval/datasets.py
scripts/run_experiment_a.py
scripts/run_experiment_b.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
focused config / runner tests if present
```

Probe must identify every experiment-owned RNG path and confirm whether any consumer intentionally requires an independently configurable evaluation seed.

## Required behavior to freeze if probe confirms

- `ExperimentConfig.random_seed` is authoritative whenever evaluation is run as part of an experiment.
- Constructing an experiment preset with `random_seed=N` results in:

```text
config.random_seed == N
config.evaluation.random_seed == N
```

- Replacing only the top-level experiment seed also resynchronizes evaluation seed.
- A caller cannot accidentally create a divergent experiment/evaluation seed inside a valid `ExperimentConfig`.
- The experiment seed controls:

```text
user split
training DataLoader shuffle
Python RNG
NumPy RNG
Torch RNG
CUDA RNG when available
linear probe initialization / shuffle
KMeans initialization
silhouette / balanced sampling
```

- `LinearProbeConfig.seed_offset` may remain as a deterministic offset derived from the authoritative experiment seed.
- Standalone `RepresentationEvaluationConfig` use outside `ExperimentConfig` may retain its own seed if current public usage requires it.
- A/B/C/D CLIs expose the same `--seed` option and pass it through the authoritative experiment config.
- Remove runner-level duplicate seed synchronization when it becomes redundant under the frozen config contract.

## Contracts to preserve

- Existing default seed remains `12345` unless probe finds a documented different value.
- Same seed produces the same split and DataLoader shuffle under the same environment/input.
- Explicit A checkpoint user lists remain authoritative for B/C/D regardless of local seed value.
- No change to metric definitions, training hyperparameters, or experiment domain selection.
- No change to checkpoint key names solely for seed unification.

## Anticipated allowed write paths

Freeze exact scope after probe. Expected maximum scope:

```text
snn/accel_reconstruction_eval/config.py
scripts/run_experiment_a.py
scripts/run_experiment_b.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
tests/test_accel_eval_config.py
tests/test_accel_eval_cli.py
```

## Forbidden writes

```text
snn/accel_reconstruction_eval/model.py
reconstruction producer scripts
notebooks/**
docs/notes/**
README.md
pyproject.toml
vendor/**
data_sample/**
```

unless probe establishes a concrete dependency requiring TaskSpec revision.

## Acceptance criteria

1. All experiment presets synchronize top-level and evaluation seed.
2. `dataclasses.replace(config, random_seed=N)` preserves the single-source contract.
3. Attempted evaluation-seed divergence inside an ExperimentConfig resolves to the experiment seed or fails explicitly according to the probed API design.
4. A/B/C/D expose consistent `--seed` CLI behavior.
5. C/D no longer silently hard-code a seed independent of CLI configuration.
6. Focused tests prove deterministic propagation to split/loader/evaluation configuration without requiring expensive model training.
7. Existing default behavior with seed `12345` is preserved.
8. Fresh verifier PASS.

## Focused validation

Expected commands, adjusted by probe to the repository's exact filenames:

```bash
conda run -n writingring-gpu python -m pytest -q tests/test_accel_eval_config.py
conda run -n writingring-gpu python scripts/run_experiment_a.py --help
conda run -n writingring-gpu python scripts/run_experiment_b.py --help
conda run -n writingring-gpu python scripts/run_experiment_c.py --help
conda run -n writingring-gpu python scripts/run_experiment_d.py --help
```

Fall back to `writingring-viz` if `writingring-gpu` is unavailable and dependencies permit.

## Replan triggers

- A documented consumer intentionally requires independent experiment and evaluation seeds.
- Seed synchronization cannot be enforced without a public API break broader than this task.
- Correct behavior requires changing evaluation algorithms rather than seed plumbing.
- Runner CLI compatibility requires changing unrelated user-facing arguments.

---

# 7. T002 — DRAFT TaskSpec: Protocol Regression Suite

## Goal

Add focused tests that lock the reusable A/B/C/D acceleration-reconstruction protocol invariants without running expensive end-to-end training for every assertion.

## Probe scope

Inspect:

```text
snn/accel_reconstruction_eval/model.py
snn/accel_reconstruction_eval/io.py
snn/accel_reconstruction_eval/datasets.py
snn/accel_reconstruction_eval/embedding.py
snn/accel_reconstruction_eval/evaluation.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
current Experiment A checkpoint artifacts or legacy model definition only if
needed for compatibility evidence
existing tests and fixtures relevant to checkpoint / dataset contracts
```

Probe must determine the smallest synthetic fixture surface needed to validate the contracts and whether a committed legacy A checkpoint fixture already exists and is appropriate.

## Required regression coverage

### A. Legacy Experiment A checkpoint compatibility

Test the documented strict compatibility invariant:

```text
legacy A state_dict
    -> current MaskAwareAccelerationCNN
    -> load_state_dict(..., strict=True)
    -> success
```

The test must be independent enough to catch accidental layer-name/key/shape changes. Prefer a compact legacy-compatible test model or a stable tiny checkpoint fixture over generating the state dict from the exact same current class being tested.

Also verify current checkpoint loader accepts the documented legacy artifact type/schema behavior.

### B. Raw / reconstruction pairing

For the same logical samples, prove raw and reconstruction views preserve:

```text
sample_id
label / label_idx
valid_length
valid_mask
sample order required by paired evaluation
```

Add negative coverage for mismatched sample order or IDs in paired embedding validation.

### C. Mixed D2 duplication

Prove `MixedAccelerationDataset` produces exactly two examples per logical sample:

```text
<sample_id>::raw
<sample_id>::reconstruction
```

with aligned label, user/action metadata, valid length, and valid mask.

### D. Normalization

Prove:

- statistics use valid time points only;
- padded sentinel values do not affect mean/std;
- `source="mixed"` accumulates raw and reconstruction valid values under the documented weighting semantics;
- normalized invalid/padded positions are exact zero;
- `fitted_on` records the correct domain/split provenance.

### E. C / D same-split contract

With an A reference checkpoint/metadata object, prove C and D reuse exactly:

```text
train_users
val_users
test_users
class_to_idx
```

Use different local random seeds in the test to prove explicit A split metadata remains authoritative.

### F. D single checkpoint / dual-domain evaluation

Using monkeypatch/fakes rather than full training where possible, prove the D runner contract:

```text
fit/train best model once
restore/select one best model state
save one experiment checkpoint
evaluate test_raw
evaluate test_reconstruction
reuse the same mixed train/validation reference embeddings for both evaluations
```

The test must fail if future code trains a separate model per test domain or writes separate raw/reconstruction checkpoints.

## Contracts to preserve

- A remains raw/raw/raw with raw-train normalization.
- B remains frozen A weights + A split/class mapping + A normalization, with raw reference train/val and reconstruction query/test.
- C trains from scratch on reconstruction while reusing A split/class mapping.
- D2 trains once on mixed raw/reconstruction duplication and evaluates one best checkpoint on both held-out domains.
- sample IDs remain stable for single-domain raw/reconstruction and source-suffixed only in mixed duplication.
- padding after normalization remains exact zero.

## Anticipated allowed write paths

Prefer new focused files instead of broad edits:

```text
tests/conftest.py                         # only if shared fixture is justified
tests/test_accel_eval_model_checkpoint.py
tests/test_accel_eval_datasets.py
tests/test_accel_eval_protocols.py
```

Implementation files may be added to allowed scope only if tests expose a real contract defect that can be repaired without redesign; otherwise return REPLAN.

## Forbidden writes

```text
vendor/**
data_sample/**
producer artifact generation code
notebooks/**
README.md
pyproject.toml
unrelated SNN/HAR code
```

## Acceptance criteria

1. Legacy A strict state-dict restore has a regression test that would catch key/shape drift.
2. Raw/reconstruction pairing is tested positively and negatively.
3. Mixed D2 duplication is tested for exact count, ordering/suffix, and metadata alignment.
4. Normalization tests exclude padding and assert exact-zero normalized padding.
5. C/D split reuse is proven against A metadata independent of local seed.
6. D orchestration is proven to train/save once and evaluate two test domains with shared references.
7. Tests are synthetic/focused and do not depend on `data_sample/**`.
8. Fresh verifier PASS.

## Focused validation

```bash
conda run -n writingring-gpu python -m pytest -q \
  tests/test_accel_eval_model_checkpoint.py \
  tests/test_accel_eval_datasets.py \
  tests/test_accel_eval_protocols.py
```

Then run the existing nearest-neighbor acceleration helper tests identified by probe.

## Replan triggers

- Legacy compatibility cannot be established without an unavailable real checkpoint or undocumented historical architecture.
- A current protocol bug requires changing a durable docs/notes contract.
- D cannot satisfy the one-checkpoint/two-domain contract without a runner architecture change outside the anticipated scope.
- Synthetic fixtures cannot faithfully represent the documented package contract without changing production loader APIs.

---

# 8. T003 — DRAFT TaskSpec: Package and CLI Smoke Contract

## Goal

Ensure the acceleration helper package and all A/B/C/D runners are importable and expose a stable CLI in an installed/test environment.

## Probe scope

Inspect:

```text
pyproject.toml
snn/__init__.py if present
snn/accel_reconstruction_eval/__init__.py
scripts/run_experiment_a.py
scripts/run_experiment_b.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
existing packaging/import tests
```

Probe must confirm whether current setuptools configuration includes `snn.accel_reconstruction_eval` in built/installable packages and identify the least invasive packaging repair if not.

## Required behavior

- `import snn.accel_reconstruction_eval` succeeds in the supported test/install environment.
- Public exports required by A/B/C/D runners import successfully.
- Each runner's `--help` succeeds without loading experiment datasets or checkpoints.
- A/B/C/D all expose the authoritative `--seed` argument after T001.
- Packaging includes `snn.accel_reconstruction_eval` and its Python modules.

## Contracts to preserve

- Existing package name remains `writingring`.
- Do not broadly convert packaging strategy unless needed.
- Do not accidentally include `vendor/**`, sample data, notebooks, generated artifacts, or unrelated directories in the installed package.
- CLI help must remain side-effect-free with respect to data loading/training.

## Anticipated allowed write paths

```text
pyproject.toml
tests/test_accel_eval_cli.py
```

Runner files should only be in scope if probe finds a CLI import/argument defect not already handled by T001.

## Acceptance criteria

1. Package import smoke test passes.
2. Public helper symbols needed by runners are importable.
3. A/B/C/D `--help` return zero.
4. A/B/C/D `--help` expose `--seed`.
5. Built/installed packaging includes the helper subpackage.
6. No unrelated package discovery expansion occurs.
7. Fresh verifier PASS.

## Focused validation

At minimum:

```bash
conda run -n writingring-gpu python -m pytest -q tests/test_accel_eval_cli.py
```

If probe confirms a practical wheel-build check is cheap and supported:

```bash
python -m build --wheel
```

and inspect/install the wheel in an isolated environment only if justified by current repository tooling.

## Replan triggers

- Current setuptools layout requires a broader package-discovery migration.
- Helper imports depend on optional dependencies not represented in the documented installation extras.
- CLI import safety requires reorganizing unrelated modules.

---

# 9. T004 — DRAFT TaskSpec: README and Entry-Point Contract

## Goal

Update README so the public acceleration-reconstruction workflow matches the reusable runner architecture and standardized A/B/C/D outputs.

## Probe scope

Inspect:

```text
README.md
scripts/run_experiment_a.py
scripts/run_experiment_b.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb
notebooks/experiment_B_reconstruction_frozen_cnn.ipynb
notebooks/experiment_C_reconstruction_trained.ipynb
notebooks/experiment_D_mixed_training.ipynb
notebooks/final_acceleration_reconstruction_comparison.ipynb
docs/notes/acceleration_reconstruction_helper_modules_guide.md
```

Do not inspect notebook outputs beyond what is required to confirm paths/roles.

## Required documentation behavior

README Entry points must include the authoritative runner paths:

```text
scripts/run_experiment_a.py
scripts/run_experiment_b.py
scripts/run_experiment_c.py
scripts/run_experiment_d.py
```

and notebook presentation/comparison entry points:

```text
notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb
notebooks/experiment_B_reconstruction_frozen_cnn.ipynb
notebooks/experiment_C_reconstruction_trained.ipynb
notebooks/experiment_D_mixed_training.ipynb
notebooks/final_acceleration_reconstruction_comparison.ipynb
```

README must state the execution roles accurately:

```text
A runner:
    trains raw baseline and produces authoritative A checkpoint.

B runner:
    restores/freeze A checkpoint and evaluates reconstruction without
    adaptation.

C runner:
    trains from scratch on reconstruction while reusing A split/class mapping.

D runner:
    trains one mixed-domain model and evaluates the same best checkpoint on
    both raw and reconstruction test domains.

notebooks:
    thin configuration / presentation / interpretation layers.

final notebook:
    compares standardized A/B/C/D artifacts.
```

README should mention `--seed` as the single experiment seed only if that statement is verified after T001.

## Contracts to preserve

- Durable technical detail remains in `docs/notes/**`; README stays concise.
- Do not claim acquisition rates, units, channel semantics, or dataset meanings not established by current notes.
- Do not document nonexistent CLI flags or artifact names.
- Do not present notebooks as the authoritative implementation if runners are authoritative.

## Anticipated allowed write paths

```text
README.md
```

No notebook edits are expected.

## Acceptance criteria

1. README lists A/B/C/D runners.
2. README lists C/D/final notebooks in addition to A/B.
3. Runner/notebook roles match actual implementation.
4. D is documented as one trained checkpoint evaluated on two domains.
5. Seed wording matches the final T001 behavior.
6. No stale or nonexistent entry point remains in the edited section.
7. Fresh verifier PASS.

## Validation

Focused documentation checks:

```text
confirm every listed path exists
confirm --help for every listed runner passes
compare protocol wording against acceleration reconstruction helper guide
```

No full pytest is required for this documentation-only task.

## Replan triggers

- Runner or notebook names change in T001-T003.
- Actual standardized output locations differ from the current documented assumptions.
- README conflicts with a higher-priority docs/notes contract.

---

# 10. T005 — DRAFT TaskSpec: Integration and Plan Closure

## Goal

Verify the completed hardening work as one coherent A/B/C/D contract and close the plan only after focused tasks are independently verified.

## Dependencies

T001, T002, T003, and T004 must each have verifier PASS.

## Probe

No new probe by default. PRIMARY may directly inspect final state. Spawn a targeted probe only if integration reveals a new cross-task contract ambiguity.

## Required integration checks

Confirm final repository state satisfies all of the following simultaneously:

```text
Seed contract:
    one ExperimentConfig seed controls experiment-owned randomness.

Legacy compatibility:
    current model restores legacy A-compatible state dict strictly.

Pairing:
    raw/reconstruction logical sample identity remains aligned.

D2:
    mixed dataset duplicates each sample exactly once per domain.

Normalization:
    valid-only statistics and exact-zero normalized padding.

Split contract:
    B/C/D reference A user split/class mapping where required.

D evaluation:
    one trained/checkpointed D model, two held-out domain evaluations.

Packaging:
    helper package imports after supported installation/test setup.

CLI:
    A/B/C/D --help and --seed are consistent.

Documentation:
    README lists the actual runners/notebooks/final comparison entry point.
```

## Allowed writes

By default, none except plan/task/workboard closure documentation if the repository workflow uses it.

If integration finds a defect inside an already frozen task contract, route back to that task's worker for repair and re-verification. Do not silently expand T005 implementation scope.

## Validation ladder

Run once per unchanged final implementation state, from focused to broad:

```bash
# 1. focused acceleration evaluation tests
conda run -n writingring-gpu python -m pytest -q tests/test_accel_eval_*.py

# 2. nearest related existing tests identified by the task probes
conda run -n writingring-gpu python -m pytest -q <related-test-files>

# 3. full repository regression because this plan is HIGH_RISK
conda run -n writingring-gpu python -m pytest -q
```

Use `writingring-viz` only when the preferred environment is unavailable and record that deviation.

Optional E2E experiment execution is required only if the relevant dataset/checkpoint artifacts are locally available and the frozen TaskSpecs explicitly require it. Do not make unavailable private/local artifacts a reason to fail otherwise valid synthetic protocol tests.

## Final acceptance

- Every prior TaskSpec has independent verifier PASS.
- Full relevant diff is within approved scopes.
- Focused acceleration tests pass.
- Full pytest passes, or any environment-only inability is documented and independently assessed under verifier rules.
- No undocumented behavior change exists.
- README and durable guide are mutually consistent for the changed surface.
- Plan can move from DRAFT/ACTIVE to DONE.

## Replan triggers

- Full regression exposes a real cross-module contract conflict not repairable within an existing frozen TaskSpec.
- A packaging or public API change requires a broader migration decision.
- Legacy A compatibility evidence contradicts the documented helper contract.

---

# 11. Suggested Task File State Machine

If this plan is split into a companion `*_TASKS.md`, use this state progression for each task:

```text
DRAFT TaskSpec
    |
    v
luna_probe
    |
    +--> CONFIRMED -> PRIMARY freezes TaskSpec
    |                    |
    |                    v
    |                luna_worker
    |                    |
    |                    +--> DONE
    |                    +--> NEEDS_REPLAN -> PRIMARY revises / re-probes
    |                    +--> BLOCKED -> stop only for genuine blocker
    |                    |
    |                    v
    |                luna_verifier
    |                    |
    |                    +--> PASS -> task DONE
    |                    +--> FAIL -> worker repairs inside same frozen contract
    |                    +--> REPLAN -> PRIMARY revises / re-probes
    |
    +--> REVISE -> PRIMARY updates DRAFT and re-probes
    +--> BLOCKED -> request external/user input only when genuinely required
```

Do not freeze implementation choices such as exact helper-function names, fixture construction style, or internal refactor structure unless they are required to preserve a public/cross-module contract.

---

# 12. Probe Packet Expectations

Each task probe returns the repository-defined compact packet:

```text
CONTRACT_PROBE_PACKET
task_id
verdict: CONFIRMED | REVISE | BLOCKED
validated_against_commit
files_read
observed_contracts
assumptions_rejected
required_task_changes
dependencies_discovered
risks
```

PRIMARY must incorporate all `required_task_changes` before freezing a TaskSpec.

---

# 13. Worker Packet Expectations

Each worker returns:

```text
TASK_SUMMARY_PACKET
task_id
status: DONE | NEEDS_REPLAN | BLOCKED
changed_files
behavior_changed
contracts_preserved
contracts_changed
validation_run
acceptance_results
deviations
followups
```

A worker must self-review every acceptance criterion and the complete diff before returning DONE.

---

# 14. Verifier Packet Expectations

Each verifier returns:

```text
VERIFICATION_PACKET
task_id
verdict: PASS | FAIL | REPLAN
scope_check
spec_compliance
contract_check
acceptance_check
test_evidence
regression_risks
undocumented_behavior_changes
findings
```

On PASS, `findings` must be empty.

A verifier FAIL routes back to the same frozen task worker when the defect is repairable inside the existing contract. REPLAN is reserved for a materially wrong/stale TaskSpec.

---

# 15. Completion Definition

This plan is DONE only when the repository has an evidence-backed contract that:

1. one experiment seed controls all experiment-owned stochastic paths;
2. Experiment A legacy checkpoints remain strictly restorable by the current model;
3. raw and reconstruction views remain pairable by logical sample identity;
4. D2 duplication is exact and source-explicit;
5. normalization excludes padding and preserves zero padding after transform;
6. C/D preserve A's split/class mapping contract;
7. D trains/checkpoints once and evaluates that same model on both domains;
8. helper package import and A/B/C/D CLIs are regression-tested;
9. packaging actually includes the helper package;
10. README exposes the complete A/B/C/D/final workflow;
11. focused and broad regression evidence is recorded;
12. every implementation task has independent verifier PASS.

