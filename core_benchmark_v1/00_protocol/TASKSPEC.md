# Core benchmark v1: frozen implementation contract

## Goal
Implement the agreed standardized SNN benchmark in an independent root folder, with one result-analysis notebook per block and one integrated notebook.

## Required behavior
1. Freeze one concrete user split and dataset identity; seeds 11/23/37 change initialization/training, not users. Reuse one canonical REF per seed across blocks.
2. Isolate Objective, Tau and Depth at the shared anchor; independent cases can execute in parallel. Only actual checkpoint dependencies create later stages.
3. Use unnormalized synaptic integration, a no-bias analog accumulator, explicit valid-step loss geometry, and matched affine/no-bias temporal probes. Readout-to-spike controls must preserve frozen hidden representations.
4. Use one CPU per independent task with at most 50 concurrent tasks, atomic per-run train/evaluate artifacts, strict aggregation, and analysis-only notebooks.
5. Persist enough sample, source, configuration, checkpoint and decoder identity to reject accidental mixing of results.

## Preserved contracts and write scope
Do not change historical experiment behavior, raw data, existing checkpoints, vendor sources, or dataset schemas. Write only core_benchmark_v1, its focused root test, and its path-filtered CI workflow. Reuse the validated repository dataset loader; preserve event-channel semantics and right-padding masks.

## Acceptance criteria
The default manifest contains 27 unique backbone runs and 6 readout tasks; all aliases map to O0. Synthetic end-to-end tests exercise optional depth/membrane branches, every probe coordinate, both output dynamics and output-only adaptation. Missing, mismatched or modified artifacts prevent a PASS aggregate. All eight notebooks validate and execute from finalized artifacts. Python/Bash source syntax and focused contracts pass under Python 3.11.

## Validation
Run the repository source-syntax check and the focused benchmark contract suite. Validate Slurm through dry-run and Bash syntax, not production job submission. Distinguish synthetic execution evidence from real-data performance; do not claim that research runs have completed.

## Dependencies / replan triggers
Real training needs the existing D0 action0/action1 padded datasets and a compatible cluster Conda environment. Stop and report incompatibility rather than guessing channel schemas, silently modifying user cohorts, or substituting old experiment checkpoints.
