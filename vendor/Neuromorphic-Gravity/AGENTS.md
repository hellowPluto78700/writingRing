# AGENTS.md

## Repository purpose

This repository investigates wavelet-based IMU signal encoding,
spike generation, spiking neural networks, NeuroBench evaluation,
Rockpool simulation, and Xylo IMU deployment.

## Current analysis task

The active repository-analysis specification is:

- `docs/PROJECT_SUMMARY_TASK.MD`

The approved execution plan, once created, is:

- `docs/PROJECT_SUMMARY_PLAN.md`

The final deliverable is:

- `docs/PROJECT_SUMMARY_REPORT.md`

The persistent investigation log is:

- `PROGRESS.md`

## Required workflow

1. Read `docs/PROJECT_SUMMARY_TASK.MD` completely.
2. Read `PROGRESS.md` before continuing existing work.
3. Inspect the actual implementation and call graph.
4. Produce and obtain approval for a plan before writing the final report.
5. Do not modify `docs/PROJECT_SUMMARY_TASK.MD` unless explicitly instructed
   by the user.
6. After every completed investigation phase, update `PROGRESS.md`.
7. Generate the final report only after the plan has been reviewed.

## Analysis rules

- Base important conclusions on repository evidence.
- Trace actual imports, calls, parameter propagation, tensor shapes,
  configuration values, and command-line arguments.
- Do not infer behavior solely from filenames or comments.
- Distinguish:
  - Confirmed
  - Strongly inferred
  - Unresolved
- Preserve Python identifiers, class names, function names, command-line
  flags, equations, tensor dimensions, and paths in English.
- Write explanatory report text in Chinese.
- Cite Python and shell evidence using:
  `path/to/file.py:L10-L25`
- Cite notebook evidence using:
  `path/to/notebook.ipynb`, cell N
- Clearly state when notebook execution order is uncertain.

## Repository safety

Do not modify:

- source code
- datasets
- checkpoints
- notebooks
- environment files
- existing result files

Files permitted to be created or updated for this task:

- `docs/PROJECT_SUMMARY_PLAN.md`
- `docs/PROJECT_SUMMARY_REPORT.md`
- `PROGRESS.md`

Do not run:

- long training jobs
- full dataset preprocessing
- GPU-intensive experiments
- hardware-dependent Xylo experiments

Small deterministic CPU-only checks are permitted when needed to verify:

- tensor shapes
- default parameters
- import relationships
- call behavior
- encoder output dimensions

Record every executed verification command and its result in `PROGRESS.md`.

## Progress-log rules

Keep `PROGRESS.md` concise and cumulative.

For every phase, record:

- date or session
- files inspected
- call paths confirmed
- parameters traced
- commands executed
- conclusions reached
- unresolved questions
- next step

Do not erase previous investigation history.

## Language

All planning documents shall be written entirely in English.

All reports shall be written in English.

Always preserve:

- source code
- identifiers
- filenames
- paths
- class names
- function names
- command-line options

in English.
