# Exp19 — Cross-user generalization mechanisms

Exp19 keeps the locked CoreBenchmark v1 split, seeds, 30-channel input, 128-wide (2,3,4)(2,3,4) SNN, bias-free accumulator, WCCE objective, and native validation checkpoint selection fixed. It tests three distinct explanations for the train-to-unseen-user gap.

## Families

- **19.1 dominant-evidence self-challenging**: random, high-rate, evidence-targeted, and evidence-strength-matched-random neuron masking. Evidence masking starts only after 25% of training and ramps to full strength by 40%.
- **19.2 native decision invariance**: class-conditioned native margin consistency and user-risk variance (REx). The task batch remains an ordinary CoreBenchmark batch; the auxiliary branch uses a separate class-by-user batch. These losses operate in native decision space, not a train-only projection head.
- **19.3 nuisance coverage**: matched random noise, schema-safe nearest-neighbor writing-speed temporal warp of the locked pre-SNN input, and same-class/different-user feature-statistics style mixing.

The Core cache already contains the 30-channel pre-SNN representation. Exp19 therefore **does not invent raw sensor channel semantics or apply guessed xyz rotations**. A future raw-domain extension must route through the documented upstream preprocessing/encoder contract.

## Calibration and final runs

Calibration uses seed 101 only:

- 19.1 evidence-targeted: q in {0.1,0.2,0.3} × lambda in {0.5,1.0} = 6 tasks.
- 19.2 margin consistency: 4 lambda values.
- 19.2 REx: 4 lambda values.
- 19.3 physical strength: weak/medium/strong = 3 tasks.
- 19.3 style-mix probability: 0.25/0.50/0.75 = 3 tasks.

Total calibration = **20 one-CPU tasks**. Selection uses validation native BA only, with weaker intervention as the tie-break.

The selected settings are frozen for seeds 11/23/37:

- 19.1: 4 variants × 3 seeds = 12 tasks.
- 19.2: 2 variants × 3 seeds = 6 tasks.
- 19.3: 3 variants × 3 seeds = 9 tasks.

Total final = **27 one-CPU tasks**. Exp19.0 reuses the three canonical CoreBenchmark O0 references and does not retrain them.

## Execution contract

Required per-run artifacts:

- checkpoint.pt
- history.json
- probes.json plus canonical probe artifacts
- complete.json

A run is complete only when complete.json matches the locked Core identity and current Exp19 source hash. Finalization fails closed if any expected final run is absent.

Slurm DAG:

    environment smoke
      -> experiment smoke
      -> prepare
      -> calibration array (20 independent tasks, 1 CPU each)
      -> select
      -> final array (27 independent tasks, 1 CPU each)
      -> finalize/aggregate

Training, checkpoint selection, native evaluation, and completion artifacts are co-located in the same array task. The finalizer only aggregates completed artifacts.

Recommended concurrency is 20 for calibration and 27 for final, both below the repository ceiling of 50. Each task requests one CPU core and uses scripts/bash_script/SNN_Bash/slurm_cpu_env.bash, which pins BLAS/OpenMP threads to one.

Resume behavior: a valid complete.json with matching provenance skips the run. A stale or missing completion marker causes the task to execute. The finalizer never silently regenerates missing runs.

Formal submit:

    bash scripts/bash_script/SNN_Bash/submit_exp_19_cpu.bash

Focused validation:

    python -m pytest -q tests/test_repository_source_syntax.py tests/test_experiment_19_cross_user_generalization_contract.py

[executed on device: acd20ea31325 (425a23ad-a806-44e3-abed-ce7b563d3969)]