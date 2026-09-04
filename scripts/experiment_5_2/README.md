# Experiment 5.2 — Frozen-local FF/RSNN long-term tauR sweep

## Question

With the validated Exp3 local representation frozen and fed at the original 64 Hz rate, how much gesture-level temporal information can be causally consolidated into a single endpoint membrane state by passive LIF memory versus learned recurrence?

Exp5.2 deliberately removes the Exp5.1 leaky interface and all hard temporal pooling from the temporal decoder path.

```text
Raw64
  -> frozen Exp3-exact 2-layer local SNN
  -> raw L2 spike trajectory at 64 Hz
  -> FF-LIF or dense RSNN (H=128)
  -> hidden Uend
  -> Linear(128, 12)
  -> endpoint CE
```

## Frozen local source

The source model is the exact `exp3_exact_analog` condition from Exp5.0.1 / historical Exp3:

```text
Raw64 30-channel weighted events
  -> L1: 128 Synaptic neurons, shifts (2, 3, 4)
  -> L2: 128 Synaptic neurons, shifts (2, 3, 4)
```

Exp5.2 uses seeds:

```text
11, 23, 37, 53, 71
```

For each seed, one preparation task trains or reuses the exact Exp3-compatible local checkpoint under the same fixed user split, then caches the L2 binary trajectory once. All ten temporal-decoder conditions for that seed consume exactly that same cached trajectory.

There is no Fixed250 pooling, Relative10 pooling, leaky242 interface, learned compressor, or temporal downsampling before the FF/RSNN decoder.

## Temporal decoder sweep

Architectures:

- `ff`: no learned recurrent matrix; temporal memory is only passive LIF membrane state.
- `rsnn`: adds dense `128 x 128` hidden-to-hidden recurrence from the previous hidden spike.

Stage-2 width is fixed at 128.

The recurrent/passive membrane decay is parameterized by repository-style shifts:

```text
shift_mem_R = 3, 4, 5, 6, 7
beta_R      = 1 - 2^-shift
```

At 64 Hz these correspond approximately to:

```text
117, 242, 492, 992, 1992 ms
```

Threshold is fixed at 0.5. No separate temporal synaptic state is introduced in Exp5.2, so the swept temporal mechanism is the LIF membrane state plus optional learned recurrence only.

## Primary readout and loss

Exp5.2 is intentionally hardware-agnostic. The primary representation is the hidden membrane at the last valid gesture timestep:

```text
Uend_i = U_i[length_i - 1]
logits = Linear(Uend_i)
loss   = CE(logits, y_i)
```

Padding timesteps never define the endpoint. The implementation explicitly gathers `length - 1` and rejects non-positive or overlong valid lengths.

Checkpoint selection is:

1. maximum validation native endpoint balanced accuracy;
2. tie-break minimum validation endpoint CE.

Test balanced accuracy is never used to select an epoch or tau.

## Run matrix

```text
2 architectures
x 5 shift/tau values
x 5 frozen-local seeds
= 50 independent temporal-decoder runs
```

The scientific comparisons are:

```text
FF(tau_long) - FF(tau_short)
    -> passive membrane-memory contribution

RSNN(tau) - FF(tau)
    -> learned recurrence contribution beyond passive memory
```

## Evaluation matrix

Every selected temporal checkpoint reports:

- native endpoint-head BA/accuracy/macro-F1/loss;
- `hidden_whole_count` + train-only scaled Linear probe;
- `hidden_fixed250_ordered` + train-only scaled Linear probe;
- `hidden_relative10_ordered` + train-only scaled Linear probe;
- `hidden_uend` + train-only scaled Linear probe;
- hidden events/neuron/s;
- mean endpoint membrane L2 norm;
- 2 s zero-input post-end tail-event fraction and tail rate.

The local cache preparation also produces, once per seed:

- `local_whole_count`;
- `local_fixed250_ordered`;
- `local_relative10_ordered`.

These provide the bag-of-events, hard-boundary temporal, and offline phase-aware reference levels for the same frozen local trajectory.

The key mechanistic diagnostic is whether `hidden_uend` and the native endpoint head approach the strong local Fixed250/Relative10 trajectory references. A large `hidden_relative10_ordered - hidden_uend` gap indicates that the temporal decoder forms an informative trajectory but does not fully consolidate it into endpoint state.

## Multi-CPU execution

Exp5.2 follows `AGENTS.md` exactly:

```text
5 frozen-local preparation tasks
  -> Slurm array 0-4%5

local-cache afterok
  -> 50 temporal decoder tasks
  -> Slurm array 0-49%50
  -> one CPU core per task
  -> train -> select checkpoint -> evaluate -> probes -> per-run artifacts

all 50 decoder tasks succeed
  -> afterok finalizer
  -> concatenate existing artifacts only

finalizer succeeds
  -> analysis-only notebook reads finalized CSV/JSON artifacts
```

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_2_cpu.bash
```

Every compute task initializes Conda locally and sets single-thread CPU environment variables.

## Notebook aggregation policy

`notebooks/experiment_5_2_frozen_local_tauR_sweep.ipynb` is analysis-only.

It must:

1. load only finalized `runs.csv`, `histories.csv`, `local_reference.csv`, and `manifest.json`;
2. compute mean ± SD by architecture/tau;
3. select the RSNN tau by **mean validation native endpoint BA only**;
4. report the selected test result only after validation selection;
5. plot test endpoint BA vs tau for FF and RSNN with frozen-local reference lines;
6. compute paired `RSNN - FF` effects by seed at each tau;
7. show the hidden readout matrix (`whole_count`, `fixed250`, `relative10`, `uend`) for validation-selected conditions;
8. compute timing-recovery and Fixed250-gap diagnostics without clipping;
9. plot validation learning curves for the validation-selected FF and RSNN conditions;
10. never train, launch Slurm, regenerate missing runs, or select on test BA.

## Finalized artifacts

```text
notebooks/artifacts/experiment_5_2_frozen_local_tauR_sweep/
  frozen_exp3_l2_endpoint_tauR_v1/
    runs.csv
    histories.csv
    local_reference.csv
    manifest.json
```

Per-run checkpoints/evaluations/histories and per-seed local caches/references remain in subdirectories under the same protocol directory.

## Required checks

```bash
python -m pytest -q tests/test_repository_source_syntax.py tests/test_experiment_5_2_contract.py
```
