# Exp10.2 — L1 Membrane Memory and Subthreshold-Evidence Recovery

## Question

Exp10.0/10.1 hidden-state probes show a repeated gap between L1 pre-reset
membrane decodability and L1 transmitted binary-spike decodability. Exp10.2
tests whether the current short membrane constant is discarding useful
subthreshold evidence before it can be converted into spike timing/count
evidence.

The experiment deliberately changes only L1 membrane decay. It does **not**
change synaptic shifts, width, threshold, binary coding, preprocessing,
objective, or readout.

## Fixed contract

- Dataset: D1 post-encode airborne mask
- User split: Exp10.0/10.1 rotation 0 only
  - test fold 0
  - validation fold 1
  - train folds 2/3/4
- Backbone: `30 -> 128 -> 128 -> 12`
- Coding: BB, binary L1 + binary L2
- Synaptic shifts:
  - L1: `(2,3,4)`
  - L2: `(2,3,4)`
- Objective: L2 time-shared WCCE / valid-mean evidence CE
- L2 membrane shift: fixed at 1
- Seeds: `11,23,37`
- No phase-aware readout
- No multi-threshold coding
- No joint or TSCE auxiliary objective

For membrane shift (s),

[
\beta_s = 1 - 2^{-s},
\qquad
\tau_{mem}=-\frac{\Delta t}{\ln \beta_s}.
]

At 64 Hz:

| L1 shift_mem | beta | tau_mem |
|---:|---:|---:|
| 1 | ~0.499968 | 22.54 ms |
| 2 | 0.75 | ~54.3 ms |
| 3 | 0.875 | ~117 ms |
| 4 | 0.9375 | ~242 ms |

## Stage A — frozen-weight dynamics replay

First train the normal D1+BB A2 baseline with L1/L2 `shift_mem=1` for the
three paired seeds.

For every baseline checkpoint, reload the exact learned matrices into four
models whose only difference is:

```text
L1 shift_mem = 1, 2, 3, or 4
L2 shift_mem = 1
```

No model parameter is optimized during replay. LIF beta is not part of the
state dict, so loading the baseline state dict into the target-shift model
keeps all learned weights identical while changing only L1 membrane dynamics.

Primary Stage-A diagnostic:

[
\Delta_{L1,quant}
=
BA(L1\ spike\ Fixed250)
-
BA(L1\ pre-reset\ Fixed250).
]

A useful membrane-memory effect should make this negative gap less severe by
raising L1 spike decodability, not merely by lowering pre-reset decodability.

Also report:

- L2 pre-reset/spike Fixed250 probes;
- L2 whole-count probe;
- native time-shared BA;
- output-LIF transfer BA;
- L1/L2 firing rate;
- dead-neuron fraction;
- fraction of neurons firing on >=50% of valid steps;
- near-saturation fraction firing on >=95% of valid steps.

Stage A has 12 frozen replays:

```text
4 L1 membrane shifts x 3 seeds = 12
```

plus the three baseline training jobs that produce the source checkpoints.

## Stage B — end-to-end retraining

Train the same D1+BB A2 independently for:

```text
L1 shift_mem = 1,2,3,4
L2 shift_mem = 1
seeds = 11,23,37
```

The shift-1 models are the Stage-A source baselines, so only nine additional
long-memory training jobs are needed.

Total E2E conditions:

```text
4 shifts x 3 seeds = 12
```

The initialization stream and DataLoader order depend only on the model seed,
not on membrane shift. Therefore shift comparisons are paired.

## Interpretation

The desired pattern is:

```text
L1 pre-reset Fixed250: approximately preserved
L1 spike Fixed250:     increases
|pre-reset - spike|:   decreases
L2 Fixed250:           preserved or increases
native BA:             preserved or increases
saturation:            remains controlled
```

A smaller gap caused only by a decrease in L1 pre-reset BA is **not**
considered recovery.

A very long tau may instead produce temporal smearing or persistent firing.
Therefore shift 4 (~242 ms) is a diagnostic upper end, not an assumed optimum.

## Multi-CPU execution

### Stage A only

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_2_stage_a_cpu.bash
```

Dependency graph:

```text
prepare
  -> 3 baseline training jobs
      -> 12 frozen replay jobs
          -> Stage-A finalizer
```

### Stage B only, after Stage A has been reviewed

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_2_stage_b_cpu.bash
```

This submits only the nine end-to-end `shift_mem=2/3/4` runs and the full
finalizer. The submit script now performs a hard preflight: a matching
`stage_a_manifest.json` must exist with `status=PASS` and `run_count=12`.
If Stage A is still running or failed, Stage B exits before submitting any
Slurm jobs. This prevents the full finalizer from racing unfinished replay
tasks.

### Full Stage A + Stage B

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_2_cpu.bash
```

Dependency graph:

```text
prepare
  |-> 3 baseline shift-1 trainings -> 12 frozen replays --|
  |-> 9 E2E shift-2/3/4 trainings ------------------------|
                                                           -> finalizer
```

Every independent task uses one CPU core. BLAS/OpenMP thread counts are pinned
to one.

## Artifacts

Root:

```text
notebooks/artifacts/experiment_10_2_l1_membrane_memory/
  d1_bb_l1_mem_shift_sweep_v1/
```

Stage A:

- `stage_a_replay_runs.csv`
- `stage_a_replay_summary.csv`
- `stage_a_replay_shift_contrasts.csv`
- `stage_a_replay_activity_summary.csv`
- `stage_a_shift1_replay_sanity.csv`
- `stage_a_manifest.json`

Full experiment:

- `all_runs.csv`
- `all_summary.csv`
- `all_shift_contrasts.csv`
- `all_shift_contrast_summary.csv`
- `all_activity_runs.csv`
- `all_activity_summary.csv`
- `e2e_minus_replay.csv`
- `manifest.json`

The notebook is aggregation-only and never trains a model.
