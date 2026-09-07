# Experiment 5.3.2 - Supervised causal WHEN representation benchmark

## Scientific question

Experiment 5.3.2 isolates the temporal-context problem from WHAT x WHEN fusion. The frozen Local SNN already provides the WHAT trajectory `z_t`; this experiment asks which SNN memory mechanism can turn the causal history `z_1..z_t` into the most readable WHEN state.

```text
Raw64
  -> frozen Exp3-exact Local SNN
  -> z_t in R^128 (binary Local L2 spikes)
  -> WHEN branch
  -> U_t / synaptic state / spike trajectory
  -> temporary phase + progress heads during training
```

No Fusion SNN, letter classifier, WholeCount letter readout, Fixed250 letter decoder, Relative10 letter decoder, or dynamic weight mechanism is trained here.

## Frozen WHAT source

All conditions reuse the Exp5.2 frozen Local-L2 cache:

```text
experiment_5_2_frozen_local_tauR_sweep
frozen_exp3_l2_endpoint_tauR_v1
```

The Local SNN is never fine-tuned. Every WHEN condition within a seed sees exactly the same cached WHAT trajectory and valid lengths.

Seeds are fixed and paired:

```text
(11, 23, 37, 53, 71)
```

## Nine pre-specified WHEN conditions

All models use one 128-neuron causal WHEN layer. Threshold, reset, optimizer, hidden width, supervision, and split are fixed. Only the memory mechanism changes.

### Membrane-memory family

Synaptic decay is kept short (`shift_syn=1`) while membrane decay is swept.

| Condition | shift_mem | approximate tau_mem at 64 Hz | shift_syn |
|---|---|---|---|
| `mem_single_s4` | 4 | 242 ms | 1 |
| `mem_single_s5` | 5 | 492 ms | 1 |
| `mem_single_s6` | 6 | 992 ms | 1 |
| `mem_multi_s456` | 4,5,6 | 242/492/992 ms | 1 |

For `mem_multi_s456`, the 128 neurons receive deterministic round-robin assignments, giving 43/43/42 neurons across the three membrane shifts.

### Synaptic-memory family

Membrane decay is kept short (`shift_mem=1`) while synaptic decay is swept.

| Condition | shift_mem | shift_syn | approximate tau_syn at 64 Hz |
|---|---|---|---|
| `syn_single_s4` | 1 | 4 | 242 ms |
| `syn_single_s5` | 1 | 5 | 492 ms |
| `syn_single_s6` | 1 | 6 | 992 ms |
| `syn_multi_s456` | 1 | 4,5,6 | 242/492/992 ms |

For `syn_multi_s456`, the 128 neurons receive the same deterministic 43/43/42 assignment across synaptic shifts.

### Learned-recurrence control

`rsnn_shortmem` keeps both membrane and synaptic decay short (`shift=1`) and adds a trainable previous-spike recurrent matrix.

Thus the experiment directly asks where useful WHEN memory should live:

```text
slow membrane
vs
slow synaptic trace
vs
learned recurrence
```

## Causal state equations

The frozen WHAT spike vector is projected with a fixed trainable input matrix:

```math
c_t = W_{in} z_t + 1_{rec} W_{rec} s_{t-1}.
```

Synaptic state:

```math
I_t = alpha I_{t-1} + c_t.
```

Membrane state:

```math
U_t^- = beta U_{t-1} + I_t,
```

followed by repository LIF threshold/reset:

```math
s_t = H(U_t^- - theta).
```

The primary WHEN interface is the post-reset membrane `U_t`. Synaptic state and spike codes are diagnostics/deployment-oriented readouts.

After a sample reaches its valid length, synaptic and membrane states are frozen and no padded timestep contributes to loss or metrics.

## Supervised WHEN targets

For a valid sequence of length `T`, training labels use the known segment end only to construct targets. Final duration is never an input to the WHEN branch.

Continuous progress:

```math
p_t = t/(T-1),
```

with the length-one case safely mapped to zero.

Ten-way phase:

```math
q_t = min(9, floor(10 p_t)).
```

The WHEN network therefore sees only causal `z_1..z_t`, but receives supervised feedback about the current relative state during training.

## Training objective

Two temporary heads read `U_t`:

```math
phase_logits_t = W_phase U_t + b_phase,
```

```math
progress_hat_t = sigmoid(W_progress U_t + b_progress).
```

The fixed objective is

```math
L_WHEN = L_phase(U_t) + L_progress(U_t),
```

where

```math
L_phase = CE(phase_logits_t, q_t)
```

and

```math
L_progress = SmoothL1(progress_hat_t, p_t).
```

Both terms have coefficient 1.0. The reduction is sample-balanced: first average valid timesteps within each gesture, then average gestures in the batch. Long gestures therefore do not receive larger loss weight merely because they contain more timesteps.

The temporary heads are not part of the future Fusion SNN deployment contract.

## Checkpoint selection

Checkpoints are selected using validation only:

1. maximize native validation phase balanced accuracy from `U_t`;
2. tie-break with lower validation sample-balanced progress MAE;
3. tie-break with lower total validation loss.

Test data never participates in checkpoint selection.

## Evaluation matrix

### Native `U_t` heads

Report phase BA/accuracy/macro-F1 and progress MAE/R2 from the trained heads.

### Common post-hoc probes

After checkpoint selection the WHEN branch is frozen. Identical probe machinery is fitted to:

```text
membrane U_t
synaptic state I_t
instantaneous spike S_t
causal trailing 250 ms spike count
causal trailing 500 ms spike count
```

The 250/500 ms windows use only `[t-W, t]`; no future timestep is included.

### Multi-tau group probes

For `mem_multi_s456`, probe membrane groups corresponding to shifts 4,5,6 separately and jointly through the full membrane probe.

For `syn_multi_s456`, probe synaptic groups corresponding to shifts 4,5,6 separately and jointly through the full synaptic probe.

These group diagnostics show which timescale actually carries WHEN information.

## Baselines

Each frozen-local preparation task also computes two phase/progress baselines with identical probe machinery.

### WHAT-only

```text
z_t -> phase/progress probe
```

This asks how much phase can be guessed from the current Local WHAT pattern without temporal memory.

### Elapsed-time-only

```text
t / fs -> phase/progress probe
```

Only absolute elapsed seconds are supplied; final duration `T` is never provided. This asks whether a model is merely acting as a clock rather than using gesture dynamics.

## Causal attribution

Every trained checkpoint is evaluated under:

```text
ordered
state_reset
temporal_shuffle (5 deterministic replicates)
```

`state_reset` zeros synaptic, membrane, and recurrent-spike history before each valid timestep while preserving the current WHAT input.

`temporal_shuffle` permutes each valid WHAT trajectory while retaining sequence length and target positions.

The main causal quantities are

```math
H_reset = PhaseBA_ordered - PhaseBA_reset
```

and

```math
H_shuffle = PhaseBA_ordered - PhaseBA_shuffle.
```

A high ordered score without positive history attribution may indicate a current-WHAT shortcut rather than a true causal WHEN representation.

## Primary interpretation

A strong WHEN branch should show the convergent pattern:

```text
high ordered U_t phase BA
low progress MAE
ordered > state_reset
preferably ordered > temporal_shuffle
preferably useful phase remains accessible in 250/500 ms spike counts
```

This experiment does not choose a fusion architecture. Only after a convincing WHEN branch is identified should Experiment 5.3.3 build a fixed-weight SNN-native WHAT x WHEN fusion layer.

## Multi-CPU execution

This follows `AGENTS.md` task-level multi-CPU rules.

### Frozen-local preparation and baselines

```text
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=1
```

Each task validates/reuses one Exp5.2 frozen Local cache and computes WHAT-only plus elapsed-time baselines.

### Independent train/evaluate tasks

Nine conditions x five seeds = **45 independent runs**:

```text
#SBATCH --array=0-44%45
#SBATCH --cpus-per-task=1
```

Each task atomically performs:

```text
load frozen WHAT cache
-> train one WHEN condition/seed
-> validation-only checkpoint selection
-> ordered native evaluation
-> U/I/S/Spike250/Spike500 probes
-> multi-tau group probes when applicable
-> state-reset evaluation
-> five temporal-shuffle evaluations
-> write checkpoint/history/evaluation artifacts
```

All compute jobs initialize Conda locally and force OMP/MKL/OPENBLAS/NUMEXPR to one thread.

### Dependency chain

```text
5-task local/baseline array
  -> afterok
45-task WHEN train/evaluate array
  -> afterok
aggregation-only finalizer
```

Submit from repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_3_2_cpu.bash
```

The finalizer never retrains models and fails on missing run, history, or baseline artifacts.

## Finalized artifacts

```text
notebooks/artifacts/experiment_5_3_2_when_representation/
  supervised_causal_when_v1/
    runs.csv
    histories.csv
    probe_runs.csv
    group_probe_runs.csv
    ablation_runs.csv
    baseline_runs.csv
    local_reference.csv
    manifest.json
```

`runs.csv` contains native trained-head performance. `probe_runs.csv` contains the common U/I/S/window phase and progress probes. `group_probe_runs.csv` contains multi-tau group diagnostics. `ablation_runs.csv` contains ordered/reset/shuffle results. `baseline_runs.csv` contains WHAT-only and elapsed-time controls.

## Analysis-only notebook

`notebooks/experiment_5_3_2_when_representation.ipynb` reads finalized CSV/JSON artifacts only. It does not train models, fit probes, run multiprocessing, or submit Slurm jobs.

The notebook is organized around:

1. native and probe `U_t` WHEN quality across all nine conditions;
2. membrane-vs-synaptic 242/492/992 ms sweep curves;
3. state-to-spike accessibility (`U`, `S`, `Spike250`, `Spike500`);
4. ordered/reset/shuffle causal attribution;
5. multi-tau subgroup diagnostics;
6. WHAT-only and elapsed-time baselines;
7. validation training curves and paired-seed summaries.
