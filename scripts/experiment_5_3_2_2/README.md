# Experiment 5.3.2.2 — RSNN hidden-state capacity sweep

## Scientific question

Exp5.3.2 showed that the recurrent short-memory WHEN branch is stronger than passive long-`tau` alternatives. Exp5.3.2.1 then held the 128-neuron RSNN fixed and showed that objective reweighting does not by itself remove the remaining WHEN ceiling.

Exp5.3.2.2 asks one question only:

```math
\boxed{\text{Is the supervised causal WHEN ceiling limited by RSNN hidden-state capacity?}}
```

The experiment is intentionally a **pure width sweep**. It does not change the frozen WHAT representation, objective, recurrent topology, time constants, threshold, reset rule, surrogate gradient, optimizer, learning rate, weight decay, batch size, epoch budget, data split, valid-mask policy, checkpoint policy, or common post-hoc probe protocol.

## Fixed pipeline

```math
Raw \rightarrow Frozen\ LocalSNN \rightarrow z_t^{WHAT}\in\mathbb{R}^{128}
\rightarrow RSNN_H \rightarrow (I_t^H,U_t^H,S_t^H)
```

The Local-SNN is frozen and the same per-seed WHAT cache is reused by every width. There is no intermediate pooling and no width-dependent input rescaling.

The only experimental factor is

```math
H\in\{16,32,64,128,256\}.
```

Condition names are:

```text
rsnn_h16
rsnn_h32
rsnn_h64
rsnn_h128
rsnn_h256
```

Five master seeds are used:

```text
11, 23, 37, 53, 71
```

Therefore the primary matrix is **5 widths x 5 seeds = 25 independent runs**.

## RSNN architecture

For each width `H`:

```math
W_{in}\in\mathbb{R}^{H\times128},\qquad
W_{rec}\in\mathbb{R}^{H\times H}.
```

The dynamics match the Exp5.3.2.1 `rsnn_shortmem` branch:

```math
I_t = \alpha I_{t-1} + W_{in}z_t + W_{rec}S_{t-1}
```

```math
U_t^- = \beta U_{t-1} + I_t
```

```math
S_t = H(U_t^- - \theta)
```

followed by the unchanged Exp5.3.2.1 reset rule.

Fixed dynamics:

- `shift_mem = 1` (short membrane time constant);
- `shift_syn = 1` (short synaptic time constant);
- `threshold = 0.5` through the inherited Exp5.3.2 contract;
- recurrent weights enabled for every condition;
- no additional FF layer;
- same surrogate gradient as Exp5.3.2.1.

The recurrent parameter term grows as `H^2`, so parameter count is reported for every run. With the two supervision heads included, the trainable total is

```math
N(H)=128H+H^2+(10H+10)+(H+1)=H^2+139H+11.
```

## Fixed joint objective

The objective is exactly the Exp5.3.2.1 `joint` condition:

```math
L_{WHEN}=L_{phase}+L_{progress}.
```

For sample `i` with valid length `T_i`:

```math
q_{i,t}=\min\left(9,\left\lfloor10\frac{t}{T_i-1}\right\rfloor\right)
```

and

```math
p_{i,t}=\frac{t}{T_i-1}.
```

The heads read the post-reset membrane `U_t`:

```math
L_{phase}=\frac1B\sum_i\frac1{T_i}\sum_{t<T_i}
CE(W_pU_{i,t}+b_p,q_{i,t})
```

```math
L_{progress}=\frac1B\sum_i\frac1{T_i}\sum_{t<T_i}
SmoothL1(\sigma(w_r^TU_{i,t}+b_r),p_{i,t}).
```

The reduction is sample-balanced: valid timesteps are averaged within each gesture first, then gestures are averaged. Padding never contributes to the objective.

## Causality contract

`T_i` is used **only to construct supervision targets and the valid mask**. It is never an input to:

- the RSNN;
- `I_t`, `U_t`, or `S_t`;
- the phase head;
- the progress head;
- the deployment path.

The representation therefore remains causal:

```math
z_{1:t}\rightarrow U_t.
```

## Paired execution across widths

Widths cannot share elementwise-identical parameter matrices because their shapes differ. Pairing therefore means:

1. the same master seed for corresponding runs;
2. exactly the same minibatch order within a seed, using the unchanged Exp5.3.2 loader seed;
3. deterministic component initialization using `dseed(seed, experiment, width, component)`;
4. no prefix-copy trick such as copying the first 64 neurons from an H128 model into H64.

This keeps the initialization distribution independent while making each architecture reproducible.

## Checkpoint selection

Checkpoint selection is unchanged from the actual Exp5.3.2.1 `joint` implementation:

1. minimize validation joint objective loss;
2. tie-break by lower validation sample-balanced progress MAE;
3. then higher validation phase BA.

The test set is never used for checkpoint selection.

## Common post-hoc probes

After checkpoint selection the RSNN is frozen. The training heads are not the primary cross-width comparison interface.

Every width uses the same Exp5.3.2.1 common probe contract:

- StandardScaler + LogisticRegression grid for ten-way phase;
- StandardScaler + Ridge grid for continuous progress;
- hyperparameters selected on validation data only.

The probe is applied to:

- `U_t`: post-reset membrane;
- `I_t`: synaptic state (`synaptic` in artifacts);
- `S_t`: instantaneous spike state;
- `C250_t`: trailing 250 ms spike count;
- `C500_t`: trailing 500 ms spike count.

For all five feature types the runner stores train/val/test phase metrics and progress metrics. Sample-balanced trajectory metrics are computed per gesture, not by pooling long gestures more heavily.

## Primary metrics

### Phase accessibility

```math
PhaseBA(U_t)
```

is the main phase metric. Analysis derives reference lines from the validated baseline artifact:

- chance = 10%;
- elapsed-time-only Phase BA (current Exp5.3.2.1 artifact: about 36.84%);
- WHAT-only Phase BA (current five-seed mean: about 32.3%).

### Sample-balanced progress MAE

For gesture `i`:

```math
MAE_i=\frac1{T_i}\sum_{t<T_i}|\hat p_{i,t}-p_{i,t}|,
```

then

```math
MAE=\frac1N\sum_i MAE_i.
```

The elapsed-time-only reference is read from the reused baseline artifact (currently about `0.13819`). A useful RSNN should ideally beat that clock baseline.

### Spearman trajectory ordering

Per gesture:

```math
\rho_i=Spearman(\hat p_{i,1:T_i},t).
```

The report averages `rho_i` across gestures.

### Monotonic violation rate

```math
V_i=\frac1{T_i-1}\sum_t 1[\hat p_{t+1}<\hat p_t].
```

This distinguishes a smooth progress trajectory from a wider model that merely fits local fluctuations.

## History attribution

History attribution is a required winner-selection diagnostic.

### Ordered

Normal causal execution:

```math
z_1,z_2,\ldots,z_t\rightarrow U_t.
```

### State reset

Before each current input, carried synaptic, membrane, and recurrent spike state is cleared while retaining the current `z_t`.

### Temporal shuffle

The valid WHAT sequence is shuffled within each gesture while the target timeline remains unchanged. Five deterministic shuffle replicates are used, exactly as in Exp5.3.2.1.

Phase history gains:

```math
H_{reset}^{phase}=BA_{ordered}-BA_{reset}
```

```math
H_{shuffle}^{phase}=BA_{ordered}-BA_{shuffle}.
```

Progress history gains are positive when removing history makes MAE worse:

```math
H_{reset}^{progress}=MAE_{reset}-MAE_{ordered}
```

```math
H_{shuffle}^{progress}=MAE_{shuffle}-MAE_{ordered}.
```

The key capacity curve is therefore not only `H -> PhaseBA`, but also `H -> history gain`.

## Firing-rate utilization

For each run, test-set instantaneous spikes are used to compute

```math
FR=\frac{\sum_{i,t,j}S_{i,t,j}}{\sum_i T_iH}.
```

Artifacts also report:

- mean firing rate;
- mean firing rate in Hz;
- median neuron firing rate;
- dead-neuron fraction, defined as `FR_j < 1e-4` spikes/timestep;
- highly-active-neuron fraction, defined here as `FR_j > 0.10` spikes/timestep (about 6.4 Hz at 64 Hz sampling).

The explicit high-activity threshold is part of the protocol so it cannot drift after results are seen.

## Effective state dimension

All valid test `U_t` rows are centered and their covariance eigenspectrum is computed. Two diagnostics are stored:

```math
D_{eff}=\frac{(\sum_k\lambda_k)^2}{\sum_k\lambda_k^2}
```

and

```math
D_{90}=\text{minimum number of leading PCs explaining at least 90% variance}.
```

This distinguishes nominal width from actually used state dimension. For example, if H128 and H256 have nearly identical `D_eff` and `D90`, adding neurons has not created a meaningfully larger accessible state subspace.

## Generalization diagnostics

Capacity can improve training fit while degrading held-out representation quality. The experiment therefore stores:

- checkpoint-native train/val/test metrics;
- common `U_t` probe train/val/test phase BA;
- common `U_t` probe train/val/test sample-balanced progress MAE;
- train-minus-val phase probe gap;
- val-minus-train progress probe gap.

The analysis notebook plots these quantities against width to expose underfitting, plateauing, or overcapacity.

## Baselines

The WHAT-only and elapsed-time-only baselines are not retrained for each width. The five Exp5.3.2.1 baseline JSON artifacts are identity-validated and copied into the Exp5.3.2.2 artifact namespace.

This avoids pointless recomputation and guarantees that all width conditions are compared against the same baseline protocol.

## Multi-CPU execution

This experiment follows `AGENTS.md` task-level multi-CPU execution:

```text
5 frozen-WHAT / baseline validation tasks
        -> afterok
25 independent width x seed train -> evaluate tasks
        -> afterok
aggregation-only finalizer
        -> analysis-only notebook
```

The 25-run array is width-major:

```text
tasks  0-4   -> H16  seeds 11,23,37,53,71
tasks  5-9   -> H32  seeds 11,23,37,53,71
tasks 10-14  -> H64  seeds 11,23,37,53,71
tasks 15-19  -> H128 seeds 11,23,37,53,71
tasks 20-24  -> H256 seeds 11,23,37,53,71
```

Every primary task performs its own complete atomic work:

```text
load frozen WHAT
-> build RSNN_H
-> joint training
-> validation checkpoint selection
-> ordered evaluation
-> state-reset evaluation
-> five temporal-shuffle evaluations
-> U/I/S/C250/C500 common probes
-> progress trajectory metrics
-> firing-rate diagnostics
-> effective-dimension diagnostics
-> save checkpoint/history/evaluation
```

Submit from the repository root on Unity:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_3_2_2_cpu.bash
```

Each Slurm task uses one CPU core. The array is capped at 25 concurrent tasks, below the repository maximum of 50.

## Per-run artifacts

Each `(width, seed)` produces:

```text
checkpoints/rsnn_hH__seedS.pt
histories/rsnn_hH__seedS.csv
evaluations/rsnn_hH__seedS.json
```

Each evaluation contains native metrics, all five common probe readouts, top-level trajectory metrics, ordered/reset/shuffle attribution with history gains, activity diagnostics, representation diagnostics, and explicit parameter counts.

## Finalizer contract

The finalizer **only aggregates** existing artifacts. It does not train, evaluate, or fit probes.

It requires exactly:

```text
25/25 completed primary evaluations
25/25 histories
5/5 seeds for every width
5/5 validated baseline artifacts
```

Any missing artifact is a hard failure; missing runs are never silently skipped.

Aggregate outputs:

```text
runs.csv
histories.csv
probe_runs.csv
ablation_runs.csv
history_gain_runs.csv
activity_runs.csv
representation_runs.csv
baseline_runs.csv
local_reference.csv
manifest.json
```

The finalizer writes under:

```text
notebooks/artifacts/experiment_5_3_2_2_when_width_sweep/rsnn_when_width_v1/
```

## Analysis-only notebook

`notebooks/experiment_5_3_2_2_when_width_sweep.ipynb` is analysis-only. It reads the finalized CSV/JSON artifacts and is organized as:

1. Capacity curve — parameter count, Phase BA, progress MAE;
2. Generalization — train/val/test probe curves;
3. Temporal quality — Spearman and monotonic violation;
4. History contribution — `H_{reset}` and `H_{shuffle}`;
5. SNN-native accessibility — `U/I/S/C250/C500`;
6. Activity utilization — firing rate and dead/highly-active fractions;
7. Effective state dimension — `D_eff` and `D90`;
8. Hierarchical winner table.

The notebook never trains a model, refits a probe, or submits Slurm work.

## Winner-selection hierarchy

A width is first considered eligible only when the five-seed mean satisfies:

```math
ProgressMAE < MAE_{elapsed}
```

and

```math
H_{reset}^{phase}>0,\qquad H_{shuffle}^{phase}>0,
```

and

```math
H_{reset}^{progress}>0,\qquad H_{shuffle}^{progress}>0.
```

Among eligible widths, interpretation prioritizes:

1. higher Phase BA;
2. lower sample-balanced Progress MAE;
3. larger positive history gains;
4. then fewer trainable parameters;
5. then stronger spike-domain accessibility.

No single scalar score is used.

## Decision cases

### Capacity-limited

If quality and history gain continue to improve through H256, capacity still matters. Further width should be considered only against hardware cost, because recurrent parameters scale quadratically.

### Sweet spot / overcapacity

If H64 or another smaller model has the best test probe quality and history gain while H128/H256 improve train fit but flatten or regress on held-out users, that width becomes the preferred WHEN baseline.

### Dynamics-limited plateau

If H16 through H256 all converge to essentially the same WHEN ceiling and effective dimension saturates early, the bottleneck is not neuron count. That result motivates the next architectural test:

```math
WHAT_{128}\rightarrow FF\text{-}SNN\rightarrow RSNN
```

rather than continuing to increase recurrent width.
