# Exp7.3.1 — Frozen-L2 Linear Optimization and Same-W LIF Substitution

## Goal

Explain why the frozen-L2 linear accessibility probe in Exp7.3 can exceed the Stage-2 Analog head, while separating four factors without changing the backbone:

1. training objective (`TSCE` vs `WCCE`),
2. optimization budget / optimizer (`Adam` vs `LBFGS`),
3. regularization and per-neuron feature scaling,
4. the final Analog-to-LIF interface loss when the exact same trained weight matrix is deployed through the output LIF.

This experiment does **not** retrain L1/L2 and does **not** train through the output LIF.

## Frozen inputs

Reuse the six committed Exp7.3 Stage-2 L2 caches:

- backbone source `A1_e2e_linear_tsce` for the TSCE backbone,
- backbone source `A2_e2e_linear_wcce` for the WCCE backbone,
- seeds `11, 23, 37`,
- architecture `234x234`,
- binary L2 spikes and the original valid lengths.

All conditions train only a bias-free `128 -> 12` weight matrix `W`.

## Parallel head objectives

Every case is run under both objectives.

### TSCE

For valid L2 states `z_t`:

`L_TS = mean_{valid t} CE(W z_t, y)`.

### WCCE

For each sequence:

`z_bar = mean_{valid t} z_t`,

`L_WC = CE(W z_bar, y)`.

`CE_GAIN=1`, `bias=False` in every case.

## Cases

### C0 — Adam, 100-epoch budget

- Same Adam hyperparameters as Exp7.3 Stage-2.
- C0 and C1 share one uninterrupted 500-epoch trajectory.
- C0 is the best validation-BA checkpoint found within epochs 1–100.
- No regularization.

This is budget-matched to the 100-epoch Exp7.3 Stage-2 setting while preserving an exact shared trajectory with C1; it is not a byte-for-byte reuse of an early-stopped Exp7.3 checkpoint.

### C1 — Adam, 500-epoch budget

- Same trajectory as C0.
- Best validation-BA checkpoint found within epochs 1–500.
- No regularization.

Contrast `C1 - C0` isolates additional Adam training budget.

### C2 — LBFGS, no regularization

- Same objective and raw features.
- `max_eval=100` full-objective evaluations.
- Strong-Wolfe line search.
- No regularization.

Contrast `C2 - C0` asks whether a curvature-aware optimizer can solve the same frozen-feature linear problem better under a roughly 100-full-pass budget.

### C3 — LBFGS + L2 regularization

- Raw objective-native features.
- `max_eval=100` per candidate.
- Sweep `lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1}`.
- Optimize `L_total = L_task + 0.5 * lambda * ||W||_F^2`.
- Select `lambda` by validation Analog BA, then validation task CE, then smaller lambda.

Contrast `C3 - C2` isolates validation-selected L2 regularization.

### C4 — scale-only + LBFGS + L2 regularization

- Compute per-neuron `sigma` from the **training feature matrix for the current objective only**.
- Use `x_scaled = x / sigma`; do not subtract the mean.
- `bias=False`.
- Same lambda sweep and `max_eval=100` as C3.
- Fold scaling into the deployable raw matrix after training:

`W_raw = W_scaled diag(1/sigma)`.

Therefore `W_scaled (z_t / sigma) == W_raw z_t` exactly, so no runtime scaler is required before LIF deployment.

Contrast `C4 - C3` isolates feature-scale conditioning / regularization geometry while preserving exact deployment compatibility.

### Ref — fully converged deployment-compatible probe

- Same formulation as C4.
- Same lambda sweep.
- `max_eval=5000` per candidate (or earlier solver convergence).
- Same validation-only lambda selection rule.

Contrast `Ref - C4` asks whether the 100-evaluation LBFGS budget itself remains limiting.

The existing Exp7.3 `StandardScaler(with_mean=True) + intercept + LogisticRegression` L2 probes are retained only as auxiliary accessibility references. They are not treated as exact same-W LIF-deployable heads because centering/intercept creates a sequence-level affine offset that cannot be naively injected at every timestep.

## Evaluation contract

Every selected case is evaluated twice with the exact same raw `W`.

### Analog

`score_linear = sum_{valid t} W z_t`.

Report train/val/test BA; test BA is the main continuous-head metric.

### Same-W LIF substitution

Without retraining or changing `W`:

`z_t -> W z_t -> LIF(beta=0.5, threshold=0.5, cap=1) -> valid spike count -> argmax`.

No surrogate training, gain calibration, threshold sweep, beta sweep, or post-hoc weight scaling is allowed.

Report:

`Delta_interface = 100 * (Analog test BA - LIF test BA)`.

## Loss logging

For every selected case report:

- train task CE,
- validation task CE,
- train total objective,
- validation total objective,
- optimizer budget metadata,
- final Analog and same-W LIF BA.

Adam stores an epoch-wise history. LBFGS stores closure-evaluation task/total loss traces.

Absolute TSCE and WCCE loss values are not interpreted as directly comparable; optimizer comparisons are made within the same objective.

## Task counts and multi-CPU execution

Logical result rows:

`2 backbones x 2 head objectives x 6 cases x 3 seeds = 72`.

Physical tasks:

- 12 Adam tasks; each emits both C0 and C1,
- 12 C2 LBFGS no-reg tasks,
- 60 C3 lambda-candidate tasks,
- 60 C4 lambda-candidate tasks,
- 60 Ref lambda-candidate tasks,
- 1 finalizer.

Each independent run is one CPU / one Slurm array task. Candidate arrays are capped at `%50`. The finalizer uses only existing per-run artifacts and performs no training.

## Pre-registered contrasts

Within each backbone/objective:

- `C1 - C0`: extra Adam budget,
- `C2 - C0`: LBFGS vs Adam-100,
- `C3 - C2`: regularization,
- `C4 - C3`: scale-only feature conditioning,
- `Ref - C4`: additional LBFGS convergence budget.

Within each backbone/case:

- `WCCE - TSCE`: objective effect after holding optimization recipe fixed.

Every contrast is reported for both Analog and same-W LIF test BA.
