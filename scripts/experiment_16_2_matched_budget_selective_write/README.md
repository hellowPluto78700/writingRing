# Experiment 16.2 — matched-budget selective write

## Question

Under the same **mean gate budget**, can WCCE learn a functionally useful temporal write allocation?

Only the scalar z-only gate is retained: `g_t = sigmoid(G_z z_t + b_g)`. Prefix, z+h gating, entropy/L0/hard gates, and counterfactual utility are excluded.

## Phase 0 — budget-controller calibration

- calibration seed: **101** only;
- rho: **0.9, 0.7, 0.5**;
- lambda_B candidates: **0.1, 0.3, 1, 3**;
- 12 calibration runs total;
- initialize `G_z=0`, `b_g=logit(rho)`, so the initial gate is uniform at rho;
- budget is computed per sample over valid timesteps only;
- compliance is the mean validation `E_i[abs(mean_gate_i-rho)]` over the last 5 epochs <= 0.02;
- log coverage `P(abs(mean_gate_i-rho)<=0.05)`;
- choose the smallest compliant lambda_B, never by BA;
- prefer one common lambda_B across all rho values; otherwise select the smallest compliant value per rho.

## Phase 1 — formal matched-budget experiment

Formal seeds are **11, 23, 37**. Each seed runs: `C0, GZ0, S90, G90, S70, G70, S50, G50` for **24 formal runs**.

`Sxx` uses a static gate equal to rho. `Gxx` uses the dynamic z-only gate with WCCE plus the calibrated budget loss. `GZ0` is the unconstrained z-only WCCE control.

### Hard contracts

- one C0 shared initialization is explicitly cloned to all cases for a seed;
- each S/G pair must have identical first-forward evidence, L2 spikes, L2 synaptic current and L2 membrane state;
- C0 must be exactly equivalent to an S100 control;
- every epoch uses an explicit deterministic sample permutation derived from seed and epoch;
- every epoch stores the permutation SHA256 hash;
- a paired task verifies both members saw the same batch trajectory for all common epochs.

### Slurm pairing

One CPU per task. Each matched S/G pair runs sequentially in the same Slurm task. There are 12 Phase-1 tasks: 3 baseline C0/GZ0 tasks and 9 S/G tasks.

Counterbalance: seed 11 = S→G, G→S, S→G for rho .9/.7/.5; seed 23 = G→S, S→G, G→S; seed 37 = S→G, G→S, S→G.

Priority: exact clone > deterministic batches > same node > order counterbalance.

### Constrained checkpoint selection

For G90/G70/G50, only epochs with validation budget MAE <= 0.02 are eligible. Among eligible epochs use max validation BA, then min validation mean-logit CE, then earliest epoch. If no epoch is compliant, mark the run `BUDGET_INVALID`; the closest-budget epoch is diagnostic only.

Static cases, C0 and GZ0 use the normal BA-first selector.

## Metrics and diagnostics

Primary adaptive-gating gain is `Metric(G_rho)-Metric(S_rho)`, reported per seed and as a mean for native BA, L2 WholeCount no-bias, L2 Fixed250 ordered no-bias, L2 Relative10 ordered no-bias, and collapse gap.

Every gated checkpoint logs mean/sample dispersion of the gate, within-sample temporal gate std, P90-P10, fractions above/below rho±0.1, mean `||u_t||`, mean `||g_t u_t||`, `||W2||_F`, L1/L2 firing rate, mean L2 synaptic-current norm and mean L2 membrane norm.

The matched quantity is **mean gate budget**, not memory capacity.

## Phase 2 — mandatory functional replay

Run inference-only replay for `GZ0, G90, G70, G50`: learned schedule; exact per-sample mean; within-valid-range shuffle with seeds 101/211/307; circular shifts by 0.25T/0.50T/0.75T.

Interpretation hierarchy: G>S supports adaptive gating under matched mean budget; Learned>Mean supports functional use of temporal nonuniformity; Learned>Shuffle/Shift supports functional importance of temporal placement.

## Submission

From the Unity repository root:

```bash
git pull origin main
bash scripts/bash_script/SNN_Bash/submit_exp_16_2_cpu.bash
```

Pipeline: prepare -> 12 Phase0 jobs -> calibration finalizer -> 12 paired Phase1 tasks -> 12 Phase2 replay tasks -> finalizer.

Main artifacts are under `notebooks/artifacts/experiment_16_2_matched_budget_selective_write/matched_budget_selective_write_v1/`, including calibration selection, formal native/probe tables, adaptive-gating deltas, write/state diagnostics, replay tables and causal deltas.