# Experiment 14 — Cross-user History Organization Loss

## Question

CoreBenchmark and Exp13/13.1 show that the two-layer multi-tau SNN uses long
history, but WCCE does not specify how that history should be organized.
Exp14 keeps the temporal substrate fixed and changes only representation
supervision.

Question: can the same SNN temporal memory be organized into a representation
whose history contribution transfers better across users?

This is a loss study, not a tau/depth/readout study.

## Frozen backbone/data contract

Exp14 consumes the finalized CoreBenchmark dataset cache and split and hard
fails if the locked production geometry changes.

- action0 + action1, 12 benchmark labels, 30 event input channels, 64 Hz
- train users: 0,1,2,5,7,8,11,12,13,14,15,18,19,20
- validation users: 4,9,16
- test/OOD users: 3,6,10
- seeds: 11,23,37
- two hidden layers, width 128
- L1 shifts (2,3,4), L2 shifts (2,3,4)
- tau_mem=22.54 ms, threshold=0.5
- unnormalized synaptic update
- bias-free analog output accumulator
- valid-mean-logit WCCE

BenchmarkNet initializes common SNN parameters by seed+parameter role, so all
Exp14 cases with the same seed begin from identical SNN weights.

## Structured training sampler

Auxiliary losses require same-character examples from different users in each
batch. Training therefore uses a deterministic class/user-balanced sampler:

- 8 classes/batch
- 4 users/class
- 4 samples/user
- 128 samples/batch

Sampling uses training users only. Validation and test users never enter
training batches, projection-head fitting, or auxiliary pair construction.

The sampler is shared by all Exp14 loss cases, including C0, so loss
comparisons do not confound the auxiliary objective with batch composition.
Existing CoreBenchmark O0 remains the external reference for the effect of the
sampler change.

## Auxiliary representation

The native classifier still consumes the original L2 spike output. Auxiliary
losses operate on continuous L2 pre_reset state through a training-only
projection head:

128 -> 64 -> ReLU -> 32 -> L2 normalize

The projection head is discarded at inference.

Full per-timestep `traces.npz` files are intentionally not persisted. All probes and diagnostics consume traces in memory during evaluation, while `checkpoint.pt` is retained so traces can be regenerated later if needed.

## Cases

C0: matched WCCE baseline. No auxiliary loss.

C1: whole-sequence cross-user SupCon. Positives are same character and
different user.

C2: phase-conditioned cross-user SupCon at 25%, 50%, 75%, 100% normalized
phase. This does not impose temporal smoothing.

C3: history-delta cross-user SupCon. For each sample compare the full-history
L2 endpoint with an endpoint after both hidden states are reset so only the
final 250 ms remain. Delta z = z_full - z_250ms.

C4: combined C2 + C3. After phase 1, select C2 and C3 lambdas independently by
mean held-out-user validation BA across seeds. Run selected strengths and half
of both selected strengths.

C5: shuffled auxiliary control. Uses C2 at lambda 0.1 but shuffles labels only
for the auxiliary loss. Native WCCE always uses true labels.

## Lambda sweep

For C1/C2/C3: lambda in {0.01, 0.03, 0.1, 0.3}.

Auxiliary weight is zero through epoch 10, ramps linearly through epoch 30,
then remains fixed. Checkpoint selection uses held-out-user validation BA,
then validation valid-mean-logit CE, then earliest epoch.

## Evaluation

Each Slurm task trains and immediately evaluates its selected checkpoint.

- Native train/val/test BA, accuracy, macro-F1, train-test gap.
- Existing CoreBenchmark L1/L2 temporal probes.
- Existing lag-similarity and state-reset diagnostics at
  50/100/250/500/1000 ms.
- Cross-user geometry on L1/L2 spike and pre_reset trajectories.
- Train-user to test-user trajectory retrieval BA.
- L2 history-generalization analysis for 50/100/250/500/1000 ms and full
  history, reporting ID gain, OOD gain and excess-seen gain.

The desired mechanism is not reduced history use. It is preserved useful
history with larger OOD history gain and smaller excess-seen gain.

## Slurm execution

bash scripts/bash_script/SNN_Bash/submit_exp_14_cpu.bash

Execution graph:

prepare -> phase-1 42 tasks -> lambda selection -> C4 6 tasks -> finalizer

One CPU is used per task. Array concurrency defaults to 20 and is capped at 50.

## Validation

python -m pytest -q tests/test_experiment_14_history_organization_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
