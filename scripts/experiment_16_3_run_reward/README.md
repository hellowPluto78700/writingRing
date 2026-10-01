# Experiment 16.3 — diminishing reward for repeated spikes within one firing episode

## Core question

Exp16.2 and the sustained-firing diagnostics showed two distinct facts:

1. different synaptic time constants reach similar population-level L2 activity because learned parameter scales compensate for gain differences;
2. longer synaptic time constants primarily increase persistence length rather than simply increasing mean firing occupancy.

This makes the problem deeper than an "unnormalized synapse bug". With unconstrained weights,

I_t = alpha I_{t-1} + W z_t

and

I_t = alpha I_{t-1} + (1-alpha) W' z_t

are a row-wise reparameterization of one another when W' = W/(1-alpha). Likewise a static scalar gate can be absorbed into the following weight matrix. Therefore Exp16.3 does not treat synaptic normalization, a static gate, or a generic firing-rate penalty as the primary scientific manipulation.

The formal question is:

Does linear repeated-spike reward make a persistent firing episode artificially valuable?

and, if so:

Can diminishing within-run reward reduce persistent/redundant activity while preserving whole-character classification?

## Locked task semantics

Every formal case keeps:

- the same two-layer 128-neuron multi-tau SNN;
- the same CoreBenchmark dataset and frozen user split;
- formal seeds 11, 23, 37;
- the same optimizer, learning rate, training budget and early stopping;
- the same bias-free linear class head;
- one whole-character label and one sequence-level cross-entropy only;
- no TSCE, prefix loss, auxiliary loss, gate, activity penalty or additional regularizer.

Only the way L2 spikes are aggregated before the existing class head is changed.

All reward features are divided by the valid sequence length T_i, preserving the duration normalization of current WCCE.

## Why the manipulation is run-based, not total-rate based

Current WCCE is equivalent to:

A_j^LIN = (1/T) sum_t s_(t,j).

Within a fixed sequence, every additional spike from the same persistent episode has the same linear marginal contribution.

Exp16.3 changes the marginal reward inside a contiguous firing run while allowing a later, genuinely new firing episode to contribute again.

For neuron j, let its contiguous run lengths be ell_j1, ell_j2, ... . The reward feature is:

A_j = (1/T) sum_b phi(ell_jb).

### Formal cases

Each seed runs seven cases, for 21 formal runs total.

| Case | Run reward phi(ell) | Meaning |
|---|---|---|
| LIN | ell | current linear WCCE baseline |
| CAP4 | min(ell,4) | a run contributes at most 4 spikes |
| CAP8 | min(ell,8) | a run contributes at most 8 spikes |
| CAP16 | min(ell,16) | a run contributes at most 16 spikes |
| CAP24 | min(ell,24) | mild cap |
| SUB8 | 8(1-exp(-ell/8)) | smooth diminishing repeated reward |
| EP1 | 1[ell>0] | episode/onset count; one reward per firing episode |

The head receives A and the only training loss is:

L = CE(R A, y).

LIN must be exactly equivalent to the existing WCCE forward readout because the class head is bias-free:

R[(1/T) sum_t s_t] = (1/T) sum_t R s_t.

## Gradient contract

The forward firing run is binary. To make the training manipulation correspond directly to marginal repeated-spike reward, the run position is computed from detached forward spikes.

At timestep t, the current spike variable receives a gradient weight equal to the reward of a hypothetical spike given the detached previous run length.

Examples:

- LIN: marginal reward 1 at every valid timestep;
- CAP8: reward 1 for run positions 1–8, then 0;
- EP1: reward 1 at a new episode onset and 0 for continuation spikes;
- SUB8: smooth decreasing marginal reward.

This is intentional: the reward rule itself is not backpropagated through the discrete run index, while the current spike remains differentiable through the existing surrogate gradient.

## Shared initialization and sampler contract

For a given seed, all seven cases explicitly clone the same BenchmarkNet parameter state. Case names do not determine initialization.

Every epoch uses an explicit deterministic sample permutation derived only from seed and epoch. Every training history stores its permutation SHA256. Therefore all cases for one seed see the same sample trajectory through their common epochs.

Checkpoint selection uses only the case's native reward readout on validation:

1. maximum validation BA;
2. minimum validation CE;
3. earliest epoch.

Test data never participate in training or checkpoint selection.

# Phase 0A — scale-identifiability diagnostic

No new SNN is trained.

For all 24 existing Exp16.2 formal checkpoints, run on the test split and measure per L2 neuron:

- E[u], E[|u|], RMS(u);
- positive and negative drive parts;
- E[g u], E[|g u|], RMS(g u);
- incoming row norm ||W_2,j||_2.

Report these by synaptic tau group together with:

- RMS(u)/(1-alpha);
- RMS(g u)/(1-alpha);
- ||W_2,j||_2/(1-alpha).

Purpose: test whether the compensation observed in signed mean drive is genuinely a scale-level phenomenon rather than positive/negative cancellation.

This phase is diagnostic only and never selects a formal Exp16.3 case.

# Phase 0B — existing-checkpoint run-tail utility and objective comparison

No new SNN is trained.

Use the existing CoreBenchmark O0 (WCCE) and O1 (TSCE) checkpoints, all three seeds.

## Activity comparison

For L1 and L2, report by tau group:

- mean occupancy;
- P90 neuron occupancy;
- fraction of neurons with mean occupancy > 0.5;
- mean and P90 longest firing run;
- mean episode count.

This comparison is descriptive only. O0 and O1 change more than repeated-spike reward, so O0-vs-O1 cannot by itself prove that WCCE repeated counting causes sustained firing.

## Run-cap curve

On frozen O0/O1 spike traces, retrain matched linear probes using K = 1, 2, 4, 8, 16, 24, infinity run-capped features, plus the smooth SUB8 feature.

This asks how much class information remains when late spikes in one continuous run receive diminishing representational weight.

## Early-tail incremental utility

For K = 4, 8, 16, decompose each run into:

E_j^(K) = (1/T) sum_b min(ell_jb,K)

and

T_j^(K) = A_j^LIN - E_j^(K).

Train probes on:

- earlyK;
- tailK;
- concatenated [earlyK, tailK].

The most useful quantity is whether BA([E,T]) - BA(E) is small. If so, the late sustained tail has little incremental class information after the early run component is known.

A matched random-spike-removal WholeCount control is intentionally excluded: if the same number of spikes is removed from the same sample/neuron, WholeCount cannot distinguish their temporal positions.

# Formal evaluation

For every trained case save:

## Native readouts

1. reward: the case's own run-reward readout;
2. linear_same_head: ordinary linear mean-spike readout using the same trained head.

## Standard representation probes

On raw L1/L2 spikes, keep the existing matched probes:

- WholeCount;
- Fixed250 ordered/shuffled;
- Relative10 ordered/shuffled;
- no-bias and affine decoders.

## Activity

By layer and tau group:

- occupancy;
- longest run;
- episode count;
- high-occupancy neuron fraction.

Primary comparison is each case minus LIN, paired by seed.

The desired result is not merely lower firing. Mechanistic support requires diminishing-run-reward cases to reduce sustained occupancy/longest runs while retaining or improving transferable classification metrics.

# Interpretation hierarchy

### Scale compensation

If RMS/absolute drive and W2 row norms also scale with 1-alpha, the parameter-scale degeneracy is strongly supported.

### Tail redundancy

If early+tail provides little gain over early-only, late repeated spikes are largely redundant for class decoding.

### Objective effect

O0-vs-O1 differences show that training objective affects the learned temporal regime, but do not isolate repeated-count causality.

### Formal causal test

If changing only phi(ell) changes sustained firing while all task semantics remain whole-character sequence-level CE, then repeated within-run reward is a causal contributor to the learned code.

# Submission

From the Unity repository root:

    git pull origin main
    bash scripts/bash_script/SNN_Bash/submit_exp_16_3_cpu.bash

The pipeline is:

    prepare
      |- Phase0A scale diagnostics: 24 parallel tasks
      |- Phase0B O0/O1 diagnostics: 6 parallel tasks
      |- Phase1 formal training: 21 parallel tasks
      |- finalizer after all three branches complete

All arrays use one CPU per task.

Artifacts are written to:

    notebooks/artifacts/experiment_16_3_run_reward/run_reward_v1/

with aggregate CSVs under aggregate/.
