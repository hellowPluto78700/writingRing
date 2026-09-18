# Exp10.2.1 — L2 Membrane Memory × Hidden Communication Coding

## Question

Exp10.2 showed that increasing L1 membrane memory from ~22.5 ms to ~54.3 ms improved the current time-shared A2. Exp10.2.1 asks whether L2 also benefits from longer membrane memory and whether integer weighted hidden communication preserves more analog evidence than binary communication.

The experiment is:

\[
2\ coding\ conditions\times3\ L2\ membrane\ settings\times3\ seeds=18\ E2E\ runs.
\]

## Fixed contract

- Dataset: D1 post-encode airborne mask
- User split: rotation0 (test fold0, validation fold1, train folds2/3/4)
- Backbone: 30 -> 128 -> 128 -> 12
- L1/L2 synaptic shifts: (2,3,4)
- L1 membrane: fixed shift_mem=2, beta=0.75, tau_mem~54.31 ms
- Objective: L2 valid-mean time-shared WCCE
- No Joint, TSCE, MT threshold bank, or phase-aware readout
- Seeds: 11,23,37
- Same optimizer / early stopping / checkpoint selection as Exp10.2

## Factor A — L2 membrane memory

| L2 shift_mem | beta | tau_mem |
|---:|---:|---:|
| 1 | ~0.499968 | 22.54 ms |
| 2 | 0.75 | 54.31 ms |
| 3 | 0.875 | 117.01 ms |

Thus (L1,L2) is one of (54,22), (54,54), or (54,117) ms.

## Factor B — hidden communication

### Binary

Both hidden layers use max_spikes_per_dt=1, so communication is 0/1.

### Weighted31

Both hidden layers use the existing MacroMultiSpikeLIF with:

\[
s_t=clip\left(\left\lfloor\frac{\max(U_t^-,0)}{\theta}\right\rfloor,0,31\right),
\]

and subtractive reset:

\[
U_t=U_t^- - s_t\theta.
\]

This is integer event-count communication, not MT threshold heterogeneity and not a continuous analog activation. No /31 normalization is applied before L1->L2 or L2->Linear.

## Full matrix

| ID | Coding | L1 mem | L2 mem | Objective |
|---|---|---:|---:|---|
| B22 | Binary | 54 ms | 22 ms | WCCE |
| B54 | Binary | 54 ms | 54 ms | WCCE |
| B117 | Binary | 54 ms | 117 ms | WCCE |
| W22 | Weighted31 | 54 ms | 22 ms | WCCE |
| W54 | Weighted31 | 54 ms | 54 ms | WCCE |
| W117 | Weighted31 | 54 ms | 117 ms | WCCE |

Each condition uses three paired seeds.

## Primary contrasts

Within Binary: B54-B22 and B117-B22.

Within Weighted31: W54-W22 and W117-W22.

At each L2 memory: Weighted31-Binary.

Interaction:

\[
I_2=(W54-W22)-(B54-B22),
\]

\[
I_3=(W117-W22)-(B117-B22).
\]

## Representation diagnostics

Main communication gaps:

\[
\Delta_{L1comm}=BA(L1_{communication})-BA(L1_{pre-reset}),
\]

\[
\Delta_{L2comm}=BA(L2_{communication})-BA(L2_{pre-reset}).
\]

A useful coding improvement should raise communication BA while preserving pre-reset BA; a smaller gap caused only by worse pre-reset representation is not interpreted as recovery.

Temporal-collapse diagnostic:

\[
\Delta_{temporal}=BA(L2_{Fixed250})-BA(L2_{whole}).
\]

The historical probe name still uses __spike__, but in Weighted31 that tensor is the integer communication count.

## Communication-cost diagnostics

For each hidden layer and synaptic-shift group, report:

- mean count per neuron per valid timestep;
- mean events/neuron/s;
- nonzero communication fraction;
- P(count>=2);
- P(count>=4);
- cap-hit fraction;
- dead-neuron fraction.

For Weighted31, near-zero P(count>=2) means the weighted channel is barely used. A large cap-hit fraction indicates an unhealthy burst regime.

Because Weighted31 is not divided by 31, the finalizer also records output-weight Frobenius/mean-absolute norms, per-valid-timestep evidence magnitude, and native score norm. These diagnose whether an apparent accuracy change is accompanied by a pure logit-scale change.

## Training and pairing

All 18 cases are independently end-to-end trained. There is no frozen-replay main branch because changing Binary to Weighted31 alters communication magnitude and subtractive reset.

For a fixed seed, learned-layer initialization and DataLoader order are identical across all six conditions. Only the hidden event cap and L2 membrane beta differ.

## Multi-CPU execution

~~~bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_2_1_cpu.bash
~~~

Dependency graph:

~~~text
prepare
   -> 18-way one-core CPU array
       -> finalizer
~~~

Array contract: 0-17%18 with cpus-per-task=1. BLAS/OpenMP thread counts are pinned to one.

## Outputs

Artifact root:

~~~text
notebooks/artifacts/experiment_10_2_1_l2_membrane_weighted/
  d1_l1mem2_l2mem123_binary_weighted31_v1/
~~~

Primary aggregate files:

- run_metrics.csv
- method_summary.csv
- paired_contrasts.csv
- paired_contrast_summary.csv
- interaction_runs.csv
- interaction_summary.csv
- activity_runs.csv
- activity_summary.csv
- probe_runs.csv
- probe_summary.csv
- manifest.json

The notebook is aggregation-only.
