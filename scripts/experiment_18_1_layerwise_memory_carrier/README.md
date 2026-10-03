# Exp18.1 — Layer-wise Memory-Carrier Localization

## Scientific question

Exp18.1 localizes the Exp18 degradation to a hidden layer. The experiment asks whether placing the long-timescale pole in a resettable membrane state is harmful mainly in L1, mainly in L2, or only when both layers use that carrier.

The full conceptual 2 x 2 design is:

| Case | L1 carrier | L2 carrier | Source |
| --- | --- | --- | --- |
| `II_REF` | I | I | CoreBenchmark O0, reused |
| `UI` | U | I | Exp18.1, trained |
| `IU` | I | U | Exp18.1, trained |
| `UU_REF` | U | U | Exp18 U_NORMAL, reused |

Only `UI` and `IU` are newly trained. With seeds 11/23/37 this is 6 formal runs.

## Layer dynamics

For an I-carrier layer:

```text
I[t]    = alpha_slow I[t-1] + W x[t]
Upre[t] = beta_fast U[t-1] + I[t]
U[t]    = Upre[t] - theta s[t]
```

For a U-carrier layer:

```text
I[t]    = beta_fast I[t-1] + W x[t]
Upre[t] = alpha_slow U[t-1] + I[t]
U[t]    = Upre[t] - theta s[t]
```

Both use normal subtractive-reset differentiation. Exp18.1 does not include detached reset, tau/threshold/width sweeps, Prefix, TSCE, firing regularization, or extra objectives.

The slow shifts remain (2,3,4) in both layers; the fast pole is the CoreBenchmark 22.54 ms membrane pole. Before a threshold/reset event, I and U carriers contain the same two cascaded linear poles in reversed order.

## Locked inherited contract

Exp18.1 inherits the CoreBenchmark O0 contract unchanged:

- locked CoreBenchmark dataset cache and concrete user split;
- labels, input channels, width 128, 43/43/42 tau groups;
- threshold 0.5, surrogate slope 25, one binary spike per timestep;
- paired bias-free hidden weights and bias-free accumulator;
- WCCE with valid-mean logits;
- Adam lr 0.001, weight decay 0, batch 128;
- max 100 epochs, min 20, patience 30;
- checkpoint selection by validation native BA, then validation mean-logit CE, then earliest epoch;
- seeds 11/23/37;
- canonical CoreBenchmark probe implementation and shuffle seeds.

For a fixed seed, `layers.0.weight`, `layers.1.weight`, and `head.weight` use the same named RNG streams as CoreBenchmark O0 and Exp18, so the initial parameter tensors are paired across all four corners.

## Primary interpretation

The primary localization compares the four corners using native BA and L2 spike/no-bias probes.

Key contrasts for metric Y are:

```text
L1 U effect when L2=I : Y_UI - Y_II
L1 U effect when L2=U : Y_UU - Y_IU
L2 U effect when L1=I : Y_IU - Y_II
L2 U effect when L1=U : Y_UU - Y_UI
interaction            : Y_UU - Y_UI - Y_IU + Y_II
```

Primary representation metrics are WholeCount, Fixed250 ordered/shuffled, Relative10 ordered/shuffled, plus:

```text
G_order_250 = Fixed250 ordered - Fixed250 shuffled
G_order_rel = Relative10 ordered - Relative10 shuffled
```

Interpretation is descriptive and paired across the three locked seeds.

- `UI ~= II` and `IU ~= UU`: deep U is the main harmful intervention.
- `IU ~= II` and `UI ~= UU`: shallow U is the main harmful intervention.
- both hybrids near II but UU poor: harmful non-additive interaction requires U in both layers.
- both hybrids intermediate: both layers contribute independent carrier effects.

## Required per-run outputs

Each formal `UI` or `IU` task must perform, in the same Slurm task:

```text
train
 -> select checkpoint
 -> native train/val/test evaluation
 -> canonical probes
 -> per-layer/per-tau activity diagnostics
 -> complete.json
```

Required artifacts:

- `initial.pt`
- `checkpoint.pt`
- `history.json`
- `native.json`
- `activity.json`
- `traces.npz`
- `probes.json`
- `probe_search.json`
- `probe_decoders.npz`
- `probe_predictions.npz`
- `complete.json`

Activity diagnostics include firing Hz, mean absolute pre-reset state, pre-reset/threshold ratio conditioned on firing, and silent-neuron fraction for every layer/tau group.

Gradient diagnostics, rasters, and burst/run-length analysis are optional in Exp18.1 and do not gate completion.

## Reference artifacts

`II_REF` reuses finalized CoreBenchmark O0 artifacts. `UU_REF` reuses finalized Exp18 `U_NORMAL` artifacts.

Neither reference receives a training task or a separate diagnostic task in Exp18.1. `prepare` records checkpoint/native/probe hashes for both references. `finalize` fails if any reference hash changes.

## Provenance and resume behavior

`prepare` locks:

- Exp18.1 source hash;
- CoreBenchmark identity, dataset hash, and protocol hash;
- Exp18 protocol-lock hash;
- exact reference artifact hashes;
- seed/case/carrier mapping.

Resume behavior:

- valid `complete.json` plus matching provenance -> skip the run;
- valid checkpoint plus matching provenance -> skip training and rerun required evaluation/probes/activity;
- mismatched source/core identity/case/seed/carrier assignment -> fail rather than silently reuse.

Any implementation change after `prepare` requires a new results directory or a deliberate new protocol lock.

## Slurm execution contract

Recommended concurrency is 6.

Exact DAG:

```text
compute-node smoke
  -> prepare
      -> 6-task UI/IU train+evaluate+probe+activity array
          -> finalize
```

No post-training CPU array is created because all required per-run analysis is lightweight and run-local.

All jobs use:

`scripts/bash_script/SNN_Bash/slurm_cpu_env.bash`

which loads the intended Conda environment and one-core CPU thread limits without sourcing Unity's full `/etc/profile`.

Formal submission:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_18_1_cpu.bash
```

The smoke job uses the real locked Core/Exp18 artifacts and checks both hybrid cases for forward/loss/backward/optimizer behavior, checkpoint roundtrip, finite restored logits, and the canonical probe feature path.

## Finalizer requirements

Finalization requires all six hybrid `complete.json` files and unchanged `II_REF`/`UU_REF` reference hashes. It does not retrain or regenerate missing runs.

It writes `aggregate.json` containing:

- all four corners x three seeds;
- native train/val/test metrics;
- primary L2 spike/no-bias probes;
- Fixed250 and Relative10 order gaps;
- seed-wise factorial contrasts;
- three-seed mean factorial contrasts.
