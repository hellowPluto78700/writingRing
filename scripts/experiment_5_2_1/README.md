# Experiment 5.2.1 — FF Multi-Tau Memory × Objective

## Question

Experiment 5.2 showed that a one-layer FF temporal decoder benefits from longer passive membrane memory but plateaus around the 1–2 s range, while its hidden temporal trajectory still contains more linearly accessible information than its final endpoint state. Experiment 5.2.1 asks two controlled questions:

1. Does a heterogeneous membrane-time-constant bank improve the FF endpoint representation beyond the validation-selected homogeneous long-`tau_mem` control?
2. Does explicit temporal supervision during training help preserve hidden temporal information and close the gap between the hidden trajectory, `Uend`, and the native endpoint classifier?

The experiment is a paired 2 × 2 factorial with the exact same frozen local representation and five seeds.

## Frozen input contract

The temporal decoder reuses the committed Experiment 5.2 frozen L2 caches:

```text
Experiment 5.2 frozen Exp3 local model
Raw64 -> L1_128 shifts(2,3,4) -> L2_128 shifts(2,3,4)
                                      |
                                      +-> frozen binary L2 trajectory at 64 Hz
```

Experiment 5.2.1 does **not** retrain the local SNN. Each run validates the Experiment 5.2 cache identity before training and fails if the source artifact is missing or mismatched.

Source protocol:

```text
experiment_5_2_frozen_local_tauR_sweep
frozen_exp3_l2_endpoint_tauR_v1
```

## Factorial design

All temporal decoders are one-layer FF models with width 128, no recurrent connection, no temporal pooling/compressor/downsampling, the same threshold/reset settings as Experiment 5.2, and the same deployed endpoint readout.

| Memory mode | Hidden membrane time constants | Objective |
|---|---|---|
| `single_shift7` | all 128 neurons use shift 7 (~1992 ms at 64 Hz) | `endpoint_ce` |
| `multitau_4567` | 32 neurons each at shifts 4/5/6/7 (~242/492/992/1992 ms) | `endpoint_ce` |
| `single_shift7` | all 128 neurons use shift 7 | `endpoint_fixed250_aux_ce` |
| `multitau_4567` | 32 neurons each at shifts 4/5/6/7 | `endpoint_fixed250_aux_ce` |

Seeds:

```text
11, 23, 37, 53, 71
```

Total: **20 independent runs**.

```text
4 conditions × 5 seeds = 20 runs
```

The single-tau control uses shift 7 because Experiment 5.2 selected that FF condition by mean native validation endpoint BA. The multi-tau bank uses only time constants already covered by the Experiment 5.2 sweep.

## Decoder and deployed readout

For frozen local spike vector `z_t`:

```text
I_t = W_in z_t
(S_t, U_t) = LIF(I_t, U_{t-1}; beta_i)
```

There is no recurrent term. The deployed prediction is always:

```text
Uend = U[length - 1]
logits = W_end Uend + b_end
prediction = argmax(logits)
```

The primary deployment loss remains:

```text
L_end = CE(W_end Uend + b_end, y)
```

Therefore Experiment 5.2.1 does not introduce an objective/readout mismatch: checkpoint selection and the final native classifier are both based on the endpoint representation.

## New temporal auxiliary objective

`endpoint_fixed250_aux_ce` adds a training-only auxiliary head over ordered 250 ms hidden-spike counts:

```text
H250 = flatten(Fixed250Count(S_1:T))
L_aux = CE(W_aux H250 + b_aux, y)
L_total = L_end + lambda_aux L_aux
lambda_aux = 1.0
```

The auxiliary head is discarded at deployment. Inference remains:

```text
frozen Local -> FF -> hidden Uend -> Linear -> class logits
```

The purpose of the auxiliary loss is not to deploy Fixed250. It tests whether temporal supervision can prevent useful `WHAT + WHEN` information from being destroyed while the FF state is trained to form an endpoint summary.

## What multi-tau changes

`single_shift7` gives every hidden neuron the same passive decay kernel. `multitau_4567` keeps the hidden width and deployed parameter count fixed but gives four equal neuron groups different membrane decay constants. The endpoint therefore contains a bank of passive temporal traces spanning short-to-long horizons instead of one homogeneous horizon.

This isolates temporal-basis diversity from network depth and recurrence.

## Checkpoint selection

Each independent run trains its temporal FF and endpoint head jointly. For auxiliary conditions, the Fixed250 auxiliary head is trained jointly as well.

Checkpoint selection is strictly validation-only:

1. maximize native endpoint validation balanced accuracy;
2. break exact ties with lower validation endpoint CE.

Test BA is never used for checkpoint selection.

## Mandatory evaluation probes

Every selected checkpoint is evaluated with the deployed native endpoint head and four frozen post-hoc linear probes:

```text
hidden_whole_count
hidden_fixed250_ordered
hidden_relative10_ordered
hidden_uend
```

The probes use the same Experiment 5.2 probe protocol so the following gaps can be interpreted directly:

- `hidden_relative10 - hidden_uend`: trajectory-to-endpoint consolidation gap;
- `hidden_uend_probe - native_endpoint_BA`: endpoint-head utilization gap;
- local-reference probes versus hidden probes: information lost from frozen Local L2 into the temporal FF.

## Multi-CPU execution

Experiment 5.2.1 follows the repository default one-run-per-CPU Slurm strategy.

```text
20 independent condition/seed runs
        -> Slurm array 0-19%20
        -> one CPU core per task
        -> load/validate reused Exp5.2 frozen L2 cache
        -> train one FF run
        -> select best checkpoint on validation endpoint BA
        -> evaluate the same checkpoint immediately
        -> write per-run checkpoint/history/evaluation
all 20 tasks succeed
        -> afterok finalizer
        -> concatenate finalized CSV/JSON artifacts only
        -> analysis-only notebook
```

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_5_2_1_cpu.bash
```

The finalizer does not train models or regenerate missing runs. Missing per-run artifacts are a hard failure.

## Finalized artifacts

```text
notebooks/artifacts/experiment_5_2_1_ff_multitau_objectives/
  frozen_exp3_l2_ff_multitau_objectives_v1/
    runs.csv
    histories.csv
    local_reference.csv
    manifest.json
    checkpoints/
    evaluations/
    histories/
```

## Notebook contract

`notebooks/experiment_5_2_1_ff_multitau_objectives.ipynb` is analysis-only. It reads finalized artifacts and computes:

1. per-condition mean ± SD for native validation/test BA;
2. post-hoc `Uend`, Fixed250, Relative10, and WholeCount probes;
3. paired multi-tau minus single-tau effects within each objective;
4. paired auxiliary-objective minus endpoint-only effects within each memory mode;
5. the 2 × 2 interaction effect;
6. native-head versus post-hoc `Uend` utilization gap;
7. hidden Relative10 versus `Uend` consolidation gap;
8. validation learning curves;
9. local-reference comparison.

The notebook never launches training, Slurm, or regenerates missing experiment artifacts.

## Interpretation rules

A multi-tau gain is only an architecture effect when it appears in the paired comparison at the same objective. An auxiliary-objective gain is only a supervision effect when it appears in the paired comparison at the same memory mode.

Key diagnostic cases:

- higher hidden Relative10 and higher `Uend` probe: richer temporal representation;
- unchanged hidden Relative10 but higher `Uend` probe: improved endpoint consolidation;
- higher `Uend` probe but unchanged native endpoint BA: endpoint-head optimization remains limiting;
- native endpoint BA approaches the `Uend` probe: the representation–classifier utilization gap is closing.
