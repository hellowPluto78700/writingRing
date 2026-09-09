# Experiment 0.1.2 — Regularized General Comparison

## Question

Does hidden-state membrane/activity regularization improve the direct multi-`tau_syn` SNNs from Experiment 0.1 without changing the input, architecture, objective, output readout, data split, or random pairing?

This experiment is a paired extension of `experiment_0_1_general_comparison`. It trains only the new `reg_on` direct-SNN conditions and reuses the finalized Experiment 0.1 direct runs as frozen `reg_off` controls.

## Preserved Exp0.1 contracts

Every new run keeps the original direct-SNN contract unchanged:

- Raw64 `[T,30]` unsigned weighted event input;
- the same three hierarchical multi-`tau_syn` architectures;
- hidden width 128;
- binary (`hidden_cap=1`) vs multi-H (`hidden_cap=31`) communication;
- binary 12-neuron output layer (`output_cap=1`);
- `whole_count_ce` vs `timestep_ce` as the main task loss;
- validation Output WholeCount BA checkpoint selection, tie-broken by normalized WholeCount CE;
- valid Output WholeCount inference;
- Phase-C user-disjoint split seed 12345;
- training seeds `(11,23,37,53,71)`;
- paired model initialization and loader randomness inherited from Exp0.1.

No Fixed250/Relative10 pooling or additional scaling is inserted before the direct SNN.

## Regularized training condition

For every direct-SNN condition, the new training loss is

```text
L_total = L_task + lambda_p2 * L_P2 + lambda_a1 * L_A1
```

with fixed protocol constants:

| parameter | value |
| --- | ---: |
| `tau` | 0.99 |
| `lambda_p2` | 0.01 |
| `lambda_a1` | 0.1 |
| `P1` | disabled |
| potential layers | all hidden spiking layers |
| activity layers | all hidden spiking layers |
| output regularization | disabled |

`L_task` is exactly the Exp0.1 `whole_count_ce` or `timestep_ce` objective.

### P2: cumulative hidden membrane regularization

For each hidden layer and sample, track

```text
U_t = tau * U_{t-1} + v_pre(t)
```

only while `t < valid_length`. After the sample endpoint, `U` is frozen rather than allowed to continue decaying through padding.

The layer term is the mean squared final cumulative potential across batch and neurons; the final `L_P2` is the mean across hidden layers. This makes the regularization scale independent of whether the architecture has two or three hidden layers.

### Why Exp0.1.2 uses pre-reset membrane

The source regularizer is described using the neuron membrane returned after the neuron update. Exp0.1 uses `MacroMultiSpikeLIF`, where the post-reset membrane depends directly on the event cap:

```text
pre_reset = beta * membrane + current
spikes = capped threshold crossings(pre_reset)
post_reset = pre_reset - spikes * threshold
```

For the same `pre_reset`, binary and multi-H neurons can therefore have very different post-reset membranes solely because one can remove one threshold crossing and the other can remove many. Exp0.1.2 uses the cap-independent `pre_reset` trace for P2 so the regularizer acts on the same excitation quantity in the paired binary/multi-H comparison. This is an explicit MacroMultiSpikeLIF adaptation, not a claim that it is identical to the original post-call-membrane formulation.

### A1: cap-normalized hidden activity regularization

For each hidden layer,

```text
normalized_activity = hidden_event_count / hidden_cap
L_A1(layer) = mean(normalized_activity over valid neuron-timesteps)
```

and the final `L_A1` is the mean across hidden layers.

Thus binary events (`0/1`) and multi-H events (`0..31`) are both regularized on a per-step `[0,1]` scale. Output class neurons are not regularized; their spike evidence remains controlled only by the main classification objective.

## Run matrix

Only new regularized direct-SNN runs are trained:

```text
3 architectures
x 2 objectives
x 2 hidden-cap variants
x 5 seeds
= 60 new runs
```

The corresponding 60 `reg_off` direct runs are loaded from the finalized Exp0.1 `runs.csv`; they are never retrained by Exp0.1.2.

The Exp0.1 SNN+Fixed250+Linear family and Raw Fixed250/Relative10 + Linear baselines are reused as frozen context in the final comparison table.

## Multi-CPU execution

Following `AGENTS.md`:

```text
60 independent reg-on train/evaluate tasks
 -> Slurm array 0-59%50
 -> one CPU core per task
 -> train -> select checkpoint -> evaluate -> write JSON

all 60 tasks succeed
 -> afterok finalizer
 -> require all 60 new evaluation JSON files
 -> require frozen Exp0.1 finalized runs/baselines
 -> aggregate only; never retrain or regenerate missing runs
 -> analysis-only notebook
```

The finalizer writes:

- `runs.csv`: frozen Exp0.1 `reg_off`, new Exp0.1.2 `reg_on`, and frozen non-direct context;
- `summary.csv`;
- `baseline_results.csv` copied from finalized Exp0.1;
- `comparison_summary.csv`;
- `paired_regularization_effects.csv`;
- `regularization_objective_interactions.csv`;
- `regularization_capacity_interactions.csv`;
- `regularization_architecture_interactions.csv`;
- `manifest.json`.

## Diagnostics

Each `reg_on` evaluation records per-hidden-layer:

- events per neuron per step / second;
- cap-normalized activity per neuron-step;
- active-step fraction;
- fraction at event cap;
- dead-neuron fraction over the evaluated split;
- mean absolute pre-reset membrane.

The frozen Exp0.1 rows already contain last-hidden event rate but do not contain the newer dead/cap diagnostics, so those extra fields remain `NaN` for `reg_off` in the combined CSV rather than being silently regenerated.

Training histories additionally record task loss, P2, A1, weighted regularization terms, total loss, and the regularization-to-task-loss ratio.

## Run

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_0_1_2_regularized_general_comparison_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_0_1_2_regularized_general_comparison_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_0_1_2_regularized_general_comparison/regularized_general_comparison_v1/
```
