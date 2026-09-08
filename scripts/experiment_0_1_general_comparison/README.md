# Experiment 0.1 — General Comparison

## Question

Can a feed-forward hierarchical multi-`tau_syn` SNN turn the raw 64-Hz weighted spike train directly into class evidence that approaches explicit temporal decoding with Fixed250 or Relative10 + Linear?

This experiment uses the unique implementation namespace `experiment_0_1_general_comparison` because the repository already contains an older Experiment 0.1 growing-prefix study. No existing Experiment 0.1 files are modified.

All systems use the same Phase-C user-disjoint split (`SPLIT_SEED=12345`) and the same 12-class label set.

## Raw input contract

Every SNN receives the raw 30-channel unsigned weighted event trajectory:

```text
Raw64 [T,30]
```

There is no Fixed250/Relative10 aggregation before the SNN and no extra Exp4 input scaling.

## Direct-SNN architectures

All direct SNNs end in 12 binary spiking class neurons and use valid WholeCount for inference.

### A — `short_mid_long`

```text
30
 -> 128 shifts (2,3)
 -> 128 shifts (2,3,4,5)
 -> 128 shifts (2,3,4,5,6,7)
 -> 12 binary spiking class neurons
 -> valid WholeCount -> argmax
```

### B — `short_mid`

```text
30
 -> 128 shifts (2,3)
 -> 128 shifts (2,3,4,5)
 -> 12 binary spiking class neurons
 -> valid WholeCount -> argmax
```

### C — `mid_long`

```text
30
 -> 128 shifts (2,3,4,5)
 -> 128 shifts (2,3,4,5,6,7)
 -> 12 binary spiking class neurons
 -> valid WholeCount -> argmax
```

At 64 Hz, shifts 2..7 correspond approximately to 54, 117, 242, 492, 992, and 1992 ms synaptic time constants.

## Training objectives

Every direct architecture is trained with both:

- `whole_count_ce`: one CE per gesture from normalized valid Output WholeCount evidence;
- `timestep_ce`: one CE per valid raw timestep.

Checkpoint selection is always based on validation Output WholeCount BA, tie-broken by normalized WholeCount CE. Final inference is always Output WholeCount regardless of training objective.

## Spike/event capacity

The experiment reuses the Exp4.0.1 / Exp5.0 custom `MacroMultiSpikeLIF` semantics and compares only:

| variant | hidden cap | output cap |
| --- | ---: | ---: |
| `binary` | 1 | 1 |
| `multi_h` | 31 | 1 |

Thus binary vs multi-H isolates hidden communication capacity while keeping the class output strictly binary.

## SNN representation + analog temporal decoder

A fourth SNN family is included:

```text
Raw64 [T,30]
 -> 128 shifts (2,3,4)
 -> 128 shifts (2,3,4)
 -> temporary Linear(128,12)
```

The SNN is trained end-to-end with timestep CE. After checkpoint selection:

1. freeze the SNN;
2. discard the temporary analog head;
3. aggregate final hidden spikes into ordered 250-ms bins;
4. flatten the bins;
5. fit train-only StandardScaler;
6. choose LogisticRegression `C` using validation BA;
7. report the fixed probe on test.

This family also compares binary vs multi-H hidden capacity.

## Raw Linear baselines

A separate one-CPU baseline job computes:

```text
Raw64 -> Fixed250 ordered -> flatten -> Linear
Raw64 -> Relative10 ordered -> flatten -> Linear
```

Both reuse the same train-only scaler and validation-selected LogisticRegression protocol. These deterministic baselines are computed once rather than repeated for every SNN seed.

## Run matrix

Direct SNN:

```text
3 architectures
x 2 objectives
x 2 capacity variants
x 5 seeds (11,23,37,53,71)
= 60 runs
```

SNN + Fixed250 Linear:

```text
1 architecture
x timestep CE
x 2 capacity variants
x 5 seeds
= 10 runs
```

Total trained SNN runs: **70**.

Raw baselines are one deterministic baseline job.

## Primary comparisons

The finalizer writes paired deltas for:

- `multi_h - binary` within every trained system;
- `timestep_ce - whole_count_ce` within every direct architecture/capacity;
- `short_mid_long - short_mid` to test the added long-timescale third layer;
- `mid_long - short_mid` to test shifting the same two-layer depth toward longer temporal coverage.

The common comparison table places all SNN systems beside Raw Fixed250 + Linear and Raw Relative10 + Linear.

## Multi-CPU execution

Following `AGENTS.md`:

```text
70 independent train/evaluate tasks
 -> Slurm array 0-69%50
 -> one CPU core per task
 -> train -> select checkpoint -> evaluate -> write JSON

1 raw-baseline CPU job

array + baseline succeed
 -> afterok finalizer
 -> aggregate existing artifacts only
 -> analysis-only notebook
```

No finalizer retraining or silent regeneration is allowed.

## Run

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_0_1_general_comparison_contract.py

bash scripts/bash_script/SNN_Bash/submit_exp_0_1_general_comparison_cpu.bash
```

Artifacts are written under:

```text
notebooks/artifacts/experiment_0_1_general_comparison/general_comparison_v1/
```
