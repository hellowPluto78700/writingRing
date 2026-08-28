# Experiment 3.4.1 — Raw causal temporal Linear readout

## Question

Does progressive fixed-duration prefix supervision improve Linear decoding of raw 30-channel encoder event spike trains?

This is a decoder/readout experiment. It does **not** train an SNN.

## Cohort and paired splits

The implementation imports the Experiment 3.2 cohort, split, representation, metric, and seed utilities directly so the raw-spike protocol remains aligned with Experiment 1.3.3 / 3.2:

- 853 samples;
- 20 users;
- 12 classes;
- 30 unsigned event channels;
- 64 Hz;
- global padded length 256;
- Action0 + Action1;
- split seeds `(11, 23, 37, 53, 71)`;
- 12 train users / 4 validation users / 4 test users.

All five methods for a split use exactly the same train/validation/test users and sample hash.

## Five methods

```text
Relative10 Whole
Fixed250 Whole
Fixed500 Whole
Fixed250 Prefix
Fixed500 Prefix
```

Relative10 is the offline phase-normalized oracle/reference. Prefix training is only defined for the absolute-time Fixed250 and Fixed500 representations.

Fixed250 uses 16 samples/bin, 16 bins, and 480 flattened features. Fixed500 uses 32 samples/bin, 8 bins, and 240 flattened features. Relative10 uses 10 bins and 300 flattened features.

The final partial fixed-duration bin is retained. Fixed validity is `ceil(valid_length / samples_per_bin)`.

## Whole Linear

Whole training is exactly the existing Linear protocol:

```text
flatten full temporal representation
-> fit per-feature z-score on whole training samples only
-> LogisticRegression(solver="lbfgs", max_iter=5000)
```

The finalizer requires Whole results to reproduce the saved Experiment 1.3.3 Linear baseline to `1e-10` absolute test-BA tolerance.

## Prefix Linear means cumulative prefix classification

Prefix Linear does **not** classify each fixed-duration bin independently.

It classifies cumulative temporal prefixes with one fixed-dimensional Linear head:

```text
[z1, 0, 0, ...]          -> y
[z1, z2, 0, ...]         -> y
[z1, z2, z3, 0, ...]     -> y
...
[z1, ..., zK, 0, ...]     -> y
```

For a sample with `K` valid bins, the last prefix is exactly the Whole raw feature. A partial final bin is used only at its endpoint and is never discarded.

## Prefix loss through sample weights

Every prefix is one LogisticRegression training row, but each original gesture has total weight exactly 1.

For `K > 1`:

```text
intermediate prefix k < K: 0.5 / (K - 1)
final prefix k = K:        0.5
```

For `K = 1`, the only/final row has weight 1.

This implements equal per-gesture contribution with `0.5 * L_prefix + 0.5 * L_final`; long gestures do not receive more total training weight merely because they have more prefixes.

## Shared normalization contract

For each split and fixed duration, the scaler is fit once on **whole training samples**. Whole and Prefix models use that same scaler definition.

Prefix construction order is fixed:

```text
raw fixed-bin counts
-> zero all future bins
-> flatten
-> apply whole-training scaler
```

A raw future-zero bin may therefore become a nonzero z-score value. This is still causal because it contains no future event observation; only the known absolute temporal position is encoded.

## Causal contract

Changing any raw events after prefix `k` must leave the raw prefix feature unchanged. Therefore model output at prefix `k` is invariant to future event values.

The strongest identity contract is:

```text
final raw prefix == whole raw feature
```

and, because the same scaler is used:

```text
final standardized prefix == standardized whole feature
```

## Evaluation

Every run reports train/validation/test:

- balanced accuracy;
- accuracy;
- macro-F1.

Balanced accuracy is primary.

For Fixed250 and Fixed500, both Whole-trained and Prefix-trained models are also evaluated at all causal prefixes.

Two streaming curves are saved:

- `active`: only gestures with duration reaching the current tick; includes `n_samples` and coverage;
- `endpoint_aware`: unfinished gestures use the current prefix, already-finished gestures carry their endpoint prediction forward.

The long-format curve artifact permits direct common-time comparisons such as 500, 1000, 1500, and 2000 ms.

## Primary hypotheses

Primary paired effect:

```text
Delta250 = BA(Fixed250 Prefix) - BA(Fixed250 Whole)
```

Preregistered support rule:

```text
mean(Delta250) > 0
and
at least 4/5 split deltas > 0
```

`Delta500` is secondary.

Oracle gaps are:

```text
Gap250 = BA(Relative10 Whole) - BA(Fixed250 Prefix)
Gap500 = BA(Relative10 Whole) - BA(Fixed500 Prefix)
```

If a later experiment must choose Fixed250 Prefix vs Fixed500 Prefix for SNN training, that choice must use mean validation BA; test BA is confirmation only.

## Run matrix and multi-CPU execution

There are 25 independent CPU tasks:

```text
0-4:   relative10 whole, split 11/23/37/53/71
5-9:   fixed250 whole,  split 11/23/37/53/71
10-14: fixed500 whole,  split 11/23/37/53/71
15-19: fixed250 prefix, split 11/23/37/53/71
20-24: fixed500 prefix, split 11/23/37/53/71
```

Slurm follows the repository default:

```text
#SBATCH --array=0-24%25
#SBATCH --cpus-per-task=1
```

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` are all set to 1.

Each task loads the cohort, constructs its paired user split, builds the representation, fits the whole-training scaler, trains exactly one Linear model, evaluates final/streaming metrics, and writes durable JSON/model artifacts.

The afterok finalizer aggregates existing artifacts only and fails if any of the 25 runs are missing or inconsistent.

## Artifacts

```text
notebooks/artifacts/
  experiment_3_4_1_raw_causal_temporal_readout/
    prefix_linear_v1/
      runs/
      models/
      experiment_3_4_1_results.csv
      experiment_3_4_1_summary.csv
      experiment_3_4_1_prefix_curves.csv
      experiment_3_4_1_paired_deltas.csv
      experiment_3_4_1_baseline_parity.csv
      provenance.json
```

Each model `.npz` contains coefficients, intercept, classes, scaler mean, and scaler standard deviation.

## Validate

```bash
python -m pytest -q tests/test_experiment_3_4_1_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```

## Submit

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_3_4_1_pipeline.bash
```
