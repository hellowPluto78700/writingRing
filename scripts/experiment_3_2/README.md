# Experiment 3.2 — Nonlinear temporal interaction probe

## Question

Does the 30-channel event representation contain discriminative information in nonlinear interactions between neighboring temporal segments that is inaccessible to a position-aware Linear decoder?

The experiment is a decoder probe. It does not train an SNN.

## Protocol inherited from Experiment 1.3.3

Experiment 3.2 intentionally reproduces the Experiment 1.3.3 cohort and representation semantics:

- 64 Hz unsigned event representation;
- 30 event channels;
- 853 samples, 20 users, 12 classes;
- user-disjoint split seeds `(11, 23, 37, 53, 71)`;
- each split has 12 train users, 4 validation users, and 4 test users;
- user order is shuffled with `derive_seed(split_seed, "user_split")`;
- Relative10 uses `np.array_split(valid_sequence, 10)` and sums each chunk;
- Fixed250 uses 16 samples/bin and 16 padded bins;
- Fixed500 uses 32 samples/bin and 8 padded bins;
- the final partial fixed-duration bin is retained with its true event count;
- fixed-duration validity is `ceil(valid_length / samples_per_bin)`.

The finalizer compares every newly computed Linear baseline with the saved Experiment 1.3.3 per-split Linear result and fails if the absolute Balanced Accuracy difference exceeds 0.01.

## Linear baseline

For every representation and split, the position-aware Linear baseline exactly follows Experiment 1.3.3:

```text
flatten temporal representation
-> train-only per-feature z-score
-> LogisticRegression(solver="lbfgs", max_iter=5000)
```

The resulting train/validation/test decision-function logits are frozen. The neural residual is trained on top of those fixed logits:

```text
final logits = frozen Linear logits + residual logits
```

This makes the experiment ask what discriminative information remains after the position-aware Linear solution is fixed.

## Residual decoders

All residual projections and heads are bias-free. Residual output heads are zero initialized, so epoch 0 is exactly the frozen Linear classifier. Epoch 0 is a valid checkpoint candidate.

The residual rank is fixed to 16; there is no rank sweep in the primary experiment.

### Local

For each valid bin `b`:

```text
u_b = m_b * GELU(P_local z_b)
```

The flattened local features are mapped to class-logit corrections. This controls for generic within-bin nonlinear channel mixing.

### Transition

For each neighboring bin pair:

```text
v_b = m_b * m_(b+1) * ((P z_b) elementwise-multiply (Q z_(b+1)))
```

Only pairs where both bins are valid contribute. This is the primary adjacent temporal interaction probe.

### Local + transition

The model contains separate Local and Transition projections and heads, then adds both residual logit corrections to the frozen Linear logits. The branches do not share projections.

## Masking and scaling

Fixed-duration padding is never provided as an explicit classifier feature.

- Local residual: invalid bins are multiplied by `m_b`.
- Transition residual: invalid/boundary pairs are multiplied by `m_b * m_(b+1)`.
- Relative10 has ten valid bins for every sample.
- Residual inputs use a train-only per-channel RMS scale computed over valid bins only.
- Validation/test data never fit a scaler.

## Training

Primary settings:

```text
rank:                 16
batch size:           64
optimizer:            AdamW
learning rate:        1e-3
weight decay:         1e-4
max epochs:           200
minimum epochs:       20
early-stop patience:  25
grad clip norm:       1.0
model init seed:      2026
```

The effective initialization seed depends on representation and decoder type, but not on user split seed. Therefore the five user splits for a fixed decoder architecture share the same initialization rule.

Validation Balanced Accuracy selects the checkpoint. Ties use lower validation CE, then the earlier epoch because the incumbent checkpoint is retained when both are equal. Test is evaluated after checkpoint selection.

## Run matrix

There are 45 independent neural tasks:

```text
3 representations
x 3 residual decoders
x 5 user-disjoint splits
= 45 runs
```

Representations:

- `relative10`
- `fixed250`
- `fixed500`

Residual decoders:

- `local`
- `transition`
- `local_transition`

Split seeds:

- `11`
- `23`
- `37`
- `53`
- `71`

## Multi-CPU execution

The repository default multi-CPU policy is used:

```text
one independent run
-> one Slurm array task
-> one CPU core
-> fit deterministic Linear baseline
-> train one residual decoder
-> select best validation checkpoint
-> evaluate test once
-> save JSON + checkpoint

all 45 tasks succeed
-> afterok finalizer
-> aggregate existing artifacts only
```

The array is capped below the repository concurrency limit:

```text
#SBATCH --array=0-44%45
#SBATCH --cpus-per-task=1
```

CPU thread environment variables are set to 1 to avoid oversubscription.

## Primary interpretation

The primary representation is Fixed250.

Primary effect:

```text
Transition - Linear
```

Mechanism control:

```text
Transition - Local
```

Interpretation:

- `Transition > Linear` and `Transition > Local`: evidence that adjacent temporal interaction contributes beyond both a position-aware Linear decoder and generic within-bin nonlinearity.
- `Local > Linear` and `Transition ~= Local`: evidence for generic nonlinearity, but weak evidence for temporal composition specifically.
- `Local+Transition > both`: evidence that within-bin and cross-bin nonlinear information are complementary.
- residual ~= Linear: the representation is largely linearly accessible under this probe family.

Cross-representation gain differences are secondary evidence because the number of temporal positions and residual parameters differ across Relative10, Fixed250, and Fixed500. Trainable parameter counts are therefore reported explicitly.

## Outputs

Artifacts are written under:

```text
notebooks/artifacts/
  experiment_3_2_nonlinear_temporal_interaction/
    low_rank_residual_v1/
      runs/
      checkpoints/
      experiment_3_2_results.csv
      experiment_3_2_paired_deltas.csv
      experiment_3_2_summary.csv
      experiment_3_2_mechanism_summary.csv
      experiment_3_2_linear_baseline_summary.csv
      experiment_3_2_baseline_reproduction.csv
      experiment_3_2_parameter_counts.csv
      experiment_3_2_conclusion.json
      provenance.json
```

The finalizer rejects missing runs, mixed run identities, cohort/split mismatches, repeated Linear-baseline mismatches, and baseline drift relative to Experiment 1.3.3.

## Validate

From repository root:

```bash
python -m pytest -q \
  tests/test_repository_source_syntax.py \
  tests/test_experiment_3_2_contract.py
```

## Submit

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_3_2_pipeline.bash
```
