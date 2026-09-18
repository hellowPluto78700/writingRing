# Exp11.0 v4 plan — frozen-L1 temporal representation control

## Scientific question

Determine whether RSNN/Fusion can internalize the temporal information already present in the pretrained L1 representation, or whether the final gain requires WCCE to substantially reshape L1.

## Factors

- Dataset: D0 / D1.
- L1 mode:
  - dynamics_only;
  - pretrained_trainable;
  - pretrained_frozen.
- Context topology:
  - ff;
  - diagonal recurrence;
  - dense recurrence.
- Fusion:
  - off;
  - on.
- Seed:
  - 11 / 23 / 37.

Total main runs: 108.

## Frozen contract

For `pretrained_frozen`:

- load the same seed-matched source L1 input matrix used by `pretrained_trainable`;
- set only `model.l1_input.weight.requires_grad=False`;
- exclude that parameter from the optimizer;
- leave context, recurrence, Fusion, and Linear readout trainable;
- assert post-training source-relative L1 drift <= 1e-12.

Neuron dynamics are fixed in all conditions, so freezing the input matrix freezes the only trainable L1 parameter.

## Representation-drift diagnostics

For both pretrained modes record:

- relative Frobenius drift from the source matrix;
- cosine similarity to the source matrix;
- L1 pre-reset Fixed250 and whole probes;
- L1 spike Fixed250 and whole probes.

The frozen branch should reproduce the source weight exactly. The trainable branch quantifies how strongly WCCE reshapes L1.

## Core comparisons

- B-A under pretrained_frozen: recurrence can use the original L1 representation.
- D-B under pretrained_frozen: history-conditioned Fusion adds value without L1 adaptation.
- pretrained_trainable - pretrained_frozen: value of L1 adaptation.
- D1-D0: preprocessing effect at fixed architecture and L1 mode.
- recurrence × Fusion: `(D-C)-(B-A)`.

## Execution

- 3 D0 matched-source runs.
- 108 main runs.
- D0/D1 remain adjacent array tasks.
- one CPU core per task.
- main array cap 50 concurrent: `0-107%50`.
- finalizer aggregates only.
- notebook performs analysis only.

## Acceptance criteria

- Exactly 108 main specs.
- L1 modes are exactly dynamics_only / pretrained_trainable / pretrained_frozen.
- Frozen L1 is absent from optimizer parameter list.
- Synthetic training step cannot change frozen L1.
- Frozen post-training drift is zero within 1e-12.
- Trainable/frozen paired contrasts are generated.
- L1 pre-reset Fixed250 appears in core contrasts.
- A/B/C/D and D0/D1 pairing contracts remain unchanged.
- focused Exp11.0 CI passes.
