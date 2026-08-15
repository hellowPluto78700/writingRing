# Plan: A/B/C/D CNN Probe Complexity Ladder

## Plan identity

- **Repository:** `hellowPluto78700/writingRing`
- **Workflow authority:** `AGENTS.md`
- **Risk class:** **HIGH_RISK**
- **Reason:** changes the shared CNN architecture protocol and persisted Experiment A checkpoints consumed by B/C/D.

## Goal

Replace the current single-CNN evaluation with three matched CNN probe levels so encoder/reconstruction quality is evaluated across increasing feature-extraction complexity rather than through one relatively strong classifier.

Use:

| Probe | Architecture |
| --- | --- |
| `CNN-S` | `Conv1d(3,8,k=7) -> ReLU -> masked GAP -> Linear` |
| `CNN-M` | `Conv1d(3,32,k=7) -> Conv1d(32,64,k=5,s=2) -> ReLU -> masked GAP -> Linear` |
| `CNN-L` | existing `64 -> 128 -> 256 -> 256` MaskAwareAccelerationCNN |

The primary comparison is the matched raw/reconstruction gap for each probe complexity.

## Scientific contract

For each probe \(P_k\), preserve the existing A/B/C/D protocol:

- **A:** raw train / raw val / raw test.
- **B:** freeze the matching `A_k` model and evaluate reconstruction without retraining or renormalization.
- **C:** same architecture as `A_k`, trained from scratch on reconstruction.
- **D:** same architecture, trained once on mixed raw + reconstruction and evaluated separately on raw and reconstruction.

Primary comparisons:

- `A_k - C_k`: reconstruction accessibility/recoverability gap.
- `A_k - B_k`: raw-to-reconstruction compatibility gap.
- `|D_raw_k - D_recon_k|`: mixed-domain invariance gap.

Raw and reconstruction must always use the **same architecture, split, class mapping, sample cohort, training policy, and evaluation metrics** within a comparison.

---

# Agent workflow

```text
PRIMARY
  -> freeze TaskSpec
  -> luna_worker
       inspect -> implement -> focused tests -> fix -> retest -> self-review
  -> luna_verifier
  -> PRIMARY integration/documentation
```

Do not use `luna_probe` unless implementation discovers a concrete unresolved checkpoint/model-contract fact.

---

# T001 — Shared probe architecture contract

## Required behavior

1. Introduce an explicit probe/model variant with exactly:
   - `cnn_s`
   - `cnn_m`
   - `cnn_l`

2. All three models:
   - accept `(B, 3, T)`;
   - preserve mask-aware right-padding behavior;
   - expose logits plus a pre-classifier embedding;
   - support the shared training/evaluation pipeline.

3. `cnn_l` remains behaviorally/checkpoint compatible with the current `MaskAwareAccelerationCNN`.

4. Architecture identity and architecture configuration are persisted in checkpoints and provenance.

## Preserved contracts

- acceleration input remains `paddedSpikeIMU[..., 15:18]`;
- user-disjoint splitting is unchanged;
- normalization remains fit only from protocol-defined valid train samples;
- metric definitions and label semantics are unchanged;
- no preprocessing/reconstruction producer changes.

## Acceptance criteria

- all three architectures pass shape/mask/model tests;
- `cnn_l` can restore existing compatible baseline weights;
- checkpoints unambiguously identify their probe architecture;
- mismatched checkpoint/model architecture fails explicitly.

---

# T002 — Extend A/B/C/D across the probe ladder

## Required behavior

1. A/B/C/D runners accept a probe architecture selection.

2. For each `k in {cnn_s, cnn_m, cnn_l}`:
   - A trains and saves `A_k`;
   - B must consume the matching `A_k` checkpoint;
   - C reuses A's cohort/split/class mapping but trains `C_k` from scratch;
   - D trains one `D_k` mixed-domain model and evaluates both test domains.

3. B/C/D must reject architecture or cohort mismatches rather than silently adapting them.

4. Existing ABCD scientific semantics remain unchanged apart from making probe complexity an experiment variable.

5. Artifact/provenance paths must prevent results from different probe variants from overwriting each other.

## Acceptance criteria

- setup/integration tests cover A/B/C/D for all three probes;
- B rejects a checkpoint produced by a different probe;
- paired raw/reconstruction views retain identical logical sample IDs and labels;
- changing probe architecture does not change the authoritative A split/cohort.

---

# T003 — Comparison outputs and notebook integration

## Required behavior

Expose the three probes in the A/B/C/D notebooks as configuration rather than duplicating model/training logic.

Produce a compact comparison table suitable for the final analysis:

| Probe | A Raw | B Frozen→Recon | C Recon | D Raw | D Recon | A-C | A-B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-S | | | | | | | |
| CNN-M | | | | | | | |
| CNN-L | | | | | | | |

The comparison should support an accuracy/balanced-accuracy versus probe-complexity plot, with **balanced accuracy as the primary metric** and macro-F1 as a secondary metric.

Documentation must interpret probe complexity as **feature-extraction complexity**, not as a pure parameter-count experiment.

## Acceptance criteria

- notebooks remain thin configuration/presentation layers;
- the comparison can be generated from saved standardized experiment artifacts;
- no metric is recomputed with notebook-only implementations;
- README / relevant `docs/notes/**` describe the new three-probe interpretation.

---

## Validation

Run from cheap to expensive:

1. model shape/mask/checkpoint tests;
2. focused A/B/C/D runner tests for each probe;
3. checkpoint/cohort mismatch tests;
4. notebook structural/integration checks;
5. affected acceleration-reconstruction test suite;
6. `git diff --check`;
7. independent `luna_verifier` review.

Run full pytest only if the affected shared-contract surface or focused failures justify it.

## Replan triggers

Return `NEEDS_REPLAN` only if correct implementation requires:

- changing padded/reconstruction file schemas;
- changing acceleration channel semantics;
- changing A-authoritative cohort/split semantics;
- breaking legacy `cnn_l` checkpoint compatibility;
- redesigning the shared evaluation metric protocol;
- expanding the task to direct spike-domain probes or linear probes.

## Non-goals

This plan does not add:

- linear/logistic-regression input probes;
- direct 15-channel spike probes;
- stronger-than-current CNN architectures;
- preprocessing, segmentation, padding, or reconstruction changes;
- changes under `data_sample/**` or `vendor/**`.
