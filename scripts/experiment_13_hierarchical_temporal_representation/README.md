# Experiment 13 — Hierarchical Temporal Representation Validation

## Scientific question

Exp13 asks whether deeper multi-tau SNN layers merely produce slower temporal filtering, or causally transform local temporal evidence into a current, communicated, stroke-organized contextual representation.

Three hypotheses are separated:

```text
H0: temporal filtering only
H1: longer history exists internally, but spike communication loses it
H2: longer history becomes a communicated higher-level contextual representation
```

The fixed trained backbone is Exp7.3 A2:

```text
input -> L1 (128, shifts 2/3/4) -> L2 (128, shifts 2/3/4)
      -> bias-free Linear -> WCCE
```

No Exp13 sub-experiment retrains the A2 backbone. Exp7.3.9 C2 is the negative-depth control because the extra L3 is already known not to improve the native A2 result. A same-tau random A2 is used only for dynamics/mechanism controls, not as a classification-BA baseline.

## Representation hierarchy

Paper-style analyses use the communicated representation as primary:

```text
spike50_t = sum of the last 4 binary spike timesteps
```

At 64 Hz this is approximately a 50–60 ms communication trace. Mechanistic secondary states are `syn_current = I_t` and `pre_reset = U_t^-`. This separates hidden-state hierarchy from communicated-representation hierarchy.

## Sub-experiments

Every sub-experiment writes into its own directory under the Exp13 protocol root. Per-run artifacts never share mutable output files.

### A1 — Recurrence / self-similarity

Question: do deeper layers have slower population dynamics?

For A2, same-tau random A2, and Exp7.3.9 C2, A1 measures layer/state similarity at increasing temporal lag. It also records state norm, per-neuron variance, nonzero fraction, and spike firing rate so trained-vs-random recurrence is not interpreted without activity-statistic controls.

Task matrix: `3 seeds x 3 model kinds = 9 CPU tasks`.

### A2 — Same-suffix causal history truncation

Question: does the current deeper-layer representation causally depend on older history?

```text
full:      x[1:t]       -> z_t(full)
truncated: x[t-D:t]     -> z_t(D), zero initial state
```

The current suffix is identical; only older history is removed. History sweep: `50, 100, 250, 500, 750, 1000 ms`. Phase anchors are 25%, 50%, 75%, and 100%; weak stroke annotations additionally provide mid-stroke and stroke-end anchors on test.

A2 reports correlation, cosine similarity, and normalized L2 distance. Task matrix: `3 seeds x {A2,C2} x 6 histories = 36 CPU tasks`.

### A3 — Layer-specific reset

Question: where is causal history stored?

At 25%, 50%, and 75% of the gesture, apply `intact`, `reset L1`, `reset L2`, or `reset both`. Reset means `I=0` and `U=0`; future input is unchanged. Recovery is measured for roughly 0–500 ms.

Task matrix: `3 seeds x 4 interventions = 12 CPU tasks`.

### B — Context expression in the current representation

Question: is older history merely changing state values, or is it currently decodable?

B reads durable A1 full-history anchors and A2 truncated-history anchors. The primary causal probe is trained once on full-history train/val states and then applied unchanged to both full and truncated test states.

Primary no-bias probe:

```text
StandardScaler(with_mean=False)
LogisticRegression(fit_intercept=False)
```

A secondary affine probe preserves the historical mean-centered/intercepted convention. A secondary probe is also retrained directly on each truncated representation.

Interpretation:

```text
shared probe down + retrained probe down
    -> information itself is lost

shared probe down + retrained probe ~= full
    -> information remains but representation geometry changed
```

B reports `context_drop_shared_pp`, `information_loss_retrained_pp`, and `geometry_shift_pp`. Task matrix: `36 CPU tasks`.

WholeCount and Fixed250 remain auxiliary sequence-accessibility references from Exp7.3/Exp7.3.9. WholeCount means temporally collapsed accessibility; Fixed250 means explicitly temporally indexed accessibility. Fixed250 is not interpreted as an instantaneous local-state probe.

### C — Stroke-level representational organization

Question: do deeper representations organize around true stroke units?

C uses the weak stroke start/end annotations already validated in Exp12.1. It aligns transitions around true press/lift boundaries, compares within-stroke versus across-stroke similarity at matched lag, and adds random plus motion-matched pseudo-boundary controls.

The motion match uses event activity, the aligned six continuous sensor channels, and local derivative magnitude. The primary boundary result is the residual real-boundary effect relative to the motion-matched control.

Task matrix: `3 CPU tasks`.

### D — Paper-style stroke-order scrambling

Question: does the deeper representation depend more on coarse preceding stroke context?

Each true stroke waveform is kept intact. Synthetic original and permuted sequences use the same stroke segments, then outputs are realigned by stroke identity and within-stroke phase.

Primary hierarchy scales are whole-stroke and multi-stroke-block permutation. Gap sweep is `0, 50, 100, 250 ms`: 0 ms exposes direct transition artifacts, 50/100 ms are primary practical conditions, and 250 ms is a history-washout control.

Metric: `ContextSensitivity = 1 - corr(original, permuted+realigned)`.

Task matrix: `3 seeds x 4 gaps x 2 scales = 24 CPU tasks`.

### E — Cross-tau temporal decoding

Question: has the learned inter-layer projection converted distributed multi-tau traces into localized delayed temporal features?

E uses both a unit impulse and a real 6-step event motif followed by a zero tail. It records I, pre-reset U, and spikes. Per-neuron outputs include peak latency, peak amplitude, FWHM, response duration, and peak-to-tail ratio; population responses are saved for neuron-by-time heatmaps.

A2, same-tau random A2, and C2 are compared. E is mechanistic evidence only; delayed peaks alone are not proof of hierarchy.

Task matrix: `9 CPU tasks`.

## Multi-CPU execution

Exp13 serializes sub-experiment arrays while parallelizing independent runs inside each sub-experiment. This keeps the repository-wide running-task ceiling enforceable.

```text
prepare
  -> A1 (9-task array)
  -> A2 (36-task array)
  -> A3 (12-task array)
  -> B  (36-task array)
  -> C  (3-task array)
  -> D  (24-task array)
  -> E  (9-task array)
  -> finalizer
```

Each task requests one CPU core. Default maximum concurrency is 20 and may be changed without editing scripts:

```bash
SLURM_MAX_CONCURRENCY=30 bash scripts/bash_script/SNN_Bash/submit_exp_13_cpu.bash
```

The wrapper rejects values outside 1–50.

Formal submission:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_13_cpu.bash
```

Inspect exact task mappings without Slurm:

```bash
python -m scripts.experiment_13_hierarchical_temporal_representation list-runs
```

## Artifact structure

```text
notebooks/artifacts/
  experiment_13_hierarchical_temporal_representation/
    hierarchical_temporal_representation_v2/
      source_manifest.json
      manifest.json
      A1_recurrence/
        recurrence_runs/
        activity_runs/
        anchors/
      A2_history_truncation/
        similarity_runs/
        features/
      A3_layer_reset/
        recovery_runs/
      B_context_expression/
        probe_runs/
      C_stroke_organization/
        boundary_runs/
        within_across_runs/
      D_stroke_scrambling/
        sensitivity_runs/
      E_cross_tau_decoding/
        neuron_runs/
        population_response/
```

The finalizer writes both all-run CSVs and grouped summary CSVs inside each sub-experiment directory, then writes the top-level `manifest.json`.

## Interpretation ladder

```text
L2 slower only
    -> temporal filtering

L2 slower + longer causal TRW
    -> longer memory

I/U longer but spike50 not longer
    -> history stored but communication loses it

spike50 longer + shared-probe context gain
    -> older history is expressed in the current communicated representation

+ true-boundary residual > motion-matched control
    -> stroke-level temporal organization

+ L2 is more stroke-order sensitive
    -> hierarchical contextual representation

+ trained network develops localized delayed fields
    -> mechanistic evidence for learned cross-timescale decoding
```

The strongest claim requires the full chain. Exp13 never infers hierarchy merely from slower dynamics or final classification accuracy.

## Required validation

```bash
python -m pytest -q tests/test_experiment_13_hierarchical_temporal_representation_contract.py
python -m pytest -q tests/test_repository_source_syntax.py
```
