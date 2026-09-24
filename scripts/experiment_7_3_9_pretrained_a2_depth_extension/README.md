# Experiment 7.3.9 — Pretrained A2 depth extension

## Question

Exp7.3 A2 already learns a two-layer local SNN with a bias-free Linear head and WCCE:

```text
Raw64 events -> L1(128, shifts 2/3/4) -> L2(128, shifts 2/3/4)
             -> bias-free Linear 128->12 -> valid-time mean evidence -> CE
```

This experiment asks whether a third layer can improve that already-trained hierarchy without conflating three different effects:

1. the value of inserting a third spiking transformation;
2. the need for upstream representation adaptation;
3. the contribution of class-specific probe bias.

## Cases

### C0 — reuse Exp7.3 A2

```text
L1(234) -> L2(234) -> Wout
```

No retraining. The exact Exp7.3 `A2_e2e_linear_wcce` checkpoints for seeds `11,23,37` are reused.

### C1 — freeze A2, train only the new third-layer matrix

```text
frozen A2 W1 -> frozen A2 W2 -> train W3 -> L3(234) -> frozen A2 Wout
```

The inserted L3 has:

- width 128;
- shifts `(2,3,4)`;
- the same hidden membrane/threshold/reset/event-cap contract as Exp7.3;
- binary spike communication.

Only `W3: 128 -> 128` is trainable. `W1`, `W2`, and the original bias-free A2 `Wout` remain bitwise frozen. Training uses the unchanged A2 WCCE objective.

This case asks whether a new spiking transformation can reorganize a fixed A2 L2 representation into something that the original A2 classifier already understands.

### C2 — initialize from selected C1, then full E2E WCCE

C2 loads the selected C1 checkpoint rather than creating another random L3. It then:

- creates a fresh Adam optimizer;
- reuses no C1 optimizer state;
- treats the C1 checkpoint as epoch 0 and a valid checkpoint candidate;
- unfreezes `W1/W2/W3/Wout`;
- continues with the same bias-free Linear/WCCE objective.

This case asks whether coordinated upstream adaptation is required for the third layer to become useful.

## Probe matrix

Every available hidden layer is evaluated with four probes:

| Representation | Affine probe | True no-bias probe |
|---|---|---|
| WholeCount | yes | yes |
| ordered Fixed250 | yes | yes |

The affine probe preserves the historical Exp7.3 contract:

```text
train-only StandardScaler(mean + scale)
-> LogisticRegression(fit_intercept=True)
```

The no-bias probe is deliberately different:

```text
train-only StandardScaler(with_mean=False)
-> LogisticRegression(fit_intercept=False)
```

No mean subtraction is allowed for the no-bias probe. Otherwise `W((x-mu)/sigma)` would introduce an effective constant class offset even when the classifier intercept is disabled.

Both probe families use the repository C grid and select C by validation balanced accuracy only. Test is never used for C selection.

## Primary comparisons

Native WCCE:

```text
C1 - C0 : value of inserting/training L3 while A2 is frozen
C2 - C1 : value of coordinated E2E adaptation
C2 - C0 : total gain/loss from staged deepening
```

Representation diagnostics:

```text
L3 - L2
affine - no_bias
Fixed250 - WholeCount
```

C1 also verifies after training that the source A2 `W1`, `W2`, and `Wout` are unchanged.

## Multi-CPU execution

The formal pipeline is:

```text
prepare/validate 3 source A2 checkpoints
        |
        v
3 C1 CPU tasks
        |
        v
3 C2 CPU tasks
        |
        v
9 probe tasks = 3 cases x 3 seeds
        |
        v
finalizer
```

Every training task uses one CPU core. The probe array also uses one CPU per case/seed and performs one forward extraction per split before fitting all layer/probe combinations.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_9_cpu.bash
```

## Artifacts

Finalized outputs are written under:

```text
notebooks/artifacts/
  experiment_7_3_9_pretrained_a2_depth_extension/
    pretrained_a2_depth_extension_v1/
```

Durable aggregate files:

- `manifest.json`
- `source_manifest.json`
- `native_runs.csv`
- `native_summary.csv`
- `probe_runs.csv`
- `probe_summary.csv`
- `depth_contrasts.csv`
- `bias_contrasts.csv`
- `temporal_contrasts.csv`
- `layer_contrasts.csv`

The notebook `notebooks/experiment_7_3_9_pretrained_a2_depth_extension.ipynb` is analysis-only and never trains or regenerates missing artifacts.
