# Exp7.3.4 — Linear-pretrained LIF output fine-tuning

## Scientific question

Exp7.3 showed that a weight matrix learned through a Linear/WCCE path can outperform a weight matrix learned directly through the beta=0.5 LIF surrogate path when both are deployed through the same output LIF.

Exp7.3.4 asks whether that gap is primarily an initialization problem or whether the LIF training objective itself moves a good Linear solution toward a worse region.

The experiment freezes the Exp7.3 WCCE-trained SNN representation and trains only the final `128 -> 12` matrix through the standard output LIF.

```text
Frozen L1/L2 -> W -> LIF(beta=0.5, threshold=0.5, cap=1)
                         |
                         +-> valid spike-count WCCE
```

For the two pretrained conditions:

```text
W(epoch 0) = pretrained Linear W from A2 or B6
```

Then epochs >= 1 optimize only `W` through the LIF readout using the same WCCE objective used by Exp7.3.

## Locked protocol

- architecture: `234x234`;
- hidden representation: exact Exp7.3 WCCE backbone, sourced from `A2_e2e_linear_wcce`;
- L1/L2 are frozen by reusing the Exp7.3 cached L2 trajectories;
- seeds: `11,23,37`;
- output head: bias-free `128 -> 12` matrix;
- output LIF: `beta=0.5`, repository threshold `0.5`, max one spike/neuron/timestep;
- training objective: LIF spike WCCE only;
- optimizer/LR/weight decay/max epochs/min epochs/patience reuse Exp7.3;
- only trainable parameter: `output_linear.weight`;
- no auxiliary Linear loss;
- no bias, gain, threshold, beta, or backbone tuning.

## Conditions

Each seed runs three independent head-training tasks on the same frozen WCCE L2 cache.

### 1. A2 pretrained initialization

Load the output matrix from the selected Exp7.3 checkpoint:

```text
A2_e2e_linear_wcce
```

Use that matrix unchanged at epoch 0 for the direct-transfer reference, then fine-tune only that matrix through the output LIF.

### 2. B6 pretrained initialization

Load the selected Exp7.3 Stage-2 Linear/WCCE matrix:

```text
B6_wcbackbone_linear_wcce
```

Use that matrix unchanged at epoch 0 for the direct-transfer reference, then fine-tune only that matrix through the output LIF.

Because B6 already uses the same frozen WCCE backbone sourced from A2, A2-vs-B6 comparisons isolate the pretrained projection rather than hidden representation differences.

### 3. Random initialization control

Instantiate the same Exp7.3 Stage-2 LIF/WCCE head with the paired Exp7.3 Stage-2 model-init seed and loader order. This control should reproduce:

```text
B8_wcbackbone_lif_wcce
```

This makes the initialization comparison explicit:

```text
random W -> LIF/WCCE training
A2 Linear W -> LIF/WCCE fine-tuning
B6 Linear W -> LIF/WCCE fine-tuning
```

## Epoch-0 direct-transfer reference

For A2 and B6, epoch 0 is evaluated before any optimizer update.

```text
Frozen L2 -> pretrained Linear W -> same beta=0.5 LIF -> spike-count readout
```

Epoch 0 is retained as a baseline but is **not eligible** for the fine-tuned checkpoint selection. This keeps the following comparison honest:

```text
fine-tuned best epoch >= 1  vs  unchanged pretrained W at epoch 0
```

## Checkpoint selection

Training runs retain:

- `epoch0`;
- `best_val_ba` — validation LIF balanced accuracy primary, validation LIF WCCE loss tiebreak;
- `best_val_loss` — lowest validation LIF WCCE loss, validation BA tiebreak;
- `final`.

The primary reported trained result is `best_val_ba`.

Test is evaluated only after training/checkpoint selection is complete.

## Per-epoch diagnostics

For the same current `W`, every epoch records both:

1. LIF readout:

```text
L2 -> W -> LIF -> valid spike-count BA/loss
```

2. Linear bypass diagnostic:

```text
L2 -> W -> valid mean evidence BA/loss
```

The Linear bypass is evaluation-only and never contributes to the training loss.

This distinguishes useful LIF adaptation from destruction of the original class geometry.

Each epoch also records:

- relative Frobenius drift from initialization;
- weight norm ratio;
- mean/min class-row cosine similarity to initialization;
- validation mean output spikes/sample;
- validation silent-sample fraction;
- validation output-spike fraction;
- validation spike-count margin.

Selected checkpoints additionally retain train/val/test LIF metrics, Linear-bypass metrics, and spike diagnostics.

## Primary method table

The finalized `method_runs.csv` contains five methods x three seeds:

```text
A2_linearW_direct_lif
A2_linearW_lif_finetune
B6_linearW_direct_lif
B6_linearW_lif_finetune
randomW_lif_train
```

The most important paired contrasts are:

```text
A2 fine-tune - A2 direct transfer
B6 fine-tune - B6 direct transfer
A2 fine-tune - random LIF training
B6 fine-tune - random LIF training
B6 direct - A2 direct
B6 fine-tune - A2 fine-tune
```

Interpretation:

- `fine-tune > direct`: LIF optimization provides useful dynamics adaptation after Linear pretraining;
- `direct > fine-tune`: LIF optimization moves a good pretrained solution in a harmful direction;
- `pretrained fine-tune > random`: Linear pretraining provides a better optimization basin;
- `pretrained fine-tune ~= random` while direct is better: initialization alone is not the dominant problem.

## Multi-CPU strategy

Exp7.3.4 follows `AGENTS.md` task-level CPU execution.

```text
3 seeds x 3 initialization sources = 9 independent training tasks
        |
        +-> each task: load frozen cache -> train W -> select checkpoints
                       -> evaluate selected states -> write artifacts
        |
        +-> one afterok finalizer aggregates existing artifacts only
```

Each task uses one CPU core. No training occurs in the finalizer or notebook.

Launch from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_3_4_cpu.bash
```

## Finalized outputs

Artifacts are written under:

```text
notebooks/artifacts/experiment_7_3_4_linear_pretrained_lif_finetuning/linear_pretrained_lif_finetuning_v1/
```

Important files:

- `manifest.json`
- `method_runs.csv`
- `method_summary.csv`
- `contrast_runs.csv`
- `contrast_summary.csv`
- `checkpoint_diagnostics.csv`
- `source_reproduction_checks.csv`
- `history_runs.csv`
- `history_summary.csv`
- per-run `evaluations/*.json`
- per-run `checkpoints/*.pt`
- per-run `histories/*.csv`

## Notebook contract

`notebooks/experiment_7_3_4_linear_pretrained_lif_finetuning.ipynb` is analysis-only. It reads finalized CSV/JSON artifacts and presents method-level summaries only. It does not import Torch, invoke Slurm, retrain models, or regenerate missing artifacts.
