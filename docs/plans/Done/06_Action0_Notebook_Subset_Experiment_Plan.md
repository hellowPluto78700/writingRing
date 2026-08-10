# Plan: Action0 Notebook Subset Experiment Completion

## Goal

Complete and verify the supported Action0 notebook subset experiment without
changing the canonical CLI trainer or producer contracts.

The notebook must select a reproducible label subset present in every user
split, use one selected representation throughout, restore its in-memory best
validation model, evaluate all splits, and publish user-facing documentation
that distinguishes this experiment from CLI full-label training.

## Task DAG

```text
T001 Notebook subset workflow and result completion
  ↓
T002 Documentation and end-to-end acceptance
```
