# Exp11.0 plan — paired D0/D1 RSNN history internalization

## Goal

Test whether recurrent SNN state can internalize the temporal information exposed by Exp10.2.2 Fixed250 probes, and whether the mechanism behaves differently for D0 original events versus D1 post-encode-masked events.

## Preserved contracts

- D0 and D1 contain identical sample identities, labels, valid lengths, and user split geometry.
- Locked cross-user rotation0.
- Seeds 11/23/37.
- L1 width 128, membrane shift2, synaptic shifts (2,3,4), binary communication.
- RSNN and Fusion tau_syn ~= tau_mem ~= 54 ms.
- Fusion is non-recurrent and receives both L1 local spikes and RSNN context spikes.
- Primary objective is valid-length time-shared WCCE only.
- Exp10.2.2 valid/window probe semantics.
- Validation BA selects checkpoints; validation objective loss is the tiebreak.

## Matched pretrained-source stage

D1 already has seed-matched Exp10.2.1 binary/l1mem2/l2mem1 checkpoints.

No equivalent D0 source exists. Before the main factorial, train exactly three D0 source checkpoints with the same 30 -> 128 -> 128 -> 12 backbone, binary communication, L1 mem shift2, L2 mem shift1, synaptic shifts (2,3,4), valid-mean WCCE, optimizer, early stopping, checkpoint selection, and seed-specific initialization/loader order.

The main D0 pretrained branch copies only the source input->L1 matrix. The main D1 pretrained branch copies only the corresponding Exp10.2.1 input->L1 matrix. Copied weights remain trainable.

## Main factorial

2 datasets x 2 L1 init modes x 3 recurrence topologies x 3 seeds = 36 runs.

Topologies:

- ff: no recurrence
- diagonal: neuron-wise self recurrence
- dense: full 128x128 recurrence

All common random weights and DataLoader order are paired across D0/D1 for a fixed seed.

## Required outputs

Per main run: checkpoint/history, valid-native metrics, whole-window metrics, output-LIF diagnostic, L1/RSNN/Fusion valid/window probes, firing/activity diagnostics, recurrent input diagnostics, and parameter counts.

Finalized outputs: method summaries grouped by variant/init/topology, paired recurrence contrasts, paired pretrained-input contrasts, paired D1-D0 contrasts, D1 x recurrence interactions, pretraining x recurrence interactions, temporal-gap summaries through L1 -> RSNN -> Fusion, and D0 source metrics.

## Multi-CPU execution

Use one 3-task CPU array for D0 matched-source training, one 36-task CPU array for the main factorial, one CPU core per task, afterok dependencies, an aggregation-only finalizer, and an analysis-only notebook.

The 36-task array intentionally runs D0 and D1 concurrently.

## Acceptance criteria

- Exactly 3 D0 source specs.
- Exactly 36 unique main specs.
- D0/D1 geometry is verified paired before training.
- D0 pretrained mode cannot use a D1 source checkpoint.
- D1 pretrained mode must use the seed-matched Exp10.2.1 binary/l1mem2/l2mem1 checkpoint.
- Only input->L1 is copied; copied parameters remain trainable.
- ff/diagonal/dense recurrent parameter counts are 0/128/16384.
- Main Slurm array is 0-35 with maximum concurrency 36.
- D0 source Slurm array is 0-2 with maximum concurrency 3.
- Finalizer contains paired D1-D0 comparisons.
- Notebook is aggregation-only.
- Focused Exp11.0 contract tests and repository source-syntax test pass.
