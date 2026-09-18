# Exp11.0 plan — RSNN history internalization

## Goal

Test whether recurrent SNN state can convert the temporal information revealed by the Exp10.2.2 Fixed250 probes into a history-aware representation that one time-shared output matrix can decode.

## Preserved contracts

- D1 post-encode-mask dataset.
- Exp10.2 locked cross-user rotation0.
- Seeds 11/23/37.
- L1 128 neurons, membrane shift2, original synaptic shifts (2,3,4), binary spikes.
- Primary objective is valid-length time-shared WCCE only.
- Exp10.2.2 valid/window dual-support probe semantics.
- Validation BA selects checkpoints; validation objective loss breaks ties.

## Required behavior

1. Train two L1 initialization modes: dynamics-only random input weight, and seed-matched pretrained Exp10.2.1 input->L1 weight.
2. Train three RSNN topologies: ff recurrence-off control, diagonal self-recurrence, and dense 128x128 recurrence.
3. Use tau_syn ~= tau_mem ~= 54 ms for RSNN and Fusion.
4. Fusion must receive both current L1 binary local spikes and RSNN binary context spikes and must not itself be recurrent.
5. All conditions remain end-to-end trainable.
6. Save per-run checkpoint, history, native/window metrics, output-LIF diagnostic, hidden probes, firing activity, recurrent/external-input diagnostics, parameter counts.
7. Finalizer must report paired recurrence effects, pretrained-L1 effects, interactions, and Fixed250-minus-whole temporal-gap movement through L1 -> RSNN -> Fusion.
8. Use one CPU core per run in an 18-task Slurm array; finalizer uses afterok dependency.
9. Notebook reads finalized artifacts only.

## Acceptance criteria

- Exactly 18 unique run specifications.
- Pretrained mode copies only the seed-matched Exp10.2.1 binary/l1mem2/l2mem1 input->L1 weight and leaves it trainable.
- Diagonal recurrence has exactly 128 trainable recurrent parameters and no cross-neuron mixing.
- Dense recurrence has exactly 16,384 recurrent parameters.
- FF control has zero recurrent parameters.
- RSNN and Fusion decay constants correspond to approximately 54 ms at 64 Hz.
- Primary probes exist for L1, RSNN and Fusion under valid and window supports.
- Finalized outputs contain native BA, Fusion whole/Fixed250 BA, and their temporal gap.
- Focused Exp11.0 contract test and repository source-syntax test pass before merge.
