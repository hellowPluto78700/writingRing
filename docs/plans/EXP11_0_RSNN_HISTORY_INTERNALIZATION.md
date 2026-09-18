# Exp11.0 plan — full A/B/C/D context × fusion factorial

## Goal

Test whether the temporal information exposed by external Fixed250 probes can be internalized by recurrent SNN dynamics, by a history-conditioned Fusion transform, or by their interaction.

## Architecture factor

Use the complete 3 context topologies × 2 Fusion states:

- A: FF context, Fusion off — `L1 -> FF -> Linear`
- B-diag/B-dense: recurrent context, Fusion off — `L1 -> RSNN -> Linear`
- C: FF context, Fusion on — `L1 -> FF context -> Fusion -> Linear`
- D-diag/D-dense: recurrent context, Fusion on — `L1 -> RSNN context -> Fusion -> Linear`

Fusion off must instantiate no Fusion weight matrices. Fusion on must combine both L1 local spikes and context spikes.

The final hidden state is exposed uniformly as `readout`: context output when Fusion is off, Fusion output when Fusion is on.

## Preserved contracts

- D0 original and D1 post-encode-mask.
- Paired D0/D1 sample identities, labels, valid lengths, split geometry, seed streams, and DataLoader ordering.
- Rotation0.
- Seeds 11/23/37.
- L1 width 128, membrane shift2, synaptic shifts (2,3,4), binary spikes.
- Context width 128, tau_syn ~= tau_mem ~= 54 ms.
- Fusion width 128, tau_syn ~= tau_mem ~= 54 ms, no recurrence.
- Valid-mean time-shared WCCE only.
- Exp10.2.2 valid/window probe semantics.
- Validation BA primary checkpoint criterion and validation loss tiebreak.
- D0 pretrained-input source remains matched D0 source; D1 source remains seed-matched Exp10.2.1.

## Main factorial

2 datasets × 2 L1 init modes × 3 context topologies × 2 Fusion states × 3 seeds = 72 main runs.

## Required contrasts

Finalizer must report:

- diagonal/dense recurrence minus FF separately for Fusion off and on;
- Fusion on minus off separately for FF/diagonal/dense;
- recurrence × Fusion interaction `(D-C)-(B-A)`;
- paired D1-D0;
- pretrained-input minus dynamics-only;
- D1 × recurrence;
- D1 × Fusion;
- pretraining × recurrence.

## Mechanism diagnostics

For L1, context/RSNN, and readout, evaluate valid/window whole and Fixed250 probes. Main temporal diagnostic is `Fixed250 BA - Whole BA`.

For Fusion-off runs, readout state must equal context state exactly. For Fusion-on runs, readout is the Fusion output.

Record recurrent/external input ratio, recurrent weight norm, firing activity, dead-neuron fraction, and post-valid residual firing.

## Multi-CPU execution

- 3-task D0 matched-source array.
- 72-task main array with one CPU core per task.
- Cap main concurrency at 50: `#SBATCH --array=0-71%50`.
- afterok finalizer.
- finalizer aggregates only.
- notebook is analysis-only.

## Acceptance criteria

- Exactly 72 unique main run specs.
- A/B/C/D case mapping is correct for all topology/Fusion combinations.
- Fusion-off models contain no Fusion weight matrices.
- Fusion-off readout state is identical to context state.
- Fusion-on models instantiate both local and context Fusion projections.
- D0/D1 common initializations remain paired.
- Diagonal recurrence has 128 recurrent parameters; dense has 16,384; FF has 0.
- Probe inventory uses `l1 / rsnn / readout`.
- Finalizer contains `fusion_on_minus_off` and `recurrence_x_fusion`.
- Slurm main array is `0-71%50`.
- Focused Exp11.0 contract tests and repository source-syntax test pass.
