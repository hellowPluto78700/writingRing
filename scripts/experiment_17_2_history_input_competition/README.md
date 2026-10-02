# Experiment 17.2 — History–input competition and spike-decision dominance

## Purpose

Exp17.2 is an **artifact-only diagnostic extension** of Exp17. It does not
train a new model, change the CoreBenchmark contract, add a loss, add a gate,
or modify any source checkpoint.

The motivating observation is that ordinary WCCE training moves much of the L2
population into a high-firing regime. Exp17.1 also showed that neurons which
were in the epoch-20 low-occupancy quartile (about 4 Hz) naturally rise to
roughly 20+ Hz under continued baseline WCCE training.

The remaining mechanism question is:

\[
\boxed{
\text{Does accumulated synaptic history become strong enough that current
incoming evidence loses control over fire/not-fire decisions?}
}
\]

This is distinct from the already-established fact that history exists in the
network. Exp13 showed strong history dependence; Exp17.2 asks whether that
history increasingly dominates the **state-to-spike decision**.

## Locked source

Exp17.2 reads only the finalized Exp17 trajectory artifacts:

\`\`\`
notebooks/artifacts/
  experiment_17_persistent_pathway_story/
    persistent_pathway_story_v1/
      trajectory/seed11/
      trajectory/seed23/
      trajectory/seed37/
\`\`\`

For each seed it consumes:

- \`snapshot_manifest.json\`;
- every referenced \`snapshots/epoch_*.pt\`;
- \`trajectory_neurons.csv\`.

Formal seeds remain \`11, 23, 37\`. The locked two-layer CoreBenchmark
LIN/WCCE model remains width 128, shifts \`((2,3,4),(2,3,4))\`, 64 Hz,
\`tau_mem=22.54 ms\`, threshold 0.5, bias-free accumulator/WholeCount head.

## Exact drive decomposition

For each layer and valid timestep the implemented dynamics are decomposed as

\[
M_t = \beta V_{t-1},
\qquad
H_t = \alpha I_{t-1},
\qquad
N_t = W z_t,
\]

so that

\[
I_t = H_t + N_t,
\qquad
P_t = M_t + H_t + N_t,
\qquad
s_t = \mathbf 1[P_t \ge \theta].
\]

Interpretation:

- \(M_t\): short membrane carry;
- \(H_t\): persistent synaptic-history drive;
- \(N_t\): current incoming layer drive.

For L1, \(N_t=W_1x_t\). For L2, \(N_t=W_2s_t^{L1}\). Therefore the L2
"incoming-drive" diagnostic is a layer-local question; L1 spikes may already
contain history. A separate raw-\(x_t\) counterfactual addresses this caveat.

## Block A — Instantaneous spike-decision dominance

Every finalized Exp17 snapshot and all train/validation/test samples are
replayed. The diagnostic path must reproduce factual \`spike\`, \`pre_reset\`,
\`evidence\`, \`final_syn\`, and \`final_mem\` exactly.

### A1. Remove current incoming drive

Hold the factual previous states fixed and set only

\[
N_t = 0.
\]

Then

\[
s_t^{no-new}
=
\mathbf 1[M_t + H_t \ge \theta].
\]

Primary metrics:

\[
F_{new}
=
P(s_t^{full} \ne s_t^{no-new})
\]

and

\[
S_H
=
P(s_t^{no-new}=1 \mid s_t^{full}=1).
\]

The four decision outcomes are retained explicitly:

1. history sufficient: full=1, no-new=1;
2. new input triggers: full=1, no-new=0;
3. new input suppresses: full=0, no-new=1;
4. both off.

The third case is mandatory because signed weights allow current input to
actively correct a history-driven spike.

### A2. Separate synaptic history from membrane carry

Two additional instantaneous counterfactuals are computed:

\[
P_t^{no-syn-history}=M_t+N_t,
\]

\[
P_t^{no-mem-carry}=H_t+N_t.
\]

This reports \`syn_history_flip_fraction\` and
\`membrane_carry_flip_fraction\` separately. It directly tests the working
mechanistic assumption that the long history state resides mainly in synaptic
current rather than the short membrane state.

### A3. Raw-current-input counterfactual

At each timestep the factual previous L1/L2 states are held fixed, but the
current raw event vector is replaced with

\[
x_t'=0.
\]

The counterfactual L1 spike is propagated through L2 for that timestep only.
The primary metric is

\[
P(s_t^{L2,full}\ne s_t^{L2,x_t=0}\mid x_t\ne0).
\]

Conditioning on active raw input prevents already-zero input timesteps from
artificially diluting the result.

### A4. Analog drive statistics

Per neuron the experiment records:

- firing rate;
- mean absolute membrane carry;
- mean absolute synaptic-history drive;
- mean absolute current incoming drive;
- synaptic-history dominance
  \[
  |H|/(|H|+|N|+\epsilon);
  \]
- total carried-state dominance
  \[
  (|M|+|H|)/(|M|+|H|+|N|+\epsilon);
  \]
- factual and history-only threshold margins.

These continuous statistics prevent a binary flip metric from hiding large
subthreshold changes.

## Block B — Training trajectory

The principal evidence is longitudinal, not only final-snapshot correlation.

For every Exp17 snapshot the experiment relates:

- firing rate;
- incoming-drive flip fraction;
- history-sufficient spike fraction;
- raw-input flip fraction;
- history/new-drive magnitude balance.

The key pattern supporting the hypothesis is

\[
q(e)\uparrow,
\qquad
S_H(e)\uparrow,
\qquad
F_{new}(e)\downarrow
\]

consistently across seeds.

### Fixed epoch-20 groups

Two grouping systems are retained:

- dynamic high25/middle50/low25 at each epoch, ranked by that epoch's
  training firing rate;
- fixed epoch-20 high25/middle50/low25, defined once from **training-split
  occupancy only** and never reassigned.

The fixed groups directly track the known epoch-20 low-rate population as it
moves toward the later high-firing regime.

### Tau groups

All L2 metrics are also summarized by the locked shift groups 2/3/4. A stronger
history-dominance effect for larger alpha is a mechanistically plausible
pattern, not a hard success criterion because learned weights can compensate.

### Relative sequence time

Valid timesteps are divided into ten normalized \(t/T\) bins for diagnostics
only. This is not a Relative10 training objective or classifier. It tests
whether input leverage is greater early in a gesture and history dominance
greater later.

## Block C — Short-horizon write influence

A low instantaneous flip rate does **not** by itself show that current input is
useless: \(N_t\) can change \(I_t\) without changing the current binary spike.

For this reason Exp17.2 includes a second intervention on L2.

At an anchor timestep:

1. preserve factual \(I_{t-1}^{L2},V_{t-1}^{L2}\);
2. set the anchor L2 incoming drive to zero;
3. from the next timestep onward restore the factual L1 spike drives;
4. propagate only the counterfactual L2 state forward.

Metrics are evaluated after \`1,2,4,8,16\` steps
(15.625–250 ms at 64 Hz):

- future spike flip fraction;
- absolute synaptic-state delta;
- absolute membrane-state delta;
- current native-evidence delta norm;
- cumulative spike-count delta;
- cumulative native-evidence delta norm.

To keep this artifact replay bounded, this block uses epochs
\`0,20,40,60,80\` plus each seed's selected-best and stopped epochs, train and
test splits, and anchor stride 8. Anchors require an actual factual L1 spike at
that timestep.

## Interpretation contract

### History-dominance support

Support requires a seed-consistent longitudinal pattern in which firing rises,
history-sufficient spikes rise, and current incoming-drive flip fraction falls.

### State-to-spike communication bottleneck

If instantaneous input leverage falls but the short-horizon intervention still
causes substantial future state/evidence changes, new information is being
written into \(I\) but is poorly expressed in the immediate binary spike.

### State-updateability bottleneck

If both instantaneous input leverage and short-horizon future state/evidence
effects collapse, old history is dominating not only communication but the
ability to update the persistent state. That outcome would motivate selective
forgetting/adaptive decay rather than only an output-communication mechanism.

### Hypothesis not supported

If firing rises without a seed-consistent reduction in current-input leverage,
high firing cannot be explained by history drowning out new input.

These rules are descriptive mechanism criteria, not automatic causal proof.

## Outputs

\`\`\`
notebooks/artifacts/
  experiment_17_2_history_input_competition/
    history_input_competition_v1/
      protocol.json
      seed11/
        neuron_dominance.csv
        snapshot_summary.csv
        relative_time_summary.csv
        short_horizon.csv
        audit.json
      seed23/
      seed37/
      aggregate/
        neuron_dominance.csv
        trajectory_summary.csv
        relative_time_summary.csv
        short_horizon_summary.csv
        epoch20_fixed_group.csv
        dynamic_group_summary.csv
        tau_group_summary.csv
        hypothesis_summary.json
        manifest.json
\`\`\`

Raw \`[sample,time,neuron]\` current tensors are deliberately not persisted;
statistics are aggregated while replaying batches.

## Hard contracts

- no SNN training or checkpoint modification;
- factual instrumented replay must exactly equal \`BenchmarkNet.forward()\`;
- only valid timesteps contribute to diagnostics;
- epoch-20 fixed groups use training occupancy only;
- test data never choose checkpoints, groups, intervention settings, or
  hyperparameters;
- CoreBenchmark source files and protocol are unchanged;
- finalizer only aggregates already-written task outputs and fails if any are
  missing.

## Slurm execution

\`\`\`bash
bash scripts/bash_script/SNN_Bash/submit_exp_17_2_cpu.bash
\`\`\`

The DAG is:

\`\`\`
prepare
  -> replay[seed11,seed23,seed37]
  -> horizon[seed11,seed23,seed37]
  -> finalize
\`\`\`

Replay and horizon arrays use one CPU core per seed. The finalizer performs no
model replay.
