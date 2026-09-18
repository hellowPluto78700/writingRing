# Exp10.2.2 — Valid-Length vs Whole-Window Hidden-State Probes

## Goal

Exp10.2.2 does not retrain the SNN. It reuses all 18 Exp10.2.1 checkpoints and re-evaluates hidden representations under the repository's dual temporal-support convention:

- `valid`: aggregate only timesteps `t < valid_length`;
- `window`: aggregate the complete padded inference window without zeroing hidden state after `valid_length`.

The experiment asks whether residual post-valid SNN dynamics contain useful class information and how that depends on coding (`binary` vs `weighted31`) and L2 membrane memory (22/54/117 ms).

## Source checkpoints

Source experiment:

~~~text
experiment_10_2_1_l2_membrane_weighted
d1_l1mem2_l2mem123_binary_weighted31_v1
~~~

The full 18-checkpoint matrix is reused:

~~~text
2 coding types x 3 L2 membrane settings x 3 seeds = 18 checkpoints
~~~

No SNN optimizer, backward pass, or weight update is allowed in Exp10.2.2.

## Hidden representations

For both L1 and L2:

- `syn_current`
- `pre_reset`
- `communication` (`spike` tensor in the source implementation)
- `post_reset`

For analog states (`syn_current`, `pre_reset`, `post_reset`):

- `whole_mean`
- `fixed250_ordered_mean`

For communication:

- `whole_mean`
- `fixed250_ordered_mean`
- `whole_count`
- `fixed250_count`

Each definition is evaluated under both temporal supports, giving:

~~~text
10 probes/layer x 2 layers x 2 supports = 40 probes/checkpoint
18 checkpoints x 40 probes = 720 probe rows
~~~

Probe names explicitly include support, e.g.:

~~~text
l1__pre_reset__valid__fixed250_ordered_mean
l1__pre_reset__window__fixed250_ordered_mean
l2__communication__valid__fixed250_count
l2__communication__window__fixed250_count
~~~

## Temporal-support semantics

### Valid-length masked

For counts:

\[
C_{valid}=\sum_{t<T_{valid}}s_t.
\]

For means, only valid timesteps contribute to numerator and denominator.

### Whole-window unmasked

For counts:

\[
C_{window}=\sum_{t<T_{window}}s_t.
\]

For means, the full padded window contributes. The input may already be zero after `valid_length`, but synaptic current, membrane state, reset dynamics, and resulting communication are not manually cleared.

Fixed250 uses absolute 16-timestep bins anchored at the beginning of the padded window in both modes.

## Probe classifier protocol

For every probe:

~~~text
Train representation
  -> StandardScaler fit on train only
  -> LogisticRegression
  -> choose C by validation balanced accuracy
  -> evaluate train/val/test
~~~

The C grid and LogisticRegression settings are inherited from the existing Exp10 probe protocol.

Changing temporal support does not change the selection criterion: validation BA remains the only hyperparameter-selection metric.

## Reported metrics

Every probe reports:

- Accuracy
- Balanced Accuracy
- Macro-F1

for train, validation, and test.

The main reported endpoint is test BA plus test Macro-F1.

## Primary paired contrast

For the same checkpoint/layer/state/aggregation:

\[
\Delta^{BA}_{window-valid}=BA_{window}-BA_{valid},
\]

\[
\Delta^{F1}_{window-valid}=F1_{window}-F1_{valid}.
\]

The same paired contrast is also stored for accuracy.

## Key diagnostics

The main summary highlights:

- L1 pre-reset Fixed250 mean;
- L1 communication Fixed250 count;
- L2 pre-reset Fixed250 mean;
- L2 communication Fixed250 count.

These directly test whether residual post-valid dynamics improve or degrade the representations that motivated Exp10.2/10.2.1.

## Statistical scope

Exp10.2.2 inherits Exp10.2.1 rotation0. Seeds 11/23/37 are paired optimization replicates, not independent user splits.

## Multi-CPU execution

~~~bash
bash scripts/bash_script/SNN_Bash/submit_exp_10_2_2_cpu.bash
~~~

Dependency graph:

~~~text
prepare / checkpoint preflight
    -> 18-way evaluation-only CPU array
        -> finalizer
~~~

Each array task loads exactly one frozen Exp10.2.1 checkpoint, performs forward inference, fits its 40 post-hoc probes, and writes evaluation artifacts.

## Outputs

Artifact root:

~~~text
notebooks/artifacts/experiment_10_2_2_valid_window_probes/
  exp10_2_1_checkpoints_valid_window_v1/
~~~

Primary outputs:

- `probe_runs.csv`
- `probe_summary.csv`
- `support_contrasts.csv`
- `support_contrast_summary.csv`
- `key_probe_contrasts.csv`
- `key_probe_summary.csv`
- `manifest.json`

The notebook is aggregation-only and never trains or evaluates checkpoints itself.
