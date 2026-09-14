# Exp7.2.5 — Output Synaptic Alpha: Frozen Readout vs End-to-End

Exp7.2.5 asks whether adding a short synaptic state to the 12-neuron output layer improves only readout/decoding or also improves end-to-end representation learning.

The output membrane decay is fixed at `beta=0.5`. The only output-dynamics ablation is:

- control: `alpha_out=0.0`, so `I_t = W z_t`;
- proposed: `alpha_out=0.5`, so `I_t = 0.5 I_{t-1} + W z_t`.

Both use `U_t^- = 0.5 U_{t-1} + I_t`, the same threshold/reset, binary output spikes, no output bias, and no `(1-alpha)` gain normalization.

## Scientific blocks

### Block A — Frozen-L2 output-head refit

Reuse the trained Exp7.2.3 S1-S4 representations:

- `S1 = s1_spike_wc_beta05`
- `S2 = s2_spike_tsce_beta05`
- `S3 = s3_spike_wc_beta10`
- `S4 = s4_spike_tsce_beta10`

For every source checkpoint, cache the complete L2 binary trajectory, freeze it, and train only a fresh `Linear(128,12,bias=False)` plus the output neuron dynamics.

Each source representation is decoded under a 2x2 matrix:

| objective | alpha=0, beta=.5 | alpha=.5, beta=.5 |
|---|---|---|
| WholeCount CE | control | proposed |
| timestep CE | control | proposed |

There are `2 architectures x 4 source families x 2 regularizers x 3 seeds = 48` frozen L2 sources and `48 x 2 objectives x 2 alpha = 192` output-head fits.

This block measures the readout-dynamics effect with L1/L2 fixed.

### Block B — Paired end-to-end training

Train a new two-hidden-layer SNN end-to-end with the same two Exp7.2.3 architectures:

- `234x234`: L1 `(2,3,4)`, L2 `(2,3,4)`
- `34x345`: L1 `(3,4)`, L2 `(3,4,5)`

Sweep:

- objective: `whole_count_ce`, `timestep_ce`
- output alpha: `0.0`, `0.5`
- regularization: `task_only`, `task_plus_reg`
- seed: `11,23,37`

Total: `2 x 2 x 2 x 2 x 3 = 48` end-to-end runs.

For a fixed `(architecture, regularization, seed, objective)`, `alpha=0` and `alpha=.5` use the exact same model-initialization seed and loader-order seeds. Alpha is excluded from all pairing seeds.

## Valid-length contract

Training always forwards the complete 256-step padded sequence, but the task objective only uses `t < valid_length`:

- WholeCount CE uses valid-length output spike aggregation.
- timestep CE evaluates CE only at valid timesteps.
- checkpoint selection always uses validation **valid-length WholeCount balanced accuracy**, with validation task-objective loss as tie-break.

The hidden-layer rate/persistence regularizer also uses valid length and remains hidden-only. The output synaptic state is not regularized.

After checkpoint selection, full-256 WholeCount BA and output tail activity are recorded only as diagnostics. They never affect checkpoint selection.

## Frozen-L2 cache

The cache-preparation stage runs each S1-S4 source checkpoint once and stores train/val/test L2 trajectories as compressed `uint8` arrays. The cache is shared by the four frozen output-head conditions, avoiding four repeated L1/L2 forward passes.

## E2E frozen probes

Every new E2E checkpoint receives two post-hoc valid-length probes:

- `l2_wholecount_linear`: 128-D valid L2 count + train-only StandardScaler + validation-selected LogisticRegression;
- `l2_fixed250_linear`: 16 x 128 ordered valid Fixed250 counts + the same probe protocol.

Probe randomness is paired across alpha conditions.

These probes separate output-readout changes from upstream representation changes.

## Main contrasts

Frozen readout gain:

`Delta_alpha_frozen = BA(alpha=.5) - BA(alpha=0)`

End-to-end gain:

`Delta_alpha_e2e = BA(alpha=.5) - BA(alpha=0)`

The finalizer also emits a diagnostic comparison against the matched historical beta=.5 source representation:

- WholeCount E2E is matched to S1;
- timestep-CE E2E is matched to S2.

`shaping_diagnostic = Delta_alpha_e2e - Delta_alpha_frozen_matched`

This is a mechanism diagnostic, not a strict additive causal decomposition.

## Multi-CPU execution

Submit with:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_5_cpu.bash
```

The submitter intentionally serializes the arrays to respect the repository-wide <=50 concurrent experiment CPU-task default:

1. 48-task `0-47%48` frozen-L2 cache array;
2. 192-task `0-191%50` frozen-head training/evaluation array after the cache array;
3. 48-task `0-47%48` end-to-end training/evaluation array after the frozen-head array;
4. one `afterok` finalizer after all E2E runs.

Each array task requests one CPU core and initializes Conda locally. The finalizer only aggregates existing artifacts; missing runs are an error.

## Finalized artifacts

The finalizer writes run-level CSVs for reproducibility and summary-level CSVs for the notebook. Important summary outputs are:

- `frozen_head_performance_summary.csv`
- `frozen_alpha_delta_summary.csv`
- `frozen_head_tail_summary.csv`
- `e2e_performance_summary.csv`
- `e2e_l2_probe_summary.csv`
- `e2e_alpha_delta_summary.csv`
- `e2e_probe_alpha_delta_summary.csv`
- `e2e_tail_summary.csv`
- `mechanism_alpha_gain_summary.csv`

Presentation tables are split by regularization rather than averaged together:

- `report_frozen_task_only.csv`
- `report_e2e_task_only_wc.csv`
- `report_e2e_task_only_tsce.csv`
- `report_frozen_task_plus_reg.csv`
- `report_e2e_task_plus_reg_wc.csv`
- `report_e2e_task_plus_reg_tsce.csv`

## Notebook

`notebooks/experiment_7_2_5_output_synaptic_alpha.ipynb` is analysis-only. It reads finalized summary/report CSVs and never trains models, loads checkpoints, or dispatches Slurm jobs.

The default reading order is:

1. task-only frozen-L2 table;
2. task-only E2E WholeCount table;
3. task-only E2E timestep-CE table;
4. task+regularizer frozen-L2 table;
5. task+regularizer E2E WholeCount table;
6. task+regularizer E2E timestep-CE table;
7. paired alpha gains, L2-probe gains, tail diagnostics, and architecture breakdowns.
