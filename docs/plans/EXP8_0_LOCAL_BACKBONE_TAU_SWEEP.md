# Exp8.0 — Local Backbone Temporal-Scale Sweep

## Goal

Exp8.0 isolates one question:

> How should the two local SNN layers distribute synaptic time scales so that the learned representation is most discriminative, temporally informative, and compatible with a downstream spiking readout?

The experiment changes only the hidden-layer synaptic shift sets. Training otherwise follows Exp7.3 A2 (`A2_e2e_linear_wcce`) exactly: two 128-neuron binary LIF hidden layers, a bias-free 128→12 Linear classifier, end-to-end WCCE training, three paired seeds, the same data split, optimizer, checkpoint rule, epoch budget, and task-only objective.

## Main backbone sweep

| ID | Architecture | L1 shifts | L2 shifts | Primary interpretation |
|---|---|---|---|---|
| B0 | `234x234` | `(2,3,4)` | `(2,3,4)` | Exp7.3 A2 baseline |
| B1 | `123x234` | `(1,2,3)` | `(2,3,4)` | Faster L1 with the original L2 |
| B2 | `123x123` | `(1,2,3)` | `(1,2,3)` | Short/local hierarchy in both layers |
| B3 | `123x345` | `(1,2,3)` | `(3,4,5)` | Fast-to-long hierarchical expansion |
| B4 | `234x345` | `(2,3,4)` | `(3,4,5)` | Longer L2 without a faster L1 |

At 64 Hz the shifts correspond approximately to the existing repository shift-to-tau mapping. The experiment records the exact tau values in the final manifest rather than hard-coding rounded values into analysis.

## Controlled comparisons

The main interpretation uses paired seed differences rather than ranking all five architectures indiscriminately.

1. `123x234 - 234x234`: effect of making only L1 faster.
2. `123x234 - 123x123`: effect of expanding L2 from short to medium while L1 stays fast.
3. `123x345 - 123x234`: effect of expanding L2 from medium to long while L1 stays fast.
4. `123x345 - 234x345`: effect of a faster L1 when L2 is already long.

These contrasts distinguish fast primitive extraction from deeper temporal integration.

## Training contract

For every architecture and seed:

```text
30 input event channels
  -> L1: 128 binary LIF neurons
  -> L2: 128 binary LIF neurons
  -> bias-free Linear(128, 12)
```

The hidden membrane dynamics, threshold, reset rule, width, optimizer, LR, weight decay, batch size, train/val/test split, early stopping, and checkpoint selection are inherited from Exp7.3 A2.

The training trajectory is

```math
z_t = S^{L2}_t,
\qquad e_t = W z_t,
\qquad \bar e = \frac{1}{T}\sum_{t=1}^{T}e_t,
\qquad \mathcal L = CE(\bar e, y).
```

No auxiliary loss is added in Exp8.0. This is deliberate: the only intended training variable is the hidden synaptic time-scale allocation.

Seeds are `11, 23, 37`. Exp8.0 reuses the Exp7.3 A2 paired model-initialization and loader-order seeds so that `234x234` is a strict A2 replication control and all backbone comparisons are seed-paired.

## Post-training same-W Linear → LIF transfer

After selecting the best Linear checkpoint, freeze L1, L2, and the trained matrix `W`. Do not retrain or fine-tune anything.

Replace only the analog Linear readout realization with 12 output LIF neurons:

```math
I_t = Wz_t,
```

with output synaptic recurrence disabled:

```math
\alpha_{out}=0,
```

and membrane leak fixed to

```math
\beta_{out}=0.5.
```

The output threshold and one-spike-per-timestep cap remain the standard Exp7.3 deployment values. Classification uses valid-region output spike count.

The key derived quantity is

```math
P_{LIF}=BA_{Linear}-BA_{LIF}.
```

This separates representation/readout quality from LIF realization loss.

## Representation probes

After training, freeze the backbone and extract binary spike trajectories from both hidden layers.

For both L1 and L2, fit two repository-standard affine diagnostic probes using only train data for fitting/scaling and the established Exp7.3 probe implementation:

1. Whole-count probe
   - valid spike counts over the full sequence;
   - StandardScaler + multinomial linear probe with intercept, following the existing Exp7.3/P7-style diagnostic path.

2. Ordered Fixed250 probe
   - 250 ms = 16 timesteps at 64 Hz;
   - retain bin order;
   - flatten ordered per-bin spike counts;
   - use the same affine probe implementation.

Report train/val/test metrics, but the main summary table uses test balanced accuracy.

Derived representation metrics:

```math
G^{L1}_{temporal}=BA^{L1}_{F250}-BA^{L1}_{whole}
```

```math
G^{L2}_{temporal}=BA^{L2}_{F250}-BA^{L2}_{whole}
```

```math
G^{whole}_{L2-L1}=BA^{L2}_{whole}-BA^{L1}_{whole}
```

```math
G^{F250}_{L2-L1}=BA^{L2}_{F250}-BA^{L1}_{F250}.
```

## Spike-activity diagnostics

Because changing `tau_syn` can change firing statistics even when the task representation does not improve, every trained run records test-set activity separately for each shift group in L1 and L2:

- mean spikes/neuron/s;
- dead-neuron fraction;
- group neuron range and shift;
- output-LIF mean spikes/neuron/s and dead-neuron fraction.

This prevents interpreting a change caused only by a large firing-rate shift as a clean temporal-receptive-field gain.

## Raster contract

Every one of the 15 training runs must save rasters from the same deterministic representative test sample. The representative sample is selected once from the fixed test split as the sample whose valid length is closest to the test-set median; therefore the same input is used for all architectures and seeds.

Each run writes three raster PNGs:

```text
L1 neurons x timestep
L2 neurons x timestep
12 output-LIF neurons x timestep
```

L1/L2 figures mark shift-group boundaries and the valid-sequence endpoint. Output-LIF rasters use the post-training same-W `alpha=0, beta=0.5` transfer described above. A compressed trace NPZ is also saved for reproducibility.

The aggregation notebook contains a raster appendix that displays all three rasters for every architecture/seed run.

## CPU parallelization

There are exactly:

```text
5 architectures x 3 seeds = 15 training jobs
```

Use one Slurm CPU array task per `(architecture, seed)` pair. Each task uses one CPU thread and sets OMP/MKL/OpenBLAS/NumExpr thread counts to one. The array may run up to 15 tasks concurrently.

Each array task performs, in order:

1. A2-style training;
2. native Linear evaluation;
3. same-W output-LIF transfer evaluation;
4. L1/L2 whole-count and Fixed250 probes;
5. activity diagnostics;
6. L1/L2/output raster generation;
7. per-run JSON, checkpoint, history CSV, and raster artifacts.

A dependent finalizer runs only after the entire array succeeds.

## Finalizer and notebook aggregation

The finalizer must fail if any of the 15 evaluation JSON files are missing. It writes:

```text
method_runs.csv
method_summary.csv
contrast_runs.csv
contrast_summary.csv
activity_runs.csv
activity_summary.csv
raster_index.csv
manifest.json
```

The notebook reads only those finalized aggregate files plus raster images. It should not recompute models or probes.

The notebook main section shows method-level aggregated results only:

- Linear vs same-W LIF test BA;
- LIF realization penalty;
- L1/L2 whole-count and Fixed250 probe BA;
- temporal-information gains;
- L1→L2 representation gains;
- paired controlled contrasts;
- activity summaries.

A separate raster appendix displays L1, L2, and output-LIF rasters for every one of the 15 trained runs.

## Primary output table

The main table is organized as:

| Backbone | Linear BA | LIF BA | LIF penalty | L1 whole | L1 F250 | L2 whole | L2 F250 |
|---|---:|---:|---:|---:|---:|---:|---:|

All method-level values are reported as mean ± std across the three paired seeds.

## Interpretation rules

Exp8.0 should answer mechanism questions rather than simply selecting the highest bar:

- If `123x234 > 234x234`, faster first-layer dynamics improve primitive/local representation under an otherwise identical A2 objective.
- If `123x234 > 123x123`, L2 benefits from integrating over a broader temporal range than L1.
- If `123x345 > 123x234`, longer second-layer integration still adds useful information; if it decreases, L2 is over-smoothing or mixing phases.
- If `123x345 > 234x345`, a long L2 still benefits from a fast local front end.
- A high Fixed250 probe with a much lower whole-count probe means discriminative information exists but remains strongly tied to temporal order/phase.
- A large Linear→LIF penalty means the backbone may be informative but its evidence is poorly realized by the constrained output neuron.
