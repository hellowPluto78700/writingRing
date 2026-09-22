# Experiment 12.0 — Local Primitive Bottleneck

## Goal

Test whether the Exp7.3 A2 local SNN representation can be reorganized into a compact
local-primitive representation and then sparsified into useful events without first
adding a long-memory classifier.

The local SNN is fixed to:

[
L1:(2,3,4), qquad L2:(2,3,4).
]

The experiment uses the repository's single fixed user-disjoint split and three
training seeds: `11, 23, 37`.

## Temporal evidence modes

All three modes are retained as first-class empirical cases:

- `t0`: no added integrator, (h_t=z_t).
- `raw`: (h_t=eta h_{t-1}+z_t).
- `ema`: (h_t=eta h_{t-1}+(1-eta)z_t).

The shared decay is (eta=15/16), approximately 242 ms at 64 Hz. RAW and EMA
have the same temporal kernel shape and differ in injection scale; their comparison
is therefore interpreted as a parameterization/optimization comparison, not a
memory-duration comparison.

## Primitive representation

A bias-free 128-to-16 projection produces:

[
s_t=W_ph_t.
]

The main representation separates three quantities:

[
hat{s}_t =
rac{s_t-mathrm{mean}(s_t)}
{mathrm{std}(s_t)+epsilon},
]

[
q_t=mathrm{softmax}(hat{s}_t)
quad	ext{(WHAT)},
]

[
a_t=mathrm{RMS}(h_t)
quad	ext{(HOW MUCH)},
]

and winner margin in normalized primitive space is used as confidence
(HOW CERTAIN).

The dense main representation is:

[
r_t=a_tq_t.
]

## Dense training cases

For each temporal mode and seed the CPU array trains four cases:

- `r0b_analog`: analog 16-D primitive projection.
- `r1_what`: normalized WHAT-only representation.
- `r2_main_frozen`: WHAT × HOW-MUCH with the A2 L1/L2 backbone frozen.
- `r2_main_e2e`: same R2 representation with L1/L2 fine-tuned from the same A2
  checkpoint; backbone learning rate is 0.1 times the head learning rate.

All use valid-timestep summed evidence and a bias-free 12-class linear classifier.

## Fixed-transform diagnostics

The trained `r0b_analog` encoder is reused without changing L1/L2 or (W_p) to
fit matched linear probes on:

- R0: direct L2 (z_t).
- R0a: 128-D integrated state (h_t).
- R0b: 16-D analog primitive state (s_t).
- R1 Main: normalized primitive distribution (q_t).
- R2 Main: (a_tq_t).
- Legacy R1: (mathrm{softmax}(s_t)).
- Legacy R2: (mathrm{RMS}(s_t)mathrm{softmax}(s_t)).

The existing Exp7.4 artifacts remain the exact historical C/D reproduction
reference; the legacy transforms here are controls under the new Exp12 setup.

## Post-hoc sparse tokenizer

R3 and R4 do not train through a discrete selection operator in this phase. They
operate on fixed R2 encoders and only refit the final linear probe.

R3 keeps timesteps whose normalized winner margin exceeds a training-set
confidence quantile.

R4 keeps causal same-primitive local confidence peaks. At time (t), the
candidate is (t-1); the candidate primitive (k^*) is fixed first and its margin
against all other primitives is recomputed at (t-2,t-1,t). No refractory is
used in this phase.

Quantiles are 0.20, 0.40, 0.60, 0.80, and 0.90. This produces BA-vs-sparsity and
BA-vs-events/sample curves without retraining an encoder for every threshold.

Additional diagnostics include hard-token refits, magnitude-only features,
R4 count-only features, primitive effective vocabulary size, L2 Fixed250 probe,
and L2 whole-count probe.

## Multi-CPU execution

Independent training runs use one CPU core each:

- Dense array: 36 tasks = 3 temporal modes × 4 dense methods × 3 seeds.
- Post-hoc array: 27 tasks = 3 temporal modes × 3 source families × 3 seeds.
- Finalizer runs only after all post-hoc tasks succeed.

Submit from the repository root:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_12_0_cpu.bash
```

The finalizer writes aggregate CSVs and a PASS manifest under:

```text
notebooks/artifacts/experiment_12_0_local_primitive_bottleneck/
  local_primitive_bottleneck_v1/
```
