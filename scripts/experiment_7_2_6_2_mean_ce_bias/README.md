# Exp7.2.6.2 — Mean-CE output-bias ablation

## Question

Exp7.2.6.1 showed that replacing `CE(valid_sum(logits))` with `CE(valid_mean(logits))` recovers most of the Exp7.2.4 A2 performance gap. The remaining controlled difference is that Exp7.2.4 A2 uses a shared analog linear head with bias, while Exp7.2.6.1 keeps the hardware-matched projection bias-free.

This experiment isolates that remaining variable:

\[
\boxed{\text{Mean-CE, bias=False} \quad \text{vs} \quad \text{Mean-CE, bias=True}}
\]

The primary question is whether output bias explains the residual accuracy gap and, if it helps, whether the benefit is a direct readout effect or an E2E representation-shaping effect.

## Fixed protocol

- architecture: `234x234`, i.e. L1 shifts `(2,3,4)` and L2 shifts `(2,3,4)`;
- hidden width: 128;
- regularization: `task_only` only;
- seeds: `11,23,37`;
- objective: valid-length Mean-CE;
- optimizer, LR, weight decay, batch size, max epochs, minimum epochs, patience, model-init seed and loader seed streams: exactly paired to Exp7.2.6.1;
- checkpoint selection: highest validation BA; validation Mean-CE loss is the tie-break.

## Conditions

### B0 — reused bias-free baseline

Reuse the completed Exp7.2.6.1 Mean-CE run:

\[
\ell = W\left(\frac{1}{T}\sum_t z_t\right)
\]

with `bias=False`.

No B0 retraining is performed.

### B1 — Mean-CE with bias

The new model uses:

\[
\ell = W\left(\frac{1}{T}\sum_t z_t\right)+b.
\]

For exact pairing, each B1 model is first constructed as the same Exp7.2.6.1 bias-free model. The hidden weights and output `W` therefore receive exactly the same initialization. The output layer is then replaced by `Linear(128,12,bias=True)`, the paired `W` is copied exactly, and:

\[
b_0=0.
\]

Thus the only additional trainable parameters are the 12 bias terms.

## Direct-vs-training decomposition

For every trained B1 checkpoint, inference is performed twice on the same representation:

1. `bias kept`: \(W\bar z+b\)
2. `bias zeroed`: \(W\bar z\)

Together with the separately trained B0 baseline, the test-BA effect is decomposed exactly as:

\[
\Delta_{total}
=BA_{B1,+b}-BA_{B0}
\]

\[
\Delta_{direct}
=BA_{B1,+b}-BA_{B1,b=0}
\]

\[
\Delta_{E2E}
=BA_{B1,b=0}-BA_{B0}
\]

so that:

\[
\boxed{\Delta_{total}=\Delta_{direct}+\Delta_{E2E}}
\]

`direct` measures the final decision-boundary contribution of the learned bias on a fixed B1 representation. `E2E` measures changes induced during training, including any upstream L1/L2 representation change and any paired-W change caused by training with a bias parameter.

## L2 probes

After B1 training, L2 is frozen and evaluated with the same probes as Exp7.2.6/7.2.6.1:

- matched bias-free WholeCount linear;
- standardized WholeCount linear;
- Fixed250 linear.

The probe delta `B1-trained L2 - B0-trained L2` is the main independent test of representation shaping.

## Bias diagnostics

For each seed the experiment stores:

- learned 12-D bias vector;
- bias L2 norm, mean absolute value, standard deviation and range;
- ratio of bias L2 magnitude to the mean zero-bias logit L2 magnitude;
- fraction of predictions changed by restoring the bias;
- fraction rescued from wrong to correct;
- fraction harmed from correct to wrong;
- sum/mean readout equivalence check for the complete per-timestep logits including bias.

Because

\[
\sum_t(Wz_t+b)=T\left(W\bar z+b\right),
\]

sum and mean inference predictions must remain identical for a fixed model.

## Multi-CPU execution

Only three new E2E runs are required:

1. 3-way CPU array: B1 Mean-CE bias=True training, one seed per task;
2. `afterok`: 3-way CPU array for L2 probes;
3. `afterok`: single finalizer producing aggregate CSV/JSON outputs.

Run:

```bash
bash scripts/bash_script/SNN_Bash/submit_exp_7_2_6_2_cpu.bash
```

The analysis notebook is aggregate-only:

`notebooks/experiment_7_2_6_2_mean_ce_bias.ipynb`
