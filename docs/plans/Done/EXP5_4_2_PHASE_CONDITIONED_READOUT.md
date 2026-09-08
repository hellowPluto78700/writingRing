# Exp5.4.2 — Phase-Conditioned Readout Mechanism Search

Status: implemented

## Scientific contract

Exp5.4.2 is a mechanism-search/oracle experiment. It freezes the Exp5.4.1 direct-WHAT whole-count classifier and the Exp5.3.2.3 `ffsnn128_rsnn64` causal WHEN source, then asks whether a low-capacity interaction can recover stable phase-dependent class evidence.

Stage A contains exactly ten conditions × five seeds with fixed `K=4`, `rank=4`. Candidate mechanisms are only bilinear or readout-bank interactions with spike/membrane WHEN. Additive WHEN, time-averaged WHEN, causal elapsed time, and true relative phase are controls.

Membrane scaling is fit from training valid timesteps only. Padding is excluded and the same train-derived scaler is reused on validation/test.

True interaction candidates preserve the architecture contract:

- frozen WHAT base;
- frozen WHEN branch;
- epoch 0 exactly equals WHAT-only;
- context zero gives zero correction;
- residual WHAT zero gives zero correction.

Screen/refine selection is validation-only. Test is not constructed during either selection stage. `selection.json` must be locked before the final test command can run.

Stage B refines only the winning mechanism: `K,r in {2,4,8}` for bank (45 tasks) or `r in {2,4,8}` for bilinear (15 tasks).

Final locked-test diagnostics include mean/shuffle/circular/reset/zero ablations, residual magnitude, rescue/harm, prefix BA, source activity, and bank-gate diagnostics.

Slurm execution follows `AGENTS.md`: one CPU per independent task, maximum 50 concurrent experiment tasks, and aggregation-only finalizers.

See `scripts/experiment_5_4_2/README.md` for the complete protocol and execution commands.
