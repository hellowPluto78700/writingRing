# -----------------------------------------------------------------------------
# Development: choose one lambda per contrastive condition by mean validation
# SNN-only probe BA across all three seeds. lambda=0 is included explicitly, so
# contrastive training must beat the no-contrastive control to be selected.
# Test is never accessed here.
# -----------------------------------------------------------------------------
CONTRASTIVE_CONDITIONS = ("con250", "con500")

development_rows = []
for condition in CONTRASTIVE_CONDITIONS:
    for lambda_con in LAMBDA_CON_GRID:
        for seed in SEEDS:
            ckpt = CHECKPOINT_DIR / f"dev_{condition}_lambda_{lambda_con:g}_seed_{seed}.pt"
            payload = train_run(condition, seed, lambda_con, ckpt)
            model = model_from_payload(payload)
            ztr250, _ = extract_features(model, X_train)
            zva250, _ = extract_features(model, X_val)
            probe = fit_logreg_probe(
                ztr250,
                zva250,
                None,
                seed_tag=(RUN_VARIANT, "dev", condition, lambda_con, seed),
            )
            development_rows.append({
                "condition": condition,
                "lambda_con": lambda_con,
                "seed": seed,
                "best_epoch": payload["best_epoch"],
                "global_val_BA": payload["val_best"]["balanced_accuracy"],
                "probe_selected_C": probe["selected_C"],
                "probe_val_BA": probe["val"]["balanced_accuracy"],
                "train_l3_fr": payload["train_best"]["l3_fr"],
                "active_item_fraction": payload["train_best"]["active_item_fraction"],
                "valid_anchor_fraction": payload["train_best"]["valid_anchor_fraction"],
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

development_df = pd.DataFrame(development_rows)
development_summary = (
    development_df.groupby(["condition", "lambda_con"], as_index=False)
    .agg(
        mean_probe_val_BA=("probe_val_BA", "mean"),
        sd_probe_val_BA=("probe_val_BA", "std"),
        mean_global_val_BA=("global_val_BA", "mean"),
        mean_train_l3_fr=("train_l3_fr", "mean"),
        mean_active_item_fraction=("active_item_fraction", "mean"),
        mean_valid_anchor_fraction=("valid_anchor_fraction", "mean"),
    )
)
development_df.to_csv(RESULTS_DIR / "development_sweep_by_seed.csv", index=False)
development_summary.to_csv(RESULTS_DIR / "development_sweep_summary.csv", index=False)

BEST_LAMBDA = {}
for condition in CONTRASTIVE_CONDITIONS:
    sub = development_summary[development_summary.condition == condition].sort_values(
        ["mean_probe_val_BA", "lambda_con"], ascending=[False, True]
    )
    BEST_LAMBDA[condition] = float(sub.iloc[0].lambda_con)

print("Selected lambdas from validation only:", BEST_LAMBDA)
for condition in CONTRASTIVE_CONDITIONS:
    if BEST_LAMBDA[condition] == 0.0:
        print(
            f"{condition}: lambda=0 won development; "
            "contrastive regularization did not beat the control."
        )
display(development_summary)

# -----------------------------------------------------------------------------
# Final confirmation: REUSE the already-trained development checkpoints.
#
# Hyperparameter selection is finished using validation only. There is no reason
# to retrain an identical seed / lambda / architecture a second time before test.
# The selected development checkpoint is therefore frozen and loaded directly,
# and test is accessed for the first time only after BEST_LAMBDA is fixed.
#
# cls_only reuses dev_con250_lambda_0 because lambda=0 makes that run exactly
# CE-only: no contrastive items are constructed and the objective is L_cls.
# No final_*.pt training checkpoints are created by this stage.
# -----------------------------------------------------------------------------
FINAL_CONDITIONS = {
    "cls_only": {
        "source_condition": "con250",
        "lambda_con": 0.0,
    },
    "con250": {
        "source_condition": "con250",
        "lambda_con": BEST_LAMBDA["con250"],
    },
    "con500": {
        "source_condition": "con500",
        "lambda_con": BEST_LAMBDA["con500"],
    },
}


def load_selected_development_checkpoint(
    report_condition: str,
    source_condition: str,
    seed: int,
    lambda_con: float,
):
    """Load one frozen development checkpoint; never train in final evaluation."""
    ckpt = CHECKPOINT_DIR / (
        f"dev_{source_condition}_lambda_{lambda_con:g}_seed_{seed}.pt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            "Final evaluation requires the selected development checkpoint, but it "
            f"does not exist: {ckpt}. Complete/resume the development sweep first; "
            "the final stage intentionally does not retrain models."
        )

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    expected_config = run_config_dict(source_condition, seed, lambda_con)
    if payload.get("config") != expected_config:
        raise ValueError(
            f"Selected development checkpoint config mismatch: {ckpt}. "
            "Refusing to evaluate a stale/incompatible checkpoint."
        )

    print(
        f"Final {report_condition:8s} seed={seed:3d} lambda={lambda_con:g} "
        f"<- reuse {ckpt.name}"
    )
    return payload, ckpt


training_rows = []
history_rows = []
probe_rows = []
probe_sweeps = []
feature_cache = {}

raw_probe = fit_logreg_probe(
    RAW250["train"],
    RAW250["val"],
    RAW250["test"],
    seed_tag=(RUN_VARIANT, "raw250"),
)
RAW_TEST_BA = float(raw_probe["test"]["balanced_accuracy"])

for condition, final_spec in FINAL_CONDITIONS.items():
    source_condition = final_spec["source_condition"]
    lambda_con = float(final_spec["lambda_con"])

    for seed in SEEDS:
        payload, ckpt = load_selected_development_checkpoint(
            condition,
            source_condition,
            seed,
            lambda_con,
        )
        model = model_from_payload(payload)

        ztr250, ztr125 = extract_features(model, X_train)
        zva250, zva125 = extract_features(model, X_val)
        zte250, zte125 = extract_features(model, X_test)
        feature_cache[(condition, seed, "250")] = (ztr250, zva250, zte250)
        feature_cache[(condition, seed, "125")] = (ztr125, zva125, zte125)

        # Evaluate the objective associated with the checkpoint source. For the
        # cls_only report row this is con250 with lambda=0, which is exactly CE-only.
        with torch.no_grad():
            test_metrics = run_epoch(
                model.to(DEVICE),
                make_loader("test", seed, False),
                source_condition,
                lambda_con,
                None,
                epoch=payload["best_epoch"],
            )

        training_rows.append({
            "condition": condition,
            "checkpoint_source_condition": source_condition,
            "checkpoint_file": ckpt.name,
            "seed": seed,
            "lambda_con": lambda_con,
            "best_epoch": payload["best_epoch"],
            "train_global_BA": payload["train_best"]["balanced_accuracy"],
            "val_global_BA": payload["val_best"]["balanced_accuracy"],
            "test_global_accuracy": test_metrics["accuracy"],
            "test_global_BA": test_metrics["balanced_accuracy"],
            "test_global_macro_f1": test_metrics["macro_f1"],
            "test_l1_fr": test_metrics["l1_fr"],
            "test_l2_fr": test_metrics["l2_fr"],
            "test_l3_fr": test_metrics["l3_fr"],
            "test_l1_dead_fraction": test_metrics["l1_dead_fraction"],
            "test_l2_dead_fraction": test_metrics["l2_dead_fraction"],
            "test_l3_dead_fraction": test_metrics["l3_dead_fraction"],
            "train_active_item_fraction": payload["train_best"]["active_item_fraction"],
            "val_active_item_fraction": payload["val_best"]["active_item_fraction"],
            "train_valid_anchor_fraction": payload["train_best"]["valid_anchor_fraction"],
            "val_valid_anchor_fraction": payload["val_best"]["valid_anchor_fraction"],
        })
        for row in payload["history"]:
            history_rows.append({
                "condition": condition,
                "checkpoint_source_condition": source_condition,
                "seed": seed,
                "lambda_con": lambda_con,
                **row,
            })

        for readout, arrays in (
            ("SNN250", (ztr250, zva250, zte250)),
            ("SNN125", (ztr125, zva125, zte125)),
        ):
            probe = fit_logreg_probe(
                *arrays,
                seed_tag=(RUN_VARIANT, "final", readout, condition, seed),
            )
            probe_rows.append({
                "representation": readout,
                "condition": condition,
                "seed": seed,
                "feature_dim": int(np.prod(arrays[0].shape[1:])),
                "selected_C": probe["selected_C"],
                "val_BA": probe["val"]["balanced_accuracy"],
                "test_accuracy": probe["test"]["accuracy"],
                "test_BA": probe["test"]["balanced_accuracy"],
                "test_macro_f1": probe["test"]["macro_f1"],
                "delta_test_BA_vs_Raw250": (
                    probe["test"]["balanced_accuracy"] - RAW_TEST_BA
                ),
            })
            for sweep_row in probe["sweep"]:
                probe_sweeps.append({
                    "representation": readout,
                    "condition": condition,
                    "seed": seed,
                    **sweep_row,
                })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

training_df = pd.DataFrame(training_rows)
history_df = pd.DataFrame(history_rows)
probe_df = pd.DataFrame(probe_rows)
probe_sweep_df = pd.DataFrame(probe_sweeps)
probe_summary = (
    probe_df.groupby(["representation", "condition"], as_index=False)
    .agg(
        n_runs=("seed", "count"),
        mean_test_BA=("test_BA", "mean"),
        sd_test_BA=("test_BA", "std"),
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_macro_f1=("test_macro_f1", "mean"),
        mean_delta_test_BA_vs_Raw250=("delta_test_BA_vs_Raw250", "mean"),
    )
)

training_df.to_csv(RESULTS_DIR / "final_training_results.csv", index=False)
history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
probe_df.to_csv(RESULTS_DIR / "snn_probe_results.csv", index=False)
probe_summary.to_csv(RESULTS_DIR / "snn_probe_summary.csv", index=False)
probe_sweep_df.to_csv(RESULTS_DIR / "snn_probe_c_sweeps.csv", index=False)

raw_summary = pd.DataFrame([{
    "representation": "Raw250",
    "selected_C": raw_probe["selected_C"],
    "val_BA": raw_probe["val"]["balanced_accuracy"],
    "test_accuracy": raw_probe["test"]["accuracy"],
    "test_BA": raw_probe["test"]["balanced_accuracy"],
    "test_macro_f1": raw_probe["test"]["macro_f1"],
}])
raw_summary.to_csv(RESULTS_DIR / "raw250_baseline.csv", index=False)

display(training_df)
display(probe_summary)
display(raw_summary)
