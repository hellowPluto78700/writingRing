# -----------------------------------------------------------------------------
# Development sweep: choose auxiliary weights with validation SNN-only probe BA.
# Test is not touched in this section.
# -----------------------------------------------------------------------------
dev_rows = []

# Step 1: choose lambda_count with condition B and development seed 11.
for lambda_count in LAMBDA_COUNT_GRID:
    ckpt = DEV_DIR / f"dev_cls_count_lc_{lambda_count:g}_seed_{DEV_SEED}.pt"
    payload = train_run(
        condition="cls_count",
        seed=DEV_SEED,
        lambda_count=lambda_count,
        lambda_temp=0.0,
        checkpoint_path=ckpt,
    )
    model = model_from_payload(payload)
    z_train = extract_local_features(model, X_train)
    z_val = extract_local_features(model, X_val)
    probe = fit_logreg_probe(
        z_train,
        z_val,
        z_test=None,
        seed_tag=("dev_count", DEV_SEED, lambda_count),
    )
    dev_rows.append({
        "stage": "count",
        "lambda_count": lambda_count,
        "lambda_temp": 0.0,
        "best_epoch": payload["best_epoch"],
        "global_val_BA": payload["val_best"]["balanced_accuracy"],
        "probe_selected_C": probe["selected_C"],
        "probe_val_BA": probe["val"]["balanced_accuracy"],
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

count_dev = pd.DataFrame(dev_rows)
BEST_LAMBDA_COUNT = float(
    count_dev.sort_values(
        ["probe_val_BA", "lambda_count"],
        ascending=[False, True],
    ).iloc[0].lambda_count
)

print("Selected lambda_count =", BEST_LAMBDA_COUNT)
display(count_dev)


# Step 2: fix lambda_count and choose lambda_temp with condition C.
for lambda_temp in LAMBDA_TEMP_GRID:
    ckpt = DEV_DIR / (
        f"dev_cls_count_temp_lc_{BEST_LAMBDA_COUNT:g}_lt_{lambda_temp:g}_seed_{DEV_SEED}.pt"
    )
    payload = train_run(
        condition="cls_count_temp",
        seed=DEV_SEED,
        lambda_count=BEST_LAMBDA_COUNT,
        lambda_temp=lambda_temp,
        checkpoint_path=ckpt,
    )
    model = model_from_payload(payload)
    z_train = extract_local_features(model, X_train)
    z_val = extract_local_features(model, X_val)
    probe = fit_logreg_probe(
        z_train,
        z_val,
        z_test=None,
        seed_tag=("dev_temp", DEV_SEED, BEST_LAMBDA_COUNT, lambda_temp),
    )
    dev_rows.append({
        "stage": "temp",
        "lambda_count": BEST_LAMBDA_COUNT,
        "lambda_temp": lambda_temp,
        "best_epoch": payload["best_epoch"],
        "global_val_BA": payload["val_best"]["balanced_accuracy"],
        "probe_selected_C": probe["selected_C"],
        "probe_val_BA": probe["val"]["balanced_accuracy"],
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

dev_df = pd.DataFrame(dev_rows)
temp_dev = dev_df[dev_df.stage == "temp"].copy()
BEST_LAMBDA_TEMP = float(
    temp_dev.sort_values(
        ["probe_val_BA", "lambda_temp"],
        ascending=[False, True],
    ).iloc[0].lambda_temp
)

dev_df.to_csv(RESULTS_DIR / "development_sweep.csv", index=False)

print("Selected lambda_count =", BEST_LAMBDA_COUNT)
print("Selected lambda_temp  =", BEST_LAMBDA_TEMP)
display(dev_df)


# -----------------------------------------------------------------------------
# Hyperparameters are now locked. Test evaluation may begin.
# -----------------------------------------------------------------------------
FINAL_CONDITIONS = {
    "cls_only": (0.0, 0.0),
    "cls_count": (BEST_LAMBDA_COUNT, 0.0),
    "cls_count_temp": (BEST_LAMBDA_COUNT, BEST_LAMBDA_TEMP),
}

raw_probe = fit_logreg_probe(
    RAW250["train"],
    RAW250["val"],
    RAW250["test"],
    seed_tag="raw250_final",
)
RAW_TEST_BA = raw_probe["test"]["balanced_accuracy"]
print("Locked Raw250 selected C:", raw_probe["selected_C"])
print("Locked Raw250 test BA:", RAW_TEST_BA)


# -----------------------------------------------------------------------------
# Final 3 conditions x 3 seeds.
# -----------------------------------------------------------------------------
final_payloads = {}
training_rows = []
history_rows = []

for condition, (lambda_count, lambda_temp) in FINAL_CONDITIONS.items():
    for seed in SEEDS:
        print("=" * 110)
        print(
            "FINAL:",
            condition,
            "seed=", seed,
            "lambda_count=", lambda_count,
            "lambda_temp=", lambda_temp,
        )

        ckpt = FINAL_DIR / f"{condition}_seed_{seed}.pt"
        payload = train_run(
            condition=condition,
            seed=seed,
            lambda_count=lambda_count,
            lambda_temp=lambda_temp,
            checkpoint_path=ckpt,
        )
        model = model_from_payload(payload).to(DEVICE)

        with torch.no_grad():
            test_metrics = run_epoch(
                model,
                make_loader("test", seed, shuffle=False),
                lambda_count=lambda_count,
                lambda_temp=lambda_temp,
                optimizer=None,
            )

        final_payloads[(condition, seed)] = payload
        training_rows.append({
            "condition": condition,
            "seed": seed,
            "lambda_count": lambda_count,
            "lambda_temp": lambda_temp,
            "best_epoch": payload["best_epoch"],
            "train_global_BA": payload["train_best"]["balanced_accuracy"],
            "val_global_BA": payload["val_best"]["balanced_accuracy"],
            "test_global_accuracy": test_metrics["accuracy"],
            "test_global_BA": test_metrics["balanced_accuracy"],
            "test_global_macro_f1": test_metrics["macro_f1"],
            "test_loss_count": test_metrics["loss_count"],
            "test_loss_temp": test_metrics["loss_temp"],
            "test_l1_fr": test_metrics["l1_fr"],
            "test_l2_fr": test_metrics["l2_fr"],
            "test_l3_fr": test_metrics["l3_fr"],
            "test_l1_dead_fraction": test_metrics["l1_dead_fraction"],
            "test_l2_dead_fraction": test_metrics["l2_dead_fraction"],
            "test_l3_dead_fraction": test_metrics["l3_dead_fraction"],
        })

        for history_row in payload["history"]:
            history_rows.append({
                "condition": condition,
                "seed": seed,
                "lambda_count": lambda_count,
                "lambda_temp": lambda_temp,
                **history_row,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

training_df = pd.DataFrame(training_rows)
history_df = pd.DataFrame(history_rows)
training_df.to_csv(RESULTS_DIR / "final_training_results.csv", index=False)
history_df.to_csv(RESULTS_DIR / "training_histories.csv", index=False)

display(training_df)


# -----------------------------------------------------------------------------
# Primary evaluation: freeze SNN, ignore all training heads, and fit a fresh
# SNN-only Logistic Regression probe on the 16 x 64 = 1024-D local-feature map.
# -----------------------------------------------------------------------------
probe_rows = []
probe_c_rows = []
feature_cache = {}

for condition in FINAL_CONDITIONS:
    for seed in SEEDS:
        payload = final_payloads[(condition, seed)]
        model = model_from_payload(payload)

        z_train = extract_local_features(model, X_train)
        z_val = extract_local_features(model, X_val)
        z_test = extract_local_features(model, X_test)
        feature_cache[(condition, seed)] = (z_train, z_val, z_test)

        probe = fit_logreg_probe(
            z_train,
            z_val,
            z_test,
            seed_tag=("final_probe", condition, seed),
        )

        probe_rows.append({
            "representation": "SNN250",
            "condition": condition,
            "seed": seed,
            "feature_dim": probe["feature_dim"],
            "selected_C": probe["selected_C"],
            "val_BA": probe["val"]["balanced_accuracy"],
            "test_accuracy": probe["test"]["accuracy"],
            "test_BA": probe["test"]["balanced_accuracy"],
            "test_macro_f1": probe["test"]["macro_f1"],
            "delta_test_BA_vs_Raw250": probe["test"]["balanced_accuracy"] - RAW_TEST_BA,
        })
        for row in probe["c_sweep"].to_dict("records"):
            probe_c_rows.append({
                "condition": condition,
                "seed": seed,
                **row,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

probe_df = pd.DataFrame(probe_rows)
probe_c_df = pd.DataFrame(probe_c_rows)
probe_summary = (
    probe_df.groupby("condition", as_index=False)
    .agg(
        n_runs=("seed", "count"),
        mean_test_BA=("test_BA", "mean"),
        sd_test_BA=("test_BA", "std"),
        mean_test_accuracy=("test_accuracy", "mean"),
        mean_test_macro_f1=("test_macro_f1", "mean"),
        mean_delta_test_BA_vs_Raw250=("delta_test_BA_vs_Raw250", "mean"),
        sd_delta_test_BA_vs_Raw250=("delta_test_BA_vs_Raw250", "std"),
    )
)

raw_row = pd.DataFrame([{
    "condition": "Raw250 baseline",
    "n_runs": 1,
    "mean_test_BA": raw_probe["test"]["balanced_accuracy"],
    "sd_test_BA": np.nan,
    "mean_test_accuracy": raw_probe["test"]["accuracy"],
    "mean_test_macro_f1": raw_probe["test"]["macro_f1"],
    "mean_delta_test_BA_vs_Raw250": 0.0,
    "sd_delta_test_BA_vs_Raw250": np.nan,
}])
probe_summary_with_raw = pd.concat([raw_row, probe_summary], ignore_index=True)

probe_df.to_csv(RESULTS_DIR / "snn_only_probe_results.csv", index=False)
probe_c_df.to_csv(RESULTS_DIR / "snn_only_probe_c_sweeps.csv", index=False)
probe_summary_with_raw.to_csv(RESULTS_DIR / "snn_only_probe_summary.csv", index=False)

display(probe_df)
display(probe_summary_with_raw)
