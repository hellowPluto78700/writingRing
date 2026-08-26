# -----------------------------------------------------------------------------
# Fresh post-hoc semantic decodability probe.
# This deliberately does NOT reuse the training semantic decoder weights.
# -----------------------------------------------------------------------------
def semantic_linear_probe(z_train, z_val, z_test):
    a = z_train.reshape(-1, z_train.shape[-1])
    b = z_val.reshape(-1, z_val.shape[-1])
    c = z_test.reshape(-1, z_test.shape[-1])

    ytr = count_norm_train.reshape(-1, EVENT_CHANNEL_COUNT)
    yva = count_norm_val.reshape(-1, EVENT_CHANNEL_COUNT)
    yte = count_norm_test.reshape(-1, EVENT_CHANNEL_COUNT)
    wtr = targets_train["count_weight"].reshape(-1)
    wva = targets_val["count_weight"].reshape(-1)
    wte = targets_test["count_weight"].reshape(-1)

    train_keep = wtr > 0
    scaler = StandardScaler()
    a_fit = scaler.fit_transform(a[train_keep])

    reg = LinearRegression()
    reg.fit(a_fit, ytr[train_keep], sample_weight=wtr[train_keep])

    def score(xmat, ymat, weights):
        keep = weights > 0
        pred = reg.predict(scaler.transform(xmat[keep]))
        mae = mean_absolute_error(ymat[keep], pred, sample_weight=weights[keep])
        r2 = r2_score(
            ymat[keep],
            pred,
            sample_weight=weights[keep],
            multioutput="uniform_average",
        )
        return float(mae), float(r2)

    val_mae, val_r2 = score(b, yva, wva)
    test_mae, test_r2 = score(c, yte, wte)
    return {
        "semantic_val_MAE": val_mae,
        "semantic_val_R2": val_r2,
        "semantic_test_MAE": test_mae,
        "semantic_test_R2": test_r2,
    }


# -----------------------------------------------------------------------------
# Fresh post-hoc temporal decodability probe.
# Each raw event channel has its own regression because the target is only
# defined for fully valid bins where that channel is active.
# -----------------------------------------------------------------------------
def temporal_linear_probe(z_train, z_val, z_test):
    zflat = {
        "train": z_train.reshape(-1, z_train.shape[-1]),
        "val": z_val.reshape(-1, z_val.shape[-1]),
        "test": z_test.reshape(-1, z_test.shape[-1]),
    }
    y = {
        "train": targets_train["temporal"].reshape(-1, EVENT_CHANNEL_COUNT),
        "val": targets_val["temporal"].reshape(-1, EVENT_CHANNEL_COUNT),
        "test": targets_test["temporal"].reshape(-1, EVENT_CHANNEL_COUNT),
    }
    mask = {
        "train": targets_train["temporal_mask"].reshape(-1, EVENT_CHANNEL_COUNT).astype(bool),
        "val": targets_val["temporal_mask"].reshape(-1, EVENT_CHANNEL_COUNT).astype(bool),
        "test": targets_test["temporal_mask"].reshape(-1, EVENT_CHANNEL_COUNT).astype(bool),
    }

    train_any = mask["train"].any(axis=1)
    if not train_any.any():
        raise RuntimeError("No active fully-valid training bins for temporal probe")

    scaler = StandardScaler()
    scaler.fit(zflat["train"][train_any])
    zs = {name: scaler.transform(values) for name, values in zflat.items()}

    val_abs_errors = []
    test_abs_errors = []
    val_r2s = []
    test_r2s = []
    fitted_channels = 0

    for channel in range(EVENT_CHANNEL_COUNT):
        tr = mask["train"][:, channel]
        va = mask["val"][:, channel]
        te = mask["test"][:, channel]

        if tr.sum() < 2 or va.sum() == 0 or te.sum() == 0:
            continue

        reg = LinearRegression()
        reg.fit(zs["train"][tr], y["train"][tr, channel])

        pred_val = reg.predict(zs["val"][va])
        pred_test = reg.predict(zs["test"][te])

        val_abs_errors.extend(np.abs(pred_val - y["val"][va, channel]).tolist())
        test_abs_errors.extend(np.abs(pred_test - y["test"][te, channel]).tolist())

        if va.sum() >= 2:
            val_r2s.append(r2_score(y["val"][va, channel], pred_val))
        if te.sum() >= 2:
            test_r2s.append(r2_score(y["test"][te, channel], pred_test))
        fitted_channels += 1

    return {
        "temporal_channels_fit": fitted_channels,
        "temporal_val_MAE": float(np.mean(val_abs_errors)),
        "temporal_test_MAE": float(np.mean(test_abs_errors)),
        "temporal_val_mean_channel_R2": float(np.mean(val_r2s)),
        "temporal_test_mean_channel_R2": float(np.mean(test_r2s)),
    }


aux_rows = []
for condition in FINAL_CONDITIONS:
    for seed in SEEDS:
        z_train, z_val, z_test = feature_cache[(condition, seed)]
        aux_rows.append({
            "condition": condition,
            "seed": seed,
            **semantic_linear_probe(z_train, z_val, z_test),
            **temporal_linear_probe(z_train, z_val, z_test),
        })

aux_probe_df = pd.DataFrame(aux_rows)
aux_probe_summary = (
    aux_probe_df.groupby("condition", as_index=False)
    .agg(
        semantic_test_MAE_mean=("semantic_test_MAE", "mean"),
        semantic_test_R2_mean=("semantic_test_R2", "mean"),
        temporal_test_MAE_mean=("temporal_test_MAE", "mean"),
        temporal_test_R2_mean=("temporal_test_mean_channel_R2", "mean"),
    )
)
aux_probe_df.to_csv(RESULTS_DIR / "fresh_auxiliary_linear_probes.csv", index=False)
aux_probe_summary.to_csv(RESULTS_DIR / "fresh_auxiliary_probe_summary.csv", index=False)

display(aux_probe_df)
display(aux_probe_summary)


# -----------------------------------------------------------------------------
# Raw + SNN is a diagnostic only. It is NOT the target deployment architecture.
# If SNN-only improves while fusion remains above Raw, the new representation
# has absorbed more robust count information without losing complementarity.
# -----------------------------------------------------------------------------
fusion_rows = []
for condition in FINAL_CONDITIONS:
    for seed in SEEDS:
        z_train, z_val, z_test = feature_cache[(condition, seed)]

        f_train = np.concatenate([
            RAW250["train"].reshape(len(X_train), -1),
            z_train.reshape(len(X_train), -1),
        ], axis=1)
        f_val = np.concatenate([
            RAW250["val"].reshape(len(X_val), -1),
            z_val.reshape(len(X_val), -1),
        ], axis=1)
        f_test = np.concatenate([
            RAW250["test"].reshape(len(X_test), -1),
            z_test.reshape(len(X_test), -1),
        ], axis=1)

        fusion = fit_logreg_probe(
            f_train,
            f_val,
            f_test,
            seed_tag=("fusion_diagnostic", condition, seed),
        )
        snn_test_ba = float(
            probe_df[
                (probe_df.condition == condition) & (probe_df.seed == seed)
            ].iloc[0].test_BA
        )

        fusion_rows.append({
            "condition": condition,
            "seed": seed,
            "snn_only_test_BA": snn_test_ba,
            "raw250_test_BA": RAW_TEST_BA,
            "raw_plus_snn_test_BA": fusion["test"]["balanced_accuracy"],
            "fusion_delta_vs_raw": fusion["test"]["balanced_accuracy"] - RAW_TEST_BA,
            "fusion_delta_vs_snn": fusion["test"]["balanced_accuracy"] - snn_test_ba,
            "selected_C": fusion["selected_C"],
        })

fusion_df = pd.DataFrame(fusion_rows)
fusion_summary = (
    fusion_df.groupby("condition", as_index=False)
    .agg(
        mean_snn_only_test_BA=("snn_only_test_BA", "mean"),
        mean_raw_plus_snn_test_BA=("raw_plus_snn_test_BA", "mean"),
        mean_fusion_delta_vs_raw=("fusion_delta_vs_raw", "mean"),
        sd_fusion_delta_vs_raw=("fusion_delta_vs_raw", "std"),
    )
)
fusion_df.to_csv(RESULTS_DIR / "raw_plus_snn_diagnostic.csv", index=False)
fusion_summary.to_csv(RESULTS_DIR / "raw_plus_snn_diagnostic_summary.csv", index=False)

display(fusion_df)
display(fusion_summary)


# -----------------------------------------------------------------------------
# Visualization.
# -----------------------------------------------------------------------------
plot_df = probe_summary_with_raw.copy()
plt.figure(figsize=(9, 5))
x = np.arange(len(plot_df))
plt.bar(
    x,
    plot_df.mean_test_BA,
    yerr=plot_df.sd_test_BA.fillna(0),
    capsize=4,
)
plt.xticks(
    x,
    ["Raw250", "A: cls", "B: cls+count", "C: cls+count+temp"],
    rotation=12,
)
plt.ylabel("Test balanced accuracy")
plt.title("Experiment 1.3.8 - SNN-only local feature quality")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for (condition, seed), group in history_df.groupby(["condition", "seed"]):
    plt.plot(
        group.epoch,
        group.val_balanced_accuracy,
        label=f"{condition} / seed {seed}",
        alpha=0.85,
    )
plt.xlabel("Epoch")
plt.ylabel("Validation balanced accuracy")
plt.title("Final training - validation BA vs epoch")
plt.grid(alpha=0.25)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for (condition, seed), group in history_df.groupby(["condition", "seed"]):
    plt.plot(
        group.epoch,
        group.train_l3_fr,
        label=f"{condition} / seed {seed}",
        alpha=0.85,
    )
plt.xlabel("Epoch")
plt.ylabel("Train L3 firing rate")
plt.title("Final training - L3 firing regime")
plt.grid(alpha=0.25)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

print("Selected lambda_count:", BEST_LAMBDA_COUNT)
print("Selected lambda_temp:", BEST_LAMBDA_TEMP)
print("Raw250 test BA:", RAW_TEST_BA)
print("Saved outputs to:", RESULTS_DIR)
display(probe_summary_with_raw)
display(aux_probe_summary)
display(fusion_summary)
