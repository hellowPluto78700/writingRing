# -----------------------------------------------------------------------------
# Cross-user phase-aware retrieval / kNN diagnostic.
# Query and candidate pool are always from different users and the same phase.
# -----------------------------------------------------------------------------
def build_retrieval_items(z250, labels, users, valid_lengths, mode="250"):
    if mode == "250":
        rep = np.log1p(z250)
        meta = local_meta_numpy(valid_lengths, labels, users, "250")
    elif mode == "500":
        rep = np.log1p(np.concatenate([z250[:, :-1], z250[:, 1:]], axis=-1))
        meta = local_meta_numpy(valid_lengths, labels, users, "500")
    else:
        raise ValueError(mode)
    full = meta["full"]
    rep = rep[full]
    labels_flat = meta["label"][full]
    users_flat = meta["user"][full]
    phases_flat = meta["phase"][full]
    norms = np.linalg.norm(rep, axis=1, keepdims=True)
    rep = rep / np.maximum(norms, EPS)
    return rep.astype(np.float32), labels_flat, users_flat, phases_flat


def cross_user_retrieval_metrics(rep, labels, users, phases, ks=(1,5)):
    same_label_hits = {k: [] for k in ks}
    knn_preds, knn_true = [], []
    for i in range(len(rep)):
        candidate = (users != users[i]) & (phases == phases[i])
        if candidate.sum() == 0:
            continue
        idx = np.flatnonzero(candidate)
        sims = rep[idx] @ rep[i]
        order = idx[np.argsort(-sims)]
        for k in ks:
            top = order[: min(k, len(order))]
            same_label_hits[k].append(float(np.mean(labels[top] == labels[i])))
        k5 = order[: min(5, len(order))]
        vals, counts = np.unique(labels[k5], return_counts=True)
        pred = vals[np.argmax(counts)]
        knn_preds.append(int(pred)); knn_true.append(int(labels[i]))
    out = {f"SameLabel@{k}": float(np.mean(same_label_hits[k])) for k in ks}
    out["n_queries"] = len(knn_true)
    out["knn5_BA"] = float(balanced_accuracy_score(knn_true, knn_preds)) if knn_true else np.nan
    return out


retrieval_rows = []
for condition in FINAL_CONDITIONS:
    for seed in SEEDS:
        ztr250, _, _ = feature_cache[(condition, seed, "250")]
        for mode in ("250", "500"):
            rep, lab, usr, ph = build_retrieval_items(ztr250, y_train, user_train, valid_train, mode)
            metrics = cross_user_retrieval_metrics(rep, lab, usr, ph)
            retrieval_rows.append({"condition": condition, "seed": seed, "mode": mode, **metrics})

retrieval_df = pd.DataFrame(retrieval_rows)
retrieval_summary = retrieval_df.groupby(["condition","mode"], as_index=False).agg(
    mean_SameLabel_at_1=("SameLabel@1","mean"),
    mean_SameLabel_at_5=("SameLabel@5","mean"),
    mean_knn5_BA=("knn5_BA","mean"),
    sd_knn5_BA=("knn5_BA","std"),
)
retrieval_df.to_csv(RESULTS_DIR / "cross_user_retrieval.csv", index=False)
retrieval_summary.to_csv(RESULTS_DIR / "cross_user_retrieval_summary.csv", index=False)

# -----------------------------------------------------------------------------
# Raw + SNN diagnostic only. Not the deployment architecture.
# -----------------------------------------------------------------------------
fusion_rows = []
for condition in FINAL_CONDITIONS:
    for seed in SEEDS:
        ztr250, zva250, zte250 = feature_cache[(condition, seed, "250")]
        ftr = np.concatenate([RAW250["train"].reshape(len(X_train), -1), ztr250.reshape(len(X_train), -1)], axis=1)
        fva = np.concatenate([RAW250["val"].reshape(len(X_val), -1), zva250.reshape(len(X_val), -1)], axis=1)
        fte = np.concatenate([RAW250["test"].reshape(len(X_test), -1), zte250.reshape(len(X_test), -1)], axis=1)
        fusion = fit_logreg_probe(ftr, fva, fte, seed_tag=("fusion", condition, seed))
        snn_row = probe_df[(probe_df.representation == "SNN250") & (probe_df.condition == condition) & (probe_df.seed == seed)].iloc[0]
        fusion_rows.append({
            "condition": condition, "seed": seed,
            "snn_only_test_BA": float(snn_row.test_BA),
            "raw250_test_BA": RAW_TEST_BA,
            "raw_plus_snn_test_BA": fusion["test"]["balanced_accuracy"],
            "fusion_delta_vs_raw": fusion["test"]["balanced_accuracy"] - RAW_TEST_BA,
            "fusion_delta_vs_snn": fusion["test"]["balanced_accuracy"] - float(snn_row.test_BA),
            "selected_C": fusion["selected_C"],
        })

fusion_df = pd.DataFrame(fusion_rows)
fusion_summary = fusion_df.groupby("condition", as_index=False).agg(
    mean_snn_only_test_BA=("snn_only_test_BA","mean"),
    mean_raw_plus_snn_test_BA=("raw_plus_snn_test_BA","mean"),
    mean_fusion_delta_vs_raw=("fusion_delta_vs_raw","mean"),
    sd_fusion_delta_vs_raw=("fusion_delta_vs_raw","std"),
)
fusion_df.to_csv(RESULTS_DIR / "raw_plus_snn_diagnostic.csv", index=False)
fusion_summary.to_csv(RESULTS_DIR / "raw_plus_snn_diagnostic_summary.csv", index=False)

# Compact summary and plots.
summary_payload = {
    "best_lambda_con250": BEST_LAMBDA["con250"],
    "best_lambda_con500": BEST_LAMBDA["con500"],
    "raw250_test_BA": RAW_TEST_BA,
}
with open(RESULTS_DIR / "selected_hyperparameters.json", "w") as f:
    json.dump(summary_payload, f, indent=2)

plt.figure(figsize=(10,5))
plot_df = probe_summary[probe_summary.representation == "SNN250"].copy()
x = np.arange(len(plot_df))
plt.bar(x, plot_df.mean_test_BA, yerr=plot_df.sd_test_BA.fillna(0), capsize=4)
plt.axhline(RAW_TEST_BA, linestyle="--", linewidth=1.5, label="Raw250")
plt.xticks(x, plot_df.condition)
plt.ylabel("Test balanced accuracy")
plt.title("Experiment 1.3.9 - SNN250 local feature quality")
plt.legend(); plt.grid(axis="y", alpha=0.25); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
for (condition, seed), group in history_df.groupby(["condition","seed"]):
    plt.plot(group.epoch, group.val_balanced_accuracy, label=f"{condition}/seed{seed}", alpha=0.85)
plt.xlabel("Epoch"); plt.ylabel("Validation balanced accuracy"); plt.title("Validation BA vs epoch")
plt.grid(alpha=0.25); plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
for (condition, seed), group in history_df.groupby(["condition","seed"]):
    plt.plot(group.epoch, group.train_valid_anchor_fraction, label=f"{condition}/seed{seed}", alpha=0.85)
plt.xlabel("Epoch"); plt.ylabel("Valid contrastive anchor fraction"); plt.title("Contrastive anchor coverage")
plt.grid(alpha=0.25); plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

display(development_summary)
display(probe_summary)
display(retrieval_summary)
display(fusion_summary)
print("Selected lambdas:", BEST_LAMBDA)
print("Raw250 test BA:", RAW_TEST_BA)
print("Saved outputs to:", RESULTS_DIR)
