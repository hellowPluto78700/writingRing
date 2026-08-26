BETA = float(math.exp(-(1000.0 / SAMPLING_RATE_HZ) / TAU_MEM_MS))


class LocalFeaturePreservationSNN(nn.Module):
    def __init__(self):
        super().__init__()
        h1, h2, d = SNN_LAYER_WIDTHS
        grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)

        self.fc1 = nn.Linear(EVENT_CHANNEL_COUNT, h1, bias=False)
        self.lif1 = snn.Synaptic(
            alpha=alpha_vector(h1, SNN_LAYER_SHIFTS[0]),
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=grad,
            reset_mechanism=RESET_MECHANISM,
        )
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = snn.Synaptic(
            alpha=alpha_vector(h2, SNN_LAYER_SHIFTS[1]),
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=grad,
            reset_mechanism=RESET_MECHANISM,
        )
        self.fc3 = nn.Linear(h2, d, bias=False)
        self.lif3 = snn.Synaptic(
            alpha=alpha_vector(d, SNN_LAYER_SHIFTS[2]),
            beta=BETA,
            threshold=THRESHOLD,
            spike_grad=grad,
            reset_mechanism=RESET_MECHANISM,
        )

        # Same whole-gesture classification head used by Experiment 1.3.5.
        self.classifier = nn.Linear(N_BINS * d, N_CLASSES, bias=True)

        # Training-only linear decoders. These do not assign semantics to
        # individual SNN neurons; they constrain information to be linearly
        # decodable from the distributed 64-D local feature.
        self.semantic_decoder = nn.Linear(d, EVENT_CHANNEL_COUNT, bias=True)
        self.temporal_decoder = nn.Linear(d, EVENT_CHANNEL_COUNT, bias=True)

    def forward(self, x):
        batch, timesteps, _ = x.shape
        h1, h2, d = SNN_LAYER_WIDTHS

        syn1 = torch.zeros(batch, h1, device=x.device, dtype=x.dtype)
        mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, h2, device=x.device, dtype=x.dtype)
        mem2 = torch.zeros_like(syn2)
        syn3 = torch.zeros(batch, d, device=x.device, dtype=x.dtype)
        mem3 = torch.zeros_like(syn3)

        local_acc = torch.zeros(batch, d, device=x.device, dtype=x.dtype)
        local_features = []
        spike_sum_l1 = torch.zeros(h1, device=x.device, dtype=x.dtype)
        spike_sum_l2 = torch.zeros(h2, device=x.device, dtype=x.dtype)
        spike_sum_l3 = torch.zeros(d, device=x.device, dtype=x.dtype)

        for t in range(timesteps):
            s1, syn1, mem1 = self.lif1(self.fc1(x[:, t]), syn1, mem1)
            s2, syn2, mem2 = self.lif2(self.fc2(s1), syn2, mem2)
            s3, syn3, mem3 = self.lif3(self.fc3(s2), syn3, mem3)

            spike_sum_l1 += s1.sum(dim=0)
            spike_sum_l2 += s2.sum(dim=0)
            spike_sum_l3 += s3.sum(dim=0)

            local_acc = local_acc + s3
            if (t + 1) % BIN_SAMPLES == 0:
                local_features.append(local_acc)
                local_acc = torch.zeros_like(local_acc)

        z = torch.stack(local_features, dim=1)
        if z.shape[1] != N_BINS:
            raise RuntimeError((z.shape, N_BINS))

        logits = self.classifier(z.flatten(1))
        return {
            "logits": logits,
            "local_features": z,
            "count_pred": self.semantic_decoder(z),
            "temporal_pred": self.temporal_decoder(z),
            "spike_sum_l1": spike_sum_l1,
            "spike_sum_l2": spike_sum_l2,
            "spike_sum_l3": spike_sum_l3,
        }


SPLIT_ARRAYS = {
    "train": (
        X_train,
        y_train,
        count_norm_train,
        targets_train["count_weight"],
        targets_train["temporal"],
        targets_train["temporal_mask"],
    ),
    "val": (
        X_val,
        y_val,
        count_norm_val,
        targets_val["count_weight"],
        targets_val["temporal"],
        targets_val["temporal_mask"],
    ),
    "test": (
        X_test,
        y_test,
        count_norm_test,
        targets_test["count_weight"],
        targets_test["temporal"],
        targets_test["temporal_mask"],
    ),
}


def make_loader(split: str, seed: int, shuffle: bool):
    tensors = [torch.from_numpy(a) for a in SPLIT_ARRAYS[split]]
    ds = TensorDataset(*tensors)
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, split, "loader"))
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None,
        generator=generator,
    )


def masked_count_loss(pred, target, bin_weight):
    per_bin = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    return (per_bin * bin_weight).sum() / bin_weight.sum().clamp_min(1.0)


def masked_temporal_loss(pred, target, mask):
    per_element = F.smooth_l1_loss(pred, target, reduction="none")
    return (per_element * mask).sum() / mask.sum().clamp_min(1.0)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def run_epoch(model, loader, lambda_count: float, lambda_temp: float, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_n = 0
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_count": 0.0, "loss_temp": 0.0}
    all_y, all_pred = [], []
    widths = SNN_LAYER_WIDTHS
    spike_sums = [
        torch.zeros(widths[0], dtype=torch.float64),
        torch.zeros(widths[1], dtype=torch.float64),
        torch.zeros(widths[2], dtype=torch.float64),
    ]

    for xb, yb, ctarget, cweight, ttarget, tmask in loader:
        xb = xb.to(DEVICE, non_blocking=True).float()
        yb = yb.to(DEVICE, non_blocking=True).long()
        ctarget = ctarget.to(DEVICE, non_blocking=True).float()
        cweight = cweight.to(DEVICE, non_blocking=True).float()
        ttarget = ttarget.to(DEVICE, non_blocking=True).float()
        tmask = tmask.to(DEVICE, non_blocking=True).float()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(xb)
            l_cls = F.cross_entropy(out["logits"], yb)
            l_count = masked_count_loss(out["count_pred"], ctarget, cweight)
            l_temp = masked_temporal_loss(out["temporal_pred"], ttarget, tmask)
            loss = l_cls + lambda_count * l_count + lambda_temp * l_temp

            if training:
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        n = len(xb)
        total_n += n
        totals["loss"] += float(loss.detach()) * n
        totals["loss_cls"] += float(l_cls.detach()) * n
        totals["loss_count"] += float(l_count.detach()) * n
        totals["loss_temp"] += float(l_temp.detach()) * n

        pred = out["logits"].argmax(dim=1)
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        spike_sums[0] += out["spike_sum_l1"].detach().double().cpu()
        spike_sums[1] += out["spike_sum_l2"].detach().double().cpu()
        spike_sums[2] += out["spike_sum_l3"].detach().double().cpu()

    metrics = classification_metrics(np.concatenate(all_y), np.concatenate(all_pred))
    fr, dead = [], []
    for layer_sum, width in zip(spike_sums, widths, strict=True):
        fr.append(float(layer_sum.sum() / (total_n * PADDED_LENGTH * width)))
        dead.append(float((layer_sum == 0).double().mean()))

    return {
        **{k: v / total_n for k, v in totals.items()},
        **metrics,
        "l1_fr": fr[0],
        "l2_fr": fr[1],
        "l3_fr": fr[2],
        "l1_dead_fraction": dead[0],
        "l2_dead_fraction": dead[1],
        "l3_dead_fraction": dead[2],
    }


def run_config_dict(condition: str, seed: int, lambda_count: float, lambda_temp: float):
    return {
        "experiment_id": EXPERIMENT_ID,
        "condition": condition,
        "seed": int(seed),
        "split_seed": SPLIT_SEED,
        "fs": float(SAMPLING_RATE_HZ),
        "event_channels": EVENT_CHANNEL_COUNT,
        "padded_length": PADDED_LENGTH,
        "bin_samples": BIN_SAMPLES,
        "half_bin_samples": HALF_BIN_SAMPLES,
        "widths": tuple(SNN_LAYER_WIDTHS),
        "shifts": tuple(tuple(v) for v in SNN_LAYER_SHIFTS),
        "tau_mem_ms": TAU_MEM_MS,
        "threshold": THRESHOLD,
        "lambda_count": float(lambda_count),
        "lambda_temp": float(lambda_temp),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
    }


def train_run(condition: str, seed: int, lambda_count: float, lambda_temp: float, checkpoint_path: Path):
    expected_cfg = run_config_dict(condition, seed, lambda_count, lambda_temp)

    if RESUME_EXISTING and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("config") != expected_cfg:
            raise ValueError(f"Stale checkpoint config mismatch: {checkpoint_path}")
        print("Loaded completed checkpoint:", checkpoint_path)
        return payload

    seed_everything(seed)
    model = LocalFeaturePreservationSNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader("train", seed, shuffle=True)
    val_loader = make_loader("val", seed, shuffle=False)

    history = []
    best_state, best_epoch = None, None
    best_val_ba, best_val_loss = -np.inf, np.inf

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, lambda_count, lambda_temp, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, lambda_count, lambda_temp, None)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        val_ba, val_loss = val_metrics["balanced_accuracy"], val_metrics["loss"]
        improved = (
            val_ba > best_val_ba + 1e-12
            or (abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss)
        )
        if improved:
            best_val_ba, best_val_loss = val_ba, val_loss
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(
                f"{condition:16s} seed={seed:3d} epoch={epoch:3d} | "
                f"train BA={train_metrics['balanced_accuracy']:.4f} "
                f"val BA={val_metrics['balanced_accuracy']:.4f} | "
                f"L={train_metrics['loss']:.4f} "
                f"(cls={train_metrics['loss_cls']:.4f}, count={train_metrics['loss_count']:.4f}, "
                f"temp={train_metrics['loss_temp']:.4f}) | L3 FR={train_metrics['l3_fr']:.4f}"
            )

    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        train_best = run_epoch(model, make_loader("train", seed, False), lambda_count, lambda_temp, None)
        val_best = run_epoch(model, make_loader("val", seed, False), lambda_count, lambda_temp, None)

    payload = {
        "config": expected_cfg,
        "best_epoch": int(best_epoch),
        "best_val_ba": float(best_val_ba),
        "model_state_dict": best_state,
        "history": history,
        "train_best": train_best,
        "val_best": val_best,
        "count_target_mean": COUNT_LOG_MEAN,
        "count_target_std": COUNT_LOG_STD,
    }
    if SAVE_CHECKPOINTS:
        torch.save(payload, checkpoint_path)
        print("Saved:", checkpoint_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def model_from_payload(payload):
    model = LocalFeaturePreservationSNN()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_local_features(model, X: np.ndarray):
    model = model.to(DEVICE)
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    chunks = []
    for (xb,) in loader:
        xb = xb.to(DEVICE, non_blocking=True).float()
        chunks.append(model(xb)["local_features"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def fit_logreg_probe(z_train, z_val, z_test=None, seed_tag="probe"):
    a = np.asarray(z_train, np.float32).reshape(len(z_train), -1)
    b = np.asarray(z_val, np.float32).reshape(len(z_val), -1)
    scaler = StandardScaler()
    a = scaler.fit_transform(a)
    b = scaler.transform(b)

    best, sweep = None, []
    for c_value in LOGREG_C_GRID:
        clf = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=LOGREG_MAX_ITER,
            random_state=derive_seed(SPLIT_SEED, seed_tag, c_value),
        )
        clf.fit(a, y_train)
        val_metrics = classification_metrics(y_val, clf.predict(b))
        sweep.append({"C": c_value, **val_metrics})
        if best is None or val_metrics["balanced_accuracy"] > best["val"]["balanced_accuracy"] + 1e-12:
            best = {"C": c_value, "model": clf, "val": val_metrics}

    result = {
        "feature_dim": int(a.shape[1]),
        "selected_C": float(best["C"]),
        "val": best["val"],
        "c_sweep": pd.DataFrame(sweep),
    }
    if z_test is not None:
        cmat = scaler.transform(np.asarray(z_test, np.float32).reshape(len(z_test), -1))
        result["train"] = classification_metrics(y_train, best["model"].predict(a))
        result["test"] = classification_metrics(y_test, best["model"].predict(cmat))
    return result


RAW250 = {
    "train": targets_train["raw_count"],
    "val": targets_val["raw_count"],
    "test": targets_test["raw_count"],
}

# Development is validation-only; test remains untouched until lambdas are locked.
raw_probe_dev = fit_logreg_probe(RAW250["train"], RAW250["val"], None, "raw250_dev")
print("Raw250 validation-selected C:", raw_probe_dev["selected_C"])
print("Raw250 val BA:", raw_probe_dev["val"]["balanced_accuracy"])
print("Test remains untouched during development.")
