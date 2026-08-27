BETA = float(math.exp(-(1000.0 / SAMPLING_RATE_HZ) / TAU_MEM_MS))


class PhaseAwareContrastiveSNN(nn.Module):
    def __init__(self):
        super().__init__()
        h1, h2, d = SNN_LAYER_WIDTHS
        grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        self.fc1 = nn.Linear(EVENT_CHANNEL_COUNT, h1, bias=False)
        self.lif1 = snn.Synaptic(alpha=alpha_vector(h1, SNN_LAYER_SHIFTS[0]), beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism=RESET_MECHANISM)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.lif2 = snn.Synaptic(alpha=alpha_vector(h2, SNN_LAYER_SHIFTS[1]), beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism=RESET_MECHANISM)
        self.fc3 = nn.Linear(h2, d, bias=False)
        self.lif3 = snn.Synaptic(alpha=alpha_vector(d, SNN_LAYER_SHIFTS[2]), beta=BETA, threshold=THRESHOLD, spike_grad=grad, reset_mechanism=RESET_MECHANISM)
        self.classifier = nn.Linear(N_BINS * d, N_CLASSES, bias=True)

    def forward(self, x, return_fine=False):
        batch, timesteps, _ = x.shape
        h1, h2, d = SNN_LAYER_WIDTHS
        syn1 = torch.zeros(batch, h1, device=x.device, dtype=x.dtype); mem1 = torch.zeros_like(syn1)
        syn2 = torch.zeros(batch, h2, device=x.device, dtype=x.dtype); mem2 = torch.zeros_like(syn2)
        syn3 = torch.zeros(batch, d, device=x.device, dtype=x.dtype); mem3 = torch.zeros_like(syn3)

        coarse_acc = torch.zeros(batch, d, device=x.device, dtype=x.dtype)
        fine_acc = torch.zeros(batch, d, device=x.device, dtype=x.dtype)
        coarse_features, fine_features = [], []
        spike_sum_l1 = torch.zeros(h1, device=x.device, dtype=x.dtype)
        spike_sum_l2 = torch.zeros(h2, device=x.device, dtype=x.dtype)
        spike_sum_l3 = torch.zeros(d, device=x.device, dtype=x.dtype)

        for t in range(timesteps):
            s1, syn1, mem1 = self.lif1(self.fc1(x[:, t]), syn1, mem1)
            s2, syn2, mem2 = self.lif2(self.fc2(s1), syn2, mem2)
            s3, syn3, mem3 = self.lif3(self.fc3(s2), syn3, mem3)
            spike_sum_l1 += s1.sum(dim=0); spike_sum_l2 += s2.sum(dim=0); spike_sum_l3 += s3.sum(dim=0)
            coarse_acc = coarse_acc + s3
            if return_fine:
                fine_acc = fine_acc + s3
                if (t + 1) % FINE_BIN_SAMPLES == 0:
                    fine_features.append(fine_acc)
                    fine_acc = torch.zeros_like(fine_acc)
            if (t + 1) % BIN_SAMPLES == 0:
                coarse_features.append(coarse_acc)
                coarse_acc = torch.zeros_like(coarse_acc)

        z250 = torch.stack(coarse_features, dim=1)
        if z250.shape[1] != N_BINS:
            raise RuntimeError((z250.shape, N_BINS))
        logits = self.classifier(z250.flatten(1))
        out = {
            "logits": logits,
            "z250": z250,
            "spike_sum_l1": spike_sum_l1,
            "spike_sum_l2": spike_sum_l2,
            "spike_sum_l3": spike_sum_l3,
        }
        if return_fine:
            out["z125"] = torch.stack(fine_features, dim=1)
        return out


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def make_loader(split: str, seed: int, shuffle: bool):
    arrays = {
        "train": (X_train, y_train, user_train, valid_train),
        "val": (X_val, y_val, user_val, valid_val),
        "test": (X_test, y_test, user_test, valid_test),
    }[split]
    ds = TensorDataset(*(torch.from_numpy(a) for a in arrays))
    g = torch.Generator(); g.manual_seed(derive_seed(seed, split, "loader"))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None, generator=g)


def make_contrastive_items(z250, labels, users, valid_lengths, mode: str):
    batch = z250.shape[0]
    if mode == "250":
        rep = torch.log1p(z250)
        n_items = N_BINS
        ends = (torch.arange(n_items, device=z250.device) + 1) * BIN_SAMPLES
        centers = torch.arange(n_items, device=z250.device, dtype=z250.dtype) * BIN_SAMPLES + BIN_SAMPLES / 2.0
    elif mode == "500":
        rep = torch.cat([z250[:, :-1], z250[:, 1:]], dim=-1)
        n_items = N_BINS - 1
        ends = (torch.arange(n_items, device=z250.device) + 2) * BIN_SAMPLES
        centers = (torch.arange(n_items, device=z250.device, dtype=z250.dtype) + 1.0) * BIN_SAMPLES
    else:
        raise ValueError(mode)

    full = ends.unsqueeze(0) <= valid_lengths.unsqueeze(1)
    progress = centers.unsqueeze(0) / valid_lengths.clamp_min(1).to(z250.dtype).unsqueeze(1)
    phase = torch.clamp((progress * N_PHASE_BUCKETS).long(), 0, N_PHASE_BUCKETS - 1)
    label_grid = labels.unsqueeze(1).expand(batch, n_items)
    user_grid = users.unsqueeze(1).expand(batch, n_items)

    rep = rep[full]
    label_flat = label_grid[full]
    user_flat = user_grid[full]
    phase_flat = phase[full]
    rep = F.normalize(rep, p=2, dim=-1, eps=EPS)
    return rep, label_flat, user_flat, phase_flat


def cross_user_phase_supcon(rep, labels, users, phases, temperature=CONTRASTIVE_TEMPERATURE):
    n = rep.shape[0]
    if n <= 1:
        return rep.sum() * 0.0, 0.0, 0
    sim = rep @ rep.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=rep.device)
    same_phase = phases[:, None] == phases[None, :]
    same_label = labels[:, None] == labels[None, :]
    diff_user = users[:, None] != users[None, :]
    positives = same_phase & same_label & diff_user & (~eye)
    negatives = same_phase & (~same_label) & (~eye)
    allowed = positives | negatives
    valid_anchor = positives.any(dim=1) & negatives.any(dim=1)
    n_valid = int(valid_anchor.sum().item())
    if n_valid == 0:
        return rep.sum() * 0.0, 0.0, 0

    logits = sim.masked_fill(~allowed, float("-inf"))
    log_denom = torch.logsumexp(logits, dim=1)
    pos_logits = sim.masked_fill(~positives, 0.0)
    pos_count = positives.sum(dim=1).clamp_min(1)
    mean_pos = pos_logits.sum(dim=1) / pos_count
    loss_per_anchor = -(mean_pos - log_denom)
    loss = loss_per_anchor[valid_anchor].mean()
    return loss, float(valid_anchor.float().mean().item()), n_valid


def run_epoch(model, loader, condition: str, lambda_con: float, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_n = 0
    totals = {"loss": 0.0, "loss_cls": 0.0, "loss_con": 0.0, "valid_anchor_fraction": 0.0}
    all_y, all_pred = [], []
    widths = SNN_LAYER_WIDTHS
    spike_sums = [torch.zeros(w, dtype=torch.float64) for w in widths]

    for xb, yb, ub, vb in loader:
        xb = xb.to(DEVICE, non_blocking=True).float(); yb = yb.to(DEVICE).long(); ub = ub.to(DEVICE).long(); vb = vb.to(DEVICE).long()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            out = model(xb)
            l_cls = F.cross_entropy(out["logits"], yb)
            if condition == "cls_only":
                l_con = out["z250"].sum() * 0.0; valid_frac = 0.0
            elif condition == "con250":
                items = make_contrastive_items(out["z250"], yb, ub, vb, "250")
                l_con, valid_frac, _ = cross_user_phase_supcon(*items)
            elif condition == "con500":
                items = make_contrastive_items(out["z250"], yb, ub, vb, "500")
                l_con, valid_frac, _ = cross_user_phase_supcon(*items)
            else:
                raise ValueError(condition)
            loss = l_cls + lambda_con * l_con
            if training:
                loss.backward()
                if GRAD_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

        n = len(xb); total_n += n
        totals["loss"] += float(loss.detach()) * n; totals["loss_cls"] += float(l_cls.detach()) * n; totals["loss_con"] += float(l_con.detach()) * n; totals["valid_anchor_fraction"] += valid_frac * n
        pred = out["logits"].argmax(dim=1)
        all_y.append(yb.detach().cpu().numpy()); all_pred.append(pred.detach().cpu().numpy())
        for i, key in enumerate(("spike_sum_l1", "spike_sum_l2", "spike_sum_l3")):
            spike_sums[i] += out[key].detach().double().cpu()

    metrics = classification_metrics(np.concatenate(all_y), np.concatenate(all_pred))
    fr, dead = [], []
    for layer_sum, width in zip(spike_sums, widths, strict=True):
        fr.append(float(layer_sum.sum() / (total_n * PADDED_LENGTH * width)))
        dead.append(float((layer_sum == 0).double().mean()))
    return {
        **{k: v / total_n for k, v in totals.items()}, **metrics,
        "l1_fr": fr[0], "l2_fr": fr[1], "l3_fr": fr[2],
        "l1_dead_fraction": dead[0], "l2_dead_fraction": dead[1], "l3_dead_fraction": dead[2],
    }


def run_config_dict(condition, seed, lambda_con):
    return {
        "experiment_id": EXPERIMENT_ID, "condition": condition, "seed": int(seed), "split_seed": SPLIT_SEED,
        "fs": float(SAMPLING_RATE_HZ), "event_channels": EVENT_CHANNEL_COUNT, "padded_length": PADDED_LENGTH,
        "bin_samples": BIN_SAMPLES, "widths": tuple(SNN_LAYER_WIDTHS), "shifts": tuple(tuple(v) for v in SNN_LAYER_SHIFTS),
        "tau_mem_ms": TAU_MEM_MS, "threshold": THRESHOLD, "temperature": CONTRASTIVE_TEMPERATURE,
        "n_phase_buckets": N_PHASE_BUCKETS, "lambda_con": float(lambda_con), "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "num_epochs": NUM_EPOCHS,
    }


def train_run(condition, seed, lambda_con, checkpoint_path: Path):
    cfg = run_config_dict(condition, seed, lambda_con)
    if RESUME_EXISTING and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("config") != cfg:
            raise ValueError(f"Stale checkpoint config mismatch: {checkpoint_path}")
        return payload

    seed_everything(seed)
    model = PhaseAwareContrastiveSNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader("train", seed, True); val_loader = make_loader("val", seed, False)
    history = []; best_state = None; best_epoch = None; best_val_ba = -np.inf; best_val_loss = np.inf

    for epoch in range(1, NUM_EPOCHS + 1):
        tr = run_epoch(model, train_loader, condition, lambda_con, opt)
        with torch.no_grad():
            va = run_epoch(model, val_loader, condition, lambda_con, None)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}})
        improved = va["balanced_accuracy"] > best_val_ba + 1e-12 or (abs(va["balanced_accuracy"] - best_val_ba) <= 1e-12 and va["loss"] < best_val_loss)
        if improved:
            best_val_ba = va["balanced_accuracy"]; best_val_loss = va["loss"]; best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 10 == 0 or epoch == NUM_EPOCHS:
            print(f"{condition:8s} seed={seed:3d} lambda={lambda_con:.3f} epoch={epoch:3d} | train BA={tr['balanced_accuracy']:.4f} val BA={va['balanced_accuracy']:.4f} | con={tr['loss_con']:.4f} anchors={tr['valid_anchor_fraction']:.3f} | L3 FR={tr['l3_fr']:.4f}")

    model.load_state_dict(best_state, strict=True); model.eval()
    with torch.no_grad():
        train_best = run_epoch(model, make_loader("train", seed, False), condition, lambda_con, None)
        val_best = run_epoch(model, make_loader("val", seed, False), condition, lambda_con, None)
    payload = {"config": cfg, "best_epoch": int(best_epoch), "best_val_ba": float(best_val_ba), "model_state_dict": best_state, "history": history, "train_best": train_best, "val_best": val_best}
    if SAVE_CHECKPOINTS:
        torch.save(payload, checkpoint_path)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return payload


def model_from_payload(payload):
    model = PhaseAwareContrastiveSNN(); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, X):
    model = model.to(DEVICE); model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=BATCH_SIZE, shuffle=False)
    z250, z125 = [], []
    for (xb,) in loader:
        out = model(xb.to(DEVICE).float(), return_fine=True)
        z250.append(out["z250"].cpu().numpy().astype(np.float32)); z125.append(out["z125"].cpu().numpy().astype(np.float32))
    return np.concatenate(z250), np.concatenate(z125)


def fit_logreg_probe(z_train, z_val, z_test=None, seed_tag="probe"):
    a = np.asarray(z_train, np.float32).reshape(len(z_train), -1); b = np.asarray(z_val, np.float32).reshape(len(z_val), -1)
    scaler = StandardScaler(); a = scaler.fit_transform(a); b = scaler.transform(b)
    best, sweep = None, []
    for c_value in LOGREG_C_GRID:
        clf = LogisticRegression(C=c_value, solver="lbfgs", max_iter=LOGREG_MAX_ITER, random_state=derive_seed(SPLIT_SEED, seed_tag, c_value))
        clf.fit(a, y_train); vm = classification_metrics(y_val, clf.predict(b)); sweep.append({"C": c_value, **vm})
        if best is None or vm["balanced_accuracy"] > best["val"]["balanced_accuracy"] + 1e-12:
            best = {"C": c_value, "model": clf, "val": vm}
    out = {"selected_C": best["C"], "val": best["val"], "sweep": sweep}
    if z_test is not None:
        c = scaler.transform(np.asarray(z_test, np.float32).reshape(len(z_test), -1)); out["test"] = classification_metrics(y_test, best["model"].predict(c))
    return out
