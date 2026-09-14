from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_0_1_general_comparison as exp01
from scripts import experiment_3_0_1_single_tau_objective_comparison as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_0_hierarchical_context_snn as exp70

EXPERIMENT_ID = "experiment_7_2_two_layer_tau_training"
PROTOCOL_VERSION = "two_layer_tau_training_v1"
FAMILY_E2E = "e2e_wc"
FAMILY_LOCAL = "local_tsce"
TRAINING_FAMILIES = (FAMILY_E2E, FAMILY_LOCAL)
TASK_ONLY = "task_only"
TASK_PLUS_REG = "task_plus_reg"
REGULARIZATION_CONDITIONS = (TASK_ONLY, TASK_PLUS_REG)
ARCHITECTURES = {
    "234x234": ((2, 3, 4), (2, 3, 4)),
    "34x234": ((3, 4), (2, 3, 4)),
    "34x34": ((3, 4), (3, 4)),
    "34x345": ((3, 4), (3, 4, 5)),
    "34x45": ((3, 4), (4, 5)),
    "4x4": ((4,), (4,)),
    "5x5": ((5,), (5,)),
}
ARCHITECTURE_ORDER = tuple(ARCHITECTURES)
TRAIN_SEEDS = (11, 23, 37)
HIDDEN_WIDTH = 128
HIDDEN_CAP = 1
OUTPUT_CAP = 1
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30
BATCH_SIZE = exp3.BATCH_SIZE
LR = exp3.LR
WEIGHT_DECAY = 0.0
WARMUP_EPOCHS = 10
CALIBRATION_BATCHES = 5
TARGET_RATE_GRAD_RATIO = 0.025
TARGET_PERSIST_GRAD_RATIO = 0.05
PERSIST_SHORT_WINDOW = 3
PERSIST_SHORT_ALLOWED = 2
PERSIST_LONG_WINDOW = 8
PERSIST_LONG_ALLOWED = 4
PERSIST_LONG_WEIGHT = 0.5
TAU_MEM_MS = exp70.TAU_MEM_MS
OUTPUT_TAU_MEM_MS = exp70.OUTPUT_TAU_MEM_MS
THRESHOLD = exp70.THRESHOLD
SURROGATE_SLOPE = exp70.SURROGATE_SLOPE
EXPECTED_FS = exp70.EXPECTED_FS
EXPECTED_CHANNELS = exp70.EXPECTED_CHANNELS
EXPECTED_STEPS = exp70.EXPECTED_STEPS
EXPECTED_SPLIT_SEED = exp70.EXPECTED_SPLIT_SEED


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    training_family: str
    regularization: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__{self.training_family}__{self.regularization}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    max_epochs: int = MAX_EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp70.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(a, f, r, s)
        for a in ARCHITECTURE_ORDER
        for f in TRAINING_FAMILIES
        for r in REGULARIZATION_CONDITIONS
        for s in TRAIN_SEEDS
    ]


def validate_spec(spec: RunSpec) -> None:
    if spec.architecture not in ARCHITECTURES:
        raise ValueError(spec.architecture)
    if spec.training_family not in TRAINING_FAMILIES:
        raise ValueError(spec.training_family)
    if spec.regularization not in REGULARIZATION_CONDITIONS:
        raise ValueError(spec.regularization)
    if spec.seed not in TRAIN_SEEDS:
        raise ValueError(spec.seed)


def paired_seed(seed: int, role: str) -> int:
    return exp3.dseed(seed, "exp7_2_two_layer_tau_training", role)


def prepare_data(repo_root: Path) -> exp3.Data:
    data = exp70.prepare_data(repo_root)
    if not np.isclose(data.fs, EXPECTED_FS):
        raise ValueError("Exp7.2 requires 64 Hz")
    if int(exp3.SPLIT_SEED) != EXPECTED_SPLIT_SEED:
        raise ValueError("Exp7.2 split seed changed")
    if data.Xtr.shape[1] != EXPECTED_STEPS or data.bin_steps != 16:
        raise ValueError("Exp7.2 requires 256 padded steps and 16-step Fixed250 bins")
    return data


def _parts(data: exp3.Data):
    return {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }


def loaders(data: exp3.Data, seed: int, batch: int, shuffle: bool):
    return {
        name: exp3.loader(X, y, l, batch, shuffle if name == "train" else False,
                         paired_seed(seed, f"{name}_loader"))
        for name, (X, y, l) in _parts(data).items()
    }


def _group_counts(width: int, shifts: tuple[int, ...]) -> tuple[int, ...]:
    q, r = divmod(width, len(shifts))
    return tuple(q + (i < r) for i in range(len(shifts)))


def shift_groups(shifts: tuple[int, ...], width: int = HIDDEN_WIDTH) -> list[dict[str, int]]:
    out, start = [], 0
    for shift, count in zip(shifts, _group_counts(width, shifts), strict=True):
        out.append({"shift": int(shift), "start": start, "stop": start + int(count), "count": int(count)})
        start += int(count)
    return out


def tau_ms_from_shift(shift: int, fs: float = EXPECTED_FS) -> float:
    return exp70.tau_ms_from_shift(shift, fs)


class TwoLayerTauSNN(nn.Module):
    def __init__(self, shifts, family: str, n_classes: int, fs: float) -> None:
        super().__init__()
        self.shifts = tuple(tuple(int(v) for v in x) for x in shifts)
        self.family = family
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.hidden_linears = nn.ModuleList([
            nn.Linear(EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
        ])
        beta = math.exp(-(1000.0 / fs) / TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList([
            exp401.MacroMultiSpikeLIF(beta=beta, threshold=THRESHOLD,
                max_spikes_per_dt=HIDDEN_CAP, surrogate_slope=SURROGATE_SLOPE)
            for _ in range(2)
        ])
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, self.shifts[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, self.shifts[1]))
        if family == FAMILY_E2E:
            self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
            out_beta = math.exp(-(1000.0 / fs) / OUTPUT_TAU_MEM_MS)
            self.output_lif = exp401.MacroMultiSpikeLIF(beta=out_beta, threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP, surrogate_slope=SURROGATE_SLOPE)
            self.analog_head = None
        elif family == FAMILY_LOCAL:
            self.output_linear = None
            self.output_lif = None
            self.analog_head = nn.Linear(HIDDEN_WIDTH, n_classes, bias=True)
        else:
            raise ValueError(family)

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        B, T, C = x.shape
        if C != EXPECTED_CHANNELS:
            raise ValueError(f"Expected {EXPECTED_CHANNELS} channels")
        syn = [torch.zeros(B, HIDDEN_WIDTH, device=x.device, dtype=x.dtype) for _ in range(2)]
        mem = [torch.zeros_like(v) for v in syn]
        hidden = [[], []]
        out_mem = torch.zeros(B, self.n_classes, device=x.device, dtype=x.dtype)
        pre, out, analog = [], [], []
        for t in range(T):
            cur = x[:, t]
            for i in range(2):
                alpha = getattr(self, f"alpha_{i}")
                syn[i] = alpha * syn[i] + self.hidden_linears[i](cur)
                spk, mem[i], _ = self.hidden_lifs[i](syn[i], mem[i])
                hidden[i].append(spk)
                cur = spk
            if self.family == FAMILY_E2E:
                evidence = self.output_linear(cur)
                spk, out_mem, _ = self.output_lif(evidence, out_mem)
                pre.append(evidence)
                out.append(spk)
            else:
                analog.append(self.analog_head(cur))
        payload = {"hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden)}
        if self.family == FAMILY_E2E:
            payload["pre_output_evidence"] = torch.stack(pre, dim=1)
            payload["output_spikes"] = torch.stack(out, dim=1)
        else:
            payload["analog_logits"] = torch.stack(analog, dim=1)
        return payload


def new_model(spec: RunSpec, data: exp3.Data) -> TwoLayerTauSNN:
    return TwoLayerTauSNN(ARCHITECTURES[spec.architecture], spec.training_family,
                          len(data.labels), data.fs)


def task_loss(spec: RunSpec, tr: dict[str, Any], lengths: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if spec.training_family == FAMILY_E2E:
        return F.cross_entropy(exp50.deployment_ce_logits(tr["output_spikes"], lengths, OUTPUT_CAP), y)
    logits = tr["analog_logits"]
    valid = exp50.valid_mask(lengths, logits.shape[1])
    targets = y[:, None].expand(len(y), logits.shape[1])
    return F.cross_entropy(logits[valid], targets[valid])


def masked_rate_loss(spikes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid = exp50.valid_mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    return (spikes * valid).sum() / (valid.sum() * spikes.shape[2]).clamp_min(1.0)


def masked_persistence_loss(spikes: torch.Tensor, lengths: torch.Tensor,
                            window: int, allowed: int) -> torch.Tensor:
    windows = spikes.unfold(1, window, 1).sum(dim=-1)
    penalty = F.relu(windows - float(allowed)).square()
    starts = torch.arange(penalty.shape[1], device=lengths.device).unsqueeze(0)
    valid = (starts + window <= lengths.unsqueeze(1)).to(penalty.dtype).unsqueeze(-1)
    return (penalty * valid).sum() / (valid.sum() * penalty.shape[2]).clamp_min(1.0)


def regularization_terms(hidden, lengths):
    rates, p32, p84 = [], [], []
    for spikes in hidden:
        rates.append(masked_rate_loss(spikes, lengths))
        p32.append(masked_persistence_loss(spikes, lengths, PERSIST_SHORT_WINDOW, PERSIST_SHORT_ALLOWED))
        p84.append(masked_persistence_loss(spikes, lengths, PERSIST_LONG_WINDOW, PERSIST_LONG_ALLOWED))
    rate = torch.stack(rates).mean()
    short = torch.stack(p32).mean()
    long = torch.stack(p84).mean()
    return rate, short, long, short + PERSIST_LONG_WEIGHT * long


def warmup_scale(epoch: int) -> float:
    return min(1.0, float(epoch) / WARMUP_EPOCHS)


def _grad_norm(loss, params, retain):
    grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


def calibrate(spec: RunSpec, model: TwoLayerTauSNN, data: exp3.Data, config: Config) -> dict[str, object]:
    if spec.regularization == TASK_ONLY:
        return {"lambda_rate": 0.0, "lambda_persist": 0.0, "calibrated": False}
    ld = exp3.loader(data.Xtr, data.ytr, data.ltr, config.batch_size, True,
                     paired_seed(spec.seed, "calibration_loader"))
    params = tuple(layer.weight for layer in model.hidden_linears)
    rr, pr, rows = [], [], []
    device = torch.device(config.device)
    for i, (X, y, lengths) in enumerate(ld):
        if i >= CALIBRATION_BATCHES:
            break
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        tr = model.forward_trajectory(X)
        task = task_loss(spec, tr, lengths, y)
        rate, _, _, persist = regularization_terms(tr["hidden_spikes"], lengths)
        tg = _grad_norm(task, params, True)
        rg = _grad_norm(rate, params, True)
        pg = _grad_norm(persist, params, False)
        if tg > 1e-12:
            rr.append(rg / tg); pr.append(pg / tg)
            rows.append({"batch": i, "task_grad": tg, "rate_grad": rg, "persist_grad": pg})
    if not rr:
        raise RuntimeError(f"No usable calibration batches for {spec.key}")
    rmed, pmed = float(np.median(rr)), float(np.median(pr))
    lrate = TARGET_RATE_GRAD_RATIO / rmed if rmed > 1e-12 else 1.0
    lpersist = TARGET_PERSIST_GRAD_RATIO / pmed if pmed > 1e-12 else 1.0
    return {
        "calibrated": True, "lambda_rate": lrate, "lambda_persist": lpersist,
        "raw_rate_to_task": rmed, "raw_persist_to_task": pmed,
        "achieved_rate_grad_ratio": lrate * rmed,
        "achieved_persist_grad_ratio": lpersist * pmed,
        "parameter_scope": "both hidden input matrices", "batches": rows,
    }


def _metrics(y, pred):
    return exp70._classification_metrics(np.asarray(y), np.asarray(pred))


def evaluate_native(spec, model, loader, device):
    labels, preds, loss_sum, n_total = [], [], 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            if spec.training_family == FAMILY_E2E:
                ce_logits = exp50.deployment_ce_logits(tr["output_spikes"], lengths, OUTPUT_CAP)
                pred_logits = exp50.deployment_logits(tr["output_spikes"], lengths, OUTPUT_CAP)
            else:
                pred_logits = exp50.valid_mean(tr["analog_logits"], lengths)
                ce_logits = pred_logits
            loss = F.cross_entropy(ce_logits, y)
            labels.extend(y.cpu().tolist()); preds.extend(pred_logits.argmax(1).cpu().tolist())
            loss_sum += float(loss) * len(y); n_total += len(y)
    out = _metrics(labels, preds); out["loss"] = loss_sum / max(n_total, 1)
    return out


def _early_stop_triggered(epoch: int, best_epoch: int) -> bool:
    return epoch >= MIN_EPOCHS + PATIENCE and best_epoch > 0 and epoch - best_epoch >= PATIENCE


def _path(root: Path, kind: str, spec: RunSpec, suffix: str) -> Path:
    return root / kind / spec.training_family / f"{spec.key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plot_history(df, path, spec, best, stopped):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.epoch, df.train_ba, label="train BA")
    ax.plot(df.epoch, df.val_ba, label="val BA")
    ax2 = ax.twinx(); ax2.plot(df.epoch, df.train_task_loss, linestyle="--", label="task loss")
    ax.axvline(best, linestyle=":"); ax.set_ylim(0, 1)
    ax.set_title(f"{spec.key} (stop {stopped})"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def train_one(spec: RunSpec, data: exp3.Data, config: Config, force=False) -> Path:
    validate_spec(spec)
    ckpt = _path(config.results_dir, "checkpoints", spec, ".pt")
    if ckpt.exists() and not force:
        return ckpt
    torch.set_num_threads(config.threads)
    device = torch.device(config.device)
    exp3.seed_all(paired_seed(spec.seed, "model_init"))
    model = new_model(spec, data).to(device)
    cal = calibrate(spec, model, data, config)
    _save_json(_path(config.results_dir, "calibrations", spec, ".json"), cal)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = loaders(data, spec.seed, config.batch_size, True)["train"]
    eval_loaders = loaders(data, spec.seed, config.batch_size, False)
    best_state, best_epoch, best_ba, best_loss = None, -1, -1.0, float("inf")
    rows, stopped = [], config.max_epochs
    for epoch in range(1, config.max_epochs + 1):
        model.train(); task_sum = reg_sum = total_sum = 0.0; nb = 0
        scale = warmup_scale(epoch) if spec.regularization == TASK_PLUS_REG else 0.0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            opt.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            task = task_loss(spec, tr, lengths, y)
            rate, p32, p84, persist = regularization_terms(tr["hidden_spikes"], lengths)
            reg = scale * (float(cal["lambda_rate"]) * rate + float(cal["lambda_persist"]) * persist)
            total = task + reg
            total.backward(); opt.step()
            task_sum += float(task.detach()); reg_sum += float(reg.detach()); total_sum += float(total.detach()); nb += 1
        trm = evaluate_native(spec, model, eval_loaders["train"], device)
        vam = evaluate_native(spec, model, eval_loaders["val"], device)
        improved = vam["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(vam["balanced_accuracy"] - best_ba) <= 1e-12 and vam["loss"] < best_loss)
        if improved:
            best_ba, best_loss, best_epoch = vam["balanced_accuracy"], vam["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({"epoch": epoch, "train_ba": trm["balanced_accuracy"], "val_ba": vam["balanced_accuracy"],
                     "train_loss": trm["loss"], "val_loss": vam["loss"],
                     "train_task_loss": task_sum / max(nb, 1), "train_reg_loss": reg_sum / max(nb, 1),
                     "train_total_loss": total_sum / max(nb, 1), "reg_scale": scale})
        if _early_stop_triggered(epoch, best_epoch):
            stopped = epoch; break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
                "spec": asdict(spec), "best_epoch": best_epoch, "stopped_epoch": stopped,
                "best_val_ba": best_ba, "best_val_loss": best_loss,
                "model_state_dict": best_state, "calibration": cal}, ckpt)
    hist = pd.DataFrame(rows); hp = _path(config.results_dir, "histories", spec, ".csv")
    hp.parent.mkdir(parents=True, exist_ok=True); hist.to_csv(hp, index=False)
    _plot_history(hist, _path(config.results_dir, "training_curves", spec, ".png"), spec, best_epoch, stopped)
    return ckpt


def load_model(spec, data, config):
    ckpt = torch.load(_path(config.results_dir, "checkpoints", spec, ".pt"),
                      map_location=config.device, weights_only=False)
    model = new_model(spec, data).to(config.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True); model.eval()
    return model, ckpt


def _extract(spec, model, loader, device, source, bin_steps):
    xs, ys = [], []
    with torch.no_grad():
        for X, y, lengths in loader:
            X, lengths_d = X.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            if source == "l2_fixed250": values = tr["hidden_spikes"][-1]
            elif source == "pre_output_fixed250": values = tr["pre_output_evidence"]
            elif source == "post_output_fixed250": values = tr["output_spikes"]
            else: raise ValueError(source)
            xs.append(exp3.fixed_counts(values, lengths_d, bin_steps).flatten(1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def fit_probe(spec, model, data, config, source):
    lds = loaders(data, spec.seed, config.batch_size, False); device = torch.device(config.device)
    feat = {k: _extract(spec, model, v, device, source, data.bin_steps) for k, v in lds.items()}
    probe = exp01._fit_linear_probe(feat["train"][0], feat["train"][1], feat["val"][0], feat["val"][1],
                                    feat["test"][0], feat["test"][1],
                                    exp3.dseed(spec.seed, "exp7_2_probe", spec.key, source))
    payload = {"source": source, "feature_dim": int(probe["feature_dim"]), "probe_C": float(probe["probe_C"]),
               "metrics": {x: probe[x] for x in ("train", "val", "test")}}
    _save_json(config.results_dir / "probes" / source / f"{spec.key}.json", payload)
    return payload


def _visual_sample(data):
    med = float(np.median(data.lva)); i = int(np.argmin(np.abs(data.lva.astype(float) - med)))
    return i, int(data.lva[i])


def _raster(arr, path, valid, title, groups=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.imshow(arr.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.axvline(valid, linestyle="--", linewidth=1)
    if groups:
        for g in groups[:-1]: ax.axhline(g["stop"] - 0.5, linestyle=":", linewidth=0.8)
        title += "\n" + ", ".join(f"s{g['shift']}:{g['start']}-{g['stop']-1}" for g in groups)
    ax.set_xlabel("Timestep"); ax.set_ylabel("Neuron"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_rasters(spec, model, data, config):
    i, valid = _visual_sample(data); device = torch.device(config.device)
    sample = torch.tensor(data.Xva[i:i+1], dtype=torch.float32, device=device); sample[:, valid:] = 0
    with torch.no_grad(): tr = model.forward_trajectory(sample)
    for li, spk in enumerate(tr["hidden_spikes"], 1):
        _raster(spk[0].cpu().numpy(), config.results_dir / "rasters" / spec.training_family / f"{spec.key}__L{li}.png",
                valid, f"{spec.key} — L{li}", shift_groups(ARCHITECTURES[spec.architecture][li-1]))
    if spec.training_family == FAMILY_E2E:
        _raster(tr["output_spikes"][0].cpu().numpy(), config.results_dir / "rasters" / spec.training_family / f"{spec.key}__output.png",
                valid, f"{spec.key} — output")
    return {"split": "val", "sample_index": i, "valid_length": valid, "label": str(data.labels[int(data.yva[i])])}


def _activity_for_group(arr, lengths, start, stop, fs):
    total = positions = active = pairs = p32 = n32 = p84 = n84 = 0
    max_runs = []
    for sample, L in zip(arr, lengths, strict=True):
        x = sample[:int(L), start:stop] > 0
        total += int(x.sum()); positions += x.size; active += int(x.any(0).sum()); pairs += x.shape[1]
        run = np.zeros(x.shape[1], int); best = np.zeros_like(run)
        for row in x:
            run = np.where(row, run + 1, 0); best = np.maximum(best, run)
        max_runs.extend(best.tolist())
        if len(x) >= 3:
            c = np.lib.stride_tricks.sliding_window_view(x, 3, axis=0).sum(-1); p32 += int((c > 2).sum()); n32 += c.size
        if len(x) >= 8:
            c = np.lib.stride_tricks.sliding_window_view(x, 8, axis=0).sum(-1); p84 += int((c > 4).sum()); n84 += c.size
    return {"valid_firing_fraction": total / max(positions, 1),
            "valid_spikes_per_neuron_second": total * fs / max(positions, 1),
            "active_neuron_sample_fraction": active / max(pairs, 1),
            "p32_violation_fraction": p32 / max(n32, 1), "p84_violation_fraction": p84 / max(n84, 1),
            "mean_max_run_length": float(np.mean(max_runs)) if max_runs else 0.0,
            "p95_max_run_length": float(np.quantile(max_runs, .95)) if max_runs else 0.0,
            "global_max_run_length": int(max(max_runs)) if max_runs else 0}


def save_activity(spec, model, data, config):
    device = torch.device(config.device); ld = loaders(data, spec.seed, config.batch_size, False)["test"]
    parts = [[], []]; lens = []
    with torch.no_grad():
        for X, _, lengths in ld:
            tr = model.forward_trajectory(X.to(device)); lens.append(lengths.numpy())
            for i in range(2): parts[i].append(tr["hidden_spikes"][i].cpu().numpy())
    lengths = np.concatenate(lens); rows = []
    for li in range(2):
        arr = np.concatenate(parts[li]); shifts = ARCHITECTURES[spec.architecture][li]
        for g in shift_groups(shifts):
            row = {"architecture": spec.architecture, "training_family": spec.training_family,
                   "regularization": spec.regularization, "seed": spec.seed,
                   "layer": f"L{li+1}", "shift": g["shift"], "tau_syn_ms": tau_ms_from_shift(g["shift"], data.fs)}
            row.update(_activity_for_group(arr, lengths, g["start"], g["stop"], data.fs)); rows.append(row)
    df = pd.DataFrame(rows); p = _path(config.results_dir, "activities", spec, ".csv")
    p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False); return df


def evaluate_one(spec, data, config, force=False):
    ep = _path(config.results_dir, "evaluations", spec, ".json")
    if ep.exists() and not force: return json.loads(ep.read_text())
    model, ckpt = load_model(spec, data, config); device = torch.device(config.device)
    lds = loaders(data, spec.seed, config.batch_size, False)
    native = {k: evaluate_native(spec, model, v, device) for k, v in lds.items()}
    probes = {}
    if spec.training_family == FAMILY_LOCAL:
        probes["l2_fixed250"] = fit_probe(spec, model, data, config, "l2_fixed250")
        deployment_source = "l2_fixed250"; deployment_metrics = probes["l2_fixed250"]["metrics"]
    else:
        for src in ("l2_fixed250", "pre_output_fixed250", "post_output_fixed250"):
            probes[src] = fit_probe(spec, model, data, config, src)
        deployment_source = "e2e_wholecount"; deployment_metrics = native
    raster = save_rasters(spec, model, data, config); save_activity(spec, model, data, config)
    payload = {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION,
               "spec": asdict(spec), "best_epoch": int(ckpt["best_epoch"]), "stopped_epoch": int(ckpt["stopped_epoch"]),
               "native_metrics": native, "deployment_source": deployment_source,
               "deployment_metrics": deployment_metrics, "probes": probes, "raster_sample": raster,
               "calibration": ckpt["calibration"]}
    _save_json(ep, payload); return payload


def run_one(spec, data, config, force=False):
    train_one(spec, data, config, force); return evaluate_one(spec, data, config, force)


def _performance_rows(payload):
    spec = payload["spec"]; base = {"architecture": spec["architecture"], "training_family": spec["training_family"],
                                    "regularization": spec["regularization"], "seed": int(spec["seed"])}
    rows = []
    def add(source, metrics):
        for split, m in metrics.items():
            rows.append({**base, "source": source, "split": split, "accuracy": float(m["accuracy"]),
                         "balanced_accuracy": float(m["balanced_accuracy"]), "macro_f1": float(m["macro_f1"])})
    add("e2e_wholecount" if spec["training_family"] == FAMILY_E2E else "temporary_analog_head", payload["native_metrics"])
    for source, probe in payload["probes"].items(): add(source, probe["metrics"])
    return rows


def _agg(df, groups, values):
    g = df.groupby(groups, dropna=False)[values]
    return pd.concat([g.size().rename("n"), g.mean().add_suffix("_mean"), g.std(ddof=1).fillna(0).add_suffix("_std")], axis=1).reset_index()


def finalize(repo_root: Path):
    root = results_dir(repo_root); perf, dyn, cal, missing = [], [], [], []
    for spec in run_specs():
        ep = _path(root, "evaluations", spec, ".json")
        if not ep.exists(): missing.append(spec.key); continue
        payload = json.loads(ep.read_text()); perf.extend(_performance_rows(payload))
        ap = _path(root, "activities", spec, ".csv")
        if ap.exists(): dyn.append(pd.read_csv(ap))
        cp = _path(root, "calibrations", spec, ".json")
        if cp.exists():
            c = json.loads(cp.read_text()); cal.append({"architecture": spec.architecture, "training_family": spec.training_family,
                "regularization": spec.regularization, "seed": spec.seed, "lambda_rate": c.get("lambda_rate", 0),
                "lambda_persist": c.get("lambda_persist", 0), "achieved_rate_grad_ratio": c.get("achieved_rate_grad_ratio", 0),
                "achieved_persist_grad_ratio": c.get("achieved_persist_grad_ratio", 0)})
    if missing: raise RuntimeError(f"Missing {len(missing)} Exp7.2 evaluations; first={missing[:8]}")
    pr = pd.DataFrame(perf); pr.to_csv(root / "performance_runs.csv", index=False)
    ps = _agg(pr, ["architecture", "training_family", "regularization", "source", "split"], ["accuracy", "balanced_accuracy", "macro_f1"])
    ps.to_csv(root / "performance_summary.csv", index=False)
    dr = pd.concat(dyn, ignore_index=True); dr.to_csv(root / "dynamics_runs.csv", index=False)
    ds = _agg(dr, ["architecture", "training_family", "regularization", "layer", "shift"],
              ["tau_syn_ms", "valid_firing_fraction", "valid_spikes_per_neuron_second", "active_neuron_sample_fraction",
               "p32_violation_fraction", "p84_violation_fraction", "mean_max_run_length", "p95_max_run_length", "global_max_run_length"])
    ds.to_csv(root / "dynamics_summary.csv", index=False)
    cr = pd.DataFrame(cal); cr.to_csv(root / "calibration_runs.csv", index=False)
    cs = _agg(cr, ["architecture", "training_family", "regularization"],
              ["lambda_rate", "lambda_persist", "achieved_rate_grad_ratio", "achieved_persist_grad_ratio"])
    cs.to_csv(root / "calibration_summary.csv", index=False)
    test = pr[pr.split == "test"].copy(); deltas = []
    for (a, f, src, seed), g in test.groupby(["architecture", "training_family", "source", "seed"]):
        p = g.set_index("regularization")
        if TASK_ONLY in p.index and TASK_PLUS_REG in p.index:
            for m in ("accuracy", "balanced_accuracy", "macro_f1"):
                deltas.append({"contrast": "reg_minus_task_only", "architecture": a, "training_family": f, "source": src,
                               "seed": seed, "metric": m, "delta": float(p.loc[TASK_PLUS_REG, m] - p.loc[TASK_ONLY, m])})
    e2e = test[(test.training_family == FAMILY_E2E)]
    for (a, r, seed), g in e2e.groupby(["architecture", "regularization", "seed"]):
        p = g.set_index("source")
        for name, left, right in (("l2_probe_minus_e2e", "l2_fixed250", "e2e_wholecount"),
                                  ("pre_minus_post_output_probe", "pre_output_fixed250", "post_output_fixed250")):
            if left in p.index and right in p.index:
                for m in ("accuracy", "balanced_accuracy", "macro_f1"):
                    deltas.append({"contrast": name, "architecture": a, "training_family": FAMILY_E2E,
                                   "source": f"{left}-{right}", "regularization": r, "seed": seed,
                                   "metric": m, "delta": float(p.loc[left, m] - p.loc[right, m])})
    local = test[(test.training_family == FAMILY_LOCAL) & (test.source == "l2_fixed250")]
    e2ed = test[(test.training_family == FAMILY_E2E) & (test.source == "e2e_wholecount")]
    mg = local.merge(e2ed, on=["architecture", "regularization", "seed", "split"], suffixes=("_local", "_e2e"))
    for row in mg.itertuples(index=False):
        for m in ("accuracy", "balanced_accuracy", "macro_f1"):
            deltas.append({"contrast": "local_linear_minus_e2e_wholecount", "architecture": row.architecture,
                           "training_family": "cross_family", "source": "deployment", "regularization": row.regularization,
                           "seed": row.seed, "metric": m, "delta": float(getattr(row, m + "_local") - getattr(row, m + "_e2e"))})
    ddf = pd.DataFrame(deltas); ddf.to_csv(root / "paired_delta_runs.csv", index=False)
    pdout = _agg(ddf, ["contrast", "architecture", "training_family", "source", "regularization", "metric"], ["delta"])
    pdout.to_csv(root / "paired_deltas.csv", index=False)
    arch = []
    for name, shifts in ARCHITECTURES.items():
        arch.append({"architecture": name, "L1_shifts": "-".join(map(str, shifts[0])), "L2_shifts": "-".join(map(str, shifts[1])),
                     "L1_tau_ms": "/".join(f"{tau_ms_from_shift(s):.1f}" for s in shifts[0]),
                     "L2_tau_ms": "/".join(f"{tau_ms_from_shift(s):.1f}" for s in shifts[1])})
    pd.DataFrame(arch).to_csv(root / "architecture_table.csv", index=False)
    manifest = {"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "expected_training_runs": len(run_specs()),
                "notebook_inputs": ["architecture_table.csv", "performance_summary.csv", "dynamics_summary.csv", "paired_deltas.csv", "calibration_summary.csv"]}
    _save_json(root / "manifest.json", manifest); return manifest


def _parser():
    p = argparse.ArgumentParser(); p.add_argument("--repo-root", type=Path); p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1); p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS); sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run-one"); r.add_argument("--array-task-id", type=int); r.add_argument("--architecture", choices=ARCHITECTURE_ORDER)
    r.add_argument("--training-family", choices=TRAINING_FAMILIES); r.add_argument("--regularization", choices=REGULARIZATION_CONDITIONS)
    r.add_argument("--seed", type=int, choices=TRAIN_SEEDS); r.add_argument("--force", action="store_true")
    sub.add_parser("finalize"); sub.add_parser("list-runs"); return p


def main():
    args = _parser().parse_args(); root = find_repo_root(args.repo_root)
    config = Config(root, results_dir(root), args.device, args.max_epochs, args.batch_size, args.threads)
    if args.command == "list-runs":
        for i, spec in enumerate(run_specs()): print(i, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(root), indent=2)); return
    specs = run_specs()
    if args.array_task_id is not None:
        spec = specs[args.array_task_id]
    else:
        spec = RunSpec(args.architecture, args.training_family, args.regularization, args.seed)
    data = prepare_data(root); payload = run_one(spec, data, config, args.force)
    print(json.dumps({"spec": asdict(spec), "deployment_source": payload["deployment_source"]}, indent=2))


if __name__ == "__main__":
    main()
