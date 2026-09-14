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
from scripts import experiment_3_0_1_single_tau_objectives as exp3
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_7_2_two_layer_tau_training as exp72

EXPERIMENT_ID = "experiment_7_2_3_objective_temporal_dynamics"
PROTOCOL_VERSION = "objective_temporal_dynamics_v1"

A1 = "a1_shared_tsce"
A2 = "a2_shared_wholecount"
F1 = "f1_fixed250_tsce"
F2 = "f2_fixed250_wholecount"
S1 = "s1_spike_wc_beta05"
S2 = "s2_spike_tsce_beta05"
S3 = "s3_spike_wc_beta10"
S4 = "s4_spike_tsce_beta10"
FAMILIES = (A1, A2, F1, F2, S1, S2, S3, S4)
REUSED_FAMILIES = (A1, S1)
NEW_FAMILIES = (A2, F1, F2, S2, S3, S4)
ANALOG_FAMILIES = (A1, A2)
FIXED_FAMILIES = (F1, F2)
SPIKING_FAMILIES = (S1, S2, S3, S4)

ARCHITECTURES = {
    "234x234": ((2, 3, 4), (2, 3, 4)),
    "34x345": ((3, 4), (3, 4, 5)),
}
ARCHITECTURE_ORDER = tuple(ARCHITECTURES)
SEEDS = (11, 23, 37)
REGULARIZATIONS = exp72.REGULARIZATION_CONDITIONS
HIDDEN_WIDTH = exp72.HIDDEN_WIDTH
OUTPUT_CAP = 1
N_BINS = 16
MAX_EPOCHS = 100
MIN_EPOCHS = 20
PATIENCE = 30


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    family: str
    regularization: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.architecture}__{self.family}__{self.regularization}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp72.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def all_specs() -> list[RunSpec]:
    return [RunSpec(a, f, r, s) for a in ARCHITECTURE_ORDER for f in FAMILIES for r in REGULARIZATIONS for s in SEEDS]


def new_specs() -> list[RunSpec]:
    return [spec for spec in all_specs() if spec.family in NEW_FAMILIES]


def reused_specs() -> list[RunSpec]:
    return [spec for spec in all_specs() if spec.family in REUSED_FAMILIES]


def _base_exp72_spec(spec: RunSpec) -> exp72.RunSpec:
    if spec.family == A1:
        family = exp72.FAMILY_LOCAL
    elif spec.family == S1:
        family = exp72.FAMILY_E2E
    else:
        raise ValueError(f"{spec.family} is not a reused Exp7.2 family")
    return exp72.RunSpec(spec.architecture, family, spec.regularization, spec.seed)


def _run_path(root: Path, kind: str, spec: RunSpec, suffix: str) -> Path:
    return root / kind / spec.family / f"{spec.key}{suffix}"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _loaders(data: exp3.Data, spec: RunSpec, batch: int, shuffle: bool):
    parts = {
        "train": (data.Xtr, data.ytr, data.ltr),
        "val": (data.Xva, data.yva, data.lva),
        "test": (data.Xte, data.yte, data.lte),
    }
    return {
        name: exp3.loader(X, y, lengths, batch, shuffle if name == "train" else False,
                         exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, name))
        for name, (X, y, lengths) in parts.items()
    }


def _fixed_means_sizes(l2: torch.Tensor, lengths: torch.Tensor, bin_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    counts = exp3.fixed_counts(l2, lengths, bin_steps)
    positions = torch.arange(l2.shape[1], device=l2.device)[None, :]
    valid = (positions < lengths[:, None]).to(l2.dtype)
    pad = counts.shape[1] * bin_steps - l2.shape[1]
    valid_pad = F.pad(valid, (0, pad))
    sizes = valid_pad.reshape(len(l2), counts.shape[1], bin_steps).sum(dim=2)
    means = counts / sizes.clamp_min(1.0).unsqueeze(-1)
    return means, sizes


class Exp723Model(nn.Module):
    def __init__(self, spec: RunSpec, n_classes: int, fs: float) -> None:
        super().__init__()
        self.spec = spec
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        shifts = ARCHITECTURES[spec.architecture]
        self.hidden_linears = nn.ModuleList([
            nn.Linear(exp72.EXPECTED_CHANNELS, HIDDEN_WIDTH, bias=False),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=False),
        ])
        beta_hidden = math.exp(-(1000.0 / fs) / exp72.TAU_MEM_MS)
        self.hidden_lifs = nn.ModuleList([
            exp401.MacroMultiSpikeLIF(beta=beta_hidden, threshold=exp72.THRESHOLD,
                                      max_spikes_per_dt=1, surrogate_slope=exp72.SURROGATE_SLOPE)
            for _ in range(2)
        ])
        self.register_buffer("alpha_0", exp50.alpha_vector(HIDDEN_WIDTH, shifts[0]))
        self.register_buffer("alpha_1", exp50.alpha_vector(HIDDEN_WIDTH, shifts[1]))

        self.shared_head: nn.Linear | None = None
        self.phase_weight: nn.Parameter | None = None
        self.phase_bias: nn.Parameter | None = None
        self.output_linear: nn.Linear | None = None
        self.output_lif: exp401.MacroMultiSpikeLIF | None = None

        if spec.family == A2:
            self.shared_head = nn.Linear(HIDDEN_WIDTH, n_classes, bias=True)
        elif spec.family in FIXED_FAMILIES:
            self.phase_weight = nn.Parameter(torch.empty(N_BINS, n_classes, HIDDEN_WIDTH))
            self.phase_bias = nn.Parameter(torch.zeros(N_BINS, n_classes))
            nn.init.kaiming_uniform_(self.phase_weight, a=math.sqrt(5))
        elif spec.family in (S2, S3, S4):
            self.output_linear = nn.Linear(HIDDEN_WIDTH, n_classes, bias=False)
            beta = 0.5 if spec.family in (S2,) else 1.0
            self.output_lif = exp401.MacroMultiSpikeLIF(beta=beta, threshold=exp72.THRESHOLD,
                                                       max_spikes_per_dt=1,
                                                       surrogate_slope=exp72.SURROGATE_SLOPE)
        else:
            raise ValueError(f"New model cannot instantiate family {spec.family}")

    def forward_trajectory(self, x: torch.Tensor) -> dict[str, Any]:
        batch, steps, channels = x.shape
        if channels != exp72.EXPECTED_CHANNELS:
            raise ValueError(channels)
        syn = [torch.zeros(batch, HIDDEN_WIDTH, device=x.device, dtype=x.dtype) for _ in range(2)]
        mem = [torch.zeros_like(syn[0]), torch.zeros_like(syn[1])]
        hidden = [[], []]
        analog_logits, pre_output, output_spikes = [], [], []
        out_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        for t in range(steps):
            cur = x[:, t]
            for li in range(2):
                alpha = getattr(self, f"alpha_{li}")
                syn[li] = alpha * syn[li] + self.hidden_linears[li](cur)
                spk, mem[li], _ = self.hidden_lifs[li](syn[li], mem[li])
                hidden[li].append(spk)
                cur = spk
            if self.shared_head is not None:
                analog_logits.append(self.shared_head(cur))
            if self.output_linear is not None and self.output_lif is not None:
                evidence = self.output_linear(cur)
                out_spk, out_mem, _ = self.output_lif(evidence, out_mem)
                pre_output.append(evidence)
                output_spikes.append(out_spk)
        payload: dict[str, Any] = {"hidden_spikes": tuple(torch.stack(v, dim=1) for v in hidden)}
        if analog_logits:
            payload["analog_logits"] = torch.stack(analog_logits, dim=1)
        if output_spikes:
            payload["pre_output_evidence"] = torch.stack(pre_output, dim=1)
            payload["output_spikes"] = torch.stack(output_spikes, dim=1)
        return payload

    def phase_logits(self, l2: torch.Tensor, lengths: torch.Tensor, bin_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase_weight is None or self.phase_bias is None:
            raise RuntimeError("phase head unavailable")
        means, sizes = _fixed_means_sizes(l2, lengths, bin_steps)
        if means.shape[1] != N_BINS:
            raise ValueError(f"Expected {N_BINS} bins, got {means.shape[1]}")
        logits = torch.einsum("bnd,nkd->bnk", means, self.phase_weight) + self.phase_bias.unsqueeze(0)
        return logits, sizes


def _objective(spec: RunSpec, model: Exp723Model, tr: dict[str, Any], lengths: torch.Tensor,
               y: torch.Tensor, bin_steps: int) -> torch.Tensor:
    if spec.family == A2:
        return F.cross_entropy(exp50.valid_mean(tr["analog_logits"], lengths), y)
    if spec.family in FIXED_FAMILIES:
        logits, sizes = model.phase_logits(tr["hidden_spikes"][-1], lengths, bin_steps)
        weights = sizes / lengths.clamp_min(1).to(sizes.dtype).unsqueeze(1)
        if spec.family == F1:
            targets = y[:, None].expand(len(y), logits.shape[1])
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
            ce = ce.reshape(len(y), logits.shape[1])
            return (ce * weights).sum(dim=1).mean()
        return F.cross_entropy((logits * weights.unsqueeze(-1)).sum(dim=1), y)
    if spec.family in (S2, S4):
        return exp50.objective_loss(tr["output_spikes"], lengths, y, "timestep_ce", OUTPUT_CAP, model.fs)
    if spec.family == S3:
        return exp50.objective_loss(tr["output_spikes"], lengths, y, "whole_count_ce", OUTPUT_CAP, model.fs)
    raise ValueError(spec.family)


def _native_logits(spec: RunSpec, model: Exp723Model, tr: dict[str, Any], lengths: torch.Tensor,
                   bin_steps: int) -> torch.Tensor:
    if spec.family == A2:
        return exp50.valid_mean(tr["analog_logits"], lengths)
    if spec.family in FIXED_FAMILIES:
        logits, sizes = model.phase_logits(tr["hidden_spikes"][-1], lengths, bin_steps)
        weights = sizes / lengths.clamp_min(1).to(sizes.dtype).unsqueeze(1)
        return (logits * weights.unsqueeze(-1)).sum(dim=1)
    if spec.family in SPIKING_FAMILIES:
        return exp50.deployment_logits(tr["output_spikes"], lengths, OUTPUT_CAP)
    raise ValueError(spec.family)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return exp72._metrics(y, pred)


def _evaluate_native(spec: RunSpec, model: Exp723Model, loader, device: torch.device, bin_steps: int) -> dict[str, float]:
    ys, preds, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            tr = model.forward_trajectory(X)
            logits = _native_logits(spec, model, tr, lengths, bin_steps)
            ys.append(y.cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            losses.append((float(F.cross_entropy(logits, y)) * len(y), len(y)))
    out = _metrics(np.concatenate(ys), np.concatenate(preds))
    out["loss"] = sum(v for v, _ in losses) / max(sum(n for _, n in losses), 1)
    return out


def _regularizer(spec: RunSpec, tr: dict[str, Any], lengths: torch.Tensor, calibration: dict[str, Any], epoch: int):
    if spec.regularization == exp72.TASK_ONLY:
        return torch.zeros((), device=lengths.device)
    rate, _, _, persist = exp72.regularization_terms(tr["hidden_spikes"], lengths)
    scale = exp72.warmup_scale(epoch)
    return scale * (float(calibration["lambda_rate"]) * rate + float(calibration["lambda_persist"]) * persist)


def _calibrate(spec: RunSpec, model: Exp723Model, data: exp3.Data, config: Config) -> dict[str, Any]:
    if spec.regularization == exp72.TASK_ONLY:
        return {"calibrated": False, "lambda_rate": 0.0, "lambda_persist": 0.0}
    loader = _loaders(data, spec, config.batch_size, True)["train"]
    params = tuple(layer.weight for layer in model.hidden_linears)
    rate_ratios, persist_ratios = [], []
    device = torch.device(config.device)
    for i, (X, y, lengths) in enumerate(loader):
        if i >= exp72.CALIBRATION_BATCHES:
            break
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        tr = model.forward_trajectory(X)
        task = _objective(spec, model, tr, lengths, y, data.bin_steps)
        rate, _, _, persist = exp72.regularization_terms(tr["hidden_spikes"], lengths)
        tg = exp72._grad_norm(task, params, True)
        rg = exp72._grad_norm(rate, params, True)
        pg = exp72._grad_norm(persist, params, False)
        if tg > 1e-12:
            rate_ratios.append(rg / tg)
            persist_ratios.append(pg / tg)
    if not rate_ratios:
        raise RuntimeError(f"No calibration batches for {spec.key}")
    rr, pr = float(np.median(rate_ratios)), float(np.median(persist_ratios))
    return {
        "calibrated": True,
        "lambda_rate": exp72.TARGET_RATE_GRAD_RATIO / rr if rr > 1e-12 else 1.0,
        "lambda_persist": exp72.TARGET_PERSIST_GRAD_RATIO / pr if pr > 1e-12 else 1.0,
        "raw_rate_to_task": rr,
        "raw_persist_to_task": pr,
    }


def _plot_history(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.epoch, frame.train_ba, label="train BA")
    ax.plot(frame.epoch, frame.val_ba, label="val BA")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax2 = ax.twinx()
    ax2.plot(frame.epoch, frame.train_total_loss, linestyle="--", label="train loss")
    ax2.plot(frame.epoch, frame.val_loss, linestyle=":", label="val loss")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> Path:
    if spec.family not in NEW_FAMILIES:
        raise ValueError("train_one only accepts new families")
    checkpoint = _run_path(config.results_dir, "checkpoints", spec, ".pt")
    if checkpoint.exists() and not force:
        return checkpoint
    torch.set_num_threads(config.threads)
    exp3.seed_all(exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, "model_init"))
    device = torch.device(config.device)
    model = Exp723Model(spec, len(data.labels), data.fs).to(device)
    calibration = _calibrate(spec, model, data, config)
    _save_json(_run_path(config.results_dir, "calibrations", spec, ".json"), calibration)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp72.LR, weight_decay=exp72.WEIGHT_DECAY)
    train_loader = _loaders(data, spec, config.batch_size, True)["train"]
    eval_loaders = _loaders(data, spec, config.batch_size, False)
    best_state, best_epoch, best_ba, best_loss = None, -1, -1.0, float("inf")
    rows: list[dict[str, float]] = []
    stopped = config.max_epochs
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for X, y, lengths in train_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            tr = model.forward_trajectory(X)
            task = _objective(spec, model, tr, lengths, y, data.bin_steps)
            reg = _regularizer(spec, tr, lengths, calibration, epoch)
            loss = task + reg
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        train_metrics = _evaluate_native(spec, model, eval_loaders["train"], device, data.bin_steps)
        val_metrics = _evaluate_native(spec, model, eval_loaders["val"], device, data.bin_steps)
        rows.append({"epoch": epoch, "train_ba": train_metrics["balanced_accuracy"],
                     "val_ba": val_metrics["balanced_accuracy"], "train_total_loss": total_loss / max(batches, 1),
                     "val_loss": val_metrics["loss"]})
        improved = val_metrics["balanced_accuracy"] > best_ba + 1e-12 or (
            abs(val_metrics["balanced_accuracy"] - best_ba) <= 1e-12 and val_metrics["loss"] < best_loss)
        if improved:
            best_ba = val_metrics["balanced_accuracy"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped = epoch
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"experiment_id": EXPERIMENT_ID, "protocol_version": PROTOCOL_VERSION, "spec": asdict(spec),
                "best_epoch": best_epoch, "stopped_epoch": stopped, "best_val_ba": best_ba,
                "best_val_loss": best_loss, "model_state_dict": best_state, "calibration": calibration}, checkpoint)
    history = pd.DataFrame(rows)
    history_path = _run_path(config.results_dir, "histories", spec, ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index=False)
    _plot_history(history, _run_path(config.results_dir, "training_curves", spec, ".png"), spec.key)
    return checkpoint


def _load_new_model(spec: RunSpec, data: exp3.Data, config: Config):
    payload = torch.load(_run_path(config.results_dir, "checkpoints", spec, ".pt"), map_location=config.device, weights_only=False)
    model = Exp723Model(spec, len(data.labels), data.fs).to(config.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _load_any_model(spec: RunSpec, data: exp3.Data, config: Config):
    if spec.family in REUSED_FAMILIES:
        base = _base_exp72_spec(spec)
        base_config = exp72.Config(config.repo_root, exp72.results_dir(config.repo_root), config.device,
                                   exp72.MAX_EPOCHS, config.batch_size, config.threads)
        model, payload = exp72.load_model(base, data, base_config)
        return model, payload, True
    model, payload = _load_new_model(spec, data, config)
    return model, payload, False


def _extract_l2(model, loader, device: torch.device):
    xs, ys, lengths = [], [], []
    with torch.no_grad():
        for X, y, l in loader:
            tr = model.forward_trajectory(X.to(device))
            xs.append(tr["hidden_spikes"][-1].cpu())
            ys.append(y.numpy())
            lengths.append(l.numpy())
    return torch.cat(xs, dim=0), np.concatenate(ys), np.concatenate(lengths)


def _probe_features(l2: torch.Tensor, lengths: np.ndarray, bin_steps: int, source: str) -> np.ndarray:
    lt = torch.tensor(lengths, dtype=torch.long)
    if source == "l2_wholecount_linear":
        mask = exp50.valid_mask(lt, l2.shape[1]).to(l2.dtype).unsqueeze(-1)
        return (l2 * mask).sum(dim=1).numpy()
    if source == "l2_fixed250_linear":
        return exp3.fixed_counts(l2, lt, bin_steps).flatten(1).numpy()
    raise ValueError(source)


def _fit_probe(source: str, splits: dict[str, tuple[torch.Tensor, np.ndarray, np.ndarray]], spec: RunSpec,
               bin_steps: int) -> dict[str, Any]:
    features = {name: (_probe_features(l2, lengths, bin_steps, source), y)
                for name, (l2, y, lengths) in splits.items()}
    probe = exp01._fit_linear_probe(features["train"][0], features["train"][1],
                                    features["val"][0], features["val"][1],
                                    features["test"][0], features["test"][1],
                                    exp3.dseed(spec.seed, EXPERIMENT_ID, spec.key, source))
    return {"source": source, "feature_dim": int(probe["feature_dim"]), "probe_C": float(probe["probe_C"]),
            "metrics": {split: probe[split] for split in ("train", "val", "test")}}


def _numpy_lif(l2: np.ndarray, lengths: np.ndarray, W: np.ndarray, beta: float, full_window: bool) -> np.ndarray:
    n, steps, _ = l2.shape
    k = W.shape[0]
    mem = np.zeros((n, k), dtype=np.float64)
    counts = np.zeros((n, k), dtype=np.float64)
    for t in range(steps):
        active = np.ones(n, dtype=bool) if full_window else (t < lengths)
        evidence = l2[:, t].astype(np.float64) @ W.T
        pre = beta * mem + evidence
        spikes = (pre >= exp72.THRESHOLD).astype(np.float64)
        spikes[~active] = 0.0
        updated = pre - spikes * exp72.THRESHOLD
        mem[active] = updated[active]
        counts += spikes
    return counts


def _score_spiking_counterfactual(splits, W: np.ndarray, beta: float, full_window: bool):
    result = {}
    for name, (l2, y, lengths) in splits.items():
        scores = _numpy_lif(l2.numpy(), lengths, W, beta, full_window)
        result[name] = _metrics(y, scores.argmax(1))
    return result


def _native_reuse_metrics(spec: RunSpec, model, loaders, device: torch.device):
    base = _base_exp72_spec(spec)
    return {split: exp72.evaluate_native(base, model, loader, device) for split, loader in loaders.items()}


def _native_new_metrics(spec: RunSpec, model: Exp723Model, loaders, device: torch.device, bin_steps: int):
    return {split: _evaluate_native(spec, model, loader, device, bin_steps) for split, loader in loaders.items()}


def _native_analog_from_w(splits, W: np.ndarray, full_window: bool):
    out = {}
    for name, (l2, y, lengths) in splits.items():
        arr = l2.numpy()
        if full_window:
            summed = arr.sum(axis=1)
        else:
            mask = (np.arange(arr.shape[1])[None, :] < lengths[:, None])[..., None]
            summed = (arr * mask).sum(axis=1)
        out[name] = _metrics(y, (summed.astype(np.float64) @ W.T).argmax(1))
    return out


def save_rasters(spec: RunSpec, model, data: exp3.Data, config: Config) -> None:
    idx = int(np.argmin(np.abs(data.lva.astype(float) - float(np.median(data.lva)))))
    valid = int(data.lva[idx])
    sample = torch.tensor(data.Xva[idx:idx + 1], dtype=torch.float32, device=config.device)
    with torch.no_grad():
        tr = model.forward_trajectory(sample)
    for li, spikes in enumerate(tr["hidden_spikes"], 1):
        exp72._raster(spikes[0].cpu().numpy(),
                      config.results_dir / "rasters" / spec.family / f"{spec.key}__L{li}.png",
                      valid, f"{spec.key} — L{li}", exp72.shift_groups(ARCHITECTURES[spec.architecture][li - 1]))
    if "output_spikes" in tr:
        exp72._raster(tr["output_spikes"][0].cpu().numpy(),
                      config.results_dir / "rasters" / spec.family / f"{spec.key}__output.png",
                      valid, f"{spec.key} — output")


def evaluate_one(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False) -> dict[str, Any]:
    destination = _run_path(config.results_dir, "evaluations", spec, ".json")
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    model, checkpoint_payload, reused = _load_any_model(spec, data, config)
    device = torch.device(config.device)
    loaders = _loaders(data, spec, config.batch_size, False)
    splits = {name: _extract_l2(model, loader, device) for name, loader in loaders.items()}
    probes = {source: _fit_probe(source, splits, spec, data.bin_steps)
              for source in ("l2_wholecount_linear", "l2_fixed250_linear")}

    if reused:
        native = _native_reuse_metrics(spec, model, loaders, device)
    else:
        native = _native_new_metrics(spec, model, loaders, device, data.bin_steps)

    dynamics: dict[str, Any] = {}
    if spec.family in SPIKING_FAMILIES:
        if model.output_linear is None:
            raise RuntimeError("spiking family missing output_linear")
        W = model.output_linear.weight.detach().cpu().numpy().astype(np.float64)
        dynamics["native_w_analog_valid"] = _native_analog_from_w(splits, W, False)
        dynamics["native_w_analog_full"] = _native_analog_from_w(splits, W, True)
        dynamics["native_lif_beta05_valid"] = _score_spiking_counterfactual(splits, W, 0.5, False)
        dynamics["native_lif_beta10_valid"] = _score_spiking_counterfactual(splits, W, 1.0, False)
        dynamics["native_lif_beta05_full"] = _score_spiking_counterfactual(splits, W, 0.5, True)
        dynamics["native_lif_beta10_full"] = _score_spiking_counterfactual(splits, W, 1.0, True)

    save_rasters(spec, model, data, config)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "reuses_exp7_2_checkpoint": reused,
        "best_epoch": int(checkpoint_payload["best_epoch"]),
        "native_metrics": native,
        "probes": probes,
        "output_dynamics": dynamics,
    }
    _save_json(destination, payload)
    return payload


def run_new(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False):
    train_one(spec, data, config, force)
    return evaluate_one(spec, data, config, force)


def run_reuse(spec: RunSpec, data: exp3.Data, config: Config, force: bool = False):
    if spec.family not in REUSED_FAMILIES:
        raise ValueError(spec.family)
    return evaluate_one(spec, data, config, force)


def _performance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = payload["spec"]
    base = {"architecture": spec["architecture"], "family": spec["family"],
            "regularization": spec["regularization"], "seed": int(spec["seed"])}
    rows = []
    for split, metrics in payload["native_metrics"].items():
        rows.append({**base, "source": "native", "split": split,
                     "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                     "macro_f1": metrics["macro_f1"]})
    for source, probe in payload["probes"].items():
        for split, metrics in probe["metrics"].items():
            rows.append({**base, "source": source, "split": split,
                         "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                         "macro_f1": metrics["macro_f1"]})
    for source, split_map in payload["output_dynamics"].items():
        for split, metrics in split_map.items():
            rows.append({**base, "source": source, "split": split,
                         "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                         "macro_f1": metrics["macro_f1"]})
    return rows


def _aggregate(df: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    grouped = df.groupby(groups, dropna=False)[values]
    return pd.concat([grouped.size().rename("n"), grouped.mean().add_suffix("_mean"),
                      grouped.std(ddof=1).fillna(0).add_suffix("_std")], axis=1).reset_index()


def finalize(repo_root: Path) -> dict[str, Any]:
    root = results_dir(repo_root)
    rows, missing = [], []
    for spec in all_specs():
        path = _run_path(root, "evaluations", spec, ".json")
        if not path.exists():
            missing.append(spec.key)
            continue
        rows.extend(_performance_rows(json.loads(path.read_text(encoding="utf-8"))))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} Exp7.2.3 evaluations; first={missing[:8]}")
    runs = pd.DataFrame(rows)
    runs.to_csv(root / "performance_runs.csv", index=False)
    summary = _aggregate(runs, ["architecture", "family", "regularization", "source", "split"],
                         ["accuracy", "balanced_accuracy", "macro_f1"])
    summary.to_csv(root / "performance_summary.csv", index=False)

    probes = runs[runs.source.isin(["l2_wholecount_linear", "l2_fixed250_linear"])].copy()
    probes.to_csv(root / "representation_probe_runs.csv", index=False)
    _aggregate(probes, ["architecture", "family", "regularization", "source", "split"],
               ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "representation_probe_summary.csv", index=False)

    test = runs[runs.split == "test"].copy()
    contrasts = [
        ("probe_phase_gain", "l2_fixed250_linear", "l2_wholecount_linear"),
        ("lif_loss_beta05", "native_w_analog_valid", "native_lif_beta05_valid"),
        ("lif_loss_beta10", "native_w_analog_valid", "native_lif_beta10_valid"),
        ("leak_effect_valid", "native_lif_beta10_valid", "native_lif_beta05_valid"),
        ("tail_beta05", "native_lif_beta05_valid", "native_lif_beta05_full"),
        ("tail_beta10", "native_lif_beta10_valid", "native_lif_beta10_full"),
    ]
    delta_rows = []
    for (architecture, family, regularization, seed), group in test.groupby(["architecture", "family", "regularization", "seed"]):
        indexed = group.set_index("source")
        for name, left, right in contrasts:
            if left not in indexed.index or right not in indexed.index:
                continue
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                delta_rows.append({"contrast": name, "architecture": architecture, "family": family,
                                   "regularization": regularization, "seed": int(seed), "metric": metric,
                                   "delta": float(indexed.loc[left, metric] - indexed.loc[right, metric])})
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "paired_delta_runs.csv", index=False)
    if not deltas.empty:
        _aggregate(deltas, ["contrast", "architecture", "family", "regularization", "metric"], ["delta"]).to_csv(
            root / "paired_delta_summary.csv", index=False)
    else:
        deltas.to_csv(root / "paired_delta_summary.csv", index=False)

    dynamics = runs[runs.source.str.startswith("native_")].copy()
    dynamics.to_csv(root / "output_dynamics_runs.csv", index=False)
    _aggregate(dynamics, ["architecture", "family", "regularization", "source", "split"],
               ["accuracy", "balanced_accuracy", "macro_f1"]).to_csv(root / "output_dynamics_summary.csv", index=False)

    arch = pd.DataFrame([
        {"architecture": name, "L1_shifts": "-".join(map(str, shifts[0])), "L2_shifts": "-".join(map(str, shifts[1]))}
        for name, shifts in ARCHITECTURES.items()
    ])
    arch.to_csv(root / "architecture_table.csv", index=False)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architectures": list(ARCHITECTURE_ORDER),
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "regularizations": list(REGULARIZATIONS),
        "new_training_runs": len(new_specs()),
        "reused_checkpoint_evaluations": len(reused_specs()),
        "total_evaluations": len(all_specs()),
        "notebook_inputs": ["architecture_table.csv", "performance_summary.csv", "representation_probe_summary.csv",
                            "paired_delta_summary.csv", "output_dynamics_summary.csv"],
    }
    _save_json(root / "manifest.json", manifest)
    return manifest


def _select(specs: list[RunSpec], array_task_id: int | None, args) -> RunSpec:
    if array_task_id is not None:
        if not 0 <= array_task_id < len(specs):
            raise ValueError(array_task_id)
        return specs[array_task_id]
    if None in (args.architecture, args.family, args.regularization, args.seed):
        raise ValueError("Specify --array-task-id or all explicit run fields")
    return RunSpec(args.architecture, args.family, args.regularization, args.seed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run-new", "run-reuse"):
        p = sub.add_parser(name)
        p.add_argument("--array-task-id", type=int)
        p.add_argument("--architecture", choices=ARCHITECTURE_ORDER)
        p.add_argument("--family", choices=FAMILIES)
        p.add_argument("--regularization", choices=REGULARIZATIONS)
        p.add_argument("--seed", type=int, choices=SEEDS)
        p.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    sub.add_parser("list-new")
    sub.add_parser("list-reuse")
    return parser


def main() -> None:
    args = _parser().parse_args()
    repo_root = find_repo_root(args.repo_root)
    config = Config(repo_root, results_dir(repo_root), args.device, args.batch_size, args.threads, args.max_epochs)
    if args.command == "list-new":
        for i, spec in enumerate(new_specs()):
            print(i, spec.key)
        return
    if args.command == "list-reuse":
        for i, spec in enumerate(reused_specs()):
            print(i, spec.key)
        return
    if args.command == "finalize":
        print(json.dumps(finalize(repo_root), indent=2))
        return
    data = exp72.prepare_data(repo_root)
    specs = new_specs() if args.command == "run-new" else reused_specs()
    spec = _select(specs, args.array_task_id, args)
    payload = run_new(spec, data, config, args.force) if args.command == "run-new" else run_reuse(spec, data, config, args.force)
    print(json.dumps({"spec": payload["spec"], "reused": payload["reuses_exp7_2_checkpoint"]}, indent=2))


if __name__ == "__main__":
    main()
