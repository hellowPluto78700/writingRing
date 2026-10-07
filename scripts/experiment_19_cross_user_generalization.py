#!/usr/bin/env python3
"""Exp19: cross-user generalization mechanisms under the locked CoreBenchmark contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from core_benchmark_v1.data import loader
from core_benchmark_v1.model import BenchmarkNet, mean_logits, valid_mask, valid_sum
from core_benchmark_v1.probes import run_probes
from core_benchmark_v1.protocol import Protocol, Run, paired_seed
from core_benchmark_v1.storage import file_hash, save_json, save_torch
from core_benchmark_v1.training import cpu_state, evaluate_validation, metrics
from scripts import experiment_16_prefix_supervised_selective_memory as exp16

EXPERIMENT_ID = "experiment_19_cross_user_generalization"
PROTOCOL_VERSION = "cross_user_generalization_v1"
SEEDS = (11, 23, 37)
CALIBRATION_SEED = 101
SHIFTS = ((2, 3, 4), (2, 3, 4))
CORE_RESULTS_REL = Path("core_benchmark_v1/results/main")

F19_1_VARIANTS = ("random", "high_rate", "evidence", "evidence_matched_random")
F19_2_VARIANTS = ("margin_consistency", "rex")
F19_3_VARIANTS = ("matched_noise", "physical", "style_mix")
Q_GRID = (0.10, 0.20, 0.30)
SC_LAMBDA_GRID = (0.5, 1.0)
MARGIN_LAMBDA_GRID = (0.02, 0.05, 0.10, 0.20)
REX_LAMBDA_GRID = (0.1, 0.5, 1.0, 2.0)
PHYSICAL_STRENGTHS = ("weak", "medium", "strong")
STYLE_PROB_GRID = (0.25, 0.50, 0.75)
WARMUP_FRACTION = 0.25
RAMP_END_FRACTION = 0.40
STYLE_EPS = 1e-6


@dataclass(frozen=True)
class ExpSpec:
    family: str
    variant: str
    seed: int
    q: float | None = None
    weight: float | None = None
    strength: str | None = None
    mix_prob: float | None = None

    @property
    def key(self) -> str:
        parts = [self.family.replace(".", "_"), self.variant, f"seed{self.seed}"]
        if self.q is not None:
            parts.append(f"q{self.q:g}".replace(".", "p"))
        if self.weight is not None:
            parts.append(f"w{self.weight:g}".replace(".", "p"))
        if self.strength is not None:
            parts.append(self.strength)
        if self.mix_prob is not None:
            parts.append(f"p{self.mix_prob:g}".replace(".", "p"))
        return "__".join(parts)


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    core_results_dir: Path


def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in (path, *path.parents):
        if (parent / "AGENTS.md").exists() and (parent / "core_benchmark_v1").exists():
            return parent
    raise FileNotFoundError("repository root not found")


def default_results_dir(root: Path) -> Path:
    return root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def config_from_args(args: argparse.Namespace) -> Config:
    root = find_repo_root()
    return Config(
        root,
        (args.results or default_results_dir(root)).resolve(),
        (args.core_results or root / CORE_RESULTS_REL).resolve(),
    )


def _core(config: Config) -> tuple[Protocol, dict[str, Any], dict[str, np.ndarray]]:
    p, lock, arrays = exp16._load_core(config)
    if tuple(p.seeds) != SEEDS or p.width != 128 or p.input_channels != 30 or p.fs != 64.0:
        raise ValueError("Exp19 requires locked CoreBenchmark v1 geometry")
    if p.wcce_reduction != "valid_mean_logits" or p.native_bias:
        raise ValueError("Exp19 requires bias-free valid-mean WCCE")
    return p, lock, arrays


def _run(spec: ExpSpec) -> Run:
    return Run(spec.key, spec.seed, "19_generalization", shifts=SHIFTS, objective="wcce")


def calibration_specs() -> list[ExpSpec]:
    specs: list[ExpSpec] = []
    specs.extend(
        ExpSpec("19.1", "evidence", CALIBRATION_SEED, q=q, weight=w)
        for q in Q_GRID for w in SC_LAMBDA_GRID
    )
    specs.extend(
        ExpSpec("19.2", "margin_consistency", CALIBRATION_SEED, weight=w)
        for w in MARGIN_LAMBDA_GRID
    )
    specs.extend(
        ExpSpec("19.2", "rex", CALIBRATION_SEED, weight=w)
        for w in REX_LAMBDA_GRID
    )
    specs.extend(
        ExpSpec("19.3", "physical", CALIBRATION_SEED, strength=s)
        for s in PHYSICAL_STRENGTHS
    )
    specs.extend(
        ExpSpec("19.3", "style_mix", CALIBRATION_SEED, mix_prob=p)
        for p in STYLE_PROB_GRID
    )
    return specs


def final_specs(selection: dict[str, Any]) -> list[ExpSpec]:
    q = float(selection["19.1"]["q"])
    sc_weight = float(selection["19.1"]["weight"])
    margin_weight = float(selection["19.2"]["margin_weight"])
    rex_weight = float(selection["19.2"]["rex_weight"])
    strength = str(selection["19.3"]["strength"])
    mix_prob = float(selection["19.3"]["mix_prob"])
    out: list[ExpSpec] = []
    for seed in SEEDS:
        out.extend(ExpSpec("19.1", v, seed, q=q, weight=sc_weight) for v in F19_1_VARIANTS)
        out.append(ExpSpec("19.2", "margin_consistency", seed, weight=margin_weight))
        out.append(ExpSpec("19.2", "rex", seed, weight=rex_weight))
        out.extend((
            ExpSpec("19.3", "matched_noise", seed, strength=strength),
            ExpSpec("19.3", "physical", seed, strength=strength),
            ExpSpec("19.3", "style_mix", seed, mix_prob=mix_prob),
        ))
    return out


def challenge_weight(epoch: int, max_epochs: int, target: float) -> float:
    warm = max(1, int(round(max_epochs * WARMUP_FRACTION)))
    ramp = max(warm + 1, int(round(max_epochs * RAMP_END_FRACTION)))
    if epoch <= warm:
        return 0.0
    if epoch >= ramp:
        return float(target)
    return float(target) * (epoch - warm) / (ramp - warm)


def native_scores(trajectory: dict[str, Any], lengths: torch.Tensor) -> torch.Tensor:
    return mean_logits(trajectory["evidence"], lengths)


def true_competitor(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    masked = scores.detach().clone()
    masked[torch.arange(len(y), device=y.device), y] = -torch.inf
    return masked.argmax(dim=1)


def evidence_contributions(
    l2_spike: torch.Tensor,
    lengths: torch.Tensor,
    head_weight: torch.Tensor,
    y: torch.Tensor,
    competitor: torch.Tensor,
) -> torch.Tensor:
    counts = valid_sum(l2_spike, lengths)
    return counts * (head_weight[y] - head_weight[competitor])


def evidence_effective_neurons(contribution: torch.Tensor) -> torch.Tensor:
    positive = contribution.clamp_min(0)
    p = positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return 1.0 / p.square().sum(dim=1).clamp_min(1e-12)


def top_fraction_mask(values: torch.Tensor, q: float) -> torch.Tensor:
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    k = max(1, int(math.ceil(values.shape[1] * q)))
    indices = values.topk(k, dim=1).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def evidence_challenge_mask(contribution: torch.Tensor, q: float) -> torch.Tensor:
    positive = contribution.clamp_min(0)
    return top_fraction_mask(positive, q) & (positive > 0)


def high_rate_mask(rate_ema: torch.Tensor, batch: int, q: float) -> torch.Tensor:
    k = max(1, int(math.ceil(rate_ema.numel() * q)))
    idx = rate_ema.topk(k).indices
    mask = torch.zeros((batch, rate_ema.numel()), dtype=torch.bool, device=rate_ema.device)
    mask[:, idx] = True
    return mask


def random_mask(batch: int, width: int, q: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    score = torch.rand((batch, width), generator=generator, device="cpu").to(device)
    return top_fraction_mask(score, q)


def matched_random_mask(
    contribution: torch.Tensor,
    target_mask: torch.Tensor,
    generator: torch.Generator,
    candidates: int = 16,
) -> torch.Tensor:
    """Random neuron identity with per-sample cardinality and evidence strength matched."""
    batch, width = contribution.shape
    positive = contribution.clamp_min(0)
    result = torch.zeros_like(target_mask)
    for i in range(batch):
        k = int(target_mask[i].sum().item())
        if k == 0:
            continue
        target = (positive[i] * target_mask[i]).sum()
        best_error = torch.tensor(float("inf"), device=contribution.device)
        best = target_mask[i]
        for _ in range(candidates):
            score = torch.rand(width, generator=generator, device="cpu").to(contribution.device)
            idx = score.topk(k).indices
            mask = torch.zeros(width, dtype=torch.bool, device=contribution.device)
            mask[idx] = True
            error = ((positive[i] * mask).sum() - target).abs()
            if bool(error < best_error):
                best_error = error
                best = mask
        result[i] = best
    return result

def challenged_logits(
    model: BenchmarkNet,
    l2_spike: torch.Tensor,
    lengths: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    kept = l2_spike * (~mask).to(l2_spike.dtype)[:, None, :]
    return model.head(valid_sum(kept, lengths)) / lengths[:, None]


def margin_vectors(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = scores.gather(1, y[:, None])
    margins = true - scores
    keep = torch.ones_like(margins, dtype=torch.bool)
    keep[torch.arange(len(y), device=y.device), y] = False
    return margins[keep].reshape(len(y), scores.shape[1] - 1)


def margin_consistency_loss(scores: torch.Tensor, y: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
    margins = F.normalize(margin_vectors(scores, y), dim=1, eps=1e-8)
    losses: list[torch.Tensor] = []
    for cls in y.unique(sorted=True):
        cls_mask = y == cls
        cls_users = users[cls_mask].unique(sorted=True)
        if len(cls_users) < 2:
            continue
        prototypes = []
        for user in cls_users:
            selected = cls_mask & (users == user)
            prototypes.append(F.normalize(margins[selected].mean(0), dim=0, eps=1e-8))
        stack = torch.stack(prototypes)
        center = F.normalize(stack.mean(0), dim=0, eps=1e-8)
        losses.append((1.0 - stack @ center).mean())
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def rex_loss(scores: torch.Tensor, y: torch.Tensor, users: torch.Tensor) -> torch.Tensor:
    risks = []
    for user in users.unique(sorted=True):
        selected = users == user
        if selected.any():
            risks.append(F.cross_entropy(scores[selected], y[selected]))
    return torch.stack(risks).var(unbiased=False) if len(risks) > 1 else scores.sum() * 0.0


def _strength_params(strength: str) -> float:
    table = {"weak": 0.05, "medium": 0.10, "strong": 0.15}
    if strength not in table:
        raise ValueError(strength)
    return table[strength]


def temporal_warp(x: torch.Tensor, lengths: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor speed warp that preserves the per-channel value alphabet."""
    out = x.clone()
    for i in range(len(x)):
        n = int(lengths[i].item())
        if n <= 1:
            continue
        source = (torch.arange(n, device=x.device, dtype=torch.float32) / scale[i].clamp_min(1e-6))
        source = source.round().long().clamp(0, n - 1)
        out[i, :n] = x[i, source]
    return out


def physical_augment(
    x: torch.Tensor,
    lengths: torch.Tensor,
    strength: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Schema-safe proxy for writer-speed nuisance: temporal stretch/compression only."""
    time_delta = _strength_params(strength)
    batch = len(x)
    scale = 1 + (2 * torch.rand(batch, generator=generator) - 1) * time_delta
    return temporal_warp(x, lengths, scale.to(x.device))

def matched_noise_augment(
    x: torch.Tensor,
    lengths: torch.Tensor,
    strength: str,
    generator: torch.Generator,
) -> torch.Tensor:
    probe_gen = torch.Generator().manual_seed(generator.initial_seed() + 991)
    physical = physical_augment(x, lengths, strength, probe_gen)
    valid = valid_mask(lengths, x.shape[1]).unsqueeze(-1)
    delta_rms = torch.sqrt((((physical - x) * valid).square().sum() / valid.sum().clamp_min(1)).clamp_min(1e-12))
    noise = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
    return torch.where(valid, x + noise * delta_rms, x)


def style_mix(
    x: torch.Tensor,
    y: torch.Tensor,
    users: torch.Tensor,
    lengths: torch.Tensor,
    probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    out = x.clone()
    for i in range(len(x)):
        candidates = torch.nonzero((y == y[i]) & (users != users[i]), as_tuple=False).flatten()
        if len(candidates) == 0 or float(torch.rand((), generator=generator)) >= probability:
            continue
        j = int(candidates[int(torch.randint(len(candidates), (), generator=generator))].item())
        ni, nj = int(lengths[i].item()), int(lengths[j].item())
        xi, xj = x[i, :ni], x[j, :nj]
        mu_i, sd_i = xi.mean(0), xi.std(0, unbiased=False).clamp_min(STYLE_EPS)
        mu_j, sd_j = xj.mean(0), xj.std(0, unbiased=False).clamp_min(STYLE_EPS)
        lam = float(torch.rand((), generator=generator))
        mu = lam * mu_i + (1 - lam) * mu_j
        sd = lam * sd_i + (1 - lam) * sd_j
        out[i, :ni] = (xi - mu_i) / sd_i * sd + mu
    return out


def _user_tensor(arrays: dict[str, np.ndarray], split: str) -> torch.Tensor:
    values = arrays[f"{split}_users"]
    names = sorted(set(values.tolist()))
    mapping = {name: i for i, name in enumerate(names)}
    return torch.tensor([mapping[v] for v in values.tolist()], dtype=torch.long)


def _task_batches(arrays: dict[str, np.ndarray], p: Protocol, seed: int) -> Iterable[tuple[torch.Tensor, ...]]:
    users = _user_tensor(arrays, "train")
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(arrays["train_x"]),
        torch.from_numpy(arrays["train_y"]),
        torch.from_numpy(arrays["train_lengths"]),
        users,
    )
    gen = torch.Generator().manual_seed(paired_seed(seed, "exp19:train"))
    return torch.utils.data.DataLoader(dataset, batch_size=p.batch_size, shuffle=True, generator=gen, num_workers=0)


def _aux_index_batches(
    arrays: dict[str, np.ndarray], p: Protocol, seed: int, epoch: int, steps: int
) -> list[np.ndarray]:
    labels = arrays["train_y"]
    users = arrays["train_users"]
    rng = np.random.default_rng(paired_seed(seed, f"exp19:aux:{epoch}"))
    eligible: dict[int, dict[str, np.ndarray]] = {}
    for cls in np.unique(labels).tolist():
        per_user: dict[str, np.ndarray] = {}
        for user in np.unique(users[labels == cls]).tolist():
            idx = np.flatnonzero((labels == cls) & (users == user))
            if len(idx) >= 2:
                per_user[str(user)] = idx
        if len(per_user) >= 4:
            eligible[int(cls)] = per_user
    if len(eligible) < 8:
        raise ValueError("Exp19.2 requires at least 8 classes with 4 users and 2 segments/user")
    classes = np.asarray(sorted(eligible), dtype=np.int64)
    out: list[np.ndarray] = []
    for _ in range(steps):
        chosen_classes = rng.choice(classes, size=8, replace=False)
        batch: list[int] = []
        for cls in chosen_classes.tolist():
            per_user = eligible[int(cls)]
            chosen_users = rng.choice(np.asarray(sorted(per_user), dtype=object), size=4, replace=False)
            for user in chosen_users.tolist():
                chosen = rng.choice(per_user[str(user)], size=2, replace=False)
                batch.extend(int(v) for v in chosen.tolist())
        out.append(np.asarray(batch, dtype=np.int64))
    return out


def _aux_batch(arrays: dict[str, np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    user_ids = _user_tensor(arrays, "train")
    idx = torch.from_numpy(indices)
    return (
        torch.from_numpy(arrays["train_x"])[idx],
        torch.from_numpy(arrays["train_y"])[idx],
        torch.from_numpy(arrays["train_lengths"])[idx],
        user_ids[idx],
    )


def _batch_loss(
    model: BenchmarkNet,
    spec: ExpSpec,
    x: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    users: torch.Tensor,
    epoch: int,
    p: Protocol,
    rate_ema: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    if spec.family == "19.3":
        if spec.variant == "physical":
            x = physical_augment(x, lengths, spec.strength or "medium", generator)
        elif spec.variant == "matched_noise":
            x = matched_noise_augment(x, lengths, spec.strength or "medium", generator)
        elif spec.variant == "style_mix":
            x = style_mix(x, y, users, lengths, float(spec.mix_prob or 0.5), generator)

    tr = model(x, lengths)
    scores = native_scores(tr, lengths)
    normal = F.cross_entropy(scores, y)
    diag: dict[str, float] = {}

    with torch.no_grad():
        counts = valid_sum(tr["spike"][-1], lengths)
        batch_rate = counts.sum(0) / lengths.sum().clamp_min(1)
        rate_ema = 0.95 * rate_ema + 0.05 * batch_rate

    if spec.family == "19.1":
        comp = true_competitor(scores, y)
        contrib = evidence_contributions(tr["spike"][-1], lengths, model.head.weight, y, comp).detach()
        q = float(spec.q or 0.2)
        if spec.variant == "evidence":
            mask = evidence_challenge_mask(contrib, q)
        elif spec.variant == "high_rate":
            mask = high_rate_mask(rate_ema, len(y), q)
        elif spec.variant == "random":
            mask = random_mask(len(y), p.width, q, generator, x.device)
        elif spec.variant == "evidence_matched_random":
            target = evidence_challenge_mask(contrib, q)
            mask = matched_random_mask(contrib, target, generator)
        else:
            raise ValueError(spec.variant)
        challenged = challenged_logits(model, tr["spike"][-1], lengths, mask)
        aux = F.cross_entropy(challenged, y)
        w = challenge_weight(epoch, p.max_epochs, float(spec.weight or 1.0))
        positive = contrib.clamp_min(0)
        diag = {
            "challenge_weight": w,
            "effective_evidence_neurons": float(evidence_effective_neurons(contrib).mean()),
            "removed_positive_margin_fraction": float(
                ((positive * mask).sum(1) / positive.sum(1).clamp_min(1e-12)).mean()
            ),
            "masked_fraction": float(mask.float().mean()),
        }
        return normal + w * aux, diag, rate_ema

    if spec.family == "19.2":
        return normal, diag, rate_ema

    return normal, diag, rate_ema


def train_one(config: Config, spec: ExpSpec) -> dict[str, Any]:
    p, lock, arrays = _core(config)
    directory = config.results_dir / "runs" / spec.key
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        if payload.get("core_identity") == lock["identity"] and payload.get("source_hash") == file_hash(Path(__file__).resolve()):
            return payload

    model = BenchmarkNet(_run(spec), p)
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    best = evaluate_validation(model, arrays, p, spec.seed)
    best_state, best_epoch = cpu_state(model), 0
    history: list[dict[str, Any]] = []
    rate_ema = torch.zeros(p.width)
    generator = torch.Generator().manual_seed(paired_seed(spec.seed, f"exp19:{spec.key}"))
    task_loader = _task_batches(arrays, p, spec.seed)

    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        diagnostics: list[dict[str, float]] = []
        aux_indices = _aux_index_batches(arrays, p, spec.seed, epoch, len(task_loader)) if spec.family == "19.2" else []
        for step, (x, y, lengths, users) in enumerate(task_loader):
            optimizer.zero_grad(set_to_none=True)
            loss, diag, rate_ema = _batch_loss(model, spec, x, y, lengths, users, epoch, p, rate_ema, generator)
            if spec.family == "19.2":
                ax, ay, alengths, ausers = _aux_batch(arrays, aux_indices[step])
                aux_scores = native_scores(model(ax, alengths), alengths)
                aux = (
                    margin_consistency_loss(aux_scores, ay, ausers)
                    if spec.variant == "margin_consistency"
                    else rex_loss(aux_scores, ay, ausers)
                )
                loss = loss + float(spec.weight or 0.0) * aux
                diag = {"aux_loss": float(aux.detach())}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{spec.key}: nonfinite loss")
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError(f"{spec.key}: nonfinite gradient")
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
            diagnostics.append(diag)
        val = evaluate_validation(model, arrays, p, spec.seed)
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "val_ba": val["ba"], "val_mean_logit_ce": val["mean_logit_ce"]}
        keys = sorted(set().union(*(d.keys() for d in diagnostics))) if diagnostics else []
        row.update({k: float(np.mean([d[k] for d in diagnostics if k in d])) for k in keys})
        history.append(row)
        improved = val["ba"] > best["ba"] + 1e-12 or (
            abs(val["ba"] - best["ba"]) <= 1e-12 and val["mean_logit_ce"] < best["mean_logit_ce"] - 1e-12
        )
        if improved:
            best, best_state, best_epoch = val, cpu_state(model), epoch
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break

    model.load_state_dict(best_state)
    split_metrics: dict[str, Any] = {}
    traces: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for split in ("train", "val", "test"):
            ys, preds = [], []
            chunks: dict[str, list[np.ndarray]] = {}
            for layer in ("L1", "L2"):
                for state in ("spike", "pre_reset"):
                    chunks[f"{layer}__{state}"] = []
            for x, y, lengths in loader(arrays, split, p, spec.seed):
                trajectory = model(x, lengths)
                score = native_scores(trajectory, lengths)
                ys.append(y.numpy())
                preds.append(score.argmax(1).numpy())
                for li, layer in enumerate(("L1", "L2")):
                    chunks[f"{layer}__spike"].append(trajectory["spike"][li].numpy().astype(np.uint8))
                    chunks[f"{layer}__pre_reset"].append(trajectory["pre_reset"][li].numpy().astype(np.float32))
            y_all = np.concatenate(ys)
            pred_all = np.concatenate(preds)
            split_metrics[split] = metrics(y_all, pred_all)
            recalls = {}
            for cls in np.unique(y_all).tolist():
                selected = y_all == cls
                recalls[str(int(cls))] = float((pred_all[selected] == cls).mean())
            split_metrics[split]["per_class_recall"] = recalls
            split_metrics[split]["recall_std"] = float(np.std(list(recalls.values())))
            split_metrics[split]["worst_class_recall"] = float(min(recalls.values()))
            for key, values in chunks.items():
                traces[f"{split}__{key}"] = np.concatenate(values, axis=0)

    probe_rows = run_probes(directory, traces, arrays, _run(spec), p)
    probe_summary = [
        row for row in probe_rows
        if row["layer"] == "L2" and row["state"] == "spike" and row["decoder"] == "no_bias"
    ]

    source_hash = file_hash(Path(__file__).resolve())
    checkpoint = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "source_hash": source_hash,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_val": best,
        "selection_rule": "native validation BA, then native validation mean-logit CE, then earliest epoch",
    }
    save_torch(directory / "checkpoint.pt", checkpoint)
    save_json(directory / "history.json", {"rows": history})
    payload = {
        "spec": asdict(spec),
        "key": spec.key,
        "best_epoch": best_epoch,
        "splits": split_metrics,
        "train_test_gap": split_metrics["train"]["ba"] - split_metrics["test"]["ba"],
        "core_identity": lock["identity"],
        "source_hash": source_hash,
        "probe_summary": probe_summary,
    }
    save_json(complete, payload)
    return payload


def prepare(config: Config) -> dict[str, Any]:
    p, lock, _ = _core(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    calibration = [asdict(s) | {"key": s.key} for s in calibration_specs()]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "core_identity": lock["identity"],
        "core_dataset_hash": lock["dataset_hash"],
        "seeds": list(SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "calibration_tasks": calibration,
        "calibration_count": len(calibration),
        "expected_calibration_count": 20,
        "final_count": 27,
        "reference_count": 3,
        "execution": "one independent config x seed per one-core Slurm array task",
        "input_augmentation_scope": "locked 30-channel pre-SNN CoreBenchmark input; structured 19.3-P uses nearest-neighbor writing-speed warp only; raw-sensor geometry is intentionally not guessed",
        "references": {
            str(seed): {
                "native_hash": file_hash(config.core_results_dir / "runs" / f"O0__seed{seed}" / "native.json"),
                "checkpoint_hash": file_hash(config.core_results_dir / "runs" / f"O0__seed{seed}" / "checkpoint.pt"),
            }
            for seed in SEEDS
        },
        "selection_rule": "validation native BA only; ties prefer weaker intervention",
        "width": p.width,
        "fs_hz": p.fs,
    }
    save_json(config.results_dir / "protocol.json", payload)
    return payload


def select(config: Config) -> dict[str, Any]:
    rows = []
    for spec in calibration_specs():
        path = config.results_dir / "runs" / spec.key / "complete.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append((spec, float(payload["splits"]["val"]["ba"])))

    def best(candidates: list[tuple[ExpSpec, float]], simplicity: Any) -> ExpSpec:
        return max(candidates, key=lambda item: (item[1], -float(simplicity(item[0]))))[0]

    e = best([(s, v) for s, v in rows if s.family == "19.1"], lambda s: (s.q or 0) + (s.weight or 0))
    m = best([(s, v) for s, v in rows if s.family == "19.2" and s.variant == "margin_consistency"], lambda s: s.weight or 0)
    r = best([(s, v) for s, v in rows if s.family == "19.2" and s.variant == "rex"], lambda s: s.weight or 0)
    p = best([(s, v) for s, v in rows if s.family == "19.3" and s.variant == "physical"], lambda s: PHYSICAL_STRENGTHS.index(s.strength or "weak"))
    sm = best([(s, v) for s, v in rows if s.family == "19.3" and s.variant == "style_mix"], lambda s: s.mix_prob or 0)
    selection = {
        "19.1": {"q": e.q, "weight": e.weight},
        "19.2": {"margin_weight": m.weight, "rex_weight": r.weight},
        "19.3": {"strength": p.strength, "mix_prob": sm.mix_prob},
    }
    save_json(config.results_dir / "selected_config.json", selection)
    return selection


def finalize(config: Config) -> dict[str, Any]:
    selection = json.loads((config.results_dir / "selected_config.json").read_text(encoding="utf-8"))
    rows = []
    for seed in SEEDS:
        native_path = config.core_results_dir / "runs" / f"O0__seed{seed}" / "native.json"
        if not native_path.exists():
            raise FileNotFoundError(native_path)
        native = json.loads(native_path.read_text(encoding="utf-8"))
        test = native["splits"]["test"]
        rows.append({
            "family": "19.0",
            "variant": "core_wcce_reference",
            "seed": seed,
            "ba": float(test["ba"]),
            "accuracy": float(test["accuracy"]),
            "macro_f1": float(test["macro_f1"]),
            "train_test_gap": float(native["train_test_gap"]),
        })
    for spec in final_specs(selection):
        path = config.results_dir / "runs" / spec.key / "complete.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "family": spec.family,
            "variant": spec.variant,
            "seed": spec.seed,
            **payload["splits"]["test"],
            "train_test_gap": payload["train_test_gap"],
        })
    frame = pd.DataFrame(rows)
    agg_dir = config.results_dir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(agg_dir / "run_summary.csv", index=False)
    summary = frame.groupby(["family", "variant"], as_index=False).agg(
        test_ba_mean=("ba", "mean"),
        test_ba_std=("ba", "std"),
        train_test_gap_mean=("train_test_gap", "mean"),
    )
    summary.to_csv(agg_dir / "case_summary.csv", index=False)
    payload = {"rows": rows, "selection": selection}
    save_json(agg_dir / "summary.json", payload)
    return payload


def smoke(config: Config) -> dict[str, Any]:
    p, lock, arrays = _core(config)
    x, y, lengths = next(iter(loader(arrays, "train", p, SEEDS[0])))
    users_all = _user_tensor(arrays, "train")
    users = users_all[: len(y)]
    representatives = (
        ExpSpec("19.1", "evidence", SEEDS[0], q=0.2, weight=0.5),
        ExpSpec("19.2", "margin_consistency", SEEDS[0], weight=0.05),
        ExpSpec("19.2", "rex", SEEDS[0], weight=0.5),
        ExpSpec("19.3", "physical", SEEDS[0], strength="medium"),
        ExpSpec("19.3", "style_mix", SEEDS[0], mix_prob=0.5),
    )
    rows = []
    for spec in representatives:
        model = BenchmarkNet(_run(spec), p)
        optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
        generator = torch.Generator().manual_seed(paired_seed(spec.seed, f"exp19:smoke:{spec.variant}"))
        rate_ema = torch.zeros(p.width)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = _batch_loss(model, spec, x, y, lengths, users, max(1, p.max_epochs // 2), p, rate_ema, generator)
        if spec.family == "19.2":
            aux_idx = _aux_index_batches(arrays, p, spec.seed, 1, 1)[0]
            ax, ay, alengths, ausers = _aux_batch(arrays, aux_idx)
            aux_scores = native_scores(model(ax, alengths), alengths)
            aux = margin_consistency_loss(aux_scores, ay, ausers) if spec.variant == "margin_consistency" else rex_loss(aux_scores, ay, ausers)
            loss = loss + float(spec.weight or 0.0) * aux
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"smoke nonfinite: {spec.variant}")
        rows.append({"variant": spec.variant, "loss": float(loss.detach())})
    return {"core_identity": lock["identity"], "cases": rows}

def _spec_from_args(args: argparse.Namespace) -> ExpSpec:
    return ExpSpec(
        args.family,
        args.variant,
        args.seed,
        q=args.q,
        weight=args.weight,
        strength=args.strength,
        mix_prob=args.mix_prob,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "run-calibration-index", "run-final-index", "select", "finalize", "smoke"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--core-results", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--q", type=float)
    parser.add_argument("--weight", type=float)
    parser.add_argument("--strength")
    parser.add_argument("--mix-prob", type=float)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    config = config_from_args(args)
    if args.command == "smoke":
        print(json.dumps(smoke(config), indent=2))
        return
    if args.command == "prepare":
        print(json.dumps(prepare(config), indent=2))
    elif args.command == "run":
        if args.family is None or args.variant is None or args.seed is None:
            parser.error("run requires --family --variant --seed")
        print(json.dumps(train_one(config, _spec_from_args(args)), indent=2))
    elif args.command == "run-calibration-index":
        if args.index is None:
            parser.error("run-calibration-index requires --index")
        specs = calibration_specs()
        if not 0 <= args.index < len(specs):
            parser.error("calibration index out of range")
        print(json.dumps(train_one(config, specs[args.index]), indent=2))
    elif args.command == "run-final-index":
        if args.index is None:
            parser.error("run-final-index requires --index")
        selection = json.loads((config.results_dir / "selected_config.json").read_text(encoding="utf-8"))
        specs = final_specs(selection)
        if not 0 <= args.index < len(specs):
            parser.error("final index out of range")
        print(json.dumps(train_one(config, specs[args.index]), indent=2))
    elif args.command == "select":
        print(json.dumps(select(config), indent=2))
    else:
        print(json.dumps(finalize(config), indent=2))


if __name__ == "__main__":
    main()

[executed on device: acd20ea31325 (425a23ad-a806-44e3-abed-ce7b563d3969)]