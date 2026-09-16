from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts import experiment_7_3_training_strategy_decomposition as exp73


EXPERIMENT_ID = "experiment_7_3_6_affine_mean_head_bridge"
PROTOCOL_VERSION = "affine_mean_head_bridge_v1"
ARCHITECTURE = exp73.ARCHITECTURE
SEEDS = exp73.SEEDS
BACKBONE_OBJECTIVE = "wcce"
OBJECTIVE = "wcce"
CASES = (
    "H0_raw_no_bias",
    "H1_raw_bias",
    "H2_scale_no_bias",
    "H3_scale_bias",
)
LIF_BETA = exp73.LIF_BETA
THRESHOLD = exp73.THRESHOLD
OUTPUT_CAP = exp73.OUTPUT_CAP
MAX_EPOCHS = exp73.MAX_EPOCHS
MIN_EPOCHS = exp73.MIN_EPOCHS
PATIENCE = exp73.PATIENCE
EPS = 1e-8
CHARGE_TOL = 1e-3


@dataclass(frozen=True)
class RunSpec:
    seed: int
    case: str

    @property
    def key(self) -> str:
        return f"{ARCHITECTURE}__{self.case}__seed{self.seed}"

    @property
    def backbone(self) -> exp73.BackboneSpec:
        return exp73.BackboneSpec(self.seed, BACKBONE_OBJECTIVE)

    @property
    def use_scale(self) -> bool:
        return self.case in {"H2_scale_no_bias", "H3_scale_bias"}

    @property
    def use_bias(self) -> bool:
        return self.case in {"H1_raw_bias", "H3_scale_bias"}


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    batch_size: int = exp73.exp72.BATCH_SIZE
    threads: int = 1
    max_epochs: int = MAX_EPOCHS


def find_repo_root(start: Path | None = None) -> Path:
    return exp73.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def run_specs() -> list[RunSpec]:
    return [RunSpec(seed, case) for seed in SEEDS for case in CASES]


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp73_config(config: Config) -> exp73.Config:
    return exp73.Config(
        repo_root=config.repo_root,
        results_dir=exp73.results_dir(config.repo_root),
        device=config.device,
        batch_size=config.batch_size,
        threads=config.threads,
        max_epochs=config.max_epochs,
    )


def _load_cache(config: Config, spec: RunSpec) -> dict[str, Any]:
    cache = exp73._load_stage2_cache(spec.backbone, _exp73_config(config))
    meta = cache["metadata"]
    if meta.get("source_method") != "A2_e2e_linear_wcce":
        raise RuntimeError(
            f"{spec.key}: expected A2_e2e_linear_wcce cache, got {meta.get('source_method')}"
        )
    if not bool(meta.get("frozen_l1_l2")):
        raise RuntimeError(f"{spec.key}: Exp7.3 cache is not marked frozen")
    return cache


def _valid_mean_l2(l2: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    mask = exp73._valid_mask(lengths, l2.shape[1]).to(l2.dtype).unsqueeze(-1)
    return (l2 * mask).sum(dim=1) / lengths.clamp_min(1).to(l2.dtype).unsqueeze(1)


def _train_sigma(cache: dict[str, Any]) -> np.ndarray:
    l2_np, _, lengths_np = cache["train"]
    l2 = torch.from_numpy(l2_np).to(torch.float32)
    lengths = torch.from_numpy(lengths_np).to(torch.long)
    mean = _valid_mean_l2(l2, lengths)
    sigma = mean.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    sigma[~np.isfinite(sigma)] = 1.0
    sigma[sigma < EPS] = 1.0
    return sigma


class AffineHead(nn.Module):
    def __init__(self, n_features: int, n_classes: int, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_head(
    spec: RunSpec,
    n_features: int,
    n_classes: int,
    sigma: np.ndarray,
    device: torch.device,
) -> tuple[AffineHead, torch.Tensor]:
    exp73.exp3.seed_all(exp73._stage2_pair_seed(spec.seed, "model_init"))
    raw_reference = nn.Linear(n_features, n_classes, bias=False)
    initial_raw_w = raw_reference.weight.detach().cpu().clone()

    exp73.exp3.seed_all(exp73._stage2_pair_seed(spec.seed, "model_init"))
    head = AffineHead(n_features, n_classes, bias=spec.use_bias).to(device)
    with torch.no_grad():
        if spec.use_scale:
            scale = torch.from_numpy(sigma).to(device=device, dtype=head.linear.weight.dtype)
            head.linear.weight.copy_(initial_raw_w.to(device) * scale.unsqueeze(0))
        else:
            head.linear.weight.copy_(initial_raw_w.to(device))
        if head.linear.bias is not None:
            head.linear.bias.zero_()
    return head, initial_raw_w


def _scaled_mean(
    l2: torch.Tensor,
    lengths: torch.Tensor,
    sigma: torch.Tensor,
    use_scale: bool,
) -> torch.Tensor:
    mean = _valid_mean_l2(l2, lengths)
    return mean / sigma if use_scale else mean


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return exp73._metrics(y, scores)


def _evaluate_analog(
    head: AffineHead,
    loader: Iterable,
    device: torch.device,
    sigma: torch.Tensor,
    use_scale: bool,
    bias_enabled: bool = True,
) -> dict[str, float]:
    ys: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    head.eval()
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            x = _scaled_mean(l2, lengths, sigma, use_scale)
            scores = F.linear(
                x,
                head.linear.weight,
                head.linear.bias if bias_enabled else None,
            )
            loss = F.cross_entropy(exp73.CE_GAIN * scores, y)
            ys.append(y.cpu().numpy())
            scores_all.append(scores.cpu().numpy())
            loss_sum += float(loss) * len(y)
            n_total += len(y)
    out = _metrics(np.concatenate(ys), np.concatenate(scores_all))
    out["objective_loss"] = loss_sum / max(n_total, 1)
    return out


def _raw_parameters(
    head: AffineHead,
    sigma: np.ndarray,
    use_scale: bool,
) -> tuple[np.ndarray, np.ndarray]:
    w = head.linear.weight.detach().cpu().numpy().astype(np.float32)
    if use_scale:
        w = w / sigma[None, :]
    if head.linear.bias is None:
        b = np.zeros(w.shape[0], dtype=np.float32)
    else:
        b = head.linear.bias.detach().cpu().numpy().astype(np.float32)
    return w, b


def _simulate_lif_repeated_bias(
    l2: np.ndarray,
    lengths: np.ndarray,
    w_raw: np.ndarray,
    bias: np.ndarray,
    beta: float,
) -> np.ndarray:
    x = torch.from_numpy(l2).to(torch.float32)
    w = torch.from_numpy(w_raw).to(torch.float32)
    b = torch.from_numpy(bias).to(torch.float32)
    lengths_t = torch.from_numpy(lengths).to(torch.long)
    batch, steps, _ = x.shape
    classes = w.shape[0]
    mem = torch.zeros(batch, classes, dtype=torch.float32)
    counts = torch.zeros_like(mem)
    for t in range(steps):
        valid = (t < lengths_t).to(torch.float32).unsqueeze(1)
        current = F.linear(x[:, t], w, b)
        pre = beta * mem + current
        spikes = (pre >= THRESHOLD).to(torch.float32)
        if OUTPUT_CAP == 1:
            spikes = spikes.clamp_max(1.0)
        spikes = spikes * valid
        mem = torch.where(valid.bool(), pre - spikes * THRESHOLD, mem)
        counts += spikes
    return counts.numpy()


def _simulate_if_charge(
    l2: np.ndarray,
    lengths: np.ndarray,
    w_raw: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = torch.from_numpy(l2).to(torch.float32)
    w = torch.from_numpy(w_raw).to(torch.float32)
    b = torch.from_numpy(bias).to(torch.float32)
    lengths_t = torch.from_numpy(lengths).to(torch.long)
    batch, steps, _ = x.shape
    classes = w.shape[0]
    mem = torch.zeros(batch, classes, dtype=torch.float32)
    counts = torch.zeros_like(mem)
    analog_sum = torch.zeros_like(mem)
    for t in range(steps):
        valid = (t < lengths_t).to(torch.float32).unsqueeze(1)
        current = F.linear(x[:, t], w, b)
        analog_sum += current * valid
        pre = mem + current
        spikes = (pre >= THRESHOLD).to(torch.float32).clamp_max(1.0) * valid
        mem = torch.where(valid.bool(), pre - spikes * THRESHOLD, mem)
        counts += spikes
    charge = THRESHOLD * counts + mem
    error = float(torch.max(torch.abs(charge - analog_sum)))
    return counts.numpy(), charge.numpy(), error


def _evaluate_realizations(
    cache: dict[str, Any],
    w_raw: np.ndarray,
    bias: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    max_charge_error = 0.0
    for split in ("train", "val", "test"):
        l2, y, lengths = cache[split]
        l2f = l2.astype(np.float32, copy=False)
        mean = _valid_mean_l2(
            torch.from_numpy(l2f), torch.from_numpy(lengths).to(torch.long)
        ).numpy()
        analog = mean @ w_raw.T + bias[None, :]
        lif = _simulate_lif_repeated_bias(l2f, lengths, w_raw, bias, LIF_BETA)
        if_counts, if_charge, charge_error = _simulate_if_charge(
            l2f, lengths, w_raw, bias
        )
        max_charge_error = max(max_charge_error, charge_error)
        result[split] = {
            "analog_mean_affine": _metrics(y, analog),
            "lif_beta05_repeated_bias": _metrics(y, lif),
            "if_beta1_count_repeated_bias": _metrics(y, if_counts),
            "if_beta1_charge_repeated_bias": _metrics(y, if_charge),
        }
    result["max_if_charge_identity_abs_error"] = max_charge_error
    return result


def _head_geometry(
    head: AffineHead,
    initial_raw_w: torch.Tensor,
    sigma: np.ndarray,
    use_scale: bool,
) -> dict[str, float]:
    w_raw, b = _raw_parameters(head, sigma, use_scale)
    w = torch.from_numpy(w_raw).double()
    w0 = initial_raw_w.double()
    denom = torch.linalg.vector_norm(w, dim=1) * torch.linalg.vector_norm(w0, dim=1)
    cos = torch.where(
        denom > 1e-12,
        (w * w0).sum(dim=1) / denom.clamp_min(1e-12),
        torch.ones_like(denom),
    )
    return {
        "weight_frobenius_norm": float(torch.linalg.vector_norm(w)),
        "bias_l2_norm": float(np.linalg.norm(b)),
        "bias_span": float(np.max(b) - np.min(b)),
        "mean_row_cosine_to_init": float(cos.mean()),
        "min_row_cosine_to_init": float(cos.min()),
    }


def run_one(spec: RunSpec, config: Config, force: bool = False) -> dict[str, Any]:
    if spec.seed not in SEEDS or spec.case not in CASES:
        raise ValueError(spec)
    evaluation_path = config.results_dir / "evaluations" / f"{spec.key}.json"
    checkpoint_path = config.results_dir / "checkpoints" / f"{spec.key}.pt"
    history_path = config.results_dir / "histories" / f"{spec.key}.csv"
    if evaluation_path.exists() and checkpoint_path.exists() and history_path.exists() and not force:
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    cache = _load_cache(config, spec)
    n_features = int(cache["train"][0].shape[-1])
    n_classes = int(np.max(cache["train"][1])) + 1
    sigma_np = _train_sigma(cache)
    sigma = torch.from_numpy(sigma_np).to(config.device)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    head, initial_raw_w = _make_head(spec, n_features, n_classes, sigma_np, device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=exp73.exp72.LR, weight_decay=exp73.exp72.WEIGHT_DECAY
    )
    train_loader = exp73._cached_loaders(cache, spec.seed, config.batch_size, True)["train"]
    eval_loaders = exp73._cached_loaders(cache, spec.seed, config.batch_size, False)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ba = -1.0
    best_loss = float("inf")
    stopped_epoch = config.max_epochs
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        head.train()
        train_loss_sum = 0.0
        n_total = 0
        for l2, y, lengths in train_loader:
            l2 = l2.to(device=device, dtype=torch.float32)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            x = _scaled_mean(l2, lengths, sigma, spec.use_scale)
            scores = head(x)
            loss = F.cross_entropy(exp73.CE_GAIN * scores, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            n_total += len(y)

        train_eval = _evaluate_analog(
            head, eval_loaders["train"], device, sigma, spec.use_scale
        )
        val_eval = _evaluate_analog(
            head, eval_loaders["val"], device, sigma, spec.use_scale
        )
        geometry = _head_geometry(head, initial_raw_w, sigma_np, spec.use_scale)
        history.append(
            {
                "epoch": epoch,
                "optimization_train_loss": train_loss_sum / max(n_total, 1),
                "train_ba": train_eval["balanced_accuracy"],
                "val_ba": val_eval["balanced_accuracy"],
                "train_loss": train_eval["objective_loss"],
                "val_loss": val_eval["objective_loss"],
                **geometry,
            }
        )
        if exp73._checkpoint_improved(val_eval, best_ba, best_loss):
            best_ba = float(val_eval["balanced_accuracy"])
            best_loss = float(val_eval["objective_loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch >= MIN_EPOCHS and best_epoch > 0 and epoch - best_epoch >= PATIENCE:
            stopped_epoch = epoch
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    head.load_state_dict(best_state)

    analog = {
        split: _evaluate_analog(head, loader, device, sigma, spec.use_scale)
        for split, loader in eval_loaders.items()
    }
    bias_off = {
        split: _evaluate_analog(
            head, loader, device, sigma, spec.use_scale, bias_enabled=False
        )
        for split, loader in eval_loaders.items()
    }
    w_raw, b_raw = _raw_parameters(head, sigma_np, spec.use_scale)
    realizations = _evaluate_realizations(cache, w_raw, b_raw)
    geometry = _head_geometry(head, initial_raw_w, sigma_np, spec.use_scale)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": asdict(spec),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "head_state_dict": best_state,
            "sigma": sigma_np,
            "raw_weight": w_raw,
            "raw_bias": b_raw,
            "selection_rule": "validation analog BA primary, validation CE loss tiebreak",
        },
        checkpoint_path,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": asdict(spec),
        "contract": {
            "architecture": ARCHITECTURE,
            "backbone_source": "A2_e2e_linear_wcce",
            "frozen_l1_l2": True,
            "objective": OBJECTIVE,
            "pooling": "valid_mean",
            "use_scale": spec.use_scale,
            "use_bias": spec.use_bias,
            "scaler": "train-only per-feature std; no centering" if spec.use_scale else "none",
            "scale_folded_into_raw_weight": True,
            "lif_beta": LIF_BETA,
            "threshold": THRESHOLD,
            "output_cap": OUTPUT_CAP,
            "paired_model_init_seed": exp73._stage2_pair_seed(spec.seed, "model_init"),
            "paired_loader_order": True,
            "primary_checkpoint": "best validation analog BA",
            "test_evaluated_after_selection": True,
        },
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "analog_metrics": analog,
        "same_w_bias_off_metrics": bias_off,
        "realization_metrics": realizations,
        "geometry": geometry,
    }
    _save_json(evaluation_path, payload)
    return payload


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    spec = payload["spec"]
    analog = payload["analog_metrics"]
    bias_off = payload["same_w_bias_off_metrics"]
    real = payload["realization_metrics"]
    g = payload["geometry"]
    return {
        "method": spec["case"],
        "seed": int(spec["seed"]),
        "use_scale": bool(payload["contract"]["use_scale"]),
        "use_bias": bool(payload["contract"]["use_bias"]),
        "best_epoch": int(payload["best_epoch"]),
        "analog_test_ba": float(analog["test"]["balanced_accuracy"]),
        "bias_off_test_ba": float(bias_off["test"]["balanced_accuracy"]),
        "lif_test_ba": float(real["test"]["lif_beta05_repeated_bias"]["balanced_accuracy"]),
        "if_count_test_ba": float(real["test"]["if_beta1_count_repeated_bias"]["balanced_accuracy"]),
        "if_charge_test_ba": float(real["test"]["if_beta1_charge_repeated_bias"]["balanced_accuracy"]),
        "if_charge_max_abs_error": float(real["max_if_charge_identity_abs_error"]),
        **g,
    }


def _contrast_rows(runs: pd.DataFrame) -> pd.DataFrame:
    defs = (
        ("bias_raw_H1_minus_H0", "H1_raw_bias", "H0_raw_no_bias"),
        ("scale_no_bias_H2_minus_H0", "H2_scale_no_bias", "H0_raw_no_bias"),
        ("bias_scaled_H3_minus_H2", "H3_scale_bias", "H2_scale_no_bias"),
        ("full_H3_minus_H0", "H3_scale_bias", "H0_raw_no_bias"),
    )
    rows: list[dict[str, Any]] = []
    for name, left, right in defs:
        merged = runs[runs.method == left][["seed", "analog_test_ba"]].merge(
            runs[runs.method == right][["seed", "analog_test_ba"]],
            on="seed",
            suffixes=("_left", "_right"),
        )
        if set(merged.seed.astype(int)) != set(SEEDS):
            raise RuntimeError(f"Incomplete contrast {name}")
        for r in merged.itertuples(index=False):
            rows.append(
                {
                    "contrast": name,
                    "left_method": left,
                    "right_method": right,
                    "seed": int(r.seed),
                    "delta_pp": 100.0
                    * (float(r.analog_test_ba_left) - float(r.analog_test_ba_right)),
                }
            )
    return pd.DataFrame(rows)


def _probe_references(config: Config) -> dict[str, float | None]:
    path = (
        config.repo_root
        / "notebooks/artifacts/experiment_7_3_2_affine_probe_bridge"
        / "affine_probe_bridge_v1/method_summary.csv"
    )
    if not path.exists():
        return {"p3_mean_affine_test_ba": None, "p7_wholecount_affine_test_ba": None}
    df = pd.read_csv(path)
    wc = df[df["backbone_objective"] == "wcce"]

    def get(case: str) -> float | None:
        row = wc[wc["case"].str.startswith(case)]
        return None if row.empty else float(row.iloc[0]["test_ba_mean"])

    return {
        "p3_mean_affine_test_ba": get("P3_"),
        "p7_wholecount_affine_test_ba": get("P7_"),
    }


def finalize(config: Config) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    missing: list[Path] = []
    for spec in run_specs():
        path = config.results_dir / "evaluations" / f"{spec.key}.json"
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Missing Exp7.3.6 evaluations:\n" + "\n".join(str(p) for p in missing[:12])
        )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame([_row(p) for p in payloads]).sort_values(["method", "seed"])
    if len(runs) != len(CASES) * len(SEEDS):
        raise RuntimeError(f"Expected 12 rows, got {len(runs)}")
    runs.to_csv(config.results_dir / "method_runs.csv", index=False)

    numeric = [c for c in runs.columns if c not in {"method", "seed", "use_scale", "use_bias"}]
    summary = runs.groupby(["method", "use_scale", "use_bias"], sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(config.results_dir / "method_summary.csv", index=False)

    contrasts = _contrast_rows(runs)
    contrasts.to_csv(config.results_dir / "contrast_runs.csv", index=False)
    contrast_summary = contrasts.groupby(["contrast", "left_method", "right_method"])["delta_pp"].agg(["count", "mean", "std"]).reset_index()
    contrast_summary.columns = ["contrast", "left_method", "right_method", "delta_pp_count", "delta_pp_mean", "delta_pp_std"]
    contrast_summary.to_csv(config.results_dir / "contrast_summary.csv", index=False)

    bias_rows = runs[runs.use_bias].copy()
    bias_rows["direct_bias_effect_pp"] = 100.0 * (
        bias_rows["analog_test_ba"] - bias_rows["bias_off_test_ba"]
    )
    bias_rows.to_csv(config.results_dir / "bias_ablation.csv", index=False)

    realization_cols = [
        "method",
        "seed",
        "analog_test_ba",
        "lif_test_ba",
        "if_count_test_ba",
        "if_charge_test_ba",
        "if_charge_max_abs_error",
    ]
    runs[realization_cols].to_csv(
        config.results_dir / "lif_realization_summary.csv", index=False
    )

    histories: list[pd.DataFrame] = []
    for spec in run_specs():
        path = config.results_dir / "histories" / f"{spec.key}.csv"
        df = pd.read_csv(path)
        df.insert(0, "seed", spec.seed)
        df.insert(1, "method", spec.case)
        histories.append(df)
    history_runs = pd.concat(histories, ignore_index=True)
    history_runs.to_csv(config.results_dir / "history_runs.csv", index=False)
    history_numeric = [c for c in history_runs.columns if c not in {"seed", "method", "epoch"}]
    hs = history_runs.groupby(["method", "epoch"], sort=False)[history_numeric].agg(["mean", "std", "count"])
    hs.columns = [f"{name}_{stat}" for name, stat in hs.columns]
    hs.reset_index().to_csv(config.results_dir / "history_summary.csv", index=False)

    refs = _probe_references(config)
    source_checks = {
        "experiment": EXPERIMENT_ID,
        "h0_is_b6_formulation": True,
        "b6_reference_note": "Exp7.3 B6 uses the same frozen A2/WCCE backbone, valid-mean WCCE, bias-free head, paired init/order.",
        **refs,
    }
    _save_json(config.results_dir / "source_reproduction_checks.json", source_checks)

    max_charge_error = float(runs["if_charge_max_abs_error"].max())
    if max_charge_error > CHARGE_TOL:
        raise RuntimeError(
            f"IF charge identity error {max_charge_error} exceeds tolerance {CHARGE_TOL}"
        )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "logical_training_runs": len(run_specs()),
        "frozen_backbone": "Exp7.3 A2_e2e_linear_wcce L2 cache",
        "pooling": "valid mean for all training cases",
        "primary_metric": "analog test BA from validation-BA-selected checkpoint",
        "multi_cpu_contract": "one CPU per seed/case task; afterok finalizer aggregates only",
        "notebook_contract": "analysis-only; reads finalized CSV/JSON artifacts",
        "scale_contract": "train-only std scaling, no centering; folded into raw W",
        "bias_contract": "H1/H3 train one class bias; repeated once per valid timestep for LIF realization",
        "max_if_charge_identity_abs_error": max_charge_error,
        **refs,
    }
    _save_json(config.results_dir / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp7.3.6 affine mean-head bridge")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=exp73.exp72.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--array-task-id", type=int, required=True)
    run.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    config = Config(
        repo_root=root,
        results_dir=results_dir(root),
        device=args.device,
        batch_size=args.batch_size,
        threads=args.threads,
        max_epochs=args.max_epochs,
    )
    if args.command == "run":
        specs = run_specs()
        if args.array_task_id < 0 or args.array_task_id >= len(specs):
            raise IndexError(args.array_task_id)
        run_one(specs[args.array_task_id], config, force=args.force)
    elif args.command == "finalize":
        finalize(config)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
