from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts import experiment_5_4_2_phase_conditioned_readout as exp542


exp541 = exp542.exp541
exp54 = exp542.exp54
base = exp542.base

EXPERIMENT_ID = "experiment_5_4_3_elapsed_readout_capacity"
PROTOCOL_VERSION = "elapsed_readout_capacity_v1"
SEEDS = exp542.SEEDS
WHAT_WIDTH = exp542.WHAT_WIDTH
WHEN_WIDTH = exp542.WHEN_WIDTH
N_CLASSES = 12
FIXED250_SECONDS = 0.250
CAPACITY_BANKS = (4, 8, 16)
CAPACITY_RANKS = (4, 8, 12)
EXTENDED_BANKS = (20, 24, 32)
FULL_MATRIX_BANKS = (4, 8, 16)
FULL_RANK = min(N_CLASSES, WHAT_WIDTH)
STRONG_RECOVERY_FRACTION = 0.90
STRONG_FIXED_GAP = 0.01
MIN_NONNEGATIVE_SEEDS = 4
EPOCHS = exp542.EPOCHS
BATCH_SIZE = exp542.BATCH_SIZE
LR = exp542.LR
WEIGHT_DECAY = exp542.WEIGHT_DECAY
LOGREG_MAX_ITER = 5000
PREFIX_FRACTIONS = exp542.PREFIX_FRACTIONS


@dataclass(frozen=True)
class RunSpec:
    stage: str
    seed: int
    n_banks: int
    rank: int
    parameterization: str = "factorized"

    @property
    def key(self) -> str:
        return f"{self.parameterization}__k{self.n_banks}_r{self.rank}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


@dataclass
class ReadoutTrajectory:
    base_logits: torch.Tensor
    residual_logits: torch.Tensor
    base_evidence: torch.Tensor
    residual_evidence: torch.Tensor
    gate_probabilities: torch.Tensor
    centered_gates: torch.Tensor


def find_repo_root(start: Path | None = None) -> Path:
    return exp542.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def reference_model_path(root: Path, seed: int) -> Path:
    return root / "reference_models" / f"seed{seed}.npz"


def reference_evaluation_path(root: Path, seed: int) -> Path:
    return root / "reference_evaluations" / f"seed{seed}.json"


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_checkpoints" / f"{spec.key}.pt"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_histories" / f"{spec.key}.csv"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / f"{spec.stage}_evaluations" / f"{spec.key}.json"


def capacity_selection_path(root: Path) -> Path:
    return root / "capacity_selection.json"


def selection_path(root: Path) -> Path:
    return root / "selection.json"


def final_evaluation_path(root: Path, seed: int) -> Path:
    return root / "final_evaluations" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _exp541_config(config: Config) -> exp541.Config:
    return exp541.Config(config.repo_root, exp541.results_dir(config.repo_root), config.device, exp541.EPOCHS, config.batch_size, config.threads)


def _exp54_config(config: Config) -> exp54.Config:
    return exp54.Config(config.repo_root, exp54.results_dir(config.repo_root), config.device, exp54.EPOCHS, config.batch_size, config.threads)


def _base_spec(seed: int) -> exp541.RunSpec:
    return exp541.RunSpec(seed)


def capacity_specs() -> list[RunSpec]:
    return [RunSpec("capacity", seed, k, r, "factorized") for k in CAPACITY_BANKS for r in CAPACITY_RANKS for seed in SEEDS]


def _extension_specs_for_mode(mode: str) -> list[RunSpec]:
    if mode == "higher_k":
        return [RunSpec("extension", seed, k, FULL_RANK, "factorized") for k in EXTENDED_BANKS for seed in SEEDS]
    if mode == "direct_matrix":
        return [RunSpec("extension", seed, k, FULL_RANK, "direct") for k in FULL_MATRIX_BANKS for seed in SEEDS]
    raise ValueError(f"Unknown extension mode: {mode}")


def _load_capacity_selection(root: Path) -> dict[str, object]:
    path = capacity_selection_path(root)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("only_validation_selected") is not True or payload.get("test_not_used_for_selection") is not True:
        raise ValueError("Invalid Exp5.4.3 capacity selection")
    return payload


def extension_specs(root: Path) -> list[RunSpec]:
    selection = _load_capacity_selection(root)
    if selection.get("extension_required") is not True:
        return []
    return _extension_specs_for_mode(str(selection["extension_mode"]))


def _labels_lengths(data: base.Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    return {"train": (data.ytr, data.ltr), "val": (data.yva, data.lva), "test": (data.yte, data.lte)}[split]


def _load_what_arrays(seed: int, data: base.Data, config: Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, meta = exp54.load_fusion_cache(seed, data, _exp54_config(config))
    out = {split: np.asarray(arrays[f"what_{split}"], dtype=np.float32) for split in ("train", "val", "test")}
    for split, values in out.items():
        labels, _ = _labels_lengths(data, split)
        if values.shape != (len(labels), data.T, WHAT_WIDTH):
            raise ValueError(f"WHAT cache shape mismatch for {split}: {values.shape}")
    return out, meta


def _masked_what(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for row, length in enumerate(lengths):
        out[row, int(length):] = 0.0
    return out


def fixed250_layout(data: base.Data) -> tuple[int, int]:
    bin_steps = max(1, int(np.rint(FIXED250_SECONDS * float(data.fs))))
    return bin_steps, int(math.ceil(int(data.T) / bin_steps))


def fixed250_features(values: np.ndarray, lengths: np.ndarray, data: base.Data) -> tuple[np.ndarray, int, int]:
    bin_steps, n_bins = fixed250_layout(data)
    padded_steps = n_bins * bin_steps
    masked = _masked_what(values, lengths)
    if padded_steps > masked.shape[1]:
        masked = np.pad(masked, ((0, 0), (0, padded_steps - masked.shape[1]), (0, 0)))
    counts = masked.reshape(masked.shape[0], n_bins, bin_steps, WHAT_WIDTH).sum(axis=2)
    return counts.reshape(masked.shape[0], n_bins * WHAT_WIDTH), bin_steps, n_bins


def streaming_fixed250_logits(values: np.ndarray, lengths: np.ndarray, weight_bins: np.ndarray, bias: np.ndarray, bin_steps: int) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    logits = np.broadcast_to(np.asarray(bias, dtype=np.float64), (values64.shape[0], len(bias))).copy()
    for timestep in range(values64.shape[1]):
        bank = timestep // int(bin_steps)
        if bank >= weight_bins.shape[0]:
            break
        valid = (np.asarray(lengths) > timestep).astype(np.float64)[:, None]
        logits += (values64[:, timestep, :] @ weight_bins[bank].T) * valid
    return logits


def _metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(logits).argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -float(log_probs[np.arange(len(true)), true].mean())
    return {"loss": loss, "balanced_accuracy": float(balanced_accuracy_score(true, pred)), "accuracy": float(accuracy_score(true, pred)), "macro_f1": float(f1_score(true, pred, average="macro")), "n_samples": int(len(true))}


def prepare_reference_seed(seed: int, data: base.Data, config: Config, force: bool = False) -> Path:
    model_path = reference_model_path(config.results_dir, seed)
    eval_path = reference_evaluation_path(config.results_dir, seed)
    if model_path.exists() and eval_path.exists() and not force:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if payload.get("experiment_id") == EXPERIMENT_ID and payload.get("protocol_version") == PROTOCOL_VERSION and payload.get("seed") == seed:
            return eval_path
        raise ValueError("Existing reference artifact identity mismatch")
    base_config = _exp541_config(config)
    exp541.prepare_inputs_seed(seed, data, base_config, force=False)
    exp541.train_base(_base_spec(seed), data, base_config, force=False)
    _, base_payload = exp541.load_base(_base_spec(seed), data, base_config)
    what, source_meta = _load_what_arrays(seed, data, config)
    x_train, bin_steps, n_bins = fixed250_features(what["train"], data.ltr, data)
    x_val, val_steps, val_bins = fixed250_features(what["val"], data.lva, data)
    if (bin_steps, n_bins) != (val_steps, val_bins):
        raise RuntimeError("Fixed250 layout changed across splits")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    classifier = LogisticRegression(max_iter=LOGREG_MAX_ITER, random_state=base.dseed(seed, EXPERIMENT_ID, "fixed250_reference"), solver="lbfgs")
    classifier.fit(x_train_scaled, data.ytr)
    if not np.array_equal(classifier.classes_, np.arange(len(data.labels), dtype=np.int64)):
        raise RuntimeError("Reference classifier class order mismatch")
    effective_weight = classifier.coef_.astype(np.float64) / scaler.scale_[None, :]
    effective_bias = classifier.intercept_.astype(np.float64) - effective_weight @ scaler.mean_.astype(np.float64)
    weight_bins = effective_weight.reshape(len(data.labels), n_bins, WHAT_WIDTH).transpose(1, 0, 2)
    train_offline = classifier.decision_function(x_train_scaled)
    val_offline = classifier.decision_function(x_val_scaled)
    train_raw = x_train @ effective_weight.T + effective_bias
    val_raw = x_val @ effective_weight.T + effective_bias
    val_stream = streaming_fixed250_logits(what["val"], data.lva, weight_bins, effective_bias, bin_steps)
    max_raw_delta = float(max(np.max(np.abs(train_offline - train_raw)), np.max(np.abs(val_offline - val_raw))))
    max_stream_delta = float(np.max(np.abs(val_offline - val_stream)))
    if max_raw_delta > 1e-6 or max_stream_delta > 1e-6 or not np.array_equal(val_offline.argmax(1), val_stream.argmax(1)):
        raise RuntimeError(f"Fixed250 offline/streaming equivalence failed: raw={max_raw_delta:.3e}, streaming={max_stream_delta:.3e}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, weight_bins=weight_bins, bias=effective_bias, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, classes=classifier.classes_, bin_steps=np.asarray([bin_steps]), n_bins=np.asarray([n_bins]))
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "source_experiment_id": exp54.EXPERIMENT_ID,
        "source_protocol_version": exp54.PROTOCOL_VERSION,
        "source_cache_seed": source_meta.get("seed", seed),
        "source_what_definition": "frozen Local-SNN L2 spikes from the Exp5.4 fusion cache",
        "source_base_experiment_id": exp541.EXPERIMENT_ID,
        "source_base_protocol_version": exp541.PROTOCOL_VERSION,
        "split_seed": base.SPLIT_SEED,
        "train_users": data.split["train_users"], "val_users": data.split["val_users"], "test_users": data.split["test_users"],
        "labels": data.labels, "sampling_rate_hz": float(data.fs),
        "fixed250": {"requested_seconds": FIXED250_SECONDS, "bin_steps": int(bin_steps), "actual_seconds": float(bin_steps / data.fs), "n_bins": int(n_bins), "feature_dim": int(n_bins * WHAT_WIDTH), "classifier": "StandardScaler(train only) + LogisticRegression(lbfgs)", "raw_space_conversion": "W_eff=coef/scale; b_eff=intercept-W_eff@mean"},
        "train_max_elapsed_seconds": float((int(np.max(data.ltr)) - 1) / data.fs),
        "what_base_val": base_payload["result"]["native"]["val"],
        "fixed250_train": _metrics_from_logits(data.ytr, train_offline),
        "fixed250_val": _metrics_from_logits(data.yva, val_offline),
        "equivalence": {"max_abs_offline_vs_raw_space_logits": max_raw_delta, "max_abs_offline_vs_streaming_logits": max_stream_delta, "val_predictions_identical": True, "tolerance": 1e-6},
        "test_evaluated": False,
    }
    _save_json(eval_path, payload)
    return eval_path


def _load_reference(seed: int, config: Config) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model_path, eval_path = reference_model_path(config.results_dir, seed), reference_evaluation_path(config.results_dir, seed)
    if not model_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Missing Exp5.4.3 reference for seed {seed}")
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("seed") != seed or payload.get("test_evaluated") is not False:
        raise ValueError("Invalid reference artifact identity")
    with np.load(model_path, allow_pickle=False) as loaded:
        model = {name: loaded[name] for name in loaded.files}
    return model, payload


def _loader(what: np.ndarray, labels: np.ndarray, lengths: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(TensorDataset(torch.tensor(what, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


def _make_loaders(what: dict[str, np.ndarray], data: base.Data, spec: RunSpec, config: Config, train_shuffle: bool, splits: tuple[str, ...] = ("train", "val")) -> dict[str, DataLoader]:
    out = {}
    for split in splits:
        labels, lengths = _labels_lengths(data, split)
        out[split] = _loader(what[split], labels, lengths, config.batch_size, train_shuffle if split == "train" else False, base.dseed(spec.seed, EXPERIMENT_ID, spec.stage, spec.parameterization, spec.n_banks, spec.rank, split))
    return out


class ElapsedConditionedReadout(nn.Module):
    def __init__(self, base_model: exp541.DirectWhatBase, n_classes: int, n_banks: int, rank: int, parameterization: str) -> None:
        super().__init__()
        if parameterization not in ("factorized", "direct") or n_banks <= 1 or rank <= 0 or rank > FULL_RANK:
            raise ValueError("Invalid elapsed readout configuration")
        self.base_model = base_model.eval()
        for p in self.base_model.parameters(): p.requires_grad_(False)
        self.n_classes, self.n_banks, self.rank, self.parameterization = int(n_classes), int(n_banks), int(rank), parameterization
        self.gate = nn.Linear(WHEN_WIDTH, n_banks, bias=False)
        if parameterization == "factorized":
            self.bank_u = nn.Parameter(torch.empty(n_banks, n_classes, rank)); self.bank_v = nn.Parameter(torch.empty(n_banks, WHAT_WIDTH, rank))
            nn.init.xavier_uniform_(self.bank_u); nn.init.xavier_uniform_(self.bank_v)
        else:
            self.bank_w = nn.Parameter(torch.empty(n_banks, n_classes, WHAT_WIDTH)); nn.init.xavier_uniform_(self.bank_w)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def train(self, mode: bool = True):
        super().train(mode); self.base_model.eval(); return self

    def bank_matrices(self) -> torch.Tensor:
        return torch.einsum("kcr,kir->kci", self.bank_u, self.bank_v) if self.parameterization == "factorized" else self.bank_w

    def forward(self, what: torch.Tensor, context: torch.Tensor, lengths: torch.Tensor, *, return_trajectory: bool = False):
        with torch.no_grad(): base_logits, base_evidence = self.base_model(what, lengths, return_evidence=True)
        valid = base.mask(lengths, what.shape[1]).to(what.dtype).unsqueeze(-1)
        probs = torch.softmax(self.gate(context), dim=-1); centered = probs - 1.0 / self.n_banks
        if self.parameterization == "factorized":
            projected = torch.einsum("bti,kir->btkr", what, self.bank_v); bank_evidence = torch.einsum("btkr,kcr->btkc", projected, self.bank_u)
        else:
            bank_evidence = torch.einsum("bti,kci->btkc", what, self.bank_w)
        residual_evidence = self.alpha * (centered.unsqueeze(-1) * bank_evidence).sum(2) * valid
        residual_logits = residual_evidence.sum(1); logits = base_logits + residual_logits
        tr = ReadoutTrajectory(base_logits, residual_logits, base_evidence, residual_evidence, probs, centered) if return_trajectory else None
        return logits, tr


def parameter_counts(model: ElapsedConditionedReadout) -> dict[str, int]:
    return {"frozen_base": int(sum(p.numel() for p in model.base_model.parameters())), "trainable_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)), "stored_total": int(sum(p.numel() for p in model.parameters()))}


def _validate_spec(spec: RunSpec) -> None:
    if spec.seed not in SEEDS or spec.rank <= 0 or spec.rank > FULL_RANK or spec.parameterization not in ("factorized", "direct"):
        raise ValueError(spec)


def _initialize_model(spec: RunSpec, data: base.Data, config: Config) -> ElapsedConditionedReadout:
    _validate_spec(spec)
    base_model, _ = exp541.load_base(_base_spec(spec.seed), data, _exp541_config(config))
    base.seed_all(base.dseed(spec.seed, EXPERIMENT_ID, spec.stage, spec.parameterization, spec.n_banks, spec.rank, "constructor"))
    return ElapsedConditionedReadout(base_model, len(data.labels), spec.n_banks, spec.rank, spec.parameterization).to(config.device)


def _elapsed_context(batch_size: int, steps: int, data: base.Data, reference: dict[str, object], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return exp54.elapsed_context(batch_size, steps, float(data.fs), float(reference["train_max_elapsed_seconds"]), device, dtype)


def _classification_metrics(true: np.ndarray, pred: np.ndarray, loss: float) -> dict[str, float | int]:
    return {"loss": float(loss), "balanced_accuracy": float(balanced_accuracy_score(true, pred)), "accuracy": float(accuracy_score(true, pred)), "macro_f1": float(f1_score(true, pred, average="macro")), "n_samples": int(len(true))}


def _evaluate(model: ElapsedConditionedReadout, loader: DataLoader, data: base.Data, reference: dict[str, object], device: torch.device, *, diagnostics: bool = False) -> dict[str, Any]:
    model.eval(); ys=[]; ps=[]; bps=[]; losses=[]; ratios=[]; max_abs=0.0
    gate_entropy=[]; utilization=np.zeros(model.n_banks, dtype=np.int64)
    bin_steps, n_bins = fixed250_layout(data); gate_sum=np.zeros((n_bins, model.n_banks)); gate_n=np.zeros(n_bins, dtype=np.int64)
    frac_records={rho: [] for rho in PREFIX_FRACTIONS}; endpoints=tuple(min(data.T,(i+1)*bin_steps) for i in range(n_bins)); abs_records={e: [] for e in endpoints}
    with torch.no_grad():
        for what, labels, lengths in loader:
            what, labels, lengths = what.to(device), labels.to(device), lengths.to(device)
            context = _elapsed_context(len(what), what.shape[1], data, reference, device, what.dtype)
            logits, tr = model(what, context, lengths, return_trajectory=True); loss=F.cross_entropy(logits, labels)
            pred, bp = logits.argmax(1), tr.base_logits.argmax(1); ys.append(labels.cpu().numpy()); ps.append(pred.cpu().numpy()); bps.append(bp.cpu().numpy()); losses.append((float(loss.item()), len(labels)))
            ratio=torch.linalg.vector_norm(tr.residual_logits,dim=1)/(torch.linalg.vector_norm(tr.base_logits,dim=1)+1e-8); ratios.extend(ratio.cpu().tolist()); max_abs=max(max_abs,float((logits-tr.base_logits).abs().max().item()))
            if diagnostics:
                valid=base.mask(lengths, what.shape[1]); vp=tr.gate_probabilities[valid]; gate_entropy.extend((-(vp*torch.log(vp.clamp_min(1e-12))).sum(1)).cpu().tolist()); utilization += np.bincount(vp.argmax(1).cpu().numpy(), minlength=model.n_banks)
                for row, lt in enumerate(lengths.cpu()):
                    length=int(lt.item()); bc=tr.base_evidence[row,:length].cumsum(0); rc=tr.residual_evidence[row,:length].cumsum(0)
                    for rho in PREFIX_FRACTIONS:
                        e=min(length-1,max(0,int(math.ceil(rho*length))-1)); bl=bc[e]+model.base_model.class_bias; cl=bc[e]+rc[e]+model.base_model.class_bias; frac_records[rho].append((int(labels[row]),int(bl.argmax()),int(cl.argmax())))
                    for endpoint in endpoints:
                        e=max(min(length,endpoint)-1,0); bl=bc[e]+model.base_model.class_bias; cl=bc[e]+rc[e]+model.base_model.class_bias; abs_records[endpoint].append((int(labels[row]),int(bl.argmax()),int(cl.argmax())))
                    for t in range(length):
                        b=min(n_bins-1,t//bin_steps); gate_sum[b]+=tr.gate_probabilities[row,t].cpu().numpy(); gate_n[b]+=1
    true,pred,bp=np.concatenate(ys),np.concatenate(ps),np.concatenate(bps); out=_classification_metrics(true,pred,sum(v*n for v,n in losses)/sum(n for _,n in losses)); r=np.asarray(ratios)
    out["base_balanced_accuracy"]=float(balanced_accuracy_score(true,bp)); out["residual_ratio"]={"mean":float(r.mean()),"median":float(np.median(r)),"p95":float(np.percentile(r,95)),"max":float(r.max())}; out["max_abs_logit_delta_vs_base"]=max_abs
    if diagnostics:
        bcorr,both=bp==true,pred==true; out["rescue_harm"]={"rescued":int((~bcorr&both).sum()),"harmed":int((bcorr&~both).sum()),"retained_correct":int((bcorr&both).sum()),"retained_error":int((~bcorr&~both).sum())}
        out["prefix_fraction"]=[{"progress":rho,"base_balanced_accuracy":float(balanced_accuracy_score(np.asarray(rec)[:,0],np.asarray(rec)[:,1])),"combined_balanced_accuracy":float(balanced_accuracy_score(np.asarray(rec)[:,0],np.asarray(rec)[:,2]))} for rho,rec in frac_records.items()]
        out["prefix_absolute"]=[{"elapsed_seconds":float(e/data.fs),"base_balanced_accuracy":float(balanced_accuracy_score(np.asarray(rec)[:,0],np.asarray(rec)[:,1])),"combined_balanced_accuracy":float(balanced_accuracy_score(np.asarray(rec)[:,0],np.asarray(rec)[:,2]))} for e,rec in abs_records.items()]
        out["gate"]={"entropy_mean":float(np.mean(gate_entropy)),"utilization":(utilization/max(utilization.sum(),1)).tolist(),"absolute_time_probability":[(gate_sum[i]/max(gate_n[i],1)).tolist() for i in range(n_bins)],"absolute_time_seconds":[float((i+0.5)*bin_steps/data.fs) for i in range(n_bins)]}
    return out


def train_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> Path:
    _validate_spec(spec); dest=checkpoint_path(config.results_dir,spec)
    if dest.exists() and not force: return dest
    _,ref=_load_reference(spec.seed,config); what,_=_load_what_arrays(spec.seed,data,config); train_loaders=_make_loaders(what,data,spec,config,True,("train","val")); eval_loaders=_make_loaders(what,data,spec,config,False,("train","val")); device=torch.device(config.device); torch.set_num_threads(config.threads); model=_initialize_model(spec,data,config)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WEIGHT_DECAY); e0tr=_evaluate(model,eval_loaders["train"],data,ref,device); e0va=_evaluate(model,eval_loaders["val"],data,ref,device)
    if float(e0va["max_abs_logit_delta_vs_base"])>1e-7: raise RuntimeError("alpha=0 must reproduce frozen WHAT")
    best_ba,best_loss,best_epoch=float(e0va["balanced_accuracy"]),float(e0va["loss"]),0; best_state=copy.deepcopy(model.state_dict()); hist=[{"epoch":0,"train_loss":e0tr["loss"],"train_balanced_accuracy":e0tr["balanced_accuracy"],"val_loss":best_loss,"val_balanced_accuracy":best_ba}]
    for epoch in range(1,config.epochs+1):
        model.train(); yt=[]; yp=[]; lsum=0.0; nt=0
        for xb,y,l in train_loaders["train"]:
            xb,y,l=xb.to(device),y.to(device),l.to(device); ctx=_elapsed_context(len(xb),xb.shape[1],data,ref,device,xb.dtype); opt.zero_grad(set_to_none=True); logits,_=model(xb,ctx,l); loss=F.cross_entropy(logits,y); loss.backward(); opt.step(); n=len(y); lsum+=float(loss.item())*n; nt+=n; yt.append(y.cpu().numpy()); yp.append(logits.detach().argmax(1).cpu().numpy())
        val=_evaluate(model,eval_loaders["val"],data,ref,device); vba,vl=float(val["balanced_accuracy"]),float(val["loss"]); hist.append({"epoch":epoch,"train_loss":lsum/max(nt,1),"train_balanced_accuracy":float(balanced_accuracy_score(np.concatenate(yt),np.concatenate(yp))),"val_loss":vl,"val_balanced_accuracy":vba})
        if vba>best_ba+1e-12 or (abs(vba-best_ba)<=1e-12 and vl<best_loss): best_ba,best_loss,best_epoch,best_state=vba,vl,epoch,copy.deepcopy(model.state_dict())
    dest.parent.mkdir(parents=True,exist_ok=True); torch.save({"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"spec":asdict(spec),"state_dict":best_state,"result":{"best_epoch":best_epoch,"best_val_balanced_accuracy":best_ba,"best_val_loss":best_loss}},dest); hp=history_path(config.results_dir,spec); hp.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(hist).to_csv(hp,index=False); return dest


def load_model(spec: RunSpec, data: base.Data, config: Config):
    path=checkpoint_path(config.results_dir,spec); payload=torch.load(path,map_location=config.device,weights_only=False)
    if payload.get("experiment_id")!=EXPERIMENT_ID or payload.get("protocol_version")!=PROTOCOL_VERSION or payload.get("spec")!=asdict(spec): raise ValueError("Checkpoint identity mismatch")
    model=_initialize_model(spec,data,config); model.load_state_dict(payload["state_dict"]); model.eval(); return model,payload


def evaluate_validation_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    dest=evaluation_path(config.results_dir,spec)
    if dest.exists() and not force: return json.loads(dest.read_text(encoding="utf-8"))
    _,ref=_load_reference(spec.seed,config); what,_=_load_what_arrays(spec.seed,data,config); loaders=_make_loaders(what,data,spec,config,False,splits=("train", "val")); model,ckpt=load_model(spec,data,config); device=torch.device(config.device); tr=_evaluate(model,loaders["train"],data,ref,device); val=_evaluate(model,loaders["val"],data,ref,device,diagnostics=True)
    b=float(val["base_balanced_accuracy"]); f=float(ref["fixed250_val"]["balanced_accuracy"]); c=float(val["balanced_accuracy"]); gap=f-b; rec=(c-b)/gap if gap>1e-12 else float("nan")
    payload={"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"spec":asdict(spec),"seed":spec.seed,"best_epoch":ckpt["result"]["best_epoch"],"parameter_counts":parameter_counts(model),"train":tr,"val":val,"reference":{"what_base_val_balanced_accuracy":b,"fixed250_val_balanced_accuracy":f,"fixed250_gap_over_base":gap,"recovery_fraction":rec},"test_evaluated":False}; _save_json(dest,payload); return payload


def run_validation_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    train_one(spec,data,config,force); return evaluate_validation_one(spec,data,config,force)


def _load_eval(root: Path, spec: RunSpec) -> dict[str, object]:
    path=evaluation_path(root,spec)
    if not path.exists(): raise FileNotFoundError(f"Missing upstream run artifact: {path}")
    p=json.loads(path.read_text(encoding="utf-8"));
    if p.get("test_evaluated") is not False: raise ValueError("Selection artifact evaluated test")
    return p


def _sem(v: np.ndarray) -> float: return 0.0 if len(v)<=1 else float(v.std(ddof=1)/math.sqrt(len(v)))


def _summary_from_payloads(specs, payloads):
    rows=[]; summaries=[]
    for s in specs:
        p=payloads[s.key]; val=p["val"]; ref=p["reference"]; b=float(ref["what_base_val_balanced_accuracy"]); f=float(ref["fixed250_val_balanced_accuracy"]); c=float(val["balanced_accuracy"]); gap=f-b; rec=(c-b)/gap if gap>1e-12 else float("nan")
        rows.append({"stage":s.stage,"parameterization":s.parameterization,"seed":s.seed,"n_banks":s.n_banks,"rank":s.rank,"best_epoch":p["best_epoch"],"trainable_parameter_count":p["parameter_counts"]["trainable_total"],"base_val_balanced_accuracy":b,"fixed250_val_balanced_accuracy":f,"val_balanced_accuracy":c,"delta_val_ba_vs_base":c-b,"gap_to_fixed250_val_ba":f-c,"oracle_gap_recovery_fraction":rec,"val_accuracy":val["accuracy"],"val_macro_f1":val["macro_f1"],"val_loss":val["loss"],"test_evaluated":False})
    groups={}
    for row in rows: groups.setdefault((row["parameterization"],row["n_banks"],row["rank"]),[]).append(row)
    for (param,k,r),group in sorted(groups.items()):
        group=sorted(group,key=lambda x:x["seed"]); d=np.asarray([x["delta_val_ba_vs_base"] for x in group],float); rec=np.asarray([x["oracle_gap_recovery_fraction"] for x in group],float); gap=np.asarray([x["gap_to_fixed250_val_ba"] for x in group],float); ba=np.asarray([x["val_balanced_accuracy"] for x in group],float); eligible=bool(d.mean()>0 and int((d>=-1e-12).sum())>=MIN_NONNEGATIVE_SEEDS); strong=bool(eligible and np.nanmean(rec)>=STRONG_RECOVERY_FRACTION and gap.mean()<=STRONG_FIXED_GAP)
        summaries.append({"stage":group[0]["stage"],"parameterization":param,"n_banks":k,"rank":r,"mean_val_balanced_accuracy":float(ba.mean()),"mean_paired_val_delta_vs_base":float(d.mean()),"sem_paired_val_delta_vs_base":_sem(d),"nonnegative_seed_count":int((d>=-1e-12).sum()),"mean_oracle_gap_recovery_fraction":float(np.nanmean(rec)),"mean_gap_to_fixed250_val_ba":float(gap.mean()),"trainable_parameter_count":int(group[0]["trainable_parameter_count"]),"eligible":eligible,"strong_recovery":strong})
    return rows,summaries


def _select_minimal_supported(summaries):
    eligible=[x for x in summaries if bool(x["eligible"])]
    if not eligible: raise RuntimeError("No capacity improves frozen WHAT on validation")
    strong=[x for x in eligible if bool(x["strong_recovery"])]
    if strong: return min(strong,key=lambda x:(int(x["trainable_parameter_count"]),-float(x["mean_paired_val_delta_vs_base"])))
    best=max(eligible,key=lambda x:float(x["mean_paired_val_delta_vs_base"])); cutoff=float(best["mean_paired_val_delta_vs_base"])-float(best["sem_paired_val_delta_vs_base"]); within=[x for x in eligible if float(x["mean_paired_val_delta_vs_base"])>=cutoff]
    return min(within,key=lambda x:(int(x["trainable_parameter_count"]),-float(x["mean_paired_val_delta_vs_base"])))


def _reference_rows(root: Path):
    rows=[]
    for seed in SEEDS:
        p=json.loads(reference_evaluation_path(root,seed).read_text(encoding="utf-8")); rows.append({"seed":seed,"base_val_balanced_accuracy":p["what_base_val"]["balanced_accuracy"],"fixed250_val_balanced_accuracy":p["fixed250_val"]["balanced_accuracy"],"fixed250_train_balanced_accuracy":p["fixed250_train"]["balanced_accuracy"],"fixed250_gap_over_base":float(p["fixed250_val"]["balanced_accuracy"])-float(p["what_base_val"]["balanced_accuracy"]),"offline_streaming_max_abs_logit_delta":p["equivalence"]["max_abs_offline_vs_streaming_logits"],"test_evaluated":False})
    return rows


def _lock_selection(root: Path, selected: dict[str, object], *, source_stage: str, capacity_selection: dict[str, object], extension_summary=None) -> Path:
    path=selection_path(root); _save_json(path,{"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"selected_stage":source_stage,"selected_parameterization":selected["parameterization"],"selected_n_banks":int(selected["n_banks"]),"selected_rank":int(selected["rank"]),"selection_metric":"validation oracle-gap recovery with strong-recovery threshold; otherwise one-standard-error paired BA rule and minimum parameters","selected_summary":selected,"capacity_selection":capacity_selection,"extension_summary":extension_summary,"only_validation_selected":True,"test_not_used_for_selection":True}); return path


def finalize_capacity(repo_root: Path) -> dict[str, Path]:
    root=results_dir(repo_root); specs=capacity_specs();
    if len(specs)!=45: raise RuntimeError("Stage-A mapping must contain 45 runs")
    payloads={s.key:_load_eval(root,s) for s in specs}; rows,summaries=_summary_from_payloads(specs,payloads); refs=_reference_rows(root); selected=_select_minimal_supported(summaries); extension_required=not any(bool(x["strong_recovery"]) for x in summaries); mode=None
    if extension_required:
        k8={x["seed"]:x for x in rows if x["parameterization"]=="factorized" and x["n_banks"]==8 and x["rank"]==FULL_RANK}; k16={x["seed"]:x for x in rows if x["parameterization"]=="factorized" and x["n_banks"]==16 and x["rank"]==FULL_RANK}; paired=np.asarray([k16[s]["val_balanced_accuracy"]-k8[s]["val_balanced_accuracy"] for s in SEEDS]); mode="higher_k" if paired.mean()>0 and int((paired>=-1e-12).sum())>=MIN_NONNEGATIVE_SEEDS else "direct_matrix"
    outputs={"reference_runs":root/"reference_runs.csv","capacity_runs":root/"capacity_runs.csv","capacity_paired_deltas":root/"capacity_paired_deltas.csv","capacity_recovery":root/"capacity_recovery.csv","capacity_summary":root/"capacity_summary.csv","capacity_selection":capacity_selection_path(root)}
    pd.DataFrame(refs).to_csv(outputs["reference_runs"],index=False); pd.DataFrame(rows).to_csv(outputs["capacity_runs"],index=False); pd.DataFrame([{k:r[k] for k in ("seed","parameterization","n_banks","rank","delta_val_ba_vs_base","gap_to_fixed250_val_ba")} for r in rows]).to_csv(outputs["capacity_paired_deltas"],index=False); pd.DataFrame([{k:r[k] for k in ("seed","parameterization","n_banks","rank","oracle_gap_recovery_fraction")} for r in rows]).to_csv(outputs["capacity_recovery"],index=False); pd.DataFrame(summaries).to_csv(outputs["capacity_summary"],index=False)
    capsel={"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"stage":"capacity","status":"extension_required" if extension_required else "capacity_closed","preliminary_selection":selected,"extension_required":extension_required,"extension_mode":mode,"capacity_summary":summaries,"strong_recovery_definition":{"mean_recovery_fraction_at_least":STRONG_RECOVERY_FRACTION,"mean_gap_to_fixed250_val_ba_at_most":STRONG_FIXED_GAP},"only_validation_selected":True,"test_not_used_for_selection":True}; _save_json(outputs["capacity_selection"],capsel)
    if not extension_required: _lock_selection(root,selected,source_stage="capacity",capacity_selection=capsel)
    return outputs


def finalize_extension(repo_root: Path) -> dict[str, Path]:
    root=results_dir(repo_root); capsel=_load_capacity_selection(root); specs=extension_specs(root)
    if len(specs)!=15: raise RuntimeError("Stage-B mapping must contain exactly 15 runs")
    rows,summaries=_summary_from_payloads(specs,{s.key:_load_eval(root,s) for s in specs}); cap=pd.read_csv(root/"capacity_summary.csv").to_dict(orient="records"); selected=_select_minimal_supported(cap+summaries); outputs={"extension_runs":root/"extension_runs.csv","extension_summary":root/"extension_summary.csv","selection":selection_path(root)}; pd.DataFrame(rows).to_csv(outputs["extension_runs"],index=False); pd.DataFrame(summaries).to_csv(outputs["extension_summary"],index=False); _lock_selection(root,selected,source_stage=str(selected["stage"]),capacity_selection=capsel,extension_summary=summaries); return outputs


def _load_selection(root: Path) -> dict[str, object]:
    path=selection_path(root)
    if not path.exists(): raise FileNotFoundError(f"Missing locked selection; test remains unopened: {path}")
    p=json.loads(path.read_text(encoding="utf-8"));
    if p.get("only_validation_selected") is not True or p.get("test_not_used_for_selection") is not True: raise ValueError("Invalid locked selection")
    return p


def selected_spec(root: Path, seed: int) -> RunSpec:
    s=_load_selection(root); return RunSpec(str(s["selected_stage"]),seed,int(s["selected_n_banks"]),int(s["selected_rank"]),str(s["selected_parameterization"]))


def _evaluate_fixed250_test(seed: int, data: base.Data, config: Config, what_test: np.ndarray) -> dict[str, object]:
    rm,ref=_load_reference(seed,config); wb=np.asarray(rm["weight_bins"],float); bias=np.asarray(rm["bias"],float); bs=int(rm["bin_steps"][0]); x,bs2,nb=fixed250_features(what_test,data.lte,data)
    if bs!=bs2 or nb!=wb.shape[0]: raise RuntimeError("Fixed250 layout mismatch")
    wflat=wb.transpose(1,0,2).reshape(len(data.labels),-1); offline=x@wflat.T+bias; streaming=streaming_fixed250_logits(what_test,data.lte,wb,bias,bs); delta=float(np.max(np.abs(offline-streaming)))
    if delta>1e-6 or not np.array_equal(offline.argmax(1),streaming.argmax(1)): raise RuntimeError("Final Fixed250 streaming equivalence failed")
    return {"metrics":_metrics_from_logits(data.yte,offline),"max_abs_offline_vs_streaming_logits":delta,"predictions_identical":True,"reference_validation":ref["fixed250_val"]}


def _effective_weight_diagnostics(model, rm, ref, data, device):
    wb=np.asarray(rm["weight_bins"],float); bs=int(rm["bin_steps"][0]);
    with torch.no_grad():
        ctx=_elapsed_context(1,data.T,data,ref,device,torch.float32); probs=torch.softmax(model.gate(ctx),dim=-1)[0]; centered=probs-1.0/model.n_banks; banks=model.bank_matrices(); eff=model.base_model.output_projection.weight.unsqueeze(0)+model.alpha*torch.einsum("tk,kci->tci",centered,banks); eff=eff.cpu().numpy(); banks=banks.cpu().numpy()
    dist=[]
    for b in range(wb.shape[0]):
        t=min(data.T-1,b*bs+bs//2); c,target=eff[t],wb[b]; tn,cn=np.linalg.norm(target),np.linalg.norm(c); dist.append({"bin_index":b,"elapsed_seconds":float(t/data.fs),"normalized_frobenius_distance":float(np.linalg.norm(c-target)/(tn+1e-12)),"cosine_similarity":float(np.sum(c*target)/(cn*tn+1e-12)),"fixed250_weight_norm":float(tn),"candidate_weight_norm":float(cn)})
    ranks=[]
    for k,m in enumerate(banks):
        sv=np.linalg.svd(m,compute_uv=False); threshold=max(float(sv[0])*1e-6,1e-12); ranks.append({"bank":k,"effective_rank":int((sv>threshold).sum()),"singular_values":sv.tolist()})
    return dist,ranks


def evaluate_final_seed(seed: int, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    dest=final_evaluation_path(config.results_dir,seed)
    if dest.exists() and not force: return json.loads(dest.read_text(encoding="utf-8"))
    selection=_load_selection(config.results_dir); spec=selected_spec(config.results_dir,seed); rm,ref=_load_reference(seed,config); what,_=_load_what_arrays(seed,data,config); loader=_make_loaders(what,data,spec,config,False,splits=("test",))["test"]; device=torch.device(config.device); model,ckpt=load_model(spec,data,config); selected=_evaluate(model,loader,data,ref,device,diagnostics=True); fixed=_evaluate_fixed250_test(seed,data,config,what["test"]); anchor_spec=RunSpec("capacity",seed,4,4,"factorized"); anchor_model,anchor_ckpt=load_model(anchor_spec,data,config); anchor=_evaluate(anchor_model,loader,data,ref,device); dist,ranks=_effective_weight_diagnostics(model,rm,ref,data,device); b=float(selected["base_balanced_accuracy"]); f=float(fixed["metrics"]["balanced_accuracy"]); c=float(selected["balanced_accuracy"]); gap=f-b; recovery=(c-b)/gap if gap>1e-12 else float("nan")
    payload={"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"selection":selection,"spec":asdict(spec),"seed":seed,"best_epoch":ckpt["result"]["best_epoch"],"parameter_counts":parameter_counts(model),"base_test_balanced_accuracy":b,"fixed250_test":fixed,"anchor_k4_r4":{"best_epoch":anchor_ckpt["result"]["best_epoch"],"test":anchor},"selected_test":selected,"oracle_gap_recovery_fraction":recovery,"effective_weight_distance":dist,"effective_rank":ranks,"test_evaluated":True}; _save_json(dest,payload); return payload


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root=results_dir(repo_root); selection=_load_selection(root)
    for path in (root/"reference_runs.csv",root/"capacity_runs.csv",root/"capacity_summary.csv",capacity_selection_path(root),selection_path(root)):
        if not path.exists(): raise FileNotFoundError(f"Finalizer will not regenerate missing upstream artifact: {path}")
    final=[]; paired=[]; gate_rows=[]; dist_rows=[]; rank_rows=[]; prefix=[]; residual=[]; rescue=[]
    for seed in SEEDS:
        p=json.loads(final_evaluation_path(root,seed).read_text(encoding="utf-8")); s=p["selected_test"]; f=p["fixed250_test"]["metrics"]; a=p["anchor_k4_r4"]["test"]; bba=float(p["base_test_balanced_accuracy"]); sba=float(s["balanced_accuracy"]); fba=float(f["balanced_accuracy"]); aba=float(a["balanced_accuracy"])
        final.append({"seed":seed,"selected_stage":selection["selected_stage"],"parameterization":selection["selected_parameterization"],"n_banks":selection["selected_n_banks"],"rank":selection["selected_rank"],"trainable_parameter_count":p["parameter_counts"]["trainable_total"],"base_test_balanced_accuracy":bba,"fixed250_test_balanced_accuracy":fba,"anchor_k4_r4_test_balanced_accuracy":aba,"selected_test_balanced_accuracy":sba,"selected_delta_test_ba_vs_base":sba-bba,"selected_gap_to_fixed250_test_ba":fba-sba,"oracle_gap_recovery_fraction":p["oracle_gap_recovery_fraction"],"selected_test_accuracy":s["accuracy"],"selected_test_macro_f1":s["macro_f1"],"selected_test_loss":s["loss"],"fixed250_streaming_max_abs_logit_delta":p["fixed250_test"]["max_abs_offline_vs_streaming_logits"]})
        paired.extend([{"seed":seed,"comparison":"selected_minus_what","delta_test_balanced_accuracy":sba-bba},{"seed":seed,"comparison":"selected_minus_fixed250","delta_test_balanced_accuracy":sba-fba},{"seed":seed,"comparison":"anchor_k4_r4_minus_what","delta_test_balanced_accuracy":aba-bba}])
        gate=s["gate"]; gate_rows.append({"seed":seed,"metric":"gate_entropy_mean","bank":None,"elapsed_seconds":None,"value":gate["entropy_mean"]})
        for k,v in enumerate(gate["utilization"]): gate_rows.append({"seed":seed,"metric":"bank_utilization","bank":k,"elapsed_seconds":None,"value":v})
        for elapsed,probs in zip(gate["absolute_time_seconds"],gate["absolute_time_probability"]):
            for k,v in enumerate(probs): gate_rows.append({"seed":seed,"metric":"bank_probability_by_elapsed_time","bank":k,"elapsed_seconds":elapsed,"value":v})
        dist_rows.extend([{"seed":seed,**row} for row in p["effective_weight_distance"]])
        for row in p["effective_rank"]: rank_rows.append({"seed":seed,"bank":row["bank"],"effective_rank":row["effective_rank"],"singular_values_json":json.dumps(row["singular_values"])})
        for row in s["prefix_fraction"]: prefix.extend([{"seed":seed,"axis":"relative_fraction","position":row["progress"],"model":"what_only","balanced_accuracy":row["base_balanced_accuracy"]},{"seed":seed,"axis":"relative_fraction","position":row["progress"],"model":"selected","balanced_accuracy":row["combined_balanced_accuracy"]}])
        for row in s["prefix_absolute"]: prefix.extend([{"seed":seed,"axis":"elapsed_seconds","position":row["elapsed_seconds"],"model":"what_only","balanced_accuracy":row["base_balanced_accuracy"]},{"seed":seed,"axis":"elapsed_seconds","position":row["elapsed_seconds"],"model":"selected","balanced_accuracy":row["combined_balanced_accuracy"]}])
        residual.append({"seed":seed,**s["residual_ratio"]}); rescue.append({"seed":seed,**s["rescue_harm"]})
    outputs={"final_runs":root/"final_runs.csv","final_paired_deltas":root/"final_paired_deltas.csv","gate_activity":root/"gate_activity.csv","effective_weight_distance":root/"effective_weight_distance.csv","effective_rank":root/"effective_rank.csv","prefix_ba":root/"prefix_ba.csv","residual_metrics":root/"residual_metrics.csv","rescue_harm":root/"rescue_harm.csv","manifest":root/"manifest.json"}
    for name,rows in (("final_runs",final),("final_paired_deltas",paired),("gate_activity",gate_rows),("effective_weight_distance",dist_rows),("effective_rank",rank_rows),("prefix_ba",prefix),("residual_metrics",residual),("rescue_harm",rescue)): pd.DataFrame(rows).to_csv(outputs[name],index=False)
    _save_json(outputs["manifest"],{"experiment_id":EXPERIMENT_ID,"protocol_version":PROTOCOL_VERSION,"seeds":list(SEEDS),"capacity_banks":list(CAPACITY_BANKS),"capacity_ranks":list(CAPACITY_RANKS),"full_rank":FULL_RANK,"fixed250_seconds":FIXED250_SECONDS,"selection":selection,"only_validation_selected":True,"test_not_used_for_selection":True,"final_test_opened_only_after_selection_json":True,"aggregation_policy":"finalizers aggregate existing per-run artifacts only; notebook is analysis-only and never trains or regenerates missing runs"}); return outputs


def _config_from_args(args):
    root=Path(args.repo_root).resolve() if args.repo_root else find_repo_root(); return Config(root,results_dir(root),args.device,args.epochs,args.batch_size,args.threads)


def _parser():
    p=argparse.ArgumentParser(description="Experiment 5.4.3 causal elapsed-time-conditioned readout capacity"); sub=p.add_subparsers(dest="command",required=True)
    def common(c): c.add_argument("--repo-root",default=None); c.add_argument("--device",default="cpu"); c.add_argument("--threads",type=int,default=1); c.add_argument("--batch-size",type=int,default=BATCH_SIZE); c.add_argument("--epochs",type=int,default=EPOCHS); c.add_argument("--force",action="store_true")
    for name in ("prepare-reference","capacity-run-one","extension-run-one","final-run-one"): c=sub.add_parser(name); common(c); c.add_argument("--array-task-id",type=int,required=True)
    for name in ("capacity-finalize","extension-finalize","finalize"): c=sub.add_parser(name); c.add_argument("--repo-root",default=None)
    return p


def main():
    args=_parser().parse_args()
    if args.command in ("capacity-finalize","extension-finalize","finalize"):
        root=Path(args.repo_root).resolve() if args.repo_root else find_repo_root(); outputs=finalize_capacity(root) if args.command=="capacity-finalize" else finalize_extension(root) if args.command=="extension-finalize" else finalize_experiment(root)
        for n,p in outputs.items(): print(f"{n}: {p}")
        return
    config=_config_from_args(args); data=base.prepare_data(config.repo_root); task=int(args.array_task_id)
    if args.command=="prepare-reference": print(prepare_reference_seed(SEEDS[task],data,config,args.force)); return
    if args.command=="capacity-run-one": print(json.dumps(run_validation_one(capacity_specs()[task],data,config,args.force),indent=2)); return
    if args.command=="extension-run-one": print(json.dumps(run_validation_one(extension_specs(config.results_dir)[task],data,config,args.force),indent=2)); return
    if args.command=="final-run-one": print(json.dumps(evaluate_final_seed(SEEDS[task],data,config,args.force),indent=2)); return
    raise RuntimeError(args.command)


if __name__ == "__main__": main()
