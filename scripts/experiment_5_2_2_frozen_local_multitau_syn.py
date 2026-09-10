from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts import experiment_3_0_1_single_tau_objectives as base
from scripts import experiment_4_0_1_multispike_macro_lif as exp401
from scripts import experiment_5_0_local_evidence_objectives as exp50
from scripts import experiment_5_1_boundary_free_temporal_decoder as exp51
from scripts import experiment_5_2_frozen_local_tauR_sweep as exp52


EXPERIMENT_ID = "experiment_5_2_2_frozen_local_multitau_syn"
PROTOCOL_VERSION = "frozen_exp51_l2_multitau_syn_wholecount_v1"
SOURCE_EXPERIMENT_ID = exp51.EXPERIMENT_ID
SOURCE_PROTOCOL_VERSION = exp51.PROTOCOL_VERSION
SEEDS = exp51.SEEDS

TEMPORAL_WIDTH = 128
LOCAL_WIDTH = exp51.LOCAL_WIDTH
READOUTS = ("hidden_count_linear", "output_lif")
PROFILE_SHIFTS: dict[str, tuple[int, ...]] = {
    "s2": (2,),
    "s3": (3,),
    "s4": (4,),
    "s5": (5,),
    "s6": (6,),
    "s7": (7,),
    "s234567": (2, 3, 4, 5, 6, 7),
    "s4567": (4, 5, 6, 7),
    "s567": (5, 6, 7),
    "s67": (6, 7),
}
PROFILE_GROUP_COUNTS: dict[str, tuple[int, ...]] = {
    "s2": (128,),
    "s3": (128,),
    "s4": (128,),
    "s5": (128,),
    "s6": (128,),
    "s7": (128,),
    "s234567": (22, 21, 21, 21, 21, 22),
    "s4567": (32, 32, 32, 32),
    "s567": (43, 42, 43),
    "s67": (64, 64),
}
INTERVENTIONS = (
    "normal",
    "resetall250",
    "reset234",
    "reset2345",
    "reset23456",
    "reset67",
    "reset567",
)
RESET_REQUESTS: dict[str, tuple[int, ...] | None] = {
    "normal": (),
    "resetall250": None,
    "reset234": (2, 3, 4),
    "reset2345": (2, 3, 4, 5),
    "reset23456": (2, 3, 4, 5, 6),
    "reset67": (6, 7),
    "reset567": (5, 6, 7),
}
PROBES = (
    "l3_whole_count",
    "l3_fixed250_ordered",
    "l3_relative10_ordered",
)
LOCAL_REFERENCE_PROBES = (
    "local_whole_count",
    "local_fixed250_ordered",
    "local_relative10_ordered",
)

TAU_MEM_MS = exp50.TAU_MEM_MS
THRESHOLD = exp50.THRESHOLD
SURROGATE_SLOPE = exp50.SURROGATE_SLOPE
RESET = exp50.RESET
OUTPUT_CAP = 1
HIDDEN_CAP = 1
EPOCHS = exp51.EPOCHS
BATCH_SIZE = exp51.BATCH_SIZE
LR = exp51.LR
WEIGHT_DECAY = 0.0
EPS = 1e-8
FIXED250_MS = 250.0


@dataclass(frozen=True)
class RunSpec:
    profile: str
    readout: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.profile}__{self.readout}__seed{self.seed}"


@dataclass(frozen=True)
class Config:
    repo_root: Path
    results_dir: Path
    device: str = "cpu"
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    threads: int = 1


def find_repo_root(start: Path | None = None) -> Path:
    return exp51.find_repo_root(start)


def results_dir(repo_root: Path) -> Path:
    return repo_root / "notebooks" / "artifacts" / EXPERIMENT_ID / PROTOCOL_VERSION


def source_results_dir(repo_root: Path) -> Path:
    return exp51.results_dir(repo_root)


def checkpoint_path(root: Path, spec: RunSpec) -> Path:
    return root / "checkpoints" / f"{spec.key}.pt"


def evaluation_path(root: Path, spec: RunSpec) -> Path:
    return root / "evaluations" / f"{spec.key}.json"


def history_path(root: Path, spec: RunSpec) -> Path:
    return root / "histories" / f"{spec.key}.csv"


def local_reference_path(root: Path, seed: int) -> Path:
    return root / "local_references" / f"seed{seed}.json"


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(profile, readout, seed)
        for profile in PROFILE_SHIFTS
        for readout in READOUTS
        for seed in SEEDS
    ]


def profile_shifts(profile: str) -> tuple[int, ...]:
    if profile not in PROFILE_SHIFTS:
        raise ValueError(f"Unknown temporal profile: {profile}")
    return PROFILE_SHIFTS[profile]


def profile_group_counts(profile: str) -> tuple[int, ...]:
    if profile not in PROFILE_GROUP_COUNTS:
        raise ValueError(f"Unknown temporal profile: {profile}")
    counts = PROFILE_GROUP_COUNTS[profile]
    if len(counts) != len(profile_shifts(profile)) or sum(counts) != TEMPORAL_WIDTH:
        raise ValueError(f"Invalid group allocation for {profile}: {counts}")
    return counts


def profile_group_slices(profile: str) -> dict[int, slice]:
    out: dict[int, slice] = {}
    start = 0
    for shift, count in zip(profile_shifts(profile), profile_group_counts(profile), strict=True):
        out[shift] = slice(start, start + count)
        start += count
    if start != TEMPORAL_WIDTH:
        raise ValueError(f"Profile {profile} allocated {start} neurons, expected {TEMPORAL_WIDTH}")
    return out


def alpha_vector(profile: str) -> torch.Tensor:
    values = torch.empty(TEMPORAL_WIDTH, dtype=torch.float32)
    for shift, group in profile_group_slices(profile).items():
        values[group] = base.alpha(shift)
    return values


def tau_syn_ms_from_shift(shift: int, fs: float) -> float:
    alpha = float(base.alpha(shift))
    return -(1000.0 / float(fs)) / math.log(alpha)


def tau_syn_profile_ms(profile: str, fs: float) -> tuple[float, ...]:
    return tuple(tau_syn_ms_from_shift(shift, fs) for shift in profile_shifts(profile))


def effective_reset_shifts(profile: str, intervention: str) -> tuple[int, ...]:
    if intervention not in RESET_REQUESTS:
        raise ValueError(f"Unknown intervention: {intervention}")
    shifts = profile_shifts(profile)
    requested = RESET_REQUESTS[intervention]
    if requested is None:
        return shifts
    requested_set = set(requested)
    return tuple(shift for shift in shifts if shift in requested_set)


def effective_reset_neuron_count(profile: str, intervention: str) -> int:
    effective = set(effective_reset_shifts(profile, intervention))
    return sum(
        count
        for shift, count in zip(profile_shifts(profile), profile_group_counts(profile), strict=True)
        if shift in effective
    )


def effective_mask_id(profile: str, intervention: str) -> str:
    shifts = effective_reset_shifts(profile, intervention)
    return "none" if not shifts else "s" + "".join(str(shift) for shift in shifts)


def _source_config(config: Config) -> exp51.Config:
    return exp51.Config(
        repo_root=config.repo_root,
        results_dir=source_results_dir(config.repo_root),
        device=config.device,
        epochs=exp51.EPOCHS,
        batch_size=config.batch_size,
        threads=config.threads,
    )


def load_source_cache(seed: int, data: base.Data, config: Config) -> dict[str, np.ndarray]:
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    return exp51.load_local_cache(seed, data, _source_config(config))


def _loader(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return exp51._loader(X, y, lengths, batch_size, shuffle, seed)


def _partitions(
    data: base.Data,
    cache: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "train": (cache["Xtrain"], data.ytr, data.ltr),
        "val": (cache["Xval"], data.yva, data.lva),
        "test": (cache["Xtest"], data.yte, data.lte),
    }


def _make_loaders(
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    train_shuffle: bool,
) -> dict[str, DataLoader]:
    return {
        split: _loader(
            *partition,
            config.batch_size,
            train_shuffle if split == "train" else False,
            base.dseed(spec.seed, "exp5_2_2", split, "loader"),
        )
        for split, partition in _partitions(data, cache).items()
    }


class SynapticPhaseDecoder(nn.Module):
    """Frozen-L2 temporal decoder with explicit heterogeneous L3 synaptic state."""

    def __init__(
        self,
        profile: str,
        readout: str,
        n_classes: int,
        fs: float,
        bin_steps: int,
    ) -> None:
        super().__init__()
        if profile not in PROFILE_SHIFTS:
            raise ValueError(f"Unknown profile: {profile}")
        if readout not in READOUTS:
            raise ValueError(f"Unknown readout: {readout}")
        self.profile = profile
        self.readout = readout
        self.n_classes = int(n_classes)
        self.fs = float(fs)
        self.bin_steps = int(bin_steps)
        self.hidden_width = TEMPORAL_WIDTH
        beta = exp50.beta_value(self.fs)

        self.input_hidden = nn.Linear(LOCAL_WIDTH, TEMPORAL_WIDTH, bias=False)
        self.output_linear = nn.Linear(TEMPORAL_WIDTH, self.n_classes, bias=False)
        self.hidden_lif = exp401.MacroMultiSpikeLIF(
            beta=beta,
            threshold=THRESHOLD,
            max_spikes_per_dt=HIDDEN_CAP,
            surrogate_slope=SURROGATE_SLOPE,
        )
        self.output_lif = (
            exp401.MacroMultiSpikeLIF(
                beta=beta,
                threshold=THRESHOLD,
                max_spikes_per_dt=OUTPUT_CAP,
                surrogate_slope=SURROGATE_SLOPE,
            )
            if readout == "output_lif"
            else None
        )
        self.register_buffer("alpha", alpha_vector(profile))

    def _reset_mask(self, reset_shifts: tuple[int, ...], device: torch.device) -> torch.Tensor:
        mask = torch.zeros(TEMPORAL_WIDTH, dtype=torch.bool, device=device)
        slices = profile_group_slices(self.profile)
        for shift in reset_shifts:
            if shift in slices:
                mask[slices[shift]] = True
        return mask

    def forward_trajectory(
        self,
        x: torch.Tensor,
        reset_shifts: tuple[int, ...] = (),
    ) -> dict[str, torch.Tensor]:
        batch, n_steps, channels = x.shape
        if channels != LOCAL_WIDTH:
            raise ValueError(f"Expected {LOCAL_WIDTH} frozen L2 channels, got {channels}")
        syn = torch.zeros(batch, TEMPORAL_WIDTH, device=x.device, dtype=x.dtype)
        mem = torch.zeros_like(syn)
        output_mem = torch.zeros(batch, self.n_classes, device=x.device, dtype=x.dtype)
        reset_mask = self._reset_mask(reset_shifts, x.device)

        spike_parts: list[torch.Tensor] = []
        syn_parts: list[torch.Tensor] = []
        mem_parts: list[torch.Tensor] = []
        pre_lif_parts: list[torch.Tensor] = []
        output_parts: list[torch.Tensor] = []

        for timestep in range(n_steps):
            if timestep > 0 and timestep % self.bin_steps == 0 and bool(reset_mask.any()):
                mask = reset_mask.unsqueeze(0)
                syn = syn.masked_fill(mask, 0.0)
                mem = mem.masked_fill(mask, 0.0)
            syn = self.alpha * syn + self.input_hidden(x[:, timestep])
            hidden_spike, mem, _ = self.hidden_lif(syn, mem)
            pre_lif = self.output_linear(hidden_spike)
            if self.output_lif is not None:
                output_spike, output_mem, _ = self.output_lif(pre_lif, output_mem)
                output_parts.append(output_spike)
            spike_parts.append(hidden_spike)
            syn_parts.append(syn)
            mem_parts.append(mem)
            pre_lif_parts.append(pre_lif)

        payload = {
            "hidden_spikes": torch.stack(spike_parts, dim=1),
            "hidden_synaptic": torch.stack(syn_parts, dim=1),
            "hidden_membranes": torch.stack(mem_parts, dim=1),
            "pre_lif_logits": torch.stack(pre_lif_parts, dim=1),
        }
        if output_parts:
            payload["output_spikes"] = torch.stack(output_parts, dim=1)
        return payload

    def native_logits(self, trajectory: dict[str, torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
        if self.readout == "hidden_count_linear":
            hidden_count = exp50.valid_sum(trajectory["hidden_spikes"], lengths)
            return self.output_linear(hidden_count)
        output_spikes = trajectory.get("output_spikes")
        if output_spikes is None:
            raise RuntimeError("output_lif readout requires output spikes")
        return exp50.valid_sum(output_spikes, lengths)

    def pre_lif_logits(self, trajectory: dict[str, torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
        return exp50.valid_sum(trajectory["pre_lif_logits"], lengths)

    def loss_logits(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.forward_trajectory(x)
        logits = self.native_logits(trajectory, lengths)
        return F.cross_entropy(logits, y), logits


def _initialize_model(
    spec: RunSpec,
    data: base.Data,
    device: torch.device,
) -> SynapticPhaseDecoder:
    base.seed_all(base.dseed(spec.seed, "exp5_2_2", "constructor"))
    model = SynapticPhaseDecoder(
        spec.profile,
        spec.readout,
        len(data.labels),
        data.fs,
        data.bin_steps,
    ).to(device)
    base.seed_all(base.dseed(spec.seed, "exp5_2_2", "W3_init"))
    model.input_hidden.reset_parameters()
    base.seed_all(base.dseed(spec.seed, "exp5_2_2", "Wo_init"))
    model.output_linear.reset_parameters()
    return model


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return base.metrics(y_true, y_pred)


def evaluate_native_only(
    model: SynapticPhaseDecoder,
    loader: DataLoader,
    device: torch.device,
    reset_shifts: tuple[int, ...] = (),
) -> dict[str, float]:
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            trajectory = model.forward_trajectory(l2, reset_shifts=reset_shifts)
            logits = model.native_logits(trajectory, lengths)
            loss = F.cross_entropy(logits, y)
            n = len(y)
            n_total += n
            loss_sum += float(loss.item()) * n
            y_true_parts.append(y.cpu().numpy())
            y_pred_parts.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))
    metrics["loss"] = loss_sum / max(n_total, 1)
    return metrics


def _activity_accumulators(profile: str) -> dict[str, dict[str, object]]:
    groups: dict[str, slice] = {"all": slice(0, TEMPORAL_WIDTH)}
    groups.update({str(shift): group for shift, group in profile_group_slices(profile).items()})
    return {
        name: {
            "slice": group,
            "events": 0.0,
            "entries": 0.0,
            "samples": 0,
            "neuron_totals": np.zeros(group.stop - group.start, dtype=np.float64),
            "syn_abs": 0.0,
            "syn_sq": 0.0,
            "mem_abs": 0.0,
            "mem_sq": 0.0,
        }
        for name, group in groups.items()
    }


def _update_activity(
    accumulators: dict[str, dict[str, object]],
    trajectory: dict[str, torch.Tensor],
    lengths: torch.Tensor,
) -> None:
    spikes = trajectory["hidden_spikes"]
    syn = trajectory["hidden_synaptic"]
    mem = trajectory["hidden_membranes"]
    valid = base.mask(lengths, spikes.shape[1]).to(spikes.dtype).unsqueeze(-1)
    n_samples = int(spikes.shape[0])
    for acc in accumulators.values():
        group = acc["slice"]
        if not isinstance(group, slice):
            raise TypeError("Activity group slice is invalid")
        group_spikes = spikes[:, :, group]
        group_syn = syn[:, :, group]
        group_mem = mem[:, :, group]
        masked_spikes = group_spikes * valid
        width = group_spikes.shape[-1]
        entries = float(valid.sum().item()) * width
        acc["events"] = float(acc["events"]) + float(masked_spikes.sum().item())
        acc["entries"] = float(acc["entries"]) + entries
        acc["samples"] = int(acc["samples"]) + n_samples
        neuron_totals = masked_spikes.sum(dim=(0, 1)).detach().cpu().numpy()
        acc["neuron_totals"] = np.asarray(acc["neuron_totals"]) + neuron_totals
        mask = valid.expand_as(group_syn)
        acc["syn_abs"] = float(acc["syn_abs"]) + float((group_syn.abs() * mask).sum().item())
        acc["syn_sq"] = float(acc["syn_sq"]) + float(((group_syn * group_syn) * mask).sum().item())
        acc["mem_abs"] = float(acc["mem_abs"]) + float((group_mem.abs() * mask).sum().item())
        acc["mem_sq"] = float(acc["mem_sq"]) + float(((group_mem * group_mem) * mask).sum().item())


def _finalize_activity(
    accumulators: dict[str, dict[str, object]],
    fs: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shift, acc in accumulators.items():
        group = acc["slice"]
        if not isinstance(group, slice):
            raise TypeError("Activity group slice is invalid")
        width = group.stop - group.start
        events = float(acc["events"])
        entries = float(acc["entries"])
        samples = int(acc["samples"])
        neuron_totals = np.asarray(acc["neuron_totals"], dtype=np.float64)
        rows.append(
            {
                "shift": shift,
                "n_neurons": width,
                "event_rate_hz": events * float(fs) / max(entries, EPS),
                "nonzero_fraction": events / max(entries, EPS),
                "active_neuron_fraction": float(np.mean(neuron_totals > 0.0)),
                "events_per_neuron_gesture": events / max(samples * width, 1),
                "mean_abs_syn": float(acc["syn_abs"]) / max(entries, EPS),
                "rms_syn": math.sqrt(float(acc["syn_sq"]) / max(entries, EPS)),
                "mean_abs_mem": float(acc["mem_abs"]) / max(entries, EPS),
                "rms_mem": math.sqrt(float(acc["mem_sq"]) / max(entries, EPS)),
            }
        )
    return rows


def _evaluate_split_detailed(
    model: SynapticPhaseDecoder,
    loader: DataLoader,
    device: torch.device,
    data: base.Data,
    reset_shifts: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray], list[dict[str, object]]]:
    model.eval()
    y_parts: list[np.ndarray] = []
    native_pred_parts: list[np.ndarray] = []
    pre_pred_parts: list[np.ndarray] = []
    whole_parts: list[np.ndarray] = []
    fixed_parts: list[np.ndarray] = []
    relative_parts: list[np.ndarray] = []
    native_loss_sum = 0.0
    pre_loss_sum = 0.0
    n_total = 0
    activity = _activity_accumulators(model.profile)

    with torch.no_grad():
        for l2, y, lengths in loader:
            l2 = l2.to(device)
            y_device = y.to(device)
            lengths_device = lengths.to(device)
            trajectory = model.forward_trajectory(l2, reset_shifts=reset_shifts)
            native_logits = model.native_logits(trajectory, lengths_device)
            pre_logits = model.pre_lif_logits(trajectory, lengths_device)
            n = len(y)
            n_total += n
            native_loss_sum += float(F.cross_entropy(native_logits, y_device).item()) * n
            pre_loss_sum += float(F.cross_entropy(pre_logits, y_device).item()) * n
            y_parts.append(y.numpy())
            native_pred_parts.append(native_logits.argmax(dim=1).cpu().numpy())
            pre_pred_parts.append(pre_logits.argmax(dim=1).cpu().numpy())

            spikes = trajectory["hidden_spikes"]
            whole_parts.append(exp50.valid_sum(spikes, lengths_device).cpu().numpy())
            fixed_parts.append(
                base.fixed_counts(spikes, lengths_device, data.bin_steps)
                .flatten(start_dim=1)
                .cpu()
                .numpy()
            )
            relative_parts.append(
                base.relative_counts(spikes, lengths_device, base.N_REL)
                .flatten(start_dim=1)
                .cpu()
                .numpy()
            )
            _update_activity(activity, trajectory, lengths_device)

    y_true = np.concatenate(y_parts)
    native = _classification_metrics(y_true, np.concatenate(native_pred_parts))
    native["loss"] = native_loss_sum / max(n_total, 1)
    pre_lif = _classification_metrics(y_true, np.concatenate(pre_pred_parts))
    pre_lif["loss"] = pre_loss_sum / max(n_total, 1)
    features = {
        "y": y_true,
        "l3_whole_count": np.concatenate(whole_parts),
        "l3_fixed250_ordered": np.concatenate(fixed_parts),
        "l3_relative10_ordered": np.concatenate(relative_parts),
    }
    return native, pre_lif, features, _finalize_activity(activity, data.fs)


def collect_intervention(
    model: SynapticPhaseDecoder,
    data: base.Data,
    cache: dict[str, np.ndarray],
    spec: RunSpec,
    config: Config,
    reset_shifts: tuple[int, ...],
) -> dict[str, object]:
    loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    native: dict[str, dict[str, float]] = {}
    pre_lif: dict[str, dict[str, float]] = {}
    features: dict[str, dict[str, np.ndarray]] = {}
    activity_rows: list[dict[str, object]] = []
    for split, loader in loaders.items():
        native_split, pre_split, feature_split, activity_split = _evaluate_split_detailed(
            model, loader, device, data, reset_shifts
        )
        native[split] = native_split
        pre_lif[split] = pre_split
        features[split] = feature_split
        for row in activity_split:
            activity_rows.append({"split": split, **row})
    probes = [
        {"probe_type": probe, **exp52._probe_metrics(features, probe, spec.seed)}
        for probe in PROBES
    ]
    return {
        "reset_shifts": reset_shifts,
        "native": native,
        "pre_lif": pre_lif,
        "probes": probes,
        "activity": activity_rows,
    }


def _local_reference_features(
    data: base.Data,
    cache: dict[str, np.ndarray],
    seed: int,
    config: Config,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    device = torch.device(config.device)
    for split, (X, y, lengths) in _partitions(data, cache).items():
        loader = _loader(
            X,
            y,
            lengths,
            config.batch_size,
            False,
            base.dseed(seed, "exp5_2_2", split, "local_reference_loader"),
        )
        y_parts: list[np.ndarray] = []
        whole_parts: list[np.ndarray] = []
        fixed_parts: list[np.ndarray] = []
        relative_parts: list[np.ndarray] = []
        with torch.no_grad():
            for l2, yb, lb in loader:
                l2 = l2.to(device)
                lb_device = lb.to(device)
                whole_parts.append(exp50.valid_sum(l2, lb_device).cpu().numpy())
                fixed_parts.append(
                    base.fixed_counts(l2, lb_device, data.bin_steps)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                )
                relative_parts.append(
                    base.relative_counts(l2, lb_device, base.N_REL)
                    .flatten(start_dim=1)
                    .cpu()
                    .numpy()
                )
                y_parts.append(yb.numpy())
        out[split] = {
            "y": np.concatenate(y_parts),
            "local_whole_count": np.concatenate(whole_parts),
            "local_fixed250_ordered": np.concatenate(fixed_parts),
            "local_relative10_ordered": np.concatenate(relative_parts),
        }
    return out


def prepare_local_seed(seed: int, data: base.Data, config: Config, force: bool = False) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unknown seed: {seed}")
    source_config = _source_config(config)
    exp51.prepare_local_seed(seed, source_config, force=force)
    cache = exp51.load_local_cache(seed, data, source_config)
    destination = local_reference_path(config.results_dir, seed)
    if destination.exists() and not force:
        return destination
    features = _local_reference_features(data, cache, seed, config)
    probes = [
        {"probe_type": probe, **exp52._probe_metrics(features, probe, seed)}
        for probe in LOCAL_REFERENCE_PROBES
    ]
    source_meta_path = exp51.local_cache_meta_path(source_config.results_dir, seed)
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    _save_json(
        destination,
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "seed": seed,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_protocol_version": SOURCE_PROTOCOL_VERSION,
            "source_cache": str(exp51.local_cache_path(source_config.results_dir, seed).relative_to(config.repo_root)),
            "source_checkpoint": source_meta["source_checkpoint"],
            "probes": probes,
        },
    )
    return destination


def parameter_counts(n_classes: int) -> dict[str, int]:
    W3 = LOCAL_WIDTH * TEMPORAL_WIDTH
    Wo = TEMPORAL_WIDTH * n_classes
    return {"W3": W3, "Wo": Wo, "trainable_total": W3 + Wo}


def provenance(spec: RunSpec, data: base.Data, config: Config) -> dict[str, object]:
    slices = profile_group_slices(spec.profile)
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": spec.seed,
        "split_seed": base.SPLIT_SEED,
        "profile": spec.profile,
        "shifts_syn": profile_shifts(spec.profile),
        "tau_syn_ms": tau_syn_profile_ms(spec.profile, data.fs),
        "group_widths": {
            str(shift): int(group.stop - group.start) for shift, group in slices.items()
        },
        "readout": spec.readout,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_protocol_version": SOURCE_PROTOCOL_VERSION,
        "source_local_frozen": True,
        "local_width": LOCAL_WIDTH,
        "local_l1_shifts": (2, 3, 4),
        "local_l2_shifts": (2, 3, 4),
        "temporal_width": TEMPORAL_WIDTH,
        "sampling_rate_hz": float(data.fs),
        "tau_mem_l3_ms": TAU_MEM_MS,
        "tau_mem_output_ms": TAU_MEM_MS if spec.readout == "output_lif" else None,
        "output_synaptic_state": False,
        "threshold": THRESHOLD,
        "reset": RESET,
        "hidden_cap": HIDDEN_CAP,
        "output_cap": OUTPUT_CAP if spec.readout == "output_lif" else None,
        "objective": "whole_count_ce",
        "native_readout": (
            "valid L3 channel-wise spike sum -> Linear(128,12,bias=False) -> CE"
            if spec.readout == "hidden_count_linear"
            else "L3 spikes -> Linear(128,12,bias=False) -> 12 short-membrane output LIF -> valid output spike count -> CE"
        ),
        "trainable_parameters": "W3 and Wo only; frozen L1/L2 source cache",
        "parameter_counts": parameter_counts(len(data.labels)),
        "checkpoint_selection": "max validation native BA; tie-break minimum validation native CE",
        "interventions": INTERVENTIONS,
        "intervention_period_ms": FIXED250_MS,
        "intervention_contract": "inference only; before every 16th raw64 step, clear selected L3 synaptic I and membrane U; never reset output state",
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "train_users": data.split["train_users"],
        "val_users": data.split["val_users"],
        "test_users": data.split["test_users"],
        "labels": data.labels,
    }


def train_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> Path:
    destination = checkpoint_path(config.results_dir, spec)
    if destination.exists() and not force:
        return destination
    cache = load_source_cache(spec.seed, data, config)
    train_loaders = _make_loaders(data, cache, spec, config, train_shuffle=True)
    eval_loaders = _make_loaders(data, cache, spec, config, train_shuffle=False)
    device = torch.device(config.device)
    torch.set_num_threads(config.threads)
    model = _initialize_model(spec, data, device)
    optimizer = torch.optim.Adam(
        [model.input_hidden.weight, model.output_linear.weight],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_ba = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for l2, y, lengths in train_loaders["train"]:
            l2 = l2.to(device)
            y = y.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits = model.loss_logits(l2, lengths, y)
            loss.backward()
            optimizer.step()
            n = len(y)
            train_n += n
            train_loss_sum += float(loss.item()) * n
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_metrics = _classification_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_metrics = evaluate_native_only(model, eval_loaders["val"], device)
        val_ba = float(val_metrics["balanced_accuracy"])
        val_loss = float(val_metrics["loss"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_n, 1),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_loss,
                "val_balanced_accuracy": val_ba,
            }
        )
        improved = val_ba > best_val_ba + 1e-12 or (
            abs(val_ba - best_val_ba) <= 1e-12 and val_loss < best_val_loss - 1e-12
        )
        if improved:
            best_val_ba = val_ba
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "provenance": provenance(spec, data, config),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val_ba,
            "best_val_loss": best_val_loss,
            "state_dict": best_state,
        },
        destination,
    )
    hpath = history_path(config.results_dir, spec)
    hpath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hpath, index=False)
    return destination


def load_model(spec: RunSpec, data: base.Data, config: Config) -> tuple[SynapticPhaseDecoder, dict[str, object]]:
    path = checkpoint_path(config.results_dir, spec)
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp5.2.2 checkpoint: {path}")
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Wrong Exp5.2.2 checkpoint identity: {path}")
    if payload.get("spec") != spec.__dict__:
        raise ValueError(f"Checkpoint spec mismatch: {path}")
    model = SynapticPhaseDecoder(
        spec.profile,
        spec.readout,
        len(data.labels),
        data.fs,
        data.bin_steps,
    ).to(torch.device(config.device))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def evaluate_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    destination = evaluation_path(config.results_dir, spec)
    if destination.exists() and not force:
        return json.loads(destination.read_text(encoding="utf-8"))
    cache = load_source_cache(spec.seed, data, config)
    model, checkpoint = load_model(spec, data, config)

    unique_results: dict[tuple[int, ...], dict[str, object]] = {}
    logical_results: list[dict[str, object]] = []
    for intervention in INTERVENTIONS:
        effective = effective_reset_shifts(spec.profile, intervention)
        if effective not in unique_results:
            unique_results[effective] = collect_intervention(
                model, data, cache, spec, config, effective
            )
        result = unique_results[effective]
        requested = RESET_REQUESTS[intervention]
        logical_results.append(
            {
                "intervention": intervention,
                "requested_reset_shifts": "all" if requested is None else requested,
                "effective_reset_shifts": effective,
                "effective_reset_neuron_count": effective_reset_neuron_count(spec.profile, intervention),
                "effective_mask_id": effective_mask_id(spec.profile, intervention),
                "is_noop": len(effective) == 0,
                "native": result["native"],
                "pre_lif": result["pre_lif"],
                "probes": result["probes"],
                "activity": result["activity"],
            }
        )

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "spec": spec.__dict__,
        "provenance": checkpoint["provenance"],
        "best_epoch": checkpoint["best_epoch"],
        "interventions": logical_results,
        "unique_effective_forward_count": len(unique_results),
    }
    _save_json(destination, payload)
    return payload


def run_one(spec: RunSpec, data: base.Data, config: Config, force: bool = False) -> dict[str, object]:
    train_one(spec, data, config, force=force)
    return evaluate_one(spec, data, config, force=force)


def _probe_lookup(payload: dict[str, object], probe_type: str) -> dict[str, object]:
    probes = payload["probes"]
    if not isinstance(probes, list):
        raise TypeError("Invalid probe payload")
    for probe in probes:
        if isinstance(probe, dict) and probe.get("probe_type") == probe_type:
            return probe
    raise KeyError(probe_type)


def finalize_experiment(repo_root: Path) -> dict[str, Path]:
    root = results_dir(repo_root)
    run_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    activity_rows: list[dict[str, object]] = []
    history_parts: list[pd.DataFrame] = []

    for spec in run_specs():
        path = evaluation_path(root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.2.2 evaluation: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or payload.get("spec") != spec.__dict__:
            raise ValueError(f"Evaluation identity mismatch: {path}")
        interventions = payload.get("interventions")
        if not isinstance(interventions, list) or len(interventions) != len(INTERVENTIONS):
            raise ValueError(f"Incomplete intervention matrix: {path}")
        by_name = {row["intervention"]: row for row in interventions}
        normal = by_name["normal"]
        run_row: dict[str, object] = {
            "profile": spec.profile,
            "readout": spec.readout,
            "seed": spec.seed,
            "shifts_syn": ",".join(str(v) for v in profile_shifts(spec.profile)),
            "tau_syn_ms": ",".join(f"{v:.3f}" for v in tau_syn_profile_ms(spec.profile, 64.0)),
            "best_epoch": payload["best_epoch"],
            "unique_effective_forward_count": payload["unique_effective_forward_count"],
        }
        for split in ("train", "val", "test"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "loss"):
                run_row[f"native_{split}_{metric}"] = normal["native"][split][metric]
                run_row[f"pre_lif_{split}_{metric}"] = normal["pre_lif"][split][metric]
        for probe in PROBES:
            p = _probe_lookup(normal, probe)
            run_row[f"{probe}_val_ba"] = p["probe_val_balanced_accuracy"]
            run_row[f"{probe}_test_ba"] = p["probe_test_balanced_accuracy"]
            run_row[f"{probe}_C"] = p["probe_C"]
        run_rows.append(run_row)

        for row in interventions:
            ablation: dict[str, object] = {
                "profile": spec.profile,
                "readout": spec.readout,
                "seed": spec.seed,
                "intervention": row["intervention"],
                "requested_reset_shifts": row["requested_reset_shifts"],
                "effective_reset_shifts": ",".join(str(v) for v in row["effective_reset_shifts"]),
                "effective_reset_neuron_count": row["effective_reset_neuron_count"],
                "effective_mask_id": row["effective_mask_id"],
                "is_noop": row["is_noop"],
            }
            for split in ("train", "val", "test"):
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "loss"):
                    ablation[f"native_{split}_{metric}"] = row["native"][split][metric]
                    ablation[f"pre_lif_{split}_{metric}"] = row["pre_lif"][split][metric]
            ablation_rows.append(ablation)

            for probe in PROBES:
                p = _probe_lookup(row, probe)
                probe_rows.append(
                    {
                        "profile": spec.profile,
                        "readout": spec.readout,
                        "seed": spec.seed,
                        "intervention": row["intervention"],
                        "effective_mask_id": row["effective_mask_id"],
                        "probe_type": probe,
                        "feature_dim": p.get("feature_dim"),
                        "probe_C": p["probe_C"],
                        "val_ba": p["probe_val_balanced_accuracy"],
                        "test_ba": p["probe_test_balanced_accuracy"],
                        "test_accuracy": p.get("probe_test_accuracy"),
                        "test_macro_f1": p.get("probe_test_macro_f1"),
                    }
                )
            for activity in row["activity"]:
                activity_rows.append(
                    {
                        "profile": spec.profile,
                        "readout": spec.readout,
                        "seed": spec.seed,
                        "intervention": row["intervention"],
                        "effective_mask_id": row["effective_mask_id"],
                        **activity,
                    }
                )

        hpath = history_path(root, spec)
        if not hpath.exists():
            raise FileNotFoundError(f"Missing Exp5.2.2 history: {hpath}")
        history = pd.read_csv(hpath)
        history.insert(0, "seed", spec.seed)
        history.insert(0, "readout", spec.readout)
        history.insert(0, "profile", spec.profile)
        history_parts.append(history)

    local_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = local_reference_path(root, seed)
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp5.2.2 local reference: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != PROTOCOL_VERSION or int(payload.get("seed")) != seed:
            raise ValueError(f"Local reference identity mismatch: {path}")
        for probe in payload["probes"]:
            local_rows.append(
                {
                    "seed": seed,
                    "probe_type": probe["probe_type"],
                    "probe_C": probe["probe_C"],
                    "val_ba": probe["probe_val_balanced_accuracy"],
                    "test_ba": probe["probe_test_balanced_accuracy"],
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": root / "runs.csv",
        "ablation_runs": root / "ablation_runs.csv",
        "probes": root / "probes.csv",
        "activity": root / "activity.csv",
        "histories": root / "histories.csv",
        "local_reference": root / "local_reference.csv",
        "manifest": root / "manifest.json",
    }
    pd.DataFrame(run_rows).to_csv(outputs["runs"], index=False)
    pd.DataFrame(ablation_rows).to_csv(outputs["ablation_runs"], index=False)
    pd.DataFrame(probe_rows).to_csv(outputs["probes"], index=False)
    pd.DataFrame(activity_rows).to_csv(outputs["activity"], index=False)
    pd.concat(history_parts, ignore_index=True).to_csv(outputs["histories"], index=False)
    pd.DataFrame(local_rows).to_csv(outputs["local_reference"], index=False)
    _save_json(
        outputs["manifest"],
        {
            "experiment_id": EXPERIMENT_ID,
            "protocol_version": PROTOCOL_VERSION,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "source_protocol_version": SOURCE_PROTOCOL_VERSION,
            "source_cache_policy": "reuse/reproduce the exact Exp5.1 frozen Exp3 L2 caches; never retrain L1/L2 inside a decoder run",
            "profiles": PROFILE_SHIFTS,
            "profile_group_counts": PROFILE_GROUP_COUNTS,
            "readouts": READOUTS,
            "seeds": SEEDS,
            "expected_training_runs": len(run_specs()),
            "interventions": INTERVENTIONS,
            "expected_logical_ablation_rows": len(run_specs()) * len(INTERVENTIONS),
            "reset_period_ms": FIXED250_MS,
            "reset_contract": "inference only; clear selected L3 synaptic I and membrane U before each 250 ms boundary; output LIF state is never reset",
            "probes": PROBES,
            "local_reference_probes": LOCAL_REFERENCE_PROBES,
            "trainable_parameters": "W3 and Wo only",
            "tau_mem_l3_ms": TAU_MEM_MS,
            "tau_mem_output_ms": TAU_MEM_MS,
            "output_synaptic_state": False,
            "checkpoint_selection": "per-run max validation native BA; tie-break validation native CE",
            "aggregation_policy": "per-run tasks train then evaluate the selected checkpoint; finalizer concatenates only; notebook performs validation-only profile selection, paired effects, organization-gap diagnostics, bottleneck analysis, and plots",
            "files": {name: path.name for name, path in outputs.items() if name != "manifest"},
        },
    )
    return outputs


def _config_from_args(args: argparse.Namespace) -> Config:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    return Config(
        repo_root=repo_root,
        results_dir=results_dir(repo_root),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        threads=args.threads,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5.2.2 frozen-local multi-tau synaptic phase-aware readout"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-local")
    prepare.add_argument("--repo-root", type=str, default=None)
    prepare.add_argument("--device", type=str, default="cpu")
    prepare.add_argument("--threads", type=int, default=1)
    prepare.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    prepare.add_argument("--epochs", type=int, default=EPOCHS)
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--array-task-id", type=int, required=True)

    run = subparsers.add_parser("run-one")
    run.add_argument("--repo-root", type=str, default=None)
    run.add_argument("--device", type=str, default="cpu")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    run.add_argument("--epochs", type=int, default=EPOCHS)
    run.add_argument("--force", action="store_true")
    run.add_argument("--array-task-id", type=int, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--repo-root", type=str, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "finalize":
        repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
        outputs = finalize_experiment(repo_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return

    config = _config_from_args(args)
    data = base.prepare_data(config.repo_root)
    if args.command == "prepare-local":
        task_id = int(args.array_task_id)
        if not 0 <= task_id < len(SEEDS):
            raise IndexError(f"prepare-local task {task_id} outside 0..{len(SEEDS)-1}")
        path = prepare_local_seed(SEEDS[task_id], data, config, force=args.force)
        print(path)
        return

    specs = run_specs()
    task_id = int(args.array_task_id)
    if not 0 <= task_id < len(specs):
        raise IndexError(f"run-one task {task_id} outside 0..{len(specs)-1}")
    payload = run_one(specs[task_id], data, config, force=args.force)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
