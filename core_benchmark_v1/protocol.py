"""Immutable protocol and deduplicated, dependency-aware experiment manifest."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

VERSION = "core_benchmark_v1.0"
SPLITS = ("train", "val", "test")
BLOCKS = ("01_objective", "02_tau", "03_depth", "04_readout")
TAUS = {
    "T1": ((1, 2, 3), (2, 3, 4)),
    "T2": ((1, 2, 3), (1, 2, 3)),
    "T3": ((1, 2, 3), (3, 4, 5)),
    "T4": ((2, 3, 4), (3, 4, 5)),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def paired_seed(seed: int, role: str) -> int:
    return int(digest([VERSION, int(seed), role])[:8], 16) % (2**31 - 1)


@dataclass(frozen=True)
class Protocol:
    version: str = VERSION
    profile: str = "production"
    dataset_roots: tuple[str, ...] = (
        "outputs/action0_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
        "outputs/action1_wavelets_0e5_1_2_4_8_sr_64/low-pass/aligned-board-events/segmentation_padded",
    )
    labels: tuple[str, ...] = ("A", "B", "C", "D", "E", "G", "H", "I", "J", "K", "L", "X")
    seeds: tuple[int, ...] = (11, 23, 37)
    split_seed: int = 12345
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    # Empty: materialize the legacy seed-12345 split ONCE, then freeze its users.
    train_users: tuple[str, ...] = ()
    val_users: tuple[str, ...] = ()
    test_users: tuple[str, ...] = ()
    fs: float = 64.0
    input_channels: int = 30
    total_channels: int = 36
    steps: int = 256
    width: int = 128
    synaptic_update: str = "unnormalized"
    input_drive_scale: float = 1.0
    tau_mem_ms: float = 22.54
    threshold: float = 0.5
    surrogate_slope: float = 25.0
    batch_size: int = 128
    max_epochs: int = 100
    min_epochs: int = 20
    patience: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    auxiliary_weight: float = 0.1
    wcce_reduction: str = "valid_mean_logits"
    tsce_reduction: str = "per_sample_valid_mean"
    native_bias: bool = False
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    probe_max_iter: int = 3000
    probe_tol: float = 0.0001
    fixed_ms: float = 250.0
    relative_bins: int = 10
    shuffle_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    probe_states: tuple[str, ...] = ("spike", "pre_reset")
    depth_controls: bool = False
    membrane_sweep: bool = False
    readout_adaptation: bool = True
    output_betas: tuple[float, ...] = (1.0, 0.5)
    readout_thresholds: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0)
    diagnostics: bool = True
    diagnostic_cases: tuple[str, ...] = ("O0", "D1")
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 48, 64)
    history_ms: tuple[float, ...] = (50.0, 100.0, 250.0, 500.0, 1000.0)

    def validate(self) -> None:
        if self.version != VERSION or self.profile not in ("production", "smoke"):
            raise ValueError("Unknown protocol version/profile")
        if self.synaptic_update != "unnormalized" or self.input_drive_scale != 1.0:
            raise ValueError("Only I[t] = alpha*I[t-1] + W*x[t] is permitted")
        if self.native_bias or self.wcce_reduction != "valid_mean_logits" or self.tsce_reduction != "per_sample_valid_mean":
            raise ValueError("Native readout/loss geometry is locked")
        positive = (self.fs, self.steps, self.width, self.tau_mem_ms, self.threshold,
                    self.batch_size, self.max_epochs, self.min_epochs, self.patience,
                    self.learning_rate, self.probe_max_iter, self.probe_tol, self.relative_bins)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Protocol sizes/rates must be finite and positive")
        if self.min_epochs > self.max_epochs or not 0 < self.train_fraction < 1 or not 0 < self.val_fraction < 1 - self.train_fraction:
            raise ValueError("Invalid training budget or split fractions")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or len(set(self.labels)) != len(self.labels) or len(self.labels) < 2:
            raise ValueError("Seeds and labels must be unique and nonempty")
        if not self.c_grid or any(c <= 0 or not math.isfinite(c) for c in self.c_grid):
            raise ValueError("Invalid probe C grid")
        if not self.shuffle_seeds or len(set(self.shuffle_seeds)) != len(self.shuffle_seeds):
            raise ValueError("Shuffle seeds must be unique and nonempty")
        if not self.probe_states or not set(self.probe_states) <= {"spike", "pre_reset"}:
            raise ValueError("Unknown probe state")
        if not self.output_betas or any(b not in (0.5, 1.0) for b in self.output_betas):
            raise ValueError("Output controls are IF beta=1 and LIF beta=0.5")
        if not self.readout_thresholds or self.threshold not in self.readout_thresholds or any(v <= 0 for v in self.readout_thresholds):
            raise ValueError("Threshold sweep must contain the native threshold")
        if self.auxiliary_weight < 0 or self.weight_decay < 0:
            raise ValueError("Negative loss/optimizer coefficient")
        if self.fixed_steps < 1 or self.steps % self.fixed_steps:
            raise ValueError("Fixed250 must divide the locked observation horizon")
        users = (self.train_users, self.val_users, self.test_users)
        if any(users) and not all(users):
            raise ValueError("Provide all three explicit user lists, or none")
        flat = [u for group in users for u in group]
        if len(flat) != len(set(flat)):
            raise ValueError("User sets overlap or contain duplicates")
        if self.profile == "production" and (self.seeds != (11, 23, 37) or self.width != 128 or self.fs != 64 or self.input_channels != 30 or self.total_channels != 36 or self.steps != 256):
            raise ValueError("Production geometry/seeds are locked; use profile=smoke for synthetic tests")

    @property
    def fixed_steps(self) -> int:
        return int(round(self.fixed_ms * self.fs / 1000))

    @property
    def fingerprint(self) -> str:
        self.validate()
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Protocol:
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown protocol fields: {sorted(unknown)}")
        default = cls()
        converted = {k: tuple(v) if isinstance(getattr(default, k), tuple) else v for k, v in payload.items()}
        result = cls(**converted)
        result.validate()
        return result

    @classmethod
    def load(cls, path: Path) -> Protocol:
        return cls.from_dict(json.loads(path.read_text()))


@dataclass(frozen=True)
class Run:
    case: str
    seed: int
    block: str
    shifts: tuple[tuple[int, ...], ...] = ((2, 3, 4), (2, 3, 4))
    objective: str = "wcce"
    kind: str = "backbone"
    parent_case: str | None = None
    phase: int = 1
    beta: float | None = None
    l1_tau_mem_ms: float | None = None

    @property
    def key(self) -> str:
        return f"{self.case}__seed{self.seed}"

    @property
    def parent_key(self) -> str | None:
        return None if self.parent_case is None else f"{self.parent_case}__seed{self.seed}"


def runs(p: Protocol) -> list[Run]:
    p.validate()
    result: list[Run] = []
    for seed in p.seeds:
        for case, objective in (("O0", "wcce"), ("O1", "tsce"), ("O2", "wcce_l1_wcce"), ("O3", "wcce_l1_tsce")):
            result.append(Run(case, seed, "01_objective", objective=objective))
        result.extend(Run(case, seed, "02_tau", shifts=shifts) for case, shifts in TAUS.items())
        deep = ((2, 3, 4),) * 3
        result.append(Run("D1", seed, "03_depth", shifts=deep))
        if p.depth_controls:
            result.extend((Run("D2", seed, "03_depth", shifts=deep, parent_case="O0", phase=2),
                           Run("D3", seed, "03_depth", shifts=deep, parent_case="D2", phase=3)))
        if p.membrane_sweep:
            for case, tau in (("M1", 54.0), ("M2", 117.0), ("M3", 242.0)):
                result.append(Run(case, seed, "07_membrane", l1_tau_mem_ms=tau))
        for beta in p.output_betas:
            name = "R_IF" if beta == 1 else "R_LIF"
            result.append(Run(name, seed, "04_readout", kind="readout", parent_case="O0", phase=2, beta=beta))
    if len({r.key for r in result}) != len(result):
        raise ValueError("Duplicate run keys")
    return result


def aliases() -> dict[str, str]:
    return {"REF": "O0", "T0": "O0", "D0": "O0", "M0": "O0", "R0": "O0"}


def phase_runs(p: Protocol, phase: int) -> list[Run]:
    return [r for r in runs(p) if r.phase == phase]
