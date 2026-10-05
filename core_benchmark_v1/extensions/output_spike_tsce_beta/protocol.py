"""Independent run manifest for the spike-only TSCE beta sweep."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BETAS = tuple(round(i / 10, 1) for i in range(11))
MODES = ("valid", "drain")
SEEDS = (11, 23, 37)
DRAIN_BETA = 1.0
MAX_DRAIN_STEPS = 1024


@dataclass(frozen=True)
class ExtensionRun:
    mode: str
    beta: float
    seed: int

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(self.mode)
        if self.beta not in BETAS:
            raise ValueError(self.beta)
        if self.seed not in SEEDS:
            raise ValueError(self.seed)

    @property
    def beta_tag(self) -> str:
        return f"B{int(round(self.beta * 100)):03d}"

    @property
    def key(self) -> str:
        mode_tag = "VALID" if self.mode == "valid" else "DRAIN"
        return f"TSCE_{mode_tag}_{self.beta_tag}__seed{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def runs() -> list[ExtensionRun]:
    result = [
        ExtensionRun(mode, beta, seed)
        for seed in SEEDS
        for mode in MODES
        for beta in BETAS
    ]
    if len(result) != 66 or len({run.key for run in result}) != 66:
        raise AssertionError("Spike-only TSCE beta sweep must contain 66 unique runs")
    return result
