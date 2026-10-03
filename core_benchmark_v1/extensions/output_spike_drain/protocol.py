"""Independent run manifest for the fully-drained spike-output extension."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BETAS = tuple(round(i / 10, 1) for i in range(11))
SEEDS = (11, 23, 37)
DRAIN_BETA = 1.0
MAX_DRAIN_STEPS = 1024


@dataclass(frozen=True)
class ExtensionRun:
    beta: float
    seed: int

    def validate(self) -> None:
        if self.beta not in BETAS:
            raise ValueError(self.beta)
        if self.seed not in SEEDS:
            raise ValueError(self.seed)

    @property
    def beta_tag(self) -> str:
        return f"B{int(round(self.beta * 100)):03d}"

    @property
    def key(self) -> str:
        return f"DRAIN_{self.beta_tag}__seed{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def runs() -> list[ExtensionRun]:
    result = [ExtensionRun(beta, seed) for seed in SEEDS for beta in BETAS]
    if len(result) != 33 or len({r.key for r in result}) != 33:
        raise AssertionError("Drain extension run matrix must contain 33 unique runs")
    return result
