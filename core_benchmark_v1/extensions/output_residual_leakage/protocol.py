"""Independent run manifest for the output residual-leakage extension."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

BETAS = tuple(round(i / 10, 1) for i in range(11))
OBJECTIVES = ("wcce", "tsce")
SEEDS = (11, 23, 37)

@dataclass(frozen=True)
class ExtensionRun:
    objective: str
    beta: float
    seed: int

    def validate(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(self.objective)
        if self.beta not in BETAS:
            raise ValueError(self.beta)
        if self.seed not in SEEDS:
            raise ValueError(self.seed)

    @property
    def beta_tag(self) -> str:
        return f"B{int(round(self.beta * 100)):03d}"

    @property
    def key(self) -> str:
        return f"{self.objective.upper()}_{self.beta_tag}__seed{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def runs() -> list[ExtensionRun]:
    result = [ExtensionRun(objective, beta, seed) for seed in SEEDS for objective in OBJECTIVES for beta in BETAS]
    if len(result) != 66 or len({r.key for r in result}) != 66:
        raise AssertionError("Extension run matrix must contain 66 unique runs")
    return result