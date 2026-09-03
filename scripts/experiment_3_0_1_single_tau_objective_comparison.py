from __future__ import annotations

from pathlib import Path

from scripts.experiment_3_0_1_single_tau_objectives import *  # noqa: F401,F403
from scripts import experiment_4_0_fixed250_temporal_snn as _exp40


def find_repo_root(start: Path | None = None) -> Path:
    return _exp40.find_repo_root(start)
