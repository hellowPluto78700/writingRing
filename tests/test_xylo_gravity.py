from __future__ import annotations

import builtins

import numpy as np
import pytest

from writingring.xylo_gravity import XyloGravityError, xylo_rotate_and_remove_gravity


def test_missing_xylo_optional_dependency_has_actionable_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def missing_rockpool(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("rockpool"):
            raise ModuleNotFoundError("No module named 'rockpool'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_rockpool)
    with pytest.raises(
        XyloGravityError,
        match=r'Install Xylo support with: pip install -e "\.\[xylo\]"',
    ):
        xylo_rotate_and_remove_gravity(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            sampling_rate_hz=200.0,
        )
