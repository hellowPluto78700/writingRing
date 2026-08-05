#!/usr/bin/env python3
"""Generate the checked-in Phase 3 golden fixture from the read-only vendor code.

This development-only script requires torch and snntorch.  The WritingRing
runtime and its ordinary tests consume the generated fixture only; they never
import ``vendor/Neuromorphic-Gravity``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


# The vendor tree is an immutable reference.  Importing it to refresh a golden
# fixture must not create or update its tracked bytecode artifacts.
sys.dont_write_bytecode = True


_SAMPLE_RATE_HZ = 64.0
_FREQUENCIES_HZ = np.array((0.5, 1.0, 2.0, 4.0, 8.0), dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    """Create a deterministic reference fixture from the vendor pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/custom_wavelet_vendor_64hz.npz"),
    )
    args = parser.parse_args(argv)
    modules, acceleration_wavelet, torch = _vendor_components()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frequencies = torch.tensor(_FREQUENCIES_HZ, dtype=torch.float32)

    probe = modules.WaveletIIRFilterModule(
        _SAMPLE_RATE_HZ,
        singleDim=False,
        frequencies=frequencies,
        waveletFunc=acceleration_wavelet,
    )
    fixture: dict[str, np.ndarray] = {
        "sampling_rate_hz": np.array(_SAMPLE_RATE_HZ, dtype=np.float64),
        "frequencies_hz": _FREQUENCIES_HZ,
        "widths_samples": (np.array(_SAMPLE_RATE_HZ) / _FREQUENCIES_HZ).astype(np.int64),
        "prony_numerator": probe.filterBankB.detach().cpu().numpy().T,
        "prony_denominator": np.column_stack(
            (
                np.ones(len(_FREQUENCIES_HZ), dtype=np.float32),
                -probe.filterBankA.detach().cpu().numpy()[0],
            )
        ),
        "torch_version": np.array(torch.__version__),
    }
    for width in fixture["widths_samples"]:
        fixture[f"wavelet_{int(width)}"] = (
            acceleration_wavelet(
                int(width), torch.tensor(int(width), dtype=torch.int64)
            ).detach().cpu().numpy()
        )
    for name, values in _input_cases().items():
        fixture[f"{name}_input"] = values
        fixture[f"{name}_iir"] = _iir_responses(
            modules,
            acceleration_wavelet,
            torch,
            frequencies,
            values,
        )
        events = _events(
            modules,
            acceleration_wavelet,
            torch,
            frequencies,
            values,
        )
        fixture[f"{name}_events"] = events
        fixture[f"{name}_flattened"] = events.reshape(len(events), -1)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **fixture)
    print(f"Wrote vendor fixture: {output}")
    return 0


def _vendor_components() -> tuple[object, object, object]:
    root = Path(__file__).resolve().parents[1]
    source = root / "vendor" / "Neuromorphic-Gravity" / "python-pipeline"
    if not source.is_dir():
        raise RuntimeError(f"vendor source is missing: {source}")
    sys.path.insert(0, str(source))
    try:
        import modules
        from wavelets import accelerationWavelet
        import torch
    except ImportError as error:
        raise RuntimeError(
            "fixture generation requires torch and snntorch; install the development dependencies"
        ) from error
    return modules, accelerationWavelet, torch


def _input_cases() -> dict[str, np.ndarray]:
    sample_count = 96
    time = np.arange(sample_count, dtype=np.float32) / _SAMPLE_RATE_HZ
    impulse = np.zeros((sample_count, 3), dtype=np.float32)
    impulse[0, 0] = 1.0
    impulse[4, 1] = -0.75
    impulse[9, 2] = 0.5
    sine_mixture = np.column_stack(
        (
            np.sin(2.0 * np.pi * 1.0 * time) + 0.25 * np.sin(2.0 * np.pi * 4.0 * time),
            -0.5 * np.sin(2.0 * np.pi * 2.0 * time),
            0.75 * np.cos(2.0 * np.pi * 3.0 * time),
        )
    ).astype(np.float32)
    plateau = np.zeros((sample_count, 3), dtype=np.float32)
    plateau[20:70, 0] = 0.5
    plateau[30:60, 1] = -0.25
    plateau[40:80, 2] = 0.75
    random = np.random.default_rng(202407).normal(0.0, 0.25, size=(sample_count, 3)).astype(np.float32)
    return {
        "impulse": impulse,
        "sine_mixture": sine_mixture,
        "plateau": plateau,
        "random": random,
    }


def _iir_responses(
    modules: object,
    acceleration_wavelet: object,
    torch: object,
    frequencies: object,
    values: np.ndarray,
) -> np.ndarray:
    bank = modules.WaveletIIRFilterModule(
        _SAMPLE_RATE_HZ,
        singleDim=False,
        frequencies=frequencies,
        waveletFunc=acceleration_wavelet,
    )
    return np.stack(
        [bank(torch.tensor(sample, dtype=torch.float32)).detach().cpu().numpy() for sample in values]
    )


def _events(
    modules: object,
    acceleration_wavelet: object,
    torch: object,
    frequencies: object,
    values: np.ndarray,
) -> np.ndarray:
    pipeline = modules.NeuromorphicIMUPipeline(
        _SAMPLE_RATE_HZ,
        globalFrame=True,
        usePower=False,
        singleDim=False,
        frequencies=frequencies,
        waveletFunc=acceleration_wavelet,
    )
    return np.stack(
        [pipeline(torch.tensor(sample, dtype=torch.float32)).detach().cpu().numpy() for sample in values]
    )


if __name__ == "__main__":
    raise SystemExit(main())
