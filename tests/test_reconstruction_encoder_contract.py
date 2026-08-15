import hashlib
import json

import numpy as np
import pytest

from scripts.reconstruct_padded_spike_accel import (
    ReconstructionError as PaddedReconstructionError,
    load_encoder_contract as load_padded_encoder_contract,
    reconstruction_kernels as padded_reconstruction_kernels,
)
from scripts.reconstruct_segmented_spike_accel import (
    ReconstructionError as SegmentedReconstructionError,
    load_encoder_contract as load_segmented_encoder_contract,
    reconstruction_kernels as segmented_reconstruction_kernels,
)


def _summary(widths=(200, 100, 50, 25, 12)):
    spec = {
        "schema": "custom_wavelet_encoder_spec_v1",
        "frequencies_hz": [1.0, 2.0, 4.0, 8.0, 16.0],
        "wavelet_widths_samples": list(widths),
    }
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return {"spike_encoder": spec, "spike_encoder_spec_sha256": digest}


def test_reconstruction_uses_authoritative_non_integer_ratio_widths() -> None:
    summary = _summary()
    _, digest, frequencies, widths = load_segmented_encoder_contract(summary, path=__file__)
    assert digest == summary["spike_encoder_spec_sha256"]
    assert frequencies == (1.0, 2.0, 4.0, 8.0, 16.0)
    assert widths == (200, 100, 50, 25, 12)
    assert summary["spike_encoder_spec_sha256"] == digest
    assert len(segmented_reconstruction_kernels(wavelet_widths_samples=widths, frequencies_hz=frequencies)) == 5
    assert len(padded_reconstruction_kernels(widths, frequencies)) == 5


@pytest.mark.parametrize("loader, error", [
    (load_segmented_encoder_contract, SegmentedReconstructionError),
    (load_padded_encoder_contract, PaddedReconstructionError),
])
def test_reconstruction_rejects_missing_or_corrupt_encoder_identity(loader, error) -> None:
    with pytest.raises(error):
        loader({}, path=__file__)
    summary = _summary()
    summary["spike_encoder_spec_sha256"] = "0" * 64
    with pytest.raises(error, match="does not match"):
        loader(summary, path=__file__)


def test_default_widths_preserve_kernel_shape() -> None:
    kernels = segmented_reconstruction_kernels(
        wavelet_widths_samples=(400, 200, 100, 50, 25),
        frequencies_hz=(0.5, 1.0, 2.0, 4.0, 8.0),
    )
    assert [kernel.shape for kernel in kernels] == [(800,)] * 5
    assert all(np.isfinite(kernel).all() for kernel in kernels)
