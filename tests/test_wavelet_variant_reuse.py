from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / "scripts" / "wavelet_variant_reuse.py"


def _helper_module():
    spec = importlib.util.spec_from_file_location("wavelet_variant_reuse_test", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_polarity_split_layout_is_accepted_for_variant_reuse(tmp_path: Path) -> None:
    helper = _helper_module()
    metadata = {
        "spike_imu": {"channel_count": 36, "event_channel_count": 30},
        "spike_encoder": {"frequencies_hz": [1, 2, 4, 8, 16]},
        "spike_encoder_spec_sha256": "a" * 64,
        "settings": {"post_encode_transform": "PolaritySplitAbs"},
    }
    values_path = tmp_path / "spikeIMU.npy"
    np.save(values_path, np.zeros((4, 36), dtype=np.float32), allow_pickle=False)
    record = helper.Record("user", "0", "3", tmp_path, values_path, tmp_path / "metadata.json", metadata)
    assert helper.spike_layout(metadata) == (36, 30)
    assert helper.check_uniform_source_records({record.key: record}) == ("a" * 64, [1.0, 2.0, 4.0, 8.0, 16.0])
