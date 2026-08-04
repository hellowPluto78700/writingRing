from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import segment_ring_imu


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    action_dir = root / "writer_a" / "letters"
    action_dir.mkdir(parents=True)
    timestamps = 1_000_000 + np.arange(700, dtype=np.float64) * 1_000
    imu = np.column_stack([timestamps + channel for channel in range(6)])
    np.column_stack((imu, timestamps)).astype(np.float64).tofile(
        action_dir / "0_ring_0.bin"
    )
    (action_dir / "0_ring_1.bin").write_bytes(b"must not be opened")
    (action_dir / "0_timestamp.txt").write_text(
        "1000000 a\n1350000 B\n", encoding="utf-8"
    )
    return root


def test_cli_exports_primary_ring_variable_segments_and_honors_overwrite(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    root = _data_root(tmp_path)
    output = tmp_path / "outputs"
    opened: list[Path] = []
    original = np.fromfile

    def tracked(path: object, *args: object, **kwargs: object) -> np.ndarray:
        if isinstance(path, (str, Path)):
            opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr("writingring.ring_loader.np.fromfile", tracked)
    args = [
        "--data-root", str(root), "--user", "writer_a", "--action", "letters",
        "--output-root", str(output),
    ]
    assert segment_ring_imu.main(args) == 0
    captured = capsys.readouterr()
    assert "Exported 2 variable-length segments with 700 total IMU samples." in captured.out
    assert opened == [root / "writer_a" / "letters" / "0_ring_0.bin"]
    base = output / "writer_a" / "action_letters"
    raw = base / "writer_a_action_letters_rawIMU.npy"
    np.testing.assert_equal(np.load(raw, allow_pickle=False).shape, (700, 6))
    np.testing.assert_array_equal(
        np.load(base / "writer_a_action_letters_segment_offsets.npy", allow_pickle=False),
        [0, 350, 700],
    )

    assert segment_ring_imu.main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert segment_ring_imu.main([*args, "--overwrite"]) == 0
