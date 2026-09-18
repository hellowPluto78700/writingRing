from __future__ import annotations

from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "bash_script" / "preprocessing_pipeline" / "build_writing_motion_variants.bash"
PYTHON = PROJECT_ROOT / "scripts" / "build_writing_motion_variants.py"


def test_writing_motion_launcher_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "MODE=local" in text
    assert "MODE=submit" in text
    assert "--array=" in text
    assert '--dependency="afterok:${worker_job}"' in text
    assert "OMP_NUM_THREADS=1" in text
    assert "WRITE_MASK_VERIFICATION" in text
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_writing_motion_python_entrypoint_exists() -> None:
    text = PYTHON.read_text(encoding="utf-8")
    assert 'sub.add_parser("list-users")' in text
    assert 'sub.add_parser("build-user")' in text
    assert 'sub.add_parser("finalize")' in text
    assert '"postencode_mask"' in text
    assert '"masked_accel_reencode"' in text
    assert '"writing_motion_verification"' in text
    assert "masked_accel_verification.png" not in text
