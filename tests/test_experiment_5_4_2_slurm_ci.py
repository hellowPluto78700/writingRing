from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "scripts/bash_script/SNN_Bash"

SCREEN_SUBMIT = SLURM_DIR / "submit_exp_5_4_2_screen_cpu.bash"
REFINE_SUBMIT = SLURM_DIR / "submit_exp_5_4_2_refine_cpu.bash"

SCREEN_TARGETS = (
    "scripts/bash_script/SNN_Bash/prepare_exp_5_4_2_source_cpu_array.bash",
    "scripts/bash_script/SNN_Bash/run_exp_5_4_2_screen_cpu_array.bash",
    "scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_screen_cpu.bash",
)
REFINE_TARGETS = (
    "scripts/bash_script/SNN_Bash/run_exp_5_4_2_refine_cpu_array.bash",
    "scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_refine_cpu.bash",
    "scripts/bash_script/SNN_Bash/run_exp_5_4_2_final_cpu_array.bash",
    "scripts/bash_script/SNN_Bash/finalize_exp_5_4_2_cpu.bash",
)
ALL_TARGETS = SCREEN_TARGETS + REFINE_TARGETS
ALL_SCRIPTS = (SCREEN_SUBMIT, REFINE_SUBMIT) + tuple(REPO_ROOT / p for p in ALL_TARGETS)


def test_sbatch_targets_are_repo_relative_and_exist() -> None:
    screen = SCREEN_SUBMIT.read_text(encoding="utf-8")
    refine = REFINE_SUBMIT.read_text(encoding="utf-8")

    assert "command -v sbatch" in screen
    assert "command -v sbatch" in refine
    assert "--wrap" not in screen
    assert "--wrap" not in refine
    assert "/usr/bin/sbatch" not in screen
    assert "/usr/bin/sbatch" not in refine

    for target in SCREEN_TARGETS:
        assert target in screen, f"screen launcher does not submit expected runner: {target}"
        assert (REPO_ROOT / target).is_file(), f"screen launcher target does not exist: {target}"

    for target in REFINE_TARGETS:
        assert target in refine, f"refine launcher does not submit expected runner: {target}"
        assert (REPO_ROOT / target).is_file(), f"refine launcher target does not exist: {target}"


def test_exp542_slurm_scripts_have_valid_bash_syntax() -> None:
    for script in ALL_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"bash -n failed for {script}: {result.stderr}"


def test_exp542_slurm_runners_match_unity_environment_contract() -> None:
    for relative in ALL_TARGETS:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "#SBATCH" in text
        assert "--cpus-per-task=1" in text
        assert "module load conda/latest" in text
        assert 'eval "$(conda shell.bash hook)"' in text
        assert "conda activate writingring-gpu" in text
        assert "conda activate writingring-viz" in text
        assert "OMP_NUM_THREADS=1" in text
        assert "MKL_NUM_THREADS=1" in text
        assert "OPENBLAS_NUM_THREADS=1" in text
        assert "NUMEXPR_NUM_THREADS=1" in text
