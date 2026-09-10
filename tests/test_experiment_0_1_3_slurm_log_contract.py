from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASH_ROOT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
ARRAY_SCRIPT = BASH_ROOT / "run_exp_0_1_3_doc_faithful_regularization_cpu_array.bash"
FINALIZE_SCRIPT = BASH_ROOT / "finalize_exp_0_1_3_doc_faithful_regularization_cpu.bash"
SUBMIT_SCRIPT = BASH_ROOT / "submit_exp_0_1_3_doc_faithful_regularization_cpu.bash"


def test_exp013_slurm_paths_match_exp012_submission_directory_pattern() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalize_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH --output=exp0_1_3_docreg_%A_%a.out" in array_text
    assert "#SBATCH --error=exp0_1_3_docreg_%A_%a.err" in array_text
    assert "#SBATCH --output=exp0_1_3_finalize_%j.out" in finalize_text
    assert "#SBATCH --error=exp0_1_3_finalize_%j.err" in finalize_text

    for text in (array_text, finalize_text, submit_text):
        assert "mkdir -p outputs" not in text
        assert "outputs/" not in text

    assert 'REPO_ROOT="${REPO_ROOT:-$PWD}"' in array_text
    assert 'REPO_ROOT="${REPO_ROOT:-$PWD}"' in finalize_text
    assert 'REPO_ROOT="${REPO_ROOT:-$(pwd)}"' in submit_text
    assert "export REPO_ROOT" in submit_text
    assert "sbatch --parsable --export=ALL" in submit_text
    assert '--export=ALL --dependency="afterok:${ARRAY_JOB}"' in submit_text
