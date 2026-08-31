from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "experiment_1_3_10_stacked_bin_snn_ablation.py"
ARRAY = ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_1_3_10_cpu_array.bash"
FINALIZER = ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_1_3_10_cpu.bash"
SUBMITTER = ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_1_3_10_cpu.bash"
NOTEBOOK = ROOT / "notebooks" / "experiment_1_3_10_stacked_bin_snn_ablation.ipynb"


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_factorial_protocol_constants() -> None:
    values = _literal_assignments(DRIVER)
    assert values["SPLIT_SEEDS"] == (11, 23, 101)
    assert values["OBJECTIVES"] == ("whole_count_ce", "timestep_ce")
    assert values["TRAIN_REGIMES"] == ("weight_only", "trainable_dynamics")
    assert values["HIDDEN_SHIFTS"] == (2, 3, 4)
    assert values["TAU_MEM_MS"] == 22.0
    assert values["THRESHOLD"] == 0.5
    assert values["BIN_MS"] == 250.0


def test_architecture_contract_and_readout() -> None:
    source = DRIVER.read_text(encoding="utf-8")
    assert '"1h128": (128,)' in source
    assert '"1h256": (256,)' in source
    assert '"2h128": (128, 128)' in source
    assert "return out_spikes.sum(dim=1)" in source
    assert "F.cross_entropy(out_spikes.sum(dim=1), labels)" in source
    assert "out_spikes.reshape(batch * timesteps, classes)" in source
    assert "trainable_dynamics=(spec.train_regime == \"trainable_dynamics\")" in source
    assert 'model_seed = derive_seed(spec.split_seed, "model_init", spec.architecture)' in source


def test_multi_cpu_launcher_contract() -> None:
    array_text = ARRAY.read_text(encoding="utf-8")
    finalizer_text = FINALIZER.read_text(encoding="utf-8")
    submitter_text = SUBMITTER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-35%36" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    assert 'export OMP_NUM_THREADS="$THREADS"' in array_text
    assert "--finalize-if-ready" not in array_text
    assert "--finalize-only" in finalizer_text
    assert "--require-complete" in finalizer_text
    assert 'afterok:${ARRAY_JOB_ID}' in submitter_text


def test_notebook_is_analysis_only() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "snntorch" not in code
    assert "torch.optim" not in code
    assert "loss.backward" not in code
    assert "train_one_run" not in code
    assert "run_*.json" in code
    assert "plt.subplots" in code
