from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_4_4_1_stage2_memory_sweep as exp441


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_4_4_1_stage2_memory_sweep.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_4_4_1_cpu_array.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_4_4_1_cpu.bash"


def test_run_matrix_is_exactly_50_paired_runs() -> None:
    specs = exp441.run_specs()
    assert len(specs) == 50
    assert exp441.TAU_MEM_MS == (125, 250, 500, 1000, 2000)
    assert exp441.MODEL_KINDS == ("ff", "rsnn")
    assert tuple(exp441.SEEDS) == (11, 23, 37, 53, 71)
    assert len({spec.key for spec in specs}) == 50
    for tau in exp441.TAU_MEM_MS:
        for seed in exp441.SEEDS:
            matched = [spec for spec in specs if spec.tau_mem_ms == tau and spec.seed == seed]
            assert {spec.model_kind for spec in matched} == {"ff", "rsnn"}


def test_stage2_tau_changes_only_stage2_beta() -> None:
    ff = exp441.Stage2SweepDecoder("ff", 125, n_classes=12, fs=64.0)
    rsnn = exp441.Stage2SweepDecoder("rsnn", 2000, n_classes=12, fs=64.0)

    assert np.isclose(ff.state_lif.beta, exp441.stage2_beta(125, ff.dt_ms))
    assert np.isclose(rsnn.state_lif.beta, exp441.stage2_beta(2000, rsnn.dt_ms))
    assert ff.state_lif.beta < rsnn.state_lif.beta
    assert torch.allclose(ff.local_lif.beta, rsnn.local_lif.beta)
    assert np.isclose(ff.output_lif.beta, rsnn.output_lif.beta)
    assert ff.hidden_cap == rsnn.hidden_cap == exp441.HIDDEN_CAP
    assert ff.output_cap == rsnn.output_cap == exp441.OUTPUT_CAP


def test_ff_and_rsnn_forward_share_shape_but_differ_in_recurrence_contract() -> None:
    X = torch.zeros(2, 7, 30)
    ff = exp441.Stage2SweepDecoder("ff", 250, n_classes=12, fs=64.0)
    rsnn = exp441.Stage2SweepDecoder("rsnn", 250, n_classes=12, fs=64.0)
    ff_traj = ff.forward_trajectory(X)
    rsnn_traj = rsnn.forward_trajectory(X)
    assert ff_traj["state_membranes"].shape == (2, 7, 128)
    assert rsnn_traj["state_membranes"].shape == (2, 7, 128)
    assert ff_traj["output_spikes"].shape == (2, 7, 12)
    assert rsnn_traj["output_spikes"].shape == (2, 7, 12)
    try:
        ff.forward_trajectory(X, recurrence_enabled=True)
    except ValueError:
        pass
    else:
        raise AssertionError("FF control must not allow recurrence to be enabled")


def test_array_and_submit_scripts_follow_multi_cpu_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-49%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in array_text
    assert "module load conda/latest" in array_text
    assert "conda activate writingring-gpu" in array_text
    assert "run-one" in array_text
    assert "afterok:${ARRAY_JOB}" in submit_text


def test_notebook_is_analysis_only_and_compiles() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    assert "runs.csv" in joined
    assert "summary.csv" in joined
    assert "paired_effects_summary.csv" in joined
    assert "Uend" in joined
    assert "RSNN - FF" in joined or "RSNN - FF" in joined.replace("\\u2013", "-")
    forbidden = ["optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one"]
    for token in forbidden:
        assert token not in joined
