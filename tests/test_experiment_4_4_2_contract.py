from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_4_0_5_temporal_resolution_event_capacity as exp405
from scripts import experiment_4_4_1_stage2_memory_sweep as exp441
from scripts import experiment_4_4_2_recurrence_topology as exp442


def test_exp442_run_matrix_and_parameter_counts() -> None:
    specs = exp442.run_specs()
    assert exp442.TAU_MEM_MS == (250, 500)
    assert len(specs) == 10
    assert len({spec.key for spec in specs}) == 10
    assert {spec.topology for spec in specs} == {"diagonal"}
    assert exp442.recurrent_param_count("ff") == 0
    assert exp442.recurrent_param_count("diagonal") == exp405.STATE_WIDTH
    assert exp442.recurrent_param_count("dense") == exp405.STATE_WIDTH**2


def test_exp442_diagonal_initialization_is_paired_with_dense() -> None:
    seed = 11
    tau = 250
    spec = exp442.RunSpec("diagonal", tau, seed)
    paired = exp442.paired_seed(spec, "model_init")

    exp405.exp40.base.seed_all(paired)
    dense = exp441.Stage2SweepDecoder(
        model_kind="rsnn", tau_mem_ms=tau, n_classes=12, fs=64.0
    )
    dense_diag = torch.diagonal(dense.recurrent.weight.detach()).clone()

    exp405.exp40.base.seed_all(paired)
    diagonal = exp442.TopologyDecoder(tau_mem_ms=tau, n_classes=12, fs=64.0)

    assert torch.equal(diagonal.input_local.weight, dense.input_local.weight)
    assert torch.equal(diagonal.local_state.weight, dense.local_state.weight)
    assert torch.equal(diagonal.state_output.weight, dense.state_output.weight)
    assert torch.equal(diagonal.recurrent.weight.detach(), dense_diag)
    assert diagonal.recurrent.weight.numel() == exp405.STATE_WIDTH


def test_exp442_diagonal_recurrence_has_no_cross_neuron_mixing() -> None:
    weight = torch.arange(exp405.STATE_WIDTH, dtype=torch.float32)
    recurrent = exp442.DiagonalRecurrent(weight)
    spikes = torch.zeros(2, exp405.STATE_WIDTH)
    spikes[0, 3] = 2.0
    spikes[1, 9] = 1.0
    output = recurrent(spikes)
    assert output[0, 3].item() == 6.0
    assert output[1, 9].item() == 9.0
    assert torch.count_nonzero(output[0]).item() == 1
    assert torch.count_nonzero(output[1]).item() == 1


def test_exp442_slurm_and_notebook_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (repo_root / "scripts/bash_script/SNN_Bash/run_exp_4_4_2_cpu_array.bash").read_text()
    submit = (repo_root / "scripts/bash_script/SNN_Bash/submit_exp_4_4_2_cpu.bash").read_text()
    assert "#SBATCH --array=0-9%10" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in runner
    assert "afterok:${ARRAY_JOB}" in submit

    notebook_path = repo_root / "notebooks/experiment_4_4_2_recurrence_topology.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{notebook_path}::{index}", "exec")
    for filename in ("runs.csv", "summary.csv", "paired_effects.csv", "paired_effects_summary.csv", "manifest.json"):
        assert filename in code
    for forbidden in ("run_one(", "train_one(", "sbatch ", "subprocess."):
        assert forbidden not in code
    assert "experiment_4_4_2_recurrence_topology" in code
