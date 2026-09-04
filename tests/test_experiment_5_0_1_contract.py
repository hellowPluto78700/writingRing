from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from scripts import experiment_3_0_3_l3_bottleneck_ablation as exp303
from scripts import experiment_5_0_1_exp3_analog_head_control as exp501


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_0_1_cpu_array.bash"
FINALIZE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_5_0_1_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_0_1_cpu.bash"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_0_1_exp3_analog_head_control.ipynb"
README = REPO_ROOT / "scripts" / "experiment_5_0_1" / "README.md"


def test_run_matrix_is_exact_exp3_seed_control() -> None:
    specs = exp501.run_specs()
    assert exp501.EXPECTED_RUNS == 3
    assert exp501.SEEDS == (11, 23, 101)
    assert [spec.seed for spec in specs] == [11, 23, 101]
    assert len({spec.key for spec in specs}) == 3
    assert exp501.OBJECTIVE == "timestep_ce"


def test_model_is_exact_exp3_two_layer_analog_head_contract() -> None:
    model = exp501.exp304.L2WidthNet(
        128,
        "timestep_ce",
        12,
        256,
        64.0,
        16,
    )
    assert model.layer_order == ("L1", "L2")
    assert model.layer_shifts["L1"] == (2, 3, 4)
    assert model.layer_shifts["L2"] == (2, 3, 4)
    assert model.layer_widths == {"L1": 128, "L2": 128}
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 128
    assert model.head.out_features == 12
    assert model.head.bias is not None
    assert not hasattr(model, "output_lif")
    assert not hasattr(model, "output_linear")
    assert exp501.base.TAU_MEM_MS == 22.54
    assert exp501.base.THRESHOLD == 0.5


def test_initialization_matches_historical_exp3_0_3_no_l3_model() -> None:
    seed = 11
    shared_seed = exp501.base.dseed(seed, "shared_backbone_init")
    head_seed = exp501.base.dseed(seed, "timestep_ce", "head_init")

    exp501.base.seed_all(shared_seed)
    control = exp501.exp304.L2WidthNet(128, "timestep_ce", 12, 256, 64.0, 16)
    exp501.base.seed_all(head_seed)
    control.head.reset_parameters()

    exp501.base.seed_all(shared_seed)
    historical = exp303.L3AblationNet("B", "timestep_ce", 12, 256, 64.0, 16)
    exp501.base.seed_all(head_seed)
    historical.head.reset_parameters()

    assert set(control.state_dict()) == set(historical.state_dict())
    for name in control.state_dict():
        assert torch.equal(control.state_dict()[name], historical.state_dict()[name]), name


def test_timestep_loss_is_applied_directly_to_analog_linear_logits() -> None:
    model = exp501.exp304.L2WidthNet(
        128,
        "timestep_ce",
        12,
        8,
        64.0,
        16,
    )
    spikes = torch.zeros(2, 8, 128)
    lengths = torch.tensor([8, 5])
    y = torch.tensor([1, 2])
    loss, segment_logits = model.loss_logits(spikes, lengths, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert segment_logits.shape == (2, 12)


def test_probe_protocol_reuses_all_exp3_0_5_probes() -> None:
    assert exp501.PROBE_TYPES == (
        "full_count",
        "fixed250_ordered",
        "fixed250_shuffled",
        "fixed250_pca128",
        "relative10_ordered",
        "relative10_shuffled",
        "relative10_pca128",
        "duration_only",
    )
    assert exp501.PRIMARY_PROBES == (
        "full_count",
        "fixed250_ordered",
        "relative10_ordered",
    )


def test_multi_cpu_afterok_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    finalizer_text = FINALIZE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%3" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in array_text
    assert "module load conda/latest" in array_text
    assert "conda activate writingring-gpu" in array_text
    assert "run-one" in array_text
    assert "finalize" in finalizer_text
    assert "afterok:${ARRAY_JOB}" in submit_text


def test_notebook_is_analysis_only_and_performs_requested_aggregation() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "runs.csv",
        "comparison_runs.csv",
        "groupby",
        "mean_delta",
        "Exp3",
        "binary",
        "multi_ho",
        "L2 FullCount + Linear",
        "L2 Fixed250 + Linear",
        "L2 Relative10 + Linear",
    ):
        assert token in joined
    for forbidden in ("optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one"):
        assert forbidden not in joined


def test_readme_states_the_causal_control_and_execution_contract() -> None:
    text = README.read_text(encoding="utf-8")
    for token in (
        "no output LIF",
        "Linear(128,12,bias=True)",
        "0-2%3",
        "Exp3.0.5",
        "Exp5.0",
        "notebook",
    ):
        assert token in text
