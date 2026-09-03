from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_5_0_local_evidence_objectives as exp50


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_5_0_local_evidence_objectives.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_5_0_cpu_array.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_5_0_cpu.bash"


def test_run_matrix_is_exactly_75_paired_runs() -> None:
    specs = exp50.run_specs()
    assert len(specs) == 75
    assert exp50.OBJECTIVES == (
        "whole_count_ce",
        "timestep_ce",
        "fixed250_bin_ce",
        "fixed500_bin_ce",
        "relative10_bin_ce",
    )
    assert exp50.VARIANTS == (
        ("binary", 1, 1),
        ("multi_h", 31, 1),
        ("multi_ho", 31, 31),
    )
    assert exp50.SEEDS == (11, 23, 37, 53, 71)
    assert len({spec.key for spec in specs}) == 75
    for seed in exp50.SEEDS:
        seed_specs = [spec for spec in specs if spec.seed == seed]
        assert len(seed_specs) == 15
        assert {spec.objective for spec in seed_specs} == set(exp50.OBJECTIVES)
        assert {spec.variant for spec in seed_specs} == {"binary", "multi_h", "multi_ho"}


def test_backbone_contract_is_exp3_style_local_multitau() -> None:
    assert exp50.L1_WIDTH == exp50.L2_WIDTH == 128
    assert exp50.L1_SHIFTS == exp50.L2_SHIFTS == (2, 3, 4)
    assert exp50.TAU_MEM_MS == 22.54
    assert exp50.THRESHOLD == 0.5
    model = exp50.LocalEvidenceSNN(12, fs=64.0, hidden_cap=1, output_cap=1)
    assert tuple(model.alpha1.shape) == (128,)
    assert tuple(model.alpha2.shape) == (128,)
    assert torch.unique(model.alpha1).numel() == 3
    assert torch.unique(model.alpha2).numel() == 3
    assert model.l1_lif.max_spikes_per_dt == 1
    assert model.l2_lif.max_spikes_per_dt == 1
    assert model.output_lif.max_spikes_per_dt == 1


def test_event_cap_variants_change_only_caps_for_same_seeded_model() -> None:
    exp50.exp3.seed_all(123)
    binary = exp50.LocalEvidenceSNN(12, fs=64.0, hidden_cap=1, output_cap=1)
    exp50.exp3.seed_all(123)
    multi_ho = exp50.LocalEvidenceSNN(12, fs=64.0, hidden_cap=31, output_cap=31)
    for name in ("f1.weight", "f2.weight", "output_linear.weight"):
        assert torch.allclose(binary.state_dict()[name], multi_ho.state_dict()[name])
    assert torch.allclose(binary.alpha1, multi_ho.alpha1)
    assert torch.allclose(binary.alpha2, multi_ho.alpha2)
    assert binary.l1_lif.beta == multi_ho.l1_lif.beta
    assert binary.l2_lif.beta == multi_ho.l2_lif.beta
    assert binary.output_lif.beta == multi_ho.output_lif.beta


def test_loss_partitions_have_requested_physical_widths_and_valid_weighting() -> None:
    assert exp50.fixed_steps(250.0, 64.0) == 16
    assert exp50.fixed_steps(500.0, 64.0) == 32
    spikes = torch.zeros(2, 40, 12)
    spikes[:, :, 3] = 1.0
    lengths = torch.tensor([40, 20])
    y = torch.tensor([3, 3])
    for objective in exp50.OBJECTIVES:
        loss = exp50.objective_loss(spikes, lengths, y, objective, output_cap=1, fs=64.0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
    means250, sizes250 = exp50.fixed_bin_means_and_sizes(spikes, lengths, 16, 1)
    assert means250.shape == (2, 3, 12)
    assert sizes250[0].tolist() == [16.0, 16.0, 8.0]
    assert sizes250[1].tolist() == [16.0, 4.0, 0.0]
    means500, sizes500 = exp50.fixed_bin_means_and_sizes(spikes, lengths, 32, 1)
    assert means500.shape == (2, 2, 12)
    assert sizes500[0].tolist() == [32.0, 8.0]
    assert sizes500[1].tolist() == [20.0, 0.0]


def test_relative10_uses_valid_length_only_for_partitioning() -> None:
    spikes = torch.zeros(2, 40, 12)
    lengths = torch.tensor([40, 20])
    means, sizes = exp50.relative_bin_means_and_sizes(
        spikes, lengths, exp50.RELATIVE_BINS, cap=1
    )
    assert means.shape == (2, 10, 12)
    assert torch.allclose(sizes[0], torch.full((10,), 4.0))
    assert torch.allclose(sizes[1], torch.full((10,), 2.0))


def test_readout_matrix_has_expected_temporal_access_contract() -> None:
    assert exp50.READOUTS == (
        "output_whole_count",
        "l1_whole_count_linear",
        "l2_whole_count_linear",
        "l2_fixed250_ordered_linear",
        "l2_relative10_ordered_linear",
        "l2_uend_linear",
    )


def test_array_submit_and_notebook_follow_multi_cpu_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-74%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in array_text
    assert "module load conda/latest" in array_text
    assert "conda activate writingring-gpu" in array_text
    assert "run-one" in array_text
    assert "afterok:${ARRAY_JOB}" in submit_text

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
        "summary.csv",
        "paired_loss_effects_summary.csv",
        "paired_variant_effects_summary.csv",
        "Output WholeCount",
        "L2 Count + Linear",
        "L2 Fixed250 + Linear",
        "L2 Relative10 + Linear",
        "L2 Uend + Linear",
    ):
        assert token in joined
    for forbidden in ("optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one"):
        assert forbidden not in joined
