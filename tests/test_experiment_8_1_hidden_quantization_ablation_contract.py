from __future__ import annotations

from pathlib import Path

import torch

from scripts import experiment_8_1_hidden_quantization_ablation as exp81


def test_run_matrix_and_locked_a2_contract() -> None:
    specs = exp81.run_specs()
    assert len(specs) == 15
    assert exp81.SEEDS == (11, 23, 37)
    assert exp81.SHIFTS == (2, 3, 4)
    assert exp81.L2_WIDTH == 128
    assert exp81.WEIGHTED_CAP == 31
    assert exp81.METHOD_ORDER == (
        "binary128",
        "weighted31_128",
        "mt3_128",
        "binary384",
        "mt3_384",
    )


def test_threshold_bank_layouts_preserve_requested_widths() -> None:
    mt128 = exp81.RunSpec("mt3_128", 11)
    mt384 = exp81.RunSpec("mt3_384", 11)
    assert exp81.l1_threshold_bank_widths(mt128) == (43, 43, 42)
    assert exp81.l1_threshold_bank_widths(mt384) == (128, 128, 128)
    for spec in (mt128, mt384):
        groups = exp81.l1_groups(spec)
        assert sum(int(group["count"]) for group in groups) == spec.l1_width
        assert {float(group["threshold_multiplier"]) for group in groups} == {
            0.5,
            1.0,
            1.5,
        }
        for multiplier in exp81.THRESHOLD_MULTIPLIERS:
            assert {
                int(group["shift"])
                for group in groups
                if float(group["threshold_multiplier"]) == multiplier
            } == {2, 3, 4}


def test_fixed_threshold_lif_is_binary_and_threshold_specific() -> None:
    thresholds = torch.tensor([0.5, 1.0, 1.5])
    lif = exp81.VectorThresholdBinaryLIF(
        beta=0.0, thresholds=thresholds, surrogate_slope=25.0
    )
    current = torch.tensor([[0.75, 1.25, 1.25]])
    membrane = torch.zeros_like(current)
    spikes, post, pre = lif(current, membrane)
    assert torch.equal(spikes, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(pre, current)
    assert torch.allclose(post, torch.tensor([[0.25, 0.25, 1.25]]))


def test_slurm_and_notebook_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    run_bash = (
        root / "scripts/bash_script/SNN_Bash/run_exp_8_1_cpu_array.bash"
    ).read_text()
    final_bash = (
        root / "scripts/bash_script/SNN_Bash/finalize_exp_8_1_cpu.bash"
    ).read_text()
    submit_bash = (
        root / "scripts/bash_script/SNN_Bash/submit_exp_8_1_cpu.bash"
    ).read_text()
    notebook = (
        root / "notebooks/experiment_8_1_hidden_quantization_ablation.ipynb"
    ).read_text()
    assert "#SBATCH --array=0-14%15" in run_bash
    assert "#SBATCH --cpus-per-task=1" in run_bash
    assert "module load conda/latest" in run_bash
    assert "OMP_NUM_THREADS=1" in run_bash
    assert " finalize" in final_bash
    assert 'dependency="afterok:${run_job}"' in submit_bash
    for forbidden in ("import torch", "sbatch", "run_one(", "subprocess"):
        assert forbidden not in notebook
