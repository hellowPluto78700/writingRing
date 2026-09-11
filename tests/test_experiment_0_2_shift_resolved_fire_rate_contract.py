from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import experiment_0_2_endpoint_tail_regularization as exp02
from scripts import experiment_0_2_shift_resolved_fire_rate as analysis


def test_eval_specs_cover_all_90_checkpoints_once() -> None:
    specs = analysis.eval_specs()
    assert len(specs) == 90
    assert len({spec.key for spec in specs}) == 90
    assert {spec.profile for spec in specs} == {"none", "a1", "a1_p2", "tail", "a1_tail"}


def test_shift_slices_span_width_and_match_configured_groups() -> None:
    for architecture, layers in exp02.DIRECT_ARCHITECTURES.items():
        shifts = tuple(layers[-1])
        slices = analysis._shift_slices(shifts, exp02.HIDDEN_WIDTH)
        assert [shift for shift, _, _ in slices] == list(shifts)
        assert slices[0][1] == 0
        assert slices[-1][2] == exp02.HIDDEN_WIDTH
        assert all(a_stop == b_start for (_, _, a_stop), (_, b_start, _) in zip(slices, slices[1:]))


def test_spike_stats_uses_neuron_seconds() -> None:
    spikes = torch.zeros(2, 4, 4)
    spikes[:, :2, 2:4] = 1.0
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    rate, spikes_per_neuron, timestep_count, neuron_count = analysis._spike_stats(
        spikes, mask, 2, 4, fs=2.0
    )
    # 8 spikes / ((4 valid sample-steps * 2 neurons) / 2 Hz) = 2 Hz.
    assert rate == 2.0
    assert spikes_per_neuron == 4.0
    assert timestep_count == 4
    assert neuron_count == 2


def test_long_tau_summary_only_uses_shift_ge_5() -> None:
    rows = []
    for shift, value in [(4, 100.0), (5, 5.0), (6, 7.0)]:
        rows.append(
            {
                "architecture": "x",
                "objective": "whole_count_ce",
                "profile": "tail",
                "seed": 11,
                "shift": shift,
                "valid_firing_rate_hz": value,
                "tail_stage1_firing_rate_hz": value,
                "tail_stage2_firing_rate_hz": value,
                "tail_stage3_firing_rate_hz": value,
                "tail_firing_rate_hz": value,
                "tail_to_valid_rate_ratio": 1.0,
            }
        )
    out = analysis._long_tau_summary(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["long_shifts"] == "5,6"
    assert np.isclose(out.iloc[0]["tail_firing_rate_hz"], 6.0)


def test_notebook_is_analysis_only() -> None:
    text = Path("notebooks/experiment_0_2_shift_resolved_fire_rate.ipynb").read_text(encoding="utf-8")
    forbidden = ["torch.load", "DataLoader", "multiprocessing", "subprocess", "optimizer", "backward("]
    assert not any(token in text for token in forbidden)
    assert "shift_fire_rate_summary.csv" in text


def test_slurm_array_is_one_core_and_90_tasks() -> None:
    text = Path("scripts/bash_script/SNN_Bash/run_exp_0_2_shift_fire_rate_cpu_array.bash").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-89%50" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
    assert "OPENBLAS_NUM_THREADS=1" in text


def test_finalizer_is_dependency_only_aggregation() -> None:
    submit = Path("scripts/bash_script/SNN_Bash/submit_exp_0_2_shift_fire_rate_cpu.bash").read_text(encoding="utf-8")
    finalize = Path("scripts/bash_script/SNN_Bash/finalize_exp_0_2_shift_fire_rate_cpu.bash").read_text(encoding="utf-8")
    assert "afterok:${ARRAY_JOB_ID}" in submit
    assert " finalize" in finalize
    assert "run-one" not in finalize
