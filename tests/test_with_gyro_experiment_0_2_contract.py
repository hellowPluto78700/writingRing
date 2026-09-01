from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import with_gyro_experiment_0_2_nonlinear_temporal_decoder_probe as exp02


EXPECTED_CONDA_PREFIX_FRAGMENT = "/work/pi_jgummeso_umass_edu/${USER}/.conda/envs/writingring-gpu"


def test_protocol_dimensions_and_run_mapping_are_stable() -> None:
    assert exp02.PROTOCOL_VERSION == "structured_residual_gru_v1"
    assert exp02.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp02.CHANNEL_SETS == {
        "accel30": (0, 30),
        "angular30": (30, 60),
        "combined60": (0, 60),
    }
    assert exp02.REPRESENTATIONS == ("fixed250", "relative10")
    assert exp02.RESIDUAL_DECODERS == ("local", "transition", "local_transition")
    assert exp02.NEURAL_DECODERS == ("local", "transition", "local_transition", "gru")
    assert exp02.RESIDUAL_RANK == 16
    assert exp02.GRU_HIDDEN_SIZE == 32
    assert exp02.EXPECTED_BASELINES == 30
    assert exp02.EXPECTED_NEURAL_RUNS == 120
    assert len(exp02.baseline_specs()) == 30
    assert len({spec.key for spec in exp02.baseline_specs()}) == 30
    assert len(exp02.run_specs()) == 120
    assert len({spec.key for spec in exp02.run_specs()}) == 120


def test_model_seed_is_architecture_specific_not_split_specific() -> None:
    first = exp02.RunSpec("combined60", "fixed250", "gru", 11)
    second = exp02.RunSpec("combined60", "fixed250", "gru", 71)
    other = exp02.RunSpec("combined60", "relative10", "gru", 11)
    assert exp02._model_seed(first) == exp02._model_seed(second)
    assert exp02._model_seed(first) != exp02._model_seed(other)


def test_residual_decoder_supports_30_and_60_channel_inputs_and_starts_at_linear() -> None:
    for input_dim in (30, 60):
        model = exp02.ResidualDecoder(
            "local_transition",
            input_dim=input_dim,
            n_bins=10,
            n_classes=12,
            rank=16,
        )
        x = torch.randn(4, 10, input_dim)
        mask = torch.ones(4, 10, dtype=torch.bool)
        base = torch.randn(4, 12)
        with torch.no_grad():
            logits = model(x, mask, base)
        torch.testing.assert_close(logits, base)


def test_transition_mask_blocks_invalid_pairs() -> None:
    model = exp02.ResidualDecoder(
        "transition",
        input_dim=3,
        n_bins=4,
        n_classes=2,
        rank=2,
    )
    with torch.no_grad():
        model.transition_p.weight.fill_(1.0)
        model.transition_q.weight.fill_(1.0)
        model.transition_head.weight.fill_(1.0)
    x = torch.ones(1, 4, 3)
    base = torch.zeros(1, 2)
    full_mask = torch.tensor([[True, True, True, True]])
    short_mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        full = model(x, full_mask, base)
        short = model(x, short_mask, base)
    assert torch.all(full > short)
    torch.testing.assert_close(short[0, 0], torch.tensor(18.0))


def test_gru_uses_last_valid_bin_and_ignores_right_padding() -> None:
    torch.manual_seed(7)
    model = exp02.GRUDecoder(input_dim=3, n_classes=2, hidden_size=4)
    valid = torch.randn(1, 2, 3)
    padded_a = torch.cat([valid, torch.zeros(1, 2, 3)], dim=1)
    padded_b = torch.cat([valid, torch.full((1, 2, 3), 99.0)], dim=1)
    mask = torch.tensor([[True, True, False, False]])
    base = torch.zeros(1, 2)
    with torch.no_grad():
        out_a = model(padded_a, mask, base)
        out_b = model(padded_b, mask, base)
    torch.testing.assert_close(out_a, out_b)


def test_nonlinear_synergy_is_difference_of_paired_gains() -> None:
    rows: list[dict[str, object]] = []
    for representation in exp02.REPRESENTATIONS:
        for split_seed in exp02.SPLIT_SEEDS:
            for channel_set, offset in (
                ("accel30", 0.01),
                ("angular30", 0.02),
                ("combined60", 0.05),
            ):
                rows.append(
                    {
                        "representation": representation,
                        "channel_set": channel_set,
                        "split_seed": split_seed,
                        "local_minus_linear": offset,
                        "transition_minus_linear": offset + 0.01,
                        "local_transition_minus_linear": offset + 0.02,
                    }
                )
    synergy = exp02._nonlinear_synergy(pd.DataFrame(rows))
    combined_minus_angular = synergy["combined_gain_minus_angular_gain"].to_numpy()
    np.testing.assert_allclose(combined_minus_angular, 0.03)


def _read_bash(name: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root / "scripts" / "bash_script" / "withGyro" / name
    ).read_text(encoding="utf-8")


def test_neural_array_is_one_core_120_tasks_and_capped_at_50() -> None:
    script = _read_bash("run_exp_0_2_cpu_array.bash")
    assert "#SBATCH --array=0-119%50" in script
    assert "TASK_ID >= 120" in script
    assert "#SBATCH --cpus-per-task=1" in script
    assert "OMP_NUM_THREADS=1" in script
    assert "MKL_NUM_THREADS=1" in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert "NUMEXPR_NUM_THREADS=1" in script
    assert EXPECTED_CONDA_PREFIX_FRAGMENT in script
    assert 'conda activate "$WRITINGRING_CONDA_PREFIX"' in script
    assert 'python -c "import numpy, pandas, sklearn, torch"' in script


def test_pipeline_orders_baseline_array_finalizer_with_afterok() -> None:
    script = _read_bash("submit_exp_0_2_pipeline.bash")
    assert "run_exp_0_2_baselines_cpu.bash" in script
    assert '--dependency="afterok:${BASELINE_JOB}"' in script
    assert "run_exp_0_2_cpu_array.bash" in script
    assert '--dependency="afterok:${ARRAY_JOB}"' in script
    assert "finalize_exp_0_2_cpu.bash" in script
    assert "120 = 3 channel sets x 2 representations x 4 neural decoders x 5 split seeds" in script
    assert EXPECTED_CONDA_PREFIX_FRAGMENT in script


def test_baseline_and_finalizer_are_single_core_jobs() -> None:
    for name in ("run_exp_0_2_baselines_cpu.bash", "finalize_exp_0_2_cpu.bash"):
        script = _read_bash(name)
        assert "#SBATCH --cpus-per-task=1" in script
        assert EXPECTED_CONDA_PREFIX_FRAGMENT in script
        assert 'conda activate "$WRITINGRING_CONDA_PREFIX"' in script
