from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_2_6_output_readout_loss_shaping as exp726


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_backbone_and_grid_contract() -> None:
    assert exp726.ARCHITECTURE == "234x234"
    assert exp726.ARCHITECTURES[exp726.ARCHITECTURE] == ((2, 3, 4), (2, 3, 4))
    assert exp726.SEEDS == (11, 23, 37)
    assert set(exp726.REGULARIZATIONS) == {"task_only", "task_plus_reg"}
    assert exp726.OBJECTIVE == "whole_count_ce"
    assert len(exp726.source_specs()) == 6
    assert len(exp726.frozen_head_specs()) == 12
    assert len(exp726.e2e_train_specs()) == 12
    assert len(exp726.probe_specs()) == 18


def test_source_is_exact_exp725_c2_anchor() -> None:
    spec = exp726.SourceSpec("task_only", 11)
    ref = spec.exp725_spec
    assert ref.architecture == "234x234"
    assert ref.regularization == "task_only"
    assert ref.seed == 11
    assert ref.objective == "whole_count_ce"
    assert ref.output_alpha == 0.0
    assert exp726.SOURCE_BETA == 0.5


def test_frozen_head_pairing_excludes_head_mode() -> None:
    src = exp726.SourceSpec("task_only", 23)
    analog = exp726.FrozenHeadSpec(src.regularization, src.seed, exp726.HEAD_ANALOG)
    lif = exp726.FrozenHeadSpec(src.regularization, src.seed, exp726.HEAD_LIF)
    assert exp726._head_pair_seed(analog.source, "model_init") == exp726._head_pair_seed(lif.source, "model_init")
    torch.manual_seed(exp726._head_pair_seed(src, "model_init"))
    m0 = exp726.FrozenMatchedHead(exp726.HEAD_ANALOG, 12)
    torch.manual_seed(exp726._head_pair_seed(src, "model_init"))
    m1 = exp726.FrozenMatchedHead(exp726.HEAD_LIF, 12)
    assert torch.equal(m0.output_linear.weight, m1.output_linear.weight)


def test_e2e_pairing_reuses_exp725_seed_stream() -> None:
    src = exp726.SourceSpec("task_plus_reg", 37)
    for role in ("model_init", "train_loader", "val_loader", "test_loader"):
        assert exp726._reference_seed(src, role) == exp726.exp725.e2e_pair_seed(src.exp725_spec, role)


def test_charge_preserving_identity_beta1() -> None:
    rng = np.random.default_rng(7)
    l2 = rng.integers(0, 2, size=(5, 17, 128), dtype=np.uint8)
    lengths = np.array([17, 16, 12, 9, 5], dtype=np.int64)
    W = rng.normal(0.0, 0.08, size=(12, 128))
    gain = 1.75
    sim = exp726._simulate_unipolar(l2, lengths, W, beta=1.0, cap=1, scale=gain)
    analog = exp726._analog_scores(l2, lengths, W, scale=gain)
    assert np.allclose(sim["charge_scores"], analog, atol=1e-10)
    assert sim["diagnostics"]["max_abs_charge_identity_error"] < 1e-10


def test_leaky_charge_accounting_identity() -> None:
    rng = np.random.default_rng(8)
    l2 = rng.integers(0, 2, size=(4, 13, 128), dtype=np.uint8)
    lengths = np.array([13, 11, 7, 3], dtype=np.int64)
    W = rng.normal(0.0, 0.05, size=(12, 128))
    sim = exp726._simulate_unipolar(l2, lengths, W, beta=0.5, cap=1, scale=1.0)
    reconstructed = sim["spike_charge"] + sim["leak"] + sim["residual"]
    assert np.allclose(sim["input_sum"], reconstructed, atol=1e-10)
    assert sim["diagnostics"]["max_abs_charge_identity_error"] < 1e-10


def test_gain_calibration_has_no_test_argument() -> None:
    names = exp726.calibrate_gain.__code__.co_varnames[: exp726.calibrate_gain.__code__.co_argcount]
    assert names == ("val_l2", "val_y", "val_lengths", "W")


def test_notebook_is_analysis_only_and_aggregate_only() -> None:
    path = REPO_ROOT / "notebooks" / "experiment_7_2_6_output_readout_loss_shaping.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for required in (
        "crosscheck_2x2_summary.csv",
        "mechanism_ladder_summary.csv",
        "beta_sweep_summary.csv",
        "e2e_l2_probe_summary.csv",
        "representation_shaping_delta_summary.csv",
    ):
        assert required in text
    for forbidden in (
        "torch.load",
        "run_e2e(",
        "run_frozen_head(",
        "run_mechanism(",
        "calibrate_gain(",
        "sbatch",
        "subprocess",
    ):
        assert forbidden not in text


def test_slurm_pipeline_contract() -> None:
    base = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    expected = {
        "prepare_exp_7_2_6_frozen_l2_cpu_array.bash": "#SBATCH --array=0-5%6",
        "run_exp_7_2_6_frozen_head_cpu_array.bash": "#SBATCH --array=0-11%12",
        "run_exp_7_2_6_mechanism_cpu_array.bash": "#SBATCH --array=0-5%6",
        "run_exp_7_2_6_e2e_cpu_array.bash": "#SBATCH --array=0-11%12",
        "run_exp_7_2_6_probe_cpu_array.bash": "#SBATCH --array=0-17%18",
    }
    for filename, token in expected.items():
        text = (base / filename).read_text(encoding="utf-8")
        assert token in text
        assert "--cpus-per-task=1" in text
        assert "--device cpu --threads 1" in text
    submit = (base / "submit_exp_7_2_6_cpu.bash").read_text(encoding="utf-8")
    assert submit.count("afterok:") == 5
    assert "finalize_exp_7_2_6_cpu.bash" in submit
