from pathlib import Path

import numpy as np
import torch

from scripts import experiment_7_3_9_pretrained_a2_depth_extension as exp739


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_cases_and_task_counts_are_exact() -> None:
    assert exp739.SOURCE_METHOD == "A2_e2e_linear_wcce"
    assert exp739.SOURCE_OBJECTIVE == "wcce"
    assert exp739.SEEDS == (11, 23, 37)
    assert exp739.SHIFTS == ((2, 3, 4), (2, 3, 4), (2, 3, 4))
    assert exp739.CASES == (
        "C0_a2_2layer",
        "C1_frozen_a2_train_l3",
        "C2_c1_init_e2e_3layer",
    )
    assert len(exp739.c1_specs()) == 3
    assert len(exp739.c2_specs()) == 3
    assert len(exp739.probe_specs()) == 9
    assert exp739.EXPECTED_PROBE_ROWS == 96


def test_three_layer_model_is_234x234x234_and_bias_free() -> None:
    model = exp739.DepthExtensionNet(n_classes=12, fs=64.0)
    assert len(model.hidden_linears) == 3
    assert len(model.hidden_lifs) == 3
    assert model.hidden_linears[0].in_features == exp739.exp72.EXPECTED_CHANNELS
    assert all(layer.out_features == 128 for layer in model.hidden_linears)
    assert model.hidden_linears[1].in_features == 128
    assert model.hidden_linears[2].in_features == 128
    assert all(layer.bias is None for layer in model.hidden_linears)
    assert model.output_linear.in_features == 128
    assert model.output_linear.out_features == 12
    assert model.output_linear.bias is None
    assert torch.equal(model.alpha_0, model.alpha_1)
    assert torch.equal(model.alpha_1, model.alpha_2)


def test_c1_trains_only_w3_and_c2_unfreezes_all() -> None:
    model = exp739.DepthExtensionNet(n_classes=12, fs=64.0)
    exp739._configure_trainable(model, exp739.C1_CASE)
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"hidden_linears.2.weight"}
    assert exp739._trainable_parameter_count(model) == 128 * 128

    exp739._configure_trainable(model, exp739.C2_CASE)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_no_bias_probe_is_not_mean_centered_and_has_no_intercept() -> None:
    train_x = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [4.0, 1.0], [5.0, 0.0]],
        dtype=np.float64,
    )
    train_y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    val_x = train_x.copy()
    val_y = train_y.copy()
    test_x = train_x.copy()
    test_y = train_y.copy()

    probe = exp739._fit_no_bias_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        seed=11,
    )
    assert probe["fit_intercept"] is False
    assert probe["scaler_with_mean"] is False
    assert probe["normalization"] == "train_only_scale_no_centering"
    assert probe["probe_C"] in exp739.PROBE_C_GRID


def test_c2_source_contract_keeps_c1_as_epoch_zero_candidate() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_7_3_9_pretrained_a2_depth_extension.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("def run_c2("):source.index("def _load_case_model(")]
    assert "_load_c1_model" in block
    assert "best_epoch = 0" in block
    assert '"epoch0_candidate": True' in block
    assert '"optimizer_state_reused": False' in block
    assert "optimizer = torch.optim.Adam(" in block


def test_training_creates_history_parent_before_csv_write() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_7_3_9_pretrained_a2_depth_extension.py"
    ).read_text(encoding="utf-8")
    c1 = source[source.index("def run_c1("):source.index("def run_c2(")]
    c2 = source[source.index("def run_c2("):source.index("def _load_case_model(")]
    for block in (c1, c2):
        assert "history_path = _history_path(" in block
        assert "history_path.parent.mkdir(parents=True, exist_ok=True)" in block
        assert "pd.DataFrame(history).to_csv(history_path, index=False)" in block


def test_slurm_arrays_and_dependency_chain() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    assert "#SBATCH --array=0-2%3" in (
        root / "run_exp_7_3_9_c1_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-2%3" in (
        root / "run_exp_7_3_9_c2_cpu_array.bash"
    ).read_text()
    assert "#SBATCH --array=0-8%9" in (
        root / "run_exp_7_3_9_probe_cpu_array.bash"
    ).read_text()

    submit = (root / "submit_exp_7_3_9_cpu.bash").read_text()
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${c1_job}' in submit
    assert 'afterok:${c2_job}' in submit
    assert 'afterok:${probe_job}' in submit


def test_notebook_is_analysis_only() -> None:
    text = (
        REPO_ROOT / "notebooks" / "experiment_7_3_9_pretrained_a2_depth_extension.ipynb"
    ).read_text(encoding="utf-8")
    for filename in (
        "native_runs.csv",
        "native_summary.csv",
        "probe_runs.csv",
        "probe_summary.csv",
        "depth_contrasts.csv",
        "bias_contrasts.csv",
        "temporal_contrasts.csv",
        "layer_contrasts.csv",
    ):
        assert filename in text
    assert "import torch" not in text
    assert "subprocess" not in text
    assert "sbatch" not in text
