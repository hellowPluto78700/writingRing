from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from scripts import experiment_11_0_rsnn_history_internalization as exp110


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_factorial_contract() -> None:
    assert exp110.PROTOCOL_VERSION == "d1_l1mem2_rsnn_fusion_internalization_v1"
    assert exp110.VARIANT == "postencode_mask"
    assert exp110.ROTATION == 0
    assert exp110.MODEL_SEEDS == (11, 23, 37)
    assert exp110.L1_INIT_MODES == ("dynamics_only", "pretrained_input")
    assert exp110.TOPOLOGIES == ("ff", "diagonal", "dense")
    specs = exp110.run_specs()
    assert len(specs) == 18
    assert exp110.EXPECTED_RUNS == 18
    assert len({spec.key for spec in specs}) == 18


def test_recurrent_parameter_counts() -> None:
    assert exp110.recurrent_param_count("ff") == 0
    assert exp110.recurrent_param_count("diagonal") == exp110.WIDTH
    assert exp110.recurrent_param_count("dense") == exp110.WIDTH * exp110.WIDTH


def test_context_and_fusion_tau_are_exp10_mem2_scale() -> None:
    expected = exp110.exp1021.tau_mem_ms_from_shift(2)
    assert math.isclose(exp110.CONTEXT_TAU_MS, expected, rel_tol=1e-12)
    assert math.isclose(exp110.FUSION_TAU_MS, expected, rel_tol=1e-12)
    decay = exp110.decay_from_tau_ms(exp110.CONTEXT_TAU_MS, 64.0)
    assert math.isclose(decay, 0.75, rel_tol=1e-9)


def test_l1_keeps_exp10_multitau_synaptic_dynamics() -> None:
    spec = exp110.RunSpec("dynamics_only", "ff", 11)
    model = exp110.Exp110Net(spec, n_classes=12, fs=64.0)
    assert tuple(exp110.L1_SYN_SHIFTS) == (2, 3, 4)
    assert torch.equal(model.l1_alpha, exp110.exp811._alpha_vector("binary"))
    assert math.isclose(
        model.l1_lif.beta,
        exp110.exp1021.beta_from_mem_shift(2, 64.0),
        rel_tol=1e-12,
    )


def test_topology_modules_are_strictly_distinct() -> None:
    ff = exp110.Exp110Net(exp110.RunSpec("dynamics_only", "ff", 11), 12, 64.0)
    diagonal = exp110.Exp110Net(
        exp110.RunSpec("dynamics_only", "diagonal", 11), 12, 64.0
    )
    dense = exp110.Exp110Net(
        exp110.RunSpec("dynamics_only", "dense", 11), 12, 64.0
    )
    assert isinstance(ff.recurrent, exp110.ZeroRecurrent)
    assert isinstance(diagonal.recurrent, exp110.DiagonalRecurrent)
    assert isinstance(dense.recurrent, nn.Linear)
    assert diagonal.recurrent.weight.shape == (exp110.WIDTH,)
    assert dense.recurrent.weight.shape == (exp110.WIDTH, exp110.WIDTH)


def test_diagonal_recurrence_has_no_cross_neuron_mixing() -> None:
    module = exp110.DiagonalRecurrent()
    with torch.no_grad():
        module.weight.copy_(torch.arange(exp110.WIDTH, dtype=torch.float32))
    spikes = torch.zeros(1, exp110.WIDTH)
    spikes[0, 7] = 1.0
    output = module(spikes)
    assert output[0, 7].item() == 7.0
    assert torch.count_nonzero(output).item() == 1


def test_paired_initialization_matches_shared_layers_and_dense_diagonal() -> None:
    diag_spec = exp110.RunSpec("dynamics_only", "diagonal", 23)
    dense_spec = exp110.RunSpec("dynamics_only", "dense", 23)
    diag = exp110.Exp110Net(diag_spec, 12, 64.0)
    dense = exp110.Exp110Net(dense_spec, 12, 64.0)
    exp110._initialize_paired(diag, diag_spec)
    exp110._initialize_paired(dense, dense_spec)

    for left, right in (
        (diag.l1_input.weight, dense.l1_input.weight),
        (diag.rsnn_input.weight, dense.rsnn_input.weight),
        (diag.fusion_local.weight, dense.fusion_local.weight),
        (diag.fusion_context.weight, dense.fusion_context.weight),
        (diag.output_linear.weight, dense.output_linear.weight),
    ):
        assert torch.equal(left, right)
    assert isinstance(diag.recurrent, exp110.DiagonalRecurrent)
    assert isinstance(dense.recurrent, nn.Linear)
    assert torch.equal(
        diag.recurrent.weight,
        torch.diagonal(dense.recurrent.weight),
    )


def test_fusion_is_feedforward_and_uses_local_plus_context() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    assert "self.fusion_local(z_t) + self.fusion_context(r_t)" in source
    assert "self.recurrent(prev_rsnn)" in source


def test_pretrained_contract_copies_only_l1_input_and_keeps_trainable() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    block = source[
        source.index("def _load_pretrained_l1"):
        source.index("def _valid_mean")
    ]
    assert 'state.get("hidden_linears.0.weight")' in block
    assert "model.l1_input.weight.copy_" in block
    assert "model.rsnn_input.weight.copy_" not in block
    assert "model.fusion_local.weight.copy_" not in block
    assert "requires_grad" in block


def test_forward_trajectory_contract() -> None:
    spec = exp110.RunSpec("dynamics_only", "diagonal", 11)
    model = exp110.Exp110Net(spec, 12, 64.0)
    exp110._initialize_paired(model, spec)
    x = torch.randn(2, 9, 30)
    trajectory = model.forward_trajectory(x)
    assert trajectory["fusion_evidence"].shape == (2, 9, 12)
    assert trajectory["rsnn_external_input"].shape == (2, 9, exp110.WIDTH)
    assert trajectory["rsnn_recurrent_input"].shape == (2, 9, exp110.WIDTH)
    for layer in exp110.LAYERS:
        for state in ("syn_current", "pre_reset", "spike", "post_reset"):
            assert trajectory["hidden"][layer][state].shape == (
                2,
                9,
                exp110.WIDTH,
            )


def test_probe_inventory_has_valid_and_window_supports() -> None:
    names = exp110.probe_names()
    assert len(names) == exp110.EXPECTED_PROBES_PER_RUN
    assert len(set(names)) == len(names)
    for layer in exp110.LAYERS:
        assert f"{layer}__communication__valid__whole_count" in names
        assert f"{layer}__communication__valid__fixed250_count" in names
        assert f"{layer}__communication__window__whole_count" in names
        assert f"{layer}__communication__window__fixed250_count" in names
        assert f"{layer}__pre_reset__valid__whole_mean" in names
        assert f"{layer}__pre_reset__valid__fixed250_ordered_mean" in names


def test_training_is_wcce_only_with_gradient_clipping() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    run_block = source[source.index("def run_one"):source.index("def _run_row")]
    assert "F.cross_entropy(scores, yd)" in run_block
    assert "clip_grad_norm_" in run_block
    assert "tsce" not in run_block.lower()


def test_slurm_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    run = (root / "run_exp_11_0_cpu_array.bash").read_text()
    submit = (root / "submit_exp_11_0_cpu.bash").read_text()
    assert "#SBATCH --array=0-17%18" in run
    assert "#SBATCH --cpus-per-task=1" in run
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f"export {name}=1" in run
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit


def test_notebook_is_aggregation_only() -> None:
    notebook = (
        REPO_ROOT / "notebooks" / "experiment_11_0_rsnn_history_internalization.ipynb"
    ).read_text()
    assert exp110.PROTOCOL_VERSION in notebook
    assert "method_summary.csv" in notebook
    assert "temporal_gap_summary.csv" in notebook
    assert "paired_contrast_summary.csv" in notebook
    assert "run_one(" not in notebook
    assert "torch.optim" not in notebook
