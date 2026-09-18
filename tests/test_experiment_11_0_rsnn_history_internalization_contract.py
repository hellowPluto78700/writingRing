from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from scripts import experiment_11_0_rsnn_history_internalization as exp110


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_spec(
    variant: str = exp110.VARIANT_ORIGINAL,
    l1_init: str = exp110.L1_INIT_DYNAMICS_ONLY,
    topology: str = exp110.TOPOLOGY_FF,
    seed: int = 11,
) -> exp110.RunSpec:
    return exp110.RunSpec(variant, l1_init, topology, seed)


def test_factorial_contract() -> None:
    assert exp110.PROTOCOL_VERSION == "d0_d1_l1mem2_rsnn_fusion_internalization_v2"
    assert exp110.VARIANTS == ("original", "postencode_mask")
    assert exp110.ROTATION == 0
    assert exp110.MODEL_SEEDS == (11, 23, 37)
    assert exp110.L1_INIT_MODES == ("dynamics_only", "pretrained_input")
    assert exp110.TOPOLOGIES == ("ff", "diagonal", "dense")
    specs = exp110.run_specs()
    assert len(specs) == 36
    assert exp110.EXPECTED_RUNS == 36
    assert len({spec.key for spec in specs}) == 36
    assert {spec.variant for spec in specs} == set(exp110.VARIANTS)


def test_d0_source_stage_contract() -> None:
    specs = exp110.source_specs()
    assert len(specs) == 3
    assert exp110.EXPECTED_SOURCE_RUNS == 3
    assert {spec.variant for spec in specs} == {exp110.VARIANT_ORIGINAL}
    assert {spec.seed for spec in specs} == set(exp110.MODEL_SEEDS)


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
    model = exp110.Exp110Net(make_spec(), n_classes=12, fs=64.0)
    assert tuple(exp110.L1_SYN_SHIFTS) == (2, 3, 4)
    assert torch.equal(model.l1_alpha, exp110.exp811._alpha_vector("binary"))
    assert math.isclose(
        model.l1_lif.beta,
        exp110.exp1021.beta_from_mem_shift(2, 64.0),
        rel_tol=1e-12,
    )


def test_topology_modules_are_strictly_distinct() -> None:
    ff = exp110.Exp110Net(make_spec(topology="ff"), 12, 64.0)
    diagonal = exp110.Exp110Net(make_spec(topology="diagonal"), 12, 64.0)
    dense = exp110.Exp110Net(make_spec(topology="dense"), 12, 64.0)
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


def test_paired_initialization_is_shared_across_variants() -> None:
    d0_spec = make_spec(
        variant=exp110.VARIANT_ORIGINAL,
        topology=exp110.TOPOLOGY_DENSE,
        seed=23,
    )
    d1_spec = make_spec(
        variant=exp110.VARIANT_POSTENCODE,
        topology=exp110.TOPOLOGY_DENSE,
        seed=23,
    )
    d0 = exp110.Exp110Net(d0_spec, 12, 64.0)
    d1 = exp110.Exp110Net(d1_spec, 12, 64.0)
    exp110._initialize_paired(d0, d0_spec)
    exp110._initialize_paired(d1, d1_spec)
    for left, right in (
        (d0.l1_input.weight, d1.l1_input.weight),
        (d0.rsnn_input.weight, d1.rsnn_input.weight),
        (d0.fusion_local.weight, d1.fusion_local.weight),
        (d0.fusion_context.weight, d1.fusion_context.weight),
        (d0.output_linear.weight, d1.output_linear.weight),
        (d0.recurrent.weight, d1.recurrent.weight),
    ):
        assert torch.equal(left, right)


def test_dense_and_diagonal_share_paired_diagonal_initialization() -> None:
    diag_spec = make_spec(topology="diagonal", seed=23)
    dense_spec = make_spec(topology="dense", seed=23)
    diag = exp110.Exp110Net(diag_spec, 12, 64.0)
    dense = exp110.Exp110Net(dense_spec, 12, 64.0)
    exp110._initialize_paired(diag, diag_spec)
    exp110._initialize_paired(dense, dense_spec)
    assert torch.equal(
        diag.recurrent.weight,
        torch.diagonal(dense.recurrent.weight),
    )


def test_pretrained_source_routing_is_variant_specific() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    block = source[
        source.index("def _source_weight_and_meta"):
        source.index("def _load_pretrained_l1")
    ]
    assert "if spec.variant == VARIANT_POSTENCODE" in block
    assert "_d1_source_checkpoint_path" in block
    assert "elif spec.variant == VARIANT_ORIGINAL" in block
    assert "_source_artifact" in block


def test_pretrained_contract_copies_only_l1_input_and_keeps_trainable() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    block = source[
        source.index("def _load_pretrained_l1"):
        source.index("def _valid_mean")
    ]
    assert "model.l1_input.weight.copy_" in block
    assert "model.rsnn_input.weight.copy_" not in block
    assert "model.fusion_local.weight.copy_" not in block
    assert "requires_grad" in block


def test_forward_trajectory_contract() -> None:
    spec = make_spec(topology="diagonal")
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


def test_finalizer_contains_paired_d1_d0_contrast() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_11_0_rsnn_history_internalization.py"
    ).read_text()
    assert '"d1_minus_d0"' in source
    assert "VARIANT_POSTENCODE" in source
    assert "VARIANT_ORIGINAL" in source


def test_slurm_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    source_run = (root / "run_exp_11_0_source_cpu_array.bash").read_text()
    run = (root / "run_exp_11_0_cpu_array.bash").read_text()
    submit = (root / "submit_exp_11_0_cpu.bash").read_text()
    assert "#SBATCH --array=0-2%3" in source_run
    assert "#SBATCH --array=0-35%36" in run
    assert "#SBATCH --cpus-per-task=1" in source_run
    assert "#SBATCH --cpus-per-task=1" in run
    for script in (source_run, run):
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f"export {name}=1" in script
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${source_job}' in submit
    assert 'afterok:${array_job}' in submit


def test_notebook_is_aggregation_only() -> None:
    notebook = (
        REPO_ROOT / "notebooks" / "experiment_11_0_rsnn_history_internalization.ipynb"
    ).read_text()
    assert exp110.PROTOCOL_VERSION in notebook
    assert "method_summary.csv" in notebook
    assert "temporal_gap_summary.csv" in notebook
    assert "paired_contrast_summary.csv" in notebook
    assert "d1_minus_d0" in notebook
    assert "run_one(" not in notebook
    assert "torch.optim" not in notebook
