from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import experiment_3_1_raw_vs_snn_representation_value as exp31
from scripts import experiment_3_3_snn_nonlinear_accessibility as exp33


def test_exp33_protocol_and_run_matrix() -> None:
    assert exp33.EXPERIMENT_ID == "experiment_3_3_snn_nonlinear_accessibility"
    assert exp33.PROTOCOL_VERSION == "matched_raw_snn_residual_v1"
    assert exp33.SNN_SEEDS == (11, 23, 101)
    assert exp33.SOURCES == ("raw", "snn_l2")
    assert exp33.TEMPORAL_REPRESENTATIONS == ("fixed250", "relative10")
    assert exp33.RESIDUAL_DECODERS == (
        "local",
        "transition",
        "local_transition",
    )
    assert exp33.PRIMARY_REPRESENTATION == "fixed250"
    assert exp33.PRIMARY_DECODER == "local_transition"
    specs = exp33.run_specs()
    assert exp33.EXPECTED_RUNS == 24
    assert len(specs) == 24
    assert len({spec.key for spec in specs}) == 24
    assert sum(spec.source == "raw" for spec in specs) == 6
    assert sum(spec.source == "snn_l2" for spec in specs) == 18
    assert all(spec.snn_seed is None for spec in specs if spec.source == "raw")
    assert {
        spec.snn_seed for spec in specs if spec.source == "snn_l2"
    } == {11, 23, 101}


def test_exp33_model_seed_is_source_and_snn_seed_independent() -> None:
    raw = exp33.RunSpec("raw", "fixed250", "transition", None)
    snn11 = exp33.RunSpec("snn_l2", "fixed250", "transition", 11)
    snn101 = exp33.RunSpec("snn_l2", "fixed250", "transition", 101)
    local = exp33.RunSpec("raw", "fixed250", "local", None)
    assert exp33._model_seed(raw) == exp33._model_seed(snn11)
    assert exp33._model_seed(snn11) == exp33._model_seed(snn101)
    assert exp33._model_seed(raw) != exp33._model_seed(local)


def test_exp33_linear_seed_reuses_exp31_contract() -> None:
    assert exp33._linear_seed("raw", "fixed250") == exp31._probe_seed(
        "raw", "fixed250_ordered"
    )
    assert exp33._linear_seed("snn_l2", "fixed250") == exp31._probe_seed(
        "snn_l2", "fixed250_ordered"
    )
    assert exp33._linear_seed("raw", "relative10") == exp31._probe_seed(
        "raw", "relative10_ordered"
    )
    assert exp33._linear_seed("snn_l2", "relative10") == exp31._probe_seed(
        "snn_l2", "relative10_ordered"
    )


def test_exp33_residual_supports_raw_and_snn_channel_counts() -> None:
    for channels in (30, 128):
        model = exp33.ResidualDecoder(
            "local_transition",
            n_bins=16,
            n_channels=channels,
            n_classes=12,
            rank=16,
        )
        assert model.local_projection is not None
        assert model.local_head is not None
        assert model.transition_p is not None
        assert model.transition_q is not None
        assert model.transition_head is not None
        assert model.local_projection.in_features == channels
        assert model.transition_p.in_features == channels
        assert model.transition_q.in_features == channels
        assert model.local_projection.bias is None
        assert model.local_head.bias is None
        assert model.transition_p.bias is None
        assert model.transition_q.bias is None
        assert model.transition_head.bias is None
        assert torch.count_nonzero(model.local_head.weight).item() == 0
        assert torch.count_nonzero(model.transition_head.weight).item() == 0


def test_exp33_epoch0_is_exact_frozen_linear_identity() -> None:
    torch.manual_seed(1)
    for channels in (30, 128):
        model = exp33.ResidualDecoder(
            "local_transition",
            n_bins=8,
            n_channels=channels,
            n_classes=12,
            rank=16,
        )
        x = torch.randn(5, 8, channels)
        mask = torch.ones(5, 8, dtype=torch.bool)
        base = torch.randn(5, 12)
        with torch.no_grad():
            out = model(x, mask, base)
        assert torch.equal(out, base)


def test_exp33_transition_mask_blocks_padding_boundary() -> None:
    model = exp33.ResidualDecoder(
        "transition",
        n_bins=8,
        n_channels=30,
        n_classes=2,
        rank=1,
    )
    assert model.transition_p is not None
    assert model.transition_q is not None
    assert model.transition_head is not None
    with torch.no_grad():
        model.transition_p.weight.fill_(1.0)
        model.transition_q.weight.fill_(1.0)
        model.transition_head.weight.fill_(1.0)

    x = torch.zeros(1, 8, 30)
    x[:, :5] = 1.0
    x[:, 5:] = 1000.0
    mask = torch.tensor(
        [[True, True, True, True, True, False, False, False]]
    )
    with torch.no_grad():
        residual = model.residual_logits(x, mask)

    expected = 4 * 30.0 * 30.0
    assert torch.allclose(
        residual,
        torch.full_like(residual, expected),
    )


def test_exp33_fixed250_mask_keeps_partial_final_bin() -> None:
    lengths = np.asarray([105, 256], dtype=np.int64)
    mask = exp33._mask_from_lengths(
        lengths,
        n_bins=16,
        representation="fixed250",
        bin_steps=16,
    )
    assert mask.shape == (2, 16)
    assert int(mask[0].sum()) == 7
    assert int(mask[1].sum()) == 16

    relative = exp33._mask_from_lengths(
        lengths,
        n_bins=10,
        representation="relative10",
        bin_steps=16,
    )
    assert relative.all()


def test_exp33_slurm_array_is_one_atomic_run_per_cpu() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "run_exp_3_3_cpu_array.bash"
    ).read_text(encoding="utf-8")
    finalizer = (
        repo_root
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "finalize_exp_3_3_cpu.bash"
    ).read_text(encoding="utf-8")
    submitter = (
        repo_root
        / "scripts"
        / "bash_script"
        / "SNN_Bash"
        / "submit_exp_3_3_pipeline.bash"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --array=0-23%24" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    for variable in (
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ):
        assert variable in runner
    assert (
        "python -u -m scripts.experiment_3_3_snn_nonlinear_accessibility "
        'run-one --array-task-id "$TASK_ID" --device cpu'
    ) in runner
    assert (
        "python -u -m scripts.experiment_3_3_snn_nonlinear_accessibility finalize"
        in finalizer
    )
    assert "--dependency=afterok:${EXP33_ARRAY}" in submitter


def test_exp33_notebook_declares_hypothesis_validation_and_conclusion() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = (
        repo_root
        / "notebooks"
        / "experiment_3_3_snn_nonlinear_accessibility.ipynb"
    )
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "Hypothesis" in markdown
    assert "Validation standard" in markdown
    assert "Conclusion" in markdown
    assert "Fixed250" in markdown
    assert "Local+Transition" in markdown
