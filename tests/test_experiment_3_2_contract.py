from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts import experiment_3_2_nonlinear_temporal_interaction as exp32


def test_exp32_protocol_and_run_matrix() -> None:
    assert exp32.EXPERIMENT_ID == "experiment_3_2_nonlinear_temporal_interaction"
    assert exp32.PROTOCOL_VERSION == "low_rank_residual_v1"
    assert exp32.SPLIT_SEEDS == (11, 23, 37, 53, 71)
    assert exp32.REPRESENTATIONS == ("relative10", "fixed250", "fixed500")
    assert exp32.RESIDUAL_DECODERS == ("local", "transition", "local_transition")
    assert exp32.RESIDUAL_RANK == 16
    assert exp32.MODEL_INIT_SEED == 2026
    specs = exp32.run_specs()
    assert len(specs) == 45
    assert exp32.EXPECTED_RUNS == 45
    assert len({(s.representation, s.decoder, s.split_seed) for s in specs}) == 45


def test_exp32_model_seed_does_not_depend_on_split_seed() -> None:
    a = exp32.RunSpec("fixed250", "transition", 11)
    b = exp32.RunSpec("fixed250", "transition", 71)
    c = exp32.RunSpec("fixed250", "local", 11)
    assert exp32._model_seed(a) == exp32._model_seed(b)
    assert exp32._model_seed(a) != exp32._model_seed(c)


def test_exp32_residual_layers_are_bias_free_and_heads_zero_initialized() -> None:
    model = exp32.ResidualDecoder("local_transition", n_bins=16, n_classes=12, rank=16)
    assert model.local_projection is not None
    assert model.local_head is not None
    assert model.transition_p is not None
    assert model.transition_q is not None
    assert model.transition_head is not None
    assert model.local_projection.bias is None
    assert model.local_head.bias is None
    assert model.transition_p.bias is None
    assert model.transition_q.bias is None
    assert model.transition_head.bias is None
    assert torch.count_nonzero(model.local_head.weight).item() == 0
    assert torch.count_nonzero(model.transition_head.weight).item() == 0


def test_exp32_epoch0_is_exact_frozen_linear_identity() -> None:
    torch.manual_seed(1)
    model = exp32.ResidualDecoder("local_transition", n_bins=8, n_classes=12, rank=16)
    x = torch.randn(5, 8, 30)
    mask = torch.ones(5, 8, dtype=torch.bool)
    base = torch.randn(5, 12)
    with torch.no_grad():
        out = model(x, mask, base)
    assert torch.equal(out, base)


def test_exp32_transition_mask_blocks_padding_boundary() -> None:
    model = exp32.ResidualDecoder("transition", n_bins=8, n_classes=2, rank=1)
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
    mask = torch.tensor([[True, True, True, True, True, False, False, False]])
    with torch.no_grad():
        residual = model.residual_logits(x, mask)

    # Four valid adjacent pairs (0-1, 1-2, 2-3, 3-4), each projection sums 30 ones.
    expected = 4 * 30.0 * 30.0
    assert torch.allclose(residual, torch.full_like(residual, expected))


def test_exp32_local_mask_blocks_invalid_bins() -> None:
    model = exp32.ResidualDecoder("local", n_bins=4, n_classes=1, rank=1)
    assert model.local_projection is not None
    assert model.local_head is not None
    with torch.no_grad():
        model.local_projection.weight.fill_(1.0)
        model.local_head.weight.fill_(1.0)
    x = torch.ones(1, 4, 30)
    x[:, 2:] = 1000.0
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        residual = model.residual_logits(x, mask)
    single = torch.nn.functional.gelu(torch.tensor(30.0))
    assert torch.allclose(residual.squeeze(), 2.0 * single)


def test_exp32_fixed_representation_dimensions_and_partial_bin_mask() -> None:
    class Package:
        def __init__(self) -> None:
            self.padded_spike_imu = np.zeros((1, 256, 36), dtype=np.float32)
            self.padded_spike_imu[0, :105, :30] = 1.0

    cohort = exp32.Cohort(
        manifest=None,  # type: ignore[arg-type]
        packages=[Package()],
        fs=64.0,
        labels=tuple(str(i) for i in range(12)),
    )
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "package_index": 0,
                "segment_index": 0,
                "label_idx": 0,
                "valid_length": 105,
            }
        ]
    )
    x250, m250, _, meta250 = exp32._fixed_representation(cohort, frame, 250.0)
    x500, m500, _, meta500 = exp32._fixed_representation(cohort, frame, 500.0)
    assert x250.shape == (1, 16, 30)
    assert x500.shape == (1, 8, 30)
    assert meta250["samples_per_bin"] == 16
    assert meta500["samples_per_bin"] == 32
    assert int(m250.sum()) == 7  # ceil(105 / 16)
    assert int(m500.sum()) == 4  # ceil(105 / 32)
    assert np.all(x250[0, 6] == 9.0)  # samples 96..104 are retained, not truncated
    assert np.all(x500[0, 3] == 9.0)


def test_exp32_slurm_array_is_one_atomic_run_per_cpu() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runner = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_3_2_cpu_array.bash"
    ).read_text(encoding="utf-8")
    finalizer = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "finalize_exp_3_2_cpu.bash"
    ).read_text(encoding="utf-8")
    submitter = (
        repo_root / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_3_2_pipeline.bash"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-44%45" in runner
    assert "#SBATCH --cpus-per-task=1" in runner
    assert "OMP_NUM_THREADS=1" in runner
    assert "MKL_NUM_THREADS=1" in runner
    assert "OPENBLAS_NUM_THREADS=1" in runner
    assert "NUMEXPR_NUM_THREADS=1" in runner
    assert (
        "python -u -m scripts.experiment_3_2_nonlinear_temporal_interaction "
        "run-one --array-task-id \"$TASK_ID\" --device cpu"
    ) in runner
    assert "python -u -m scripts.experiment_3_2_nonlinear_temporal_interaction finalize" in finalizer
    assert "--dependency=afterok:${EXP32_ARRAY}" in submitter
