from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

import scripts.run_experiment_a as runner_a
import scripts.run_experiment_b as runner_b
import scripts.run_experiment_c as runner_c
import scripts.run_experiment_d as runner_d
from snn.accel_reconstruction_eval import (
    build_probe_model,
    experiment_a_config,
    experiment_b_config,
    experiment_c_config,
    experiment_d_config,
    restore_model_from_checkpoint,
)


@pytest.mark.parametrize("variant,embedding_dim", [("cnn_s", 8), ("cnn_m", 64), ("cnn_l", 256)])
def test_probe_variants_preserve_masked_shape_and_embedding(variant: str, embedding_dim: int) -> None:
    model = build_probe_model(variant, num_classes=3)
    x = torch.randn(2, 3, 19)
    mask = torch.tensor([[True] * 11 + [False] * 8, [True] * 19])
    logits, embedding = model(x, valid_mask=mask)
    assert logits.shape == (2, 3)
    assert embedding.shape == (2, embedding_dim)
    padded = x.clone()
    padded[0, :, 11:] = 10_000
    masked_logits, _ = model(padded, valid_mask=mask)
    torch.testing.assert_close(logits[0], masked_logits[0])


def _checkpoint(path: Path, variant: str) -> Path:
    model = build_probe_model(variant, num_classes=2)
    payload = {
        "random_seed": 12345,
        "model_state_dict": model.state_dict(),
        "model_config": model.architecture_config(),
        "architecture_variant": variant,
        "class_to_idx": {"a": 0, "b": 1},
        "normalization_mean": [0.0, 0.0, 0.0],
        "normalization_std": [1.0, 1.0, 1.0],
        "train_users": ["u0"],
        "val_users": ["u1"],
        "test_users": ["u2"],
        "experiment_config": {"random_seed": 12345},
    }
    torch.save(payload, path)
    return path


def test_checkpoint_architecture_mismatch_is_explicit(tmp_path: Path) -> None:
    checkpoint = torch.load(_checkpoint(tmp_path / "s.pt", "cnn_s"), weights_only=False)
    with pytest.raises(ValueError, match="architecture mismatch"):
        restore_model_from_checkpoint(build_probe_model("cnn_m", 2), checkpoint)


def test_legacy_checkpoint_restores_only_cnn_l() -> None:
    model = build_probe_model("cnn_l", 2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"class_name": "MaskAwareAccelerationCNN"},
    }
    restore_model_from_checkpoint(model, checkpoint)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        restore_model_from_checkpoint(build_probe_model("cnn_s", 2), checkpoint)


@pytest.mark.parametrize("factory", [experiment_a_config, experiment_b_config, experiment_c_config, experiment_d_config])
@pytest.mark.parametrize("variant", ["cnn_s", "cnn_m", "cnn_l"])
def test_all_abcd_config_entrypoints_select_each_probe(factory, variant: str) -> None:
    config = factory(probe_variant=variant)
    assert config.probe_variant == variant


@pytest.mark.parametrize("module", [runner_a, runner_b, runner_c, runner_d])
def test_all_abcd_cli_entrypoints_expose_probe_selection(module) -> None:
    action = module._build_arg_parser().parse_args(["--probe-variant", "cnn_s"])
    assert action.probe_variant == "cnn_s"


@pytest.mark.parametrize("module", [runner_a, runner_c, runner_d])
def test_cli_passes_probe_variant_into_runner_config(module, monkeypatch) -> None:
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_dir=Path("/tmp/probe-test"),
            checkpoint_path=Path("/tmp/probe-test.pt"),
            baseline_checkpoint_path=Path("/tmp/probe-a.pt"),
            baseline_checkpoint_sha256="test",
            summary=__import__("pandas").DataFrame(),
            raw_summary=__import__("pandas").DataFrame(),
            reconstruction_summary=__import__("pandas").DataFrame(),
            paired_summary=__import__("pandas").DataFrame(),
        )

    runner_name = f"run_experiment_{module.__name__.rsplit('_', 1)[-1]}"
    monkeypatch.setattr(module, runner_name, fake_runner)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--probe-variant", "cnn_s"])
    module.main()
    assert captured["config"].probe_variant == "cnn_s"


@pytest.mark.parametrize("module,runner,argument", [
    (runner_b, runner_b.run_experiment_b, "baseline_checkpoint"),
    (runner_c, runner_c.run_experiment_c, "reference_checkpoint"),
    (runner_d, runner_d.run_experiment_d, "reference_checkpoint"),
])
def test_downstream_probe_mismatch_rejected_before_dataset_load(
    module, runner, argument: str, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(tmp_path / "a.pt", "cnn_s")
    kwargs = {argument: checkpoint, "output_dir": tmp_path / "artifacts", "probe_variant": "cnn_m"}
    if argument == "reference_checkpoint":
        kwargs["allow_new_split"] = False
    with pytest.raises(ValueError, match="mismatch"):
        runner(root=tmp_path / "missing", **kwargs)
