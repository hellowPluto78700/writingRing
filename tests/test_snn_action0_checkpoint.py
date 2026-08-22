from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
import torch

from snn.train_action0 import (
    CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_payload,
    _load_checkpoint,
    _validate_split_padded_lengths,
)
from snn.action0_dataset import Action0DatasetError, PaddingDatasetMetadata


def _args() -> Any:
    return SimpleNamespace(
        dataset_variant="lowpass",
        neurons_network=[24, 24, 24],
        shift_syn=2,
        shift_mem=1,
        sample_freq=200.0,
        train_users=["user_0", "user_1"],
        val_users=["user_2"],
        test_users=["user_3"],
        num_epochs=20,
        learning_rate=5e-4,
        spike_regularization=0.0,
        random_seed=12345,
    )


def _write_checkpoint(tmp_path: Path) -> tuple[Path, torch.nn.Module, Any, dict[str, int]]:
    source_model = torch.nn.Linear(15, 2)
    args = _args()
    class_to_idx = {"A": 0, "B": 1}
    producer_metadata = PaddingDatasetMetadata(
        input_kind="spike-imu",
        feature_schema="signed_wavelet_events_plus_imu_v1",
        channel_count=21,
        target_length=4,
        sampling_rate_hz=200.0,
        padding_side="right",
        event_representation="unsigned",
        event_feature_schema="custom_wavelet_abs_rectified_events_v1",
        event_channel_count=15,
        spike_encoder_spec_sha256="0" * 64,
    )
    payload = _checkpoint_payload(
        model=source_model,
        optimizer=torch.optim.Adam(source_model.parameters(), lr=args.learning_rate),
        epoch=7,
        best_val_metric=0.75,
        args=args,
        class_to_idx=class_to_idx,
        producer_metadata=producer_metadata,
    )
    path = tmp_path / "action0.pt"
    torch.save(payload, path)
    return path, source_model, args, class_to_idx


def _load(
    path: Path,
    model: torch.nn.Module,
    args: Any,
    class_to_idx: dict[str, int],
) -> None:
    producer_metadata = PaddingDatasetMetadata(
        input_kind="spike-imu",
        feature_schema="signed_wavelet_events_plus_imu_v1",
        channel_count=21,
        target_length=4,
        sampling_rate_hz=200.0,
        padding_side="right",
        event_representation="unsigned",
        event_feature_schema="custom_wavelet_abs_rectified_events_v1",
        event_channel_count=15,
        spike_encoder_spec_sha256="0" * 64,
    )
    _load_checkpoint(
        checkpoint_path=path,
        model=model,
        device=torch.device("cpu"),
        class_to_idx=class_to_idx,
        dataset_variant=args.dataset_variant,
        args=args,
        producer_metadata=producer_metadata,
    )


def test_schema_v1_payload_is_model_only_and_restores_weights(tmp_path: Path) -> None:
    path, source_model, checkpoint_args, class_to_idx = _write_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert "optimizer_state_dict" not in payload
    assert "epoch" not in payload
    assert "best_val_metric" not in payload
    assert payload["input_channel_count"] == 15
    assert payload["event_representation"] == "unsigned"
    assert payload["event_feature_schema"] == "custom_wavelet_abs_rectified_events_v1"
    assert payload["encoder_spec_sha256"] == "0" * 64

    requested_args = _args()
    requested_args.num_epochs = 3
    requested_args.learning_rate = 1e-3
    requested_args.spike_regularization = 0.25
    requested_args.random_seed = 9876
    target_model = torch.nn.Linear(15, 2)
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()

    _load(path, target_model, requested_args, class_to_idx)

    for target, source in zip(target_model.parameters(), source_model.parameters()):
        torch.testing.assert_close(target, source)
    assert checkpoint_args.train_users == requested_args.train_users


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("dataset_variant", "xylo"),
        ("boundary", "segment"),
        ("class_to_idx", {"A": 1, "B": 0}),
        ("input_channels", [1, 16]),
        ("input_channel_count", 30),
        ("num_inputs", 21),
        ("event_representation", "signed"),
        ("event_feature_schema", "other_events_v1"),
        ("encoder_spec_sha256", "1" * 64),
        ("num_outputs", 3),
        ("hidden_sizes", [16, 24, 24]),
        ("shift_syn", 3),
        ("shift_mem", 2),
        ("sample_freq", 201.0),
        ("train_users", ["user_1", "user_0"]),
        ("val_users", ["user_3"]),
        ("test_users", ["user_2"]),
    ],
)
def test_fixed_checkpoint_fields_fail_with_values(
    tmp_path: Path, field: str, replacement: object
) -> None:
    path, _, args, class_to_idx = _write_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError) as error_info:
        _load(path, torch.nn.Linear(15, 2), args, class_to_idx)
    message = str(error_info.value)
    assert field in message
    assert "checkpoint value=" in message
    assert "requested value=" in message


@pytest.mark.parametrize("schema_value", [None, 2, "1"])
def test_missing_or_unknown_checkpoint_schema_is_rejected(
    tmp_path: Path, schema_value: object
) -> None:
    path, _, args, class_to_idx = _write_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if schema_value is None:
        del payload["checkpoint_schema_version"]
    else:
        payload["checkpoint_schema_version"] = schema_value
    torch.save(payload, path)

    with pytest.raises(ValueError, match="checkpoint_schema_version"):
        _load(path, torch.nn.Linear(15, 2), args, class_to_idx)


def test_legacy_unversioned_state_dict_is_rejected(tmp_path: Path) -> None:
    path, source_model, args, class_to_idx = _write_checkpoint(tmp_path)
    torch.save(source_model.state_dict(), path)

    with pytest.raises(ValueError, match="checkpoint_schema_version"):
        _load(path, torch.nn.Linear(15, 2), args, class_to_idx)


def test_incompatible_checkpoint_is_rejected_before_model_state_application(
    tmp_path: Path,
) -> None:
    path, _, args, class_to_idx = _write_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["shift_syn"] = 99
    torch.save(payload, path)

    target_model = torch.nn.Linear(15, 2)
    with torch.no_grad():
        target_model.weight.fill_(4.0)
        target_model.bias.fill_(5.0)
    before = {name: value.detach().clone() for name, value in target_model.state_dict().items()}

    with pytest.raises(ValueError, match="shift_syn"):
        _load(path, target_model, args, class_to_idx)

    for name, value in target_model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def _producer_metadata(
    *,
    event_channel_count: int = 30,
    event_feature_schema: str = "custom_wavelet_polarity_split_abs_events_v1",
    encoder_spec_sha256: str = "0" * 64,
) -> PaddingDatasetMetadata:
    return PaddingDatasetMetadata(
        input_kind="spike-imu",
        feature_schema="polarity_split_wavelet_events_plus_imu_v1",
        channel_count=event_channel_count + 6,
        target_length=4,
        sampling_rate_hz=200.0,
        padding_side="right",
        event_representation="unsigned",
        event_feature_schema=event_feature_schema,
        event_channel_count=event_channel_count,
        spike_encoder_spec_sha256=encoder_spec_sha256,
    )


def test_split_validation_rejects_mismatched_event_channel_count() -> None:
    producer = _producer_metadata(event_channel_count=30)
    train = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)
    validation = SimpleNamespace(
        padded_length=4,
        input_channel_count=15,
        producer_metadata=_producer_metadata(event_channel_count=15),
    )
    test = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)

    with pytest.raises(Action0DatasetError, match="event-channel counts"):
        _validate_split_padded_lengths(
            producer,
            train_dataset=train,
            val_dataset=validation,
            test_dataset=test,
        )


def test_split_validation_rejects_different_event_schema_at_same_width() -> None:
    producer = _producer_metadata()
    train = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)
    validation = SimpleNamespace(
        padded_length=4,
        input_channel_count=30,
        producer_metadata=_producer_metadata(event_feature_schema="different_events_v1"),
    )
    test = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)

    with pytest.raises(Action0DatasetError, match="event_feature_schema"):
        _validate_split_padded_lengths(
            producer,
            train_dataset=train,
            val_dataset=validation,
            test_dataset=test,
        )


def test_split_validation_rejects_different_encoder_identity_at_same_schema() -> None:
    producer = _producer_metadata()
    train = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)
    validation = SimpleNamespace(
        padded_length=4,
        input_channel_count=30,
        producer_metadata=_producer_metadata(encoder_spec_sha256="1" * 64),
    )
    test = SimpleNamespace(padded_length=4, input_channel_count=30, producer_metadata=producer)

    with pytest.raises(Action0DatasetError, match="spike_encoder_spec_sha256"):
        _validate_split_padded_lengths(
            producer,
            train_dataset=train,
            val_dataset=validation,
            test_dataset=test,
        )
