"""Train the local SynNet baseline on padded, label-segmented Action0 data."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import random
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .action0_dataset import (
    BOUNDARY,
    INPUT_CHANNEL_COUNT,
    Action0DatasetError,
    Action0SegmentDataset,
    PaddingDatasetMetadata,
    discover_class_to_idx,
    load_padding_dataset_metadata,
    resolve_dataset_root,
    resolve_segmentation_root,
)
from .action0_engine import run_epoch
from .action0_losses import MaskedCrossEntropySpkReg
from .action0_parser import parse_args
from .utils_architectures import createModel


CHECKPOINT_SCHEMA_VERSION: Final[int] = 1


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run an Action0 training job and return its final metrics for callers/tests."""

    args = parse_args(argv)
    _validate_args(args)
    _set_random_seed(args.random_seed)
    device = _resolve_device(args.use_gpu)

    dataset_root = resolve_dataset_root(args.pipeline_root, args.dataset_variant)
    segmentation_root = resolve_segmentation_root(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"dataset root does not exist for variant {args.dataset_variant!r}: {dataset_root}"
        )
    if not segmentation_root.is_dir():
        raise FileNotFoundError(
            "padded segmentation root does not exist: "
            f"{segmentation_root}. The training baseline requires producer padding output."
        )
    producer_metadata = load_padding_dataset_metadata(segmentation_root)
    _validate_sampling_rate(
        cli_sampling_rate=args.sample_freq,
        producer_sampling_rate=producer_metadata.sampling_rate_hz,
    )

    print(f"Dataset variant: {args.dataset_variant}")
    print(f"Boundary: {BOUNDARY}")
    print(f"Dataset root: {dataset_root}")
    print(f"Segmentation root: {segmentation_root}")
    _print_producer_metadata(producer_metadata)

    class_to_idx = discover_class_to_idx(segmentation_root)
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    train_dataset = Action0SegmentDataset(segmentation_root, args.train_users, class_to_idx)
    val_dataset = Action0SegmentDataset(segmentation_root, args.val_users, class_to_idx)
    test_dataset = Action0SegmentDataset(segmentation_root, args.test_users, class_to_idx)
    _validate_split_padded_lengths(
        producer_metadata,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
    _print_dataset_statistics(
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    train_loader, val_loader, test_loader = _create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        device=device,
    )
    first_inputs, first_labels, first_mask = next(iter(train_loader))
    _inspect_first_batch(first_inputs, first_labels, first_mask)

    num_outputs = len(class_to_idx)
    kwargs: dict[str, int] = {}
    if "SynNet" in args.network_type:
        kwargs.update(
            [
                ("shiftSyn", args.shift_syn),
                ("shiftMem", args.shift_mem),
            ]
        )
    model = createModel(
        args.network_type,
        inputSize=INPUT_CHANNEL_COUNT,
        outputSize=num_outputs,
        hiddenSizes=args.neurons_network,
        device=device,
        sampleFreq=args.sample_freq,
        **kwargs,
    )
    _print_network_dynamics(args)
    criterion = MaskedCrossEntropySpkReg(spike_regularization=args.spike_regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.model_checkpoint is not None:
        _load_checkpoint(
            checkpoint_path=args.model_checkpoint,
            model=model,
            device=device,
            class_to_idx=class_to_idx,
            dataset_variant=args.dataset_variant,
            args=args,
        )

    if args.dry_run:
        with torch.no_grad():
            preview_output = model(
                first_inputs.to(device, dtype=torch.float32),
                valid_mask=first_mask.to(device, dtype=torch.bool),
            )
        dry_run_metrics = run_epoch(
            model,
            [(first_inputs, first_labels, first_mask)],
            criterion,
            optimizer,
            split="train",
            device=device,
            max_batches=1,
            expected_num_classes=num_outputs,
        )
        _print_dry_run_report(
            args=args,
            dataset_root=dataset_root,
            inputs=first_inputs,
            labels=first_labels,
            valid_mask=first_mask,
            output_shape=tuple(preview_output.shape),
            num_classes=num_outputs,
            metrics=dry_run_metrics,
        )
        return {
            "mode": "dry_run",
            "dataset_root": dataset_root,
            "segmentation_root": segmentation_root,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "metrics": dry_run_metrics,
        }

    run = _start_wandb_run(args) if args.use_wandb else None
    best_metric = -math.inf
    best_epoch = -1
    best_state: Mapping[str, torch.Tensor] | None = None
    try:
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                split="train",
                device=device,
                max_batches=args.max_train_batches,
                expected_num_classes=num_outputs,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                split="val",
                device=device,
                expected_num_classes=num_outputs,
            )
            _print_epoch_metrics("train", train_metrics)
            _print_epoch_metrics("val", val_metrics)
            if run is not None:
                run.log(
                    {
                        **{f"train/{name}": value for name, value in train_metrics.items()},
                        **{f"val/{name}": value for name, value in val_metrics.items()},
                        "epoch": epoch,
                    }
                )

            validation_metric = val_metrics["balanced_accuracy"]
            if validation_metric > best_metric:
                best_metric = validation_metric
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                if args.save_model:
                    checkpoint_path = _save_checkpoint(
                        model_dir=args.model_dir,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val_metric=best_metric,
                        args=args,
                        class_to_idx=class_to_idx,
                        path_name="best_action0_synnet.pt",
                    )
                    print(f"Saved best checkpoint: {checkpoint_path}")

        if best_state is None:
            raise RuntimeError("training completed without a validation result")
        model.load_state_dict(best_state)
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            split="test",
            device=device,
            expected_num_classes=num_outputs,
        )
        _print_epoch_metrics("test", test_metrics)
        if run is not None:
            run.log({f"test/{name}": value for name, value in test_metrics.items()})
        return {
            "mode": "train",
            "dataset_root": dataset_root,
            "segmentation_root": segmentation_root,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "best_epoch": best_epoch,
            "best_val_metric": best_metric,
            "test_metrics": test_metrics,
        }
    finally:
        if run is not None:
            run.finish()


def _validate_args(args: Any) -> None:
    if args.network_type != "SynNet":
        raise NotImplementedError("Action0 baseline currently supports SynNet only")
    if args.sample_freq <= 0 or not math.isfinite(args.sample_freq):
        raise ValueError("sample_freq must be a positive finite value")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning_rate must be a positive finite value")
    if args.spike_regularization < 0 or not math.isfinite(args.spike_regularization):
        raise ValueError("spike_regularization must be a finite non-negative value")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("max_train_batches must be positive when provided")
    if args.shift_syn <= 0 or args.shift_mem <= 0:
        raise ValueError("shift_syn and shift_mem must be positive")
    for layer_index, neuron_count in enumerate(args.neurons_network, start=1):
        group_count = 2**layer_index
        if neuron_count <= 0 or neuron_count % group_count:
            raise ValueError(
                f"neurons_network layer {layer_index} must be positive and divisible by {group_count} "
                "to preserve SynNet's heterogeneous alpha layout"
            )
    _validate_user_splits(args.train_users, args.val_users, args.test_users)


def _validate_sampling_rate(
    *, cli_sampling_rate: float, producer_sampling_rate: float
) -> None:
    """Require CLI and producer rates to agree without changing SynNet dynamics."""

    if not math.isclose(
        cli_sampling_rate,
        producer_sampling_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise Action0DatasetError(
            "CLI sample_freq does not match producer sampling_rate_hz: "
            f"{cli_sampling_rate!r} != {producer_sampling_rate!r} "
            "(absolute tolerance 1e-12)"
        )


def _validate_split_padded_lengths(
    producer_metadata: PaddingDatasetMetadata,
    *,
    train_dataset: Action0SegmentDataset,
    val_dataset: Action0SegmentDataset,
    test_dataset: Action0SegmentDataset,
) -> None:
    """Require all selected splits to use the producer's one padded length."""

    lengths = {
        "train": train_dataset.padded_length,
        "validation": val_dataset.padded_length,
        "test": test_dataset.padded_length,
    }
    mismatches = {
        name: length
        for name, length in lengths.items()
        if length != producer_metadata.target_length
    }
    if mismatches:
        raise Action0DatasetError(
            "selected split padded lengths must equal producer target_length "
            f"{producer_metadata.target_length}; got {lengths}"
        )
    if len(set(lengths.values())) != 1:
        raise Action0DatasetError(
            "train, validation, and test padded lengths must match; "
            f"got {lengths}"
        )


def _validate_user_splits(
    train_users: Sequence[str], val_users: Sequence[str], test_users: Sequence[str]
) -> None:
    splits = {
        "train_users": set(train_users),
        "val_users": set(val_users),
        "test_users": set(test_users),
    }
    for name, users in splits.items():
        if not users:
            raise ValueError(f"{name} must contain at least one user")
    names = tuple(splits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = splits[left_name] & splits[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} must be disjoint; overlap={sorted(overlap)}"
                )


def _set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if use_gpu:
        print("CUDA was requested but is unavailable; using CPU.")
    return torch.device("cpu")


def _create_dataloaders(
    *,
    train_dataset: Action0SegmentDataset,
    val_dataset: Action0SegmentDataset,
    test_dataset: Action0SegmentDataset,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    device: torch.device,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    generator = torch.Generator().manual_seed(random_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def _inspect_first_batch(
    inputs: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor
) -> None:
    if inputs.ndim != 3 or inputs.shape[2] != INPUT_CHANNEL_COUNT:
        raise AssertionError(
            "Action0 inputs must have shape (batch, padded_time, 15); "
            f"got {tuple(inputs.shape)}"
        )
    if labels.shape != (inputs.shape[0],):
        raise AssertionError(f"labels must have shape {(inputs.shape[0],)}, got {tuple(labels.shape)}")
    if valid_mask.shape != inputs.shape[:2]:
        raise AssertionError(
            "valid_mask must have shape (batch, padded_time); "
            f"got {tuple(valid_mask.shape)}"
        )
    print(f"First inputs shape: {tuple(inputs.shape)}")
    print(f"First labels shape: {tuple(labels.shape)}")
    print(f"First valid-mask shape: {tuple(valid_mask.shape)}")
    print(f"First-batch valid fraction: {float(valid_mask.float().mean()):.6f}")


def _print_producer_metadata(metadata: PaddingDatasetMetadata) -> None:
    """Print the producer fields that define this Action0 training input."""

    print("Producer:")
    print(f"  input_kind: {metadata.input_kind}")
    print(f"  feature_schema: {metadata.feature_schema}")
    print(f"  channel_count: {metadata.channel_count}")
    print(f"  target_length: {metadata.target_length}")
    print(f"  sampling_rate_hz: {metadata.sampling_rate_hz:g}")
    print(f"  padding_side: {metadata.padding_side}")
    print(f"Trainer input channel slice: 0:{INPUT_CHANNEL_COUNT}")


def _print_dataset_statistics(
    *,
    class_to_idx: Mapping[str, int],
    idx_to_class: Mapping[int, str],
    train_dataset: Action0SegmentDataset,
    val_dataset: Action0SegmentDataset,
    test_dataset: Action0SegmentDataset,
) -> None:
    print(f"class_to_idx: {dict(class_to_idx)}")
    print(f"idx_to_class: {dict(idx_to_class)}")
    for name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        print(
            f"{name} segments: {len(dataset)}; padded length: {dataset.padded_length}; "
            f"valid fraction: {dataset.valid_fraction:.6f}"
        )
        print(f"{name} class distribution: {dataset.class_distribution}")


def _print_network_dynamics(args: Any) -> None:
    shifts = list(range(args.shift_syn, args.shift_syn + 8))
    beta = 1 - 2 ** (-args.shift_mem)
    tau_syn = [-(1 / args.sample_freq) / math.log(1 - 2 ** (-shift)) for shift in shifts]
    tau_mem = -(1 / args.sample_freq) / math.log(1 - 2 ** (-args.shift_mem))
    print("Network: SynNet")
    print(f"Input channels: {INPUT_CHANNEL_COUNT}")
    print(f"Hidden sizes: {args.neurons_network}")
    print(f"Sample frequency: {args.sample_freq:g} Hz")
    print(f"shift_syn: {args.shift_syn}")
    print(f"hidden synaptic shifts: layer 1: {shifts[0]}-{shifts[1]}")
    print(f"hidden synaptic shifts: layer 2: {shifts[0]}-{shifts[3]}")
    print(f"hidden synaptic shifts: layer 3: {shifts[0]}-{shifts[7]}")
    print(f"shift_mem: {args.shift_mem}")
    print(f"beta: {beta:g}")
    print(
        "tau diagnostics only; alpha/beta are unchanged: "
        f"synaptic={tau_syn}, membrane={tau_mem}"
    )


def _print_dry_run_report(
    *,
    args: Any,
    dataset_root: Path,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    output_shape: tuple[int, ...],
    num_classes: int,
    metrics: Mapping[str, float],
) -> None:
    print("Dry run completed.")
    print(f"Dataset variant: {args.dataset_variant}")
    print(f"Dataset root: {dataset_root}")
    print(f"Inputs shape: {tuple(inputs.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Mask shape: {tuple(valid_mask.shape)}")
    print(f"Valid fraction: {float(valid_mask.float().mean()):.6f}")
    print(f"Number of classes: {num_classes}")
    print(f"shift_syn: {args.shift_syn}")
    print(f"shift_mem: {args.shift_mem}")
    print(f"beta: {1 - 2 ** (-args.shift_mem):g}")
    print(f"Output shape: {output_shape}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Total valid spikes: {metrics['total_valid_spikes']:.0f}")
    print(f"Output valid spikes: {metrics['total_output_spikes']:.0f}")
    print(f"Zero-output-spike fraction: {metrics['zero_output_spike_fraction']:.6f}")


def _print_epoch_metrics(name: str, metrics: Mapping[str, float]) -> None:
    report = ", ".join(
        f"{key}={value:.6f}"
        for key, value in metrics.items()
        if not key.startswith("total_")
    )
    print(f"{name}: {report}")


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    args: Any,
    class_to_idx: Mapping[str, int],
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    best_val_metric: float | None = None,
) -> dict[str, object]:
    """Build a schema-v1 model-only checkpoint for a new training run.

    The optional legacy arguments remain accepted by this private helper so
    existing in-repository callers do not break, but they intentionally do
    not enter the payload.  ``--model_checkpoint`` is a configuration-
    compatible model restore, not an optimizer or trajectory resume.
    """

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "dataset_variant": args.dataset_variant,
        "boundary": BOUNDARY,
        "class_to_idx": dict(class_to_idx),
        "input_channels": [0, INPUT_CHANNEL_COUNT],
        "num_inputs": INPUT_CHANNEL_COUNT,
        "num_outputs": len(class_to_idx),
        "hidden_sizes": list(args.neurons_network),
        "shift_syn": args.shift_syn,
        "shift_mem": args.shift_mem,
        "sample_freq": args.sample_freq,
        "train_users": list(args.train_users),
        "val_users": list(args.val_users),
        "test_users": list(args.test_users),
    }


def _save_checkpoint(
    *,
    model_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_metric: float,
    args: Any,
    class_to_idx: Mapping[str, int],
    path_name: str,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / path_name
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_metric=best_val_metric,
            args=args,
            class_to_idx=class_to_idx,
        ),
        path,
    )
    return path


def _load_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    class_to_idx: Mapping[str, int],
    dataset_variant: str,
    args: Any,
) -> None:
    """Restore model weights after validating the schema-v1 configuration.

    Only model parameters are restored.  Optimizer, epoch, best-metric, RNG,
    and DataLoader state are intentionally not part of this new-run restore.
    """

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"model checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint checkpoint_schema_version mismatch: "
            f"checkpoint value={type(payload).__name__!r}; "
            f"requested value={CHECKPOINT_SCHEMA_VERSION!r}"
        )
    _validate_checkpoint_field(
        payload,
        "checkpoint_schema_version",
        CHECKPOINT_SCHEMA_VERSION,
    )

    expected_fields = {
        "dataset_variant": dataset_variant,
        "boundary": BOUNDARY,
        "class_to_idx": dict(class_to_idx),
        "input_channels": [0, INPUT_CHANNEL_COUNT],
        "num_inputs": INPUT_CHANNEL_COUNT,
        "num_outputs": len(class_to_idx),
        "hidden_sizes": list(args.neurons_network),
        "shift_syn": args.shift_syn,
        "shift_mem": args.shift_mem,
        "sample_freq": args.sample_freq,
        "train_users": list(args.train_users),
        "val_users": list(args.val_users),
        "test_users": list(args.test_users),
    }
    for field, requested_value in expected_fields.items():
        _validate_checkpoint_field(payload, field, requested_value)

    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError(
            "checkpoint model_state_dict must be a mapping; "
            f"checkpoint value={model_state!r}; requested value=mapping"
        )
    model.load_state_dict(model_state)
    print(f"Loaded checkpoint: {checkpoint_path}")


def _validate_checkpoint_field(
    payload: Mapping[str, object], field: str, requested_value: object
) -> None:
    """Reject a missing or mismatched checkpoint field with actionable context."""

    missing = object()
    checkpoint_value = payload.get(field, missing)
    if checkpoint_value is missing:
        checkpoint_value_repr = "<missing>"
        raise ValueError(
            f"checkpoint {field} mismatch: checkpoint value={checkpoint_value_repr}; "
            f"requested value={requested_value!r}"
        )
    if checkpoint_value != requested_value:
        raise ValueError(
            f"checkpoint {field} mismatch: checkpoint value={checkpoint_value!r}; "
            f"requested value={requested_value!r}"
        )


def _start_wandb_run(args: Any) -> Any:
    try:
        import wandb
    except ImportError as error:
        raise ImportError("--use_wandb requires the optional wandb package") from error
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(project="writingring-action0", config=config)


if __name__ == "__main__":
    main()
