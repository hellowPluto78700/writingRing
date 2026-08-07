"""Command-line parsing for the independent Action0 SynNet baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .action0_dataset import DATASET_VARIANT_DIRS


def build_parser() -> argparse.ArgumentParser:
    """Create the Action0-only parser without legacy HAR data options."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train local SynNet on padded Action0 label segments.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--pipeline_root",
        type=Path,
        default=Path("outputs/action0_pipeline"),
        help="Root containing variant directories produced by the Action0 pipeline",
    )
    dataset_group.add_argument(
        "--dataset_variant",
        choices=tuple(DATASET_VARIANT_DIRS),
        required=True,
        help="Action0 producer variant",
    )
    dataset_group.add_argument(
        "--train_users",
        nargs="+",
        required=True,
        help="Users reserved for training",
    )
    dataset_group.add_argument(
        "--val_users",
        nargs="+",
        required=True,
        help="Users reserved for validation",
    )
    dataset_group.add_argument(
        "--test_users",
        nargs="+",
        required=True,
        help="Users reserved for final testing",
    )
    dataset_group.add_argument(
        "--sample_freq",
        type=float,
        required=True,
        help="Producer sampling frequency in Hz; metadata only for SynNet dynamics diagnostics",
    )
    dataset_group.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")

    architecture_group = parser.add_argument_group("architecture")
    architecture_group.add_argument(
        "--neurons_network",
        nargs=3,
        type=int,
        default=[24, 24, 24],
        help="Hidden neurons in the three SynNet layers",
    )
    architecture_group.add_argument(
        "--network_type",
        type=str,
        default="SynNet",
        help="Network architecture; this baseline currently supports SynNet only",
    )
    architecture_group.add_argument(
        "--shift_syn",
        type=int,
        default=2,
        help="Shift for I[t+Dt] = I[t] - (I[t] >> a)",
    )
    architecture_group.add_argument(
        "--shift_mem",
        type=int,
        default=1,
        help="Shift for U[t+Dt] = U[t] - (U[t] >> b)",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--batch_size", type=int, default=32, help="Batch size")
    training_group.add_argument("--num_epochs", type=int, default=20, help="Training epochs")
    training_group.add_argument("--learning_rate", type=float, default=5e-4, help="Adam learning rate")
    training_group.add_argument(
        "--spike_regularization",
        type=float,
        default=0.0,
        help="Coefficient for valid-neuron-time spike regularization",
    )

    reproducibility_group = parser.add_argument_group("reproducibility")
    reproducibility_group.add_argument("--random_seed", type=int, default=12345, help="Random seed")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--use_gpu", action="store_true", help="Use CUDA when it is available")
    runtime_group.add_argument("--use_wandb", action="store_true", help="Enable optional Weights & Biases logging")

    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument(
        "--model_checkpoint",
        type=Path,
        default=None,
        help="Optional full Action0 checkpoint to restore before training",
    )
    checkpoint_group.add_argument("--save_model", action="store_true", help="Save the best validation checkpoint")
    checkpoint_group.add_argument(
        "--model_dir",
        type=Path,
        default=Path("snn/models/action0"),
        help="Directory for Action0 checkpoints",
    )

    debug_group = parser.add_argument_group("debug")
    debug_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Run one data-to-backpropagation training step and exit",
    )
    debug_group.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Limit train batches per epoch for debugging",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an optional argument sequence for programmatic testability."""

    return build_parser().parse_args(argv)
