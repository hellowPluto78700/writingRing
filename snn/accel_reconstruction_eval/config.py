from __future__ import annotations

"""Experiment configuration for acceleration reconstruction studies.

The configuration layer intentionally stays small: it describes dataset source,
user-level split, DataLoader settings, CNN training hyperparameters, and the
representation-evaluation protocol.  It does not load data or perform I/O.

Source conventions
------------------
raw:
    Use the three acceleration channels immediately before the trailing
    gyroscope channels in paddedSpikeIMU.
reconstruction:
    Use ``*_padded_reconstructed_accel_m_s2.npy``.
mixed:
    Present both raw and reconstruction as separate examples (D2 protocol).

Experiment presets match the A/B/C/D terminology used in the notebooks.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

try:
    from .datasets import normalize_label_list, normalize_user_name
    from .evaluation import RepresentationEvaluationConfig
except ImportError:  # direct-module use in notebooks/tests
    from datasets import normalize_label_list, normalize_user_name
    from evaluation import RepresentationEvaluationConfig


DatasetSource = Literal["raw", "reconstruction", "mixed"]
NormalizationSource = Literal["raw", "reconstruction", "mixed", "checkpoint"]
ProbeVariant = Literal["cnn_s", "cnn_m", "cnn_l"]

__all__ = [
    "DatasetSource",
    "NormalizationSource",
    "ProbeVariant",
    "UserSplitConfig",
    "DataLoaderConfig",
    "CNNTrainingConfig",
    "ExperimentConfig",
    "validate_reference_seed",
    "experiment_a_config",
    "experiment_b_config",
    "experiment_c_config",
    "experiment_d_config",
]


@dataclass(frozen=True)
class UserSplitConfig:
    """User-disjoint train/validation/test split configuration."""

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    explicit_train_users: tuple[str, ...] | None = None
    explicit_val_users: tuple[str, ...] | None = None
    explicit_test_users: tuple[str, ...] | None = None
    require_all_users_assigned: bool = True
    require_all_labels_in_all_splits: bool = True
    excluded_users: Sequence[str] = ()
    included_labels: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.excluded_users, (str, bytes, bytearray)) or not isinstance(
            self.excluded_users, Sequence
        ):
            raise TypeError(
                "excluded_users must be a non-string sequence of user names"
            )

        normalized = tuple(
            normalize_user_name(value) for value in self.excluded_users
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "excluded_users contains duplicate canonical user names: "
                f"{normalized}"
            )
        object.__setattr__(self, "excluded_users", normalized)

        if self.included_labels is None:
            return
        normalized_labels = normalize_label_list(self.included_labels)
        object.__setattr__(self, "included_labels", tuple(normalized_labels))

    def validate(self) -> None:
        if self.train_fraction <= 0 or self.val_fraction <= 0:
            raise ValueError("train_fraction and val_fraction must be positive")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must be < 1")

        values = (
            self.explicit_train_users,
            self.explicit_val_users,
            self.explicit_test_users,
        )
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError(
                "Set all three explicit user lists, or leave all three as None"
            )
        if all(value is not None for value in values):
            normalized = [tuple(value or ()) for value in values]
            if any(len(value) == 0 for value in normalized):
                raise ValueError("Explicit train/val/test user lists must be non-empty")


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool | None = None
    persistent_workers: bool | None = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.persistent_workers is True and self.num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0"
            )


@dataclass(frozen=True)
class CNNTrainingConfig:
    num_epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    use_class_weights: bool = False
    early_stopping_patience: int | None = 10
    max_train_batches: int | None = None
    grad_clip_norm: float | None = None

    def validate(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive or None")
        if self.max_train_batches is not None and self.max_train_batches <= 0:
            raise ValueError("max_train_batches must be positive or None")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete protocol description for one A/B/C/D-style experiment."""

    name: str
    train_source: DatasetSource
    val_source: DatasetSource
    test_source: DatasetSource
    normalization_source: NormalizationSource
    probe_variant: ProbeVariant = "cnn_l"

    training_enabled: bool = True
    random_seed: int = 12345
    use_gpu: bool = True
    normalization_chunk_segments: int = 256

    split: UserSplitConfig = field(default_factory=UserSplitConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: CNNTrainingConfig = field(default_factory=CNNTrainingConfig)
    evaluation: RepresentationEvaluationConfig = field(
        default_factory=RepresentationEvaluationConfig
    )

    baseline_checkpoint: Path | None = None
    output_dir: Path | None = None

    def validate(self) -> None:
        if self.probe_variant not in {"cnn_s", "cnn_m", "cnn_l"}:
            raise ValueError(f"Unknown probe_variant: {self.probe_variant!r}")
        if not self.name.strip():
            raise ValueError("Experiment name must be non-empty")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.evaluation.random_seed != self.random_seed:
            raise ValueError(
                "evaluation.random_seed must match random_seed; "
                f"got evaluation={self.evaluation.random_seed}, "
                f"experiment={self.random_seed}"
            )
        if self.normalization_chunk_segments <= 0:
            raise ValueError("normalization_chunk_segments must be positive")

        valid_sources = {"raw", "reconstruction", "mixed"}
        for field_name, value in (
            ("train_source", self.train_source),
            ("val_source", self.val_source),
            ("test_source", self.test_source),
        ):
            if value not in valid_sources:
                raise ValueError(f"Invalid {field_name}: {value!r}")

        if self.normalization_source not in {
            "raw",
            "reconstruction",
            "mixed",
            "checkpoint",
        }:
            raise ValueError(
                f"Invalid normalization_source: {self.normalization_source!r}"
            )
        if self.normalization_source == "checkpoint" and self.training_enabled:
            raise ValueError(
                "checkpoint normalization is intended for frozen-model evaluation"
            )

        self.split.validate()
        self.loader.validate()
        self.training.validate()
        self.evaluation.validate()

    @property
    def requires_reconstruction(self) -> bool:
        return any(
            source in {"reconstruction", "mixed"}
            for source in (
                self.train_source,
                self.val_source,
                self.test_source,
                self.normalization_source,
            )
        )

    def with_output_dir(self, output_dir: str | Path) -> "ExperimentConfig":
        return replace(self, output_dir=Path(output_dir))

    def with_test_source(self, test_source: DatasetSource) -> "ExperimentConfig":
        """Convenience helper for D: evaluate the same model on both domains."""
        updated = replace(self, test_source=test_source)
        updated.validate()
        return updated

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.baseline_checkpoint is not None:
            value["baseline_checkpoint"] = str(self.baseline_checkpoint)
        if self.output_dir is not None:
            value["output_dir"] = str(self.output_dir)
        return value


def validate_reference_seed(
    config: ExperimentConfig,
    expected_seed: int,
    *,
    context: str = "reference checkpoint",
) -> None:
    """Require an experiment config to use a reference seed exactly."""

    if config.random_seed != expected_seed:
        raise ValueError(
            f"{context} random_seed mismatch: "
            f"config={config.random_seed}, expected={expected_seed}"
        )
    if config.evaluation.random_seed != expected_seed:
        raise ValueError(
            f"{context} evaluation.random_seed mismatch: "
            f"config={config.evaluation.random_seed}, expected={expected_seed}"
        )


def experiment_a_config(
    *,
    output_dir: str | Path | None = None,
    random_seed: int = 12345,
    probe_variant: ProbeVariant = "cnn_l",
) -> ExperimentConfig:
    """A: raw train -> raw validation -> raw test."""
    config = ExperimentConfig(
        name="A_raw",
        train_source="raw",
        val_source="raw",
        test_source="raw",
        normalization_source="raw",
        probe_variant=probe_variant,
        training_enabled=True,
        random_seed=random_seed,
        evaluation=replace(
            RepresentationEvaluationConfig(),
            random_seed=random_seed,
        ),
        output_dir=None if output_dir is None else Path(output_dir),
    )
    config.validate()
    return config


def experiment_b_config(
    *,
    baseline_checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    random_seed: int = 12345,
    probe_variant: ProbeVariant = "cnn_l",
) -> ExperimentConfig:
    """B: frozen raw-trained CNN; raw reference train/val; recon test query."""
    config = ExperimentConfig(
        name="B_frozen_raw_to_reconstruction",
        train_source="raw",
        val_source="raw",
        test_source="reconstruction",
        normalization_source="checkpoint",
        probe_variant=probe_variant,
        training_enabled=False,
        random_seed=random_seed,
        evaluation=replace(
            RepresentationEvaluationConfig(),
            random_seed=random_seed,
        ),
        baseline_checkpoint=(
            None if baseline_checkpoint is None else Path(baseline_checkpoint)
        ),
        output_dir=None if output_dir is None else Path(output_dir),
    )
    config.validate()
    return config


def experiment_c_config(
    *,
    output_dir: str | Path | None = None,
    random_seed: int = 12345,
    probe_variant: ProbeVariant = "cnn_l",
) -> ExperimentConfig:
    """C: reconstruction train -> reconstruction validation/test."""
    config = ExperimentConfig(
        name="C_reconstruction",
        train_source="reconstruction",
        val_source="reconstruction",
        test_source="reconstruction",
        normalization_source="reconstruction",
        probe_variant=probe_variant,
        training_enabled=True,
        random_seed=random_seed,
        evaluation=replace(
            RepresentationEvaluationConfig(),
            random_seed=random_seed,
        ),
        output_dir=None if output_dir is None else Path(output_dir),
    )
    config.validate()
    return config


def experiment_d_config(
    *,
    test_source: Literal["raw", "reconstruction"] = "raw",
    output_dir: str | Path | None = None,
    random_seed: int = 12345,
    probe_variant: ProbeVariant = "cnn_l",
) -> ExperimentConfig:
    """D2: raw+reconstruction duplicated training/validation.

    Run the same trained model twice, once with ``test_source='raw'`` and once
    with ``test_source='reconstruction'``, to measure domain invariance.
    """
    config = ExperimentConfig(
        name=f"D_mixed_to_{test_source}",
        train_source="mixed",
        val_source="mixed",
        test_source=test_source,
        normalization_source="mixed",
        probe_variant=probe_variant,
        training_enabled=True,
        random_seed=random_seed,
        evaluation=replace(
            RepresentationEvaluationConfig(),
            random_seed=random_seed,
        ),
        output_dir=None if output_dir is None else Path(output_dir),
    )
    config.validate()
    return config
