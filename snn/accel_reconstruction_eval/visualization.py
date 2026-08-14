from __future__ import annotations

"""Reusable visualization helpers for acceleration reconstruction Experiments A/B/C/D.

This module is intentionally presentation-only. It reads the standardized CSV/NPZ
artifacts already produced by ``snn.accel_reconstruction_eval`` and does not
recompute the experiment's reported metrics.

Expected current artifact layouts
---------------------------------

Experiment A
    <A_DIR>/
        summary.csv
        classification_splits.csv
        training_history.csv
        embeddings_test.npz
        knn_validation.csv
        same_label_at_k.csv
        intra_by_class.csv
        linear_probe_history.csv
        evaluation_arrays.npz
        provenance.json

Experiment B
    <B_DIR>/
        summary.csv                       # reconstruction-test result
        classification_splits.csv
        embeddings_test_raw.npz
        embeddings_test_reconstruction.npz
        raw_reference/
            summary.csv
            ...
        paired_preservation/
            paired_summary.csv
            paired_transitions.csv
            paired_per_class.csv
            paired_arrays.npz
        domain_comparison.csv
        provenance.json

Experiment C
    <C_DIR>/                              # same standard evaluation layout as A
        summary.csv
        classification_splits.csv
        training_history.csv
        embeddings_test.npz
        ...
        provenance.json

Experiment D
    <D_DIR>/
        classification_splits.csv
        training_history.csv
        embeddings_test_raw.npz
        embeddings_test_reconstruction.npz
        raw_test/
            summary.csv
            ...
        reconstruction_test/
            summary.csv
            ...
        paired_preservation/
            paired_summary.csv
            paired_transitions.csv
            paired_per_class.csv
            paired_arrays.npz
        domain_comparison.csv
        provenance.json

Typical notebook usage
----------------------

Experiment A::

    from snn.accel_reconstruction_eval.visualization import visualize_experiment_a
    figures = visualize_experiment_a(run.output_dir)

Experiment B::

    from snn.accel_reconstruction_eval.visualization import visualize_experiment_b
    figures = visualize_experiment_b(run.output_dir)

Experiment C::

    from snn.accel_reconstruction_eval.visualization import visualize_experiment_c
    figures = visualize_experiment_c(run.output_dir)

Experiment D::

    from snn.accel_reconstruction_eval.visualization import visualize_experiment_d
    figures = visualize_experiment_d(run.output_dir)

All high-level helpers return ``dict[str, Path]`` containing the generated PNGs.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = [
    "VisualizationConfig",
    "load_summary",
    "load_class_to_idx",
    "load_embedding_bundle",
    "plot_training_loss",
    "plot_training_balanced_accuracy",
    "plot_classification_splits",
    "plot_knn_validation",
    "plot_same_label_at_k",
    "plot_representation_dashboard",
    "plot_embedding_geometry",
    "plot_unsupervised_metrics",
    "plot_linear_probe_history",
    "plot_confusion_matrix",
    "plot_embedding_pca",
    "plot_per_class_compactness",
    "plot_average_precision_distribution",
    "plot_silhouette_distribution",
    "plot_domain_readout_comparison",
    "plot_domain_geometry_comparison",
    "plot_paired_summary",
    "plot_paired_transitions",
    "plot_paired_per_class_similarity",
    "plot_paired_per_class_agreement",
    "plot_paired_cosine_distribution",
    "visualize_standard_evaluation",
    "visualize_experiment_a",
    "visualize_experiment_b",
    "visualize_experiment_c",
    "visualize_experiment_d",
]


READOUT_METRICS: tuple[tuple[str, str], ...] = (
    ("CNN BA", "CNN_test_balanced_accuracy"),
    ("CNN Macro-F1", "CNN_test_macro_f1"),
    ("kNN BA", "kNN_test_balanced_accuracy"),
    ("Prototype BA", "prototype_test_balanced_accuracy"),
    ("Linear probe BA", "linear_probe_test_balanced_accuracy"),
    ("Retrieval macro mAP", "retrieval_macro_mAP"),
    ("SameLabel@1 macro", "SameLabel_macro_at_1"),
)

GEOMETRY_METRICS: tuple[tuple[str, str], ...] = (
    ("D_intra", "D_intra_macro"),
    ("D_inter", "D_inter_centroids"),
    ("D_inter / D_intra", "D_inter_over_D_intra"),
)

UNSUPERVISED_METRICS: tuple[tuple[str, str], ...] = (
    ("NMI", "NMI_arithmetic"),
    ("ARI", "ARI"),
    ("Silhouette macro", "silhouette_macro"),
    ("Silhouette micro", "silhouette_micro"),
)

DOMAIN_READOUT_METRICS: tuple[str, ...] = (
    "CNN_test_balanced_accuracy",
    "kNN_test_balanced_accuracy",
    "prototype_test_balanced_accuracy",
    "linear_probe_test_balanced_accuracy",
    "retrieval_macro_mAP",
    "SameLabel_macro_at_1",
)

DOMAIN_GEOMETRY_METRICS: tuple[str, ...] = (
    "D_intra_macro",
    "D_inter_centroids",
    "D_inter_over_D_intra",
    "silhouette_macro",
    "NMI_arithmetic",
    "ARI",
)


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration shared by the notebook visualization helpers."""

    dpi: int = 180
    pca_max_points: int = 5000
    random_seed: int = 12345
    show: bool = True
    close_after_save: bool = False
    annotate_bars: bool = True

    def validate(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.pca_max_points <= 0:
            raise ValueError("pca_max_points must be positive")


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _ensure_file(path: str | Path, *, description: str | None = None) -> Path:
    resolved = _as_path(path)
    if not resolved.is_file():
        label = description or "Required artifact"
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _ensure_dir(path: str | Path) -> Path:
    resolved = _as_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _read_indexed_csv(path: str | Path) -> pd.DataFrame:
    path = _ensure_file(path)
    frame = pd.read_csv(path, index_col=0)
    return frame


def _read_regular_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(_ensure_file(path))


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _require_metrics(
    summary: pd.Series,
    metrics: Sequence[str],
    *,
    name: str,
) -> None:
    missing = [metric for metric in metrics if metric not in summary.index]
    if missing:
        raise KeyError(f"{name} is missing required metrics: {missing}")


def _title(prefix: str | None, suffix: str) -> str:
    return suffix if not prefix else f"{prefix}: {suffix}"


def _save_figure(
    fig: plt.Figure,
    output_dir: str | Path | None,
    filename: str | None,
    *,
    config: VisualizationConfig,
) -> Path | None:
    saved: Path | None = None
    if output_dir is not None and filename is not None:
        directory = _ensure_dir(output_dir)
        saved = directory / filename
        fig.savefig(saved, dpi=config.dpi, bbox_inches="tight")

    if config.show:
        plt.show()
    if config.close_after_save:
        plt.close(fig)

    return saved


def _annotate_bars(ax: plt.Axes, *, config: VisualizationConfig) -> None:
    if not config.annotate_bars:
        return
    for patch in ax.patches:
        height = float(patch.get_height())
        if np.isfinite(height):
            ax.annotate(
                f"{height:.3f}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def load_summary(path_or_directory: str | Path) -> pd.Series:
    """Load a standardized one-row ``summary.csv`` as a Series."""
    path = _as_path(path_or_directory)
    if path.is_dir():
        path = path / "summary.csv"
    frame = _read_regular_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(frame)}")
    return frame.iloc[0].copy()


def load_class_to_idx(provenance_path: str | Path) -> dict[str, int]:
    """Read ``class_to_idx`` from an experiment provenance JSON."""
    path = _ensure_file(provenance_path, description="Provenance JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "class_to_idx" not in payload:
        raise KeyError(f"{path} does not contain class_to_idx")
    mapping = {str(label): int(index) for label, index in payload["class_to_idx"].items()}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class_to_idx values must be contiguous 0..C-1")
    return mapping


def load_embedding_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved embedding NPZ produced by ``save_embedding_bundle``."""
    path = _ensure_file(path, description="Embedding bundle")
    with np.load(path, allow_pickle=False) as values:
        required = {"h", "z", "y", "cnn_pred", "logits", "sample_id"}
        missing = sorted(required.difference(values.files))
        if missing:
            raise KeyError(f"{path} is missing embedding arrays: {missing}")
        return {
            "h": values["h"],
            "z": values["z"],
            "y": values["y"].astype(np.int64, copy=False),
            "cnn_pred": values["cnn_pred"].astype(np.int64, copy=False),
            "logits": values["logits"],
            "sample_id": values["sample_id"].astype(str),
        }


def _best_epoch_from_provenance(path: str | Path | None) -> tuple[int | None, float | None]:
    if path is None:
        return None, None
    provenance_path = _as_path(path)
    if not provenance_path.is_file():
        return None, None
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    epoch = payload.get("best_epoch")
    score = payload.get("best_val_balanced_accuracy")
    return (
        None if epoch is None else int(epoch),
        None if score is None else float(score),
    )


def plot_training_loss(
    history: pd.DataFrame,
    *,
    title: str | None = None,
    best_epoch: int | None = None,
    output_dir: str | Path | None = None,
    filename: str = "training_loss.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot train/validation cross-entropy loss by epoch."""
    config.validate()
    _require_columns(history, ["epoch", "split", "loss"], name="training history")
    table = history.pivot(index="epoch", columns="split", values="loss")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for split_name in table.columns:
        ax.plot(
            table.index,
            table[split_name],
            marker="o",
            markersize=3,
            label=str(split_name),
        )
    if best_epoch is not None:
        ax.axvline(
            int(best_epoch),
            linestyle="--",
            linewidth=1.3,
            label=f"best epoch = {int(best_epoch)}",
        )
    ax.set_title(_title(title, "training loss"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_training_balanced_accuracy(
    history: pd.DataFrame,
    *,
    title: str | None = None,
    best_epoch: int | None = None,
    best_val_balanced_accuracy: float | None = None,
    output_dir: str | Path | None = None,
    filename: str = "training_balanced_accuracy.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot train/validation balanced accuracy by epoch."""
    config.validate()
    _require_columns(
        history,
        ["epoch", "split", "balanced_accuracy"],
        name="training history",
    )
    table = history.pivot(
        index="epoch",
        columns="split",
        values="balanced_accuracy",
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for split_name in table.columns:
        ax.plot(
            table.index,
            table[split_name],
            marker="o",
            markersize=3,
            label=str(split_name),
        )
    if best_epoch is not None:
        ax.axvline(
            int(best_epoch),
            linestyle="--",
            linewidth=1.3,
            label=f"best epoch = {int(best_epoch)}",
        )
    if best_val_balanced_accuracy is not None:
        ax.axhline(
            float(best_val_balanced_accuracy),
            linestyle=":",
            linewidth=1.2,
            label=f"best val BA = {float(best_val_balanced_accuracy):.3f}",
        )
    ax.set_title(_title(title, "training balanced accuracy"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_classification_splits(
    classification_splits: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "classification_splits.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot accuracy, balanced accuracy and macro-F1 across named splits."""
    config.validate()
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    _require_columns(
        classification_splits,
        metrics,
        name="classification_splits",
    )
    table = classification_splits.loc[:, metrics].copy()

    fig, ax = plt.subplots(figsize=(10, 5.2))
    table.plot(kind="bar", ax=ax)
    ax.set_title(_title(title, "classification across splits/domains"))
    ax.set_xlabel("Split / domain")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric")
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_knn_validation(
    table: pd.DataFrame,
    *,
    selected_k: int | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "knn_validation_selection.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot validation balanced accuracy used to select kNN K."""
    config.validate()
    _require_columns(table, ["k", "balanced_accuracy"], name="kNN validation table")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        table["k"],
        table["balanced_accuracy"],
        marker="o",
        label="validation balanced accuracy",
    )
    if selected_k is not None:
        selected = table.loc[table["k"] == int(selected_k)]
        if len(selected) == 1:
            score = float(selected.iloc[0]["balanced_accuracy"])
            ax.scatter(
                [int(selected_k)],
                [score],
                s=90,
                label=f"selected K = {int(selected_k)}",
            )
    ax.set_title(_title(title, "kNN K selection"))
    ax.set_xlabel("K")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_same_label_at_k(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "same_label_at_k.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot micro/macro SameLabel@K neighborhood purity."""
    config.validate()
    _require_columns(
        table,
        ["k", "same_label_micro", "same_label_macro"],
        name="SameLabel@K table",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(table["k"], table["same_label_micro"], marker="o", label="micro")
    ax.plot(table["k"], table["same_label_macro"], marker="o", label="macro")
    ax.set_title(_title(title, "SameLabel@K neighborhood purity"))
    ax.set_xlabel("K")
    ax.set_ylabel("Same-label fraction")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_representation_dashboard(
    summary: pd.Series | Mapping[str, float],
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "representation_readout_dashboard.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot the main supervised/retrieval representation scores on a 0-1 scale."""
    config.validate()
    series = pd.Series(summary)
    keys = [key for _, key in READOUT_METRICS]
    _require_metrics(series, keys, name="representation summary")

    scores = pd.Series(
        {label: float(series[key]) for label, key in READOUT_METRICS},
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(10, 5.2))
    scores.plot(kind="bar", ax=ax)
    ax.set_title(_title(title, "task and representation readouts"))
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, config=config)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_embedding_geometry(
    summary: pd.Series | Mapping[str, float],
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "embedding_geometry.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot D_intra, D_inter and D_inter/D_intra."""
    config.validate()
    series = pd.Series(summary)
    keys = [key for _, key in GEOMETRY_METRICS]
    _require_metrics(series, keys, name="representation summary")

    values = pd.Series(
        {label: float(series[key]) for label, key in GEOMETRY_METRICS},
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    values.plot(kind="bar", ax=ax)
    ax.set_title(_title(title, "embedding geometry"))
    ax.set_ylabel("Cosine-distance based value")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, config=config)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_unsupervised_metrics(
    summary: pd.Series | Mapping[str, float],
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "unsupervised_embedding_quality.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot NMI, ARI and cosine silhouette diagnostics."""
    config.validate()
    series = pd.Series(summary)
    keys = [key for _, key in UNSUPERVISED_METRICS]
    _require_metrics(series, keys, name="representation summary")

    values = pd.Series(
        {label: float(series[key]) for label, key in UNSUPERVISED_METRICS},
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    values.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(_title(title, "unsupervised embedding quality"))
    ax.set_ylabel("Score")
    lower = min(-0.1, float(values.min()) - 0.05)
    ax.set_ylim(lower, 1.02)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, config=config)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_linear_probe_history(
    history: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "linear_probe_validation.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot validation balanced accuracy during linear-probe fitting."""
    config.validate()
    _require_columns(
        history,
        ["epoch", "val_balanced_accuracy"],
        name="linear probe history",
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        history["epoch"],
        history["val_balanced_accuracy"],
        marker="o",
        markersize=3,
    )
    best_index = int(history["val_balanced_accuracy"].astype(float).idxmax())
    best_epoch = int(history.loc[best_index, "epoch"])
    best_score = float(history.loc[best_index, "val_balanced_accuracy"])
    ax.scatter(
        [best_epoch],
        [best_score],
        s=90,
        label=f"best epoch = {best_epoch}, BA = {best_score:.3f}",
    )
    ax.set_title(_title(title, "linear-probe validation"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def _class_mapping(
    labels: np.ndarray,
    class_to_idx: Mapping[str, int] | None,
) -> tuple[int, dict[int, str]]:
    labels = np.asarray(labels, dtype=np.int64)
    if class_to_idx is not None:
        normalized = {str(label): int(index) for label, index in class_to_idx.items()}
        if sorted(normalized.values()) != list(range(len(normalized))):
            raise ValueError("class_to_idx values must be contiguous 0..C-1")
        return len(normalized), {index: label for label, index in normalized.items()}

    if len(labels) == 0:
        raise ValueError("Cannot infer classes from an empty label vector")
    num_classes = int(labels.max()) + 1
    return num_classes, {index: str(index) for index in range(num_classes)}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_to_idx: Mapping[str, int] | None = None,
    normalize: bool = True,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "test_confusion_matrix_normalized.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot held-out-test confusion matrix, row-normalized by default."""
    config.validate()
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be aligned 1-D arrays")

    num_classes, idx_to_class = _class_mapping(y_true, class_to_idx)
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("y_true contains labels outside class_to_idx")
    if np.any(y_pred < 0) or np.any(y_pred >= num_classes):
        raise ValueError("y_pred contains labels outside class_to_idx")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)

    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        shown = np.divide(
            matrix,
            totals,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=totals > 0,
        )
        vmax = 1.0
        colorbar_label = "Row-normalized fraction"
    else:
        shown = matrix.astype(np.float64)
        vmax = None
        colorbar_label = "Count"

    tick_labels = [idx_to_class[index] for index in range(num_classes)]
    side = max(7.0, min(18.0, 0.55 * num_classes + 4.0))

    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(
        shown,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    ax.set_title(_title(title, "held-out-test confusion matrix"))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    if num_classes <= 20:
        for i in range(num_classes):
            for j in range(num_classes):
                value = shown[i, j]
                if normalize:
                    if value >= 0.01:
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
                elif value > 0:
                    ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def _balanced_subsample_indices(
    labels: np.ndarray,
    max_points: int,
    *,
    random_seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) <= max_points:
        return np.arange(len(labels), dtype=np.int64)

    rng = np.random.default_rng(int(random_seed))
    classes = np.unique(labels)
    per_class = max(1, max_points // len(classes))
    selected: list[int] = []
    leftovers: list[int] = []

    for class_idx in classes:
        class_indices = np.flatnonzero(labels == class_idx)
        rng.shuffle(class_indices)
        selected.extend(class_indices[:per_class].tolist())
        leftovers.extend(class_indices[per_class:].tolist())

    remaining = max_points - len(selected)
    if remaining > 0 and leftovers:
        rest = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(rest)
        selected.extend(rest[:remaining].tolist())

    return np.asarray(sorted(selected[:max_points]), dtype=np.int64)


def plot_embedding_pca(
    z: np.ndarray,
    y: np.ndarray,
    *,
    class_to_idx: Mapping[str, int] | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "test_embedding_pca.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot a balanced-subsample PCA view of the held-out-test embedding."""
    config.validate()
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if z.ndim != 2:
        raise ValueError("z must be a 2-D embedding matrix")
    if y.shape != (len(z),):
        raise ValueError("y must align with z")
    if len(z) < 2 or z.shape[1] < 2:
        raise ValueError("PCA visualization requires at least 2 samples and 2 dimensions")

    _, idx_to_class = _class_mapping(y, class_to_idx)
    selected_indices = _balanced_subsample_indices(
        y,
        config.pca_max_points,
        random_seed=config.random_seed,
    )
    pca_x = z[selected_indices]
    pca_y = y[selected_indices]

    centered = pca_x - pca_x.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T

    variance = singular_values**2
    total_variance = float(variance.sum())
    if total_variance > 0:
        explained = variance[:2] / total_variance
    else:
        explained = np.zeros(2, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 7))
    for class_idx in np.unique(pca_y):
        mask = pca_y == class_idx
        label_name = idx_to_class.get(int(class_idx), str(class_idx))
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=13,
            alpha=0.55,
            label=label_name,
        )
        centroid = coords[mask].mean(axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            label_name,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        _title(
            title,
            "held-out-test embedding PCA "
            f"(PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%})",
        )
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    if len(np.unique(pca_y)) <= 12:
        ax.legend(title="Label", bbox_to_anchor=(1.02, 1.0), loc="upper left")

    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_per_class_compactness(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "per_class_intra_distance.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot per-class intra-class cosine distance; lower is tighter."""
    config.validate()
    _require_columns(
        table,
        ["label", "intra_cosine_distance"],
        name="intra_by_class",
    )
    ordered = table.sort_values(
        "intra_cosine_distance",
        ascending=False,
    ).reset_index(drop=True)

    height = max(4.5, min(14.0, 0.35 * len(ordered) + 2.0))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(
        ordered["label"].astype(str),
        ordered["intra_cosine_distance"].astype(float),
    )
    ax.invert_yaxis()
    ax.set_title(_title(title, "per-class embedding compactness"))
    ax.set_xlabel("Intra-class cosine distance (lower is tighter)")
    ax.set_ylabel("Label")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_average_precision_distribution(
    average_precision: np.ndarray,
    *,
    macro_map: float | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "average_precision_distribution.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot the distribution of per-query retrieval average precision."""
    config.validate()
    values = np.asarray(average_precision, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("average_precision must be a non-empty 1-D array")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(values, bins=30)
    if macro_map is not None:
        ax.axvline(
            float(macro_map),
            linestyle="--",
            linewidth=1.4,
            label=f"macro mAP = {float(macro_map):.3f}",
        )
        ax.legend()
    ax.set_title(_title(title, "per-query average precision distribution"))
    ax.set_xlabel("Average precision")
    ax.set_ylabel("Test samples")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_silhouette_distribution(
    silhouette_values: np.ndarray,
    *,
    macro_silhouette: float | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "silhouette_distribution.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot the distribution of cosine silhouette values."""
    config.validate()
    values = np.asarray(silhouette_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("silhouette_values must be a non-empty 1-D array")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(values, bins=30)
    ax.axvline(0.0, linewidth=1.0)
    if macro_silhouette is not None:
        ax.axvline(
            float(macro_silhouette),
            linestyle="--",
            linewidth=1.4,
            label=f"macro silhouette = {float(macro_silhouette):.3f}",
        )
        ax.legend()
    ax.set_title(_title(title, "cosine silhouette distribution"))
    ax.set_xlabel("Cosine silhouette")
    ax.set_ylabel("Evaluated samples")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def _normalize_domain_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a domain-comparison table to metrics x conditions."""
    result = frame.copy()
    unnamed = [column for column in result.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        result = result.set_index(unnamed[0])

    metric_hits_in_index = sum(metric in result.index for metric in DOMAIN_READOUT_METRICS)
    metric_hits_in_columns = sum(metric in result.columns for metric in DOMAIN_READOUT_METRICS)
    if metric_hits_in_columns > metric_hits_in_index:
        result = result.T
    return result


def plot_domain_readout_comparison(
    comparison: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "domain_readout_comparison.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Compare raw/reconstruction conditions using the main 0-1 readouts."""
    config.validate()
    comparison = _normalize_domain_comparison(comparison)
    missing = [metric for metric in DOMAIN_READOUT_METRICS if metric not in comparison.index]
    if missing:
        raise KeyError(f"domain comparison is missing metrics: {missing}")

    table = comparison.loc[list(DOMAIN_READOUT_METRICS)].astype(float).T

    fig, ax = plt.subplots(figsize=(11, 5.5))
    table.plot(kind="bar", ax=ax)
    ax.set_title(_title(title, "domain/readout comparison"))
    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_domain_geometry_comparison(
    comparison: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "domain_geometry_comparison.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Compare domain geometry/clustering diagnostics.

    This is a diagnostic plot, not a single common-scale score: D_intra,
    D_inter and their ratio have different interpretations.
    """
    config.validate()
    comparison = _normalize_domain_comparison(comparison)
    missing = [metric for metric in DOMAIN_GEOMETRY_METRICS if metric not in comparison.index]
    if missing:
        raise KeyError(f"domain comparison is missing metrics: {missing}")

    table = comparison.loc[list(DOMAIN_GEOMETRY_METRICS)].astype(float).T

    fig, ax = plt.subplots(figsize=(11, 5.5))
    table.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(_title(title, "domain geometry and clustering diagnostics"))
    ax.set_xlabel("Condition")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_paired_summary(
    summary: pd.Series | Mapping[str, float],
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "paired_preservation_summary.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot raw/reconstruction embedding similarity and prediction agreement."""
    config.validate()
    series = pd.Series(summary)
    metrics = (
        "mean_embedding_cosine_similarity",
        "median_embedding_cosine_similarity",
        "prediction_agreement",
    )
    _require_metrics(series, metrics, name="paired summary")

    values = pd.Series(
        {
            "Mean embedding cosine similarity": float(
                series["mean_embedding_cosine_similarity"]
            ),
            "Median embedding cosine similarity": float(
                series["median_embedding_cosine_similarity"]
            ),
            "Prediction agreement": float(series["prediction_agreement"]),
        }
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    values.plot(kind="bar", ax=ax)
    ax.set_title(_title(title, "paired raw/reconstruction preservation"))
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, config=config)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_paired_transitions(
    transitions: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "paired_correctness_transitions.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot correctness transitions from reference raw to reconstruction query."""
    config.validate()
    _require_columns(
        transitions,
        ["transition", "fraction"],
        name="paired transitions",
    )
    ordered_names = [
        "reference_correct -> query_correct",
        "reference_correct -> query_wrong",
        "reference_wrong -> query_correct",
        "reference_wrong -> query_wrong",
    ]
    table = transitions.set_index("transition").reindex(ordered_names).fillna(0.0)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(table.index, table["fraction"].astype(float))
    ax.invert_yaxis()
    ax.set_title(_title(title, "paired correctness transitions"))
    ax.set_xlabel("Fraction of paired test samples")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_paired_per_class_similarity(
    per_class: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "paired_per_class_similarity.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot per-class raw/reconstruction embedding cosine similarity."""
    config.validate()
    _require_columns(
        per_class,
        ["label", "mean_embedding_cosine_similarity"],
        name="paired per-class table",
    )
    ordered = per_class.sort_values(
        "mean_embedding_cosine_similarity",
        ascending=True,
    ).reset_index(drop=True)

    height = max(4.5, min(14.0, 0.35 * len(ordered) + 2.0))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(
        ordered["label"].astype(str),
        ordered["mean_embedding_cosine_similarity"].astype(float),
    )
    ax.set_title(_title(title, "per-class raw/reconstruction embedding similarity"))
    ax.set_xlabel("Mean cosine similarity")
    ax.set_xlim(0.0, 1.02)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_paired_per_class_agreement(
    per_class: pd.DataFrame,
    *,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "paired_per_class_prediction_agreement.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot per-class CNN prediction agreement between paired domains."""
    config.validate()
    _require_columns(
        per_class,
        ["label", "prediction_agreement"],
        name="paired per-class table",
    )
    ordered = per_class.sort_values(
        "prediction_agreement",
        ascending=True,
    ).reset_index(drop=True)

    height = max(4.5, min(14.0, 0.35 * len(ordered) + 2.0))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.barh(
        ordered["label"].astype(str),
        ordered["prediction_agreement"].astype(float),
    )
    ax.set_title(_title(title, "per-class prediction agreement"))
    ax.set_xlabel("Prediction agreement")
    ax.set_xlim(0.0, 1.02)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def plot_paired_cosine_distribution(
    cosine_similarity: np.ndarray,
    *,
    mean_similarity: float | None = None,
    title: str | None = None,
    output_dir: str | Path | None = None,
    filename: str = "paired_cosine_similarity_distribution.png",
    config: VisualizationConfig = VisualizationConfig(),
) -> Path | None:
    """Plot one-to-one raw/reconstruction embedding cosine similarities."""
    config.validate()
    values = np.asarray(cosine_similarity, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("cosine_similarity must be a non-empty 1-D array")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(values, bins=30)
    if mean_similarity is not None:
        ax.axvline(
            float(mean_similarity),
            linestyle="--",
            linewidth=1.4,
            label=f"mean = {float(mean_similarity):.3f}",
        )
        ax.legend()
    ax.set_title(_title(title, "paired embedding cosine similarity distribution"))
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Paired test samples")
    ax.set_xlim(-1.0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save_figure(fig, output_dir, filename, config=config)


def _register_path(
    result: dict[str, Path],
    key: str,
    value: Path | None,
) -> None:
    if value is not None:
        result[key] = value


def visualize_standard_evaluation(
    evaluation_dir: str | Path,
    *,
    embedding_path: str | Path,
    title: str,
    output_dir: str | Path | None = None,
    class_to_idx: Mapping[str, int] | None = None,
    classification_splits_path: str | Path | None = None,
    training_history_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    include_training: bool = True,
    include_classification_splits: bool = True,
    config: VisualizationConfig = VisualizationConfig(),
) -> dict[str, Path]:
    """Generate the reusable standard figure suite for one evaluation condition.

    This function is the common core used by A, B, C and D. ``evaluation_dir``
    must contain the standardized representation-evaluation files. The embedding
    file is explicit because B/D keep their test embeddings in the parent
    experiment directory.
    """
    config.validate()
    evaluation_dir = _as_path(evaluation_dir)
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")

    if output_dir is None:
        output_dir = evaluation_dir / "figures"
    output_dir = _ensure_dir(output_dir)

    summary = load_summary(evaluation_dir)
    embeddings = load_embedding_bundle(embedding_path)

    if class_to_idx is None and provenance_path is not None:
        provenance = _as_path(provenance_path)
        if provenance.is_file():
            class_to_idx = load_class_to_idx(provenance)

    figures: dict[str, Path] = {}

    if include_training and training_history_path is not None:
        training_path = _as_path(training_history_path)
        if training_path.is_file():
            history = pd.read_csv(training_path)
            best_epoch, best_val = _best_epoch_from_provenance(provenance_path)
            _register_path(
                figures,
                "training_loss",
                plot_training_loss(
                    history,
                    title=title,
                    best_epoch=best_epoch,
                    output_dir=output_dir,
                    filename="01_training_loss.png",
                    config=config,
                ),
            )
            _register_path(
                figures,
                "training_balanced_accuracy",
                plot_training_balanced_accuracy(
                    history,
                    title=title,
                    best_epoch=best_epoch,
                    best_val_balanced_accuracy=best_val,
                    output_dir=output_dir,
                    filename="02_training_balanced_accuracy.png",
                    config=config,
                ),
            )

    if include_classification_splits and classification_splits_path is not None:
        classification_path = _as_path(classification_splits_path)
        if classification_path.is_file():
            classification = _read_indexed_csv(classification_path)
            _register_path(
                figures,
                "classification_splits",
                plot_classification_splits(
                    classification,
                    title=title,
                    output_dir=output_dir,
                    filename="03_classification_splits.png",
                    config=config,
                ),
            )

    knn_validation = _read_regular_csv(evaluation_dir / "knn_validation.csv")
    same_label = _read_regular_csv(evaluation_dir / "same_label_at_k.csv")
    intra_by_class = _read_regular_csv(evaluation_dir / "intra_by_class.csv")
    linear_probe_history = _read_regular_csv(evaluation_dir / "linear_probe_history.csv")

    with np.load(
        _ensure_file(evaluation_dir / "evaluation_arrays.npz"),
        allow_pickle=False,
    ) as arrays:
        average_precision = arrays["average_precision"].astype(np.float64)
        silhouette_values = arrays["silhouette_values"].astype(np.float64)

    selected_k = int(summary["kNN_selected_k"])

    _register_path(
        figures,
        "knn_validation",
        plot_knn_validation(
            knn_validation,
            selected_k=selected_k,
            title=title,
            output_dir=output_dir,
            filename="04_knn_validation_selection.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "same_label_at_k",
        plot_same_label_at_k(
            same_label,
            title=title,
            output_dir=output_dir,
            filename="05_same_label_at_k.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "representation_dashboard",
        plot_representation_dashboard(
            summary,
            title=title,
            output_dir=output_dir,
            filename="06_representation_readout_dashboard.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "embedding_geometry",
        plot_embedding_geometry(
            summary,
            title=title,
            output_dir=output_dir,
            filename="07_embedding_geometry.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "unsupervised_metrics",
        plot_unsupervised_metrics(
            summary,
            title=title,
            output_dir=output_dir,
            filename="08_unsupervised_embedding_quality.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "linear_probe_history",
        plot_linear_probe_history(
            linear_probe_history,
            title=title,
            output_dir=output_dir,
            filename="09_linear_probe_validation.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "confusion_matrix",
        plot_confusion_matrix(
            embeddings["y"],
            embeddings["cnn_pred"],
            class_to_idx=class_to_idx,
            title=title,
            output_dir=output_dir,
            filename="10_test_confusion_matrix_normalized.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "embedding_pca",
        plot_embedding_pca(
            embeddings["z"],
            embeddings["y"],
            class_to_idx=class_to_idx,
            title=title,
            output_dir=output_dir,
            filename="11_test_embedding_pca.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "per_class_compactness",
        plot_per_class_compactness(
            intra_by_class,
            title=title,
            output_dir=output_dir,
            filename="12_per_class_intra_distance.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "average_precision_distribution",
        plot_average_precision_distribution(
            average_precision,
            macro_map=float(summary["retrieval_macro_mAP"]),
            title=title,
            output_dir=output_dir,
            filename="13_average_precision_distribution.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "silhouette_distribution",
        plot_silhouette_distribution(
            silhouette_values,
            macro_silhouette=float(summary["silhouette_macro"]),
            title=title,
            output_dir=output_dir,
            filename="14_silhouette_distribution.png",
            config=config,
        ),
    )
    return figures


def _visualize_paired_directory(
    paired_dir: str | Path,
    *,
    title: str,
    output_dir: str | Path,
    config: VisualizationConfig,
) -> dict[str, Path]:
    paired_dir = _as_path(paired_dir)
    output_dir = _ensure_dir(output_dir)

    summary = load_summary(paired_dir / "paired_summary.csv")
    transitions = _read_regular_csv(paired_dir / "paired_transitions.csv")
    per_class = _read_regular_csv(paired_dir / "paired_per_class.csv")
    with np.load(
        _ensure_file(paired_dir / "paired_arrays.npz"),
        allow_pickle=False,
    ) as arrays:
        cosine_similarity = arrays["cosine_similarity"].astype(np.float64)

    figures: dict[str, Path] = {}
    _register_path(
        figures,
        "paired_summary",
        plot_paired_summary(
            summary,
            title=title,
            output_dir=output_dir,
            filename="paired_01_summary.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "paired_transitions",
        plot_paired_transitions(
            transitions,
            title=title,
            output_dir=output_dir,
            filename="paired_02_correctness_transitions.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "paired_per_class_similarity",
        plot_paired_per_class_similarity(
            per_class,
            title=title,
            output_dir=output_dir,
            filename="paired_03_per_class_similarity.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "paired_per_class_agreement",
        plot_paired_per_class_agreement(
            per_class,
            title=title,
            output_dir=output_dir,
            filename="paired_04_per_class_prediction_agreement.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "paired_cosine_distribution",
        plot_paired_cosine_distribution(
            cosine_similarity,
            mean_similarity=float(summary["mean_embedding_cosine_similarity"]),
            title=title,
            output_dir=output_dir,
            filename="paired_05_cosine_similarity_distribution.png",
            config=config,
        ),
    )
    return figures


def _visualize_domain_comparison(
    comparison_path: str | Path,
    *,
    title: str,
    output_dir: str | Path,
    config: VisualizationConfig,
) -> dict[str, Path]:
    comparison = _read_indexed_csv(comparison_path)
    figures: dict[str, Path] = {}
    _register_path(
        figures,
        "domain_readouts",
        plot_domain_readout_comparison(
            comparison,
            title=title,
            output_dir=output_dir,
            filename="domain_01_readout_comparison.png",
            config=config,
        ),
    )
    _register_path(
        figures,
        "domain_geometry",
        plot_domain_geometry_comparison(
            comparison,
            title=title,
            output_dir=output_dir,
            filename="domain_02_geometry_comparison.png",
            config=config,
        ),
    )
    return figures


def visualize_experiment_a(
    output_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
    config: VisualizationConfig = VisualizationConfig(),
) -> dict[str, Path]:
    """Generate the Experiment A visualization suite."""
    root = _as_path(output_dir)
    figures = root / "figures" if figure_dir is None else figure_dir
    return visualize_standard_evaluation(
        root,
        embedding_path=root / "embeddings_test.npz",
        title="Experiment A — Raw → Raw",
        output_dir=figures,
        classification_splits_path=root / "classification_splits.csv",
        training_history_path=root / "training_history.csv",
        provenance_path=root / "provenance.json",
        config=config,
    )


def visualize_experiment_c(
    output_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
    config: VisualizationConfig = VisualizationConfig(),
) -> dict[str, Path]:
    """Generate the Experiment C visualization suite."""
    root = _as_path(output_dir)
    figures = root / "figures" if figure_dir is None else figure_dir
    return visualize_standard_evaluation(
        root,
        embedding_path=root / "embeddings_test.npz",
        title="Experiment C — Reconstruction → Reconstruction",
        output_dir=figures,
        classification_splits_path=root / "classification_splits.csv",
        training_history_path=root / "training_history.csv",
        provenance_path=root / "provenance.json",
        config=config,
    )


def visualize_experiment_b(
    output_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
    include_raw_reference: bool = True,
    config: VisualizationConfig = VisualizationConfig(),
) -> dict[str, Path]:
    """Generate Experiment B reconstruction, domain-shift and paired figures.

    Experiment B does not train a CNN, so no CNN training-history plot is made.
    ``include_raw_reference=True`` additionally generates the standard
    representation plots for the frozen model's raw held-out test reference.
    """
    root = _as_path(output_dir)
    base_figures = _ensure_dir(root / "figures" if figure_dir is None else figure_dir)
    provenance_path = root / "provenance.json"
    class_to_idx = load_class_to_idx(provenance_path)

    figures: dict[str, Path] = {}

    reconstruction_figures = visualize_standard_evaluation(
        root,
        embedding_path=root / "embeddings_test_reconstruction.npz",
        title="Experiment B — Frozen Raw-trained CNN → Reconstruction",
        output_dir=base_figures / "reconstruction_test",
        class_to_idx=class_to_idx,
        classification_splits_path=root / "classification_splits.csv",
        provenance_path=provenance_path,
        include_training=False,
        include_classification_splits=True,
        config=config,
    )
    figures.update(
        {f"reconstruction_{key}": path for key, path in reconstruction_figures.items()}
    )

    if include_raw_reference:
        raw_figures = visualize_standard_evaluation(
            root / "raw_reference",
            embedding_path=root / "embeddings_test_raw.npz",
            title="Experiment B — Frozen Raw-trained CNN → Raw reference",
            output_dir=base_figures / "raw_reference",
            class_to_idx=class_to_idx,
            provenance_path=provenance_path,
            include_training=False,
            include_classification_splits=False,
            config=config,
        )
        figures.update({f"raw_reference_{key}": path for key, path in raw_figures.items()})

    domain_figures = _visualize_domain_comparison(
        root / "domain_comparison.csv",
        title="Experiment B — Raw reference vs Reconstruction query",
        output_dir=base_figures / "domain_comparison",
        config=config,
    )
    figures.update({f"domain_{key}": path for key, path in domain_figures.items()})

    paired_figures = _visualize_paired_directory(
        root / "paired_preservation",
        title="Experiment B — Paired Raw vs Reconstruction",
        output_dir=base_figures / "paired_preservation",
        config=config,
    )
    figures.update({f"paired_{key}": path for key, path in paired_figures.items()})

    return figures


def visualize_experiment_d(
    output_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
    config: VisualizationConfig = VisualizationConfig(),
) -> dict[str, Path]:
    """Generate Experiment D training, dual-domain and paired figures."""
    root = _as_path(output_dir)
    base_figures = _ensure_dir(root / "figures" if figure_dir is None else figure_dir)
    provenance_path = root / "provenance.json"
    class_to_idx = load_class_to_idx(provenance_path)

    figures: dict[str, Path] = {}

    # D has one mixed-domain training history and one four-row classification
    # table; plot these once at the experiment level.
    history_path = root / "training_history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        best_epoch, best_val = _best_epoch_from_provenance(provenance_path)
        _register_path(
            figures,
            "training_loss",
            plot_training_loss(
                history,
                title="Experiment D — Mixed Raw + Reconstruction training",
                best_epoch=best_epoch,
                output_dir=base_figures,
                filename="01_training_loss.png",
                config=config,
            ),
        )
        _register_path(
            figures,
            "training_balanced_accuracy",
            plot_training_balanced_accuracy(
                history,
                title="Experiment D — Mixed Raw + Reconstruction training",
                best_epoch=best_epoch,
                best_val_balanced_accuracy=best_val,
                output_dir=base_figures,
                filename="02_training_balanced_accuracy.png",
                config=config,
            ),
        )

    classification_path = root / "classification_splits.csv"
    if classification_path.is_file():
        classification = _read_indexed_csv(classification_path)
        _register_path(
            figures,
            "classification_splits",
            plot_classification_splits(
                classification,
                title="Experiment D — Same mixed-trained checkpoint",
                output_dir=base_figures,
                filename="03_classification_splits.png",
                config=config,
            ),
        )

    raw_figures = visualize_standard_evaluation(
        root / "raw_test",
        embedding_path=root / "embeddings_test_raw.npz",
        title="Experiment D — Mixed → Raw test",
        output_dir=base_figures / "raw_test",
        class_to_idx=class_to_idx,
        provenance_path=provenance_path,
        include_training=False,
        include_classification_splits=False,
        config=config,
    )
    figures.update({f"raw_{key}": path for key, path in raw_figures.items()})

    reconstruction_figures = visualize_standard_evaluation(
        root / "reconstruction_test",
        embedding_path=root / "embeddings_test_reconstruction.npz",
        title="Experiment D — Mixed → Reconstruction test",
        output_dir=base_figures / "reconstruction_test",
        class_to_idx=class_to_idx,
        provenance_path=provenance_path,
        include_training=False,
        include_classification_splits=False,
        config=config,
    )
    figures.update(
        {f"reconstruction_{key}": path for key, path in reconstruction_figures.items()}
    )

    domain_figures = _visualize_domain_comparison(
        root / "domain_comparison.csv",
        title="Experiment D — Same checkpoint across test domains",
        output_dir=base_figures / "domain_comparison",
        config=config,
    )
    figures.update({f"domain_{key}": path for key, path in domain_figures.items()})

    paired_figures = _visualize_paired_directory(
        root / "paired_preservation",
        title="Experiment D — Paired Raw vs Reconstruction",
        output_dir=base_figures / "paired_preservation",
        config=config,
    )
    figures.update({f"paired_{key}": path for key, path in paired_figures.items()})

    return figures
