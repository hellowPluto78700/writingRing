from __future__ import annotations

"""High-level representation evaluation protocols for Experiments A/B/C/D.

The main entry point is:

    result = evaluate_representation(
        train_bundle=...,
        val_bundle=...,
        test_bundle=...,
        num_classes=...,
        device=...,
        idx_to_class=...,
    )

The same function supports both:
- same-domain evaluation (A/C/D), and
- Experiment B cross-domain evaluation by passing raw train/val bundles and a
  reconstructed test bundle.

This keeps the reference gallery, prototype centroids, K-selection, feature
standardization, and linear-probe training tied to the train/validation domain.
"""

import copy
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


try:
    from .embedding import (
        EmbeddingBundle,
        apply_feature_standardization,
        paired_embedding_cosine,
        standardize_feature_splits,
        validate_paired_bundles,
    )
    from .metrics import (
        adjusted_rand_index,
        class_centroids,
        cosine_silhouette,
        cosine_topk,
        kmeans_best_of_n,
        knn_predict_from_topk,
        macro_intra_cosine_distance,
        normalized_mutual_information,
        prototype_classification,
        retrieval_map,
        same_label_at_k,
        select_knn_k,
        centroid_inter_cosine_distance,
    )
    from .training import classification_metrics
except ImportError:
    from embedding import (
        EmbeddingBundle,
        apply_feature_standardization,
        paired_embedding_cosine,
        standardize_feature_splits,
        validate_paired_bundles,
    )
    from metrics import (
        adjusted_rand_index,
        class_centroids,
        cosine_silhouette,
        cosine_topk,
        kmeans_best_of_n,
        knn_predict_from_topk,
        macro_intra_cosine_distance,
        normalized_mutual_information,
        prototype_classification,
        retrieval_map,
        same_label_at_k,
        select_knn_k,
        centroid_inter_cosine_distance,
    )
    from training import classification_metrics


__all__ = [
    "LinearProbeConfig",
    "RepresentationEvaluationConfig",
    "RepresentationEvaluationResult",
    "PairedPreservationResult",
    "evaluate_linear_probe",
    "train_linear_probe",
    "evaluate_retrieval",
    "evaluate_geometry",
    "evaluate_unsupervised",
    "evaluate_representation",
    "evaluate_paired_preservation",
    "compare_representation_results",
]


@dataclass(frozen=True)
class LinearProbeConfig:
    """Defaults match the current acceleration-CNN evaluation notebook."""

    epochs: int = 200
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    batch_size: int = 512
    patience: int = 30
    seed_offset: int = 1

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("Linear-probe epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Linear-probe learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Linear-probe weight_decay must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("Linear-probe batch_size must be positive")
        if self.patience <= 0:
            raise ValueError("Linear-probe patience must be positive")


@dataclass(frozen=True)
class RepresentationEvaluationConfig:
    """Evaluation defaults matching the current notebook."""

    k_values: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    knn_k_candidates: tuple[int, ...] = (1, 3, 5, 7, 10, 15, 20)
    similarity_query_batch_size: int = 128

    linear_probe: LinearProbeConfig = field(
        default_factory=LinearProbeConfig
    )

    kmeans_n_init: int = 10
    kmeans_max_iter: int = 100
    silhouette_max_samples: int | None = 3000

    random_seed: int = 12345
    verbose: bool = True

    def validate(self) -> None:
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if (
            not self.knn_k_candidates
            or any(k <= 0 for k in self.knn_k_candidates)
        ):
            raise ValueError(
                "knn_k_candidates must contain positive integers"
            )
        if self.similarity_query_batch_size <= 0:
            raise ValueError(
                "similarity_query_batch_size must be positive"
            )
        if self.kmeans_n_init <= 0:
            raise ValueError("kmeans_n_init must be positive")
        if self.kmeans_max_iter <= 0:
            raise ValueError("kmeans_max_iter must be positive")
        if (
            self.silhouette_max_samples is not None
            and self.silhouette_max_samples <= 0
        ):
            raise ValueError(
                "silhouette_max_samples must be positive or None"
            )
        self.linear_probe.validate()


@dataclass
class RepresentationEvaluationResult:
    """All reusable outputs from one representation evaluation."""

    summary: pd.DataFrame
    cnn_metrics: dict[str, float]

    selected_knn_k: int
    knn_metrics: dict[str, float]
    knn_validation_table: pd.DataFrame
    knn_prediction: np.ndarray

    retrieval_summary: pd.DataFrame
    same_label_table: pd.DataFrame
    average_precision: np.ndarray

    prototype_metrics: dict[str, float]
    prototype_prediction: np.ndarray
    geometry_summary: pd.DataFrame
    intra_by_class: pd.DataFrame

    linear_probe_metrics: dict[str, float]
    linear_probe_history: pd.DataFrame
    linear_probe_best_epoch: int
    linear_probe_best_val_balanced_accuracy: float
    linear_probe_state_dict: dict[str, torch.Tensor]
    linear_probe_feature_mean: np.ndarray
    linear_probe_feature_std: np.ndarray

    unsupervised_summary: pd.DataFrame
    kmeans_assignment: np.ndarray
    kmeans_centers: np.ndarray
    silhouette_values: np.ndarray
    silhouette_indices: np.ndarray

    def metric_series(self) -> pd.Series:
        """Return the single-row summary as a metric-value Series."""
        if len(self.summary) != 1:
            raise ValueError("summary must contain exactly one row")
        return self.summary.iloc[0].copy()


@dataclass
class PairedPreservationResult:
    """Experiment-B style one-to-one raw/reconstruction diagnostics."""

    summary: pd.DataFrame
    transitions: pd.DataFrame
    per_class: pd.DataFrame
    cosine_similarity: np.ndarray
    cosine_distance: np.ndarray


def _bundle_array(
    bundle: EmbeddingBundle | Mapping[str, Any],
    key: str,
) -> np.ndarray:
    try:
        return np.asarray(bundle[key])
    except KeyError as error:
        raise KeyError(f"Embedding bundle is missing {key!r}") from error


def _validate_bundle(
    bundle: EmbeddingBundle | Mapping[str, Any],
    *,
    name: str,
) -> None:
    h = _bundle_array(bundle, "h")
    z = _bundle_array(bundle, "z")
    y = _bundle_array(bundle, "y")
    pred = _bundle_array(bundle, "cnn_pred")
    logits = _bundle_array(bundle, "logits")
    sample_id = _bundle_array(bundle, "sample_id")

    if h.ndim != 2 or z.shape != h.shape:
        raise ValueError(f"{name}: h/z shapes are invalid")
    n = len(h)
    if y.shape != (n,):
        raise ValueError(f"{name}: y must have shape {(n,)}")
    if pred.shape != (n,):
        raise ValueError(f"{name}: cnn_pred must have shape {(n,)}")
    if logits.ndim != 2 or logits.shape[0] != n:
        raise ValueError(f"{name}: logits must have shape (N,C)")
    if sample_id.shape != (n,):
        raise ValueError(f"{name}: sample_id must have shape {(n,)}")
    if n == 0:
        raise ValueError(f"{name}: bundle must be non-empty")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_linear_probe(
    probe: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate a fitted linear probe with current classification metrics."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if features.ndim != 2:
        raise ValueError("features must be 2-D")
    if labels.shape != (len(features),):
        raise ValueError("labels must align with features")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    probe.eval()
    logits_parts: list[np.ndarray] = []

    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch = torch.from_numpy(
            features[start:stop]
        ).to(
            device=device,
            dtype=torch.float32,
        )
        logits_parts.append(
            probe(batch).cpu().numpy()
        )

    logits = np.concatenate(
        logits_parts
    )
    predictions = logits.argmax(
        axis=1
    ).astype(np.int64)

    return classification_metrics(
        labels.tolist(),
        predictions.tolist(),
    )


def train_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    num_classes: int,
    device: torch.device,
    config: LinearProbeConfig | None = None,
    random_seed: int = 12345,
) -> tuple[
    nn.Module,
    pd.DataFrame,
    int,
    float,
]:
    """Train the notebook's linear probe with early stopping on validation BA."""
    config = (
        LinearProbeConfig()
        if config is None
        else config
    )
    config.validate()

    train_x = np.asarray(
        train_x,
        dtype=np.float32,
    )
    val_x = np.asarray(
        val_x,
        dtype=np.float32,
    )
    train_y = np.asarray(
        train_y,
        dtype=np.int64,
    )
    val_y = np.asarray(
        val_y,
        dtype=np.int64,
    )

    if train_x.ndim != 2 or val_x.ndim != 2:
        raise ValueError("train_x and val_x must be 2-D")
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError(
            "train_x and val_x feature dimensions must match"
        )
    if train_y.shape != (len(train_x),):
        raise ValueError("train_y must align with train_x")
    if val_y.shape != (len(val_x),):
        raise ValueError("val_y must align with val_x")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    seed = (
        int(random_seed)
        + int(config.seed_offset)
    )
    _set_random_seed(seed)

    probe = nn.Linear(
        train_x.shape[1],
        num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(
            config.batch_size,
            len(dataset),
        ),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    best_state: dict[
        str,
        torch.Tensor,
    ] | None = None
    best_epoch = -1
    best_val = -math.inf
    no_improvement = 0
    rows: list[
        dict[str, float | int]
    ] = []

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        probe.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(
                device=device,
                dtype=torch.float32,
            )
            y_batch = y_batch.to(
                device=device,
                dtype=torch.long,
            )

            optimizer.zero_grad(
                set_to_none=True
            )
            logits = probe(x_batch)
            loss = criterion(
                logits,
                y_batch,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite linear-probe loss"
                )
            loss.backward()
            optimizer.step()

            train_loss_sum += (
                float(loss.detach())
                * len(y_batch)
            )
            train_count += len(y_batch)

        val_metrics = evaluate_linear_probe(
            probe,
            val_x,
            val_y,
            device=device,
            batch_size=config.batch_size,
        )
        current = float(
            val_metrics[
                "balanced_accuracy"
            ]
        )

        rows.append(
            {
                "epoch": epoch,
                "train_loss": (
                    train_loss_sum
                    / train_count
                ),
                "val_balanced_accuracy": current,
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
            }
        )

        if current > best_val + 1e-12:
            best_val = current
            best_epoch = epoch
            best_state = copy.deepcopy(
                probe.state_dict()
            )
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(
            "Linear probe did not produce a best state"
        )

    probe.load_state_dict(
        best_state
    )
    return (
        probe,
        pd.DataFrame(rows),
        best_epoch,
        best_val,
    )


def evaluate_retrieval(
    *,
    train_bundle: EmbeddingBundle | Mapping[str, Any],
    val_bundle: EmbeddingBundle | Mapping[str, Any],
    test_bundle: EmbeddingBundle | Mapping[str, Any],
    k_values: Sequence[int],
    knn_k_candidates: Sequence[int],
    batch_size: int = 128,
) -> dict[str, object]:
    """Evaluate kNN, SameLabel@K, and retrieval mAP against train gallery."""
    train_z = _bundle_array(train_bundle, "z")
    train_y = _bundle_array(train_bundle, "y").astype(np.int64)
    val_z = _bundle_array(val_bundle, "z")
    val_y = _bundle_array(val_bundle, "y").astype(np.int64)
    test_z = _bundle_array(test_bundle, "z")
    test_y = _bundle_array(test_bundle, "y").astype(np.int64)

    k_values = tuple(
        sorted(
            set(
                int(k)
                for k in k_values
            )
        )
    )
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must be positive")
    if max(k_values) > len(train_y):
        raise ValueError(
            f"Largest SameLabel K={max(k_values)} exceeds "
            f"train gallery size={len(train_y)}"
        )

    selected_k, validation_table = select_knn_k(
        val_z=val_z,
        val_y=val_y,
        train_z=train_z,
        train_y=train_y,
        candidates=knn_k_candidates,
        batch_size=batch_size,
    )

    max_k = max(
        max(k_values),
        selected_k,
    )
    topk_indices, topk_scores = cosine_topk(
        test_z,
        train_z,
        max_k=max_k,
        batch_size=batch_size,
    )

    knn_prediction = knn_predict_from_topk(
        topk_indices,
        topk_scores,
        train_y,
        k=selected_k,
    )
    knn_metrics = classification_metrics(
        test_y.tolist(),
        knn_prediction.tolist(),
    )

    same_label_rows: list[
        dict[str, float | int]
    ] = []
    for k in k_values:
        micro, macro, _ = same_label_at_k(
            topk_indices,
            query_y=test_y,
            gallery_y=train_y,
            k=k,
        )
        same_label_rows.append(
            {
                "k": k,
                "same_label_micro": micro,
                "same_label_macro": macro,
            }
        )

    map_micro, map_macro, aps = retrieval_map(
        test_z,
        train_z,
        query_y=test_y,
        gallery_y=train_y,
        batch_size=batch_size,
    )

    return {
        "selected_knn_k": selected_k,
        "knn_metrics": knn_metrics,
        "knn_validation_table": validation_table,
        "knn_prediction": knn_prediction,
        "same_label_table": pd.DataFrame(
            same_label_rows
        ),
        "retrieval_summary": pd.DataFrame(
            [
                {
                    "mAP_micro": map_micro,
                    "mAP_macro": map_macro,
                }
            ]
        ),
        "average_precision": aps,
    }


def evaluate_geometry(
    *,
    train_bundle: EmbeddingBundle | Mapping[str, Any],
    test_bundle: EmbeddingBundle | Mapping[str, Any],
    idx_to_class: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Evaluate D_intra, D_inter, ratio, and raw-train prototype readout."""
    train_z = _bundle_array(train_bundle, "z")
    train_y = _bundle_array(train_bundle, "y").astype(np.int64)
    test_z = _bundle_array(test_bundle, "z")
    test_y = _bundle_array(test_bundle, "y").astype(np.int64)

    d_intra, intra_by_class = macro_intra_cosine_distance(
        test_z,
        test_y,
        idx_to_class=idx_to_class,
    )
    d_inter = centroid_inter_cosine_distance(
        test_z,
        test_y,
    )
    separation_ratio = (
        d_inter
        / (d_intra + 1e-12)
    )

    prototype_metrics, prototype_prediction = (
        prototype_classification(
            train_z=train_z,
            train_y=train_y,
            test_z=test_z,
            test_y=test_y,
        )
    )

    geometry_summary = pd.DataFrame(
        [
            {
                "D_intra_macro": d_intra,
                "D_inter_centroids": d_inter,
                "D_inter_over_D_intra": separation_ratio,
                "prototype_balanced_accuracy": prototype_metrics[
                    "balanced_accuracy"
                ],
                "prototype_accuracy": prototype_metrics[
                    "accuracy"
                ],
                "prototype_macro_f1": prototype_metrics[
                    "macro_f1"
                ],
            }
        ]
    )

    return {
        "D_intra_macro": d_intra,
        "D_inter_centroids": d_inter,
        "D_inter_over_D_intra": separation_ratio,
        "prototype_metrics": prototype_metrics,
        "prototype_prediction": prototype_prediction,
        "geometry_summary": geometry_summary,
        "intra_by_class": intra_by_class,
    }


def evaluate_unsupervised(
    test_bundle: EmbeddingBundle | Mapping[str, Any],
    *,
    kmeans_n_init: int = 10,
    kmeans_max_iter: int = 100,
    silhouette_max_samples: int | None = 3000,
    random_seed: int = 12345,
    idx_to_class: Mapping[int, str] | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    """Run test-only KMeans/NMI/ARI/cosine-silhouette diagnostics."""
    test_z = _bundle_array(test_bundle, "z")
    test_y = _bundle_array(test_bundle, "y").astype(np.int64)

    num_test_classes = len(
        np.unique(test_y)
    )
    if num_test_classes < 2:
        raise ValueError(
            "Unsupervised evaluation requires at least two classes"
        )

    assignment, centers, inertia = kmeans_best_of_n(
        test_z,
        k=num_test_classes,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        seed=random_seed,
    )
    nmi_value = normalized_mutual_information(
        test_y,
        assignment,
    )
    ari_value = adjusted_rand_index(
        test_y,
        assignment,
    )
    (
        silhouette_micro,
        silhouette_macro,
        silhouette_values,
        silhouette_indices,
    ) = cosine_silhouette(
        test_z,
        test_y,
        max_samples=silhouette_max_samples,
        seed=random_seed,
        idx_to_class=idx_to_class,
        verbose=verbose,
    )

    summary = pd.DataFrame(
        [
            {
                "kmeans_clusters": num_test_classes,
                "kmeans_inertia": inertia,
                "NMI_arithmetic": nmi_value,
                "ARI": ari_value,
                "silhouette_micro": silhouette_micro,
                "silhouette_macro": silhouette_macro,
                "silhouette_samples": len(
                    silhouette_indices
                ),
            }
        ]
    )

    return {
        "summary": summary,
        "kmeans_assignment": assignment,
        "kmeans_centers": centers,
        "kmeans_inertia": inertia,
        "NMI_arithmetic": nmi_value,
        "ARI": ari_value,
        "silhouette_micro": silhouette_micro,
        "silhouette_macro": silhouette_macro,
        "silhouette_values": silhouette_values,
        "silhouette_indices": silhouette_indices,
    }


def evaluate_representation(
    *,
    train_bundle: EmbeddingBundle | Mapping[str, Any],
    val_bundle: EmbeddingBundle | Mapping[str, Any],
    test_bundle: EmbeddingBundle | Mapping[str, Any],
    num_classes: int,
    device: torch.device,
    idx_to_class: Mapping[int, str] | None = None,
    config: RepresentationEvaluationConfig | None = None,
) -> RepresentationEvaluationResult:
    """Run the full current representation-evaluation protocol.

    For Experiment B, pass:
        train_bundle = raw_train
        val_bundle   = raw_val
        test_bundle  = reconstructed_test

    Thus no probe, prototype, kNN selection, or gallery is adapted to the
    reconstruction domain.
    """
    config = (
        RepresentationEvaluationConfig()
        if config is None
        else config
    )
    config.validate()

    _validate_bundle(
        train_bundle,
        name="train_bundle",
    )
    _validate_bundle(
        val_bundle,
        name="val_bundle",
    )
    _validate_bundle(
        test_bundle,
        name="test_bundle",
    )

    train_h = _bundle_array(
        train_bundle,
        "h",
    )
    val_h = _bundle_array(
        val_bundle,
        "h",
    )
    test_h = _bundle_array(
        test_bundle,
        "h",
    )

    if (
        train_h.shape[1]
        != val_h.shape[1]
        or train_h.shape[1]
        != test_h.shape[1]
    ):
        raise ValueError(
            "Embedding dimensions must match across train/val/test"
        )

    test_y = _bundle_array(
        test_bundle,
        "y",
    ).astype(np.int64)
    test_cnn_pred = _bundle_array(
        test_bundle,
        "cnn_pred",
    ).astype(np.int64)

    cnn_metrics = classification_metrics(
        test_y.tolist(),
        test_cnn_pred.tolist(),
    )

    retrieval = evaluate_retrieval(
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        k_values=config.k_values,
        knn_k_candidates=config.knn_k_candidates,
        batch_size=config.similarity_query_batch_size,
    )

    geometry = evaluate_geometry(
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        idx_to_class=idx_to_class,
    )

    (
        probe_train_x,
        probe_val_x,
        probe_test_x,
        probe_mean,
        probe_std,
    ) = standardize_feature_splits(
        train_h,
        val_h,
        test_h,
    )

    linear_probe, probe_history, probe_best_epoch, probe_best_val = (
        train_linear_probe(
            probe_train_x,
            _bundle_array(
                train_bundle,
                "y",
            ).astype(np.int64),
            probe_val_x,
            _bundle_array(
                val_bundle,
                "y",
            ).astype(np.int64),
            num_classes=num_classes,
            device=device,
            config=config.linear_probe,
            random_seed=config.random_seed,
        )
    )
    linear_probe_metrics = evaluate_linear_probe(
        linear_probe,
        probe_test_x,
        test_y,
        device=device,
        batch_size=config.linear_probe.batch_size,
    )

    unsupervised = evaluate_unsupervised(
        test_bundle,
        kmeans_n_init=config.kmeans_n_init,
        kmeans_max_iter=config.kmeans_max_iter,
        silhouette_max_samples=config.silhouette_max_samples,
        random_seed=config.random_seed,
        idx_to_class=idx_to_class,
        verbose=config.verbose,
    )

    same_label_table = retrieval[
        "same_label_table"
    ]
    same_label_k1 = (
        same_label_table.loc[
            same_label_table["k"] == 1
        ]
    )
    same_label_macro_at_1 = (
        float(
            same_label_k1.iloc[0][
                "same_label_macro"
            ]
        )
        if len(same_label_k1) == 1
        else math.nan
    )
    same_label_micro_at_1 = (
        float(
            same_label_k1.iloc[0][
                "same_label_micro"
            ]
        )
        if len(same_label_k1) == 1
        else math.nan
    )

    retrieval_summary = retrieval[
        "retrieval_summary"
    ].iloc[0]
    unsupervised_summary = unsupervised[
        "summary"
    ].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "CNN_test_balanced_accuracy": cnn_metrics[
                    "balanced_accuracy"
                ],
                "CNN_test_accuracy": cnn_metrics[
                    "accuracy"
                ],
                "CNN_test_macro_f1": cnn_metrics[
                    "macro_f1"
                ],
                "kNN_selected_k": int(
                    retrieval[
                        "selected_knn_k"
                    ]
                ),
                "kNN_test_balanced_accuracy": retrieval[
                    "knn_metrics"
                ][
                    "balanced_accuracy"
                ],
                "kNN_test_accuracy": retrieval[
                    "knn_metrics"
                ][
                    "accuracy"
                ],
                "kNN_test_macro_f1": retrieval[
                    "knn_metrics"
                ][
                    "macro_f1"
                ],
                "prototype_test_balanced_accuracy": geometry[
                    "prototype_metrics"
                ][
                    "balanced_accuracy"
                ],
                "prototype_test_accuracy": geometry[
                    "prototype_metrics"
                ][
                    "accuracy"
                ],
                "prototype_test_macro_f1": geometry[
                    "prototype_metrics"
                ][
                    "macro_f1"
                ],
                "linear_probe_test_balanced_accuracy": linear_probe_metrics[
                    "balanced_accuracy"
                ],
                "linear_probe_test_accuracy": linear_probe_metrics[
                    "accuracy"
                ],
                "linear_probe_test_macro_f1": linear_probe_metrics[
                    "macro_f1"
                ],
                "retrieval_macro_mAP": float(
                    retrieval_summary[
                        "mAP_macro"
                    ]
                ),
                "retrieval_micro_mAP": float(
                    retrieval_summary[
                        "mAP_micro"
                    ]
                ),
                "SameLabel_macro_at_1": same_label_macro_at_1,
                "SameLabel_micro_at_1": same_label_micro_at_1,
                "D_intra_macro": geometry[
                    "D_intra_macro"
                ],
                "D_inter_centroids": geometry[
                    "D_inter_centroids"
                ],
                "D_inter_over_D_intra": geometry[
                    "D_inter_over_D_intra"
                ],
                "NMI_arithmetic": float(
                    unsupervised_summary[
                        "NMI_arithmetic"
                    ]
                ),
                "ARI": float(
                    unsupervised_summary[
                        "ARI"
                    ]
                ),
                "silhouette_macro": float(
                    unsupervised_summary[
                        "silhouette_macro"
                    ]
                ),
                "silhouette_micro": float(
                    unsupervised_summary[
                        "silhouette_micro"
                    ]
                ),
            }
        ]
    )

    if config.verbose:
        print(
            "Representation evaluation complete | "
            f"CNN BA={cnn_metrics['balanced_accuracy']:.4f} | "
            f"kNN BA={retrieval['knn_metrics']['balanced_accuracy']:.4f} | "
            f"Prototype BA={geometry['prototype_metrics']['balanced_accuracy']:.4f} | "
            f"Linear probe BA={linear_probe_metrics['balanced_accuracy']:.4f}"
        )

    return RepresentationEvaluationResult(
        summary=summary,
        cnn_metrics=cnn_metrics,
        selected_knn_k=int(
            retrieval[
                "selected_knn_k"
            ]
        ),
        knn_metrics=dict(
            retrieval[
                "knn_metrics"
            ]
        ),
        knn_validation_table=retrieval[
            "knn_validation_table"
        ],
        knn_prediction=np.asarray(
            retrieval[
                "knn_prediction"
            ],
            dtype=np.int64,
        ),
        retrieval_summary=retrieval[
            "retrieval_summary"
        ],
        same_label_table=retrieval[
            "same_label_table"
        ],
        average_precision=np.asarray(
            retrieval[
                "average_precision"
            ],
            dtype=np.float64,
        ),
        prototype_metrics=dict(
            geometry[
                "prototype_metrics"
            ]
        ),
        prototype_prediction=np.asarray(
            geometry[
                "prototype_prediction"
            ],
            dtype=np.int64,
        ),
        geometry_summary=geometry[
            "geometry_summary"
        ],
        intra_by_class=geometry[
            "intra_by_class"
        ],
        linear_probe_metrics=linear_probe_metrics,
        linear_probe_history=probe_history,
        linear_probe_best_epoch=probe_best_epoch,
        linear_probe_best_val_balanced_accuracy=probe_best_val,
        linear_probe_state_dict=copy.deepcopy(
            linear_probe.state_dict()
        ),
        linear_probe_feature_mean=probe_mean,
        linear_probe_feature_std=probe_std,
        unsupervised_summary=unsupervised[
            "summary"
        ],
        kmeans_assignment=np.asarray(
            unsupervised[
                "kmeans_assignment"
            ],
            dtype=np.int64,
        ),
        kmeans_centers=np.asarray(
            unsupervised[
                "kmeans_centers"
            ],
            dtype=np.float32,
        ),
        silhouette_values=np.asarray(
            unsupervised[
                "silhouette_values"
            ],
            dtype=np.float64,
        ),
        silhouette_indices=np.asarray(
            unsupervised[
                "silhouette_indices"
            ],
            dtype=np.int64,
        ),
    )


def evaluate_paired_preservation(
    reference_test: EmbeddingBundle | Mapping[str, Any],
    query_test: EmbeddingBundle | Mapping[str, Any],
    *,
    idx_to_class: Mapping[int, str] | None = None,
) -> PairedPreservationResult:
    """Evaluate one-to-one raw/reconstruction preservation for Experiment B."""
    validate_paired_bundles(
        reference_test,
        query_test,
    )

    similarity, distance = paired_embedding_cosine(
        reference_test,
        query_test,
    )

    y = _bundle_array(
        reference_test,
        "y",
    ).astype(np.int64)
    reference_pred = _bundle_array(
        reference_test,
        "cnn_pred",
    ).astype(np.int64)
    query_pred = _bundle_array(
        query_test,
        "cnn_pred",
    ).astype(np.int64)

    reference_correct = (
        reference_pred == y
    )
    query_correct = (
        query_pred == y
    )

    prediction_agreement = float(
        np.mean(
            reference_pred
            == query_pred
        )
    )

    transitions = np.select(
        [
            reference_correct & query_correct,
            reference_correct & ~query_correct,
            ~reference_correct & query_correct,
            ~reference_correct & ~query_correct,
        ],
        [
            "reference_correct -> query_correct",
            "reference_correct -> query_wrong",
            "reference_wrong -> query_correct",
            "reference_wrong -> query_wrong",
        ],
        default="unexpected",
    )

    transition_table = (
        pd.Series(
            transitions,
            name="transition",
        )
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="count")
    )
    transition_table[
        "fraction"
    ] = (
        transition_table[
            "count"
        ]
        / len(y)
    )

    summary = pd.DataFrame(
        [
            {
                "mean_embedding_cosine_similarity": float(
                    similarity.mean()
                ),
                "median_embedding_cosine_similarity": float(
                    np.median(
                        similarity
                    )
                ),
                "mean_embedding_cosine_distance": float(
                    distance.mean()
                ),
                "median_embedding_cosine_distance": float(
                    np.median(
                        distance
                    )
                ),
                "prediction_agreement": prediction_agreement,
            }
        ]
    )

    per_class_rows = []
    for class_index in np.unique(y):
        selected = (
            y == class_index
        )
        label = (
            str(class_index)
            if idx_to_class is None
            else str(
                idx_to_class.get(
                    int(class_index),
                    str(class_index),
                )
            )
        )
        per_class_rows.append(
            {
                "label_idx": int(
                    class_index
                ),
                "label": label,
                "samples": int(
                    selected.sum()
                ),
                "mean_embedding_cosine_similarity": float(
                    similarity[
                        selected
                    ].mean()
                ),
                "mean_embedding_cosine_distance": float(
                    distance[
                        selected
                    ].mean()
                ),
                "prediction_agreement": float(
                    np.mean(
                        reference_pred[
                            selected
                        ]
                        == query_pred[
                            selected
                        ]
                    )
                ),
            }
        )

    return PairedPreservationResult(
        summary=summary,
        transitions=transition_table,
        per_class=pd.DataFrame(
            per_class_rows
        ),
        cosine_similarity=similarity,
        cosine_distance=distance,
    )


def compare_representation_results(
    results: Mapping[
        str,
        RepresentationEvaluationResult,
    ],
) -> pd.DataFrame:
    """Combine A/B/C/D one-row summaries into a condition comparison table."""
    if not results:
        raise ValueError(
            "results must contain at least one condition"
        )

    columns = {}
    for name, result in results.items():
        columns[str(name)] = result.metric_series()

    comparison = pd.DataFrame(
        columns
    )
    comparison.index.name = "metric"
    return comparison
