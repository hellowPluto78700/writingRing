from __future__ import annotations

"""Representation and clustering metrics used by acceleration CNN experiments.

The numerical definitions follow the current evaluation notebook:
- cosine kNN and SameLabel@K
- retrieval mAP
- normalized class prototypes
- macro within-class cosine distance
- centroid between-class cosine distance
- KMeans, arithmetic NMI, ARI
- cosine silhouette

Notebook-global values such as ``SIMILARITY_QUERY_BATCH_SIZE`` and
``idx_to_class`` are explicit optional parameters here.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


try:
    from .training import classification_metrics
except ImportError:
    from training import classification_metrics


__all__ = [
    "macro_average_by_class",
    "confusion_matrix_numpy",
    "cosine_topk",
    "knn_predict_from_topk",
    "same_label_at_k",
    "retrieval_map",
    "select_knn_k",
    "class_centroids",
    "macro_intra_cosine_distance",
    "centroid_inter_cosine_distance",
    "prototype_classification",
    "squared_euclidean_matrix",
    "kmeans_plus_plus_init",
    "run_kmeans_once",
    "kmeans_best_of_n",
    "contingency_matrix_numpy",
    "normalized_mutual_information",
    "adjusted_rand_index",
    "balanced_subsample_indices",
    "cosine_silhouette",
]


def _validate_labels(
    labels: np.ndarray,
    *,
    expected_length: int | None = None,
    name: str = "labels",
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {values.shape}")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"{name} length={len(values)} != expected {expected_length}"
        )
    return values


def _validate_z(
    z: np.ndarray,
    *,
    name: str = "z",
) -> np.ndarray:
    values = np.asarray(z, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {values.shape}")
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return values


def macro_average_by_class(
    values: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)

    if values.ndim != 1 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError(
            "values and labels must be aligned one-dimensional arrays"
        )
    if len(values) == 0:
        raise ValueError("Cannot compute a macro average on zero samples")

    per_class = [
        float(values[labels == class_index].mean())
        for class_index in np.unique(labels)
    ]
    return float(np.mean(per_class))


def confusion_matrix_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    true = _validate_labels(y_true, name="y_true")
    pred = _validate_labels(
        y_pred,
        expected_length=len(true),
        name="y_pred",
    )

    if np.any(true < 0) or np.any(true >= num_classes):
        raise ValueError("y_true contains an out-of-range class index")
    if np.any(pred < 0) or np.any(pred >= num_classes):
        raise ValueError("y_pred contains an out-of-range class index")

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )
    np.add.at(matrix, (true, pred), 1)
    return matrix


def cosine_topk(
    query_z: np.ndarray,
    gallery_z: np.ndarray,
    *,
    max_k: int,
    query_ids: np.ndarray | None = None,
    gallery_ids: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k gallery indices and cosine scores for each query."""
    query = _validate_z(query_z, name="query_z")
    gallery = _validate_z(gallery_z, name="gallery_z")

    if query.shape[1] != gallery.shape[1]:
        raise ValueError(
            "query_z and gallery_z must have the same feature dimension"
        )
    if max_k <= 0 or max_k > len(gallery):
        raise ValueError(
            f"max_k must be in [1, {len(gallery)}], got {max_k}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if (query_ids is None) != (gallery_ids is None):
        raise ValueError(
            "query_ids and gallery_ids must be supplied together or omitted"
        )

    if query_ids is not None:
        query_ids = np.asarray(query_ids).astype(str)
        gallery_ids = np.asarray(gallery_ids).astype(str)
        if query_ids.shape != (len(query),):
            raise ValueError("query_ids must align with query_z")
        if gallery_ids.shape != (len(gallery),):
            raise ValueError("gallery_ids must align with gallery_z")

    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        scores = query[start:stop] @ gallery.T

        if query_ids is not None and gallery_ids is not None:
            duplicate = (
                query_ids[start:stop, None]
                == gallery_ids[None, :]
            )
            scores = scores.copy()
            scores[duplicate] = -np.inf

        candidate = np.argpartition(
            -scores,
            kth=max_k - 1,
            axis=1,
        )[:, :max_k]
        candidate_scores = np.take_along_axis(
            scores,
            candidate,
            axis=1,
        )
        order = np.argsort(
            -candidate_scores,
            axis=1,
            kind="stable",
        )
        indices = np.take_along_axis(
            candidate,
            order,
            axis=1,
        )
        sorted_scores = np.take_along_axis(
            candidate_scores,
            order,
            axis=1,
        )
        all_indices.append(
            indices.astype(np.int64, copy=False)
        )
        all_scores.append(
            sorted_scores.astype(np.float32, copy=False)
        )

    return (
        np.concatenate(all_indices),
        np.concatenate(all_scores),
    )


def knn_predict_from_topk(
    neighbour_indices: np.ndarray,
    neighbour_scores: np.ndarray,
    gallery_y: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Majority-vote kNN with cosine-score and class-index tie breaks."""
    indices = np.asarray(neighbour_indices, dtype=np.int64)
    scores = np.asarray(neighbour_scores, dtype=np.float32)
    gallery_y = _validate_labels(gallery_y, name="gallery_y")

    if indices.ndim != 2 or scores.shape != indices.shape:
        raise ValueError(
            "neighbour_indices/scores must be aligned 2-D matrices"
        )
    if k <= 0 or k > indices.shape[1]:
        raise ValueError(
            f"k must be in [1, {indices.shape[1]}], got {k}"
        )
    if np.any(indices < 0) or np.any(indices >= len(gallery_y)):
        raise ValueError("neighbour_indices contains an invalid gallery index")

    predictions = np.empty(
        len(indices),
        dtype=np.int64,
    )

    for query_index in range(len(indices)):
        labels = gallery_y[
            indices[query_index, :k]
        ]
        local_scores = scores[
            query_index,
            :k,
        ]

        unique, counts = np.unique(
            labels,
            return_counts=True,
        )
        max_count = counts.max()
        candidates = unique[counts == max_count]

        if len(candidates) == 1:
            predictions[query_index] = int(
                candidates[0]
            )
            continue

        score_sums = {
            int(label): float(
                local_scores[
                    labels == label
                ].sum()
            )
            for label in candidates
        }
        best_score = max(score_sums.values())

        predictions[query_index] = min(
            label
            for label, value in score_sums.items()
            if value == best_score
        )

    return predictions


def same_label_at_k(
    neighbour_indices: np.ndarray,
    *,
    query_y: np.ndarray,
    gallery_y: np.ndarray,
    k: int,
) -> tuple[float, float, np.ndarray]:
    """Return micro, macro, and per-query same-label fraction at K."""
    indices = np.asarray(
        neighbour_indices,
        dtype=np.int64,
    )
    query_y = _validate_labels(
        query_y,
        expected_length=len(indices),
        name="query_y",
    )
    gallery_y = _validate_labels(
        gallery_y,
        name="gallery_y",
    )

    if indices.ndim != 2:
        raise ValueError("neighbour_indices must be 2-D")
    if k <= 0 or k > indices.shape[1]:
        raise ValueError(
            f"k must be in [1, {indices.shape[1]}], got {k}"
        )

    relevant = (
        gallery_y[indices[:, :k]]
        == query_y[:, None]
    )
    per_query = relevant.mean(
        axis=1,
        dtype=np.float64,
    )
    return (
        float(per_query.mean()),
        macro_average_by_class(
            per_query,
            query_y,
        ),
        per_query,
    )


def retrieval_map(
    query_z: np.ndarray,
    gallery_z: np.ndarray,
    *,
    query_y: np.ndarray,
    gallery_y: np.ndarray,
    query_ids: np.ndarray | None = None,
    gallery_ids: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[float, float, np.ndarray]:
    """Compute micro/macro mAP using cosine ranking.

    The formula intentionally matches the current notebook. Current A/B/C
    protocols use disjoint query/gallery splits, so duplicate-ID suppression
    normally has no effect.
    """
    query = _validate_z(query_z, name="query_z")
    gallery = _validate_z(gallery_z, name="gallery_z")
    query_y = _validate_labels(
        query_y,
        expected_length=len(query),
        name="query_y",
    )
    gallery_y = _validate_labels(
        gallery_y,
        expected_length=len(gallery),
        name="gallery_y",
    )

    if query.shape[1] != gallery.shape[1]:
        raise ValueError(
            "query_z and gallery_z must have the same feature dimension"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (query_ids is None) != (gallery_ids is None):
        raise ValueError(
            "query_ids and gallery_ids must be supplied together or omitted"
        )

    if query_ids is not None:
        query_ids = np.asarray(query_ids).astype(str)
        gallery_ids = np.asarray(gallery_ids).astype(str)
        if query_ids.shape != (len(query),):
            raise ValueError("query_ids must align with query_z")
        if gallery_ids.shape != (len(gallery),):
            raise ValueError("gallery_ids must align with gallery_z")

    aps = np.empty(
        len(query),
        dtype=np.float64,
    )
    ranks = np.arange(
        1,
        len(gallery) + 1,
        dtype=np.float64,
    )

    for start in range(0, len(query), batch_size):
        stop = min(
            start + batch_size,
            len(query),
        )
        scores = (
            np.asarray(
                query[start:stop],
                dtype=np.float32,
            )
            @ np.asarray(
                gallery,
                dtype=np.float32,
            ).T
        )

        if query_ids is not None and gallery_ids is not None:
            duplicate = (
                query_ids[start:stop, None]
                == gallery_ids[None, :]
            )
            scores = scores.copy()
            scores[duplicate] = -np.inf

        ordering = np.argsort(
            -scores,
            axis=1,
            kind="stable",
        )

        for local_index, global_index in enumerate(
            range(start, stop)
        ):
            relevant_count = int(
                np.count_nonzero(
                    gallery_y
                    == query_y[global_index]
                )
            )
            if relevant_count <= 0:
                raise ValueError(
                    f"Query {global_index} has no same-label "
                    "item in the gallery"
                )

            relevance = (
                gallery_y[
                    ordering[local_index]
                ]
                == query_y[global_index]
            )
            cumulative = np.cumsum(
                relevance,
                dtype=np.float64,
            )
            precision = cumulative / ranks
            aps[global_index] = float(
                (precision * relevance).sum()
                / relevant_count
            )

    return (
        float(aps.mean()),
        macro_average_by_class(
            aps,
            query_y,
        ),
        aps,
    )


def select_knn_k(
    *,
    val_z: np.ndarray,
    val_y: np.ndarray,
    train_z: np.ndarray,
    train_y: np.ndarray,
    candidates: Sequence[int],
    batch_size: int = 128,
) -> tuple[int, pd.DataFrame]:
    """Select K by validation balanced accuracy, then smaller K on ties."""
    candidates = sorted(
        set(int(value) for value in candidates)
    )
    if not candidates or any(k <= 0 for k in candidates):
        raise ValueError(
            "candidates must contain positive integer K values"
        )
    if max(candidates) > len(train_z):
        raise ValueError(
            f"Largest candidate K={max(candidates)} exceeds "
            f"gallery size={len(train_z)}"
        )

    indices, scores = cosine_topk(
        val_z,
        train_z,
        max_k=max(candidates),
        batch_size=batch_size,
    )

    rows: list[dict[str, float | int]] = []
    for k in candidates:
        prediction = knn_predict_from_topk(
            indices,
            scores,
            np.asarray(train_y),
            k=k,
        )
        metrics = classification_metrics(
            np.asarray(val_y).tolist(),
            prediction.tolist(),
        )
        rows.append(
            {
                "k": k,
                **metrics,
            }
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("k")
        .reset_index(drop=True)
    )
    best_row = table.sort_values(
        ["balanced_accuracy", "k"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]

    return int(best_row["k"]), table


def class_centroids(
    z: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return represented class IDs and L2-normalized mean directions."""
    z = _validate_z(z)
    y = _validate_labels(
        y,
        expected_length=len(z),
        name="y",
    )

    classes = np.unique(y)
    centroids: list[np.ndarray] = []

    for class_index in classes:
        centroid = z[
            y == class_index
        ].mean(
            axis=0,
            dtype=np.float64,
        )
        norm = float(
            np.linalg.norm(centroid)
        )
        if not math.isfinite(norm) or norm <= 0:
            raise FloatingPointError(
                f"Invalid centroid for class {class_index}"
            )
        centroids.append(
            (centroid / norm).astype(np.float32)
        )

    return (
        classes.astype(np.int64),
        np.stack(centroids),
    )


def _label_name(
    class_index: int,
    idx_to_class: Mapping[int, str] | None,
) -> str:
    if idx_to_class is None:
        return str(class_index)
    return str(
        idx_to_class.get(
            int(class_index),
            str(class_index),
        )
    )


def macro_intra_cosine_distance(
    z: np.ndarray,
    y: np.ndarray,
    *,
    idx_to_class: Mapping[int, str] | None = None,
    verbose: bool = False,
) -> tuple[float, pd.DataFrame]:
    """Macro mean within-class pairwise cosine distance.

    Singleton classes are omitted because no within-class pair exists.
    """
    z = _validate_z(z)
    y = _validate_labels(
        y,
        expected_length=len(z),
        name="y",
    )

    rows: list[
        dict[str, float | int | str]
    ] = []
    skipped: list[int] = []

    for class_index in np.unique(y):
        class_z = np.asarray(
            z[y == class_index],
            dtype=np.float64,
        )
        n = len(class_z)

        if n < 2:
            skipped.append(
                int(class_index)
            )
            continue

        vector_sum = class_z.sum(axis=0)

        # z is L2-normalized. This is the mean cosine similarity over
        # unordered within-class pairs, identical to the notebook formula.
        mean_pair_cosine = (
            float(vector_sum @ vector_sum)
            - n
        ) / (n * (n - 1))

        rows.append(
            {
                "label_idx": int(class_index),
                "label": _label_name(
                    int(class_index),
                    idx_to_class,
                ),
                "test_samples": int(n),
                "intra_cosine_distance": float(
                    1.0 - mean_pair_cosine
                ),
            }
        )

    if not rows:
        raise ValueError(
            "No class has at least two samples; "
            "D_intra cannot be computed"
        )

    table = (
        pd.DataFrame(rows)
        .sort_values("label_idx")
        .reset_index(drop=True)
    )
    d_intra = float(
        table[
            "intra_cosine_distance"
        ].mean()
    )

    if verbose:
        print(
            f"D_intra computed from {len(table)} eligible classes; "
            f"{len(skipped)} singleton class(es) skipped."
        )
        if skipped:
            print(
                "Skipped singleton class indices:",
                skipped,
            )
            print(
                "Skipped singleton labels:",
                [
                    _label_name(
                        class_index,
                        idx_to_class,
                    )
                    for class_index in skipped
                ],
            )

    return d_intra, table


def centroid_inter_cosine_distance(
    z: np.ndarray,
    y: np.ndarray,
) -> float:
    """Mean cosine distance over unordered class-centroid pairs."""
    _, centroids = class_centroids(
        z,
        y,
    )
    if len(centroids) < 2:
        raise ValueError(
            "D_inter requires at least two represented classes"
        )

    distances = (
        1.0
        - centroids @ centroids.T
    )
    upper = distances[
        np.triu_indices(
            len(centroids),
            k=1,
        )
    ]
    return float(
        upper.mean()
    )


def prototype_classification(
    *,
    train_z: np.ndarray,
    train_y: np.ndarray,
    test_z: np.ndarray,
    test_y: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Classify query embeddings by nearest normalized train centroid."""
    classes, centroids = class_centroids(
        train_z,
        train_y,
    )
    test_z = _validate_z(
        test_z,
        name="test_z",
    )
    test_y = _validate_labels(
        test_y,
        expected_length=len(test_z),
        name="test_y",
    )

    scores = (
        np.asarray(
            test_z,
            dtype=np.float32,
        )
        @ centroids.T
    )
    predictions = classes[
        scores.argmax(axis=1)
    ]

    return (
        classification_metrics(
            test_y.tolist(),
            predictions.tolist(),
        ),
        predictions.astype(
            np.int64,
            copy=False,
        ),
    )


def squared_euclidean_matrix(
    x: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float32,
    )
    centers = np.asarray(
        centers,
        dtype=np.float32,
    )
    if x.ndim != 2 or centers.ndim != 2:
        raise ValueError("x and centers must be 2-D")
    if x.shape[1] != centers.shape[1]:
        raise ValueError(
            "x and centers must have the same feature dimension"
        )

    x_sq = np.square(x).sum(
        axis=1,
        keepdims=True,
    )
    c_sq = np.square(centers).sum(
        axis=1,
        keepdims=True,
    ).T
    return np.maximum(
        x_sq
        + c_sq
        - 2.0 * x @ centers.T,
        0.0,
    )


def kmeans_plus_plus_init(
    x: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    x = _validate_z(
        x,
        name="x",
    )
    if k <= 0 or k > len(x):
        raise ValueError(
            f"k must be in [1, {len(x)}]"
        )

    centers = np.empty(
        (k, x.shape[1]),
        dtype=np.float32,
    )
    first = int(
        rng.integers(len(x))
    )
    centers[0] = x[first]

    closest = squared_euclidean_matrix(
        x,
        centers[:1],
    )[:, 0]

    for center_index in range(1, k):
        total = float(
            closest.sum()
        )
        if (
            not math.isfinite(total)
            or total <= 0
        ):
            chosen = int(
                rng.integers(len(x))
            )
        else:
            chosen = int(
                rng.choice(
                    len(x),
                    p=closest / total,
                )
            )

        centers[
            center_index
        ] = x[chosen]

        new_distance = squared_euclidean_matrix(
            x,
            centers[
                center_index : center_index + 1
            ],
        )[:, 0]

        closest = np.minimum(
            closest,
            new_distance,
        )

    return centers


def run_kmeans_once(
    x: np.ndarray,
    *,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    if max_iter <= 0:
        raise ValueError(
            "max_iter must be positive"
        )

    x = _validate_z(
        x,
        name="x",
    )
    centers = kmeans_plus_plus_init(
        x,
        k,
        rng=rng,
    )
    previous_assignment: (
        np.ndarray | None
    ) = None

    for _ in range(max_iter):
        distances = squared_euclidean_matrix(
            x,
            centers,
        )
        assignment = distances.argmin(
            axis=1,
        ).astype(np.int64)

        if (
            previous_assignment is not None
            and np.array_equal(
                assignment,
                previous_assignment,
            )
        ):
            break

        previous_assignment = assignment.copy()

        nearest_distance = distances[
            np.arange(len(x)),
            assignment,
        ]
        new_centers = np.empty_like(
            centers
        )

        for cluster_index in range(k):
            members = x[
                assignment
                == cluster_index
            ]

            if len(members) == 0:
                replacement = int(
                    np.argmax(
                        nearest_distance
                    )
                )
                new_centers[
                    cluster_index
                ] = x[replacement]
                nearest_distance[
                    replacement
                ] = -np.inf
            else:
                new_centers[
                    cluster_index
                ] = members.mean(axis=0)

        centers = new_centers

    final_distances = squared_euclidean_matrix(
        x,
        centers,
    )
    assignment = final_distances.argmin(
        axis=1,
    ).astype(np.int64)
    inertia = float(
        final_distances[
            np.arange(len(x)),
            assignment,
        ].sum()
    )

    return (
        assignment,
        centers,
        inertia,
    )


def kmeans_best_of_n(
    x: np.ndarray,
    *,
    k: int,
    n_init: int,
    max_iter: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    if n_init <= 0:
        raise ValueError(
            "n_init must be positive"
        )

    best_assignment: (
        np.ndarray | None
    ) = None
    best_centers: (
        np.ndarray | None
    ) = None
    best_inertia = math.inf

    for init_index in range(n_init):
        rng = np.random.default_rng(
            seed + init_index
        )
        assignment, centers, inertia = (
            run_kmeans_once(
                np.asarray(
                    x,
                    dtype=np.float32,
                ),
                k=k,
                rng=rng,
                max_iter=max_iter,
            )
        )

        if inertia < best_inertia:
            best_assignment = assignment
            best_centers = centers
            best_inertia = inertia

    if (
        best_assignment is None
        or best_centers is None
    ):
        raise RuntimeError(
            "KMeans failed to produce a solution"
        )

    return (
        best_assignment,
        best_centers,
        best_inertia,
    )


def contingency_matrix_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    true = _validate_labels(
        y_true,
        name="y_true",
    )
    pred = _validate_labels(
        y_pred,
        expected_length=len(true),
        name="y_pred",
    )

    true_values, true_inverse = np.unique(
        true,
        return_inverse=True,
    )
    pred_values, pred_inverse = np.unique(
        pred,
        return_inverse=True,
    )

    matrix = np.zeros(
        (
            len(true_values),
            len(pred_values),
        ),
        dtype=np.int64,
    )
    np.add.at(
        matrix,
        (
            true_inverse,
            pred_inverse,
        ),
        1,
    )
    return matrix


def normalized_mutual_information(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Arithmetic-mean normalized mutual information used in the notebook."""
    contingency = contingency_matrix_numpy(
        y_true,
        y_pred,
    ).astype(np.float64)

    n = contingency.sum()
    if n <= 0:
        raise ValueError(
            "Cannot compute NMI on zero samples"
        )

    row = contingency.sum(
        axis=1,
        keepdims=True,
    )
    col = contingency.sum(
        axis=0,
        keepdims=True,
    )
    nonzero = contingency > 0
    expected = row @ col / n

    mutual_information = float(
        np.sum(
            (
                contingency[nonzero]
                / n
            )
            * np.log(
                contingency[nonzero]
                / expected[nonzero]
            )
        )
    )

    row_prob = row.ravel() / n
    col_prob = col.ravel() / n

    h_true = float(
        -np.sum(
            row_prob[
                row_prob > 0
            ]
            * np.log(
                row_prob[
                    row_prob > 0
                ]
            )
        )
    )
    h_pred = float(
        -np.sum(
            col_prob[
                col_prob > 0
            ]
            * np.log(
                col_prob[
                    col_prob > 0
                ]
            )
        )
    )

    denominator = (
        h_true
        + h_pred
    )
    return (
        1.0
        if denominator == 0
        else float(
            2.0
            * mutual_information
            / denominator
        )
    )


def _comb2(
    values: np.ndarray | float,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    return (
        values
        * (values - 1.0)
        / 2.0
    )


def adjusted_rand_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    contingency = contingency_matrix_numpy(
        y_true,
        y_pred,
    ).astype(np.float64)

    n = contingency.sum()
    if n < 2:
        return 1.0

    sum_comb = float(
        _comb2(
            contingency
        ).sum()
    )
    row_comb = float(
        _comb2(
            contingency.sum(axis=1)
        ).sum()
    )
    col_comb = float(
        _comb2(
            contingency.sum(axis=0)
        ).sum()
    )
    total_comb = float(
        _comb2(n)
    )

    expected = (
        row_comb
        * col_comb
        / total_comb
    )
    maximum = 0.5 * (
        row_comb
        + col_comb
    )
    denominator = (
        maximum
        - expected
    )

    if denominator == 0:
        return (
            1.0
            if sum_comb == maximum
            else 0.0
        )

    return float(
        (
            sum_comb
            - expected
        )
        / denominator
    )


def balanced_subsample_indices(
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional"
        )
    if max_samples <= 0:
        raise ValueError(
            "max_samples must be positive"
        )

    if len(labels) <= max_samples:
        return np.arange(
            len(labels),
            dtype=np.int64,
        )

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    quota = max(
        1,
        max_samples // len(classes),
    )

    selected: list[int] = []
    remaining: list[int] = []

    for class_index in classes:
        indices = np.flatnonzero(
            labels == class_index
        )
        shuffled = rng.permutation(
            indices
        )

        selected.extend(
            shuffled[
                : min(
                    quota,
                    len(shuffled),
                )
            ].tolist()
        )
        remaining.extend(
            shuffled[
                min(
                    quota,
                    len(shuffled),
                ) :
            ].tolist()
        )

    slots = (
        max_samples
        - len(selected)
    )

    if slots > 0 and remaining:
        selected.extend(
            rng.choice(
                remaining,
                size=min(
                    slots,
                    len(remaining),
                ),
                replace=False,
            ).tolist()
        )

    return np.asarray(
        sorted(
            selected[:max_samples]
        ),
        dtype=np.int64,
    )


def cosine_silhouette(
    z: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 12345,
    idx_to_class: Mapping[int, str] | None = None,
    verbose: bool = False,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Compute micro/macro cosine silhouette on L2-normalized embeddings."""
    z = np.asarray(
        z,
        dtype=np.float64,
    )
    y = np.asarray(y)

    if z.ndim != 2:
        raise ValueError(
            f"z must be 2-D, got {z.shape}"
        )
    if y.ndim != 1 or len(y) != len(z):
        raise ValueError(
            "y must be 1-D and aligned with z"
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError(
            "max_samples must be positive or None"
        )

    rng = np.random.default_rng(seed)

    if (
        max_samples is not None
        and len(z) > max_samples
    ):
        indices = np.sort(
            rng.choice(
                len(z),
                size=max_samples,
                replace=False,
            )
        )
    else:
        indices = np.arange(
            len(z)
        )

    sample_z = z[indices]
    sample_y = y[indices]

    similarity = (
        sample_z
        @ sample_z.T
    )
    distance = (
        1.0
        - similarity
    )
    distance = np.clip(
        distance,
        0.0,
        2.0,
    )
    np.fill_diagonal(
        distance,
        0.0,
    )

    classes, counts = np.unique(
        sample_y,
        return_counts=True,
    )
    class_counts = dict(
        zip(
            classes.tolist(),
            counts.tolist(),
        )
    )

    singleton_classes = {
        int(class_index)
        for class_index, count
        in class_counts.items()
        if count < 2
    }

    silhouette_values: list[float] = []
    silhouette_indices: list[int] = []
    silhouette_labels: list[int] = []

    all_classes = np.unique(
        sample_y
    )

    for row_index, class_index in enumerate(
        sample_y
    ):
        if class_counts[class_index] < 2:
            continue

        same = (
            sample_y
            == class_index
        )
        same[row_index] = False
        a = float(
            distance[
                row_index,
                same,
            ].mean()
        )

        b = math.inf
        for other_class in all_classes:
            if other_class == class_index:
                continue

            other = (
                sample_y
                == other_class
            )
            if not other.any():
                continue

            mean_distance = float(
                distance[
                    row_index,
                    other,
                ].mean()
            )
            b = min(
                b,
                mean_distance,
            )

        if not math.isfinite(b):
            raise ValueError(
                "Silhouette requires at least two represented classes"
            )

        denominator = max(
            a,
            b,
        )
        s = (
            0.0
            if denominator <= 1e-12
            else (b - a) / denominator
        )

        silhouette_values.append(
            float(s)
        )
        silhouette_indices.append(
            int(indices[row_index])
        )
        silhouette_labels.append(
            int(class_index)
        )

    if not silhouette_values:
        raise ValueError(
            "No eligible samples remain for silhouette evaluation"
        )

    silhouette_values_array = np.asarray(
        silhouette_values,
        dtype=np.float64,
    )
    silhouette_indices_array = np.asarray(
        silhouette_indices,
        dtype=np.int64,
    )
    silhouette_labels_array = np.asarray(
        silhouette_labels,
        dtype=np.int64,
    )

    silhouette_micro = float(
        silhouette_values_array.mean()
    )
    silhouette_macro = float(
        np.mean(
            [
                silhouette_values_array[
                    silhouette_labels_array
                    == class_index
                ].mean()
                for class_index
                in np.unique(
                    silhouette_labels_array
                )
            ]
        )
    )

    if verbose:
        print(
            "Silhouette computed from "
            f"{len(silhouette_values_array)} eligible samples "
            f"across {len(np.unique(silhouette_labels_array))} classes."
        )
        if singleton_classes:
            print(
                f"Skipped {len(singleton_classes)} singleton "
                "class(es) as silhouette queries."
            )
            print(
                "Skipped singleton class indices:",
                sorted(singleton_classes),
            )
            print(
                "Skipped singleton labels:",
                [
                    _label_name(
                        class_index,
                        idx_to_class,
                    )
                    for class_index
                    in sorted(singleton_classes)
                ],
            )

    return (
        silhouette_micro,
        silhouette_macro,
        silhouette_values_array,
        silhouette_indices_array,
    )
