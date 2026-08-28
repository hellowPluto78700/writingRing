from __future__ import annotations

import numpy as np


def streaming_prefix_logits(
    counts: np.ndarray,
    *,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
) -> np.ndarray:
    """Return exact prefix logits without constructing zero-padded prefix vectors.

    ``counts`` has shape ``[n_bins, width]``. The fitted sklearn pipeline is
    StandardScaler followed by multinomial LogisticRegression on a flattened
    zero-padded prefix. For a linear classifier,

        coef @ ((x - mean) / scale) + intercept

    can be rewritten as one all-zero baseline plus an incremental contribution
    from each observed bin. This implementation is therefore numerically
    equivalent to rebuilding the full prefix feature at every checkpoint, while
    requiring only the current bin and accumulated class logits online.
    """
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D [n_bins, width], got {counts.shape}")

    n_bins, width = counts.shape
    expected_dim = n_bins * width
    mean = np.asarray(scaler_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(scaler_scale, dtype=np.float64).reshape(-1)
    weights = np.asarray(coef, dtype=np.float64)
    bias = np.asarray(intercept, dtype=np.float64).reshape(-1)

    if mean.size != expected_dim or scale.size != expected_dim:
        raise ValueError(
            f"scaler dimension mismatch: expected {expected_dim}, "
            f"got mean={mean.size}, scale={scale.size}"
        )
    if weights.ndim != 2 or weights.shape[1] != expected_dim:
        raise ValueError(
            f"classifier dimension mismatch: expected (*, {expected_dim}), "
            f"got {weights.shape}"
        )
    if weights.shape[0] != bias.size:
        raise ValueError("classifier coefficient/intercept class dimensions differ")

    safe_scale = np.where(scale == 0.0, 1.0, scale)
    effective_weights = weights / safe_scale[None, :]
    zero_prefix_logits = bias - effective_weights @ mean
    bin_weights = effective_weights.reshape(weights.shape[0], n_bins, width)

    logits = np.empty((n_bins, weights.shape[0]), dtype=np.float64)
    accumulated = zero_prefix_logits.copy()
    for bin_index in range(n_bins):
        accumulated = accumulated + bin_weights[:, bin_index, :] @ counts[bin_index]
        logits[bin_index] = accumulated
    return logits
