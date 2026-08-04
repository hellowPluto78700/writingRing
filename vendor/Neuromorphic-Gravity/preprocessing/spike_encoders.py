"""
spike_encoders.py
-----------------
Spike encoding methods for 3D acceleration time series.

Each function takes a 3D acceleration time series of shape (T, 3) and returns
a spike array of shape (T, 3).  Most encoders return polarity values in
{-1, 0, 1}; the two exceptions are documented in their respective docstrings.

This module exposes every encoding algorithm available in the `spikify`
library (innuce-spikify).  Encoders that internally produce only {0, 1}
(i.e. they require a non-negative signal) are applied independently to the
positive and negative parts of each channel so that the output polarity
convention {-1, 0, +1} is preserved throughout.

Available encoders
------------------
Temporal – Contrast
  threshold_encoding             : Threshold-Based Representation (TBR)
  step_forward_encoding          : step-forward / delta-modulation
  moving_window_encoding         : rolling-window mean comparison
  zero_cross_step_forward_encoding : zero-crossing step-forward (ZCSF)

Temporal – Global Referenced
  time_to_first_spike_encoding   : exponential-decay latency encoding
  phase_encoding                 : phase-angle quantisation (returns {0, 1})

Temporal – Deconvolution
  bsa_encoding                   : Ben Spiker Algorithm (BSA), Hanning window
  hough_spiker_encoding          : Hough Spiker Algorithm (HSA)
  modified_hough_spiker_encoding : Modified Hough Spiker Algorithm (MHSA)

Temporal – Latency
  burst_encoding                 : inter-spike-interval burst encoding

Rate
  poisson_rate_encoding          : Poisson rate encoding
"""

import numpy as np
from spikify.encoding.temporal.contrast import (
    threshold_based_representation,
    step_forward,
    moving_window,
    zero_cross_step_forward,
)
from spikify.encoding.temporal.global_referenced import (
    time_to_first_spike,
    phase_encoding as _phase_encoding,
)
from spikify.encoding.temporal.deconvolution import (
    bens_spiker,
    hough_spiker,
    modified_hough_spiker,
)
from spikify.encoding.temporal.latency import burst_encoding as _burst_encoding
from spikify.encoding.rate import poisson_rate
from spikify.filtering import FilterBank


def threshold_encoding(
    acc: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Threshold-Based Representation (TBR) spike encoding.

    Delegates to :func:`spikify.encoding.temporal.contrast.threshold_based_representation`.

    Computes the sample-to-sample variation for each axis and emits a positive
    spike (+1) or negative spike (-1) when the variation exceeds a threshold
    defined as ``mean(variation) ± threshold * std(variation)``.  The
    ``threshold`` parameter acts as the scaling factor ``γ`` of the
    noise-reduction threshold.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    threshold:
        Factor ``γ`` that scales the standard deviation of the signal
        variations to derive the firing threshold.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    spikes = threshold_based_representation(acc, threshold)
    return spikes.astype(np.int8)


def step_forward_encoding(
    acc: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Step-forward (delta-modulation) spike encoding.

    Delegates to :func:`spikify.encoding.temporal.contrast.step_forward`.

    Maintains a dynamically updated baseline per channel.  A positive spike
    (+1) is emitted when the signal exceeds ``baseline + threshold``, and the
    baseline advances by ``threshold``.  Likewise a negative spike (-1) is
    emitted when the signal falls below ``baseline - threshold``.

    This is also known as *send-on-delta* or *level-crossing* encoding.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    threshold:
        Step size that must be exceeded to trigger a spike.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    spikes = step_forward(acc, threshold)
    return spikes.astype(np.int8)


def moving_window_encoding(
    acc: np.ndarray,
    window_size: int = 16,
    threshold: float = 1.0,
) -> np.ndarray:
    """Moving-window spike encoding.

    Delegates to :func:`spikify.encoding.temporal.contrast.moving_window`.

    At each time step the signal is compared against the mean of the preceding
    ``window_size`` samples.  A positive spike (+1) is emitted when the signal
    exceeds the window mean by more than a signal-derived threshold, and a
    negative spike (-1) when it falls below by more than that threshold.  The
    threshold is computed internally by the spikify library as the mean
    absolute variation of the signal; the ``threshold`` parameter is accepted
    for API compatibility but is not forwarded to the underlying encoder.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    window_size:
        Number of past samples used to compute the rolling mean (base).
    threshold:
        Unused.  Kept for API compatibility; the spikify library derives the
        firing threshold automatically from the signal's mean absolute
        variation.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    spikes = moving_window(acc, window_size)
    return spikes.astype(np.int8)


def time_to_first_spike_encoding(
    acc: np.ndarray,
    window_size: int = 16,
    max_delay: int = None,
) -> np.ndarray:
    """Time-to-First-Spike (TTFS) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.global_referenced.time_to_first_spike`.

    The signal is segmented into non-overlapping windows of ``window_size``
    samples.  Within each window a single spike fires at the time step where
    the signal first crosses an exponentially decaying threshold; larger signal
    amplitudes produce earlier spikes.  Polarity (+1 / -1) is preserved by
    encoding the positive and negative parts of the signal independently and
    resolving any coincident spikes by the dominant signal polarity.

    Any trailing samples that do not fill a complete window are silenced.  The
    ``max_delay`` parameter is accepted for API compatibility but is not used
    by the underlying spikify encoder.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    window_size:
        Number of samples per encoding window (``interval`` in spikify).
        If ``T`` is not divisible by ``window_size``, trailing samples that
        do not fill a complete window are discarded.
    max_delay:
        Unused.  Kept for API compatibility.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    T, D = acc.shape

    # Truncate to the largest multiple of window_size
    T_trunc = (T // window_size) * window_size
    acc_trunc = acc[:T_trunc]

    # Encode positive and negative parts separately to preserve polarity
    pos_spikes = time_to_first_spike(np.maximum(acc_trunc, 0.0), window_size)
    neg_spikes = time_to_first_spike(np.maximum(-acc_trunc, 0.0), window_size)

    # Resolve conflicts (both fire at the same step) by signal polarity
    conflict = (pos_spikes == 1) & (neg_spikes == 1)
    if conflict.any():
        pos_spikes = pos_spikes.copy()
        neg_spikes = neg_spikes.copy()
        pos_spikes[conflict & (acc_trunc < 0)] = 0
        neg_spikes[conflict & (acc_trunc >= 0)] = 0

    spikes = np.zeros((T, D), dtype=np.int8)
    spikes[:T_trunc] = (pos_spikes - neg_spikes).astype(np.int8)
    return spikes


def bsa_encoding(
    acc: np.ndarray,
    filter_order: int = 11,
    threshold: float = 0.1,
) -> np.ndarray:
    """Ben Spiker Algorithm (BSA) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.deconvolution.bens_spiker` using a
    Hanning window filter.

    BSA encodes an analog signal as a sparse spike train by greedily matching
    the signal with copies of a FIR filter impulse response
    (Schrauwen & Van Campenhout, 2003).  At each time step a spike is fired if
    subtracting the filter's response from the running residual reduces the
    cumulative absolute reconstruction error by more than *threshold*.

    The algorithm is applied independently to the positive and negative parts
    of each axis so that the output polarity convention ``{-1, 0, +1}`` is
    preserved:

    * A **positive spike** (+1) is produced when the positive component of the
      signal is better explained by the filter.
    * A **negative spike** (-1) is produced when the negative component is.
    * When both passes fire at the same step, the polarity with the larger
      absolute signal value wins.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    filter_order:
        Number of taps in the Hanning FIR filter window.  Larger values give
        a smoother filter with a lower effective cut-off frequency.  Must be
        >= 2.  Default: 11.
    threshold:
        Decision threshold.  Higher values produce fewer spikes (sparser
        encoding); lower values produce more spikes (finer reconstruction).
        Default: 0.1.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.

    References
    ----------
    Schrauwen, B., & Van Campenhout, J. (2003). BSA, a fast and accurate spike
    train encoding scheme. *Proceedings of the International Joint Conference
    on Neural Networks*, 4, 2825–2830.
    """
    if filter_order < 2:
        raise ValueError("filter_order must be >= 2")

    acc = np.asarray(acc, dtype=float)
    return _bipolar_per_channel(
        lambda sig, filter_order, threshold: bens_spiker(
            sig, filter_order, threshold, window_type="hann"
        ),
        acc,
        filter_order=filter_order,
        threshold=threshold,
        normalize_input=True,
        normalize_peak=4.0,
    )


def _bipolar_per_channel(
    fn,
    acc: np.ndarray,
    normalize_input: bool = False,
    normalize_peak: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """Helper: apply a unipolar {0,1} encoder to +/- parts and combine.

    For each channel the encoder is applied once to the non-negative part of
    the signal and once to the non-positive part (sign-flipped to make it
    non-negative).  A positive spike (+1) is emitted for the positive pass and
    a negative spike (-1) for the negative pass.  Coincident firings are
    resolved by keeping the polarity that matches the dominant signal value at
    that sample.

    Parameters
    ----------
    fn:
        A callable that accepts a 1-D numpy array and returns a 1-D {0,1}
        int8 array of the same length.
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    **kwargs:
        Extra keyword arguments forwarded to ``fn``.

    Returns
    -------
    spikes : np.ndarray of shape ``(T, 3)``, dtype int8, values in {-1, 0, 1}.
    """
    T, D = acc.shape
    spikes = np.zeros((T, D), dtype=np.int8)
    for d in range(D):
        channel = acc[:, d]
        pos_sig = np.maximum(channel, 0.0)
        neg_sig = np.maximum(-channel, 0.0)

        if normalize_input:
            # Deconvolution encoders compare against FIR kernels with unit-ish
            # taps, so low-amplitude inputs can become effectively silent.
            # Rescale each polarity channel to a configurable peak.
            pos_max = np.max(pos_sig)
            neg_max = np.max(neg_sig)
            if pos_max > 0:
                pos_sig = (pos_sig / pos_max) * normalize_peak
            if neg_max > 0:
                neg_sig = (neg_sig / neg_max) * normalize_peak

        pos = fn(pos_sig, **kwargs)
        neg = fn(neg_sig, **kwargs)
        conflict = (pos == 1) & (neg == 1)
        if conflict.any():
            pos = pos.copy()
            neg = neg.copy()
            pos[conflict & (channel < 0)] = 0
            neg[conflict & (channel >= 0)] = 0
        spikes[:, d] = pos - neg
    return spikes


def zero_cross_step_forward_encoding(
    acc: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Zero-Crossing Step-Forward (ZCSF) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.contrast.zero_cross_step_forward`.

    ZCSF applies half-wave rectification before running the step-forward
    algorithm, so only crossings of the positive half are detected natively.
    Bipolar ``{-1, 0, +1}`` output is obtained here by encoding the positive
    and negative parts of the signal independently and combining the results.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    threshold:
        Step-size threshold.  A spike fires when the (rectified) signal
        exceeds the running baseline by this amount.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    return _bipolar_per_channel(
        lambda sig, threshold: zero_cross_step_forward(sig, threshold),
        acc,
        threshold=threshold,
    )


def hough_spiker_encoding(
    acc: np.ndarray,
    filter_order: int = 11,
) -> np.ndarray:
    """Hough Spiker Algorithm (HSA) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.deconvolution.hough_spiker`.

    HSA detects spikes by progressive subtraction: a spike fires at position
    *t* when the signal value exceeds the convolution of the signal with a
    boxcar filter.  The algorithm is applied to the positive and negative parts
    of each channel independently to produce bipolar ``{-1, 0, +1}`` output.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    filter_order:
        Length of the boxcar filter window.  Must be < ``T``.  Default: 11.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")
    acc = np.asarray(acc, dtype=float)
    return _bipolar_per_channel(
        lambda sig, filter_order: hough_spiker(
            sig, filter_order, window_type="hann"
        ),
        acc,
        filter_order=filter_order,
        normalize_input=True,
        normalize_peak=4.0,
    )


def modified_hough_spiker_encoding(
    acc: np.ndarray,
    filter_order: int = 11,
    threshold: float = 0.1,
) -> np.ndarray:
    """Modified Hough Spiker Algorithm (MHSA) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.deconvolution.modified_hough_spiker`.

    MHSA extends HSA with a threshold-based error-accumulation mechanism: a
    spike is fired only when the accumulated reconstruction error stays within
    ``threshold``.  The algorithm is applied to the positive and negative parts
    of each channel independently.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    filter_order:
        Length of the boxcar filter window.  Must be < ``T``.  Default: 11.
    threshold:
        Error-accumulation threshold.  Smaller values yield fewer spikes.
        Default: 0.1.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    if filter_order < 1:
        raise ValueError("filter_order must be >= 1")
    acc = np.asarray(acc, dtype=float)
    return _bipolar_per_channel(
        lambda sig, filter_order, threshold: modified_hough_spiker(
            sig, filter_order, threshold, window_type="hann"
        ),
        acc,
        filter_order=filter_order,
        threshold=threshold,
        normalize_input=True,
        normalize_peak=4.0,
    )


def burst_encoding(
    acc: np.ndarray,
    n_max: int = 4,
    t_min: int = 2,
    t_max: int = 6,
) -> np.ndarray:
    """Burst (latency) spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.latency.burst_encoding`.

    Encodes each channel as a burst of up to ``n_max`` spikes with
    inter-spike intervals drawn between ``t_min`` and ``t_max`` samples.
    The number of spikes in a burst is inversely proportional to the signal
    amplitude.  Bipolar ``{-1, 0, +1}`` output is obtained by encoding the
    positive and negative parts of each channel independently.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    n_max:
        Maximum number of spikes per burst.  Default: 4.
    t_min:
        Minimum inter-spike interval in samples.  Default: 2.
    t_max:
        Maximum inter-spike interval in samples.  Default: 6.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    T, D = acc.shape
    spikes = np.zeros((T, D), dtype=np.int8)

    # Avoid encoding the full recording as one burst block.
    # Chunking keeps local amplitude dynamics and yields non-dead streams.
    chunk_len = 64

    for d in range(D):
        channel = acc[:, d]
        pos = np.zeros(T, dtype=np.int8)
        neg = np.zeros(T, dtype=np.int8)

        for start in range(0, T, chunk_len):
            stop = min(start + chunk_len, T)
            seg_len = stop - start
            pos_seg = np.maximum(channel[start:stop], 0.0)
            neg_seg = np.maximum(-channel[start:stop], 0.0)

            pos[start:stop] = _burst_encoding(pos_seg, n_max, t_min, t_max, seg_len).astype(np.int8)
            neg[start:stop] = _burst_encoding(neg_seg, n_max, t_min, t_max, seg_len).astype(np.int8)

        conflict = (pos == 1) & (neg == 1)
        if conflict.any():
            pos = pos.copy()
            neg = neg.copy()
            pos[conflict & (channel < 0)] = 0
            neg[conflict & (channel >= 0)] = 0

        spikes[:, d] = pos - neg

    return spikes


def poisson_rate_encoding(
    acc: np.ndarray,
    interval_length: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Poisson rate spike encoding.

    Delegates to :func:`spikify.encoding.rate.poisson_rate`.

    Generates a spike train where the probability of a spike in each interval
    is proportional to the normalised signal amplitude.  Bipolar
    ``{-1, 0, +1}`` output is obtained by encoding the positive and negative
    parts of each channel independently.

    ``T`` must be divisible by ``interval_length``; any remainder is
    discarded.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    interval_length:
        Number of samples per Poisson interval.  ``T`` must be divisible by
        this value.  Default: 4.
    seed:
        Random seed for reproducibility.  Default: 0.

    Returns
    -------
    spikes:
        Integer array of shape ``(T, 3)`` with values in ``{-1, 0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    T, D = acc.shape
    T_trunc = (T // interval_length) * interval_length
    acc_trunc = acc[:T_trunc]

    pos = poisson_rate(np.maximum(acc_trunc, 0.0), interval_length, seed)
    neg = poisson_rate(np.maximum(-acc_trunc, 0.0), interval_length, seed + 1)

    conflict = (pos == 1) & (neg == 1)
    if conflict.any():
        pos = pos.copy()
        neg = neg.copy()
        pos[conflict & (acc_trunc < 0)] = 0
        neg[conflict & (acc_trunc >= 0)] = 0

    spikes = np.zeros((T, D), dtype=np.int8)
    spikes[:T_trunc] = (pos - neg).astype(np.int8)
    return spikes


def phase_encoding(
    acc: np.ndarray,
    num_bits: int = 4,
) -> np.ndarray:
    """Phase spike encoding.

    Delegates to
    :func:`spikify.encoding.temporal.global_referenced.phase_encoding`.

    Encodes the signal by quantising the phase angles of the normalised signal
    amplitudes into a binary spike train.  Unlike the other encoders in this
    module the output is binary ``{0, 1}`` (``uint8``) because phase encoding
    does not have a natural notion of polarity.

    ``T`` must be divisible by ``num_bits``.

    Parameters
    ----------
    acc:
        Input 3D acceleration array of shape ``(T, 3)``.
    num_bits:
        Number of bits (quantisation levels) used for phase encoding.
        ``T`` must be divisible by this value.  Default: 4.

    Returns
    -------
    spikes:
        Unsigned integer array of shape ``(T, 3)`` with values in ``{0, 1}``.
    """
    acc = np.asarray(acc, dtype=float)
    spikes = np.zeros_like(acc).astype(np.uint8)
    T = (len(acc) // num_bits) * num_bits
    spikes[:T] = _phase_encoding(acc[:T], num_bits)
    return spikes

