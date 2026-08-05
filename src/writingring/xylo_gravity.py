"""Xylo IMU rotation processing and gravity subtraction in the g domain.

The Rockpool Xylo ``RotationRemoval`` simulator accepts fixed-point values.
This wrapper keeps physical conversion outside the simulator: callers supply
and receive acceleration in g, while the implementation quantizes a clipped
``[-1, 1]`` normalized representation internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np


_VECTOR_WIDTH: Final[int] = 3
_XYLO_INSTALL_COMMAND: Final[str] = 'pip install -e ".[xylo]"'


class XyloGravityError(ValueError):
    """Raised when Xylo gravity processing cannot produce a valid result."""


@dataclass(frozen=True, slots=True)
class XyloGravityConfig:
    """Fixed-point and causal gravity-estimation settings for Xylo processing."""

    full_scale_g: float = 2.0
    quantizer_num_bits: int = 16
    rotation_num_avg_bitshift: int = 4
    rotation_sampling_period: int = 10
    gravity_num_avg_bitshift: int = 4
    clip_epsilon: float = 1e-7


@dataclass(frozen=True, slots=True)
class XyloGravityResult:
    """Read-only Xylo-domain acceleration arrays, all expressed in g."""

    rotation_removed_acceleration_g: np.ndarray
    gravity_g: np.ndarray
    linear_acceleration_g: np.ndarray
    normalized_quantizer_input: np.ndarray
    config: XyloGravityConfig


def xylo_rotate_and_remove_gravity(
    acceleration_g: np.ndarray,
    *,
    sampling_rate_hz: float,
    config: XyloGravityConfig = XyloGravityConfig(),
) -> XyloGravityResult:
    """Rotate Xylo-quantized acceleration, then subtract its causal baseline.

    ``RotationRemoval`` performs fixed-point rotation removal but does not
    expose a linear-acceleration signal.  The final causal moving baseline is
    therefore explicit and uses the same power-of-two style as the Xylo
    averaging blocks.  No resampling is performed.
    """

    values = _validated_acceleration(acceleration_g)
    _validated_config(config, sampling_rate_hz=sampling_rate_hz)
    normalized = np.clip(
        values / config.full_scale_g,
        -1.0 + config.clip_epsilon,
        1.0 - config.clip_epsilon,
    )
    rotated = _run_xylo_rotation(normalized, config=config)

    scale = float(2 ** (config.quantizer_num_bits - 1))
    rotation_removed = np.asarray(rotated, dtype=np.float64)
    if rotation_removed.ndim == 3 and rotation_removed.shape[0] == 1:
        rotation_removed = rotation_removed[0]
    if rotation_removed.shape != values.shape:
        raise XyloGravityError(
            "Xylo rotation removal returned unexpected shape "
            f"{rotation_removed.shape}; expected {values.shape}"
        )
    rotation_removed = rotation_removed / scale * config.full_scale_g
    gravity = _causal_power_of_two_average(
        rotation_removed, bitshift=config.gravity_num_avg_bitshift
    )
    linear = rotation_removed - gravity
    if not np.isfinite(rotation_removed).all() or not np.isfinite(linear).all():
        raise XyloGravityError("Xylo gravity processing produced non-finite values")
    for value in (rotation_removed, gravity, linear, normalized):
        value.setflags(write=False)
    return XyloGravityResult(
        rotation_removed_acceleration_g=rotation_removed,
        gravity_g=gravity,
        linear_acceleration_g=linear,
        normalized_quantizer_input=normalized,
        config=config,
    )


def _validated_acceleration(acceleration_g: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(acceleration_g, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise XyloGravityError("acceleration_g must be numeric") from error
    if values.ndim != 2 or values.shape[1] != _VECTOR_WIDTH or len(values) == 0:
        raise XyloGravityError("acceleration_g must have nonempty shape (N, 3)")
    if not np.isfinite(values).all():
        raise XyloGravityError("acceleration_g must be finite")
    return values.copy()


def _run_xylo_rotation(
    normalized_acceleration: np.ndarray, *, config: XyloGravityConfig
) -> np.ndarray:
    """Run the external fixed-point Xylo stages on already normalized input."""

    Quantizer, RotationRemoval = _load_xylo_components()
    try:
        quantizer = Quantizer(
            shape=(_VECTOR_WIDTH, _VECTOR_WIDTH),
            scale=1.0,
            num_bits=config.quantizer_num_bits,
        )
        rotation = RotationRemoval(
            shape=(_VECTOR_WIDTH, _VECTOR_WIDTH),
            num_avg_bitshift=config.rotation_num_avg_bitshift,
            sampling_period=config.rotation_sampling_period,
        )
        quantized, _, _ = quantizer(normalized_acceleration)
        rotated, _, _ = rotation(quantized)
    except (TypeError, ValueError, OverflowError) as error:
        raise XyloGravityError(f"Xylo rotation removal failed: {error}") from error
    return np.asarray(rotated)


def _load_xylo_components() -> tuple[type[object], type[object]]:
    """Lazily load the optional Rockpool Xylo simulator components."""

    try:
        # Quantizer is re-exported by syns63300 in Rockpool 2.9.1.
        from rockpool.devices.xylo.syns63300 import Quantizer
        from rockpool.devices.xylo.syns63300.imuif import RotationRemoval
    except (ImportError, ModuleNotFoundError) as error:
        raise XyloGravityError(
            "Rockpool Xylo IMU support is unavailable. Install Xylo support "
            f"with: {_XYLO_INSTALL_COMMAND}"
        ) from error
    return Quantizer, RotationRemoval


def _validated_config(config: XyloGravityConfig, *, sampling_rate_hz: float) -> None:
    if not isinstance(config, XyloGravityConfig):
        raise XyloGravityError("config must be XyloGravityConfig")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise XyloGravityError("sampling_rate_hz must be positive and finite")
    if not np.isfinite(config.full_scale_g) or config.full_scale_g <= 0.0:
        raise XyloGravityError("full_scale_g must be positive and finite")
    if not isinstance(config.quantizer_num_bits, int) or config.quantizer_num_bits < 2:
        raise XyloGravityError("quantizer_num_bits must be an integer of at least 2")
    for name, value, maximum in (
        ("rotation_num_avg_bitshift", config.rotation_num_avg_bitshift, 31),
        ("rotation_sampling_period", config.rotation_sampling_period, 2047),
        ("gravity_num_avg_bitshift", config.gravity_num_avg_bitshift, 31),
    ):
        if not isinstance(value, int) or not 0 <= value <= maximum:
            raise XyloGravityError(f"{name} must be an integer in [0, {maximum}]")
    if not np.isfinite(config.clip_epsilon) or not 0.0 < config.clip_epsilon < 1.0:
        raise XyloGravityError("clip_epsilon must be finite and in (0, 1)")


def _causal_power_of_two_average(values: np.ndarray, *, bitshift: int) -> np.ndarray:
    """Return a causal exponential average with a power-of-two coefficient."""

    result = np.empty_like(values)
    result[0] = values[0]
    alpha = 1.0 / float(2**bitshift)
    for index in range(1, len(values)):
        result[index] = result[index - 1] + alpha * (values[index] - result[index - 1])
    return result
