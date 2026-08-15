from __future__ import annotations

"""Mask-aware 1-D CNN for padded three-axis acceleration segments.

This module preserves the interface used by the current acceleration CNN
notebooks:

    logits, embedding = model(x, valid_mask=valid_mask)

Expected shapes
---------------
x:
    (B, 3, T), float tensor.
valid_mask:
    (B, T), bool-compatible tensor. Valid samples form a contiguous prefix.
logits:
    (B, num_classes).
embedding:
    (B, 256).

The layer names and ``architecture_config()`` output intentionally match the
existing baseline checkpoint so that previously saved ``model_state_dict``
objects remain loadable with ``strict=True``.
"""

from typing import Final

import torch
import torch.nn.functional as F
from torch import nn


INPUT_CHANNELS: Final[int] = 3
EMBEDDING_DIM: Final[int] = 256


__all__ = [
    "INPUT_CHANNELS",
    "EMBEDDING_DIM",
    "conv1d_output_lengths",
    "prefix_mask",
    "MaskAwareAccelerationCNN",
    "ProbeVariant",
    "build_probe_model",
    "validate_probe_variant",
]

ProbeVariant = str
PROBE_VARIANTS: Final[tuple[str, ...]] = ("cnn_s", "cnn_m", "cnn_l")


def conv1d_output_lengths(
    lengths: torch.Tensor,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """Propagate valid prefix lengths through a Conv1d layer.

    This is the standard PyTorch Conv1d output-length formula applied
    elementwise to a tensor of valid sequence lengths.
    """
    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must have shape (B,), got {tuple(lengths.shape)}"
        )
    if kernel_size <= 0 or stride <= 0 or dilation <= 0 or padding < 0:
        raise ValueError(
            "kernel_size, stride, and dilation must be positive; "
            "padding must be non-negative"
        )

    numerator = (
        lengths
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    )
    output = torch.div(numerator, stride, rounding_mode="floor") + 1

    if torch.any(output <= 0):
        raise ValueError(
            "Conv1d parameters collapse at least one valid sequence to "
            f"non-positive length: {output.detach().cpu().tolist()}"
        )
    return output


def prefix_mask(
    lengths: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    """Build a boolean right-padding mask from valid prefix lengths."""
    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must have shape (B,), got {tuple(lengths.shape)}"
        )
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if torch.any(lengths <= 0) or torch.any(lengths > time_steps):
        raise ValueError(
            f"valid lengths must lie in [1, {time_steps}], got "
            f"{lengths.detach().cpu().tolist()}"
        )

    return (
        torch.arange(time_steps, device=lengths.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )


class MaskAwareAccelerationCNN(nn.Module):
    """CNN baseline used for acceleration-only character classification.

    Notes
    -----
    - Input has exactly three acceleration channels.
    - Right-padding is masked at the input and after every convolution block.
    - The final representation is a 256-D masked global-average-pooled vector.
    - ``forward`` returns ``(logits, embedding)`` to preserve the current
      notebook/checkpoint interface.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_classes = int(num_classes)
        self.embedding_dim = EMBEDDING_DIM

        # Keep attribute names unchanged for strict checkpoint compatibility.
        self.conv1 = nn.Conv1d(
            INPUT_CHANNELS,
            64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(
            64,
            128,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(
            128,
            EMBEDDING_DIM,
            kernel_size=5,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(EMBEDDING_DIM)

        self.conv4 = nn.Conv1d(
            EMBEDDING_DIM,
            EMBEDDING_DIM,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn4 = nn.BatchNorm1d(EMBEDDING_DIM)

        self.classifier = nn.Linear(
            self.embedding_dim,
            self.num_classes,
        )

    @staticmethod
    def _masked_feature(
        feature: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            feature.shape[0] != mask.shape[0]
            or feature.shape[-1] != mask.shape[1]
        ):
            raise ValueError(
                "Feature/mask mismatch: "
                f"feature={tuple(feature.shape)}, "
                f"mask={tuple(mask.shape)}"
            )
        return feature * mask.unsqueeze(1).to(dtype=feature.dtype)

    @staticmethod
    def _validate_input(
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"x must have shape (B, {INPUT_CHANNELS}, T), "
                f"got {tuple(x.shape)}"
            )
        expected_mask_shape = (x.shape[0], x.shape[2])
        if valid_mask.shape != expected_mask_shape:
            raise ValueError(
                f"valid_mask must have shape {expected_mask_shape}, "
                f"got {tuple(valid_mask.shape)}"
            )

        valid_mask = valid_mask.to(
            device=x.device,
            dtype=torch.bool,
        )
        if not valid_mask.any(dim=1).all():
            raise ValueError(
                "Every segment must contain at least one valid time step"
            )
        return valid_mask

    def encode(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 256-D pre-classifier embedding."""
        valid_mask = self._validate_input(x, valid_mask)

        lengths = valid_mask.sum(dim=1).to(dtype=torch.long)

        # Remove arbitrary values from the padded input region.
        x = x * valid_mask.unsqueeze(1).to(dtype=x.dtype)

        x = F.relu(self.bn1(self.conv1(x)))
        lengths = conv1d_output_lengths(
            lengths,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        mask = prefix_mask(lengths, x.shape[-1])
        x = self._masked_feature(x, mask)

        x = F.relu(self.bn2(self.conv2(x)))
        lengths = conv1d_output_lengths(
            lengths,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        mask = prefix_mask(lengths, x.shape[-1])
        x = self._masked_feature(x, mask)

        x = F.relu(self.bn3(self.conv3(x)))
        lengths = conv1d_output_lengths(
            lengths,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        mask = prefix_mask(lengths, x.shape[-1])
        x = self._masked_feature(x, mask)

        x = F.relu(self.bn4(self.conv4(x)))
        lengths = conv1d_output_lengths(
            lengths,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        mask = prefix_mask(lengths, x.shape[-1])
        x = self._masked_feature(x, mask)

        weights = mask.unsqueeze(1).to(dtype=x.dtype)
        embedding = (
            (x * weights).sum(dim=-1)
            / weights.sum(dim=-1).clamp_min(1.0)
        )

        expected_shape = (x.shape[0], self.embedding_dim)
        if embedding.shape != expected_shape:
            raise AssertionError(
                f"Unexpected embedding shape: {tuple(embedding.shape)}; "
                f"expected {expected_shape}"
            )
        return embedding

    def classify_embedding(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the existing linear classifier to a 256-D embedding."""
        if embedding.ndim != 2:
            raise ValueError(
                "embedding must have shape (B, embedding_dim), "
                f"got {tuple(embedding.shape)}"
            )
        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding dimension must be {self.embedding_dim}, "
                f"got {embedding.shape[1]}"
            )
        return self.classifier(embedding)

    def forward(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, embedding)`` using the current notebook API."""
        embedding = self.encode(x, valid_mask=valid_mask)
        logits = self.classify_embedding(embedding)

        expected_shape = (x.shape[0], self.num_classes)
        if logits.shape != expected_shape:
            raise AssertionError(
                f"Unexpected logits shape: {tuple(logits.shape)}; "
                f"expected {expected_shape}"
            )
        return logits, embedding

    def architecture_config(self) -> dict[str, object]:
        """Return the exact architecture descriptor stored in current checkpoints."""
        return {
            "variant": "cnn_l",
            "input_channels": 3,
            "blocks": [
                {
                    "out_channels": 64,
                    "kernel_size": 7,
                    "stride": 1,
                    "padding": 3,
                },
                {
                    "out_channels": 128,
                    "kernel_size": 5,
                    "stride": 2,
                    "padding": 2,
                },
                {
                    "out_channels": 256,
                    "kernel_size": 5,
                    "stride": 2,
                    "padding": 2,
                },
                {
                    "out_channels": 256,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                },
            ],
            "normalization": "BatchNorm1d",
            "activation": "ReLU",
            "pooling": (
                "valid-length-propagated masked global average pooling"
            ),
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
        }


class _MaskedAccelerationProbe(nn.Module):
    """Small mask-aware convolutional probe with a shared model interface."""

    def __init__(self, num_classes: int, *, variant: str, channels: tuple[int, ...],
                 kernels: tuple[int, ...], strides: tuple[int, ...]) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("probe layer configuration lengths must match")
        self.num_classes = int(num_classes)
        self.variant = variant
        self.embedding_dim = channels[-1]
        layers: list[nn.Conv1d] = []
        in_channels = INPUT_CHANNELS
        for out_channels, kernel, stride in zip(channels, kernels, strides):
            layer = nn.Conv1d(in_channels, out_channels, kernel, stride=stride,
                              padding=kernel // 2, bias=True)
            layers.append(layer)
            in_channels = out_channels
        self.convs = nn.ModuleList(layers)
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

    @staticmethod
    def _validate_input(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"x must have shape (B, {INPUT_CHANNELS}, T), got {tuple(x.shape)}")
        if valid_mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError(f"valid_mask must have shape {(x.shape[0], x.shape[2])}, got {tuple(valid_mask.shape)}")
        mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every segment must contain at least one valid time step")
        return mask

    def encode(self, x: torch.Tensor, *, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = self._validate_input(x, valid_mask)
        lengths = mask.sum(dim=1).to(dtype=torch.long)
        x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        for conv in self.convs:
            x = F.relu(conv(x))
            lengths = conv1d_output_lengths(lengths, kernel_size=conv.kernel_size[0],
                                            stride=conv.stride[0], padding=conv.padding[0])
            mask = prefix_mask(lengths, x.shape[-1])
            x = x * mask.unsqueeze(1).to(dtype=x.dtype)
        weights = mask.unsqueeze(1).to(dtype=x.dtype)
        return (x * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, *, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(x, valid_mask=valid_mask)
        return self.classifier(embedding), embedding

    def architecture_config(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "input_channels": INPUT_CHANNELS,
            "blocks": [{"out_channels": c.out_channels, "kernel_size": c.kernel_size[0],
                        "stride": c.stride[0], "padding": c.padding[0]}
                       for c in self.convs],
            "normalization": "none",
            "activation": "ReLU",
            "pooling": "valid-length-propagated masked global average pooling",
            "embedding_dim": self.embedding_dim,
            "num_classes": self.num_classes,
        }


class AccelerationCNNS(_MaskedAccelerationProbe):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, variant="cnn_s", channels=(8,), kernels=(7,), strides=(1,))


class AccelerationCNNM(_MaskedAccelerationProbe):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, variant="cnn_m", channels=(32, 64), kernels=(7, 5), strides=(1, 2))


def validate_probe_variant(variant: str) -> str:
    value = str(variant).lower()
    if value not in PROBE_VARIANTS:
        raise ValueError(f"Unknown probe variant {variant!r}; expected one of {PROBE_VARIANTS}")
    return value


def build_probe_model(variant: str, num_classes: int) -> nn.Module:
    value = validate_probe_variant(variant)
    if value == "cnn_s":
        return AccelerationCNNS(num_classes)
    if value == "cnn_m":
        return AccelerationCNNM(num_classes)
    return MaskAwareAccelerationCNN(num_classes)
