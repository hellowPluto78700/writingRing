"""Generic, extensible spike-encoding framework."""

from writingring.spike_encoding.contracts import (
    SpikeEncoder,
    SpikeEncoderOutputMetadata,
    SpikeEncodingError,
    SpikeEncodingOutput,
    SpikeEncodingSequenceResult,
)
from writingring.spike_encoding.io import (
    SpikeEncodingInput,
    SpikeEncodingSourceSummary,
    load_and_validate_timestamps,
    load_encoder_settings,
    load_sequence_offsets,
    load_spike_encoding_input,
    load_spike_encoding_source_summary,
    resolve_sampling_rate_hz,
    single_array_offsets,
    validate_source_metadata_paths,
    validate_sequence_offsets,
)
from writingring.spike_encoding.registry import (
    ENCODER_REGISTRY,
    available_encoders,
    create_encoder,
    register_encoder,
)
from writingring.spike_encoding.runner import run_spike_encoder
from writingring.spike_encoding.publication import (
    SpikeEncodingOutputPaths,
    SpikeEncodingPublishError,
    publish_spike_encoding,
    spike_encoding_output_paths,
)
from writingring.spike_encoding.encoders.custom_wavelet import (
    CustomWaveletEncoder,
    CustomWaveletSettings,
    CustomWaveletSettingsError,
)

__all__ = [
    "SpikeEncoder",
    "SpikeEncoderOutputMetadata",
    "SpikeEncodingError",
    "SpikeEncodingInput",
    "SpikeEncodingSourceSummary",
    "SpikeEncodingOutput",
    "SpikeEncodingOutputPaths",
    "SpikeEncodingPublishError",
    "SpikeEncodingSequenceResult",
    "CustomWaveletEncoder",
    "CustomWaveletSettings",
    "CustomWaveletSettingsError",
    "ENCODER_REGISTRY",
    "available_encoders",
    "create_encoder",
    "load_encoder_settings",
    "load_and_validate_timestamps",
    "load_sequence_offsets",
    "load_spike_encoding_input",
    "load_spike_encoding_source_summary",
    "register_encoder",
    "publish_spike_encoding",
    "run_spike_encoder",
    "resolve_sampling_rate_hz",
    "single_array_offsets",
    "validate_source_metadata_paths",
    "spike_encoding_output_paths",
    "validate_sequence_offsets",
]
