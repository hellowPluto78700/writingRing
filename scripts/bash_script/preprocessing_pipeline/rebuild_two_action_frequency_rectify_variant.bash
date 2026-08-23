#!/usr/bin/env bash
set -Eeuo pipefail

# Rebuild two Action datasets from completed writingRing pipeline roots while
# allowing BOTH of these variant dimensions to change independently:
#
#   1. five-band Custom Wavelet frequency sequence
#   2. post-encode transform: none | AbsRectify | PolaritySplitAbs
#
# Unlike rebuild_wavelet_variant.bash, this script intentionally allows the
# destination post-encode transform to differ from the source transform.
#
# Safety contract for reuse:
#   - source pipeline must pass the existing preflight checks;
#   - source/destination recording sets must match exactly;
#   - timestamp provenance must match;
#   - the six trailing IMU channels must be EXACTLY equal (np.array_equal);
#   - encoder semantics other than frequency/width/post_encode_transform
#     must remain unchanged;
#   - alignment is reused only for aligned-board-events and is provenance-
#     rebound using the existing wavelet_variant_reuse.py implementation;
#   - segmentation is re-materialized from the new SpikeIMU;
#   - padding is rebuilt with each source dataset's original target length;
#   - final segment/padding geometry is verified against the source.
#
# Destination naming:
#
#   outputs/
#     action0_none_wavelets_1_2_3_4_5/...
#     action1_none_wavelets_1_2_3_4_5/...
#
# or:
#
#   outputs/
#     action0_absrectify_wavelets_0p5_1_2_4_8/...
#     action1_absrectify_wavelets_0p5_1_2_4_8/...
#
# Example:
#
# ACTION0_SOURCE_COMBINATION_ROOT="$PWD/outputs/action0_rectified/low-pass/aligned-board-events" \
# ACTION1_SOURCE_COMBINATION_ROOT="$PWD/outputs/action1_rectified/low-pass/aligned-board-events" \
# ENCODER_FREQUENCIES_HZ="1 2 3 4 5" \
# POST_ENCODE_TRANSFORM="none" \
# bash scripts/bash_script/preprocessing_pipeline/rebuild_two_action_frequency_rectify_variant.bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

# Default completed Action 0 / Action 1 pipeline roots.
# They can still be overridden through environment variables.
ACTION0_SOURCE_COMBINATION_ROOT="${ACTION0_SOURCE_COMBINATION_ROOT:-$REPO_ROOT/outputs/action0_rectified_new_wavelets/low-pass/aligned-board-events}"
ACTION1_SOURCE_COMBINATION_ROOT="${ACTION1_SOURCE_COMBINATION_ROOT:-$REPO_ROOT/outputs/action1_rectified_new_wavelets/low-pass/aligned-board-events}"

# Exactly five positive frequencies, separated by whitespace.
#
# Examples:
#   "1 2 3 4 5"
#   "0.5 1 2 4 8"
#   "1 2 4 8 16"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ:-1 2 3 4 8}"

# Destination post-encode transform.
# It is allowed to differ from the source.
#
# Valid values:
#   none
#   AbsRectify
#   PolaritySplitAbs
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"

# Reuse a completed resampling artifact instead of re-encoding the source-rate
# preprocessing output. Use `none` for source-rate inputs or the source target
# rate (for example `64`). This script never resamples in place.
RESAMPLE_RATE_HZ="${RESAMPLE_RATE_HZ:-none}"

BOUNDARY_MODE="${BOUNDARY_MODE:-aligned-board-events}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"

ENCODER_SETTINGS="${ENCODER_SETTINGS:-$REPO_ROOT/configs/spike_encoding/custom_wavelet.json}"
HELPER_SCRIPT="${HELPER_SCRIPT:-$REPO_ROOT/scripts/wavelet_variant_reuse.py}"

# Destination:
#
#   reencoded_wavelet_variants/
#     action0_<transform>_wavelets_<sequence>/
#     action1_<transform>_wavelets_<sequence>/
DEST_PARENT_ROOT="${DEST_PARENT_ROOT:-$REPO_ROOT/outputs/reencoded_wavelet_variants}"
PIPELINE_STAGE="${PIPELINE_STAGE:-low-pass}"

OVERWRITE_DEST="${OVERWRITE_DEST:-1}"

REBUILD_PADDING="${REBUILD_PADDING:-1}"

SEGMENT_VERIFICATION_DPI="${SEGMENT_VERIFICATION_DPI:-200}"

RUN_COMMON_REPAD="${RUN_COMMON_REPAD:-1}"
RUN_RECONSTRUCTION="${RUN_RECONSTRUCTION:-1}"

# ---------------------------------------------------------------------------
# INTERNALS
# ---------------------------------------------------------------------------

fail() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

note() {
    printf '[frequency-rectify-reuse] %s\n' "$1"
}

[[ -d "$REPO_ROOT" ]] ||
    fail "REPO_ROOT is not a directory: $REPO_ROOT"

[[ -d "$DATA_ROOT" ]] ||
    fail "DATA_ROOT is not a directory: $DATA_ROOT"

[[ -f "$REPO_ROOT/scripts/encode_spikes.py" ]] ||
    fail "not a writingRing checkout: $REPO_ROOT"

[[ -f "$REPO_ROOT/scripts/segment_ring_imu.py" ]] ||
    fail "segment_ring_imu.py is missing"

[[ -f "$REPO_ROOT/scripts/pad_segmented_imu.py" ]] ||
    fail "pad_segmented_imu.py is missing"

[[ -f "$HELPER_SCRIPT" ]] ||
    fail "wavelet reuse helper is missing: $HELPER_SCRIPT"

[[ -f "$ENCODER_SETTINGS" ]] ||
    fail "encoder settings are missing: $ENCODER_SETTINGS"

[[ -n "$ACTION0_SOURCE_COMBINATION_ROOT" ]] ||
    fail "ACTION0_SOURCE_COMBINATION_ROOT is required"

[[ -n "$ACTION1_SOURCE_COMBINATION_ROOT" ]] ||
    fail "ACTION1_SOURCE_COMBINATION_ROOT is required"

[[ -d "$ACTION0_SOURCE_COMBINATION_ROOT" ]] ||
    fail "Action 0 source root is missing: $ACTION0_SOURCE_COMBINATION_ROOT"

[[ -d "$ACTION1_SOURCE_COMBINATION_ROOT" ]] ||
    fail "Action 1 source root is missing: $ACTION1_SOURCE_COMBINATION_ROOT"

[[ -n "$ENCODER_FREQUENCIES_HZ" ]] ||
    fail "ENCODER_FREQUENCIES_HZ is required"

case "$POST_ENCODE_TRANSFORM" in
    none)
        TRANSFORM_SLUG="none"
        ;;
    AbsRectify)
        TRANSFORM_SLUG="absrectify"
        ;;
    PolaritySplitAbs)
        TRANSFORM_SLUG="polaritysplitabs"
        ;;
    *)
        fail "POST_ENCODE_TRANSFORM must be none, AbsRectify, or PolaritySplitAbs"
        ;;
esac

case "$RESAMPLE_RATE_HZ" in
    none) ;;
    *)
        [[ "$RESAMPLE_RATE_HZ" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            fail "RESAMPLE_RATE_HZ must be none or a positive downsampling rate"
        awk -v target="$RESAMPLE_RATE_HZ" -v source="$SAMPLING_RATE" \
            'BEGIN { exit !(target > 0 && target < source) }' ||
            fail "RESAMPLE_RATE_HZ must be lower than SAMPLING_RATE"
        ;;
esac

case "$BOUNDARY_MODE" in
    label|aligned-board-events)
        ;;
    *)
        fail "BOUNDARY_MODE must be label or aligned-board-events"
        ;;
esac

case "$OVERWRITE_DEST" in
    0|1)
        ;;
    *)
        fail "OVERWRITE_DEST must be 0 or 1"
        ;;
esac

case "$REBUILD_PADDING" in
    0|1)
        ;;
    *)
        fail "REBUILD_PADDING must be 0 or 1"
        ;;
esac

read -r -a ENCODER_FREQUENCIES <<<"$ENCODER_FREQUENCIES_HZ"

[[ "${#ENCODER_FREQUENCIES[@]}" -eq 5 ]] ||
    fail "ENCODER_FREQUENCIES_HZ must contain exactly five values"

# Sort the five frequencies numerically in ascending order.
mapfile -t ENCODER_FREQUENCIES < <(
    printf '%s\n' "${ENCODER_FREQUENCIES[@]}" | sort -g
)

# Rebuild the canonical frequency string from the sorted values.
ENCODER_FREQUENCIES_HZ="$(
    printf '%s ' "${ENCODER_FREQUENCIES[@]}"
)"
ENCODER_FREQUENCIES_HZ="${ENCODER_FREQUENCIES_HZ% }"

note "sorted wavelet frequencies: $ENCODER_FREQUENCIES_HZ Hz"

for frequency in "${ENCODER_FREQUENCIES[@]}"; do
    [[ "$frequency" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
        fail "invalid frequency value: $frequency"
done

# Select the same preferred Python environments used by the repository scripts.
if command -v conda >/dev/null 2>&1; then
    if conda env list | grep -q '^writingring-gpu '; then
        PYTHON_CMD=(conda run --no-capture-output -n writingring-gpu python)
    elif conda env list | grep -q '^writingring-viz '; then
        PYTHON_CMD=(conda run --no-capture-output -n writingring-viz python)
    else
        PYTHON_CMD=(python)
    fi
else
    PYTHON_CMD=(python)
fi

# Validate positivity/numeric conversion with Python as well.
"${PYTHON_CMD[@]}" - "${ENCODER_FREQUENCIES[@]}" <<'PY'
import math
import sys

values = [float(x) for x in sys.argv[1:]]
if len(values) != 5:
    raise SystemExit("expected exactly five frequencies")
if any((not math.isfinite(x)) or x <= 0.0 for x in values):
    raise SystemExit("all frequencies must be finite and > 0")
PY

FREQUENCY_SLUG_PARTS=()
for frequency in "${ENCODER_FREQUENCIES[@]}"; do
    # Preserve the user's numeric spelling and only make it filesystem-safe:
    #   0.5 -> 0p5
    #   1.25 -> 1p25
    slug="${frequency//./p}"
    FREQUENCY_SLUG_PARTS+=("$slug")
done

FREQUENCY_SLUG="$(
    IFS=_
    printf '%s' "${FREQUENCY_SLUG_PARTS[*]}"
)"

VARIANT_SUFFIX="${TRANSFORM_SLUG}_wavelets_${FREQUENCY_SLUG}"

ACTION0_DEST_COMBINATION_ROOT="$DEST_PARENT_ROOT/action0_${VARIANT_SUFFIX}/$PIPELINE_STAGE/$BOUNDARY_MODE"
ACTION1_DEST_COMBINATION_ROOT="$DEST_PARENT_ROOT/action1_${VARIANT_SUFFIX}/$PIPELINE_STAGE/$BOUNDARY_MODE"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# RELAXED VARIANT HELPER
# ---------------------------------------------------------------------------
#
# Existing wavelet_variant_reuse.py intentionally allows frequency changes but
# may reject a post_encode_transform change as a non-frequency semantic change.
#
# We keep its proven alignment rebinding / geometry verification code, but
# replace ONLY compare_new_encoding() at runtime with a variant-aware version
# that allows:
#
#   frequencies_hz         source -> requested destination value
#   wavelet widths         derived from new frequencies
#   post_encode_transform  source -> requested destination value
#
# Everything else remains strict.
#
# This DOES NOT modify scripts/wavelet_variant_reuse.py on disk.

variant_helper() {
    local command="$1"
    local source_root="$2"
    local dest_root="$3"
    local action="$4"
    local requested_transform="$5"
    local boundary_mode="$6"
    shift 6
    local frequencies=("$@")

    "${PYTHON_CMD[@]}" - \
        "$HELPER_SCRIPT" \
        "$command" \
        "$source_root" \
        "$dest_root" \
        "$action" \
        "$requested_transform" \
        "$boundary_mode" \
        "${frequencies[@]}" <<'PY'
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

helper_path = Path(sys.argv[1]).resolve()
command = sys.argv[2]
source_root = Path(sys.argv[3]).resolve()
dest_root = Path(sys.argv[4]).resolve()
action = sys.argv[5]
requested_transform = sys.argv[6]
boundary_mode = sys.argv[7]
requested_frequencies = [float(x) for x in sys.argv[8:13]]

if len(requested_frequencies) != 5:
    raise SystemExit("internal error: expected five requested frequencies")

spec = importlib.util.spec_from_file_location(
    "_writingring_wavelet_variant_reuse",
    helper_path,
)
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load helper: {helper_path}")

mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def normalized_transform(meta: dict[str, Any]) -> str:
    value = mod.first_recursive(meta, ("post_encode_transform",))
    if value is None or value == "none":
        return "none"
    if value in {"AbsRectify", "PolaritySplitAbs"}:
        return "AbsRectify"
    raise mod.ReuseError(
        f"unsupported post_encode_transform in metadata: {value!r}"
    )


def strip_allowed_variant_fields(value: Any) -> Any:
    """Remove fields intentionally allowed to differ in this variant rebuild."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            lowered = key.lower()

            # Frequency lists and frequency-derived widths are allowed to
            # change. This mirrors the existing helper's frequency-only rule.
            if "frequenc" in lowered or "width" in lowered:
                continue

            # This script additionally allows post-encode transform changes.
            if lowered == "post_encode_transform":
                continue

            out[key] = strip_allowed_variant_fields(child)
        return out

    if isinstance(value, list):
        return [strip_allowed_variant_fields(x) for x in value]

    return value


def relaxed_compare_new_encoding(
    source_records,
    dest_records,
    requested_freq,
):
    if set(source_records) != set(dest_records):
        missing = sorted(set(source_records) - set(dest_records))
        extra = sorted(set(dest_records) - set(source_records))
        raise mod.ReuseError(
            f"new encoding record set differs; missing={missing}, extra={extra}"
        )

    replacements: dict[str, tuple[str, str]] = {}
    rows: list[dict[str, Any]] = []

    for key in sorted(source_records):
        old = source_records[key]
        new = dest_records[key]

        old_transform = normalized_transform(old.metadata)
        new_transform = normalized_transform(new.metadata)

        if mod.frequencies(new.metadata) != requested_freq:
            raise mod.ReuseError(
                f"new frequencies mismatch for {new.user}/{new.dataset_id}: "
                f"{mod.frequencies(new.metadata)}"
            )

        if new_transform != requested_transform:
            raise mod.ReuseError(
                f"new post_encode_transform mismatch for "
                f"{new.user}/{new.dataset_id}: "
                f"{new_transform!r} != {requested_transform!r}"
            )

        old_semantics = strip_allowed_variant_fields(
            mod.encoder_spec(old.metadata)
        )
        new_semantics = strip_allowed_variant_fields(
            mod.encoder_spec(new.metadata)
        )

        if old_semantics != new_semantics:
            raise mod.ReuseError(
                f"new encoder changed semantics other than "
                f"frequency/width/post_encode_transform for "
                f"{new.user}/{new.dataset_id}"
            )

        old_ts = mod.timestamp_hash(old.metadata)
        new_ts = mod.timestamp_hash(new.metadata)

        if old_ts is None or new_ts is None or old_ts != new_ts:
            raise mod.ReuseError(
                f"timestamp provenance changed for "
                f"{new.user}/{new.dataset_id}: {old_ts} -> {new_ts}"
            )

        old_values = np.load(
            old.values_path,
            allow_pickle=False,
            mmap_mode="r",
        )
        new_values = np.load(
            new.values_path,
            allow_pickle=False,
            mmap_mode="r",
        )

        old_channels, old_events = mod.spike_layout(old.metadata)
        new_channels, new_events = mod.spike_layout(new.metadata)
        if old_values.ndim != 2 or new_values.ndim != 2 or old_values.shape[0] != new_values.shape[0]:
            raise mod.ReuseError(
                f"SpikeIMU row count changed for {new.user}/{new.dataset_id}: "
                f"{old_values.shape} -> {new_values.shape}"
            )
        if old_values.shape[1] != old_channels or new_values.shape[1] != new_channels:
            raise mod.ReuseError(f"unexpected SpikeIMU layouts: {old_values.shape} -> {new_values.shape}")

        # Core reuse invariant:
        #
        # Event-channel width may change for PolaritySplitAbs, but the trailing
        # six raw IMU channels must not.
        if not np.array_equal(
            old_values[:, old_events:],
            new_values[:, new_events:],
        ):
            raise mod.ReuseError(
                f"trailing IMU channels changed for "
                f"{new.user}/{new.dataset_id}; alignment reuse is unsafe"
            )

        old_values_hash = mod.sha256_file(old.values_path)
        new_values_hash = mod.sha256_file(new.values_path)
        old_meta_hash = mod.sha256_file(old.metadata_path)
        new_meta_hash = mod.sha256_file(new.metadata_path)

        for old_hash, new_hash, label in (
            (old_values_hash, new_values_hash, "values"),
            (old_meta_hash, new_meta_hash, "metadata"),
        ):
            if (
                old_hash in replacements
                and replacements[old_hash][0] != new_hash
            ):
                raise mod.ReuseError(
                    f"ambiguous {label} hash replacement for {old_hash}"
                )

            replacements[old_hash] = (
                new_hash,
                f"{new.user}/{new.dataset_id}:{label}",
            )

        rows.append(
            {
                "user": new.user,
                "dataset_id": new.dataset_id,
                "source_transform": old_transform,
                "dest_transform": new_transform,
                "old_encoder_hash": mod.encoder_hash(old.metadata),
                "new_encoder_hash": mod.encoder_hash(new.metadata),
                "old_values_sha256": old_values_hash,
                "new_values_sha256": new_values_hash,
                "old_metadata_sha256": old_meta_hash,
                "new_metadata_sha256": new_meta_hash,
                "timestamp_sha256": new_ts,
                "trailing_imu_equal": True,
            }
        )

    return rows, replacements


# Monkeypatch only the comparison policy. Existing alignment provenance
# rebinding, manifest refresh, segment comparison and padding comparison remain
# the repository implementation.
mod.compare_new_encoding = relaxed_compare_new_encoding

try:
    if command == "check-encoding":
        args = argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            frequencies=requested_frequencies,
        )
        result = mod.check_encoding(args)

    elif command == "rebind-alignment":
        args = argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            frequencies=requested_frequencies,
            overwrite=True,
        )
        result = mod.rebind_alignment(args)

    elif command == "verify":
        args = argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            boundary_mode=boundary_mode,
            frequencies=requested_frequencies,
        )
        result = mod.verify(args)

        source_records = mod.discover_records(source_root, action)
        dest_records = mod.discover_records(dest_root, action)

        source_transforms = sorted(
            {normalized_transform(r.metadata) for r in source_records.values()}
        )
        dest_transforms = sorted(
            {normalized_transform(r.metadata) for r in dest_records.values()}
        )

        if dest_transforms != [requested_transform]:
            raise mod.ReuseError(
                f"destination transform set is {dest_transforms}, "
                f"expected only {requested_transform!r}"
            )

        # If either variant dimension changed, the encoder identity must also
        # change. This extends the existing frequency-only identity check.
        source_freq = mod.frequencies(
            next(iter(source_records.values())).metadata
        )
        source_hash = mod.encoder_hash(
            next(iter(source_records.values())).metadata
        )
        dest_hash = mod.encoder_hash(
            next(iter(dest_records.values())).metadata
        )

        variant_changed = (
            source_freq != requested_frequencies
            or source_transforms != [requested_transform]
        )

        if variant_changed and source_hash == dest_hash:
            raise mod.ReuseError(
                "frequency and/or transform changed but encoder identity "
                "did not change"
            )

        result["source_post_encode_transforms"] = source_transforms
        result["requested_post_encode_transform"] = requested_transform
        result["dest_post_encode_transforms"] = dest_transforms
        result["contracts"][
            "frequency_change_allowed"
        ] = True
        result["contracts"][
            "post_encode_transform_change_allowed"
        ] = True
        result["contracts"]["trailing_imu_channels_equal"] = True

        output_path = dest_root / "wavelet_variant_validation.json"
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    else:
        raise mod.ReuseError(f"unsupported relaxed helper command: {command}")

    print(json.dumps(result, indent=2, sort_keys=True))

except (mod.ReuseError, OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
}

# ---------------------------------------------------------------------------
# REBUILD ONE ACTION
# ---------------------------------------------------------------------------

rebuild_action() {
    local action="$1"
    local source_root="$2"
    local dest_root="$3"

    note "============================================================"
    note "Action $action"
    note "source:      $source_root"
    note "destination: $dest_root"
    note "frequencies: $ENCODER_FREQUENCIES_HZ Hz"
    note "transform:   $POST_ENCODE_TRANSFORM"
    note "============================================================"

    note "preflight source reuse contract"
    "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" preflight \
        --source-root "$source_root" \
        --action "$action" \
        --boundary-mode "$BOUNDARY_MODE"

    local source_transform
    source_transform="$(
        "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" source-transform \
            --source-root "$source_root" \
            --action "$action"
    )"

    note "source post-encode transform: $source_transform"
    note "requested destination transform: $POST_ENCODE_TRANSFORM"

    if [[ "$source_transform" != "$POST_ENCODE_TRANSFORM" ]]; then
        note "transform change intentionally allowed: $source_transform -> $POST_ENCODE_TRANSFORM"
    fi

    if [[ -e "$dest_root" ]]; then
        if [[ "$OVERWRITE_DEST" != "1" ]]; then
            fail "destination already exists; set OVERWRITE_DEST=1 to replace it: $dest_root"
        fi

        case "$dest_root" in
            /|"$REPO_ROOT"|"$source_root")
                fail "refusing unsafe destination removal: $dest_root"
                ;;
        esac

        note "removing existing destination root"
        rm -rf -- "$dest_root"
    fi

    mkdir -p "$dest_root"

    local source_preprocess_root="$source_root/preprocessedIMU"
    local dest_preprocess_root="$dest_root/preprocessedIMU"
    local source_encode_input_root="$source_preprocess_root"
    local dest_encode_input_root="$dest_preprocess_root"
    local encode_pattern='*_preprocessedIMU.npy'
    local effective_sampling_rate="$SAMPLING_RATE"

    local dest_spike_output_root="$dest_root/spikeEncoding"
    local dest_spike_root="$dest_spike_output_root/custom-wavelet"
    local dest_segment_root="$dest_root/segmentation"
    local dest_padding_root="$dest_root/segmentation_padded"
    local dest_alignment_offset_root="$dest_root/alignment/offsets"

    [[ -d "$source_preprocess_root" ]] ||
        fail "source preprocessedIMU root is missing: $source_preprocess_root"

    note "reuse preprocessedIMU by symbolic links"

    rm -rf -- "$dest_preprocess_root"
    mkdir -p "$dest_preprocess_root"

    local symlink_count=0
    local src rel dst

    while IFS= read -r -d '' src; do
        rel="${src#"$source_preprocess_root"/}"
        dst="$dest_preprocess_root/$rel"

        mkdir -p "$(dirname -- "$dst")"
        ln -s "$(realpath "$src")" "$dst"

        symlink_count=$((symlink_count + 1))
    done < <(find "$source_preprocess_root" -type f -print0)

    [[ "$symlink_count" -gt 0 ]] ||
        fail "no reusable preprocessedIMU files found under $source_preprocess_root"

    note "preprocessedIMU symlink tree created: $symlink_count files"

    if [[ "$RESAMPLE_RATE_HZ" != "none" ]]; then
        local source_resample_root="$source_root/resampledIMU"
        local dest_resample_root="$dest_root/resampledIMU"
        [[ -d "$source_resample_root" ]] ||
            fail "source resampledIMU root is missing: $source_resample_root"
        note "reuse resampledIMU by symbolic links (rate=${RESAMPLE_RATE_HZ} Hz)"
        mkdir -p "$dest_resample_root"
        symlink_count=0
        while IFS= read -r -d '' src; do
            rel="${src#"$source_resample_root"/}"
            dst="$dest_resample_root/$rel"
            mkdir -p "$(dirname -- "$dst")"
            ln -s "$(realpath "$src")" "$dst"
            symlink_count=$((symlink_count + 1))
        done < <(find "$source_resample_root" -type f -print0)
        [[ "$symlink_count" -gt 0 ]] ||
            fail "no reusable resampledIMU files found under $source_resample_root"
        source_encode_input_root="$source_resample_root"
        dest_encode_input_root="$dest_resample_root"
        encode_pattern='*_resampledIMU.npy'
        effective_sampling_rate="$RESAMPLE_RATE_HZ"
    fi

    note "encode destination Custom Wavelet variant"
    "${PYTHON_CMD[@]}" scripts/encode_spikes.py \
        --input-root "$dest_encode_input_root" \
        --pattern "$encode_pattern" \
        --output-root "$dest_spike_output_root" \
        --encoder custom-wavelet \
        --encoder-settings "$ENCODER_SETTINGS" \
        --encoder-frequencies-hz "${ENCODER_FREQUENCIES[@]}" \
        --effective-sampling-rate-hz "$effective_sampling_rate" \
        --post-encode-transform "$POST_ENCODE_TRANSFORM" \
        --overwrite

    note "validate new encoding with frequency + rectify changes allowed"
    variant_helper \
        check-encoding \
        "$source_root" \
        "$dest_root" \
        "$action" \
        "$POST_ENCODE_TRANSFORM" \
        "$BOUNDARY_MODE" \
        "${ENCODER_FREQUENCIES[@]}"

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        note "prove trailing IMU invariance and rebind alignment provenance"
        variant_helper \
            rebind-alignment \
            "$source_root" \
            "$dest_root" \
            "$action" \
            "$POST_ENCODE_TRANSFORM" \
            "$BOUNDARY_MODE" \
            "${ENCODER_FREQUENCIES[@]}"
    fi

    note "re-materialize segmentation from the reused boundary/alignment contract"
    mkdir -p "$dest_segment_root"

    local user_count=0
    local user_dir user

    for user_dir in "$dest_spike_root"/*; do
        [[ -d "$user_dir" ]] || continue

        user="$(basename -- "$user_dir")"

        if [[ ! -d "$user_dir/$action" && ! -d "$user_dir/action_$action" ]]; then
            continue
        fi

        user_count=$((user_count + 1))

        segment_args=(
            "${PYTHON_CMD[@]}"
            scripts/segment_ring_imu.py
            --data-root "$DATA_ROOT"
            --user "$user"
            --action "$action"
            --input-kind spike-imu
            --spike-root "$dest_spike_root"
            --boundary-mode "$BOUNDARY_MODE"
            --sampling-rate "$effective_sampling_rate"
            --output-root "$dest_segment_root"
            --overwrite
        )

        if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
            segment_args+=(
                --alignment-offset-root "$dest_alignment_offset_root"
                --pre-press-context-seconds 0.2
                --post-lift-context-seconds 0.2
                --maximum-segment-duration-seconds 5.0
                --carry-in-press-lookback-seconds 0.2
                --missing-event-policy skip
                --crossing-touch-policy accept_until_next_press
                --recording-error-policy skip
                --verification-panel-seconds 10
                --verification-dpi "$SEGMENT_VERIFICATION_DPI"
                --overwrite-verification
            )
        fi

        note "segment user=$user action=$action"
        "${segment_args[@]}"
    done

    [[ "$user_count" -gt 0 ]] ||
        fail "no users found under new spike root: $dest_spike_root"

    if [[ "$REBUILD_PADDING" == "1" ]]; then
        local source_padding_target
        source_padding_target="$(
            "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" padding-target \
                --source-root "$source_root"
        )"

        if [[ "$source_padding_target" != "NONE" ]]; then
            note "rebuild padding with source target length: $source_padding_target"

            "${PYTHON_CMD[@]}" scripts/pad_segmented_imu.py \
                --input-root "$dest_segment_root" \
                --output-root "$dest_padding_root" \
                --target-length "$source_padding_target" \
                --sampling-rate "$effective_sampling_rate" \
                --padding-value 0.0 \
                --overwrite
        else
            note "source has no padded dataset; padding skipped"
        fi
    else
        note "padding rebuild disabled by REBUILD_PADDING=0"
    fi

    note "final contract verification"
    variant_helper \
        verify \
        "$source_root" \
        "$dest_root" \
        "$action" \
        "$POST_ENCODE_TRANSFORM" \
        "$BOUNDARY_MODE" \
        "${ENCODER_FREQUENCIES[@]}"

    note "Action $action PASS"
    note "validation report: $dest_root/wavelet_variant_validation.json"
}

# ---------------------------------------------------------------------------
# RUN BOTH ACTIONS
# ---------------------------------------------------------------------------

note "new variant ID: ${TRANSFORM_SLUG}_wavelets_${FREQUENCY_SLUG}"
note "Action 0 destination: $ACTION0_DEST_COMBINATION_ROOT"
note "Action 1 destination: $ACTION1_DEST_COMBINATION_ROOT"

rebuild_action \
    0 \
    "$ACTION0_SOURCE_COMBINATION_ROOT" \
    "$ACTION0_DEST_COMBINATION_ROOT"

rebuild_action \
    1 \
    "$ACTION1_SOURCE_COMBINATION_ROOT" \
    "$ACTION1_DEST_COMBINATION_ROOT"

note "============================================================"
note "ALL ACTIONS PASS"
note "variant: $VARIANT_SUFFIX"
note "Action 0 dataset: $ACTION0_DEST_COMBINATION_ROOT"
note "Action 1 dataset: $ACTION1_DEST_COMBINATION_ROOT"
note "============================================================"
