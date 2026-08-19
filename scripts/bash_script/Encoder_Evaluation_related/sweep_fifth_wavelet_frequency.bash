#!/usr/bin/env bash
set -Eeuo pipefail

# Build a fifth-wavelet frequency sweep for the standalone Experiment C benchmark.
#
# The input is one completed two-action combination. The source encoder must
# contain the fixed backbone 1, 2, 4, 8 Hz plus exactly one additional band.
# The source pair is retained as the benchmark reference condition (normally
# 0.5 Hz), while SWEEP_START:SWEEP_STEP:SWEEP_END creates additional conditions.
#
# Duplicate fifth frequencies are intentionally supported. For example, a
# sweep value of 2 Hz creates the five-band bank (1, 2, 2, 4, 8). The repository
# CustomWaveletSettings normally rejects duplicate bands, so duplicate cases are
# encoded through a runtime-only relaxed validator; the core source file is not
# modified. All other Custom Wavelet semantics remain unchanged.
#
# Destination layout:
#   <DEST_PARENT_ROOT>/
#     1Hz/action0/<PIPELINE_STAGE>/<BOUNDARY_MODE>/
#     1Hz/action1/<PIPELINE_STAGE>/<BOUNDARY_MODE>/
#     1p5Hz/action0/...
#     ...
#     experiment_c_frequency_benchmark_inputs.json
#
# Usage example:
#   ACTION0_SOURCE_COMBINATION_ROOT="$PWD/outputs/action0_rectified/low-pass/aligned-board-events" \
#   ACTION1_SOURCE_COMBINATION_ROOT="$PWD/outputs/action1_rectified/low-pass/aligned-board-events" \
#   SWEEP_START=1 SWEEP_STEP=0.5 SWEEP_END=20 \
#   POST_ENCODE_TRANSFORM=none \
#   bash scripts/bash_script/preprocessing_pipeline/sweep_fifth_wavelet_frequency.bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"

ACTION0_SOURCE_COMBINATION_ROOT="${ACTION0_SOURCE_COMBINATION_ROOT:-}"
ACTION1_SOURCE_COMBINATION_ROOT="${ACTION1_SOURCE_COMBINATION_ROOT:-}"

FIXED_WAVELET_CHANNELS_HZ="${FIXED_WAVELET_CHANNELS_HZ:-1 2 4 8}"
SWEEP_START="${SWEEP_START:-1}"
SWEEP_STEP="${SWEEP_STEP:-1}"
SWEEP_END="${SWEEP_END:-20}"

# Keep the option. The default means no post-encode transform is applied.
POST_ENCODE_TRANSFORM="${POST_ENCODE_TRANSFORM:-none}"

BOUNDARY_MODE="${BOUNDARY_MODE:-aligned-board-events}"
SAMPLING_RATE="${SAMPLING_RATE:-200}"
PIPELINE_STAGE="${PIPELINE_STAGE:-low-pass}"
ENCODER_SETTINGS="${ENCODER_SETTINGS:-$REPO_ROOT/configs/spike_encoding/custom_wavelet.json}"
HELPER_SCRIPT="${HELPER_SCRIPT:-$REPO_ROOT/scripts/wavelet_variant_reuse.py}"
COMMON_REPAD_SCRIPT="${COMMON_REPAD_SCRIPT:-$REPO_ROOT/scripts/bash_script/preprocessing_pipeline/repad_actions_to_common_target.sh}"
RECONSTRUCTION_SCRIPT="${RECONSTRUCTION_SCRIPT:-$REPO_ROOT/scripts/reconstruct_padded_spike_accel.py}"

DEST_PARENT_ROOT="${DEST_PARENT_ROOT:-$REPO_ROOT/outputs/fifth_wavelet_frequency_sweep}"
OVERWRITE_DEST="${OVERWRITE_DEST:-0}"
RESUME="${RESUME:-1}"
SEGMENT_VERIFICATION_DPI="${SEGMENT_VERIFICATION_DPI:-120}"

# ---------------------------------------------------------------------------
# INTERNALS
# ---------------------------------------------------------------------------

fail() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

note() {
    printf '[fifth-wavelet-sweep] %s\n' "$1"
}

[[ -d "$REPO_ROOT" ]] || fail "REPO_ROOT is not a directory: $REPO_ROOT"
[[ -d "$DATA_ROOT" ]] || fail "DATA_ROOT is not a directory: $DATA_ROOT"
[[ -f "$REPO_ROOT/scripts/encode_spikes.py" ]] || fail "not a writingRing checkout: $REPO_ROOT"
[[ -f "$REPO_ROOT/scripts/segment_ring_imu.py" ]] || fail "segment_ring_imu.py is missing"
[[ -f "$REPO_ROOT/scripts/pad_segmented_imu.py" ]] || fail "pad_segmented_imu.py is missing"
[[ -f "$HELPER_SCRIPT" ]] || fail "wavelet reuse helper is missing: $HELPER_SCRIPT"
[[ -f "$ENCODER_SETTINGS" ]] || fail "encoder settings are missing: $ENCODER_SETTINGS"
[[ -f "$COMMON_REPAD_SCRIPT" ]] || fail "common repad script is missing: $COMMON_REPAD_SCRIPT"
[[ -f "$RECONSTRUCTION_SCRIPT" ]] || fail "reconstruction script is missing: $RECONSTRUCTION_SCRIPT"

[[ -n "$ACTION0_SOURCE_COMBINATION_ROOT" ]] || fail "ACTION0_SOURCE_COMBINATION_ROOT is required"
[[ -n "$ACTION1_SOURCE_COMBINATION_ROOT" ]] || fail "ACTION1_SOURCE_COMBINATION_ROOT is required"
[[ -d "$ACTION0_SOURCE_COMBINATION_ROOT" ]] || fail "Action 0 source root is missing: $ACTION0_SOURCE_COMBINATION_ROOT"
[[ -d "$ACTION1_SOURCE_COMBINATION_ROOT" ]] || fail "Action 1 source root is missing: $ACTION1_SOURCE_COMBINATION_ROOT"

case "$POST_ENCODE_TRANSFORM" in
    none|AbsRectify) ;;
    *) fail "POST_ENCODE_TRANSFORM must be none or AbsRectify" ;;
esac

case "$BOUNDARY_MODE" in
    label|aligned-board-events) ;;
    *) fail "BOUNDARY_MODE must be label or aligned-board-events" ;;
esac

case "$OVERWRITE_DEST" in
    0|1) ;;
    *) fail "OVERWRITE_DEST must be 0 or 1" ;;
esac

case "$RESUME" in
    0|1) ;;
    *) fail "RESUME must be 0 or 1" ;;
esac

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

cd "$REPO_ROOT"
mkdir -p "$DEST_PARENT_ROOT"

# ---------------------------------------------------------------------------
# SOURCE CONTRACT + SWEEP PLAN
# ---------------------------------------------------------------------------

read -r -a FIXED_CHANNELS <<<"$FIXED_WAVELET_CHANNELS_HZ"
[[ "${#FIXED_CHANNELS[@]}" -eq 4 ]] || fail "FIXED_WAVELET_CHANNELS_HZ must contain exactly four values"

SOURCE_INFO_JSON="$DEST_PARENT_ROOT/source_and_sweep_plan.json"

"${PYTHON_CMD[@]}" - \
    "$ACTION0_SOURCE_COMBINATION_ROOT" \
    "$ACTION1_SOURCE_COMBINATION_ROOT" \
    "$SWEEP_START" "$SWEEP_STEP" "$SWEEP_END" \
    "${FIXED_CHANNELS[@]}" > "$SOURCE_INFO_JSON" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

source0 = Path(sys.argv[1]).resolve()
source1 = Path(sys.argv[2]).resolve()
start = float(sys.argv[3])
step = float(sys.argv[4])
end = float(sys.argv[5])
backbone = tuple(float(x) for x in sys.argv[6:10])

if len(backbone) != 4 or len(set(backbone)) != 4:
    raise SystemExit("fixed backbone must contain four distinct frequencies")
if any((not math.isfinite(x)) or x <= 0 for x in backbone):
    raise SystemExit("fixed backbone values must be finite and positive")
if not all(math.isfinite(x) for x in (start, step, end)) or step <= 0:
    raise SystemExit("SWEEP_START/STEP/END must be finite and SWEEP_STEP > 0")
if end < start:
    raise SystemExit("SWEEP_END must be >= SWEEP_START")


def metadata_frequency_sets(root: Path) -> tuple[tuple[float, ...], ...]:
    paths = sorted((root / "spikeEncoding" / "custom-wavelet").rglob("metadata.json"))
    if not paths:
        raise SystemExit(f"no Custom Wavelet metadata.json files found below {root}")
    values = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = payload.get("spike_encoder")
        if not isinstance(spec, dict):
            spec = payload.get("encoder")
            if isinstance(spec, dict):
                spec = spec.get("settings") if isinstance(spec.get("settings"), dict) else spec
        if not isinstance(spec, dict) or "frequencies_hz" not in spec:
            # publication summaries also expose encoder.settings at top-level-ish locations;
            # recursively search only for an exact five-value frequencies_hz field.
            stack = [payload]
            found = None
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    candidate = current.get("frequencies_hz")
                    if isinstance(candidate, list) and len(candidate) == 5:
                        found = candidate
                        break
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
            if found is None:
                raise SystemExit(f"missing five-band frequencies_hz in {path}")
            raw = found
        else:
            raw = spec["frequencies_hz"]
        freq = tuple(float(x) for x in raw)
        if len(freq) != 5:
            raise SystemExit(f"{path}: expected five frequencies, got {freq}")
        values.append(freq)
    unique = tuple(dict.fromkeys(values))
    if len(unique) != 1:
        raise SystemExit(f"{root}: source recordings disagree on frequencies_hz: {unique}")
    return unique


def consume_backbone(freq: tuple[float, ...]) -> float:
    remaining = list(freq)
    for fixed in backbone:
        matches = [i for i, value in enumerate(remaining) if math.isclose(value, fixed, rel_tol=0, abs_tol=1e-9)]
        if not matches:
            raise SystemExit(f"source frequency bank {freq} is missing fixed backbone frequency {fixed:g} Hz")
        remaining.pop(matches[0])
    if len(remaining) != 1:
        raise SystemExit(f"could not identify one source fifth frequency from {freq} after consuming backbone {backbone}")
    return float(remaining[0])

freq0 = metadata_frequency_sets(source0)[0]
freq1 = metadata_frequency_sets(source1)[0]
if len(freq0) != len(freq1) or not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for a, b in zip(freq0, freq1)):
    raise SystemExit(f"Action0/Action1 source frequency mismatch: {freq0} vs {freq1}")
source_fifth = consume_backbone(freq0)

# Decimal-safe-ish integer-index generation. Values are rounded only for JSON/name
# stability; the original requested arithmetic is preserved to 12 decimals.
values = []
index = 0
while True:
    value = start + index * step
    if value > end + max(1e-12, abs(step) * 1e-9):
        break
    value = round(value, 12)
    if value <= 0:
        raise SystemExit(f"sweep produced non-positive frequency {value}")
    values.append(value)
    index += 1
if not values:
    raise SystemExit("sweep produced no values")
if len(values) != len(set(values)):
    raise SystemExit("sweep arithmetic produced duplicate fifth-frequency values")
if any(math.isclose(value, source_fifth, rel_tol=0, abs_tol=1e-9) for value in values):
    raise SystemExit(
        f"sweep includes the source/reference fifth frequency {source_fifth:g} Hz; "
        "the source baseline is added automatically, so remove that value from the sweep range"
    )

conditions = []
for fifth in values:
    bank = sorted([*backbone, fifth])
    conditions.append({
        "fifth_channel_hz": fifth,
        "frequencies_hz": bank,
        "duplicates_fixed_backbone_frequency": any(math.isclose(fifth, x, rel_tol=0, abs_tol=1e-9) for x in backbone),
    })

print(json.dumps({
    "schema": "writingring_fifth_wavelet_sweep_plan_v1",
    "action0_source_root": str(source0),
    "action1_source_root": str(source1),
    "source_frequencies_hz": list(freq0),
    "source_fifth_channel_hz": source_fifth,
    "fixed_wavelet_channels_hz": list(backbone),
    "sweep_start": start,
    "sweep_step": step,
    "sweep_end": end,
    "conditions": conditions,
}, indent=2))
PY

note "source/sweep plan: $SOURCE_INFO_JSON"

# Source pipeline preflight is invariant across sweep values, so do it once.
note "preflight Action 0 source reuse contract"
"${PYTHON_CMD[@]}" "$HELPER_SCRIPT" preflight \
    --source-root "$ACTION0_SOURCE_COMBINATION_ROOT" \
    --action 0 \
    --boundary-mode "$BOUNDARY_MODE"

note "preflight Action 1 source reuse contract"
"${PYTHON_CMD[@]}" "$HELPER_SCRIPT" preflight \
    --source-root "$ACTION1_SOURCE_COMBINATION_ROOT" \
    --action 1 \
    --boundary-mode "$BOUNDARY_MODE"

SOURCE_TRANSFORM0="$(
    "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" source-transform \
        --source-root "$ACTION0_SOURCE_COMBINATION_ROOT" --action 0
)"
SOURCE_TRANSFORM1="$(
    "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" source-transform \
        --source-root "$ACTION1_SOURCE_COMBINATION_ROOT" --action 1
)"
[[ "$SOURCE_TRANSFORM0" == "$SOURCE_TRANSFORM1" ]] || \
    fail "Action0/Action1 source post-encode transforms differ: $SOURCE_TRANSFORM0 vs $SOURCE_TRANSFORM1"
note "source post-encode transform: $SOURCE_TRANSFORM0"
note "requested sweep post-encode transform: $POST_ENCODE_TRANSFORM"

# Emit tab-separated fifth-frequency | bank | duplicate flag from the saved plan.
mapfile -t SWEEP_ROWS < <(
    "${PYTHON_CMD[@]}" - "$SOURCE_INFO_JSON" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
for item in p["conditions"]:
    print(f"{item['fifth_channel_hz']:g}\t" + " ".join(f"{x:g}" for x in item["frequencies_hz"]) + f"\t{int(item['duplicates_fixed_backbone_frequency'])}")
PY
)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

frequency_slug() {
    "${PYTHON_CMD[@]}" - "$1" <<'PY'
import sys
v = float(sys.argv[1])
text = f"{v:g}".replace("-", "m").replace(".", "p")
print(text + "Hz")
PY
}

# Runtime-only duplicate-aware Custom Wavelet encoding. The repository source is
# not changed. For non-duplicate banks, use the normal CLI unchanged.
encode_variant() {
    local input_root="$1"
    local output_root="$2"
    shift 2
    local frequencies=("$@")

    local has_duplicate=0
    if [[ "$(printf '%s\n' "${frequencies[@]}" | sort -g | uniq | wc -l)" -lt 5 ]]; then
        has_duplicate=1
    fi

    if [[ "$has_duplicate" -eq 0 ]]; then
        "${PYTHON_CMD[@]}" scripts/encode_spikes.py \
            --input-root "$input_root" \
            --pattern '*_preprocessedIMU.npy' \
            --output-root "$output_root" \
            --encoder custom-wavelet \
            --encoder-settings "$ENCODER_SETTINGS" \
            --encoder-frequencies-hz "${frequencies[@]}" \
            --post-encode-transform "$POST_ENCODE_TRANSFORM" \
            --overwrite
        return
    fi

    note "duplicate fifth frequency detected; using runtime duplicate-aware Custom Wavelet validator"
    "${PYTHON_CMD[@]}" - \
        "$input_root" "$output_root" "$ENCODER_SETTINGS" "$POST_ENCODE_TRANSFORM" \
        "${frequencies[@]}" <<'PY'
from __future__ import annotations

import math
import sys
from pathlib import Path

import writingring.spike_encoding.encoders.custom_wavelet as cw

input_root = Path(sys.argv[1])
output_root = Path(sys.argv[2])
settings_path = Path(sys.argv[3])
transform = sys.argv[4]
frequencies = tuple(float(x) for x in sys.argv[5:10])

if len(frequencies) != 5:
    raise SystemExit("internal error: duplicate-aware encode requires five frequencies")


def relaxed_post_init(self) -> None:
    if not isinstance(self.wavelet_name, str) or self.wavelet_name not in cw.WAVELET_REGISTRY:
        available = ", ".join(sorted(cw.WAVELET_REGISTRY))
        raise cw.CustomWaveletSettingsError(
            f"unknown wavelet_name {self.wavelet_name!r}; available wavelets: {available}"
        )
    sampling_rate = cw._positive_finite_float("sampling_rate_hz", self.sampling_rate_hz)
    try:
        raw = tuple(self.frequencies_hz)
    except TypeError as error:
        raise cw.CustomWaveletSettingsError("frequencies_hz must be an array of numbers") from error
    if len(raw) != 5:
        raise cw.CustomWaveletSettingsError("frequencies_hz must contain exactly five bands")
    parsed = tuple(float(value) for value in raw)
    if any((not math.isfinite(value)) or value <= 0.0 for value in parsed):
        raise cw.CustomWaveletSettingsError("frequencies_hz values must be finite and positive")
    if any(value >= sampling_rate / 2.0 for value in parsed):
        raise cw.CustomWaveletSettingsError("frequencies_hz values must be below the Nyquist frequency")
    if any(left > right for left, right in zip(parsed[:-1], parsed[1:])):
        raise cw.CustomWaveletSettingsError("frequencies_hz must be nondecreasing in Hz")
    object.__setattr__(self, "frequencies_hz", parsed)
    object.__setattr__(self, "sampling_rate_hz", sampling_rate)
    cw._positive_integer("prony_denominator_order", self.prony_denominator_order)
    cw._positive_integer("prony_numerator_order", self.prony_numerator_order)
    object.__setattr__(self, "max_filter_time_s", cw._positive_finite_float("max_filter_time_s", self.max_filter_time_s))
    if not cw._is_finite_real(self.max_filter_frequency_decades) or float(self.max_filter_frequency_decades) < 0.0:
        raise cw.CustomWaveletSettingsError("max_filter_frequency_decades must be finite and nonnegative")
    object.__setattr__(self, "max_filter_frequency_decades", float(self.max_filter_frequency_decades))
    if self.output_dtype not in {"float32", "float64"}:
        raise cw.CustomWaveletSettingsError("output_dtype must be 'float32' or 'float64'")
    cw._validate_post_encode_transform(self.post_encode_transform)
    widths = self.wavelet_widths_samples
    required = self.prony_denominator_order + self.prony_numerator_order
    if any(width <= required for width in widths):
        raise cw.CustomWaveletSettingsError("wavelet width must exceed the sum of the configured Prony orders")
    if len(widths) != 5 or any(width <= 0 for width in widths):
        raise cw.CustomWaveletSettingsError("wavelet widths must be positive and match frequencies")
    # Intentionally do NOT reject duplicate wavelet widths here.


cw.CustomWaveletSettings.__post_init__ = relaxed_post_init

original_metadata_getter = cw.CustomWaveletEncoder.encoding_metadata.fget
if original_metadata_getter is None:
    raise SystemExit("CustomWaveletEncoder.encoding_metadata is unavailable")


def relaxed_encoding_metadata(self):
    result = dict(original_metadata_getter(self))
    if len(set(self.settings.frequencies_hz)) < len(self.settings.frequencies_hz):
        result["frequency_order"] = "nondecreasing_hz_duplicates_allowed_for_sweep"
        result["duplicate_frequency_control"] = True
    return result


cw.CustomWaveletEncoder.encoding_metadata = property(relaxed_encoding_metadata)

from scripts.encode_spikes import main

argv = [
    "--input-root", str(input_root),
    "--pattern", "*_preprocessedIMU.npy",
    "--output-root", str(output_root),
    "--encoder", "custom-wavelet",
    "--encoder-settings", str(settings_path),
    "--encoder-frequencies-hz", *(f"{x:g}" for x in frequencies),
    "--post-encode-transform", transform,
    "--overwrite",
]
raise SystemExit(main(argv))
PY
}

# Import the existing helper and relax only the comparison policy so frequency,
# width, post-transform, and duplicate-frequency metadata may change while raw
# trailing IMU and timestamp provenance remain strict.
variant_helper() {
    local command="$1"
    local source_root="$2"
    local dest_root="$3"
    local action="$4"
    shift 4
    local frequencies=("$@")

    "${PYTHON_CMD[@]}" - \
        "$HELPER_SCRIPT" "$command" "$source_root" "$dest_root" "$action" \
        "$POST_ENCODE_TRANSFORM" "$BOUNDARY_MODE" "${frequencies[@]}" <<'PY'
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

spec = importlib.util.spec_from_file_location("_writingring_wavelet_variant_reuse", helper_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load helper: {helper_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def normalized_transform(meta: dict[str, Any]) -> str:
    value = mod.first_recursive(meta, ("post_encode_transform",))
    if value is None or value == "none":
        return "none"
    if value == "AbsRectify":
        return "AbsRectify"
    raise mod.ReuseError(f"unsupported post_encode_transform in metadata: {value!r}")


def strip_allowed_variant_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            lowered = key.lower()
            if "frequenc" in lowered or "width" in lowered:
                continue
            if lowered in {"post_encode_transform", "duplicate_frequency_control"}:
                continue
            out[key] = strip_allowed_variant_fields(child)
        return out
    if isinstance(value, list):
        return [strip_allowed_variant_fields(x) for x in value]
    return value


def relaxed_compare_new_encoding(source_records, dest_records, requested_freq):
    if set(source_records) != set(dest_records):
        missing = sorted(set(source_records) - set(dest_records))
        extra = sorted(set(dest_records) - set(source_records))
        raise mod.ReuseError(f"new encoding record set differs; missing={missing}, extra={extra}")
    replacements = {}
    rows = []
    for key in sorted(source_records):
        old = source_records[key]
        new = dest_records[key]
        if mod.frequencies(new.metadata) != requested_freq:
            raise mod.ReuseError(
                f"new frequencies mismatch for {new.user}/{new.dataset_id}: {mod.frequencies(new.metadata)}"
            )
        if normalized_transform(new.metadata) != requested_transform:
            raise mod.ReuseError(
                f"new post_encode_transform mismatch for {new.user}/{new.dataset_id}"
            )
        if strip_allowed_variant_fields(mod.encoder_spec(old.metadata)) != strip_allowed_variant_fields(mod.encoder_spec(new.metadata)):
            raise mod.ReuseError(
                f"new encoder changed semantics other than frequency/width/post-transform for {new.user}/{new.dataset_id}"
            )
        old_ts = mod.timestamp_hash(old.metadata)
        new_ts = mod.timestamp_hash(new.metadata)
        if old_ts is None or new_ts is None or old_ts != new_ts:
            raise mod.ReuseError(f"timestamp provenance changed for {new.user}/{new.dataset_id}")
        old_values = np.load(old.values_path, allow_pickle=False, mmap_mode="r")
        new_values = np.load(new.values_path, allow_pickle=False, mmap_mode="r")
        if old_values.shape != new_values.shape:
            raise mod.ReuseError(f"SpikeIMU shape changed for {new.user}/{new.dataset_id}")
        if not np.array_equal(old_values[:, mod.EVENT_CHANNEL_COUNT:], new_values[:, mod.EVENT_CHANNEL_COUNT:]):
            raise mod.ReuseError(f"trailing IMU channels 15:21 changed for {new.user}/{new.dataset_id}")
        old_values_hash = mod.sha256_file(old.values_path)
        new_values_hash = mod.sha256_file(new.values_path)
        old_meta_hash = mod.sha256_file(old.metadata_path)
        new_meta_hash = mod.sha256_file(new.metadata_path)
        replacements[old_values_hash] = (new_values_hash, f"{new.user}/{new.dataset_id}:values")
        replacements[old_meta_hash] = (new_meta_hash, f"{new.user}/{new.dataset_id}:metadata")
        rows.append({
            "user": new.user,
            "dataset_id": new.dataset_id,
            "source_transform": normalized_transform(old.metadata),
            "dest_transform": normalized_transform(new.metadata),
            "old_encoder_hash": mod.encoder_hash(old.metadata),
            "new_encoder_hash": mod.encoder_hash(new.metadata),
            "old_values_sha256": old_values_hash,
            "new_values_sha256": new_values_hash,
            "old_metadata_sha256": old_meta_hash,
            "new_metadata_sha256": new_meta_hash,
            "timestamp_sha256": new_ts,
            "trailing_imu_equal": True,
        })
    return rows, replacements

mod.compare_new_encoding = relaxed_compare_new_encoding

try:
    if command == "check-encoding":
        result = mod.check_encoding(argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            frequencies=requested_frequencies,
        ))
    elif command == "rebind-alignment":
        result = mod.rebind_alignment(argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            frequencies=requested_frequencies,
            overwrite=True,
        ))
    elif command == "verify":
        result = mod.verify(argparse.Namespace(
            source_root=source_root,
            dest_root=dest_root,
            action=action,
            boundary_mode=boundary_mode,
            frequencies=requested_frequencies,
        ))
        output_path = dest_root / "wavelet_variant_validation.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise mod.ReuseError(f"unsupported helper command: {command}")
    print(json.dumps(result, indent=2, sort_keys=True))
except (mod.ReuseError, OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(2)
PY
}

reuse_preprocessed_tree() {
    local source_root="$1"
    local dest_root="$2"
    local source_preprocess_root="$source_root/preprocessedIMU"
    local dest_preprocess_root="$dest_root/preprocessedIMU"

    [[ -d "$source_preprocess_root" ]] || fail "source preprocessedIMU root is missing: $source_preprocess_root"
    rm -rf -- "$dest_preprocess_root"
    mkdir -p "$dest_preprocess_root"

    local count=0 src rel dst
    while IFS= read -r -d '' src; do
        rel="${src#"$source_preprocess_root"/}"
        dst="$dest_preprocess_root/$rel"
        mkdir -p "$(dirname -- "$dst")"
        ln -s "$(realpath "$src")" "$dst"
        count=$((count + 1))
    done < <(find "$source_preprocess_root" -type f -print0)
    [[ "$count" -gt 0 ]] || fail "no reusable preprocessedIMU files found under $source_preprocess_root"
}

rebuild_action_variant() {
    local action="$1"
    local source_root="$2"
    local dest_root="$3"
    shift 3
    local frequencies=("$@")

    note "Action $action -> $dest_root"

    if [[ -e "$dest_root" ]]; then
        if [[ "$OVERWRITE_DEST" != "1" ]]; then
            fail "destination exists; set OVERWRITE_DEST=1 or use RESUME on a complete condition: $dest_root"
        fi
        case "$dest_root" in
            /|"$REPO_ROOT"|"$source_root") fail "refusing unsafe destination removal: $dest_root" ;;
        esac
        rm -rf -- "$dest_root"
    fi
    mkdir -p "$dest_root"

    reuse_preprocessed_tree "$source_root" "$dest_root"

    local dest_spike_output_root="$dest_root/spikeEncoding"
    local dest_spike_root="$dest_spike_output_root/custom-wavelet"
    local dest_segment_root="$dest_root/segmentation"
    local dest_padding_root="$dest_root/segmentation_padded"
    local dest_alignment_offset_root="$dest_root/alignment/offsets"

    encode_variant "$dest_root/preprocessedIMU" "$dest_spike_output_root" "${frequencies[@]}"
    variant_helper check-encoding "$source_root" "$dest_root" "$action" "${frequencies[@]}"

    if [[ "$BOUNDARY_MODE" == "aligned-board-events" ]]; then
        variant_helper rebind-alignment "$source_root" "$dest_root" "$action" "${frequencies[@]}"
    fi

    mkdir -p "$dest_segment_root"
    local user_count=0 user_dir user
    for user_dir in "$dest_spike_root"/*; do
        [[ -d "$user_dir" ]] || continue
        user="$(basename -- "$user_dir")"
        if [[ ! -d "$user_dir/$action" && ! -d "$user_dir/action_$action" ]]; then
            continue
        fi
        user_count=$((user_count + 1))

        segment_args=(
            "${PYTHON_CMD[@]}" scripts/segment_ring_imu.py
            --data-root "$DATA_ROOT"
            --user "$user"
            --action "$action"
            --input-kind spike-imu
            --spike-root "$dest_spike_root"
            --boundary-mode "$BOUNDARY_MODE"
            --sampling-rate "$SAMPLING_RATE"
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
        "${segment_args[@]}"
    done
    [[ "$user_count" -gt 0 ]] || fail "no users found under new spike root: $dest_spike_root"

    # First reproduce the source action's own padding target. Pair-level common
    # repadding below will then re-pad ONLY the smaller action, matching the
    # repository's current logic.
    local source_padding_target
    source_padding_target="$(
        "${PYTHON_CMD[@]}" "$HELPER_SCRIPT" padding-target --source-root "$source_root"
    )"
    [[ "$source_padding_target" != "NONE" ]] || fail "source has no padded dataset: $source_root"

    "${PYTHON_CMD[@]}" scripts/pad_segmented_imu.py \
        --input-root "$dest_segment_root" \
        --output-root "$dest_padding_root" \
        --target-length "$source_padding_target" \
        --sampling-rate "$SAMPLING_RATE" \
        --padding-value 0.0 \
        --overwrite

    variant_helper verify "$source_root" "$dest_root" "$action" "${frequencies[@]}"
}

condition_ready() {
    local action0_root="$1"
    local action1_root="$2"
    "${PYTHON_CMD[@]}" - "$action0_root" "$action1_root" "$REPO_ROOT" >/dev/null 2>&1 <<'PY'
import sys
from pathlib import Path
repo = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo))
from snn.accel_reconstruction_eval import load_acceleration_data
load_acceleration_data([Path(sys.argv[1]), Path(sys.argv[2])], repository_root=repo, require_reconstruction=True)
PY
}

# ---------------------------------------------------------------------------
# EXECUTE SWEEP
# ---------------------------------------------------------------------------

MANIFEST_JSON="$DEST_PARENT_ROOT/experiment_c_frequency_benchmark_inputs.json"
MANIFEST_ROWS="$DEST_PARENT_ROOT/.benchmark_manifest_rows.tsv"
: > "$MANIFEST_ROWS"

SOURCE_FIFTH="$("${PYTHON_CMD[@]}" - "$SOURCE_INFO_JSON" <<'PY'
import json, sys
print(f"{json.load(open(sys.argv[1], encoding='utf-8'))['source_fifth_channel_hz']:g}")
PY
)"
SOURCE_FREQUENCIES_TEXT="$("${PYTHON_CMD[@]}" - "$SOURCE_INFO_JSON" <<'PY'
import json, sys
print(" ".join(f"{x:g}" for x in json.load(open(sys.argv[1], encoding='utf-8'))['source_frequencies_hz']))
PY
)"

# The benchmark reference must use the same post-encode transform as every sweep
# condition. Reuse the source baseline only when its transform already matches;
# otherwise rebuild the source fifth-frequency condition once with the requested
# transform (the generated sweep itself may still start above 0.5 Hz).
if [[ "$SOURCE_TRANSFORM0" == "$POST_ENCODE_TRANSFORM" ]]; then
    BASELINE_ACTION0_ROOT="$(realpath "$ACTION0_SOURCE_COMBINATION_ROOT")"
    BASELINE_ACTION1_ROOT="$(realpath "$ACTION1_SOURCE_COMBINATION_ROOT")"
    condition_ready "$BASELINE_ACTION0_ROOT" "$BASELINE_ACTION1_ROOT" || \
        fail "source combination is not ready for Experiment C (reconstruction required)"
    note "reusing source combination as $SOURCE_FIFTH Hz benchmark baseline"
else
    baseline_slug="$(frequency_slug "$SOURCE_FIFTH")"
    BASELINE_ACTION0_ROOT="$DEST_PARENT_ROOT/$baseline_slug/action0/$PIPELINE_STAGE/$BOUNDARY_MODE"
    BASELINE_ACTION1_ROOT="$DEST_PARENT_ROOT/$baseline_slug/action1/$PIPELINE_STAGE/$BOUNDARY_MODE"
    read -r -a baseline_frequencies <<<"$SOURCE_FREQUENCIES_TEXT"
    note "rebuilding $SOURCE_FIFTH Hz baseline because transform changes: $SOURCE_TRANSFORM0 -> $POST_ENCODE_TRANSFORM"
    if [[ "$RESUME" == "1" ]] && condition_ready "$BASELINE_ACTION0_ROOT" "$BASELINE_ACTION1_ROOT"; then
        note "complete rebuilt baseline already exists; reusing $baseline_slug"
    else
        rebuild_action_variant 0 "$ACTION0_SOURCE_COMBINATION_ROOT" "$BASELINE_ACTION0_ROOT" "${baseline_frequencies[@]}"
        rebuild_action_variant 1 "$ACTION1_SOURCE_COMBINATION_ROOT" "$BASELINE_ACTION1_ROOT" "${baseline_frequencies[@]}"
        ENCODER_FREQUENCIES_HZ="$SOURCE_FREQUENCIES_TEXT" \
            SAMPLING_RATE="$SAMPLING_RATE" \
            bash "$COMMON_REPAD_SCRIPT" "$BASELINE_ACTION0_ROOT" "$BASELINE_ACTION1_ROOT"
        "${PYTHON_CMD[@]}" "$RECONSTRUCTION_SCRIPT" "$BASELINE_ACTION0_ROOT" --overwrite
        "${PYTHON_CMD[@]}" "$RECONSTRUCTION_SCRIPT" "$BASELINE_ACTION1_ROOT" --overwrite
        condition_ready "$BASELINE_ACTION0_ROOT" "$BASELINE_ACTION1_ROOT" || \
            fail "rebuilt baseline is not ready for Experiment C"
    fi
    BASELINE_ACTION0_ROOT="$(realpath "$BASELINE_ACTION0_ROOT")"
    BASELINE_ACTION1_ROOT="$(realpath "$BASELINE_ACTION1_ROOT")"
fi

printf 'baseline\t%s\t%s\t%s\n' \
    "$SOURCE_FIFTH" "$BASELINE_ACTION0_ROOT" "$BASELINE_ACTION1_ROOT" >> "$MANIFEST_ROWS"

for row in "${SWEEP_ROWS[@]}"; do
    IFS=$'\t' read -r fifth frequencies_text duplicate_flag <<<"$row"
    read -r -a frequencies <<<"$frequencies_text"
    slug="$(frequency_slug "$fifth")"

    action0_dest="$DEST_PARENT_ROOT/$slug/action0/$PIPELINE_STAGE/$BOUNDARY_MODE"
    action1_dest="$DEST_PARENT_ROOT/$slug/action1/$PIPELINE_STAGE/$BOUNDARY_MODE"

    note "============================================================"
    note "fifth channel: $fifth Hz"
    note "five-band bank: $frequencies_text Hz"
    note "duplicate control: $duplicate_flag"
    note "POST_ENCODE_TRANSFORM: $POST_ENCODE_TRANSFORM"
    note "============================================================"

    if [[ "$RESUME" == "1" ]] && condition_ready "$action0_dest" "$action1_dest"; then
        note "complete condition already exists; reusing $slug"
    else
        rebuild_action_variant 0 "$ACTION0_SOURCE_COMBINATION_ROOT" "$action0_dest" "${frequencies[@]}"
        rebuild_action_variant 1 "$ACTION1_SOURCE_COMBINATION_ROOT" "$action1_dest" "${frequencies[@]}"

        # Preserve the repository's current common-padding behavior: compare the
        # two action padding targets and re-pad only the smaller one.
        note "common repad: only the smaller action is re-padded when needed"
        ENCODER_FREQUENCIES_HZ="$frequencies_text" \
            SAMPLING_RATE="$SAMPLING_RATE" \
            bash "$COMMON_REPAD_SCRIPT" "$action0_dest" "$action1_dest"

        # Reconstruction must be after the final common padding geometry.
        note "reconstruct Action 0"
        "${PYTHON_CMD[@]}" "$RECONSTRUCTION_SCRIPT" "$action0_dest" --overwrite
        note "reconstruct Action 1"
        "${PYTHON_CMD[@]}" "$RECONSTRUCTION_SCRIPT" "$action1_dest" --overwrite

        condition_ready "$action0_dest" "$action1_dest" || fail "Experiment C input validation failed for $slug"
    fi

    printf 'sweep\t%s\t%s\t%s\n' \
        "$fifth" "$(realpath "$action0_dest")" "$(realpath "$action1_dest")" >> "$MANIFEST_ROWS"
done

"${PYTHON_CMD[@]}" - "$SOURCE_INFO_JSON" "$MANIFEST_ROWS" "$MANIFEST_JSON" "$POST_ENCODE_TRANSFORM" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
rows_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
transform = sys.argv[4]
plan = json.loads(plan_path.read_text(encoding="utf-8"))
conditions = []
for line in rows_path.read_text(encoding="utf-8").splitlines():
    kind, fifth, action0, action1 = line.split("\t")
    conditions.append({
        "kind": kind,
        "fifth_channel_hz": float(fifth),
        "action0_root": action0,
        "action1_root": action1,
    })
payload = {
    "schema": "writingring_experiment_c_frequency_benchmark_inputs_v1",
    "fixed_wavelet_channels_hz": plan["fixed_wavelet_channels_hz"],
    "source_frequencies_hz": plan["source_frequencies_hz"],
    "source_fifth_channel_hz": plan["source_fifth_channel_hz"],
    "sweep_start": plan["sweep_start"],
    "sweep_step": plan["sweep_step"],
    "sweep_end": plan["sweep_end"],
    "post_encode_transform": transform,
    "conditions": conditions,
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(out_path)
PY

rm -f -- "$MANIFEST_ROWS"

note "============================================================"
note "SWEEP COMPLETE"
note "Benchmark input manifest: $MANIFEST_JSON"
note "Set BENCHMARK_INPUT_MANIFEST to this JSON in experiment_C_frequency_benchmark.ipynb"
note "============================================================"
