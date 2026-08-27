#!/usr/bin/env bash

# Validate and apply the known WritingRing raw-dataset corrections documented
# in README.md.
#
# The script is intentionally idempotent:
#   - already-correct state -> PASS
#   - known uncorrected state -> fix, verify, then PASS
#   - unexpected state -> fail rather than silently modifying unknown data
#
# Usage:
#   bash scripts/bash_script/preprocessing_pipeline/apply_known_raw_dataset_corrections.bash
#
# Optional override:
#   DATA_ROOT=/path/to/data bash scripts/bash_script/preprocessing_pipeline/apply_known_raw_dataset_corrections.bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

pass() {
    printf '[PASS] %s\n' "$1"
}

fix() {
    printf '[FIX ] %s\n' "$1"
}

fail() {
    printf '[FAIL] %s\n' "$1" >&2
    exit 1
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || fail "required file does not exist: ${path}"
}

printf 'WritingRing raw dataset correction check\n'
printf 'DATA_ROOT=%s\n\n' "$DATA_ROOT"

# ---------------------------------------------------------------------------
# 0. data/ must exist.
# ---------------------------------------------------------------------------
[[ -d "$DATA_ROOT" ]] || fail "data root does not exist: ${DATA_ROOT}"
pass "data root exists"

# ---------------------------------------------------------------------------
# 1. user_17 is unreliable and should be removed from the raw dataset.
# ---------------------------------------------------------------------------
USER17_PATH="${DATA_ROOT}/user_17"
if [[ -e "$USER17_PATH" || -L "$USER17_PATH" ]]; then
    fix "user_17 still exists; removing ${USER17_PATH}"
    rm -rf -- "$USER17_PATH"
    [[ ! -e "$USER17_PATH" && ! -L "$USER17_PATH" ]] \
        || fail "failed to remove ${USER17_PATH}"
    pass "user_17 removed"
else
    pass "user_17 already absent"
fi

# ---------------------------------------------------------------------------
# 2. user_4 / action 0 / recording 0:
#    remove the exact duplicate/wrong marker line
#      1720481135986401 wrong
# ---------------------------------------------------------------------------
USER4_TS="${DATA_ROOT}/user_4/0/0_timestamp.txt"
require_file "$USER4_TS"

BAD_USER4_LINE='1720481135986401 wrong'
if grep -Fqx -- "$BAD_USER4_LINE" "$USER4_TS"; then
    fix "removing known bad marker from ${USER4_TS}"
    "$PYTHON_BIN" - "$USER4_TS" "$BAD_USER4_LINE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
bad = sys.argv[2]
text = path.read_text()
lines = text.splitlines(keepends=True)
kept = [line for line in lines if line.rstrip("\r\n") != bad]
removed = len(lines) - len(kept)
if removed < 1:
    raise SystemExit(f"expected to remove at least one exact bad line from {path}")
path.write_text("".join(kept))
PY
fi

if grep -Fqx -- "$BAD_USER4_LINE" "$USER4_TS"; then
    fail "user_4 correction did not persist: ${USER4_TS}"
fi
pass "user_4 action 0 recording 0 timestamp correction is complete"

# ---------------------------------------------------------------------------
# 3. user_9 / action 0 / recording 0:
#    the unique 'l' marker timestamp must be 1720740976674429.
# ---------------------------------------------------------------------------
USER9_TS="${DATA_ROOT}/user_9/0/0_timestamp.txt"
require_file "$USER9_TS"

"$PYTHON_BIN" - "$USER9_TS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
target_timestamp = "1720740976674429"
text = path.read_text()
lines = text.splitlines(keepends=True)

matches = []
for i, line in enumerate(lines):
    stripped = line.rstrip("\r\n")
    parts = stripped.split()
    if len(parts) >= 2 and parts[-1] == "l":
        matches.append((i, parts, line[len(stripped):]))

if len(matches) != 1:
    raise SystemExit(
        f"[FAIL] expected exactly one 'l' marker in {path}, found {len(matches)}"
    )

idx, parts, newline = matches[0]
current_timestamp = parts[0]

if current_timestamp == target_timestamp:
    print("[PASS] user_9 'l' marker timestamp already corrected")
    raise SystemExit(0)

print(
    f"[FIX ] user_9 'l' marker timestamp: "
    f"{current_timestamp} -> {target_timestamp}"
)
parts[0] = target_timestamp
lines[idx] = " ".join(parts) + newline
path.write_text("".join(lines))

# Re-read and verify the exact postcondition.
verified = []
for line in path.read_text().splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[-1] == "l":
        verified.append(parts[0])

if verified != [target_timestamp]:
    raise SystemExit(
        f"[FAIL] user_9 correction verification failed in {path}: {verified}"
    )
print("[PASS] user_9 action 0 recording 0 timestamp correction is complete")
PY

# ---------------------------------------------------------------------------
# 4. user_3 / action 0 / recording 2:
#    2_board_0.gz must be a valid serialized empty Board chunk ([]).
#    The known bad raw file contains six frames. Anything else is unexpected
#    and is not overwritten automatically.
# ---------------------------------------------------------------------------
USER3_BOARD="${DATA_ROOT}/user_3/0/2_board_0.gz"
require_file "$USER3_BOARD"

"$PYTHON_BIN" - "$USER3_BOARD" <<'PY'
from pathlib import Path
import os
import sys
import tempfile

try:
    import compress_pickle
except ImportError as exc:
    raise SystemExit(
        "[FAIL] Python package 'compress_pickle' is required to validate/fix Board chunks. "
        "Activate the WritingRing environment first."
    ) from exc

path = Path(sys.argv[1])

try:
    frames = compress_pickle.load(path)
except Exception as exc:
    raise SystemExit(f"[FAIL] cannot load Board chunk {path}: {exc}") from exc

if not isinstance(frames, list):
    raise SystemExit(
        f"[FAIL] unexpected Board chunk type in {path}: {type(frames).__name__}; expected list"
    )

if len(frames) == 0:
    print("[PASS] user_3 2_board_0.gz is already a valid empty chunk")
    raise SystemExit(0)

if len(frames) != 6:
    raise SystemExit(
        f"[FAIL] unexpected non-empty Board chunk in {path}: {len(frames)} frames. "
        "Known uncorrected state is exactly 6 frames; refusing to overwrite unknown data."
    )

print(f"[FIX ] replacing known 6-frame Board chunk with serialized empty list: {path}")

fd, tmp_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
)
os.close(fd)
tmp_path = Path(tmp_name)

try:
    # Match the repository Board writer: open a binary file object and let
    # compress_pickle serialize/compress the list.
    with tmp_path.open("wb") as f:
        compress_pickle.dump([], f)

    verify = compress_pickle.load(tmp_path)
    if verify != []:
        raise RuntimeError(f"temporary replacement did not round-trip to []: {verify!r}")

    os.replace(tmp_path, path)
finally:
    if tmp_path.exists():
        tmp_path.unlink()

# Final verification from the published destination.
verify = compress_pickle.load(path)
if verify != []:
    raise SystemExit(f"[FAIL] final Board correction verification failed: {path}")

print("[PASS] user_3 action 0 recording 2 Board correction is complete")
PY

printf '\nAll known raw dataset corrections are complete.\n'
