#!/usr/bin/env python3
"""CLI and import surface for the 64 Hz 66-channel AngularAccel variant."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.angular_accel66.common import (  # noqa: E402,F401
    DERIVATIVE_METHOD,
    FREQUENCIES_HZ,
    MAX_FILTER_TIME_S,
    OUTPUT_CHANNELS,
    OUTPUT_EVENT_SCHEMA,
    OUTPUT_EVENTS,
    OUTPUT_SCHEMA,
    RATE_HZ,
    SOURCE_CHANNELS,
    SOURCE_EVENTS,
    angular_acceleration_from_gyro,
    angular_encoder,
    combine_event_branches,
)
from scripts.angular_accel66.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
