"""Select one discovered WritingRing recording by its exact identity."""

from __future__ import annotations

from collections.abc import Iterable

from writingring.discovery import Recording


class RecordingSelectionError(ValueError):
    """Base error for an invalid or unsuccessful recording selection."""


class InvalidRecordingSelectorError(RecordingSelectionError):
    """Raised when a selector cannot identify a valid recording."""


class RecordingNotFoundError(RecordingSelectionError):
    """Raised when no discovered recording matches an exact selector."""


class MultipleRecordingsMatchedError(RecordingSelectionError):
    """Raised when discovery unexpectedly contains duplicate identities."""


def select_recording(
    recordings: Iterable[Recording],
    *,
    user: str,
    action: str,
    dataset_id: int,
) -> Recording:
    """Return exactly one recording matching user, action, and dataset ID."""

    if not isinstance(user, str) or not user:
        raise InvalidRecordingSelectorError("recording user must be nonempty")
    if not isinstance(action, str) or not action:
        raise InvalidRecordingSelectorError("recording action must be nonempty")
    if (
        isinstance(dataset_id, bool)
        or not isinstance(dataset_id, int)
        or dataset_id < 0
    ):
        raise InvalidRecordingSelectorError(
            "recording dataset ID must be a nonnegative integer"
        )

    matches = tuple(
        recording
        for recording in recordings
        if recording.user == user
        and recording.action == action
        and recording.dataset_id == dataset_id
    )
    identity = f"user={user!r}, action={action!r}, dataset_id={dataset_id}"
    if not matches:
        raise RecordingNotFoundError(f"no recording matches {identity}")
    if len(matches) > 1:
        raise MultipleRecordingsMatchedError(
            f"multiple recordings unexpectedly match {identity}: "
            f"{len(matches)} matches"
        )
    return matches[0]
