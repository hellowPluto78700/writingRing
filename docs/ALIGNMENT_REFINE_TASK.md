Task: Replace Ring–Board alignment with press/lift event-to-transient peak matching

Objective

Replace the current manual-anchor-centered alignment workflow in notebooks/align_ring_board.ipynb with an automatic constant-offset alignment method.

The new method must:

Select the Board interval dynamically based on detected touch events.

Detect and retain both Board press and lift events.

Visualize the Ring IMU signals and Board events before synchronization.

Detect transient-score peak regions.

Shift the complete Board event sequence by one global constant offset.

Match as many Board press/lift events as possible to distinct transient-score peak regions.

Prefer matches near peak centers, while using peak magnitude only as a weak secondary signal.

Produce before/after plots, per-event match metadata, diagnostics, and tests.

Remove the current manual anchors as the authoritative source of the final offset.

Do not modify the original Ring or Board timestamps in place.

Current code to inspect

Start by reading:

notebooks/align_ring_board.ipynb

the writingring modules that provide:

Ring and Board loading;

recording discovery and selection;

reconstructed Ring timestamps;

transient-score-related helpers;

group candidate plotting and consensus-offset estimation.

existing tests covering Ring loading, Board loading, timestamp handling, plotting, and alignment.

Reuse existing validated loader behavior where possible. Put reusable numerical and matching logic in the writingring package rather than implementing everything only inside the notebook.

Required behavior

1. Detect Board occupancy transitions

Define Board occupancy per frame as:

occupied = board_frames["contact_count"].to_numpy() > 0

Detect:

press: False -> True;

lift: True -> False.

Do not automatically treat an initially occupied first frame as a newly observed press. The default transition calculation should preserve the initial state, for example:

transition = np.diff(
    occupied.astype(np.int8),
    prepend=occupied[0].astype(np.int8),
)

Equivalent correct implementations are acceptable.

Create a Board event table with at least:

event_index
event_type                 # "press" or "lift"
global_frame_index
frame_timestamp_raw
chunk_index
contact_count
paired_touch_index
duration_frames            # available for paired press/lift events
transient                  # short-contact/bounce diagnostic

Requirements:

Preserve chronological order.

Pair each press with the first subsequent lift before the next press.

Preserve incomplete contacts explicitly:

a press without a later lift must not be silently dropped;

mark its lift as missing in metadata.

Do not renumber events after filtering or validation.

Save all raw detected transitions for diagnostics.

Define a valid touch as a press/lift pair lasting at least 3 selected Board frames.

Shorter pairs may be marked as transient/bounce and excluded from interval selection and alignment, but must remain visible in diagnostic output.

1. Select the Board interval dynamically

Remove the fixed configured Board start and end timestamps.

The selected Board interval must start at the first available Board frame:

board_start_timestamp = first Board frame timestamp

Determine the interval end from valid press events:

If at least 20 valid presses exist:

select through 3 seconds after the 20th valid press.

If fewer than 20 valid presses exist:

select through 3 seconds after the last valid press.

If the requested end exceeds the available Board data:

clamp the interval end to the last Board frame.

If no valid press event exists:

raise a clear ValueError.

If the last selected press has a paired lift within the interval, include it.

If the corresponding lift occurs after the interval end, keep the exact requested interval rule and report that the selected contact is incomplete within the interval.

Use inclusive raw timestamp bounds for both Board frames and Board contacts.

Do not silently overwrite user configuration variables in a later notebook cell.

Save interval metadata including:

board_start_timestamp
desired_board_end_timestamp
actual_board_end_timestamp
valid_press_count_available
valid_press_count_selected
selected_event_count
selected_press_count
selected_lift_count
end_was_clamped
selected_interval_duration_s

1. Ring time axis and transient score

Retain the original Ring timestamp column.

Continue to create a separately named strictly increasing working time axis if the raw Ring timestamps contain repeated values.

Keep the current six-channel transient-score concept unless inspection reveals a tested reusable equivalent:

first differences of:

acc_x, acc_y, acc_z,

gyr_x, gyr_y, gyr_z;

per-channel median/MAD normalization;

Euclidean norm across six normalized channels.

Document clearly that transient_score is a motion-transient indicator, not a trained press/lift classifier.

Do not claim that a matched peak is definitively a physical press or lift.

1. Before-alignment visualization

Generate a seven-row shared-x figure for the first 30 seconds of Ring data:

acc_x

acc_y

acc_z

gyr_x

gyr_y

gyr_z

transient_score

Overlay unsynchronized Board events on every axis:

press events: one consistent style, such as solid red vertical lines;

lift events: a distinct style, such as dashed orange vertical lines.

Important:

Use the Ring reconstructed start timestamp as the common absolute-time reference.

Do not subtract the Board start timestamp independently, because that would hide the original offset.

Board event locations before alignment must be:

(board_event_timestamp - ring_start_timestamp) / 1_000_000.0

Limit the visible Ring signal range to the first 30 seconds.

Report how many press and lift events fall inside and outside the visible range.

Do not silently omit off-screen events without reporting their counts.

Save as:

outputs/alignment/before_alignment_imu_events_30s.png

1. Separate before-alignment transient/event figure

Generate a separate shared-x figure for the first 30 seconds containing:

upper axis: Ring transient_score;

lower axis: Board event raster or event stems with separate rows/styles for:

press;

lift.

This figure must make the original lack of synchronization visually obvious.

Save as:

outputs/alignment/before_alignment_transient_events_30s.png

Do not assign arbitrary continuous amplitudes to Board events. Represent them as discrete events.

1. Detect transient-score peak regions

Detect candidate transient regions from the Ring transient_score.

The implementation should:

optionally apply light smoothing intended only for peak detection;

detect local peaks using prominence-aware logic;

estimate a left and right region boundary for every peak;

merge very closely spaced peaks when they are likely part of one transient episode;

preserve the original unsmoothed transient score for plotting and reporting.

Each peak-region record must include at least:

peak_index
peak_timestamp
peak_elapsed_s
left_index
right_index
left_timestamp
right_timestamp
width_s
height
prominence

Avoid a raw-height-only threshold. Prefer robust statistics and prominence.

Make important peak-detection parameters configurable and record them in the alignment report.

1. Constant-offset candidate generation

The alignment model in this task is:

ring_time = board_time + offset

Estimate one global constant offset in microseconds.

Generate candidate offsets from differences between Board event times and transient peak centers:

candidate_offset = peak_timestamp - board_event_timestamp

Use both valid press and valid lift events when generating and evaluating candidates.

Reduce the candidate set through clustering, histogram voting, or another deterministic method before fine evaluation.

Do not perform an unnecessarily dense microsecond-by-microsecond brute-force scan across the entire recording.

Allow a configurable offset search range or derive a safe range from overlapping Ring and Board coverage.

1. Sequence-level event-to-peak matching

For each candidate offset, shift the complete Board event sequence:

shifted_event_time = board_event_time + candidate_offset

Match shifted Board events to transient peak regions.

The matching must satisfy all of the following:

chronological order is preserved;

one Board event matches at most one peak region;

one peak region matches at most one Board event;

press and lift events are both included;

an event is considered covered when its shifted timestamp lies inside the matched peak region;

unmatched Board events and unmatched peaks are allowed and explicitly reported;

no event may be silently assigned to a peak outside the accepted region.

Use dynamic programming or another deterministic monotonic one-to-one sequence-matching method.

Do not independently assign every Board event to its nearest peak without uniqueness and order constraints.

1. Match quality inside a peak region

For a shifted event time t, peak center p, and region [left, right], calculate a normalized within-region distance.

One acceptable definition is:

if t <= p:
    distance = (p - t) / max(p - left, epsilon)
else:
    distance = (t - p) / max(right - p, epsilon)

Interpretation:

distance = 0: event is at the peak center
distance = 1: event is at a peak-region boundary
distance > 1: event is outside the region and must not count as covered

Equivalent robust definitions are acceptable if documented and tested.

 1. Alignment objective

Use a coverage-first objective.

The primary goal is to match as many valid Board press/lift events as possible to transient peak regions.

Compare candidate offsets lexicographically in this order:

maximize matched Board event count;

maximize matched touch-pair completeness:

prefer offsets where both the press and lift of a valid touch are matched;

minimize median or total normalized distance to peak centers;

weakly prefer peaks with greater prominence;

prefer the simpler/smaller absolute offset only as a final deterministic tie-breaker, if needed.

Peak height or prominence must not dominate event coverage or temporal proximity.

Use a compressed contribution such as log1p(prominence) if prominence is included in a scalar diagnostic score.

Do not require 100% matching as a hard constraint. A missing Ring response must not force the entire Board sequence to an incorrect offset.

Report the best and second-best candidates so ambiguity can be evaluated.

 1. Press/lift pair diagnostics

For each valid Board touch pair, preserve:

press timestamp
lift timestamp
touch duration
matched press peak
matched lift peak
press residual
lift residual
both_events_matched

After alignment, check for these failure modes:

press consistently matching a later lift-like transient;

lift consistently matching the preceding press-like transient;

press and lift mapping to reversed peak order;

several touch pairs missing one side;

touch duration being incompatible with the time between matched peak centers.

Because transient_score has no event polarity or semantic label, do not claim that it inherently distinguishes press from lift.

The report must state that press/lift labels come from the Board and that Ring peaks are unlabeled motion transients.

 1. Alignment confidence and failure conditions

Calculate and report at least:

best_offset_us
second_best_offset_us
matched_event_count
total_valid_event_count
matched_press_count
total_valid_press_count
matched_lift_count
total_valid_lift_count
fully_matched_touch_pair_count
total_valid_touch_pair_count
event_coverage_ratio
press_coverage_ratio
lift_coverage_ratio
touch_pair_coverage_ratio
median_normalized_peak_distance
maximum_normalized_peak_distance
median_absolute_time_residual_ms
maximum_absolute_time_residual_ms
best_vs_second_best_margin

Define explicit configurable confidence thresholds.

At minimum, emit a warning when:

event coverage is too low;

fewer than a minimum number of valid touch pairs are available;

the best and second-best candidates are nearly tied;

press coverage and lift coverage differ substantially;

residuals show a strong trend over time;

a large fraction of events occur outside Ring coverage.

Do not label the result as reliable only because files were generated or assertions passed.

If no candidate produces a minimally acceptable alignment, raise or return a clear failed-alignment result rather than applying an arbitrary offset.

 1. Residual and clock-drift diagnostic

For each matched event:

residual = matched_peak_timestamp - shifted_board_event_timestamp

Plot residual versus Board event time, distinguishing press and lift.

Fit a diagnostic least-squares residual trend.

The task remains a constant-offset alignment task:

report possible clock drift;

do not automatically apply an affine correction;

do not silently change to ring_time = a * board_time + b.

Save as:

outputs/alignment/alignment_residuals_press_lift.png

 1. After-alignment visualization

Generate the same seven-row, first-30-seconds Ring figure after applying the selected offset to Board events.

Overlay:

shifted press events;

shifted lift events;

optionally emphasize matched versus unmatched events with distinct alpha or line style.

Use the same time range and signal scaling as the before-alignment figure wherever practical.

Save as:

outputs/alignment/after_alignment_imu_events_30s.png

Also generate the corresponding transient-score plus event-raster figure:

outputs/alignment/after_alignment_transient_events_30s.png

The before and after plots must be directly comparable.

Required output files

At minimum, produce:

outputs/alignment/
├── board_events_all.csv
├── board_events_selected.csv
├── board_touch_pairs.csv
├── board_interval_metadata.json
├── transient_peak_regions.csv
├── offset_candidates.csv
├── event_peak_matches.csv
├── touch_pair_matches.csv
├── alignment_report.json
├── before_alignment_imu_events_30s.png
├── before_alignment_transient_events_30s.png
├── after_alignment_imu_events_30s.png
├── after_alignment_transient_events_30s.png
└── alignment_residuals_press_lift.png

The exact internal filenames may be adjusted only if the notebook and tests are updated consistently.

Notebook changes

Refactor notebooks/align_ring_board.ipynb so that it:

configures the recording and algorithm parameters;

loads data through existing APIs;

calls reusable package functions;

displays key tables and figures;

explains the meaning and limitations of the results;

does not contain stale hard-coded manual anchors;

does not contain contradictory comments about interval duration or Board chunks;

does not silently overwrite configuration in later cells.

Remove or clearly deprecate the current MANUAL_ANCHORS constant-offset workflow.

The old group-candidate overlay may remain as an optional diagnostic, but it must not remain the authoritative alignment method.

Suggested reusable API structure

Inspect the existing package before naming or placing functions. A reasonable design may include equivalents of:

detect_board_events(...)
select_board_interval_from_presses(...)
pair_board_press_lift_events(...)
compute_transient_score(...)
detect_transient_peak_regions(...)
generate_offset_candidates(...)
match_shifted_events_to_peaks(...)
rank_alignment_candidates(...)
build_alignment_report(...)
plot_imu_with_board_events(...)
plot_transient_with_event_raster(...)
plot_alignment_residuals(...)

Use typed dataclasses or clearly documented DataFrames for structured results.

Avoid notebook-only global-state dependencies.

Testing requirements

Add or update automated tests for at least the following.

Board event detection

Detects press and lift from a normal occupancy sequence.

Does not invent a press when the first frame is already occupied.

Preserves an unmatched final press.

Flags a contact shorter than 3 frames as transient.

Correctly pairs multiple press/lift events in order.

Dynamic interval selection

Uses the 20th valid press plus 3 seconds when at least 20 presses exist.

Uses the final valid press plus 3 seconds when fewer than 20 exist.

Clamps to the final Board timestamp when less than 3 seconds remain.

Raises a clear error when no valid press exists.

Reports a final press whose lift lies beyond the selected interval.

Peak-region handling

Detects a synthetic peak and returns valid left/center/right ordering.

Merges configured near-duplicate peaks.

Does not let raw peak height alone determine the best alignment.

Correctly treats an event anywhere inside a peak region as covered.

Prefers the peak center over a boundary when coverage is equal.

Sequence matching

Enforces one event per peak.

Enforces one peak per event.

Preserves chronological order.

Handles unmatched events and peaks.

Includes both press and lift events.

Prefers more fully matched press/lift pairs when total event coverage is tied.

Does not force 100% matching when one Board event has no corresponding peak.

Offset recovery

Recovers a known synthetic constant offset within one Ring sample interval.

Remains stable when peak amplitudes vary strongly.

Remains stable with extra unrelated transient peaks.

Reports ambiguity when two offsets have near-equal sequence-level scores.

Reports residual trend without applying affine correction.

Plotting and outputs

Before-alignment plots use Ring start as the common time reference.

Press and lift use distinguishable visual styles.

Required output files are created and non-empty.

Event and peak metadata row counts are internally consistent.

The final report distinguishes structural success from alignment confidence.

Use deterministic synthetic signals for core matching tests.

Acceptance criteria

The task is complete when:

the selected Board interval follows the exact first-frame-to-20th-press-plus-3-seconds rule;

no-touch input raises an error;

both press and lift events are detected, paired, plotted, matched, and reported;

the before-alignment figures visibly preserve the original clock offset;

the final offset is selected by sequence-level event-to-peak matching;

matching is monotonic and one-to-one;

event coverage dominates peak amplitude in candidate ranking;

peak-center proximity breaks coverage ties;

press/lift pair completeness is explicitly evaluated;

the old stale manual anchors are no longer authoritative;

before/after plots and machine-readable reports are generated;

tests cover interval selection, event pairing, peak regions, matching constraints, offset recovery, ambiguity, and failure cases;

the full test suite passes.

Non-goals

Do not implement the following in this task:

a trained press/lift classifier;

dynamic time warping of the complete IMU signals;

nonlinear time warping;

automatic affine clock-drift correction;

modification of original raw timestamps;

automatic semantic labeling of Ring peaks as press or lift;

use of ring_1 unless the existing recording API explicitly requires it;

silent repair or reordering of source data beyond existing validated loader behavior.

Final implementation report

When finished, report:

files added or changed;

the final event and matching data model;

the dynamic interval-selection behavior;

the peak-region definition;

the candidate-ranking objective;

how press and lift pairing affects candidate ranking;

confidence and failure thresholds;

output artifacts;

tests added and their results;

known limitations and any remaining ambiguous assumptions.
