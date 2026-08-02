I want to redesign the current Board–Ring alignment candidate analysis.

Current behavior
----------------

The notebook currently processes each detected Board press event independently:

1. Read the Board press timestamp.
2. Add `coarse_offset_us` to estimate the corresponding Ring timestamp.
3. Find the nearest Ring sample.
4. Extract a Ring window around that sample.
5. Plot six IMU channels and `transient_score`.
6. Save one candidate figure per Board event.
7. Manually select one IMU sample per event and enter it in `MANUAL_ANCHORS`.

This per-event approach is too subjective. A large IMU transient in one window may be caused by writing motion, direction changes, or noise rather than physical Board contact.

Goal
----

Replace the primary candidate-selection workflow with a group-level analysis that uses all detected Board press events together.

The revised workflow should:

1. Extract one Ring IMU window around every Board press event using the current `coarse_offset_us`.
2. Express every window on the same relative-time axis.
3. Stack the windows into event-by-time matrices.
4. Compare all events jointly to identify repeatable IMU features associated with Board contact.
5. Generate overlay plots, robust summary curves, and heatmaps.
6. Estimate a candidate correction to `coarse_offset_us` from the cross-event consensus feature.
7. Keep the old single-event figures as an optional diagnostic tool.
8. Do not automatically declare every transient-score maximum to be a touch event.

Please first inspect the existing notebook/code and produce an implementation plan before editing files.

Important data and assumptions
------------------------------

- `press_events` is a DataFrame containing at least:
  - `event_index`
  - `frame_timestamp_raw`
- `ring_dataframe` contains six IMU columns listed in `SIGNAL_COLUMNS`.
- `ring_timestamp_reconstructed` is the reconstructed Ring timestamp array in microseconds.
- `transient_score` is already computed independently from the six-axis Ring IMU signals.
- `coarse_offset_us` maps Board time approximately to Ring time:

      expected_ring_time = board_time + coarse_offset_us

- `radius_samples` and `imu_sampling_rate_hz` define the candidate search window.
- Board and Ring timestamps are in microseconds.
- Ring timestamps may have small sampling irregularities, so group windows should be interpolated onto a common relative-time grid.
- Do not reorder or silently repair source data.
- Validate timestamp monotonicity before using `np.searchsorted` or interpolation.

Required implementation
-----------------------

1. Add a reusable nearest-timestamp helper

Create a helper such as:

    nearest_sorted_index(sorted_values, target)

Requirements:

- Input timestamps must be non-empty and strictly increasing.
- Use `np.searchsorted` rather than scanning the complete array for every event.
- Correctly handle targets before the first sample and after the last sample.
- Return the nearest global Ring sample index.

1. Build grouped candidate windows

Implement a function similar to:

    build_group_candidate_windows(...)

It should:

- Iterate over all rows in `press_events`.
- Compute:

      target_imu_timestamp =
          frame_timestamp_raw + coarse_offset_us

- Extract Ring data within a time-based radius around the target timestamp.
- Prefer timestamp bounds over assuming perfectly uniform sampling.
- Interpolate each six-axis signal and `transient_score` onto one common relative-time grid.
- Use a grid approximately equivalent to:

      np.arange(-radius_samples, radius_samples + 1)
      / imu_sampling_rate_hz

- Preserve NaN outside available data instead of extrapolating.
- Return:
  - common relative-time grid;
  - one `[N_events, T]` matrix per signal;
  - transient-score matrix `[N_events, T]`;
  - metadata DataFrame for all accepted events.

Metadata should include at least:

- `board_event_index`
- `board_timestamp_raw`
- `target_imu_timestamp`
- `center_imu_sample_index`
- `window_start_sample_index`
- `window_stop_sample_index_exclusive`
- nearest timestamp error in microseconds

Skip or clearly report events with insufficient Ring coverage.

1. Add robust per-event normalization

Implement a helper such as:

    robust_normalize_rows(values)

For each event independently:

- subtract the row median;
- divide by `1.4826 * MAD`;
- avoid division by zero;
- preserve NaNs.

This normalization is for comparing timing and waveform shape. It must not replace the original physical-unit signals.

1. Generate a group overlay summary

Create one figure with seven vertically stacked panels:

- six Ring IMU channels;
- transient score.

For each panel:

- plot every event as a thin, transparent line;
- plot the cross-event median as a prominent line;
- show the 25th–75th percentile range;
- draw a vertical reference line at relative time zero;
- use the same x-axis for every panel.

The x-axis label should clearly state that zero means:

    Board press timestamp + coarse_offset_us

The function should optionally plot:

- raw values; or
- per-event robust-normalized values.

Save the figure to a group-results directory.

1. Generate heatmaps

At minimum create a transient-score heatmap:

- rows: Board event indices;
- columns: common relative time;
- values: per-event robust-normalized transient score;
- vertical line at zero;
- event indices shown on the y-axis;
- include a colorbar.

Also consider six-axis heatmaps, either:

- one heatmap per signal; or
- an optional function controlled by configuration.

The purpose is to identify vertical bands that recur across many Board events.

1. Estimate a consensus offset correction

Create an initial candidate estimator based on the cross-event consensus transient score.

Suggested approach:

- robust-normalize each event’s transient-score window;
- calculate the pointwise cross-event median;
- search only within a configurable local interval around zero, for example:

      [-0.30 s, +0.30 s]

- find the strongest consensus feature in that interval;
- report:

      consensus_shift_s
      candidate_refined_offset_us =
          coarse_offset_us + consensus_shift_s * 1e6

Do not silently overwrite `coarse_offset_us`.

Label this result as a candidate refinement only, because the strongest common transient may not necessarily be physical contact.

1. Preserve single-event diagnostic plots

Keep or refactor the existing `plot_candidate_window()`.

It should remain available for:

- inspecting outlier events;
- checking events that do not match the group consensus;
- manually reviewing ambiguous candidate windows.

The group-level workflow should become the main analysis, while individual figures become diagnostics.

1. Improve manual-anchor workflow

Do not remove `MANUAL_ANCHORS`, but revise its role.

Manual anchors should be selected only after:

- inspecting group overlay plots;
- inspecting heatmaps;
- determining what repeatable IMU landmark is being used;
- checking that each selected event is consistent with the group feature.

Add validation/reporting that shows:

- available Board event indices;
- invalid manual anchors;
- duplicate Board event use;
- anchor offset distribution;
- residual from the median anchor offset.

Do not automatically renumber existing Board events without explicitly documenting it.

Validation and error handling
-----------------------------

Add explicit validation for:

- empty `press_events`;
- empty Ring data;
- missing signal columns;
- mismatched lengths among:
  - `ring_dataframe`
  - `ring_timestamp_reconstructed`
  - `transient_score`
- non-finite timestamps;
- non-monotonic Ring timestamps;
- invalid `radius_samples`;
- invalid `imu_sampling_rate_hz`;
- output directories that do not yet exist;
- candidate windows with fewer than two samples;
- candidate windows extending outside Ring coverage.

Avoid converting large timestamp integers to floating point earlier than necessary. If float conversion is required for interpolation, document why and confirm that relative timestamp differences remain precise enough.

Outputs
-------

The revised workflow should produce:

1. A candidate-window metadata DataFrame.
2. Group signal matrices with shape `[N_events, T]`.
3. A transient-score matrix with shape `[N_events, T]`.
4. A seven-panel group overlay figure.
5. A transient-score heatmap.
6. A printed candidate offset correction.
7. Optional per-event diagnostic figures.
8. A summary of skipped or invalid events.

Testing
-------

Add tests for reusable numerical helpers where practical.

At minimum test:

1. `nearest_sorted_index`
   - exact match;
   - midpoint between two samples;
   - before first sample;
   - after last sample;
   - empty input;
   - non-monotonic input.

2. Group-window extraction
   - expected output shapes;
   - correct relative-time grid;
   - correct metadata;
   - interpolation with irregular timestamps;
   - NaN behavior near boundaries;
   - skipped events with insufficient coverage.

3. Robust normalization
   - normal varying row;
   - constant row;
   - row containing NaNs;
   - output median approximately zero.

4. Consensus shift estimation
   - synthetic events with a known common shift;
   - outlier events that should not dominate the median;
   - search-range enforcement.

Acceptance criteria
-------------------

The implementation is complete when:

- all valid Board press events can be represented on one common Ring-relative time grid;
- group matrices have consistent event and time dimensions;
- overlay plots visibly distinguish individual events, median behavior, and interquartile range;
- the heatmap uses one row per Board event;
- the candidate offset correction is reproducible from the grouped data;
- the original single-event diagnostic plots remain usable;
- no source timestamps or source DataFrames are silently reordered, repaired, or modified;
- invalid events and data-quality problems are explicitly reported;
- tests pass.

Non-goals for this phase
------------------------

Do not yet implement:

- a trained touch classifier;
- automatic semantic labeling of IMU peaks as contact;
- dynamic time warping across complete writing sequences;
- a final affine clock-drift model unless existing anchor residuals clearly require it;
- automatic replacement of manually approved anchors.

After inspecting the repository, report:

1. which files or notebook cells should change;
2. existing utilities that can be reused;
3. proposed function signatures;
4. data-flow changes;
5. tests to add;
6. compatibility risks;
7. a step-by-step implementation order.

Do not edit files until the plan has been reviewed.
