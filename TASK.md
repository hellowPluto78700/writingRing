# WritingRing Local Visualization

## Goal

Build a local Python and Jupyter workflow that:

1. Discovers recordings by user, action, and dataset ID.
2. Loads *_ring_0.bin as float64 and reshapes it to (-1, 7).
3. Ignores *_ring_1.bin.
4. Loads board chunks in numerical order.
5. Plots IMU signals and touch trajectories with Matplotlib.
6. Provides command-line scripts and a notebook.
7. Includes tests.

## Implementation phases

1. Inspect actual sample files and board object structure.
2. Implement recording discovery.
3. Implement ring loader and validation.
4. Implement board loader and validation.
5. Implement Matplotlib plotting.
6. Implement command-line scripts.
7. Create notebook.
8. Add tests and README.

## Acceptance commands

```bash
python scripts/list_recordings.py --data-root data_sample/data

python scripts/plot_recording.py \
  --data-root data_sample/data \
  --user user_0 \
  --action 0 \
  --dataset-id 0 \
  --output-dir outputs/test \
  --no-show

pytest -q
