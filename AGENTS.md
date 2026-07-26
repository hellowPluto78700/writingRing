# Project Instructions

- Use Python 3.11.
- Use pathlib.Path and type hints.
- Use Matplotlib, not Gradio, Streamlit, or Plotly.
- Never modify files under data_sample/.
- Ignore all *_ring_1.bin files.
- Board chunks must be sorted by numeric chunk index.
- Run pytest after code changes.
- Use the Conda environment writingring-viz.
- The sample data root is data_sample/data.
- The official pickle classes are under core/sensel_lib/.

## Upstream reference code

The original dataset-author code is located under:

- vendor/WritingRing/ring_plot.py
- vendor/WritingRing/board_plot.py
- vendor/WritingRing/core/

Treat vendor/WritingRing as read-only upstream reference code.

Before implementing or changing any parser:

1. Read the relevant upstream files.
2. Identify the exact data loading and transformation logic.
3. Inspect the real sample data.
4. Document any intentional differences from the upstream implementation.

Do not modify files under vendor/WritingRing.
Do not blindly copy the upstream plotting scripts. Reuse their verified data
semantics while implementing modular, tested code under src/writingring.
Do not infer undocumented units.
