from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = "notebooks/experiment_4_3_long_term_memory_validation.ipynb"


def _load_notebook() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    return json.loads((repo_root / NOTEBOOK).read_text(encoding="utf-8"))


def test_exp43_notebook_is_valid_analysis_only_notebook() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert len(cells) >= 10

    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []

    code = "\n".join("".join(cell["source"]) for cell in code_cells)
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "markdown"
    )

    assert "experiment_4_3_long_term_memory_validation" in code
    assert "stage2_recurrence_memory_validation_v1" in code
    for filename in (
        "classification_runs.csv",
        "classification_summary.csv",
        "retention_runs.csv",
        "retention_summary.csv",
        "paired_effects.csv",
        "paired_effects_summary.csv",
        "manifest.json",
    ):
        assert filename in code

    for effect in (
        "rsnn_minus_ff",
        "rsnn_normal_minus_recurrent_off",
        "normal_minus_periodic_reset",
        "silent_delay_drop_from_0ms",
    ):
        assert effect in code

    assert "matplotlib.pyplot" in code
    assert "Uend" in markdown
    assert "V1" in markdown and "V2" in markdown and "V3" in markdown and "V4" in markdown

    forbidden = (
        "train_ff(",
        "run_ff_one(",
        "eval-rsnn-one",
        "sbatch ",
        "subprocess.",
    )
    for token in forbidden:
        assert token not in code


def test_exp43_notebook_keeps_primary_memory_metric_and_pairing_visible() -> None:
    notebook = _load_notebook()
    all_text = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "Uend + Linear" in all_text
    assert "mean ± SD" in all_text
    assert "NoReset - Reset" in all_text
    assert "RSNN - FF" in all_text
    assert "EXPECTED_SEEDS = (11, 23, 37, 53, 71)" in all_text
    assert "EXPECTED_DELAYS_MS = (0, 250, 500, 1000, 2000)" in all_text
    assert "EXPECTED_RESET_MS = (250, 500, 1000)" in all_text
