from __future__ import annotations

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    Path("notebooks/experiment_A_acceleration_cnn_representation_evaluation.ipynb"),
    Path("notebooks/experiment_B_reconstruction_frozen_cnn.ipynb"),
    Path("notebooks/experiment_C_reconstruction_trained.ipynb"),
    Path("notebooks/experiment_D_mixed_training.ipynb"),
)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_experiment_notebook_uses_shared_dataset_root_list(
    notebook_path: Path,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )

    assert "DATASET_ROOTS = [" in source
    assert "root=DATASET_ROOTS" in source
    assert "DATASET_ROOT =" not in source
    assert "action0_rectified" in source
    assert "action1_rectified" in source


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_experiment_notebook_has_no_stale_executed_outputs(
    notebook_path: Path,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []


@pytest.mark.parametrize("notebook_path", NOTEBOOKS[1:])
def test_downstream_notebook_explains_a_checkpoint_cohort_requirement(
    notebook_path: Path,
) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    assert "same selected dataset roots as the A checkpoint" in source
    assert "regenerating A" in source
