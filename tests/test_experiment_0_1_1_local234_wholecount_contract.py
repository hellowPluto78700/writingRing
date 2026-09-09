from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from scripts import experiment_0_1_1_local234_wholecount as exp011


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_1_local234_wholecount_cpu_array.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_1_local234_wholecount_cpu.bash"
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_1_local234_wholecount.ipynb"


def test_run_matrix_is_exactly_20_paired_runs() -> None:
    specs = exp011.run_specs()
    assert len(specs) == 20
    assert exp011.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp011.VARIANTS == (("binary", 1), ("multi_h", 31))
    assert exp011.SEEDS == (11, 23, 37, 53, 71)
    assert len({spec.key for spec in specs}) == 20
    for seed in exp011.SEEDS:
        seed_specs = [spec for spec in specs if spec.seed == seed]
        assert len(seed_specs) == 4
        assert {spec.objective for spec in seed_specs} == set(exp011.OBJECTIVES)
        assert {spec.variant for spec in seed_specs} == {"binary", "multi_h"}


def test_architecture_is_local_234x2_with_binary_output() -> None:
    assert exp011.ARCHITECTURE == "local_234x2"
    assert exp011.HIDDEN_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp011.HIDDEN_WIDTH == 128
    assert exp011.OUTPUT_CAP == 1

    binary = exp011.exp01.MultiTauHierarchySNN(
        layer_shifts=exp011.HIDDEN_SHIFTS,
        n_classes=12,
        fs=64.0,
        hidden_cap=1,
        output_spiking=True,
    )
    multi_h = exp011.exp01.MultiTauHierarchySNN(
        layer_shifts=exp011.HIDDEN_SHIFTS,
        n_classes=12,
        fs=64.0,
        hidden_cap=31,
        output_spiking=True,
    )
    assert len(binary.hidden_lifs) == 2
    assert [lif.max_spikes_per_dt for lif in binary.hidden_lifs] == [1, 1]
    assert [lif.max_spikes_per_dt for lif in multi_h.hidden_lifs] == [31, 31]
    assert binary.output_lif is not None
    assert multi_h.output_lif is not None
    assert binary.output_lif.max_spikes_per_dt == 1
    assert multi_h.output_lif.max_spikes_per_dt == 1


def test_paired_seed_namespace_matches_exp0_1() -> None:
    spec = exp011.RunSpec("whole_count_ce", "binary", 1, 23)
    parent = exp011.parent_spec(spec)
    assert exp011.paired_seed(spec, "model_init") == exp011.exp01.paired_seed(parent, "model_init")
    assert exp011.paired_seed(spec, "train_loader") == exp011.exp01.paired_seed(parent, "train_loader")


def _write_complete_artifacts(root: Path, spec: exp011.RunSpec) -> None:
    checkpoint = exp011.checkpoint_path(root, spec)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment_id": exp011.EXPERIMENT_ID,
            "protocol_version": exp011.PROTOCOL_VERSION,
            "spec": spec.__dict__,
            "best_epoch": 3,
            "model_state_dict": {"dummy": torch.tensor([1.0])},
        },
        checkpoint,
    )

    history = exp011.history_path(root, spec)
    history.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_objective_loss": 1.0,
                "val_output_whole_count_ba": 0.2,
                "val_output_whole_count_loss": 2.0,
            }
        ]
    ).to_csv(history, index=False)

    evaluation = exp011.evaluation_path(root, spec)
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    metric = {"accuracy": 0.3, "balanced_accuracy": 0.3, "macro_f1": 0.3, "loss": 1.5}
    evaluation.write_text(
        json.dumps(
            {
                "experiment_id": exp011.EXPERIMENT_ID,
                "protocol_version": exp011.PROTOCOL_VERSION,
                "spec": spec.__dict__,
                "primary_readout": "output_whole_count",
                "metrics": {"train": metric, "val": metric, "test": metric},
            }
        ),
        encoding="utf-8",
    )


def test_resume_requires_checkpoint_history_and_evaluation(tmp_path: Path) -> None:
    spec = exp011.RunSpec("timestep_ce", "multi_h", 31, 11)
    assert not exp011.run_complete(tmp_path, spec)

    _write_complete_artifacts(tmp_path, spec)
    assert exp011.checkpoint_complete(tmp_path, spec)
    assert exp011.history_complete(tmp_path, spec)
    assert exp011.evaluation_complete(tmp_path, spec)
    assert exp011.run_complete(tmp_path, spec)

    exp011.evaluation_path(tmp_path, spec).unlink()
    assert exp011.checkpoint_complete(tmp_path, spec)
    assert exp011.history_complete(tmp_path, spec)
    assert not exp011.evaluation_complete(tmp_path, spec)
    assert not exp011.run_complete(tmp_path, spec)


def test_identity_mismatch_does_not_count_as_complete(tmp_path: Path) -> None:
    spec = exp011.RunSpec("whole_count_ce", "binary", 1, 37)
    _write_complete_artifacts(tmp_path, spec)
    evaluation = exp011.evaluation_path(tmp_path, spec)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["spec"]["seed"] = 999
    evaluation.write_text(json.dumps(payload), encoding="utf-8")
    assert not exp011.evaluation_complete(tmp_path, spec)
    assert not exp011.run_complete(tmp_path, spec)


def test_array_submit_and_notebook_follow_repository_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-19%20" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in array_text
    assert "module load conda/latest" in array_text
    assert "conda activate writingring-gpu" in array_text
    assert "experiment_0_1_1_local234_wholecount run-one" in array_text
    assert "afterok:${ARRAY_JOB}" in submit_text

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    for token in (
        "summary.csv",
        "comparison_with_exp0_1.csv",
        "paired_objective_effects.csv",
        "paired_capacity_effects.csv",
        "local_234x2",
        "Output WholeCount",
    ):
        assert token in joined
    for forbidden in ("optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one"):
        assert forbidden not in joined
