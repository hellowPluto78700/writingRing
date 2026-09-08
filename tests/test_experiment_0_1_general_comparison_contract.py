from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import experiment_0_1_general_comparison as exp01


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "experiment_0_1_general_comparison.ipynb"
ARRAY_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_general_comparison_cpu_array.bash"
BASELINE_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "run_exp_0_1_general_comparison_baselines_cpu.bash"
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash" / "submit_exp_0_1_general_comparison_cpu.bash"


def test_run_matrix_is_exactly_70_runs() -> None:
    specs = exp01.run_specs()
    assert len(specs) == 70
    assert len({spec.key for spec in specs}) == 70
    assert exp01.SEEDS == (11, 23, 37, 53, 71)
    assert exp01.OBJECTIVES == ("whole_count_ce", "timestep_ce")
    assert exp01.VARIANTS == (("binary", 1), ("multi_h", 31))

    direct = [spec for spec in specs if spec.family == exp01.DIRECT_FAMILY]
    probe = [spec for spec in specs if spec.family == exp01.PROBE_FAMILY]
    assert len(direct) == 60
    assert len(probe) == 10
    assert {spec.architecture for spec in direct} == set(exp01.DIRECT_ARCHITECTURES)
    assert {spec.objective for spec in direct} == set(exp01.OBJECTIVES)
    assert {spec.variant for spec in direct} == {"binary", "multi_h"}
    assert {spec.objective for spec in probe} == {"timestep_ce"}
    assert {spec.architecture for spec in probe} == {exp01.PROBE_ARCHITECTURE}


def test_architecture_shift_contracts_are_exact() -> None:
    assert exp01.DIRECT_ARCHITECTURES == {
        "short_mid_long": (
            (2, 3),
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6, 7),
        ),
        "short_mid": (
            (2, 3),
            (2, 3, 4, 5),
        ),
        "mid_long": (
            (2, 3, 4, 5),
            (2, 3, 4, 5, 6, 7),
        ),
    }
    assert exp01.PROBE_HIDDEN_SHIFTS == ((2, 3, 4), (2, 3, 4))
    assert exp01.HIDDEN_WIDTH == 128
    assert exp01.OUTPUT_CAP == 1


def test_direct_models_keep_output_binary_for_both_capacity_variants() -> None:
    for variant, hidden_cap in exp01.VARIANTS:
        spec = exp01.RunSpec(
            exp01.DIRECT_FAMILY,
            "short_mid",
            "whole_count_ce",
            variant,
            hidden_cap,
            11,
        )
        model = exp01.MultiTauHierarchySNN(
            exp01.hidden_shifts(spec),
            n_classes=12,
            fs=64.0,
            hidden_cap=hidden_cap,
            output_spiking=True,
        )
        assert all(lif.max_spikes_per_dt == hidden_cap for lif in model.hidden_lifs)
        assert model.output_lif is not None
        assert model.output_lif.max_spikes_per_dt == 1
        assert model.analog_head is None


def test_capacity_pair_changes_caps_not_seeded_weights() -> None:
    binary_spec = exp01.RunSpec(
        exp01.DIRECT_FAMILY, "short_mid", "whole_count_ce", "binary", 1, 11
    )
    multi_spec = exp01.RunSpec(
        exp01.DIRECT_FAMILY, "short_mid", "whole_count_ce", "multi_h", 31, 11
    )
    exp01.exp3.seed_all(exp01.paired_seed(binary_spec, "model_init"))
    binary = exp01.MultiTauHierarchySNN(
        exp01.hidden_shifts(binary_spec), 12, 64.0, 1, True
    )
    exp01.exp3.seed_all(exp01.paired_seed(multi_spec, "model_init"))
    multi = exp01.MultiTauHierarchySNN(
        exp01.hidden_shifts(multi_spec), 12, 64.0, 31, True
    )
    for name, value in binary.state_dict().items():
        assert torch.allclose(value, multi.state_dict()[name])


def test_probe_family_uses_analog_training_head_and_fixed250_final_readout() -> None:
    spec = exp01.RunSpec(
        exp01.PROBE_FAMILY,
        exp01.PROBE_ARCHITECTURE,
        "timestep_ce",
        "binary",
        1,
        11,
    )
    model = exp01.MultiTauHierarchySNN(
        exp01.hidden_shifts(spec), 12, 64.0, 1, output_spiking=False
    )
    assert model.output_linear is None
    assert model.output_lif is None
    assert model.analog_head is not None
    manifest = exp01.architecture_manifest(spec, 64.0, 12)
    assert manifest["output"]["training_only"] is True
    assert "Fixed250" in manifest["output"]["final_readout"]
    assert exp01.fixed_steps(250.0, 64.0) == 16


def test_array_baseline_submit_and_notebook_follow_repository_contract() -> None:
    array_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    baseline_text = BASELINE_SCRIPT.read_text(encoding="utf-8")
    submit_text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-69%50" in array_text
    assert "#SBATCH --cpus-per-task=1" in array_text
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert f"export {name}=1" in array_text
        assert f"export {name}=1" in baseline_text
    assert "module load conda/latest" in array_text
    assert "conda activate writingring-gpu" in array_text
    assert "run-one" in array_text
    assert "run-baselines" in baseline_text
    assert 'afterok:${ARRAY_JOB}:${BASELINE_JOB}' in submit_text

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
        "comparison_summary.csv",
        "summary.csv",
        "baseline_results.csv",
        "paired_capacity_effects.csv",
        "paired_objective_effects.csv",
        "paired_architecture_effects.csv",
        "short_mid_long",
        "short_mid",
        "mid_long",
        "Raw Fixed250 + Linear",
        "Raw Relative10 + Linear",
    ):
        assert token in joined
    for forbidden in ("optimizer.step(", ".backward()", "subprocess", "sbatch", "run-one", "run-baselines"):
        assert forbidden not in joined
