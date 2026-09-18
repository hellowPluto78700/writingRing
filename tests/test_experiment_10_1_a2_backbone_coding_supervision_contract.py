from __future__ import annotations

from pathlib import Path

import torch

from scripts import experiment_10_1_a2_backbone_coding_supervision as exp101


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_factorial_run_contract() -> None:
    assert exp101.VARIANTS == ("original", "postencode_mask")
    assert exp101.CODINGS == ("bb", "mm")
    assert exp101.OBJECTIVES == (
        "l2_wcce",
        "l1_l2_joint",
        "l2_wcce_plus_0p1_l1_tsce",
    )
    assert exp101.ROTATIONS == (0, 1, 2, 3, 4)
    assert exp101.MODEL_SEEDS == (11, 23, 37)
    assert exp101.TSCE_LAMBDA == 0.1
    specs = exp101.run_specs()
    assert len(specs) == 180
    assert len({spec.key for spec in specs}) == 180
    assert specs[0].key == "original__bb__l2_wcce__rotation0__seed11"
    assert specs[-1].key == (
        "postencode_mask__mm__l2_wcce_plus_0p1_l1_tsce__rotation4__seed37"
    )


def test_bb_mm_coding_contract() -> None:
    bb = exp101.RunSpec("original", "bb", "l2_wcce", 0, 11)
    mm = exp101.RunSpec("original", "mm", "l2_wcce", 0, 11)
    assert bb.l1_coding == bb.l2_coding == "binary"
    assert mm.l1_coding == mm.l2_coding == "hetero3"
    assert exp101.THRESHOLD_MULTIPLIERS == (0.5, 1.0, 1.5)


def test_dataset_roots_only_cover_d0_d1() -> None:
    repo = Path("/repo")
    d0 = exp101._variant_roots(repo, "original")
    d1 = exp101._variant_roots(repo, "postencode_mask")
    assert len(d0) == len(d1) == 2
    assert str(d0[0]).endswith("aligned-board-events/segmentation_padded")
    assert str(d1[0]).endswith(
        "aligned-board-events_writing_motion_ablation/postencode_mask/segmentation_padded"
    )


def test_l1_tsce_is_valid_timestep_mean_ce() -> None:
    evidence = torch.tensor(
        [
            [[3.0, 0.0], [2.0, 0.0], [0.0, 3.0]],
            [[0.0, 3.0], [3.0, 0.0], [3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([2, 1], dtype=torch.long)
    y = torch.tensor([0, 1], dtype=torch.long)
    loss = exp101._l1_tsce(evidence, lengths, y)

    selected = torch.stack([evidence[0, 0], evidence[0, 1], evidence[1, 0]])
    targets = torch.tensor([0, 0, 1], dtype=torch.long)
    expected = torch.nn.functional.cross_entropy(selected, targets)
    assert torch.allclose(loss, expected)


def test_tsce_head_is_training_only_for_native_evidence() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_10_1_a2_backbone_coding_supervision.py"
    ).read_text()
    native_block = source[
        source.index("def _native_evidence"):source.index("def _loss_terms")
    ]
    assert "OBJECTIVE_JOINT" in native_block
    assert "OBJECTIVE_L1_TSCE" not in native_block
    loss_block = source[
        source.index("def _loss_terms"):source.index("def _evaluate_native")
    ]
    assert "TSCE_LAMBDA * aux" in loss_block


def test_joint_is_single_cooperative_evidence_path() -> None:
    source = (
        REPO_ROOT / "scripts" / "experiment_10_1_a2_backbone_coding_supervision.py"
    ).read_text()
    native_block = source[
        source.index("def _native_evidence"):source.index("def _loss_terms")
    ]
    assert "return l1 + l2" in native_block
    assert "F.cross_entropy" not in native_block


def test_model_init_is_factor_independent() -> None:
    specs = [
        exp101.RunSpec(variant, coding, objective, 3, 23)
        for variant in exp101.VARIANTS
        for coding in exp101.CODINGS
        for objective in exp101.OBJECTIVES
    ]
    seeds = {
        exp101.exp73._e2e_pair_seed(spec.seed, "model_init")
        for spec in specs
    }
    assert len(seeds) == 1


def test_probe_inventory_reuses_exp10_diagnostics() -> None:
    names = exp101.exp10.probe_names()
    assert len(names) == 22
    assert "l1__pre_reset__fixed250_ordered_mean" in names
    assert "l1__spike__fixed250_count" in names
    assert "l2__pre_reset__fixed250_ordered_mean" in names
    assert "l2__spike__fixed250_count" in names


def test_slurm_multi_cpu_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    prepare = (root / "prepare_exp_10_1_cpu.bash").read_text()
    run = (root / "run_exp_10_1_cpu_array.bash").read_text()
    submit = (root / "submit_exp_10_1_cpu.bash").read_text()
    finalize = (root / "finalize_exp_10_1_cpu.bash").read_text()

    assert "#SBATCH --array=0-179%20" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "MKL_NUM_THREADS=1" in run
    assert "OPENBLAS_NUM_THREADS=1" in run
    assert "NUMEXPR_NUM_THREADS=1" in run
    assert 'CUDA_VISIBLE_DEVICES=""' in run
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit
    assert " prepare" in prepare
    assert " run-one" in run
    assert " finalize" in finalize


def test_notebook_is_aggregation_only() -> None:
    path = REPO_ROOT / "notebooks" / "experiment_10_1_a2_backbone_coding_supervision.ipynb"
    notebook = path.read_text()
    assert "d0_d1_bb_mm_supervision_factorial_v1" in notebook
    assert "condition_summary.csv" in notebook
    assert "information_path_summary.csv" in notebook
    assert "contrast_summary.csv" in notebook
    assert "run_one(" not in notebook
    assert ".fit(" not in notebook
