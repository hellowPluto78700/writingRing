from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import experiment_10_0_airborne_motion_ablation as exp10


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_factorial_contract() -> None:
    assert exp10.VARIANTS == (
        "original",
        "postencode_mask",
        "masked_accel_reencode",
    )
    assert exp10.ROTATIONS == (0, 1, 2, 3, 4)
    assert exp10.MODEL_SEEDS == (11, 23, 37)
    specs = exp10.run_specs()
    assert len(specs) == 45
    assert len({spec.key for spec in specs}) == 45
    assert specs[0].key == "original__rotation0__seed11"
    assert specs[14].key == "original__rotation4__seed37"
    assert specs[15].key == "postencode_mask__rotation0__seed11"
    assert specs[30].key == "masked_accel_reencode__rotation0__seed11"
    assert specs[-1].key == "masked_accel_reencode__rotation4__seed37"


def test_probe_inventory() -> None:
    names = exp10.probe_names()
    assert len(names) == 22
    assert "input__events__whole_count" in names
    assert "input__events__fixed250_count" in names
    for layer in ("l1", "l2"):
        for state in ("syn_current", "pre_reset", "spike", "post_reset"):
            assert f"{layer}__{state}__whole_mean" in names
            assert f"{layer}__{state}__fixed250_ordered_mean" in names
        assert f"{layer}__spike__whole_count" in names
        assert f"{layer}__spike__fixed250_count" in names


def test_variant_roots_cover_both_actions() -> None:
    repo = Path("/repo")
    original = exp10._variant_roots(repo, exp10.VARIANT_ORIGINAL)
    post = exp10._variant_roots(repo, exp10.VARIANT_POSTENCODE)
    reencoded = exp10._variant_roots(repo, exp10.VARIANT_REENCODE)
    assert len(original) == len(post) == len(reencoded) == 2
    assert "action0_wavelets" in str(original[0])
    assert "action1_wavelets" in str(original[1])
    assert str(post[0]).endswith(
        "aligned-board-events_writing_motion_ablation/postencode_mask/segmentation_padded"
    )
    assert str(reencoded[1]).endswith(
        "aligned-board-events_writing_motion_ablation/masked_accel_reencode/segmentation_padded"
    )


def test_geometry_comparison_only_allows_feature_value_changes() -> None:
    left = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "user": ["user_0", "user_1"],
            "action": [0, 1],
            "label": ["A", "B"],
            "valid": [10, 11],
            "pad": [16, 16],
            "source_trial": ["t0", "t1"],
        }
    )
    exp10._assert_paired_geometry(
        left, left.copy(), reference_name="D0", candidate_name="D1"
    )
    right = left.copy()
    right.loc[1, "valid"] = 12
    with pytest.raises(RuntimeError, match="geometry mismatch"):
        exp10._assert_paired_geometry(
            left, right, reference_name="D0", candidate_name="D1"
        )


def test_variant_is_excluded_from_paired_model_and_probe_seeds() -> None:
    specs = [
        exp10.RunSpec(variant, 2, 23)
        for variant in exp10.VARIANTS
    ]
    model_seeds = {
        exp10.exp73._e2e_pair_seed(spec.seed, "model_init") for spec in specs
    }
    probe_seeds = {
        exp10._probe_seed(spec, "l2__spike__fixed250_count") for spec in specs
    }
    assert len(model_seeds) == 1
    assert len(probe_seeds) == 1


def test_slurm_multi_cpu_contract() -> None:
    root = REPO_ROOT / "scripts" / "bash_script" / "SNN_Bash"
    prepare = (root / "prepare_exp_10_0_cpu.bash").read_text()
    run = (root / "run_exp_10_0_cpu_array.bash").read_text()
    submit = (root / "submit_exp_10_0_cpu.bash").read_text()
    finalize = (root / "finalize_exp_10_0_cpu.bash").read_text()

    assert "#SBATCH --array=0-44%45" in run
    assert "#SBATCH --cpus-per-task=1" in run
    assert "OMP_NUM_THREADS=1" in run
    assert "CUDA_VISIBLE_DEVICES=\"\"" in run
    assert 'afterok:${prepare_job}' in submit
    assert 'afterok:${array_job}' in submit
    assert " prepare" in prepare
    assert " run-one" in run
    assert " finalize" in finalize
