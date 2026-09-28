"""Focused mathematical, artifact, scheduler, and notebook contracts."""
from __future__ import annotations
import ast
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from core_benchmark_v1.aggregate import finalize, probe_seed_table
from core_benchmark_v1.data import choose_users, prepare, synthetic_source, validate_arrays
from core_benchmark_v1.model import BenchmarkNet, lif_step, mean_logits, sequence_loss, spike_readout, valid_sum
from core_benchmark_v1.probes import fit_probe, probe_specs, temporal_features
from core_benchmark_v1.protocol import Protocol, Run, aliases, phase_runs, runs
from core_benchmark_v1.runner import required_artifacts, run_one, smoke_protocol
from core_benchmark_v1.storage import file_hash, load_lock, load_torch, read_json, save_json, validate_complete

REPO = Path(__file__).resolve().parents[1]


def test_primary_manifest_and_true_dependencies() -> None:
    p = Protocol()
    manifest = runs(p)
    assert len(manifest) == 36
    assert len(phase_runs(p, 1)) == 30
    assert len(phase_runs(p, 2)) == 6
    assert not phase_runs(p, 3)
    assert len({r.key for r in manifest}) == len(manifest)
    assert set(aliases().values()) == {'O0'}
    assert set(r.case for r in manifest if r.kind == 'backbone') == {'O0', 'O1', 'O2', 'O3', 'T1', 'T2', 'T3', 'T4', 'D1'}
    assert set(r.case for r in manifest if r.kind == 'e2e_readout') == {'R_LIF_E2E'}
    optional = runs(replace(p, depth_controls=True, membrane_sweep=True))
    assert len(optional) == 51
    assert all(r.parent_key is None or next(parent.phase for parent in optional if parent.key == r.parent_key) < r.phase for r in optional)


@pytest.mark.parametrize('changes', [
    {'synaptic_update': 'normalized'}, {'input_drive_scale': 0.25}, {'native_bias': True},
    {'wcce_reduction': 'sum'}, {'tsce_reduction': 'flatten_valid'}, {'seeds': (11, 11)},
    {'train_users': ('user_0',)}, {'max_epochs': 1}, {'c_grid': (0.0,)},
    {'shuffle_seeds': ()}, {'output_betas': (0.7,)}, {'probe_states': ('current',)},
    {'steps': 255}, {'version': 'v0'}, {'train_users': ('u',), 'val_users': ('u',), 'test_users': ('x',)},
])
def test_invalid_protocol_fails(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(Protocol(), **changes).validate()


def test_json_roundtrip_and_unknown_fields() -> None:
    p = Protocol.load(REPO / 'core_benchmark_v1/00_protocol/default.json')
    assert p == Protocol()
    assert Protocol.from_dict(json.loads(json.dumps(asdict(p)))).fingerprint == p.fingerprint
    with pytest.raises(ValueError, match='Unknown protocol'):
        Protocol.from_dict({'random_typo': 3})


def test_user_split_is_locked_and_independent_of_model_seed() -> None:
    users = [f'user_{i}' for i in range(20)]
    actual = choose_users(users, Protocol())
    assert [len(actual[s]) for s in ('train', 'val', 'test')] == [14, 3, 3]
    shuffled = np.array(sorted(users), dtype=str)
    np.random.default_rng(12345).shuffle(shuffled)
    assert set(actual['test']) == set(shuffled[17:])
    explicit = replace(Protocol(), train_users=actual['train'], val_users=actual['val'], test_users=actual['test'])
    assert choose_users(list(reversed(users)), explicit) == actual


def test_paired_initialization_does_not_depend_on_auxiliary_or_depth() -> None:
    p = smoke_protocol()
    selected = {r.case: r for r in runs(p)}
    base = BenchmarkNet(selected['O0'], p)
    for case in ('O1', 'O2', 'O3', 'T1', 'T3', 'D1', 'R_LIF_E2E'):
        candidate = BenchmarkNet(selected[case], p)
        for name, weight in base.named_parameters():
            assert torch.equal(weight, dict(candidate.named_parameters())[name]), (case, name)
    assert not torch.equal(base.alpha_0, BenchmarkNet(selected['T1'], p).alpha_0)


def test_unnormalized_synapse_equation_numerically() -> None:
    p = replace(smoke_protocol(), width=1, input_channels=1, threshold=10.0, readout_thresholds=(10.0,))
    model = BenchmarkNet(Run('O0', 11, '01_objective'), p)
    with torch.no_grad():
        model.layers[0].weight.fill_(0.1)
    x = torch.zeros(1, p.steps, 1)
    x[0, :3, 0] = torch.tensor([1.0, 2.0, 0.0])
    tr = model(x, torch.tensor([3]))
    expected = 0.75 * (0.75 * 0.1 + 0.2)
    assert torch.allclose(tr['final_syn'][0], torch.tensor([[expected]]), atol=1e-7)
    assert not torch.allclose(tr['final_syn'][0], torch.tensor([[expected * 0.25]]))


def historical_lif_class() -> type:
    # Execute the exact reference primitives without importing unrelated experiment drivers.
    path = REPO / 'scripts/experiment_4_0_1_multispike_macro_lif.py'
    tree = ast.parse(path.read_text())
    names = {'_MultiThresholdSpike', 'multi_threshold_spike', 'MacroMultiSpikeLIF'}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace: dict[str, Any] = {'torch': torch, 'nn': nn, 'SURROGATE_SLOPE': 25.0}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['MacroMultiSpikeLIF']


@pytest.mark.parametrize('beta', [0.0, 0.5, 1.0])
def test_neuron_forward_and_backward_match_repository(beta: float) -> None:
    current = torch.tensor([[-1.0, 0.0, 0.4, 0.5, 0.8, 2.5]], requires_grad=True)
    membrane = torch.tensor([[0.1, -0.2, 0.3, 0.0, -0.4, 0.6]], requires_grad=True)
    a = lif_step(current, membrane, beta, 0.5, 25.0)
    ga = torch.autograd.grad(sum(v.sum() for v in a), (current, membrane), retain_graph=True)
    reference = historical_lif_class()(beta=beta, threshold=0.5, max_spikes_per_dt=1, surrogate_slope=25)
    b = reference(current, membrane)
    gb = torch.autograd.grad(sum(v.sum() for v in b), (current, membrane))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(torch.allclose(x, y) for x, y in zip(ga, gb))


def test_padding_cannot_change_valid_state_or_accumulator() -> None:
    p = smoke_protocol()
    model = BenchmarkNet(Run('O0', 11, '01_objective'), p)
    x = torch.rand(2, p.steps, p.input_channels)
    lengths = torch.tensor([5, 19])
    alternate = x.clone()
    alternate[0, 5:] = 1e6
    alternate[1, 19:] = -1e6
    a, b = model(x, lengths), model(alternate, lengths)
    assert torch.equal(a['evidence'], b['evidence'])
    assert a['spike'][0][0, 5:].count_nonzero() == 0
    assert a['evidence'][1, 19:].count_nonzero() == 0
    assert torch.equal(valid_sum(a['evidence'], lengths).argmax(1), mean_logits(a['evidence'], lengths).argmax(1))


def test_tsce_equal_sample_weight_and_masked_wcce() -> None:
    values = torch.tensor([[[3., 0., 0.], [100., -100., 0.], [100., -100., 0.]],
                           [[0., 0., 1.], [0., 0., 2.], [0., 0., 3.]]], requires_grad=True)
    y, lengths = torch.tensor([0, 1]), torch.tensor([1, 3])
    expected = (F.cross_entropy(values[0, :1], y[:1]) + F.cross_entropy(values[1], y[1:].expand(3))) / 2
    actual = sequence_loss(values, lengths, y, 'tsce')
    assert torch.allclose(actual, expected)
    actual.backward()
    assert values.grad[0, 1:].count_nonzero() == 0
    assert torch.allclose(sequence_loss(values, lengths, y, 'wcce'), F.cross_entropy(torch.stack((values[0, 0], values[1].mean(0))), y))


def test_if_reset_sign_and_padding() -> None:
    e = torch.tensor([[[0.6], [-1.0], [0.6], [0.6], [100.0]]])
    lengths = torch.tensor([4])
    out = spike_readout(e, lengths, beta=1.0, threshold=0.5, slope=25)
    assert out.flatten().tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    # Reset is subtractive, negative membrane is retained, and padding cannot fire.
    assert torch.all((out == 0) | (out == 1))


def test_temporal_shuffle_preserves_partial_padding_and_counts() -> None:
    p = replace(smoke_protocol(), steps=64)
    z = np.repeat(np.arange(1, 5, dtype=np.float32), 16)[None, :, None].repeat(20, axis=0)
    lengths = np.full(20, 40)
    ids = np.array([f'sample_{i}' for i in range(20)])
    ordered = temporal_features(z, lengths, ids, 'fixed250_ordered', p)
    shuffled = temporal_features(z, lengths, ids, 'fixed250_shuffled', p, 101)
    assert np.array_equal(ordered[:, 2:], shuffled[:, 2:])
    assert np.all(ordered[:, 2] == 24)
    assert np.all(ordered[:, 3] == 0)
    assert np.array_equal(ordered.sum(1), shuffled.sum(1))
    assert len(np.unique(shuffled[:, :2], axis=0)) == 2
    assert np.array_equal(shuffled, temporal_features(z, lengths, ids, 'fixed250_shuffled', p, 101))
    whole = temporal_features(z, lengths, ids, 'whole_count', p)
    relative = temporal_features(z, lengths, ids, 'relative10_ordered', p)
    assert np.array_equal(relative.sum(1), whole[:, 0])
    assert np.array_equal(ordered.sum(1), whole[:, 0])


def test_scaler_has_no_hidden_bias_and_uses_train_only() -> None:
    p = smoke_protocol()
    x = np.array([[1, 0], [2, 0], [0, 1], [0, 2], [2, 2], [3, 3]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])
    scale_nb, model_nb, _ = fit_probe(x, y, 100 * x, y, 'no_bias', p)
    scale_aff, _, _ = fit_probe(x, y, 100 * x, y, 'affine', p)
    assert scale_nb.with_mean is False and scale_aff.with_mean is False
    assert np.array_equal(scale_nb.scale_, scale_aff.scale_)
    assert np.allclose(scale_nb.scale_, x.std(axis=0))
    assert np.all(model_nb.intercept_ == 0)


def test_cache_validation_rejects_user_leakage_and_padding() -> None:
    p = smoke_protocol()
    arrays, _ = synthetic_source(p)
    validate_arrays(arrays, p)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed['test_users'][:] = arrays['train_users'][0]
    with pytest.raises(ValueError, match='leakage'):
        validate_arrays(changed, p)
    changed = {k: v.copy() for k, v in arrays.items()}
    i = int(np.flatnonzero(changed['train_lengths'] < p.steps)[0])
    changed['train_x'][i, changed['train_lengths'][i], 0] = 1
    with pytest.raises(ValueError, match='padding|padded'):
        validate_arrays(changed, p)


def test_shuffles_are_averaged_inside_model_seed() -> None:
    rows = []
    for seed, values in [(11, [0.2, 0.4]), (23, [0.6, 0.8])]:
        for value in values:
            rows.append({'case': 'O0', 'block': '01_objective', 'layer': 'L1', 'state': 'spike',
                'aggregation': 'fixed250_shuffled', 'decoder': 'no_bias', 'seed': seed,
                'train_ba': value, 'val_ba': value, 'test_ba': value, 'train_test_gap': 0})
    table = probe_seed_table(pd.DataFrame(rows))
    assert len(table) == 2
    assert np.allclose(table.test_ba, [0.3, 0.7])


@pytest.fixture(scope='module')
def smoke_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp('core_benchmark_integration')
    p = smoke_protocol()
    prepare(REPO, root, p, synthetic=True)
    for phase in (1, 2, 3):
        for run in phase_runs(p, phase):
            assert run_one(root, run.key)['status'] == 'PASS'
    report = finalize(root, REPO)
    assert report['completed_runs'] == 17
    assert report['probe_rows'] == 620
    return root


def test_full_pipeline_manifest_and_frozen_depth(smoke_root: Path) -> None:
    p, lock = load_lock(smoke_root, REPO)
    for run in runs(p):
        directory = smoke_root / 'runs' / run.key
        validate_complete(directory, run, lock)
        assert set(read_json(directory / 'complete.json')['files']) == set(required_artifacts(run, p))
    source = load_torch(smoke_root / 'runs/O0__seed11/checkpoint.pt')['model_state_dict']
    frozen = load_torch(smoke_root / 'runs/D2__seed11/checkpoint.pt')['model_state_dict']
    for name in source:
        if name.startswith(('layers.0.', 'layers.1.')):
            assert torch.equal(source[name], frozen[name])
    assert read_json(smoke_root / 'runs/R_IF__seed11/readout.json')['same_W_verified']
    e2e = read_json(smoke_root / 'runs/R_LIF_E2E__seed11/readout.json')
    assert [row['mode'] for row in e2e['rows']] == ['R3_e2e_LIF', 'R4_e2e_accumulator_swap']
    assert e2e['training_scope'] == 'full_end_to_end'
    e2e_checkpoint = load_torch(smoke_root / 'runs/R_LIF_E2E__seed11/checkpoint.pt')
    assert e2e_checkpoint['paired_o0_initialization_verified'] is True
    native = pd.read_csv(smoke_root / 'aggregate/native_summary.csv')
    assert set(native.test_ba_count) == {1}


def test_finalizer_fails_closed_on_missing_and_tampered_outputs(smoke_root: Path) -> None:
    marker = smoke_root / 'runs/O0__seed11/complete.json'
    original = marker.read_text()
    marker.unlink()
    try:
        with pytest.raises(FileNotFoundError, match='missing'):
            finalize(smoke_root, REPO)
        assert not (smoke_root / 'aggregate/manifest.json').exists()
    finally:
        marker.write_text(original)
    altered = json.loads(original)
    altered['files'].pop('traces.npz')
    marker.write_text(json.dumps(altered))
    try:
        with pytest.raises(ValueError, match='artifact set'):
            finalize(smoke_root, REPO)
    finally:
        marker.write_text(original)
    assert finalize(smoke_root, REPO)['status'] == 'PASS'


def test_evaluation_only_preserves_checkpoint_and_invalidates_summary(smoke_root: Path) -> None:
    checkpoint = smoke_root / 'runs/O0__seed11/checkpoint.pt'
    before = file_hash(checkpoint)
    run_one(smoke_root, 'O0__seed11', evaluation_only=True, reevaluate=True)
    assert file_hash(checkpoint) == before
    assert not (smoke_root / 'aggregate/manifest.json').exists()
    assert finalize(smoke_root, REPO)['status'] == 'PASS'


def test_prepare_rejects_changed_protocol(smoke_root: Path) -> None:
    with pytest.raises(ValueError, match='another protocol'):
        prepare(REPO, smoke_root, replace(smoke_protocol(), learning_rate=0.002), synthetic=True)


@pytest.mark.parametrize('cap', ['0', '51', 'abc'])
def test_slurm_rejects_invalid_concurrency(cap: str) -> None:
    env = {**os.environ, 'DRY_RUN': '1', 'SLURM_MAX_CONCURRENCY': cap}
    result = subprocess.run(['bash', 'core_benchmark_v1/slurm/submit.bash'], cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'sbatch ' not in result.stdout + result.stderr


def test_slurm_dry_run_is_deduplicated_and_dependency_safe() -> None:
    env = {**os.environ, 'DRY_RUN': '1', 'SLURM_MAX_CONCURRENCY': '20'}
    result = subprocess.run(['bash', 'core_benchmark_v1/slurm/submit.bash'], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    output = result.stdout + result.stderr
    assert '--array=0-29%20' in output
    assert '--array=0-5%6' in output
    assert 'afterok:DRY_prepare' in output and 'afterok:DRY_phase1' in output and 'afterok:DRY_phase2' in output
    assert 'phase 3:' not in output
    for name in ('prepare', 'run_array', 'finalize'):
        text = (REPO / f'core_benchmark_v1/slurm/{name}.bash').read_text()
        assert '--cpus-per-task=1' in text and 'common.bash' in text


def test_notebooks_are_valid_analysis_only() -> None:
    notebooks = list((REPO / 'core_benchmark_v1').rglob('*.ipynb'))
    assert len(notebooks) == 8
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                assert cell.execution_count is None and not cell.outputs
                compile(cell.source, str(path), 'exec')
                assert all(term not in cell.source for term in ('.fit(', '.backward(', 'sbatch', 'run_one(', 'multiprocessing'))


def test_notebooks_execute_on_finalized_artifacts(smoke_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nbclient import NotebookClient
    monkeypatch.setenv('CORE_BENCHMARK_RESULTS', str(smoke_root))
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
    for path in sorted((REPO / 'core_benchmark_v1').rglob('*.ipynb')):
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}}).execute()
        output = tmp_path / (path.parent.name + '_' + path.name)
        nbformat.write(nb, output)
        assert not any(item.output_type == 'error' for cell in nb.cells if cell.cell_type == 'code' for item in cell.outputs)
