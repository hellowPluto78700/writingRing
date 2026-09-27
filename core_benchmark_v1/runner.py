"""CLI and atomic train/evaluate workers for the standardized benchmark."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from typing import Any

from .protocol import Protocol, Run, aliases, phase_runs, runs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def required_artifacts(run: Run, p: Protocol) -> list[str]:
    if run.kind == 'readout':
        return ['readout.json', 'predictions.npz', 'provenance.json'] + (['head.pt', 'history.json'] if p.readout_adaptation else [])
    return ['initial.pt', 'checkpoint.pt', 'history.json', 'native.json', 'traces.npz',
            'probes.json', 'probe_search.json', 'probe_decoders.npz', 'probe_predictions.npz',
            'diagnostics.json', 'provenance.json']


def configure_cpu() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('This repository benchmark requires Python 3.11')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def run_one(root: Path, run_key: str, *, evaluation_only: bool = False, reevaluate: bool = False) -> dict[str, Any]:
    configure_cpu()
    from threadpoolctl import threadpool_limits
    from .data import load_cache
    from .storage import (environment, file_hash, load_lock, mark_complete, read_json,
                          run_lock, save_json, save_npz, validate_complete)
    repo = repo_root()
    p, lock = load_lock(root, repo)
    available = {run.key: run for run in runs(p)}
    if run_key not in available:
        raise ValueError(f'Unknown run: {run_key}')
    run = available[run_key]
    directory = root / 'runs' / run.key
    directory.mkdir(parents=True, exist_ok=True)
    current_environment = environment(repo)
    if current_environment['packages'] != lock['environment']['packages']:
        raise ValueError('Worker packages differ from prepare; use the same Conda environment')
    with run_lock(directory / '.worker.lock'), threadpool_limits(limits=1):
        if (directory / 'complete.json').exists() and not reevaluate:
            validate_complete(directory, run, lock)
            if set(read_json(directory / 'complete.json')['files']) != set(required_artifacts(run, p)):
                raise ValueError('Completion artifact set differs from the protocol')
            return {'run': run.key, 'status': 'already_complete'}
        if run.parent_key is not None:
            parent = available[run.parent_key]
            validate_complete(root / 'runs' / parent.key, parent, lock)
        # A completion marker must never survive a partially rewritten evaluation.
        (directory / 'complete.json').unlink(missing_ok=True)
        (root / 'aggregate/manifest.json').unlink(missing_ok=True)
        arrays = load_cache(root, p, lock)
        save_json(directory / 'provenance.json', {
            'identity': lock['identity'], 'run_key': run.key, 'environment': current_environment,
            'cpu_threads': 1, 'dataloader_workers': 0, 'evaluation_only': evaluation_only,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')})
        if run.kind == 'readout':
            from .readout import run_readout
            run_readout(directory, root, run, p, lock, arrays, evaluation_only=evaluation_only)
        else:
            from .training import extract, load_model, train
            from .probes import run_probes
            from .diagnostics import run_diagnostics
            model = load_model(root, run, p, lock)[0] if evaluation_only else train(root, run, p, lock, arrays)
            checkpoint_sha = file_hash(directory / 'checkpoint.pt')
            traces, native = extract(model, arrays, p, run)
            save_npz(directory / 'traces.npz', traces)
            save_json(directory / 'native.json', native)
            run_probes(directory, traces, arrays, run, p)
            run_diagnostics(directory, model, traces, arrays, run, p)
            if file_hash(directory / 'checkpoint.pt') != checkpoint_sha:
                raise AssertionError('Evaluation modified the selected checkpoint')
        mark_complete(directory, run, lock, required_artifacts(run, p))
    return {'run': run.key, 'status': 'PASS'}


def smoke_protocol() -> Protocol:
    return replace(Protocol(), profile='smoke', seeds=(11,), width=6, input_channels=3,
                   total_channels=9, steps=32, labels=('A', 'B', 'C'), batch_size=9,
                   max_epochs=1, min_epochs=1, patience=1, c_grid=(0.1,), probe_max_iter=1000,
                   shuffle_seeds=(101,), relative_bins=4, depth_controls=True,
                   membrane_sweep=True, lags=(1, 2), history_ms=(125.0, 250.0))


def plan(p: Protocol) -> dict[str, Any]:
    specs = runs(p)
    return {'protocol_hash': p.fingerprint, 'profile': p.profile,
            'backbone_training_runs': sum(r.kind == 'backbone' for r in specs),
            'readout_tasks': sum(r.kind == 'readout' for r in specs),
            'readout_training_runs': sum(r.kind == 'readout' and p.readout_adaptation for r in specs),
            'phase_counts': {str(phase): len(phase_runs(p, phase)) for phase in (1, 2, 3)},
            'aliases': aliases(), 'runs': [{'key': r.key, **asdict(r)} for r in specs]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=repo_root() / 'core_benchmark_v1/00_protocol/default.json')
    parser.add_argument('--results', type=Path, default=repo_root() / 'core_benchmark_v1/results/main')
    sub = parser.add_subparsers(dest='command', required=True)
    planned = sub.add_parser('plan', help='List deduplicated cases without reading data')
    planned.add_argument('--counts', action='store_true', help='Print phase 1/2/3 counts as space-separated integers')
    prepared = sub.add_parser('prepare', help='Validate data and lock users, samples, source and protocol')
    prepared.add_argument('--synthetic', action='store_true')
    for command in ('run', 'evaluate'):
        worker = sub.add_parser(command)
        selected = worker.add_mutually_exclusive_group(required=True)
        selected.add_argument('--run-key')
        selected.add_argument('--task-id', type=int)
        worker.add_argument('--phase', type=int, choices=(1, 2, 3), default=1)
        worker.add_argument('--reevaluate', action='store_true', help='Rewrite evaluations only; preserve completed training checkpoints')
    sub.add_parser('finalize', help='Strict aggregation of existing completed artifacts only')
    sub.add_parser('smoke', help='Synthetic end-to-end test in a separate results directory')
    args = parser.parse_args(argv)
    root = args.results.expanduser().resolve()
    if args.command == 'plan':
        p = Protocol.load(args.config)
        result = plan(p)
        if args.counts:
            print(' '.join(str(result['phase_counts'][str(i)]) for i in (1, 2, 3)))
        else:
            print(json.dumps(result, indent=2))
        return
    configure_cpu()
    if args.command in ('prepare', 'smoke'):
        from .data import prepare
        p = smoke_protocol() if args.command == 'smoke' else Protocol.load(args.config)
        if args.command == 'smoke' and root == (repo_root() / 'core_benchmark_v1/results/main').resolve():
            parser.error('smoke requires an explicit --results directory, separate from production')
        lock = prepare(repo_root(), root, p, synthetic=args.command == 'smoke' or args.synthetic)
        if args.command == 'prepare':
            print(json.dumps({'status': 'prepared', 'identity': lock['identity'], **plan(p)}, indent=2))
            return
        for phase in (1, 2, 3):
            for run in phase_runs(p, phase):
                run_one(root, run.key)
    if args.command in ('run', 'evaluate'):
        from .storage import load_lock
        p, _ = load_lock(root, repo_root())
        if args.run_key is None:
            subset = phase_runs(p, args.phase)
            if not 0 <= args.task_id < len(subset):
                parser.error(f'task-id must be in [0, {len(subset)-1}] for phase {args.phase}')
            key = subset[args.task_id].key
        else:
            key = args.run_key
        print(json.dumps(run_one(root, key, evaluation_only=args.command == 'evaluate',
                                 reevaluate=args.reevaluate or args.command == 'evaluate'), indent=2))
        return
    if args.command in ('finalize', 'smoke'):
        from .aggregate import finalize
        from .storage import run_lock
        with run_lock(root / '.finalize.lock'):
            print(json.dumps(finalize(root, repo_root()), indent=2))


if __name__ == '__main__':
    main()
