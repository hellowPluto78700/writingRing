"""Prepare once; workers consume a hashed cache and locked user split."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .protocol import Protocol, SPLITS, aliases, digest, paired_seed, runs
from .storage import environment, file_hash, load_lock, run_lock, save_json, save_npz, source_identity


def choose_users(users: list[str], p: Protocol) -> dict[str, tuple[str, ...]]:
    supplied = {s: getattr(p, f'{s}_users') for s in SPLITS}
    if any(supplied.values()):
        if set(users) != set(u for group in supplied.values() for u in group):
            raise ValueError('Explicit users must partition the complete selected cohort')
        return supplied
    if len(users) < 3:
        raise ValueError('Cross-user evaluation needs at least three users')
    shuffled = np.array(sorted(users), dtype=str)
    np.random.default_rng(p.split_seed).shuffle(shuffled)
    nt = max(1, int(np.floor(p.train_fraction * len(shuffled))))
    nv = max(1, int(np.floor(p.val_fraction * len(shuffled))))
    if nt + nv >= len(shuffled):
        nt, nv = len(shuffled) - 2, 1
    return {'train': tuple(sorted(shuffled[:nt].tolist())),
            'val': tuple(sorted(shuffled[nt:nt + nv].tolist())),
            'test': tuple(sorted(shuffled[nt + nv:].tolist()))}


def validate_arrays(arrays: dict[str, np.ndarray], p: Protocol) -> None:
    ids: set[str] = set()
    users: set[str] = set()
    for split in SPLITS:
        x, y, lengths = (arrays[f'{split}_{k}'] for k in ('x', 'y', 'lengths'))
        sample_ids, split_users = arrays[f'{split}_ids'], arrays[f'{split}_users']
        if x.shape != (len(y), p.steps, p.input_channels) or not len(y):
            raise ValueError(f'{split}: bad or empty data shape')
        if lengths.shape != y.shape or sample_ids.shape != y.shape or split_users.shape != y.shape:
            raise ValueError(f'{split}: unaligned identities/lengths')
        if set(np.unique(y)) != set(range(len(p.labels))):
            raise ValueError(f'{split}: every benchmark class must be represented')
        if not np.isfinite(x).all() or (x < 0).any() or (lengths <= 0).any() or (lengths > p.steps).any():
            raise ValueError(f'{split}: invalid unsigned events or valid lengths')
        invalid = np.arange(p.steps)[None, :] >= lengths[:, None]
        if np.any(x[invalid] != 0):
            raise ValueError(f'{split}: padded input must be exactly zero')
        these_ids, these_users = set(sample_ids.tolist()), set(split_users.tolist())
        if len(these_ids) != len(y) or ids & these_ids or users & these_users:
            raise ValueError('Duplicate samples or cross-user leakage')
        ids |= these_ids
        users |= these_users


def load_source(repo: Path, p: Protocol) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from snn.accel_reconstruction_eval.datasets import load_acceleration_data
    loaded = load_acceleration_data([repo / root for root in p.dataset_roots],
                                    repository_root=repo, require_reconstruction=False)
    for meta in loaded.producer_metadatas:
        expected = {'event_representation': 'unsigned', 'event_feature_schema': 'custom_wavelet_polarity_split_abs_events_v1', 'event_channel_count': 30}
        if any(meta.raw.get(k) != v for k, v in expected.items()):
            raise ValueError('Event schema differs from the locked unsigned-event protocol')
        if meta.channel_count != p.total_channels or not np.isclose(meta.sampling_rate_hz, p.fs) or meta.target_length != p.steps:
            raise ValueError('Dataset channel count, sampling rate, or horizon mismatch')
    rows: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    label_to_idx = {label: i for i, label in enumerate(p.labels)}
    for pi, package in enumerate(loaded.packages):
        ri = loaded.padded_roots.index(package.source_padded_root)
        for name in ('padded_spike_imu_path', 'labels_path', 'valid_lengths_path', 'valid_mask_path', 'padding_manifest_path', 'padding_summary_path'):
            source_paths.add(getattr(package, name))
        for si, label in enumerate(package.labels.astype(str)):
            if label in label_to_idx:
                rows.append({'id': f'{ri}:{package.user}:{package.action}:{package.stem}:{si}',
                             'user': str(package.user), 'action': str(package.action), 'label': label,
                             'y': label_to_idx[label], 'length': int(package.valid_lengths[si]), 'pi': pi, 'si': si})
    if not rows:
        raise ValueError('No selected samples')
    allocation = choose_users(sorted({r['user'] for r in rows}), p)
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for split in SPLITS:
        part = sorted((r for r in rows if r['user'] in allocation[split]), key=lambda r: r['id'])
        x = np.zeros((len(part), p.steps, p.input_channels), dtype=np.float32)
        for i, row in enumerate(part):
            length = row['length']
            if not 0 < length <= p.steps:
                raise ValueError(f'Out-of-range length: {row["id"]}')
            x[i, :length] = loaded.packages[row['pi']].padded_spike_imu[row['si'], :length, :p.input_channels]
            manifest.append({'split': split, **{k: v for k, v in row.items() if k not in ('pi', 'si')}})
        arrays.update({f'{split}_x': x, f'{split}_y': np.array([r['y'] for r in part], dtype=np.int64),
                       f'{split}_lengths': np.array([r['length'] for r in part], dtype=np.int64),
                       f'{split}_ids': np.array([r['id'] for r in part], dtype=str),
                       f'{split}_users': np.array([r['user'] for r in part], dtype=str)})
    metadata = {'users': allocation, 'samples': manifest,
                'sources': {str(path.resolve()): file_hash(path) for path in sorted(source_paths)}, 'synthetic': False}
    return arrays, metadata


def synthetic_source(p: Protocol) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if p.profile != 'smoke':
        raise ValueError('Synthetic data is forbidden in production')
    rng = np.random.default_rng(718)
    arrays: dict[str, np.ndarray] = {}
    for si, split in enumerate(SPLITS):
        y = np.tile(np.arange(len(p.labels)), 3).astype(np.int64)
        lengths = rng.integers(max(2, p.steps // 2), p.steps + 1, size=len(y), dtype=np.int64)
        x = (rng.random((len(y), p.steps, p.input_channels)) < 0.13).astype(np.float32)
        x[np.arange(p.steps)[None, :] >= lengths[:, None]] = 0
        arrays.update({f'{split}_x': x, f'{split}_y': y, f'{split}_lengths': lengths,
                       f'{split}_ids': np.array([f'synthetic:{split}:{i}' for i in range(len(y))]),
                       f'{split}_users': np.array([f'smoke_user_{si}'] * len(y))})
    return arrays, {'synthetic': True, 'users': {s: [f'smoke_user_{i}'] for i, s in enumerate(SPLITS)}, 'sources': {}}


def prepare(repo: Path, root: Path, p: Protocol, *, synthetic: bool = False) -> dict[str, Any]:
    p.validate()
    if synthetic != (p.profile == 'smoke'):
        raise ValueError('Profile and synthetic input disagree')
    with run_lock(root / '.prepare.lock'):
        if (root / 'protocol.lock.json').exists():
            existing, lock = load_lock(root, repo)
            if existing.fingerprint != p.fingerprint:
                raise ValueError('Results directory belongs to another protocol')
            for source, sha in lock['data_metadata'].get('sources', {}).items():
                if file_hash(Path(source)) != sha:
                    raise ValueError(f'Raw source changed: {source}')
            load_cache(root, p, lock)
            return lock
        arrays, metadata = synthetic_source(p) if synthetic else load_source(repo, p)
        validate_arrays(arrays, p)
        save_npz(root / 'dataset.npz', arrays)
        code = source_identity(repo)
        lock = {'protocol': asdict(p), 'protocol_hash': p.fingerprint,
                'dataset_hash': file_hash(root / 'dataset.npz'), 'data_metadata': metadata, 'metadata_hash': digest(metadata),
                'source_files': code, 'source_hash': digest(code), 'environment': environment(repo),
                'aliases': aliases(), 'runs': [asdict(r) for r in runs(p)]}
        lock['identity'] = digest([lock['protocol_hash'], lock['dataset_hash'], lock['metadata_hash'], lock['source_hash']])
        save_json(root / 'protocol.lock.json', lock)
        return lock


def load_cache(root: Path, p: Protocol, lock: dict[str, Any]) -> dict[str, np.ndarray]:
    if file_hash(root / 'dataset.npz') != lock['dataset_hash']:
        raise ValueError('Frozen dataset cache changed')
    with np.load(root / 'dataset.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate_arrays(arrays, p)
    return arrays


def loader(arrays: dict[str, np.ndarray], split: str, p: Protocol, seed: int, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(*(torch.from_numpy(arrays[f'{split}_{k}']) for k in ('x', 'y', 'lengths')))
    generator = torch.Generator().manual_seed(paired_seed(seed, f'loader:{split}'))
    return DataLoader(dataset, batch_size=p.batch_size, shuffle=shuffle, generator=generator, num_workers=0)
