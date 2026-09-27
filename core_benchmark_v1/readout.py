"""Accumulator -> same-W IF/LIF -> frozen-backbone W-only adaptation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .model import spike_readout, valid_sum
from .protocol import Protocol, Run, SPLITS, paired_seed, runs
from .storage import checkpoint_metadata, file_hash, load_torch, save_json, save_npz, save_torch, validate_checkpoint
from .training import metrics, load_model


def head_loader(z: np.ndarray, arrays: dict[str, np.ndarray], split: str, run: Run, p: Protocol, shuffle: bool = False) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(z.astype(np.float32)), torch.from_numpy(arrays[f'{split}_y']), torch.from_numpy(arrays[f'{split}_lengths']))
    generator = torch.Generator().manual_seed(paired_seed(run.seed, f'readout_loader:{split}'))
    return DataLoader(dataset, batch_size=p.batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def head_scores(head: nn.Linear, z: torch.Tensor, lengths: torch.Tensor, beta: float, threshold: float, p: Protocol, analog: bool = False) -> torch.Tensor:
    evidence = head(z)
    return valid_sum(evidence if analog else spike_readout(evidence, lengths, beta, threshold, p.surrogate_slope), lengths)


@torch.no_grad()
def evaluate_head(head: nn.Linear, batches: DataLoader, beta: float, threshold: float, p: Protocol, analog: bool = False) -> tuple[dict[str, float], np.ndarray]:
    targets, predictions, scores_all = [], [], []
    loss = 0.0
    for z, y, lengths in batches:
        scores = head_scores(head, z, lengths, beta, threshold, p, analog)
        targets.append(y.numpy())
        predictions.append(scores.argmax(1).numpy())
        scores_all.append(scores.numpy())
        loss += float(F.cross_entropy(scores, y, reduction='sum'))
    y, prediction, scores = np.concatenate(targets), np.concatenate(predictions), np.concatenate(scores_all)
    result = metrics(y, prediction)
    result.update({'count_ce': loss / len(y), 'tie_fraction': float(((scores == scores.max(1, keepdims=True)).sum(1) > 1).mean()),
                   'all_zero_fraction': float((np.abs(scores).sum(1) == 0).mean())})
    return result, prediction


def run_readout(directory: Path, root: Path, run: Run, p: Protocol, lock: dict[str, Any], arrays: dict[str, np.ndarray], evaluation_only: bool = False) -> None:
    parent_run = next(r for r in runs(p) if r.key == run.parent_key)
    parent, _ = load_model(root, parent_run, p, lock)
    source_dir = root / 'runs' / parent_run.key
    source_hash = file_hash(source_dir / 'checkpoint.pt')
    with np.load(source_dir / 'traces.npz', allow_pickle=False) as cache:
        z = {s: cache[f'{s}__L2__spike'] for s in SPLITS}
    batches = {s: head_loader(z[s], arrays, s, run, p) for s in SPLITS}
    head = nn.Linear(p.width, len(p.labels), bias=False)
    head.load_state_dict(parent.head.state_dict())
    original_w = head.weight.detach().clone()
    beta = float(run.beta)
    calibration = []
    for threshold in sorted(p.readout_thresholds):
        val, _ = evaluate_head(head, batches['val'], beta, threshold, p)
        calibration.append({'threshold': threshold, **val})
    # Prefer the native threshold in BA ties, then the closest scale to native.
    selected = max(calibration, key=lambda row: (row['ba'], -abs(np.log(row['threshold'] / p.threshold))))
    threshold = selected['threshold']
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {f'{s}__ids': arrays[f'{s}_ids'] for s in SPLITS}

    def record(mode: str, model: nn.Linear, theta: float, analog: bool = False) -> None:
        row: dict[str, Any] = {'run_key': run.key, 'case': run.case, 'seed': run.seed, 'block': run.block,
                               'mode': mode, 'beta': beta, 'threshold': theta, 'parent_checkpoint_hash': source_hash}
        for split in SPLITS:
            scores, pred = evaluate_head(model, batches[split], beta, theta, p, analog)
            row.update({f'{split}_{name}': value for name, value in scores.items()})
            predictions[f'{mode}__{split}'] = pred
        row['train_test_gap'] = row['train_ba'] - row['test_ba']
        rows.append(row)

    record('R0_accumulator', head, p.threshold, analog=True)
    record('R1_sameW_fixed', head, p.threshold)
    record('R1_sameW_calibrated', head, threshold)
    if not torch.equal(original_w, head.weight):
        raise AssertionError('Same-W conversion changed the output weights')
    if p.readout_adaptation:
        checkpoint_path = directory / 'head.pt'
        if checkpoint_path.exists():
            checkpoint = load_torch(checkpoint_path)
            validate_checkpoint(checkpoint, run, lock)
            if checkpoint['parent_checkpoint_hash'] != source_hash or checkpoint['threshold'] != threshold:
                raise ValueError('Readout source/threshold changed')
            head.load_state_dict(checkpoint['head_state_dict'])
        else:
            if evaluation_only:
                raise FileNotFoundError(checkpoint_path)
            optimizer = torch.optim.Adam(head.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
            train_batches = head_loader(z['train'], arrays, 'train', run, p, shuffle=True)
            best, best_epoch = selected.copy(), 0
            best_w = original_w.clone()
            history = [{'epoch': 0, 'train_loss': None, 'val_ba': best['ba'], 'val_count_ce': best['count_ce']}]
            for epoch in range(1, p.max_epochs + 1):
                loss_sum, count = 0.0, 0
                for spikes, y, lengths in train_batches:
                    optimizer.zero_grad(set_to_none=True)
                    scores = head_scores(head, spikes, lengths, beta, threshold, p)
                    loss = F.cross_entropy(scores, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'{run.key}: nonfinite output training loss')
                    loss.backward()
                    if not torch.isfinite(head.weight.grad).all():
                        raise FloatingPointError(f'{run.key}: nonfinite output gradient')
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(y)
                    count += len(y)
                val, _ = evaluate_head(head, batches['val'], beta, threshold, p)
                history.append({'epoch': epoch, 'train_loss': loss_sum / count, 'val_ba': val['ba'], 'val_count_ce': val['count_ce']})
                if val['ba'] > best['ba'] + 1e-12 or (abs(val['ba'] - best['ba']) <= 1e-12 and val['count_ce'] < best['count_ce'] - 1e-12):
                    best, best_epoch, best_w = val, epoch, head.weight.detach().clone()
                if epoch == 1 or epoch % 10 == 0:
                    print(f'{run.key} head epoch={epoch} val_ba={val["ba"]:.5f}', flush=True)
                if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
                    break
            with torch.no_grad():
                head.weight.copy_(best_w)
            save_torch(checkpoint_path, {**checkpoint_metadata(run, lock), 'head_state_dict': head.state_dict(),
                'best_epoch': best_epoch, 'stopped_epoch': epoch, 'threshold': threshold, 'parent_checkpoint_hash': source_hash,
                'frozen_backbone': True, 'training_complete': True})
            save_json(directory / 'history.json', {'rows': history})
        record('R2_adaptW', head, threshold)
    save_json(directory / 'readout.json', {'rows': rows, 'calibration': calibration,
              'fixed_negative_membrane_policy': 'signed, no clamp; positive cap=1 spikes, subtractive reset',
              'output_synaptic_alpha': 0.0, 'same_W_verified': True})
    save_npz(directory / 'predictions.npz', predictions)
