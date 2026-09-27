"""Training and extraction are separate: evaluation never retrains a checkpoint."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from .data import loader
from .model import BenchmarkNet, mean_logits, objective_loss, valid_sum
from .protocol import Protocol, Run, SPLITS, runs
from .storage import checkpoint_metadata, file_hash, load_torch, save_json, save_torch, state_hash, validate_checkpoint


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if not len(y) or y.shape != prediction.shape:
        raise ValueError('Metrics require nonempty aligned labels/predictions')
    labels = np.unique(y)
    recall = [float((prediction[y == k] == k).mean()) for k in labels]
    f1 = []
    for k in labels:
        tp = int(((prediction == k) & (y == k)).sum())
        fp = int(((prediction == k) & (y != k)).sum())
        fn = int(((prediction != k) & (y == k)).sum())
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {'ba': float(np.mean(recall)), 'accuracy': float((y == prediction).mean()), 'macro_f1': float(np.mean(f1))}


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_model(root: Path, run: Run, p: Protocol, lock: dict[str, Any]) -> tuple[BenchmarkNet, dict[str, Any]]:
    checkpoint = load_torch(root / 'runs' / run.key / 'checkpoint.pt')
    validate_checkpoint(checkpoint, run, lock)
    if not checkpoint.get('training_complete'):
        raise ValueError(f'Checkpoint is not a completed training run: {run.key}')
    model = BenchmarkNet(run, p)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, checkpoint


@torch.no_grad()
def evaluate_validation(model: BenchmarkNet, arrays: dict[str, np.ndarray], p: Protocol, seed: int) -> dict[str, float]:
    model.eval()
    targets, predictions, losses = [], [], []
    for x, y, lengths in loader(arrays, 'val', p, seed):
        evidence = model(x, lengths)['evidence']
        scores = mean_logits(evidence, lengths)
        targets.append(y.numpy())
        predictions.append(scores.argmax(1).numpy())
        losses.append(float(F.cross_entropy(scores, y, reduction='sum')))
    result = metrics(np.concatenate(targets), np.concatenate(predictions))
    result['mean_logit_ce'] = sum(losses) / sum(len(y) for y in targets)
    return result


def train(root: Path, run: Run, p: Protocol, lock: dict[str, Any], arrays: dict[str, np.ndarray]) -> BenchmarkNet:
    directory = root / 'runs' / run.key
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'checkpoint.pt').exists():
        return load_model(root, run, p, lock)[0]
    model = BenchmarkNet(run, p)
    parent_hash = None
    if run.parent_key is not None:
        parent_run = next(r for r in runs(p) if r.key == run.parent_key)
        parent, _ = load_model(root, parent_run, p, lock)
        parent_hash = file_hash(root / 'runs' / parent_run.key / 'checkpoint.pt')
        if run.case == 'D2':
            for i in range(2):
                model.layers[i].load_state_dict(parent.layers[i].state_dict())
                model.layers[i].requires_grad_(False)
        elif run.case == 'D3':
            model.load_state_dict(parent.state_dict(), strict=True)
        else:
            raise ValueError(f'Unexpected backbone dependency: {run.key}')
    initial = cpu_state(model)
    save_torch(directory / 'initial.pt', {**checkpoint_metadata(run, lock), 'model_state_dict': initial})
    parameter_hashes = {name: state_hash({name: tensor}) for name, tensor in model.named_parameters()}
    optimizer = torch.optim.Adam((v for v in model.parameters() if v.requires_grad), lr=p.learning_rate, weight_decay=p.weight_decay)
    train_loader = loader(arrays, 'train', p, run.seed, shuffle=True)
    best = evaluate_validation(model, arrays, p, run.seed)
    best_state, best_epoch = initial, 0
    history = [{'epoch': 0, 'train_loss': None, 'val_ba': best['ba'], 'val_mean_logit_ce': best['mean_logit_ce']}]
    for epoch in range(1, p.max_epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective_loss(model(x, lengths), lengths, y, run, p)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{run.key}: nonfinite loss at epoch {epoch}')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise FloatingPointError(f'{run.key}: nonfinite gradient')
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
        val = evaluate_validation(model, arrays, p, run.seed)
        history.append({'epoch': epoch, 'train_loss': total_loss / count, 'val_ba': val['ba'], 'val_mean_logit_ce': val['mean_logit_ce']})
        improved = val['ba'] > best['ba'] + 1e-12 or (abs(val['ba'] - best['ba']) <= 1e-12 and val['mean_logit_ce'] < best['mean_logit_ce'] - 1e-12)
        if improved:
            best_state, best_epoch, best = cpu_state(model), epoch, val
        if epoch == 1 or epoch % 10 == 0:
            print(f'{run.key} epoch={epoch} train_loss={total_loss/count:.5f} val_ba={val["ba"]:.5f}', flush=True)
            save_json(directory / 'progress.json', history[-1])
        if epoch >= p.min_epochs and epoch - best_epoch >= p.patience:
            break
    model.load_state_dict(best_state)
    if run.case == 'D2':
        names = [name for name in initial if name.startswith(('layers.0.', 'layers.1.'))]
        if any(not torch.equal(initial[name], best_state[name]) for name in names):
            raise AssertionError('Frozen upstream parameters changed')
    payload = {**checkpoint_metadata(run, lock), 'model_state_dict': best_state,
               'training_complete': True, 'best_epoch': best_epoch, 'stopped_epoch': epoch,
               'best_val': best, 'parent_checkpoint_hash': parent_hash,
               'initial_parameter_hashes': parameter_hashes,
               'trainable_parameters': sum(v.numel() for v in model.parameters() if v.requires_grad),
               'selection_rule': 'native validation BA, then common valid-mean-logit CE, then earliest epoch'}
    save_json(directory / 'history.json', {'rows': history})
    save_torch(directory / 'checkpoint.pt', payload)
    return model


@torch.no_grad()
def extract(model: BenchmarkNet, arrays: dict[str, np.ndarray], p: Protocol, run: Run) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model.eval()
    traces: dict[str, np.ndarray] = {}
    native: dict[str, Any] = {'case': run.case, 'seed': run.seed, 'block': run.block, 'splits': {}, 'users': [], 'activity': []}
    for split in SPLITS:
        chunks: dict[str, list[np.ndarray]] = {'evidence': []}
        for state in ('spike', 'pre_reset'):
            for li in range(len(run.shifts)):
                chunks[f'L{li+1}__{state}'] = []
        for x, y, lengths in loader(arrays, split, p, run.seed):
            tr = model(x, lengths)
            chunks['evidence'].append(tr['evidence'].numpy())
            for state in ('spike', 'pre_reset'):
                for li in range(len(run.shifts)):
                    value = tr[state][li].numpy()
                    chunks[f'L{li+1}__{state}'].append(value.astype(np.uint8 if state == 'spike' else np.float32))
        for key, pieces in chunks.items():
            traces[f'{split}__{key}'] = np.concatenate(pieces)
        y, lengths = arrays[f'{split}_y'], arrays[f'{split}_lengths']
        scores = traces[f'{split}__evidence'].sum(1)
        prediction = scores.argmax(1)
        native['splits'][split] = {**metrics(y, prediction), 'n_samples': len(y)}
        traces[f'{split}__native_prediction'] = prediction
        for user in np.unique(arrays[f'{split}_users']):
            selected = arrays[f'{split}_users'] == user
            native['users'].append({'split': split, 'user': str(user), 'n': int(selected.sum()), **metrics(y[selected], prediction[selected])})
        mask = np.arange(p.steps)[None, :] < lengths[:, None]
        for li, shifts in enumerate(run.shifts):
            z = traces[f'{split}__L{li+1}__spike'][mask]
            v = traces[f'{split}__L{li+1}__pre_reset'][mask]
            native['activity'].append({'split': split, 'layer': f'L{li+1}',
                'spikes_per_timestep': float(z.sum(1).mean()), 'nonzero_fraction': float((z != 0).mean()),
                'silent_neuron_fraction': float((z.sum(0) == 0).mean()), 'mean_abs_pre_reset': float(np.abs(v).mean()),
                'tau_syn_ms': [-(1000 / p.fs) / np.log(1 - 2.0 ** (-s)) for s in shifts]})
    native['train_test_gap'] = native['splits']['train']['ba'] - native['splits']['test']['ba']
    return traces, native
