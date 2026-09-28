"""Strict artifact-only finalization. Never train or regenerate missing runs."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .protocol import Protocol, SPLITS, runs
from .probes import probe_specs
from .storage import atomic_path, file_hash, load_lock, load_torch, read_json, save_json, validate_complete
from .runner import required_artifacts

METRICS = ['train_ba', 'val_ba', 'test_ba', 'train_test_gap']
PROBE_GROUPS = ['case', 'block', 'layer', 'state', 'aggregation', 'decoder']
BLOCK_CASES = {'01_objective': ['O0', 'O1', 'O2', 'O3'], '02_tau': ['O0', 'T1', 'T2', 'T3', 'T4'],
               '03_depth': ['O0', 'D1', 'D2', 'D3'], '07_membrane': ['O0', 'M1', 'M2', 'M3']}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as temporary:
        frame.to_csv(temporary, index=False)


def summarize(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=groups)
    out = frame.groupby(groups, dropna=False, sort=False)[metrics].agg(['mean', 'std', 'count'])
    out.columns = ['_'.join(column) for column in out.columns]
    return out.reset_index()


def probe_seed_table(frame: pd.DataFrame) -> pd.DataFrame:
    # Average shuffle replicates WITHIN each model seed first; never call 15 shuffles 15 models.
    return frame.groupby(PROBE_GROUPS + ['seed'], as_index=False, dropna=False, sort=False)[METRICS].mean()


def probe_gains(seeds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouping = ['case', 'block', 'seed', 'layer', 'state', 'decoder']
    for key, group in seeds.groupby(grouping, dropna=False, sort=False):
        lookup = group.set_index('aggregation')
        for name, left, right in (
            ('G_resolved', 'fixed250_ordered', 'whole_count'),
            ('G_order', 'fixed250_ordered', 'fixed250_shuffled'),
            ('G_relative_order', 'relative10_ordered', 'relative10_shuffled')):
            if left in lookup.index and right in lookup.index:
                rows.append({**dict(zip(grouping, key)), 'gain': name,
                    **{f'{split}_delta': float(lookup.loc[left, f'{split}_ba'] - lookup.loc[right, f'{split}_ba']) for split in SPLITS}})
    geometry_group = ['case', 'block', 'seed', 'layer', 'state', 'aggregation']
    for key, group in seeds.groupby(geometry_group, dropna=False, sort=False):
        lookup = group.set_index('decoder')
        if {'affine', 'no_bias'} <= set(lookup.index):
            rows.append({**dict(zip(geometry_group, key)), 'decoder': 'affine_minus_no_bias', 'gain': 'G_geometry',
                **{f'{split}_delta': float(lookup.loc['affine', f'{split}_ba'] - lookup.loc['no_bias', f'{split}_ba']) for split in SPLITS}})
    return pd.DataFrame(rows)


def paired_native(native: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [('01_objective', 'O1', 'O0'), ('01_objective', 'O2', 'O0'), ('01_objective', 'O3', 'O0'),
             ('02_tau', 'T1', 'O0'), ('02_tau', 'T2', 'T1'), ('02_tau', 'T3', 'T1'), ('02_tau', 'T4', 'O0'),
             ('03_depth', 'D1', 'O0'), ('03_depth', 'D2', 'O0'), ('03_depth', 'D3', 'D2'),
             ('07_membrane', 'M1', 'O0'), ('07_membrane', 'M2', 'O0'), ('07_membrane', 'M3', 'O0')]
    for block, left, right in pairs:
        a, b = (native[native.case == case].set_index('seed') for case in (left, right))
        if a.empty or b.empty:
            continue
        if set(a.index) != set(b.index):
            raise ValueError(f'Unpaired seeds for {left} vs {right}')
        for seed in a.index:
            rows.append({'block': block, 'contrast': f'{left}_minus_{right}', 'seed': seed,
                         **{f'{split}_delta': float(a.loc[seed, f'{split}_ba'] - b.loc[seed, f'{split}_ba']) for split in SPLITS}})
    return pd.DataFrame(rows)


def readout_gains(readout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, seed, beta), group in readout.groupby(['case', 'seed', 'beta'], sort=False):
        lookup = group.set_index('mode')
        for name, left, right in (
            ('fixed_conversion_loss', 'R0_accumulator', 'R1_sameW_fixed'),
            ('calibrated_conversion_loss', 'R0_accumulator', 'R1_sameW_calibrated'),
            ('W_adaptation_gain', 'R2_adaptW', 'R1_sameW_calibrated'),
            ('remaining_gap', 'R0_accumulator', 'R2_adaptW'),
            ('e2e_realization_loss', 'R4_e2e_accumulator_swap', 'R3_e2e_LIF')):
            if left in lookup.index and right in lookup.index:
                rows.append({'case': case, 'seed': seed, 'beta': beta, 'contrast': name,
                    **{f'{split}_delta': float(lookup.loc[left, f'{split}_ba'] - lookup.loc[right, f'{split}_ba']) for split in SPLITS}})
    # Paired cross-case contrasts for beta=0.5. These separate the benefit of
    # full E2E adaptation from the residual LIF realization loss.
    baseline = readout[(readout['case'] == 'R_LIF') & (readout['beta'] == 0.5)]
    e2e = readout[(readout['case'] == 'R_LIF_E2E') & (readout['beta'] == 0.5)]
    for seed in sorted(set(baseline['seed']) & set(e2e['seed'])):
        b = baseline[baseline['seed'] == seed].set_index('mode')
        e = e2e[e2e['seed'] == seed].set_index('mode')
        pairs = (
            ('e2e_LIF_gain_vs_sameW_calibrated', e, 'R3_e2e_LIF', b, 'R1_sameW_calibrated'),
            ('e2e_LIF_gain_vs_W_adapted', e, 'R3_e2e_LIF', b, 'R2_adaptW'),
            ('e2e_accumulator_swap_gain_vs_O0', e, 'R4_e2e_accumulator_swap', b, 'R0_accumulator'),
        )
        for name, left_frame, left_mode, right_frame, right_mode in pairs:
            if left_mode in left_frame.index and right_mode in right_frame.index:
                rows.append({'case': 'R_LIF_E2E', 'seed': seed, 'beta': 0.5, 'contrast': name,
                    **{f'{split}_delta': float(left_frame.loc[left_mode, f'{split}_ba'] - right_frame.loc[right_mode, f'{split}_ba']) for split in SPLITS}})
    return pd.DataFrame(rows)


def finalize(root: Path, repo: Path) -> dict[str, Any]:
    p, lock = load_lock(root, repo)
    expected = runs(p)
    (root / 'aggregate/manifest.json').unlink(missing_ok=True)
    missing = [r.key for r in expected if not (root / 'runs' / r.key / 'complete.json').exists()]
    if missing:
        save_json(root / 'incomplete.json', {'status': 'INCOMPLETE', 'missing': missing, 'expected_runs': len(expected)})
        raise FileNotFoundError(f'{len(missing)} required runs are missing: {missing}')
    native_rows, probe_rows, readout_rows, activity, users, lag, history = [], [], [], [], [], [], []
    for run in expected:
        directory = root / 'runs' / run.key
        validate_complete(directory, run, lock)
        completion = read_json(directory / 'complete.json')
        if set(completion['files']) != set(required_artifacts(run, p)):
            raise ValueError(f'Incorrect artifact set: {run.key}')
        if run.parent_key is not None and (run.kind == 'backbone' or p.readout_adaptation):
            child = load_torch(directory / ('head.pt' if run.kind == 'readout' else 'checkpoint.pt'))
            if child['parent_checkpoint_hash'] != file_hash(root / 'runs' / run.parent_key / 'checkpoint.pt'):
                raise ValueError(f'Stale parent checkpoint: {run.key}')
        if run.kind in ('readout', 'e2e_readout'):
            payload = read_json(directory / 'readout.json')
            expected_rows = 2 if run.kind == 'e2e_readout' else (4 if p.readout_adaptation else 3)
            if len(payload['rows']) != expected_rows:
                raise ValueError(f'Incomplete readout rows: {run.key}')
            if any(row['run_key'] != run.key or row['case'] != run.case or row['seed'] != run.seed for row in payload['rows']):
                raise ValueError(f'Readout row identity mismatch: {run.key}')
            readout_rows.extend(payload['rows'])
            continue
        payload = read_json(directory / 'native.json')
        if payload['case'] != run.case or payload['seed'] != run.seed:
            raise ValueError(f'Native row identity mismatch: {run.key}')
        row = {'run_key': run.key, 'case': run.case, 'seed': run.seed, 'block': run.block, 'depth': len(run.shifts),
               'train_test_gap': payload['train_test_gap']}
        for split in SPLITS:
            row.update({f'{split}_{metric}': value for metric, value in payload['splits'][split].items()})
        native_rows.append(row)
        probes = read_json(directory / 'probes.json')['rows']
        if len(probes) != len(run.shifts) * len(p.probe_states) * len(probe_specs(p)):
            raise ValueError(f'Incomplete probe rows: {run.key}')
        keys = [(r['layer'], r['state'], r['aggregation'], r['decoder'], r['shuffle_seed']) for r in probes]
        expected_keys = {(f'L{i+1}', state, aggregation, decoder, shuffle) for i in range(len(run.shifts)) for state in p.probe_states for aggregation, decoder, shuffle in probe_specs(p)}
        if len(set(keys)) != len(keys) or set(keys) != expected_keys:
            raise ValueError(f'Duplicate/missing probe coordinates: {run.key}')
        if any(row['run_key'] != run.key or row['case'] != run.case or row['seed'] != run.seed for row in probes):
            raise ValueError(f'Probe row identity mismatch: {run.key}')
        probe_rows.extend(probes)
        activity.extend({'case': run.case, 'seed': run.seed, **r} for r in payload['activity'])
        users.extend({'case': run.case, 'seed': run.seed, **r} for r in payload['users'])
        diagnostics = read_json(directory / 'diagnostics.json')
        lag.extend(diagnostics['lag'])
        history.extend(diagnostics['history'])
    native, probes, readout = map(pd.DataFrame, (native_rows, probe_rows, readout_rows))
    probe_seeds = probe_seed_table(probes)
    gains = probe_gains(probe_seeds)
    paired = paired_native(native)
    output_gains = readout_gains(readout)
    tables = {
        'native_runs': native,
        'native_summary': summarize(native, ['case', 'block', 'depth'], METRICS),
        'probe_runs': probes, 'probe_seed_means': probe_seeds,
        'probe_summary': summarize(probe_seeds, PROBE_GROUPS, METRICS),
        'probe_gains': gains,
        'probe_gains_summary': summarize(gains, ['case', 'block', 'layer', 'state', 'decoder', 'aggregation', 'gain'], ['train_delta', 'val_delta', 'test_delta']),
        'paired_contrasts': paired,
        'paired_summary': summarize(paired, ['block', 'contrast'], ['train_delta', 'val_delta', 'test_delta']),
        'readout_runs': readout,
        'readout_summary': summarize(readout, ['case', 'mode', 'beta'], METRICS),
        'readout_gains': output_gains,
        'readout_gains_summary': summarize(output_gains, ['case', 'beta', 'contrast'], ['train_delta', 'val_delta', 'test_delta']),
        'activity': pd.DataFrame(activity), 'native_per_user': pd.DataFrame(users),
        'lag_similarity': pd.DataFrame(lag), 'history_reset': pd.DataFrame(history),
    }
    with np.load(root / 'dataset.npz', allow_pickle=False) as data:
        sample_rows = []
        for split in SPLITS:
            for sample_id, user, y, length in zip(data[f'{split}_ids'], data[f'{split}_users'], data[f'{split}_y'], data[f'{split}_lengths'], strict=True):
                sample_rows.append({'split': split, 'sample_id': str(sample_id), 'user': str(user), 'label': p.labels[int(y)],
                    'valid_steps': int(length), 'full_fixed_bins': int(length // p.fixed_steps),
                    'partial_bin_steps': int(length % p.fixed_steps), 'shuffle_eligible': bool(length // p.fixed_steps >= 2)})
    tables['sample_manifest'] = pd.DataFrame(sample_rows)
    tables['sample_counts'] = tables['sample_manifest'].groupby(['split', 'user', 'label'], as_index=False).size()
    summary_root = root / 'aggregate'
    for name, table in tables.items():
        write_csv(table, summary_root / f'{name}.csv')
    for block, cases in BLOCK_CASES.items():
        if block == '07_membrane' and not p.membrane_sweep:
            continue
        for name in ('native_runs', 'native_summary', 'probe_summary', 'probe_gains', 'probe_gains_summary'):
            subset = tables[name][tables[name]['case'].isin(cases)].copy()
            subset['display_case'] = subset['case'].replace({'O0': {'02_tau': 'T0', '03_depth': 'D0', '07_membrane': 'M0'}.get(block, 'O0')})
            write_csv(subset, summary_root / block / f'{name}.csv')
    for name in ('readout_runs', 'readout_summary', 'readout_gains', 'readout_gains_summary'):
        write_csv(tables[name], summary_root / '04_readout' / f'{name}.csv')
    report = {'status': 'PASS', 'identity': lock['identity'], 'profile': p.profile,
              'expected_runs': len(expected), 'completed_runs': len(expected), 'backbone_runs': len(native),
              'readout_runs': len({r['run_key'] for r in readout_rows}),
              'e2e_readout_runs': sum(r.kind == 'e2e_readout' for r in expected), 'probe_rows': len(probes),
              'aliases': lock['aliases'], 'model_seeds': list(p.seeds),
              'uncertainty_scope': 'model-seed variability on one locked user split, not cross-split uncertainty',
              'shuffle_averaging': 'replicates averaged within model seed before seed statistics',
              'notes': ['Fixed250 shuffle holds partial/padding bins fixed.', 'Relative10 is an offline duration-aware reference, not an upper bound.',
                        'G_order measures decoder-accessible temporal alignment, not pure information or abstraction.']}
    save_json(summary_root / 'manifest.json', report)
    (root / 'incomplete.json').unlink(missing_ok=True)
    return report
