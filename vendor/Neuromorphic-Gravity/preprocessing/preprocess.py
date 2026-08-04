import sys
import os

import numpy as np
import pandas as pd
from scipy import signal
from rockpool.devices.xylo.syns63300.imuif import RotationRemoval
from rockpool.devices.xylo.syns63300 import Quantizer

import argparse
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_encoders import (
    threshold_encoding,
    step_forward_encoding,
    moving_window_encoding,
    zero_cross_step_forward_encoding,
    time_to_first_spike_encoding,
    phase_encoding,
    bsa_encoding,
    hough_spiker_encoding,
    modified_hough_spiker_encoding,
    burst_encoding,
    poisson_rate_encoding,
    FilterBank
)

# Maps CLI encoder name → (function, parameter names it accepts)
SPIKE_ENCODER_MAP = {
    'TBR':      (threshold_encoding,              ['threshold']),
    'SF':       (step_forward_encoding,           ['threshold']),
    'MW':       (moving_window_encoding,          ['window_size']),
    'ZCSF':     (zero_cross_step_forward_encoding,['threshold']),
    'TTFS':     (time_to_first_spike_encoding,    ['window_size']),
    'Phase':    (phase_encoding,                  ['num_bits']),
    'BSA':      (bsa_encoding,                    ['filter_order', 'threshold']),
    'HSA':      (hough_spiker_encoding,           ['filter_order']),
    'MHSA':     (modified_hough_spiker_encoding,  ['filter_order', 'threshold']),
    'Burst':    (burst_encoding,                  ['n_max', 't_min', 't_max']),
    'Poisson':  (poisson_rate_encoding,           ['interval_length', 'seed']),
}


# Dataset-specific configuration
DATASET_CONFIGS = {
    'pamap':       {'native_freq': 100, 'repeat': 1000, 'is_normalized': True,  'unique_pid': False}, #
    'wisdm':       {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': False}, #
    'opportunity': {'native_freq': 33,  'repeat': 330,  'is_normalized': True,  'unique_pid': False}, #
    'mhealth':     {'native_freq': 50,  'repeat': 1,    'is_normalized': False, 'unique_pid': False}, # subjects 1-10
    'umahand':     {'native_freq': 100, 'repeat': 1,    'is_normalized': True,  'unique_pid': False}, # subjects 1-25
    'capture24':   {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},  #
    'adl':         {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},  #
    'realworld':   {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},  # subjects 0-13
}

DEFAULT_BASE_PATH = '/work/pi_sunghoonlee_umass_edu/Ignacio'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()),
                    help='Dataset name')
parser.add_argument('--subject_ID', type=int, default=1, help='Subject ID')
parser.add_argument('--use_xylo', action='store_true', help='Use Xylo pipeline')
parser.add_argument('--spike_encoder', type=str, default=None,
                    choices=list(SPIKE_ENCODER_MAP.keys()),
                    help='Spike encoder from spike_encoders.py to apply after gravity removal')
parser.add_argument('--window_size', type=float, default=10.0, help='Size of windows (in seconds)')
parser.add_argument('--raw_data', action='store_true', help='Use raw data for windowing')
parser.add_argument('--with_gravity', action='store_true', help='Use data with gravity for windowing')
parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH,
                    help='Base path for dataset storage')
# Encoder-specific parameters
parser.add_argument('--encoder_threshold', type=float, default=0.1,
                    help='Threshold used by threshold, step_forward, zero_cross_step_forward, '
                         'bsa, and modified_hough_spiker encoders')
parser.add_argument('--encoder_window_size', type=int, default=10,
                    help='Window size used by moving_window and time_to_first_spike encoders')
parser.add_argument('--encoder_filter_order', type=int, default=11,
                    help='Filter order used by bsa, hough_spiker, and modified_hough_spiker encoders')
parser.add_argument('--encoder_interval_length', type=int, default=4,
                    help='Interval length used by poisson_rate encoder')
parser.add_argument('--encoder_seed', type=int, default=0,
                    help='Random seed used by poisson_rate encoder')
parser.add_argument('--encoder_num_bits', type=int, default=4,
                    help='Number of bits used by phase encoder')
parser.add_argument('--encoder_n_max', type=int, default=4,
                    help='Maximum number of spikes per burst (burst encoder)')
parser.add_argument('--encoder_t_min', type=int, default=2,
                    help='Minimum inter-spike interval in samples (burst encoder)')
parser.add_argument('--encoder_t_max', type=int, default=6,
                    help='Maximum inter-spike interval in samples (burst encoder)')
args = parser.parse_args()

dataset = args.dataset
subjID = args.subject_ID
useXylo = args.use_xylo
spikeEncoder = args.spike_encoder
sampleFreq = 64
winSize = args.window_size

cfg = DATASET_CONFIGS[dataset]
native_freq = cfg['native_freq']
repeat = cfg['repeat']
unique_pid = cfg['unique_pid']
is_normalized = cfg['is_normalized']

dataset_path = f'{args.base_path}/{dataset}'


# ── Step 1: Gravity removal ──────────────────────────────────────────────────

if not args.raw_data:
    P = np.load(f'{dataset_path}/pid.npy')
    X = np.load(f'{dataset_path}/X.npy')
    Y = np.load(f'{dataset_path}/Y.npy')

    if unique_pid:
        _, P = np.unique(P, return_inverse=True)

    mask = P == subjID
    P, X, Y = P[mask], X[mask].reshape(-1, 3), Y[mask].repeat(repeat)
    time_index = pd.to_datetime(np.arange(len(X)) / float(native_freq), unit='s').set_names('time')

    data = pd.DataFrame(data=np.concatenate([X, Y[:, None]], 1), index=time_index,
                        columns=['x', 'y', 'z', 'annotation']) \
        .astype({'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
    data = data.resample(f'{1000 / sampleFreq}ms').nearest()

    accL = data[['x', 'y', 'z']].to_numpy()
    if not is_normalized: accL /= 9.8 # Normalize if not normalized

    def XyloRotateAndRemoveGravity(accL, fS):
        # Define Xylo modules
        quantizer = Quantizer()
        rot_removal = RotationRemoval(num_avg_bitshift=4, sampling_period=10)

        # Quantize and remove rotation
        accL_quant, _, _ = quantizer(np.clip(accL / 2, -1+1e-5, 1-1e-5))
        accG, _, _ = rot_removal(accL_quant)

        # Convert back to signed integer
        accG = accG[0].astype(int)

        # Renormalize data
        accLMean = np.mean(np.linalg.norm(accL, axis=1))
        accGMean = np.mean(np.linalg.norm(accG, axis=1))
        factor = accLMean / accGMean
        accG = accG * factor

        # Remove gravity
        b, a = signal.butter(4, 0.2, 'highpass', fs=fS)
        accG = signal.lfilter(b, a, accG, axis=0)

        return accG


    accG = XyloRotateAndRemoveGravity(accL, sampleFreq)
    data.loc[:, ['x', 'y', 'z']] = accG

    if not os.path.isdir(f'{dataset_path}/data/{dataset}_gr/'):
        os.makedirs(f'{dataset_path}/data/{dataset}_gr/')
    gr_path = f'{dataset_path}/data/{dataset}_gr/P{subjID:03d}.csv.gz'
    data.to_csv(gr_path, compression='gzip')


    # ── Step 2: Pipeline forward pass ───────────────────────────────────────────

    data = pd.read_csv(gr_path, index_col='time', parse_dates=['time'],
                    dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
    data = data.resample(f'{1000 / sampleFreq}ms').nearest()
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']].interpolate()
    data.reset_index(inplace=True)
    data.time = data.time.apply(lambda x: x.timestamp())
    data.time = data.time - data.time[0]

    if useXylo:
        from rockpool.devices.xylo.syns63300 import IMUIFSim
        from rockpool.devices.xylo.syns63300.imuif import FilterBank

        input_data = data[['x', 'y', 'z']].to_numpy()
        input_data = np.clip(input_data / 2, -1+1e-5, 1-1e-5)
        quantizer = Quantizer(shape=3, num_bits=16)
        Q_data, _, _ = quantizer(input_data)

        # Using default values for frequency bands
        frequencies = [sampleFreq / 200 * np.sqrt(fi * fj)
                    for fi, fj in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]]
        mod = IMUIFSim(bypass_jsvd=True, sampling_freq=sampleFreq, select_iaf_output=True)

        try:
            spikes, _, r_d = mod(Q_data, record=True)
        except ValueError:
            sys.exit()

        outLabels = [f'spikes_{d}_{1/frequencies[j]:.2f}' for d in 'xyz' for j in range(5)]
        data[outLabels] = spikes.reshape(-1, 3 * 5).astype(bool)

    elif spikeEncoder:
        # Build keyword arguments for the selected encoder from CLI parameters
        _enc_all_params = {
            'threshold':        args.encoder_threshold,
            'window_size':      args.encoder_window_size,
            'filter_order':     args.encoder_filter_order,
            'interval_length':  args.encoder_interval_length,
            'seed':             args.encoder_seed,
            'num_bits':         args.encoder_num_bits,
            'n_max':            args.encoder_n_max,
            't_min':            args.encoder_t_min,
            't_max':            args.encoder_t_max,
        }
        enc_fn, enc_param_names = SPIKE_ENCODER_MAP[spikeEncoder]
        enc_kwargs = {k: _enc_all_params[k] for k in enc_param_names}

        acc_data = data[['x', 'y', 'z']].to_numpy(dtype=float)
        filterbank = FilterBank(fs=sampleFreq, channels=5, f_min=0.32, f_max=10.24, order=1)
        channels_data = filterbank.decompose(acc_data).reshape(-1, 15)
        encoded = enc_fn(channels_data, **enc_kwargs)

        frequencies = filterbank.center_frequencies
        outLabels = [f'spikes_{d}_{1/frequencies[j]:.2f}' for d in 'xyz' for j in range(5)]
        data[outLabels] = encoded

    else:
        import torch
        sys.path.append('/home/igavier_umass_edu/Documents/Neuromorphic-IMU/python-pipeline/')
        from modules import *
        from wavelets import *

        # Set parameters for the IMU pipeline
        globalFrame = True
        usePower = False
        waveletFunc = powerWavelet if usePower else accelerationWavelet
        frequencies = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0])
        singleDim = False
        dim = 1 if singleDim else 3
        numFilt = len(frequencies)

        # Initialize IMU pipeline
        nimup = NeuromorphicIMUPipeline(
            sampleFreq,
            globalFrame=globalFrame,
            usePower=usePower,
            singleDim=singleDim,
            frequencies=frequencies,
            waveletFunc=waveletFunc
        )

        # Forward pass data
        out = []
        for t, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            inp = row[['x', 'y', 'z']].to_numpy(dtype=float)
            res = nimup(torch.Tensor(inp))
            out += [res.numpy()]

        outLabels = [f'events_{d}_{1/frequencies[j]:.2f}'
                    for d in ('xyz' if dim == 3 else '_') for j in range(numFilt)]
        data[outLabels] = np.stack(out, 0).reshape(-1, dim * numFilt)

    if spikeEncoder:
        pipeline_out_dir = f'spikes{spikeEncoder}'
    elif useXylo:
        pipeline_out_dir = 'spikesXylo'
    else:
        pipeline_out_dir = 'eventsNIMU'
    if not os.path.isdir(f'{dataset_path}/data/{pipeline_out_dir}/'):
        os.makedirs(f'{dataset_path}/data/{pipeline_out_dir}/')
    pipeline_out_path = f'{dataset_path}/data/{pipeline_out_dir}/P{subjID:03d}.csv.gz'
    data.to_csv(pipeline_out_path, compression='gzip', index=False)


# ── Step 3: Windowing ────────────────────────────────────────────────────────

if args.raw_data:
    if args.with_gravity and dataset != 'capture24':
        # Load original raw data with gravity from npy files
        P_w = np.load(f'{dataset_path}/pid.npy')
        X_w = np.load(f'{dataset_path}/X.npy')
        Y_w = np.load(f'{dataset_path}/Y.npy')
        if unique_pid:
            _, P_w = np.unique(P_w, return_inverse=True)
        mask_w = P_w == subjID
        P_w, X_w, Y_w = P_w[mask_w], X_w[mask_w].reshape(-1, 3), Y_w[mask_w].repeat(repeat)
        time_index_w = pd.to_datetime(np.arange(len(X_w)) / float(native_freq), unit='s').set_names('time')
        win_data = pd.DataFrame(data=np.concatenate([X_w, Y_w[:, None]], 1), index=time_index_w,
                                columns=['x', 'y', 'z', 'annotation'])
        win_data = win_data.resample(f'{1000 / sampleFreq}ms').nearest()
    elif args.with_gravity and dataset == 'capture24':
        win_data = pd.read_csv(f'{dataset_path}/data/{dataset}_wg/P{subjID:03d}.csv.gz',
                               index_col='time', parse_dates=['time'], dtype={'annotation': 'string'})
        win_data = win_data.resample(f'{1000 / sampleFreq}ms').nearest()
    else:
        win_data = pd.read_csv(gr_path, index_col='time', parse_dates=['time'],
                               dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
        win_data = win_data.resample(f'{1000 / sampleFreq}ms').nearest()

    win_data[['x', 'y', 'z']] = win_data[['x', 'y', 'z']].interpolate()
    win_data.reset_index(inplace=True)
    win_data.time = win_data.time.apply(lambda x: x.timestamp())
    win_data.time = win_data.time - win_data.time[0]

else:
    # Pipeline output was produced in Step 2 above; load it directly
    win_data = pd.read_csv(pipeline_out_path)

win_data['datetime'] = pd.to_datetime(win_data.time, unit='s')

if dataset == 'capture24':
    annot_data = pd.read_csv(
        f'{dataset_path}/data/{dataset}_wg/annotation-label-dictionary.csv',
        index_col='annotation', dtype=str)

S, y = [], []

for t, d in tqdm.tqdm(win_data.resample(f'{winSize}s', origin='start', on='datetime')):
    if len(d) != int(sampleFreq * winSize):
        continue
    if d['annotation'].isna().any():
        continue

    if args.raw_data:
        S += [d[['x', 'y', 'z']].to_numpy()]
    elif spikeEncoder:
        S += [d[[c for c in d.columns if 'spikenc' in c]].to_numpy()]
    else:
        S += [d[[c for c in d.columns if ('spikes' if useXylo else 'events') in c]].to_numpy()]

    if dataset == 'capture24':
        y += [annot_data.loc[d['annotation'].mode()[0], 'label:Willetts2018']]
    else:
        y += [d['annotation'].mode()[0]]

S = np.stack(S)
y = np.stack(y)

if args.raw_data:
    if args.with_gravity:
        wg_dir = f'{dataset}_wg'
        if not os.path.isdir(f'{dataset_path}/data/{wg_dir}/windows'):
            os.makedirs(f'{dataset_path}/data/{wg_dir}/windows')
        np.save(f'{dataset_path}/data/{wg_dir}/windows/P{subjID:03d}_spikes.npy', S)
        np.save(f'{dataset_path}/data/{wg_dir}/windows/P{subjID:03d}_labels.npy', y)
    else:
        if not os.path.isdir(f'{dataset_path}/data/{dataset}_gr/windows'):
            os.makedirs(f'{dataset_path}/data/{dataset}_gr/windows')
        np.save(f'{dataset_path}/data/{dataset}_gr/windows/P{subjID:03d}_spikes.npy', S)
        np.save(f'{dataset_path}/data/{dataset}_gr/windows/P{subjID:03d}_labels.npy', y)
else:
    if not os.path.isdir(f'{dataset_path}/data/{pipeline_out_dir}/windows'):
        os.makedirs(f'{dataset_path}/data/{pipeline_out_dir}/windows')
    np.save(f'{dataset_path}/data/{pipeline_out_dir}/windows/P{subjID:03d}_spikes.npy', S)
    np.save(f'{dataset_path}/data/{pipeline_out_dir}/windows/P{subjID:03d}_labels.npy', y)
