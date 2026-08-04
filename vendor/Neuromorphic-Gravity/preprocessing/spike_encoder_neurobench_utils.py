"""spike_encoder_neurobench_utils.py

Compute neurobenchmark encoding metrics (NRMSE, SNR, spike rate,
SNR-per-spike-rate) for spike-encoded IMU datasets produced by
preprocess.py.

Usage
-----
Scan every existing ``spikesEncoder_*`` directory under ``--base_path``
and compute metrics for all found dataset / subject / encoder combinations::

    python spike_encoder_neurobench_utils.py --compute_metrics

Or run for a specific dataset / subject / encoder::

    python spike_encoder_neurobench_utils.py \\
        --dataset realworld --subject_ID 3 --spike_encoder step_forward
"""

import sys
import os

import numpy as np
import pandas as pd
from scipy.signal.windows import get_window
from scipy.ndimage import uniform_filter1d

import argparse
import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike_encoders import (  # noqa: E402 -- after sys.path update
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

# Dataset-specific configuration (mirrors preprocess.py)
DATASET_CONFIGS = {
    'pamap':       {'native_freq': 100, 'repeat': 1000, 'is_normalized': True,  'unique_pid': False},
    'wisdm':       {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': False},
    'opportunity': {'native_freq': 33,  'repeat': 330,  'is_normalized': True,  'unique_pid': False},
    'mhealth':     {'native_freq': 50,  'repeat': 1,    'is_normalized': False, 'unique_pid': False},
    'umahand':     {'native_freq': 100, 'repeat': 1,    'is_normalized': True,  'unique_pid': False},
    'capture24':   {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},
    'adl':         {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},
    'realworld':   {'native_freq': 30,  'repeat': 300,  'is_normalized': True,  'unique_pid': True},
}

DEFAULT_BASE_PATH = '/work/pi_sunghoonlee_umass_edu/Ignacio'

# Fixed resampling rate used throughout preprocess.py
SAMPLE_FREQ = 64  # Hz


# ── Reconstruction helpers ───────────────────────────────────────────────────

def reconstruct_from_spikes(
    spikes: np.ndarray,
    encoder_name: str,
    *,
    threshold: float = 0.1,
    filter_order: int = 11,
    window_size: int = 10,
    interval_length: int = 4,
    num_bits: int = 4,
    n_max: int = 4,
    t_min: int = 2,
    t_max: int = 6,
) -> np.ndarray:
    """Reconstruct a continuous acceleration signal from its spike encoding.

    Applies the inverse (or pseudo-inverse) of each encoding algorithm so
    that quality metrics can be computed against the original signal.

    Parameters
    ----------
    spikes:
        Encoded spike array of shape ``(T, 3)``.
    encoder_name:
        Name of the spike encoder as listed in ``SPIKE_ENCODER_MAP``.
    threshold:
        Step size used by threshold / step-forward / ZCSF encoders.
        Default 0.1.
    filter_order:
        FIR filter length used by deconvolution encoders (BSA / HSA / MHSA).
        Default 11.
    window_size:
        Window length (samples) used by moving-window and TTFS encoders.
        Default 10.
    interval_length:
        Interval length (samples) used by the Poisson-rate encoder.
        Default 4.
    num_bits:
        Number of quantisation bits used by the phase encoder.  Default 4.
    n_max:
        Maximum burst length used by the burst encoder.  Default 4.
    t_min:
        Minimum inter-spike interval used by the burst encoder.  Default 2.
    t_max:
        Maximum inter-spike interval used by the burst encoder.  Default 6.

    Returns
    -------
    rec:
        Reconstructed signal array of shape ``(T, 3)``.
    """
    spikes = np.asarray(spikes, dtype=float)
    T, D = spikes.shape
    rec = np.zeros((T, D), dtype=float)

    if encoder_name in ('TBR', 'SF', 'ZCSF'):
        # Delta-modulation decoding: each spike advances the running baseline
        # by ±threshold, so the reconstruction is a cumulative sum of steps.
        for t in range(1, T):
            rec[t] = rec[t - 1] + spikes[t] * threshold

    elif encoder_name == 'MW':
        # The spikify moving_window encoder derives its threshold from the mean
        # absolute variation of the signal internally, so the exact step size
        # is not available for inversion.  A unit step (matching the original
        # preprocess.py implementation) gives a reasonable approximation.
        for t in range(1, T):
            rec[t] = rec[t - 1] + spikes[t] * 1.0

    elif encoder_name in ('BSA', 'HSA', 'MHSA'):
        # Deconvolution encoders subtract the filter kernel from the residual
        # at each spike location, so the signal is reconstructed by convolving
        # the signed spike train with the same kernel.
        # HSA/MHSA wrappers use Hann windows to avoid near-dead outputs on
        # low-amplitude multiband channels.
        filt = get_window('hann', filter_order).astype(float)
        filt /= filt.sum()
        for d in range(D):
            rec[:, d] = np.convolve(spikes[:, d], filt, mode='full')[:T]

    elif encoder_name == 'TTFS':
        # TTFS encodes amplitude by spike latency within each window:
        # earlier firing → larger amplitude.  The spikify algorithm places the
        # spike at t = floor(window_size * 0.1 * log(1 / normalised_amp)),
        # so the inverse is normalised_amp = exp(-t_spike / (window_size * 0.1)).
        # The global normalisation constant (signal_max) is not recoverable
        # from the spike train alone, so the reconstruction is relative.
        T_trunc = (T // window_size) * window_size
        for d in range(D):
            for w in range(0, T_trunc, window_size):
                w_spk = spikes[w:w + window_size, d]
                pos_times = np.where(w_spk > 0)[0]
                neg_times = np.where(w_spk < 0)[0]
                amp = 0.0
                if len(pos_times) > 0:
                    t_pos = pos_times[0]
                    amp += np.exp(-t_pos / (window_size * 0.1))
                if len(neg_times) > 0:
                    t_neg = neg_times[0]
                    amp -= np.exp(-t_neg / (window_size * 0.1))
                rec[w:w + window_size, d] = amp

    elif encoder_name == 'Phase':
        # Phase encoding returns a binary spike train in time rather than a
        # packed bitstream, so reconstruction should treat it as a rate code.
        # A short windowed average gives a smoother relative amplitude proxy
        # than interpreting consecutive samples as bits.
        T_trunc = (T // num_bits) * num_bits
        for d in range(D):
            for w in range(0, T_trunc, num_bits):
                rec[w:w + num_bits, d] = np.mean(spikes[w:w + num_bits, d])

    elif encoder_name == 'Burst':
        # Burst encoding maps signal amplitude to burst density: fewer spikes
        # per burst → larger amplitude.  Since burst boundaries are implicit,
        # reconstruction uses a sliding average over the maximum burst window.
        burst_window = n_max * t_max
        for d in range(D):
            rec[:, d] = uniform_filter1d(spikes[:, d], size=burst_window, mode='nearest')

    elif encoder_name == 'Poisson':
        # Poisson-rate encoding: spike probability within each interval
        # represents the normalised signal amplitude.  Reconstruction takes
        # the per-interval mean of the signed spike train.
        T_trunc = (T // interval_length) * interval_length
        for d in range(D):
            for w in range(0, T_trunc, interval_length):
                rec[w:w + interval_length, d] = np.mean(spikes[w:w + interval_length, d])

    return rec


def compute_encoding_metrics(
    orig: np.ndarray,
    rec: np.ndarray,
    spikes: np.ndarray,
    sample_freq: float,
) -> dict:
    """Compute signal encoding quality metrics.

    Parameters
    ----------
    orig:
        Original (gravity-removed) signal of shape ``(T, 3)``.
    rec:
        Reconstructed signal of shape ``(T, 3)``.
    spikes:
        Encoded spike array of shape ``(T, 3)``.
    sample_freq:
        Sampling frequency in Hz.

    Returns
    -------
    dict with keys ``nrmse``, ``snr``, ``spks``, ``snrperspks``.
    """
    # Validate shapes - ensure both are (T, 3)
    if orig.shape[0] != rec.shape[0] or orig.shape[1] != 3 or rec.shape[1] != 3:
        print(f'[warning] shape mismatch: orig={orig.shape}, rec={rec.shape}')
        return {
            'nrmse': float('nan'),
            'snr': float('nan'),
            'spks': float('nan'),
            'snrperspks': float('nan'),
        }

    # Invalid spike/reconstruction arrays should not produce seemingly valid
    # metrics; return NaNs so upstream aggregation can detect corrupted inputs.
    if np.isnan(spikes).any():
        print('[warning] spikes contain NaN values; metrics marked as invalid')
        return {
            'nrmse': float('nan'),
            'snr': float('nan'),
            'spks': float('nan'),
            'snrperspks': float('nan'),
        }
    
    # Handle NaN values in reconstruction
    if np.isnan(rec).all():
        print('[warning] reconstruction is all NaN; metrics marked as invalid')
        return {
            'nrmse': float('nan'),
            'snr': float('nan'),
            'spks': float('nan'),
            'snrperspks': float('nan'),
        }
    elif np.isnan(rec).any():
        print('[warning] reconstruction contains NaN values; metrics marked as invalid')
        return {
            'nrmse': float('nan'),
            'snr': float('nan'),
            'spks': float('nan'),
            'snrperspks': float('nan'),
        }
    
    diff = orig - rec
    # NRMSE scaled to physical units: multiply by standard gravity (9.81 m/s²)
    # and divide by the reference acceleration scale (9 m/s² ≈ 1 g), matching
    # the convention used in the original preprocess.py.
    nrmse = float(np.sqrt((diff ** 2).mean()) * 9.81 / 9)
    signal_rms = float(np.sqrt((orig ** 2).mean()))
    noise_rms = float(np.sqrt((diff ** 2).mean()))
    if noise_rms > 0 and signal_rms > 0:
        snr = float(10 * np.log10(signal_rms / noise_rms))
    elif noise_rms == 0:
        # Perfect reconstruction: cap at a large but finite value so that
        # downstream aggregation (mean, CSV storage) stays well-defined.
        snr = 100.0
    else:
        snr = float('nan')
    # spks = float((spikes != 0).mean() * sample_freq)
    spks = np.abs(spikes).mean() * sample_freq
    snrperspks = float(snr / spks) if spks > 0 else float('nan')
    return {'nrmse': nrmse, 'snr': snr, 'spks': spks, 'snrperspks': snrperspks}


def _process_one(dataset_name, subj_id, encoder_name, base_path, out_dir, enc_params):
    """Load one spike-encoded subject file, reconstruct, and save metrics.

    Parameters
    ----------
    dataset_name, subj_id, encoder_name:
        Identify the file to process.
    base_path:
        Root directory that contains per-dataset subdirectories.
    out_dir:
        Directory where the per-subject metrics CSV is written.
    enc_params:
        Keyword arguments forwarded to :func:`reconstruct_from_spikes`.

    Returns
    -------
    dict with result fields, or ``None`` when the file is missing / invalid.
    """
    spike_path = os.path.join(
        base_path, dataset_name, 'data',
        f'spikes{encoder_name}',
        f'P{subj_id:03d}.csv.gz',
    )
    if not os.path.isfile(spike_path):
        print(f'[skip] file not found: {spike_path}')
        return None

    df = pd.read_csv(spike_path, compression='gzip')
    # required_cols = {'x', 'y', 'z', 'spikenc_x', 'spikenc_y', 'spikenc_z'}
    required_cols = {'x', 'y', 'z', *[c for c in df.columns if 'spikes' in c]}
    if not required_cols.issubset(df.columns):
        print(f'[skip] missing expected columns in {spike_path}')
        return None

    orig   = df[['x',        'y',        'z'       ]].to_numpy(dtype=float)
    # spikes = df[['spikenc_x','spikenc_y','spikenc_z']].to_numpy(dtype=float)
    spikes = df[[c for c in df.columns if 'spikes' in c]].to_numpy(dtype=float)

    rec = reconstruct_from_spikes(spikes, encoder_name, **enc_params)
    
    # Aggregate multi-band reconstructed signal: reshape 15 channels to (N, 3, 5)
    # where 3 = spatial dimensions and 5 = frequency bands, then sum across bands
    # to recover 3D representation for comparison with original signal
    try:
        if rec.shape[1] == 15:
            rec = rec.reshape(-1, 3, 5).sum(2)  # Sum across frequency bands to get 3D
        elif rec.shape[1] != 3:
            # If shape is unexpected, try to handle gracefully
            print(f'[warning] unexpected reconstruction shape: {rec.shape}')
            rec = rec[:, :3] if rec.shape[1] >= 3 else np.zeros((rec.shape[0], 3))
    except (ValueError, IndexError) as e:
        print(f'[warning] reconstruction shape handling failed: {e}')
        rec = np.zeros((orig.shape[0], 3))

    # Deconvolution encoders normalize inputs per-polarity before encoding
    # (see preprocessing/spike_encoders._bipolar_per_channel where
    # `normalize_peak=4.0` is used).  The spike-only reconstruction is therefore
    # in the normalized units; rescale positive/negative components back to
    # the original signal amplitude using per-channel polarity peaks so that
    # metric comparisons are meaningful.
    if encoder_name in ('BSA', 'HSA', 'MHSA'):
        normalize_peak = 4.0
        T_rec = rec.shape[0]
        # Ensure orig has same length as rec (truncate/pad if necessary)
        if orig.shape[0] != T_rec:
            n = min(orig.shape[0], T_rec)
            orig_crop = orig[:n]
            rec_crop = rec[:n]
            pad_tail = T_rec - n
        else:
            orig_crop = orig
            rec_crop = rec
            pad_tail = 0

        for d in range(3):
            pos_max = float(np.max(np.maximum(orig_crop[:, d], 0.0)))
            neg_max = float(np.max(np.maximum(-orig_crop[:, d], 0.0)))

            rec_pos = np.maximum(rec_crop[:, d], 0.0)
            rec_neg = np.maximum(-rec_crop[:, d], 0.0)

            if pos_max > 0:
                rec_pos = rec_pos * (pos_max / normalize_peak)
            else:
                rec_pos = rec_pos * 0.0

            if neg_max > 0:
                rec_neg = rec_neg * (neg_max / normalize_peak)
            else:
                rec_neg = rec_neg * 0.0

            rec_crop[:, d] = rec_pos - rec_neg

        if pad_tail:
            # place adjusted portion back and leave any extra tail zeros as-is
            rec[:rec_crop.shape[0], :] = rec_crop
        else:
            rec = rec_crop
    
    # Compare original 3D signal to reconstructed 3D signal (from summed multiband reconstruction)
    metrics = compute_encoding_metrics(orig, rec, spikes, SAMPLE_FREQ)

    result = {
        'dataset':    dataset_name,
        'subject_ID': subj_id,
        'method':     encoder_name,
        'duration':   len(orig) / SAMPLE_FREQ,
        **metrics,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f'spike_encoder_neurobench_{dataset_name}_{subj_id:03d}_{encoder_name}.csv',
    )
    pd.DataFrame([result]).to_csv(out_path, index=False)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--dataset', type=str, default=None,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset name (required when not using --compute_metrics)',
    )
    p.add_argument('--subject_ID', type=int, default=1, help='Subject ID')
    p.add_argument(
        '--spike_encoder', type=str, default=None,
        choices=list(SPIKE_ENCODER_MAP.keys()),
        help='Spike encoder to evaluate (required when not using --compute_metrics)',
    )
    p.add_argument(
        '--base_path', type=str, default=DEFAULT_BASE_PATH,
        help='Base path for dataset storage',
    )
    p.add_argument(
        '--results_dir', type=str, default=None,
        help='Directory for metrics CSV output (defaults to <base_path>/metrics)',
    )
    p.add_argument(
        '--compute_metrics', action='store_true',
        help=(
            'Scan all existing spikes* directories under --base_path '
            'and compute encoding metrics for every dataset / subject / encoder '
            'combination found'
        ),
    )
    # Encoder-specific parameters (defaults match preprocess.py)
    p.add_argument('--encoder_threshold',       type=float, default=0.1)
    p.add_argument('--encoder_window_size',     type=int,   default=10)
    p.add_argument('--encoder_filter_order',    type=int,   default=11)
    p.add_argument('--encoder_interval_length', type=int,   default=4)
    p.add_argument('--encoder_seed',            type=int,   default=0)
    p.add_argument('--encoder_num_bits',        type=int,   default=4)
    p.add_argument('--encoder_n_max',           type=int,   default=4)
    p.add_argument('--encoder_t_min',           type=int,   default=2)
    p.add_argument('--encoder_t_max',           type=int,   default=6)
    return p


if __name__ == '__main__':
    parser = _build_parser()
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(args.base_path, 'metrics')

    enc_params = dict(
        threshold=args.encoder_threshold,
        filter_order=args.encoder_filter_order,
        window_size=args.encoder_window_size,
        interval_length=args.encoder_interval_length,
        num_bits=args.encoder_num_bits,
        n_max=args.encoder_n_max,
        t_min=args.encoder_t_min,
        t_max=args.encoder_t_max,
    )

    # ── --compute_metrics mode: scan all existing datasets ──────────────────

    if args.compute_metrics:
        all_results = []
        for dataset_name in sorted(DATASET_CONFIGS):
            data_dir = os.path.join(args.base_path, dataset_name, 'data')
            if not os.path.isdir(data_dir):
                continue
            for entry in sorted(os.listdir(data_dir)):
                if not entry.startswith('spikes'):
                    continue
                encoder_name = entry[len('spikes'):]
                if encoder_name not in SPIKE_ENCODER_MAP:
                    continue
                enc_dir = os.path.join(data_dir, entry)
                subject_files = sorted(
                    f for f in os.listdir(enc_dir)
                    if f.startswith('P') and f.endswith('.csv.gz')
                )
                for fname in tqdm.tqdm(
                    subject_files,
                    desc=f'{dataset_name}/{encoder_name}',
                    unit='subj',
                ):
                    try:
                        subj_id = int(fname[1:4])
                    except ValueError:
                        continue
                    result = _process_one(
                        dataset_name, subj_id, encoder_name,
                        args.base_path, results_dir, enc_params,
                    )
                    if result is not None:
                        all_results.append(result)
                        print(
                            f'  {dataset_name}/P{subj_id:03d}/{encoder_name}: '
                            f'SNR={result["snr"]:.2f} dB  '
                            f'spks={result["spks"]:.1f} Hz  '
                            f'NRMSE={result["nrmse"]:.4f}'
                        )

        if all_results:
            os.makedirs(results_dir, exist_ok=True)
            summary_path = os.path.join(results_dir, 'spike_encoder_neurobench_all.csv')
            pd.DataFrame(all_results).to_csv(summary_path, index=False)
            print(f'\nSummary ({len(all_results)} records) saved to {summary_path}')
        else:
            print('No spike-encoded datasets found under base_path. Run preprocess.py first.')
        sys.exit(0)

    # ── Single dataset / subject / encoder mode ─────────────────────────────

    if args.dataset is None or args.spike_encoder is None:
        parser.error('--dataset and --spike_encoder are required when not using --compute_metrics')

    result = _process_one(
        args.dataset, args.subject_ID, args.spike_encoder,
        args.base_path, results_dir, enc_params,
    )
    if result is not None:
        print(result)
