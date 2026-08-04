import os
import sys
sys.path.append('/home/igavier_umass_edu/Documents/Neuromorphic-IMU/python-pipeline/')
from modules import *
from wavelets import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, integrate, stats, interpolate
import seaborn as sn
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 150

import argparse

DEFAULT_BASE_PATH = '/work/pi_sunghoonlee_umass_edu/Ignacio'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--subject_ID', type=int, default=1, help='Subject ID')
parser.add_argument('--dataset', type=str, default='wisdm', help='Dataset')
parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH, help='Base path for dataset storage')
parser.add_argument('--results_dir', type=str, default=None, help='Directory for metrics CSV output (defaults to <base_path>/metrics)')
parser.add_argument('--use_xylo', action='store_true', help='Use Xylo pipeline')
args = parser.parse_args()

subjID = args.subject_ID
datasetName = args.dataset
useXylo = args.use_xylo
base_path = args.base_path
results_dir = args.results_dir or os.path.join(args.base_path, 'metrics')

fileName = f'{base_path}/{datasetName}/data/{datasetName}_gr/P{subjID:03d}.csv.gz'
print(f'Processing {fileName} with {"xylo" if useXylo else "nimu"}')

# Load data
data = pd.read_csv(fileName,
                   index_col='time', parse_dates=['time'],
                   dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'})
data = data.resample(f'{1000 / 64}ms').nearest() # Resample to sampleFreq
data[['x','y','z']] = data[['x','y','z']].interpolate()
# data['annotation'] = data['annotation'].ffill()
data.reset_index(inplace=True)
data.time = data.time.apply(lambda x: x.timestamp())
data.time = data.time - data.time[0]
dataFrameLeft = data[['time','x','y','z']]
dataFrameLeft.rename(columns={'x': 'rawacc_x', 'y': 'rawacc_y', 'z': 'rawacc_z'}, inplace=True)

# Interpolate data
interpFx = interpolate.interp1d(dataFrameLeft.time, dataFrameLeft.values, axis=0)
sampleFreq = 64
dataFrameLeft = pd.DataFrame(interpFx(dataFrameLeft.time.iloc[0] +
                                      np.arange(int((dataFrameLeft.time.iloc[-1] - dataFrameLeft.time.iloc[0]) * sampleFreq),
                                                dtype=float) / sampleFreq),
                             columns=dataFrameLeft.columns)

# Get sample frequency
sampleFreq = round(1 / np.diff(dataFrameLeft['time']).mean())

b, a = signal.butter(4, 0.1 / (sampleFreq / 2), btype='highpass')
for d in 'xyz': dataFrameLeft[f'rawacc_{d}'] = signal.filtfilt(b, a, dataFrameLeft[f'rawacc_{d}'])

# Select a time window
tIni, tFin = 100, dataFrameLeft['time'].iloc[-1]-100 # 19230, 19330
tMask = (dataFrameLeft['time'] >= tIni) * (dataFrameLeft['time'] < tFin)
dataFrameLeft = dataFrameLeft[tMask]
dataFrameLeft.reset_index(inplace=True, drop=True)

# Filter signal
b, a = signal.butter(4, 20.0 / (sampleFreq / 2), btype='lowpass')
for d in 'xyz': dataFrameLeft[f'acc_{d}'] = signal.filtfilt(b, a, dataFrameLeft[f'rawacc_{d}'])

# Get velocity and power
for d in 'xyz': dataFrameLeft[f'vel_{d}'] = integrate.cumulative_trapezoid(dataFrameLeft[f'acc_{d}'], dataFrameLeft['time'], initial=0)
for d in 'xyz': dataFrameLeft[f'pwr_{d}'] = dataFrameLeft[f'acc_{d}'] * dataFrameLeft[f'vel_{d}']

if not useXylo:
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
    rec = []
    for t, row in tqdm(dataFrameLeft.iterrows(), total=len(dataFrameLeft)):
        inp = row[[f'rawacc_{d}' for d in 'xyz']].to_numpy(dtype=float)
        res = nimup(torch.Tensor(inp))
        # rec += [res[0].numpy()]
        # out += [res[1].numpy()]
        out += [res.numpy()]

    # Store it in the dataframe
    outLabels = [f'events_{d}_{1/frequencies[j]:.2f}'
                 for d in ('xyz' if dim == 3 else '_') for j in range(numFilt)]
    dataFrameLeft[outLabels] = np.stack(out, 0).reshape(-1, dim * numFilt)

    maxS = int((sampleFreq / frequencies).max())
    maxM = int(np.min([2 * maxS, len(dataFrameLeft)]))
    delayMaxFilter = nimup.get_submodule('model.3.maxPooler').kernel_size[1]
    for d in 'xyz':
        dataFrameLeft[f'rec_{d}'] = 0.0
        for j in range(numFilt):
            si = sampleFreq / frequencies[j]
            M = np.min([2 * si, len(dataFrameLeft)]).astype(int)
            impulse = signal.unit_impulse(maxM + delayMaxFilter, maxM // 2 - int(si) // 2)
            # nimup.get_submodule('model.2').filterBank.shape[0] // 2
            wavelet = signal.convolve(impulse, waveletFunc(si, si).flip(0), 'same')
            dataFrameLeft[f'rec_{d}'] += signal.convolve(dataFrameLeft[f'events_{d}_{1/frequencies[j]:.2f}'],
                                                         wavelet, 'same') / 2.5
            # plt.plot(wavelet)
        # dataFrameLeft[f'rec_{d}'] = integrate.cumulative_trapezoid(dataFrameLeft[f'rec_{d}'], dataFrameLeft['time'], initial=0)
        # dataFrameLeft[f'rec_{d}'] = np.sqrt(2 * np.abs(dataFrameLeft[f'rec_{d}']))
        # dataFrameLeft[f'rec_{d}'] = np.gradient(dataFrameLeft[f'rec_{d}'], dataFrameLeft['time']) 
        # dataFrameLeft[f'rec_{d}'] = signal.filtfilt(b, a, dataFrameLeft[f'rec_{d}'])
    
    delaySamples = maxS // 2 + delayMaxFilter // 2

    inp = dataFrameLeft[[f'acc_{d}' for d in 'xyz']].to_numpy()[:-delaySamples]
    diff = (dataFrameLeft[[f'acc_{d}' for d in 'xyz']].to_numpy()[:-delaySamples] -
           dataFrameLeft[[f'rec_{d}' for d in 'xyz']].to_numpy()[:-delaySamples])
    spk = dataFrameLeft[[f'events_{d}_{1/frequencies[j]:.2f}' for d in ('xyz' if dim == 3 else '_')
                         for j in range(numFilt)]].to_numpy()[:-delaySamples]
    
    nrmse = np.sqrt(((diff) ** 2).mean()) * 9.81 / 9
    snr = 10 * np.log10(np.sqrt((inp ** 2).mean()) / np.sqrt((diff ** 2).mean()))
    # spks = (spk != 0).mean() * 64
    spks = (16. * (np.abs(spk) / 16.) ** (1./3)).mean() * sampleFreq
    snrperspks = snr / spks
    
    # print(len(dataFrameLeft)/64)
    # print(nrmse, snr, spks, snrperspks)
    
    result = {'subject_ID':[subjID], 'method':['NIMU'], 'duration':[len(dataFrameLeft)/64], 'nrmse':[nrmse], 'snr':[snr], 'spks':[spks], 'snrperspks':[snrperspks]}

else:
    from rockpool.devices.xylo.syns63300 import IMUIFSim
    from rockpool.devices.xylo.syns63300.imuif import FilterBank
    from rockpool.devices.xylo.syns63300 import Quantizer
    
    input_data = dataFrameLeft[[f'rawacc_{d}' for d in 'xyz']].to_numpy(dtype=float) #/ 9.81
    input_data = np.clip(input_data / 2, -1+1e-5, 1-1e-5)
    quantizer = Quantizer(shape=3, num_bits=16)
    Q_data, _, _ = quantizer(input_data)
    
    # Using default values for frequency bands
    frequencies = [(sampleFreq / 200 * fi, sampleFreq / 200 * fj)
                   for fi, fj in [(1,2),(2,4),(4,8),(8,16),(16,32)]]
    fb = FilterBank.from_specification((3, 15), *(frequencies * 3))
    mod = IMUIFSim(bypass_jsvd=True, sampling_freq=sampleFreq, select_iaf_output=True)#, filter_list=fb._filters)
    
    spikes , _, r_d = mod(Q_data, record = True)

    # Here we should use rec_data from the spikes and then sum across channels
    rec_data, _, r_d = mod.model[0](Q_data, record = True)
    # rec_data = spikes
    rec_data = rec_data[0].astype(int).T.reshape(3,5,-1).sum(1).T / 2 ** 15
    
    nrmse = np.sqrt(((input_data - rec_data) ** 2).mean()) * 9.81 / 9
    snr = 10 * np.log10(np.sqrt((input_data ** 2).mean()) / np.sqrt(((input_data - rec_data) ** 2).mean()))
    # spks = (spikes != 0).mean() * 64
    spks = np.abs(spikes).mean() * sampleFreq
    snrperspks = snr / spks

    # print(len(dataFrameLeft)/64)
    # print(nrmse, snr, spks, snrperspks)

    result = {'subject_ID':[subjID], 'method':['Xylo'], 'duration':[len(dataFrameLeft)/64], 'nrmse':[nrmse], 'snr':[snr], 'spks':[spks], 'snrperspks':[snrperspks]}

print(result)
result = pd.DataFrame(result)
result.to_csv(f'{results_dir}/spike_encoder_neurobench_{datasetName}_{subjID:03d}_{"Xylo" if useXylo else "NIMU"}.csv')