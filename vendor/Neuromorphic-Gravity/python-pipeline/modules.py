from typing import Union
from collections.abc import Callable
import warnings

import torch
import torch.nn as nn
import snntorch as snn
import math

from wavelets import powerWavelet, accelerationWavelet, velocityWavelet
from utils import prony, butter

# Define default frequencies to work with
# (frequencies should be < fs/2; for better performance they should be <= fs/4)
defaultFrequencies = torch.logspace(math.log10(1e-1), math.log10(1e1), 5, base=10)


# Define Neuromorphic pipeline preprocessing
class NeuromorphicIMUPipeline(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 globalFrame: bool = True,
                 usePower: bool = True,
                 singleDim: bool = True,
                 frequencies: torch.Tensor = defaultFrequencies,
                 waveletFunc: Callable = powerWavelet,
                 maxFiltWin: Union[float, float] = (0.3, 0.5),
                 beta: float = 1.0,
                 threshold: float = 1.0,
                 quantized: bool = False):
        '''Initializes neuromorphic pipeline preprocessing
        
        :param sampleFreq: sampling frequency of input signal
        :param globalFrame: whether the input signal is in global or local frame
        :param usePower: whether to use power signal or acceleration signal
        :param singleDim: whether to collapse 3D into 1D or not
        :param frequencies: the ones to be utilized in the wavelet transform
        :param waveletFunc: the function to be utilized as wavelet
        :param maxFiltWin: tuple containing window width in the max filter,
            first is is time (in seconds), and second is frequencies (in decades)
        :param beta: spike-encoder membrane potential decay rate (between 0 and 1)
        :param threshold: spike-encoder threshold for spikes
        :param quantized: whether to use quantized data
        '''
        
        super().__init__()

        # Initialize modules
        if globalFrame:
            localToGlobal = nn.Identity()
        else:
            localToGlobal = LocalToGlobalModule(sampleFreq)
        
        if usePower:
            accelToPower = AccelToPowerModule(
                sampleFreq,
                singleDim=singleDim
            )
        else:
            accelToPower = nn.Identity()
        
        waveletFilter = WaveletIIRFilterModule(
            sampleFreq,
            singleDim=singleDim,
            frequencies=frequencies,
            waveletFunc=waveletFunc
        )
        
        maxFilter = MaxFilterModule(
            sampleFreq,
            usePower=usePower,
            singleDim=singleDim,
            frequencies=frequencies,
            maxFiltWin=maxFiltWin
        )
        
        rectifier = AbsModule()
        
        spikeEncoder = snn.Leaky(
            beta=beta,
            threshold=threshold,
            init_hidden=True
        )

        self.model = nn.Sequential(
            localToGlobal,
            accelToPower,
            waveletFilter,
            maxFilter,
        )
    
    def forward(self,
                accCurr):

        spikesCurr = self.model(accCurr)

        return spikesCurr


class QuantizerModule(nn.Module):

    def __init__(self,
                 numBits: int = 16,
                 scaleFactor: float = 1.0):
        '''Initializes module to quantize acceleration
        
        :param numBits: number of bits to quantize
        :param scaleFactor: scale factor to apply before quantization
        '''

        super().__init__()
        self.scaleFactor = scaleFactor
        self.bitFactor = 2 ** (numBits - 1)
    
    def forward(self,
                accCurr: torch.Tensor): # shape=(dim,)
        
        accQCurr = accCurr * self.scaleFactor * self.bitFactor
        
        if ((accQCurr > self.bitFactor - 1).any() or (accQCurr < -self.bitFactor).any()):
            warnings.warn(f'Overflow encountered accCurr = {accCurr}', RuntimeWarning)
            accQCurr = accQCurr.clamp(-self.bitFactor, self.bitFactor - 1)
        
        accQCurr = accQCurr.to(torch.int)
        
        return accQCurr # shape=(dim,)


class LocalToGlobalModule(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 cutoffFreq: float = 0.2):
        '''Initializes module to convert local to global
        
        :param sampleFreq: sampling frequency of input signal
        '''

        super().__init__()
        dim = 3

        self.gravGlob = torch.Tensor([0.0, 0.0, 1.0]) # shape=(3,)
        coefficients = torch.cat(butter(2, cutoffFreq * 2 / sampleFreq)) # shape=(5,)

        self.register_buffer('filterBankB', coefficients[[0,1,2]]) # shape=(3,)
        self.register_buffer('filterBankA', -coefficients[[4,5]]) # shape=(2,)
        self.register_buffer('accPrev', torch.zeros((dim, 2))) # shape=(dim, 2)
        self.register_buffer('gravPrev', torch.zeros((dim, 2))) # shape=(dim, 2)
    
    def forward(self,
                accCurr: torch.Tensor): # shape=(dim,)
        
        # Prepare vector to be filtered
        accMem = torch.cat([accCurr.unsqueeze(1), self.accPrev], dim=1) # shape=(dim, 3)
            
        # Apply linear filtering using matrix multiplication
        gravCurr = (accMem @ self.filterBankB + self.gravPrev @ self.filterBankA) # shape=(dim,)
        normGrav = torch.norm(gravCurr) + 1e-6
        
        # Apply Rodrigues' rotation formula
        cosAngle = self.gravGlob.dot(gravCurr) / normGrav
        sinAngle = torch.sqrt(1 - cosAngle ** 2)
        vectorRot = torch.cross(gravCurr, self.gravGlob, dim=0) # shape=(dim,)
        vectorRot = vectorRot / (torch.norm(vectorRot) + 1e-6) # shape=(dim,)
        accCorr = (accCurr * cosAngle + torch.cross(vectorRot, accCurr, dim=0) * sinAngle
                   + vectorRot * vectorRot.dot(accCurr) * (1 - cosAngle)) # shape=(dim,)
        
        # Correct gravity
        accCorr = accCorr - self.gravGlob * normGrav # shape=(dim,)

        # Update memory
        self.accPrev = accMem[:,:2] # shape=(dim, 2)
        self.gravPrev = torch.stack([gravCurr, self.gravPrev[:,0]], dim=1) # shape=(dim, 2)
        
        return accCorr # shape=(dim,)


class AccelToPowerModule(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 singleDim: bool = True):
        '''Initializes module to convert acceleration to power
        
        :param sampleFreq: sampling frequency of input signal
        :param singleDim: whether to collapse 3D into 1D or not
        '''
        
        super().__init__()
        self.sampleFreq = sampleFreq
        self.singleDim = singleDim

        self.register_buffer('velPrev', torch.zeros((3,)))
        self.register_buffer('velCurr', torch.zeros((3,)))
        self.register_buffer('accPrev', torch.zeros((3,)))
    
    def forward(self,
                accCurr: torch.Tensor): # shape=(dim,)
        
        # Apply trapezoid method to obtain velocity by integrating acceleration
        velCurr = self.velPrev + (accCurr + self.accPrev) / 2 / self.sampleFreq # shape=(dim,)
        
        # Compute energy (3D or 1D depending on singleDim)
        powerCurr = accCurr * velCurr # shape=(dim,)
        if self.singleDim: powerCurr = torch.sum(powerCurr, dim=0, keepdim=True) # shape=(1,)

        # Update memory
        self.velPrev = velCurr # shape=(dim,)
        self.accPrev = accCurr # shape=(dim,)

        return powerCurr # shape=(dim,)


class WaveletFIRFilterModule(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 singleDim: bool = True,
                 frequencies: torch.Tensor = defaultFrequencies,
                 waveletFunc: Callable = powerWavelet):
        '''Initializes module for CWT
        
        :param sampleFreq: sampling frequency of input signal
        :param singleDim: whether to collapse 3D into 1D or not
        :param frequencies: the ones to be utilized in the wavelet transform
        :param waveletFunc: the function to be utilized as wavelet
        '''
        
        super().__init__()
        dim = 1 if singleDim else 3
        
        widths = (sampleFreq / frequencies).to(torch.int) # in samples
        maxWidth = torch.max(widths).item() # in samples

        self.register_buffer('filterBank',
                             torch.stack([waveletFunc(maxWidth, w) for w in widths]).T)
            # shape=(maxWidth-1, numfilt)
        self.register_buffer('signalPrev', torch.zeros((dim, maxWidth - 1)))
            # shape=(dim, maxWidth-1)
    
    def forward(self,
                signalCurr: torch.Tensor): # shape=(dim,)
        
        # Prepare vector to be filtered
        signalMem = torch.cat([signalCurr.unsqueeze(1), self.signalPrev], dim=1)
            # shape=(dim, maxWidth)
        
        # Apply linear filtering using matrix multiplication
        signalFiltCurr = signalMem @ self.filterBank # shape=(dim, numfilt)

        # Update memory
        self.signalPrev = signalMem[:,:-1] # shape=(dim, maxWidth-1)
        
        return signalFiltCurr # shape=(dim, numfilt)


class WaveletIIRFilterModule(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 singleDim: bool = True,
                 frequencies: torch.Tensor = defaultFrequencies,
                 waveletFunc: Callable = powerWavelet):
        '''Initializes module for CWT implemented using IIR
        
        :param sampleFreq: sampling frequency of input signal
        :param singleDim: whether to collapse 3D into 1D or not
        :param frequencies: the ones to be utilized in the wavelet transform
        :param waveletFunc: the function to be utilized as wavelet
        '''
        
        super().__init__()
        dim = 1 if singleDim else 3
        numfilt = len(frequencies)
        
        widths = (sampleFreq / frequencies).to(torch.int) # in samples
        coefficients = torch.stack([torch.cat(prony(waveletFunc(w, w), 2, 2))
                                    for w in widths]).T # shape=(5, numfilt)
        
        self.register_buffer('filterBankB', coefficients[[0,1,2]])
            # shape=(3, numfilt)
        self.register_buffer('filterBankA', -coefficients[[4,5]].T[None,:])
            # shape=(1, numfilt, 2)
        self.register_buffer('signalPrev', torch.zeros((dim, 2)))
            # shape=(dim, 2)
        self.register_buffer('signalFiltPrev', torch.zeros((dim, numfilt, 2)))
            # shape=(dim, numfilt, 2)
    
    def forward(self,
                signalCurr: torch.Tensor): # shape=(dim,)
        
        # Prepare vector to be filtered
        signalMem = torch.cat([signalCurr.unsqueeze(1), self.signalPrev], dim=1)
            # shape=(dim, 3)
        
        # Apply linear filtering using matrix multiplication
        signalFiltCurr = (signalMem @ self.filterBankB +
                          (self.signalFiltPrev * self.filterBankA).sum(2)) # shape=(dim, numfilt)

        # Update memory
        self.signalPrev = signalMem[:,:2] # shape=(dim, 2)
        self.signalFiltPrev = torch.stack([signalFiltCurr, self.signalFiltPrev[:,:,0]], dim=2)
            # shape=(dim, numfilt, 2)
        
        return signalFiltCurr # shape=(dim, numfilt)


class MaxFilterModule(nn.Module):

    def __init__(self,
                 sampleFreq: float,
                 usePower: bool = True,
                 singleDim: bool = True,
                 frequencies: torch.Tensor = defaultFrequencies,
                 maxFiltWin: Union[float, float] = (0.3, 0.5)):
        '''Initializes max-filter module
        
        :param sampleFreq: sampling frequency of input signal
        :param usePower: whether to use power signal or acceleration signal
        :param singleDim: whether to collapse 3D into 1D or not
        :param frequencies: the ones to be utilized in the wavelet transform
        :param maxFiltWin: tuple containing window width in the max filter,
            first is is time (in seconds), and second is frequencies (in decades)
        '''
        
        super().__init__()
        self.usePower = usePower
        dim = 1 if singleDim else 3
        
        # Convert window to odd number of samples (time) and number of channels (frequencies)
        self.win = {'time': int(maxFiltWin[0] * sampleFreq),
                    'freq': int(maxFiltWin[1] / torch.diff(torch.log10(frequencies)).mean())}
        self.win = {'time': self.win['time'] + (self.win['time'] % 2 == 0),
                    'freq': self.win['freq'] + (self.win['freq'] % 2 == 0)} # make them odd
        
        # PyTorch module for max-filter
        self.maxPooler = nn.MaxPool2d(kernel_size=(self.win['freq'], self.win['time']),
                                      stride=(1,1), padding=(self.win['freq']//2, 0))
        
        self.register_buffer('signalPrev',
                             torch.zeros((dim, len(frequencies), self.win['time'] - 1)))
            # shape=(dim, numfilt, win['time']-1)
    
    def forward(self,
                signalCurr: torch.Tensor): # shape=(dim, numfilt)
        
        # Prepare vector to be filtered
        signalMem = torch.cat([self.signalPrev, signalCurr.unsqueeze(2)], dim=2)
            # shape=(dim, numfilt, win['time'])
        
        # Apply max (or max/min) filtering
        signalMaxCurr = self.maxPooler(signalMem).squeeze(2) # shape=(dim, numfilt)
        if self.usePower:
            eventCurr = signalMaxCurr * (signalMaxCurr == signalMem[:,:,self.win['time']//2])
                # shape=(dim, numfilt)
        else:
            signalMinCurr = -self.maxPooler(-signalMem).squeeze(2) # shape=(dim, numfilt)
            eventCurr = (signalMaxCurr * (signalMaxCurr == signalMem[:,:,self.win['time']//2]) +
                         signalMinCurr * (signalMinCurr == signalMem[:,:,self.win['time']//2]))
                # shape=(dim, numfilt)

        # Update memory
        self.signalPrev = signalMem[:,:,1:] # shape=(dim, numfilt, win['time']-1)

        return eventCurr # shape=(dim, numfilt)


class AbsModule(nn.Module):
    '''Module that takes absolute value'''
    def forward(self, x): return torch.abs(x)



if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import signal
    import seaborn as sn


    # Load data
    dataFrameLeft = pd.read_csv(f'data/Subject001_left_global_acc.csv')
    dataFrameLeft.rename(columns={'# time': 'time'}, inplace=True)

    # Select a time window
    tIni, tFin = 100, 120
    tMask = (dataFrameLeft['time'] >= tIni) * (dataFrameLeft['time'] < tFin)
    dataFrameLeft = dataFrameLeft[tMask]
    dataFrameLeft.reset_index(inplace=True, drop=True)

    # Get sample frequency
    sampleFreq = round(1 / np.diff(dataFrameLeft['time']).mean())

    # Filter signal
    b, a = signal.butter(2, [0.1 / (sampleFreq / 2), 5.0 / (sampleFreq / 2)], btype='bandpass')
    for d in 'xyz': dataFrameLeft[f'acc_{d}'] = signal.filtfilt(b, a, dataFrameLeft[f'acc_{d}'])
    
    # Set parameters for the IMU pipeline
    numFreq = 15
    frequencies = torch.logspace(math.log10(5e-1), math.log10(5e0), numFreq, base=10)
    numFilt = len(frequencies)
    singleDim = True
    dim = 1 if singleDim else 3
    rateFactor = 30.0

    # Initialize IMU pipeline
    nimup = NeuromorphicIMUPipeline(
        sampleFreq,
        singleDim=singleDim,
        frequencies=frequencies,
        rateFactor=rateFactor
    )
    
    # Forward pass data
    out = []
    for t, row in dataFrameLeft.iterrows():
        inp = row[[f'acc_{d}' for d in 'xyz']].to_numpy()
        out += [nimup(torch.Tensor(inp)).numpy()]

    # Store it in the dataframe
    outLabels = [f'events_{d}_{1/frequencies[j]:.2f}'
                 for j in range(numFilt) for d in ('xyz' if dim == 3 else '_')]
    dataFrameLeft[outLabels] = np.stack(out, 0).reshape(-1, dim * numFilt)

    # Plot spikes
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    sn.heatmap(dataFrameLeft[outLabels].T,
               xticklabels=len(dataFrameLeft)//10,
               yticklabels=True, cmap='crest', cbar=False)
    ax.set_xticklabels([f"{dataFrameLeft.loc[int(t.get_text()), 'time'].item():.2f}"
                        for t in ax.get_xticklabels()])

    plt.show()
