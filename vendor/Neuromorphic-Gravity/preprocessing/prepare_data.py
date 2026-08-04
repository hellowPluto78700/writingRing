import numpy as np
import pandas as pd

import argparse
import tqdm

def prepareWindows(dataPath, annotPath, sampleFreq=64, winSize=10.0, subjID=1, useXylo=False):

    # Load dataframe
    data = pd.read_csv(f'{dataPath}/P{subjID:03d}.csv.gz')
            
    data['datetime'] = pd.to_datetime(data.time, unit='s')

    S, y = [], []
    
    for t, d in tqdm.tqdm(data.resample(f'{winSize}s', origin='start', on='datetime')):
        if len(d) != int(sampleFreq * winSize): continue
        if d['annotation'].isna().any(): continue
    
        S += [d[[c for c in d.columns if ('spikes' if useXylo else 'events') in c]].to_numpy()]
        y += [d['annotation'].mode()[0]]
    
    S = np.stack(S)
    y = np.stack(y)
    
    np.save(f'{dataPath}/windows/P{subjID:03d}_spikes.npy', S)
    np.save(f'{dataPath}/windows/P{subjID:03d}_labels.npy', y)

    return windowsPath