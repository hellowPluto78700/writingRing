import os
import sys
import glob
import re
import warnings

import numpy as np
from scipy.interpolate import interp1d
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trf

from sklearn.preprocessing import LabelEncoder

####################################################################################################
# Dataloaders
def createLoaders(
    dataPath,
    datasetName='Wisdm',
    labelType='Willetts2018',
    subjectSplits=(80,10,10),
    batchSize=256,
    randomSeed=12345,
    balance=True,
    sampleFreq=64,
    **kwargs
):
    # Create transforms for the dataset
    trfm = []
    
    rs = kwargs['rectifySpikes'] if 'rectifySpikes' in kwargs else False
    pb = kwargs['polarityBichannel'] if 'polarityBichannel' in kwargs else False
    tc = kwargs['timeCompress'] if 'timeCompress' in kwargs else 1.0
    tr = kwargs['timeResolution'] if 'timeResolution' in kwargs else 1 / sampleFreq
    st = kwargs['systemType'] if 'systemType' in kwargs else 'trimmed_intermittent'
        # padded or trimmed intermittent / downsampled
    qs = kwargs['quantizeSpikes'] if 'quantizeSpikes' in kwargs else False
    
    ag = kwargs['addGravity'] if 'addGravity' in kwargs else False
    cc = kwargs['compressChannels'] if 'compressChannels' in kwargs else False
    
    if rs: trfm.append(rectifySpikes())
        # (T, C) --> (T, C)
    elif pb: trfm.append(polarityBichannel())
        # (T, C) --> (T, 2C)
    if 'intermittent' in st:
        if tc * tr * sampleFreq != 1: trfm.append(downsampleSpikes(factor=tc*tr*sampleFreq))
            # (T, C) --> (T / (tc * tr * sampleFreq), C)
        if 'padded' in st and tc != 1: trfm.append(padSpikes(Tpad=int(10/tr*(1-1/tc))))
            # (T / (tc * tr * sampleFreq), C) --> (T / (tr * sampleFreq), C)
    else:
        if tc != 1: trfm.append(downsampleSpikes(factor=tc))
            # (T, C) --> (T / tc, C)
        if tr * sampleFreq != tc: trfm.append(downsampleSpikes(factor=tr*sampleFreq/tc))
            # (T / tc, C) --> (T / (tr * sampleFreq), C)
    if qs: trfm.append(quantizeSpikes(scale=1.0, numBits=5, qtype='1/3'))
        # (T, C) --> (T, C)
            
    trfm = trf.Compose(trfm)
    
    # Create datasets and balance among classes
    dataset_class = {'Capture24':CAPTURE24, 'ADL':ADL, 'MHealth':MHEALTH,
                     'Opportunity':OPPORTUNITY, 'PAMAP':PAMAP, 'Realworld':REALWORLD,
                     'UMAHand':UMAHAND, 'Wisdm':WISDM}[datasetName]

    dataset_train = dataset_class(dataPath, addGravity=ag, compressChannels=cc, split='train',
                                  subjectSplits=subjectSplits, transform=trfm,
                                  labelType=labelType, seed=randomSeed)
    if balance: dataset_train.balance()
    dataset_val = dataset_class(dataPath, addGravity=ag, compressChannels=cc, split='val',
                                subjectSplits=subjectSplits, transform=trfm,
                                labelType=labelType, seed=randomSeed)
    if balance: dataset_val.balance()
    dataset_test = dataset_class(dataPath, addGravity=ag, compressChannels=cc, split='test',
                                 subjectSplits=subjectSplits, transform=trfm,
                                 labelType=labelType, seed=randomSeed)
    # if balance: dataset_test.balance()
    
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batchSize, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batchSize, shuffle=False)

    return train_loader, val_loader, test_loader

####################################################################################################
# HAR Datasets
class BaseHARDataset(Dataset):
    def __init__(
        self,
        data_path,
        label_names,
        isCapture=False,
        addGravity=False,
        compressChannels=False,
        split='train',
        transform=None,
        labelType='Willetts2018',
        subjectSplits=(80,10,10),
        seed=12345
    ):
        self.add_gravity = addGravity
        self.compress_channels = compressChannels
        
        self._label_names = label_names
        self._label_numbers = list(range(len(self._label_names)))
        names2numbers = dict([p for p in zip(self._label_names, self._label_numbers)])
        numbers2names = dict([p for p in zip(self._label_numbers, self._label_names)])
        
        rng = np.random.default_rng(seed)
        
        # Find files and sort by subject ID
        spikes_files = sorted(glob.glob(f'{data_path}/P*_spikes.npy'),
                              key=lambda fn: int(os.path.basename(fn).split('_')[0][1:]))
        # labels_files = sorted(glob.glob(f'{data_path}/P*_labels{labelType if isCapture else ""}.npy'),
        #                       key=lambda fn: int(os.path.basename(fn).split('_')[0][1:]))
        labels_files = sorted(glob.glob(f'{data_path}/P*_labels.npy'),
                              key=lambda fn: int(os.path.basename(fn).split('_')[0][1:]))
        
        # Shuffle subjects once and split
        S = len(spikes_files)
        order = rng.permutation(S)
        splits = np.asarray(subjectSplits, dtype=float)
        splits /= splits.sum()  # normalize
        
        sizes = np.maximum(1, np.round(S * splits).astype(int))
        while sizes.sum() > S: sizes[0] -= 1 # take from training split
        while sizes.sum() < S: sizes[0] += 1 # add to training split
        chunks = np.split(order, np.cumsum(sizes)[:-1])
        print(f'Subject splits: {chunks}')

        if split == 'train':
            spikes_files = [spikes_files[i] for i in chunks[0]]
            labels_files = [labels_files[i] for i in chunks[0]]
        elif split == 'val':
            spikes_files = [spikes_files[i] for i in chunks[1]]
            labels_files = [labels_files[i] for i in chunks[1]]
        elif split == 'test':
            spikes_files = [spikes_files[i] for i in chunks[-1]]
            labels_files = [labels_files[i] for i in chunks[-1]]
        else:
            raise ValueError('Must choose one of train/val/test for the dataset')
        
        self.spikes_files = spikes_files
        self.transform = transform
        self.seed = seed
        
        # Figure out how many samples per file, build a global index
        lengths = []
        for lf in labels_files:
            lab = np.load(lf, allow_pickle=True)
            lengths.append(lab.shape[0])
        self.cum_lengths = np.concatenate([[0], np.cumsum(lengths)])
        self.total_len = int(self.cum_lengths[-1])
        
        # Load and encode all labels into one int array
        all_targets = []
        for lf in labels_files:
            lab = np.load(lf, allow_pickle=True)
            all_targets.extend(names2numbers[s] for s in lab)
        self.targets = np.array(all_targets, dtype=np.int64)  # size = total_len
        
        # Initial index list is the full range
        self.indices = np.arange(self.total_len, dtype=np.int64)
        
        # Dictionary for numbers and names
        self.targ_dict = numbers2names
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # which global sample?
        gidx = int(self.indices[idx])
        
        # locate which file holds that index
        file_idx = np.searchsorted(self.cum_lengths, gidx, side='right') - 1
        local_i  = gidx - self.cum_lengths[file_idx]
        
        # lazy‐load the correct memmap and slice out one sample
        spikes_mmap = np.load(self.spikes_files[file_idx], mmap_mode='r')
        x = np.array(spikes_mmap[local_i])       # shape (640,15)

        # if gravity included
        if self.add_gravity:
            grav_mmap = np.load(re.sub(r'([^/]+)/data/.*?/windows', r'\1/data/\1_wg/windows',
                                       self.spikes_files[file_idx]), mmap_mode='r')
            gr = np.array(grav_mmap[local_i])       # shape (640,3)
            gr = 1 + np.stack([np.convolve(gr_i, np.ones((129,))/129., mode='same') for gr_i in gr.T], 1)
            x[:,[0,5,10]] = gr
        
        if self.transform:
            x = self.transform(x).astype(np.float32)
        
        y = int(self.targets[gidx])

        return x, y
    
    
    def balance(self):
        """Downsample each class to the size of the smallest."""
        rng = np.random.default_rng(self.seed)
        
        # count how many we have _before_ balancing
        counts = np.bincount(self.targets[self.indices], minlength=len(self._label_numbers))
        min_c  = counts.min()
        if min_c == 0:
            raise RuntimeError(f"Some class missing: {counts}")
        
        # for each class, pick min_c global indices
        new_idxs = []
        for t in self._label_numbers:
            # mask = which entries of self.indices correspond to class t
            mask = np.where(self.targets[self.indices] == t)[0]
            pick = rng.choice(mask, size=min_c, replace=False)
            new_idxs.append(self.indices[pick])
        
        self.indices = np.concatenate(new_idxs)
        rng.shuffle(self.indices)

# 5,1,1
class ADL(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            ['climb_stairs', 'drink_glass', 'getup_bed', 'pour_water', 'walk'],
            **kwargs
        )

# 100,20,20
class CAPTURE24(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            ['bicycling', 'mixed', 'sit-stand', 'sleep', 'vehicle', 'walking'],
            isCapture=True,
            **kwargs
        )

class OPPORTUNITY(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            [1, 2, 3, 4],
            # ['lying down', 'standing', 'sitting', 'walking'],
            **kwargs
        )


class PAMAP(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            [1, 2, 3, 4, 12, 13, 16, 17],
            # ['lying down', 'sitting', 'standing', 'ironing', 'vacuum cleaning',
            #  'walking', 'ascending stairs', 'descending stairs'],
            **kwargs
        )


class WISDM(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            ['catch', 'chips', 'clapping', 'dribbling', 'drinking', 'folding',
             'jogging', 'kicking', 'pasta', 'sandwich', 'sitting', 'soup',
             'stairs', 'standing', 'teeth', 'typing', 'walking', 'writing'],
            **kwargs
        )


class REALWORLD(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            ['climbingdown', 'climbingup', 'jumping', 'lying',
             'running', 'sitting', 'standing', 'walking'],
            **kwargs
        )


class MHEALTH(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            # ['null', 'standing still', 'sitting and relaxing', 'lying down', 'walking',
            #  'climbing stairs', 'waist bends forward', 'frontal elevation of arms',
            #  'knees bending (crouching)', 'cycling', 'jogging', 'running', 'jump front & back'],
            **kwargs
        )


class UMAHAND(BaseHARDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            # ['Brushing teeth with a manual toothbrush (during a complete brushing)',
            #  'Brushing teeth with an electric toothbrush (during a complete brushing)',
            #  'Washing hands', 'Eat soup (or any other liquid) with a spoon from a bowl',
            #  'Cutting food (e.g., a slice of bread) with a knife and fork',
            #  'Peeling a fruit with a knife (an apple or a kiwi fruit)', 'Aplauding', 'combing hair',
            #  'Cleaning (Wiping with a cloth with some energy on a smooth, flat surface)',
            #  'Sweep with a broom', 'Write on a sheet of paper a sentence with a pen',
            #  'Writing a sentence with a keyboard on the computer',
            #  'Send a message through the whatsap application on a cell phone',
            #  'Fold a piece of paper several times',
            #  'Mark a phone number on a cell phone and hold the cell phone to your ear',
            #  'Picking up an object from the floor starting from the upright position and bringing it to chest height',
            #  'Opening a bottle with thread (already opened beforehand so that the thread does not offer too much resistance)',
            #  'Drinking water from a glass', 'Pouring water into a glass with a pitcher',
            #  'Putting on a pair of glasses', 'Putting on a jacket/sweatshirt',
            #  'Remove a jacket/sweatshirt', 'Putting on a shoe and tying the laces',
            #  'Waving goodbye (in the way they would normally do it, i.e., without instructing the volunteer)',
            #  'Nose blowing', 'Opening and closing a door (by moving a doorknob)',
            #  'Buttoning a shirt button', 'Raising and lowering a zipper', 'Screwing a screw'],
            **kwargs
        )


####################################################################################################
# Transforms

class quantizeSpikes(object):
    def __init__(self, scale=1.0, numBits=16, qtype='lin'):
        self.scale = scale
        self.bitFactor = 2 ** (numBits - 1)
        self.qtype = qtype

    def __call__(self, spikes):
        # Assume input is shape (T, C)
        if self.qtype == 'lin': spikes = self.scale * spikes
        elif self.qtype == 'sqrt': spikes = self.bitFactor * np.sqrt(self.scale * spikes / self.bitFactor)
        elif self.qtype == 'log': spikes = self.bitFactor * np.log(1 + self.scale * spikes) / np.log(1 + self.bitFactor)
        elif len(self.qtype.split('/')) == 2: spikes = self.bitFactor * (self.scale * spikes / self.bitFactor) ** eval(self.qtype)
        else: raise NotImplementedError(f'Quantization method {self.qtype} not implemented')
        spikes = np.ceil(spikes)
        spikes = np.clip(spikes, -self.bitFactor, self.bitFactor - 1)

        return spikes # shape (T, C)

class downsampleSpikes(object):
    def __init__(self, factor=2):
        self.factor = factor

    def __call__(self, spikes):
        # Assume input is shape (T, C)
        T, C = spikes.shape
        Tnew = int(T / self.factor)
        
        spikes_cum = np.cumsum(spikes, axis=0)
        spikes_cum_f = interp1d(np.arange(T), spikes_cum,
                                kind='nearest', axis=0, fill_value='extrapolate')
        spikes_cum_down = spikes_cum_f(np.linspace(0, T, Tnew, endpoint=False))
        spikes_down = np.diff(spikes_cum_down, axis=0, prepend=0)

        return spikes_down # shape (Tnew, C)


class padSpikes(object):
    def __init__(self, Tpad=0):
        self.Tpad = Tpad

    def __call__(self, spikes):
        # Assume input is shape (T, C)
        spikes_padded = np.pad(spikes, ((self.Tpad, 0), (0, 0)))

        return spikes_padded # shape (T + Tpad, C)


class polarityBichannel(object):
    def __call__(self, spikes):
        # Assume input is shape (T, C)
        spikes_pos, spikes_neg = np.clip(spikes, a_min=0, a_max=None), np.clip(spikes, a_min=None, a_max=0)
        spikes = np.concatenate([spikes_pos, -spikes_neg], 1)

        return spikes # shape (T, 2C)


class rectifySpikes(object):
    def __call__(self, spikes):
        # Assume input is shape (T, C)
        return np.abs(spikes) # shape (T, C)