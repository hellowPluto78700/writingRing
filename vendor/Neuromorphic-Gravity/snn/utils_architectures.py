import torch
import torch.nn as nn
import snntorch as snn

import numpy as np
import math

# import sys
# sys.path.append('/home/igavier_umass_edu/Documents/Neuromorphic-IMU/snn/')
from rockpool_nn_networks_synnet import SynNet as SNRP
from rockpool_nn_networks_synnet import bitshift_to_tau

####################################################################################################
# Get model
def createModel(
    network_type,
    inputSize,
    outputSize,
    hiddenSizes,
    device,
    sampleFreq=64,
    **kwargs
):
    try: model_class = {'LSTMNet':LSTMNet, 'LSTMNetANN':LSTMNetANN, 'RSynNet':RSynNet,
                        'SynNet':SynNet, 'SynNetRP':SynNetRockPool, 'ANNMLP':ANNMLP, 'SNNMLP': SNNMLP,
                        'CNNNetANN':CNNNetANN, 'CNNNetSNN':CNNNetSNN}[network_type]
    except KeyError: raise NotImplementedError(f'Network type {network_type} not supported')

    model = model_class(
        inputSize=inputSize,
        outputSize=outputSize,
        hiddenSizes=hiddenSizes,
        sampleFreq=sampleFreq,
        **kwargs
    ).to(device)
    
    return model

# Define SynNet Network
class SynNet(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], sampleFreq=64,
                 shiftSyn=1, shiftMem=1, tausFactor=1, **kwargs):
        super().__init__()

        shiftsSyn = list(range(shiftSyn, shiftSyn + 8))
            # number of bits to shift in I[t+Dt] = I[t] - (I[t] >> a)
        shiftMem = shiftMem
            # number of bits to shift in U[t+Dt] = U[t] - (U[t] >> b)

        alphas = [[(1 - 2 ** (-shiftSyn)) for shiftSyn in shiftsSyn[:2**(l+1)]
                   for _ in range(hiddenSizes[l] // 2**(l+1))] for l in range(len(hiddenSizes))]
        beta = (1 - 2 ** (-shiftMem))

        tausSyn = [-(1 / sampleFreq) / tausFactor / math.log(1 - 2 ** (-shiftSyn)) for shiftSyn in shiftsSyn]
        tauMem = -(1 / sampleFreq) / tausFactor / math.log(1 - 2 ** (-shiftMem))

        # Initialize layers
        self.fc1 = nn.Linear(inputSize, hiddenSizes[0])
        self.lif1 = snn.Synaptic(alpha=alphas[0], beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.lif2 = snn.Synaptic(alpha=alphas[1], beta=beta, learn_threshold=True)
        self.fc3 = nn.Linear(hiddenSizes[1], hiddenSizes[2])
        self.lif3 = snn.Synaptic(alpha=alphas[2], beta=beta, learn_threshold=True)
        self.fc4 = nn.Linear(hiddenSizes[2], outputSize)
        self.lif4 = snn.Synaptic(alpha=beta, beta=beta, learn_threshold=True)
        
        # Total spike record
        self.spkTotal = None

    def reset_state(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif4.reset_hidden()

    def forward(self, x):

        # Initialize hidden states at t=0
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []

        for step in range(x.shape[1]):
            cur1 = self.fc1(x[:,step])
            spk1, _, _ = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _, _ = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, _, _ = self.lif3(cur3)
            cur4 = self.fc4(spk3)
            spk4, _, _ = self.lif4(cur4)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
        
        self.spkTotal = (torch.stack(spk1_rec, dim=1).sum() +
                         torch.stack(spk2_rec, dim=1).sum() +
                         torch.stack(spk3_rec, dim=1).sum() +
                         torch.stack(spk4_rec, dim=1).sum())
        
        return torch.stack(spk4_rec, dim=1)


# Define SynNet Network using RockPool
class SynNetRockPool(SNRP):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], sampleFreq=64,
                 shiftSyn=1, shiftMem=1, tausFactor=1, timeResolution=1/64, **kwargs):
        super().__init__(
            n_channels=inputSize,
            n_classes=outputSize,
            size_hidden_layers=hiddenSizes,
            time_constants_per_layer=[2,2,2],
            tau_syn_base=bitshift_to_tau(timeResolution, torch.tensor(shiftSyn))[0],
            tau_mem=bitshift_to_tau(timeResolution, torch.tensor(shiftMem))[0],
            tau_syn_out=bitshift_to_tau(timeResolution, torch.tensor(shiftMem))[0],
            quantize_time_constants=True,
            train_threshold=True,
            max_spikes_per_dt=31,
            max_spikes_per_dt_out=1,
            dt=timeResolution,
            **kwargs
        )
    def __call__(self, x, record=False):
        return super().__call__(x, record=record)

# Define ANN Network using RockPool
class ANNMLP(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], **kwargs):
        super().__init__()
        
        self.fc1 = nn.Linear(inputSize, hiddenSizes[0])
        self.fc2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.fc3 = nn.Linear(hiddenSizes[1], hiddenSizes[2])
        self.fc4 = nn.Linear(hiddenSizes[2], outputSize)
        self.relu = nn.ReLU()
        self.spkTotal = torch.tensor(sum(hiddenSizes) + outputSize)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out.unsqueeze(1)

# Define SNN Network using RockPool
class SNNMLP(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], alpha=0.9, beta=0.8, num_steps=25, **kwargs):
        super().__init__()
        
        self.fc1 = nn.Linear(inputSize, hiddenSizes[0])
        self.fc2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.fc3 = nn.Linear(hiddenSizes[1], hiddenSizes[2])
        self.fc4 = nn.Linear(hiddenSizes[2], outputSize)

        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)

        self.num_steps = num_steps
        
        self.spkTotal = None

    def forward(self, x):

        # Initialize hidden states at t=0
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(torch.flatten(x, 1))
            spk1, _, _ = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _, _ = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, _, _ = self.lif3(cur3)
            cur4 = self.fc4(spk3)
            spk4, _, _ = self.lif4(cur4)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
        
        self.spkTotal = (torch.stack(spk1_rec, dim=1).sum() +
                         torch.stack(spk2_rec, dim=1).sum() +
                         torch.stack(spk3_rec, dim=1).sum() +
                         torch.stack(spk4_rec, dim=1).sum())
        
        return torch.stack(spk4_rec, dim=1)


# Define SynNet Network
class RSynNet(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], sampleFreq=64, **kwargs):
        super().__init__()
        
        tausSyn = [2 ** n * 5e-2 for n in range(1, 9)] # in seconds
        tauMem = 1e-2
        alphas = [[math.exp(-(1 / sampleFreq) / tauSyn) for tauSyn in tausSyn[:2**(l+1)]
                   for _ in range(hiddenSizes[l] // 2**(l+1))] for l in range(len(hiddenSizes))]
        beta = math.exp(-(1 / sampleFreq) / tauMem)

        # Initialize layers
        self.fc1 = nn.Linear(inputSize, hiddenSizes[0])
        self.lif1 = snn.RSynaptic(alpha=alphas[0], beta=beta, linear_features=hiddenSizes[0], learn_threshold=True)
        self.fc2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.lif2 = snn.RSynaptic(alpha=alphas[1], beta=beta, linear_features=hiddenSizes[1], learn_threshold=True)
        self.fc3 = nn.Linear(hiddenSizes[1], hiddenSizes[2])
        self.lif3 = snn.RSynaptic(alpha=alphas[2], beta=beta, linear_features=hiddenSizes[2], learn_threshold=True)
        self.fc4 = nn.Linear(hiddenSizes[2], outputSize)
        self.lif4 = snn.Synaptic(alpha=beta, beta=beta, learn_threshold=True)
        
        # Total spike record
        self.spkTotal = None

    def reset_state(self):
        self.lif1.reset_hidden()
        self.lif2.reset_hidden()
        self.lif3.reset_hidden()
        self.lif4.reset_hidden()

    def forward(self, x):

        # Initialize hidden states at t=0
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []

        for step in range(x.shape[1]):
            cur1 = self.fc1(x[:,step])
            spk1, _, _ = self.lif1(cur1)
            cur2 = self.fc2(spk1)
            spk2, _, _ = self.lif2(cur2)
            cur3 = self.fc3(spk2)
            spk3, _, _ = self.lif3(cur3)
            cur4 = self.fc4(spk3)
            spk4, _, _ = self.lif4(cur4)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
        
        self.spkTotal = (torch.stack(spk1_rec, dim=1).sum() +
                         torch.stack(spk2_rec, dim=1).sum() +
                         torch.stack(spk3_rec, dim=1).sum() +
                         torch.stack(spk4_rec, dim=1).sum())
        
        return torch.stack(spk4_rec, dim=1)



# Define LSTM Network
class LSTMNet(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], sampleFreq=64, **kwargs):
        super().__init__()
        
        # Initialize layers
        self.lstm1 = snn.SLSTM(inputSize, hiddenSizes[0], learn_threshold=True)
        self.lstm2 = snn.SLSTM(hiddenSizes[0], hiddenSizes[1], learn_threshold=True)
        self.lstm3 = snn.SLSTM(hiddenSizes[1], hiddenSizes[2], learn_threshold=True)
        self.lstm4 = snn.SLSTM(hiddenSizes[2], outputSize, learn_threshold=True)

        self.reset = kwargs.get('reset', False)
        
        # Total spike record
        self.spkTotal = None

    def reset_state(self):
        self.lstm1.reset_hidden()
        self.lstm2.reset_hidden()
        self.lstm3.reset_hidden()
        self.lstm4.reset_hidden()

    def forward(self, x):

        # Initialize hidden states at t=0
        self.lstm1.reset_mem()
        self.lstm2.reset_mem()
        self.lstm3.reset_mem()
        self.lstm4.reset_mem()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []

        for step in range(x.shape[1]):
            spk1, _, _ = self.lstm1(x[:,step])
            spk2, _, _ = self.lstm2(spk1)
            spk3, _, _ = self.lstm3(spk2)
            spk4, _, _ = self.lstm4(spk3)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4)
        
        self.spkTotal = (torch.stack(spk1_rec, dim=1).sum() +
                         torch.stack(spk2_rec, dim=1).sum() +
                         torch.stack(spk3_rec, dim=1).sum() +
                         torch.stack(spk4_rec, dim=1).sum())
        
        return torch.stack(spk4_rec, dim=1)


# Define LSTM Network
class LSTMNetANN(nn.Module):
    def __init__(self, inputSize, outputSize=1, hiddenSizes=[24,24,24], sampleFreq=64, **kwargs):
        super().__init__()
        
        # Save size parameters
        self.hiddenSizes = hiddenSizes
        self.outputSize = outputSize
        
        # Initialize layers
        self.lstm1 = nn.LSTMCell(inputSize, hiddenSizes[0])
        self.lstm2 = nn.LSTMCell(hiddenSizes[0], hiddenSizes[1])
        self.lstm3 = nn.LSTMCell(hiddenSizes[1], hiddenSizes[2])
        self.lstm4 = nn.LSTMCell(hiddenSizes[2], outputSize)
        
        self.h1c1 = None
        self.h2c2 = None
        self.h3c3 = None
        self.h4c4 = None

        self.reset = kwargs.get('reset', False)
        
        # Total spike record
        self.spkTotal = None
    
    def init_hidden(self, batch_size, lstm):
        hidden_size = len(lstm.bias_ih) // 4
        hidden_state = torch.zeros(batch_size, hidden_size).to(lstm.bias_ih.device)
        cell_state = torch.zeros(batch_size, hidden_size).to(lstm.bias_ih.device)
        return hidden_state, cell_state

    def forward(self, x):

        # Initialize hidden states at t=0
        if self.reset:
            self.h1c1 = self.init_hidden(x.shape[0], self.lstm1)
            self.h2c2 = self.init_hidden(x.shape[0], self.lstm2)
            self.h3c3 = self.init_hidden(x.shape[0], self.lstm3)
            self.h4c4 = self.init_hidden(x.shape[0], self.lstm4)
        elif self.h1c1 != None:
            self.h1c1 = tuple(hid1[:1].expand(x.shape[0],-1).detach() for hid1 in self.h1c1)
            self.h2c2 = tuple(hid2[:1].expand(x.shape[0],-1).detach() for hid2 in self.h2c2)
            self.h3c3 = tuple(hid3[:1].expand(x.shape[0],-1).detach() for hid3 in self.h3c3)
            self.h4c4 = tuple(hid4[:1].expand(x.shape[0],-1).detach() for hid4 in self.h4c4)
            
        # Record the final layer
        h1_rec = []
        h2_rec = []
        h3_rec = []
        h4_rec = []

        for step in range(x.shape[1]):
            self.h1c1 = self.lstm1(x[:,step], self.h1c1)
            self.h2c2 = self.lstm2(self.h1c1[0], self.h2c2)
            self.h3c3 = self.lstm3(self.h2c2[0], self.h3c3)
            self.h4c4 = self.lstm4(self.h3c3[0], self.h4c4)
            
            h1_rec.append(self.h1c1[0])
            h2_rec.append(self.h2c2[0])
            h3_rec.append(self.h3c3[0])
            h4_rec.append(self.h4c4[0])
        
        self.spkTotal = (torch.stack(h1_rec, dim=1).sum() +
                         torch.stack(h2_rec, dim=1).sum() +
                         torch.stack(h3_rec, dim=1).sum() +
                         torch.stack(h4_rec, dim=1).sum())
        
        return torch.stack(h4_rec, dim=1)

class CNNNetANN(nn.Module):
    """
    Input:  x of shape (N, T, inputSize)
    Output: logits of shape (N, 1, outputSize)
            (you can squeeze the middle dim for CE loss)
    """
    def __init__(
        self,
        inputSize: int,
        outputSize: int,
        channels=(64, 128, 128),
        kernel_sizes=(7, 5, 3),
        dropout=0.1,
        causal: bool = False,  # set True if you need strictly causal convs
        **kwargs
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        layers = []
        in_ch = inputSize
        for out_ch, k in zip(channels, kernel_sizes):
            # padding to keep L the same (or causal if requested)
            if causal:
                padding = (k - 1)  # left-pad only; we’ll handle via padding module
                conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=0)
                block = nn.Sequential(
                    nn.ConstantPad1d((padding, 0), 0.0),
                    conv,
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            else:
                padding = k // 2  # "same" padding
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            layers.append(block)
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        # Collapse time dimension to length 1
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # 1x1 conv as a linear head over channels -> classes
        self.head = nn.Conv1d(in_ch, outputSize, kernel_size=1)

        # optional: initialize head a bit smaller
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5)) if hasattr(nn.init, "kaiming_uniform_") else None
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        self.spkTotal = torch.tensor(0)

    def forward(self, x):
        # x: (N, T, inputSize) -> Conv1d expects (N, C, L) = (N, inputSize, T)
        x = x.permute(0, 2, 1)

        feats = self.backbone(x)           # (N, C_last, T)
        pooled = self.pool(feats)          # (N, C_last, 1)
        logits = self.head(pooled)         # (N, outputSize, 1)

        # Return as (N, 1, outputSize) per your spec
        return logits.permute(0, 2, 1)


class CNNNetSNN(nn.Module):
    """
    Spiking version of CNNNetANN using snn.Synaptic neurons.

    Input:  x of shape (N, T, inputSize)
    Output: spk_out_avg of shape (N, 1, outputSize)
            (you can squeeze the middle dim for CE loss if desired)

    Notes:
    - Uses rate-coding over `num_steps`: the same analog input is
      presented at each step, spikes are accumulated/averaged.
    - Conv structure mirrors your ANN.
    """

    def __init__(
        self,
        inputSize: int,
        outputSize: int,
        channels=(80, 80, 80),
        kernel_sizes=(5, 5, 5),
        dropout=0.1,
        causal: bool = False,
        num_steps: int = 25,
        alpha: float = 0.9,  # synaptic current decay
        beta: float = 0.8,   # membrane potential decay
        **kwargs,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)

        self.num_steps = num_steps
        self.spkTotal = None

        # --- Backbone convs (no ReLU; BN kept, dropout optional) ---
        conv_layers = []
        lif_layers = []

        in_ch = inputSize
        for out_ch, k in zip(channels, kernel_sizes):
            if causal:
                padding = (k - 1)
                conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=0)
                block = nn.Sequential(
                    nn.ConstantPad1d((padding, 0), 0.0),
                    conv,
                    nn.BatchNorm1d(out_ch),
                    nn.Dropout(p=dropout),
                )
            else:
                padding = k // 2
                block = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    nn.Dropout(p=dropout),
                )

            conv_layers.append(block)
            in_ch = out_ch

        
        self.backbone = nn.ModuleList(conv_layers)

        # --- Pool + head (same as ANN, but with spiking output) ---
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Conv1d(in_ch, outputSize, kernel_size=1)

        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif3 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)
        self.lif4 = snn.Synaptic(alpha=alpha, beta=beta, learn_alpha=True, learn_beta=True, learn_threshold=True)

        # Optional: init head like in your ANN
        if hasattr(nn.init, "kaiming_uniform_"):
            nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)


    def forward(self, x):

        # Initialize hidden states at t=0
        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()
        self.lif4.reset_mem()
        
        # Record the final layer
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []

        for step in range(self.num_steps):
            cur1 = self.backbone[0](x.permute(0, 2, 1))
            spk1, _, _ = self.lif1(cur1)
            cur2 = self.backbone[1](spk1)
            spk2, _, _ = self.lif2(cur2)
            cur3 = self.backbone[2](spk2)
            spk3, _, _ = self.lif3(cur3)
            spk3_pooled = self.pool(spk3)
            cur4 = self.head(spk3_pooled)
            spk4, _, _ = self.lif4(cur4)
            
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
            spk3_rec.append(spk3)
            spk4_rec.append(spk4.squeeze(2))
        
        self.spkTotal = (torch.stack(spk1_rec, dim=1).sum() +
                         torch.stack(spk2_rec, dim=1).sum() +
                         torch.stack(spk3_rec, dim=1).sum() +
                         torch.stack(spk4_rec, dim=1).sum())
        
        return torch.stack(spk4_rec, dim=1)