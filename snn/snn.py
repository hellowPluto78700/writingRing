import torch
import torch.nn as nn
import snntorch as snn

from architectures import SynNet

# Define SNN
class SNNIMU(nn.Module):
    def __init__(self, numInput=1*15, numHidNeur=100, numOutput=2, beta=0.5):
        super().__init__()

        self.lin1 = nn.Linear(numInput, numHidNeur)
        self.rlif = snn.RLeaky(beta=beta, linear_features=numHidNeur)
        self.lin2 = nn.Linear(numHidNeur, numOutput)
        self.lif = snn.Leaky(beta=beta)

        self.reset_mem()

    def reset_mem(self):
        spk1, mem1 = self.rlif.reset_mem()
        mem2 = self.lif.reset_mem()

        self.register_buffer('spk1', spk1)
        self.register_buffer('mem1', mem1)
        self.register_buffer('mem2', mem2)

    def forward(self, x):
        
        x = x.flatten(1)

        cur1 = self.lin1(x)
        self.spk1, self.mem1 = self.rlif(cur1, self.spk1, self.mem1)
        cur2 = self.lin2(self.spk1)
        self.spk2, self.mem2 = self.lif(cur2, self.mem2)
        
        return self.spk2, self.mem2


if __name__ == '__main__':

    # Set device
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    # Set parameters of the network
    numHidNeur = 100
    beta = 0.5 # decay parameter for snnTorch
    dim = 1
    numFreq = 15

    snnIMU = SNNIMU(
        numInput=dim*numFreq,
        numHidNeur=numHidNeur,
        numOutput=2,
        beta=beta,
    ).to(device)

    inp = 10 * torch.rand(1, dim, numFreq).to(device)
    snnIMU(inp)
    print(inp)
    print(snnIMU.spk1)
    print(snnIMU.mem1)
    print(snnIMU.spk2)
    print(snnIMU.mem2)

