import torch
import torch.nn as nn

class CrossEntropySpkReg(nn.Module):
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, output, label, spikes=None):
        # output (N, T, C)
        # label (N,)
        # spikes ()
        N, T, C = output.shape
        
        label = torch.repeat_interleave(label, output.shape[1]) # (N * T,)
        output = output.view(-1,C) # (N * T, C)
        
        loss = nn.functional.cross_entropy(output, label)
        if self.alpha > 0: loss += self.alpha * spikes
        
        return loss

class TimeFirstWin__Old__(nn.Module):
    
    def __init__(self, beta=1.0, cutoff=10, alpha=1.0):
        super().__init__()
        self.beta = beta
        self.cutoff = cutoff
        self.alpha = alpha
        
    def forward(self, output, label, spikes):
        # output (N, T, 6)
        # label (N,)
        # spikes ()
        
        # Pad with ones in the end of temporal dimension (in case spikes is all zeros)
        output = torch.cat([output, torch.ones_like(output[:,:1])], dim=1)
        (N, T, C), device = output.shape, output.device

        # Calculate times of first spike using softmin (keep loss differentiable)
        times = torch.arange(T, device=device).view(1,T,1).expand(N,T,C)
        soft_time = (times * nn.functional.softmax(-10 * (output * times + (1 - output) * T), 1)).sum(1)

        # Get ground-truth class and other classes' first spike times
        gt_time = soft_time[torch.arange(N),label]
        mask = torch.ones_like(soft_time).bool()
        mask[torch.arange(N),label] = False
        other_times = soft_time.masked_select(mask).view(N,C-1)

        # Calculate loss
        loss = (gt_time + self.beta * nn.functional.relu(self.cutoff - other_times).sum(1)).mean() / T

        if self.alpha > 0: loss += self.alpha * spikes
        
        return loss



class TimeFirstWin(nn.Module):
    
    def __init__(self, beta=1.0, cutoff=None, alpha=1.0):
        super().__init__()
        self.beta = beta
        self.cutoff = cutoff
        self.alpha = alpha
        
    def forward(self, output, label, spikes):
        (N, T, C), device = output.shape, output.device

        times = torch.arange(T, device=device).view(1,T,1)
        spike_times = (output * times + (1 - output) * T)
        ttfs_all = spike_times.min(1).values

        ttfs_gt = ttfs_all.gather(1, label.view(-1, 1)).squeeze(1)
        
        loss = ttfs_gt.mean() / T

        if self.beta > 0:
            mask = torch.ones_like(ttfs_all).scatter(1, label.view(-1, 1), 0)
            ttfs_others = torch.where(mask.bool(), ttfs_all, float('inf'))
            
            loss += self.beta * nn.functional.relu(self.cutoff - ttfs_others).sum(1).mean() / T
        
        if self.alpha > 0: loss += self.alpha * spikes

        return loss.mean()