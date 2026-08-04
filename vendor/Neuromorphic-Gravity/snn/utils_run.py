from tqdm import tqdm
import torch
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, cohen_kappa_score

# import sys
# sys.path.append('/home/igavier_umass_edu/Documents/Neuromorphic-IMU/snn/')
from utils_architectures import *

def run_epoch(model, dataloader, criterion, optimizer=None, split='train', device=None):
    is_train = split == 'train'
    is_val = split == 'val'
    is_test = split == 'test'
    assert is_train or is_val or is_test, 'Must choose one of train/val/test for the epoch'
    
    if is_train: model.train()
    else: model.eval()
    
    samples_count = 0
    running_loss = 0.0
    spike_count = 0.0
    y_true = []
    y_score = []
    y_pred = []
    
    with torch.set_grad_enabled(is_train):
        for (inputs, labels) in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device, dtype=torch.float) # (N, T, 15)
            labels = labels.to(device, dtype=torch.long) # (N, C)

            if is_train: optimizer.zero_grad()

            # model.reset_state()
            if isinstance(model, SNRP):
                output, _, rec = model(inputs, record=True) # (N, T, C)
                model.spkTotal = sum([(model._record_dict[lyr]['spikes'] > 0).sum()
                                      for lyr in model.lif_names])
            else:
                output = model(inputs) # (N, T, C)
            
            loss = criterion(output, labels, model.spkTotal)

            if is_train: loss.backward()
            if is_train: optimizer.step()

            if isinstance(model, (ANNMLP, LSTMNetANN, CNNNetANN)):
                scores = torch.softmax(output.sum(1), 1) # (N, C)
            else:
                scores = (output + 1e-6).sum(1) / (output + 1e-6).sum((1,2)).unsqueeze(1) # (N, C)
            _, preds = scores.max(1) # _, (N,)

            samples_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            spike_count += model.spkTotal.item()

            y_true += labels.detach().cpu().numpy().flatten().tolist()
            y_score += scores.detach().cpu().numpy().tolist()
            y_pred += preds.detach().cpu().numpy().flatten().tolist()
        
        conf_mat = confusion_matrix(y_true, y_pred, labels=list(dataloader.dataset.targ_dict.keys()))
        conf_mat = wandb.Table(data=conf_mat.tolist(),
                               columns=[f'Pred {c}' for c in dataloader.dataset.targ_dict.values()],
                               rows=[f'True {c}' for c in dataloader.dataset.targ_dict.values()])
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        auroc_mic = roc_auc_score(y_true, y_score, average='micro', multi_class='ovr',
                                  labels=list(dataloader.dataset.targ_dict.keys()))
        auroc_mac = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr',
                                  labels=list(dataloader.dataset.targ_dict.keys()))
        auroc_wei = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr',
                                  labels=list(dataloader.dataset.targ_dict.keys()))
        f1_mic = f1_score(y_true, y_pred, zero_division=0, average='micro',
                          labels=list(dataloader.dataset.targ_dict.keys()))
        f1_mac = f1_score(y_true, y_pred, zero_division=0, average='macro',
                          labels=list(dataloader.dataset.targ_dict.keys()))
        f1_wei = f1_score(y_true, y_pred, zero_division=0, average='weighted',
                          labels=list(dataloader.dataset.targ_dict.keys()))
        phi = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, labels=list(dataloader.dataset.targ_dict.keys()))

        print(f'####################### {split} #######################{"#"*(5-len(split))}')
        print(classification_report(y_true, y_pred, labels=list(dataloader.dataset.targ_dict.keys()),
                                    target_names=list(map(str, list(dataloader.dataset.targ_dict.values()))), zero_division=0))
        print(f'#####################################################')
    
    return {
        'loss': running_loss / samples_count,
        'spk': spike_count / samples_count,
        'conf_mat':conf_mat,
        'acc':acc, 'bacc':bacc,
        'auroc_mic':auroc_mic, 'auroc_mac':auroc_mac, 'auroc_wei':auroc_wei,
        'f1_mic':f1_mic, 'f1_mac':f1_mac, 'f1_wei':f1_wei,
        'phi':phi, 'kappa':kappa
    }