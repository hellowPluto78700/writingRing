from tqdm import tqdm
from copy import deepcopy
import sys
sys.path.append('/home/igavier_umass_edu/Documents/Neuromorphic-IMU/snn/')
from utils_architectures import *
from utils_datasets import *
from utils_losses import *
from utils_run import *
from utils_parser import *

import torch
import torch.nn as nn

import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, cohen_kappa_score


if __name__ == '__main__':
    args = getArgsParser()
    
    # Initialize run
    tags = ['31_spikes_per_dt', 'without_reset_after_batch', 'saving_models'] # , 'no_balancing', 'xylo_size_sweep'
    if not args.use_xylo: tags += ['NIMU_using_JSVD']
    if args.spike_regularization > 0: tags += ['regularization_spikes']
    if args.use_wandb: run = wandb.init(project='Neuromorphic-IMU', config=args, tags=tags)
    
    # Define some variables
    data_path = f'/work/pi_sunghoonlee_umass_edu/Ignacio/{args.dataset_name.lower()}/'
    if args.raw_data_wg:
        data_path += f'data/{args.dataset_name.lower()}_wg/windows/'
    elif args.raw_data_gr:
        data_path += f'data/{args.dataset_name.lower()}_gr/windows/'
    elif args.use_xylo:
        data_path += f'data/spikesXylo/windows/'
    else:
        data_path += f'data/eventsNIMU/windows/'
        
    model_path = '/home/igavier_umass_edu/Documents/Neuromorphic-IMU/snn/models/'
    device = torch.device('cuda') if torch.cuda.is_available() and args.use_gpu else torch.device('cpu')
    sample_freq = 64 # int(64 / time_compress)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = createLoaders(
        data_path,
        datasetName=args.dataset_name,
        addGravity=args.add_gravity,
        compressChannels=args.compress_channels,
        labelType=args.label_type,
        subjectSplits=args.subject_splits,
        batchSize=args.batch_size,
        randomSeed=args.random_seed,
        balance='no_balancing' not in tags,
        sampleFreq=sample_freq,
        rectifySpikes=args.rectify_spikes,
        polarityBichannel=args.polarity_bichannel,
        timeCompress=args.time_compress,
        timeResolution=args.time_resolution,
        systemType=args.system_type,
        quantizeSpikes=args.quantize_spikes
    )

    # Create model
    num_inputs = 30 if args.polarity_bichannel else 15
    if args.network_type == 'SNNMLP': num_inputs = int(sample_freq * 10 * 3 * 5 / args.time_compress)
    if args.network_type == 'ANNMLP': num_inputs = sample_freq * 10 * 3
    if args.network_type == 'LSTMNetANN': num_inputs = 3
    if args.network_type == 'CNNNetANN': num_inputs = 3
    num_outputs = {'Capture24':{'Willetts2018':6, 'Walmsley2020':4}[args.label_type], 'ADL':5,
                   'MHealth':13, 'Opportunity':4, 'PAMAP':8, 'Realworld':8, 'UMAHand':29,
                   'Wisdm':18}[args.dataset_name]
    
    kwargs = {}
    if 'SynNet' in args.network_type: kwargs.update([('shiftSyn', args.shift_syn), ('shiftMem', args.shift_mem)])
    if args.network_type == 'SynNetRP': kwargs['timeResolution'] = args.time_resolution
    if 'intermittent' in args.system_type:
        assert args.time_resolution * args.time_compress * sample_freq <= 1 or args.relax_taus, (
            f'Need a finer time resolution to satisfy hw constraints (try with {1 / args.time_compress / sample_freq})'
        )
        kwargs['tausFactor'] = args.time_compress if not args.relax_taus else 1.0
    elif 'downsampled' in args.system_type:
        kwargs['tausFactor'] = 1.0 if not args.relax_taus else 1.0 / args.time_compress
    
    model = createModel(
        args.network_type,
        inputSize=num_inputs,
        outputSize=num_outputs,
        hiddenSizes=args.neurons_network,
        device=device,
        sampleFreq=sample_freq,
        **kwargs
    )
    
    # Define criterion
    alpha = (args.spike_regularization * 2.0 /
             (args.batch_size * sum(args.neurons_network) * 10 * sample_freq)) # 10s windows @ 64 Hz ## TODO: review this
    if args.criterion_type == 'CrossEntropy':
        criterion = CrossEntropySpkReg(alpha=alpha)
    elif args.criterion_type == 'FirstWin':
        criterion = TimeFirstWin(beta=1.0, cutoff=320, alpha=alpha) ## TODO: review this
    else: raise NotImplementedError(f'Criterion type {args.criterion_type} not supported')
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters() if 'RP' not in args.network_type
        else model.parameters().astorch(), lr=args.learning_rate)

    # Load pretrained model
    if args.model_checkpoint != 'None':
        # Load parameters up to layer 3 (included)
        state_dict = model.state_dict()
        state_dict.update({k: v for k, v in torch.load(f'{model_path}/{args.model_checkpoint}').items()
                          if k in state_dict and '4' not in k})
        model.load_state_dict(state_dict)
        # Set learning rates
        optimizer = torch.optim.Adam([
            {'params': model.fc1.parameters(), 'lr': args.learning_rate/8},
            {'params': model.lif1.parameters(), 'lr': args.learning_rate/8},
            {'params': model.fc2.parameters(), 'lr': args.learning_rate/4},
            {'params': model.lif2.parameters(), 'lr': args.learning_rate/4},
            {'params': model.fc3.parameters(), 'lr': args.learning_rate/2},
            {'params': model.lif3.parameters(), 'lr': args.learning_rate/2},
            {'params': model.fc4.parameters(), 'lr': args.learning_rate},
            {'params': model.lif4.parameters(), 'lr': args.learning_rate},
        ])
    
    # Outer loop
    best_metric = 0.0
    for epoch in range(args.num_epochs):
        print(f'>>>>>>>>>>>> EPOCH {epoch} <<<<<<<<<<<<')
        
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, split='train', device=device)
        val_metrics = run_epoch(model, val_loader, criterion, split='val', device=device)
        test_metrics = run_epoch(model, test_loader, criterion, split='test', device=device)

        # Log metrics
        if args.use_wandb:
            all_metrics = {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
            metrics_names = train_metrics.keys()
            wandb.log(dict([(f'{k}/{split}', all_metrics[split][k])
                            for k in metrics_names for split in all_metrics.keys()]))
        
        if args.save_model and val_metrics['auroc_mac'] > best_metric:
            best_metric = val_metrics['auroc_mac']
            created_at = pd.to_datetime(run.start_time, unit='s', utc=True).isoformat()
            torch.save(deepcopy(model.state_dict()), f'{model_path}/{created_at}_{run.id}.pth')