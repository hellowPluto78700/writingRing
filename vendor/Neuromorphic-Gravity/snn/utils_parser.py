import argparse

def getArgsParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Reproducibility
    parser.add_argument('--random_seed', type=int, default=12345, help='Random seed')
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='Capture24', help='Name of dataset')
    parser.add_argument('--label_type', type=str, default='Willetts2018', help='Annotation criterion for Capture24')
    parser.add_argument('--subject_splits', nargs=3, type=int, default=[20,5,5],
                        help='Subject splits for train/val/test')
    parser.add_argument('--use_xylo', action='store_true', help='Use Xylo pipeline')
    parser.add_argument('--raw_data_wg', action='store_true', help='Use raw data with gravity')
    parser.add_argument('--raw_data_gr', action='store_true', help='Use raw data with gravity removed')

    # Transforms
    parser.add_argument('--time_compress', type=float, default=1.0, help='Compress time for sparse spikes')
    parser.add_argument('--add_gravity', action='store_true', help='Add gravity information to the pipeline')
    parser.add_argument('--compress_channels', action='store_true', help='Compress channels information to width+ampl')
    parser.add_argument('--polarity_bichannel', action='store_true', help='Use two channels for polarity')
    parser.add_argument('--rectify_spikes', action='store_true', help='Take absolute value of spikes')
    parser.add_argument('--system_type', type=str, default='trimmed_intermittent',
                        help='Padded or trimmed intermitent, or downsampled spikes')
    parser.add_argument('--time_resolution', type=float, default=1/64, help='Temporal resolution for neurons')
    parser.add_argument('--quantize_spikes', action='store_true', help='Use quantized spikes')
    
    # Architecture
    parser.add_argument('--neurons_network', nargs=3, type=int, default=[24,24,24],
                        help='Neurons per layer in the SNN')
    parser.add_argument('--network_type', type=str, default='SynNet',
                        help='Type of network to use (SynNet, SynNetRP, RSynNet, LSTMNet)')
    parser.add_argument('--shift_syn', type=int, default=2, help='Shift for I[t+Dt] = I[t] - (I[t] >> a)')
    parser.add_argument('--shift_mem', type=int, default=1, help='Shift for U[t+Dt] = U[t] - (U[t] >> b)')
    parser.add_argument('--relax_taus', action='store_true', help='Relax taus to satisfy hardware constraints')
    parser.add_argument('--model_checkpoint', type=str, default='None', help='Path to model checkpoint')
    parser.add_argument('--save_model', action='store_true', help='Save model')
    
    # Training setup
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--spike_regularization', type=float, default=0, help='Regularization for spikes')
    
    # Criterion
    parser.add_argument('--criterion_type', type=str, default='CrossEntropy',
                        help='Type of criterion to use (CrossEntropy, FirstWin)')

    # Others
    parser.add_argument('--use_gpu', action='store_true', help='Train on GPU')
    parser.add_argument('--use_wandb', action='store_true', help='Log to WandB')

    return parser.parse_args()