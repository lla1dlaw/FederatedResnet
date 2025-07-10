import pretty_errors
import argparse
import os
import torch
from utils import *
from datetime import datetime
from Server_Process import Server
import copy
from tqdm import tqdm

def parse_arguments():
    """Parses command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description='PyTorch Federated ResNet Training')

    # --- Model Configuration ---
    parser.add_argument('--model', '-a', metavar='MODEL', required=True, choices=['RealResNet', 'ComplexResNet'],
                        help='Model architecture to train.')
    parser.add_argument('--architecture_type', '-arch', metavar='ARCH', required=True, type=str, choices=['WS', 'DN', 'IB'],
                        help="The architecture type (e.g., 'WS', 'DN', 'IB').")
    parser.add_argument('--complex_activations', '-act', metavar='ACT', type=str, default='crelu', choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'],
                        help="Activation function for ComplexResNet.")
    parser.add_argument('--learn_imaginary', '-learn_imag', action='store_true',
                        help='Enable learning the imaginary component for ComplexResNet.')
    parser.add_argument('--aggregation_strategy', '-agg', type=str, default='arithmetic', choices=['arithmetic', 'circular', 'hybrid'],
                        help='Server parameter aggregation strategy.')

    # --- Training Control ---
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='Number of total global epochs to run.')
    parser.add_argument('--numclients', '-n', type=int, default=10,
                        help='Number of clients for federated learning.')
    parser.add_argument('--num_trials', type=int, default=1, metavar='N',
                        help='Number of times to repeat the experiment for stability.')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N', help='Mini-batch size.')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='Initial learning rate.')

    # --- Housekeeping ---
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                        help='Directory to save results.')
    parser.add_argument('--save', metavar='SAVE', default='',
                        help='Custom name for the saved model folder. If not set, a name is generated automatically.')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        help='Dataset name or folder.')
    parser.add_argument('--num_saves', type=int, default=4,
                        help="Number of times to save the model during training.")
    parser.add_argument('--saved_models_dir', type=str, default='checkpoints',
                        help="The directory to save model checkpoints.")
    parser.add_argument('--tqdm_mode', '-tqdm', type=str, default='local', choices=['global', 'local'],
                        help='Use tqdm progress bar during training.')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='Manual epoch number (useful on restarts).')
    parser.add_argument('--type', default='torch.cuda.FloatTensor' if torch.cuda.is_available() else torch.Tensor,
                        help='Type of tensor to use.')

    args = parser.parse_args()
    validate_arguments(args)
    return args

def validate_arguments(args):
    """Validates the parsed arguments to prevent invalid configurations."""
    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    if args.model == 'RealResNet' and args.learn_imaginary:
        raise ValueError("RealResNets cannot learn an imaginary component. Remove '--learn_imaginary' flag.")

def main():
    """Main function to run the training process."""
    args = parse_arguments()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-- Device: {args.device} --")

    # --- Dynamic Save Name Generation ---
    if not args.save:
        if args.model == 'ComplexResNet':
            imag_str = 'learn_imag' if args.learn_imaginary else 'zero_imag'
            args.save = f"{args.model}-{args.architecture_type}-{args.complex_activations}-{args.numclients}_clients-{args.aggregation_strategy}-{imag_str}"
        else: # RealResNet
            args.save = f"{args.model}-{args.architecture_type}-{args.numclients}_clients-{args.aggregation_strategy}"

    if args.num_saves > 0 and args.epochs > 0:
        args.save_frequency = max(1, args.epochs // args.num_saves)

    # --- Training Loop ---
    print(f"Starting Experiment: {args.save}")
    print(f"Number of Trials: {args.num_trials}")

    for trial in range(1, args.num_trials + 1):
        trial_args = copy.deepcopy(args)
        trial_args.trial = f"results_tr{trial}"
        
        prompt = f" Trial {trial} of {args.num_trials} "
        print(f"\n{prompt:=^80}")
        
        save_path = os.path.join(trial_args.results_dir, trial_args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        server = Server(trial_args)

        epochs_iterator = tqdm(range(trial_args.start_epoch, trial_args.epochs), desc=trial_args.save, dynamic_ncols=True) if trial_args.tqdm_mode == 'global' else range(trial_args.start_epoch, trial_args.epochs)

        for epoch in epochs_iterator:
            server.train_epoch(epoch)
            if hasattr(trial_args, 'save_frequency') and (epoch + 1) % trial_args.save_frequency == 0:
                server.save_model(epoch, path=os.path.join(trial_args.results_dir, trial_args.save, trial_args.saved_models_dir))
                print(f"✅ Saving model at epoch {epoch + 1}")

if __name__ == '__main__':
    now = datetime.now()
    print(f"-- Script Started at: {now.strftime('%Y-%m-%d %H:%M:%S')} --")
    main()
    now = datetime.now()
    print(f"-- Script Finished at: {now.strftime('%Y-%m-%d %H:%M:%S')} --")
