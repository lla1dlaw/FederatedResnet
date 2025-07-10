#Simulation for the paper: https://arxiv.org/abs/2405.13365
#The base settings of the FLL is taken from: https://github.com/yuzhiyang123/FL-BNN

import pretty_errors
import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from utils import *
from datetime import datetime
from Server_Process import Server
import copy
import importlib
import models
importlib.reload(models)
from datetime import datetime
from tqdm import tqdm
from itertools import product

now = datetime.now()
# Print the current date and time with a specific format
formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("-- Script Started at:", formatted_date_time, "--")

model_names = ['RealResNet', 'ComplexResNet']

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')

    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name or folder')

    parser.add_argument('--model', '-a', metavar='MODEL', default='RealResNet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
    parser.add_argument('--architecture_type', '-arch', metavar='ARCH', type=str, nargs='+', default=['WS'], choices=['WS', 'DN', 'IB'], help="Pick any combination of the following separated by spaces: 'WS', 'DN', 'IB'.")
    parser.add_argument('--complex_activations', '-act', metavar='ACT', type=str, nargs='+', default=['crelu'], choices=['crelu', 'zrelu', 'modrelu', 'complex_cardioid'], help="Pick any combination of the following separated by spaces: 'crelu', 'zrelu', 'modrelu', 'complex_cardioid'.")
    parser.add_argument('--learn_imaginary', '-learn_imag',  action='store_true', help='Enable learning the imaginary component of real-valued input. If disabled, imaginary component is set to 0.')
    parser.add_argument('--aggregation_strategy', '-agg', type=str, default='arethmetic', choices=['arethmetic', 'circular', 'hybrid'],
                    help='server parameters updating algorithm')

    parser.add_argument('--input_size', type=int, default=28, help='image input size')
    parser.add_argument('--model_config', default='', help='additional architecture configuration')
    parser.add_argument('--type', default='torch.cuda.FloatTensor' if torch.cuda.is_available() else torch.Tensor, help='type of tensor - e.g torch.cuda.HalfTensor')
    parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,3')
# Setting the argument num_workers as a positive integer will turn on multi-process
# data loading in torch.utils.data.DataLoader with the specified number of loader
# worker processes:
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
    parser.add_argument('-n', '--numclients', type=int, default=4,
                    help='number of clients')

    parser.add_argument('--workmode', type=str, default='fullfull',
                    help='system working mode')
    parser.add_argument('--alpha', type=float, default=0.2,
                    help='client parameters updating algorithm')

    parser.add_argument('--tqdm_mode', '-tqdm', type=str, choices=['global', 'local'], help='Use tqdm progress bar during training to show global training progress.')
    
    args = parser.parse_args()
    validate_arguments(args) #Validate argument values early on to catch potential errors before they propagate further into the system
    return args

def validate_arguments(args):
    if args.batch_size < 1:
        raise ValueError("Batch size must be a positive integer")

    if args.model == 'RealResNet' and args.learn_imaginary:
        raise ValueError("RealResNets cannot learn an imaginary component. Remove '--learn_imaginary' if you wish to use RealResNet")
    


def configure_device(args):
    if 'cuda' in args.type and torch.cuda.is_available():
        try:
            # Try to set GPU if specified and available
            args.gpus = [int(i) for i in args.gpus.split(',') if i.isdigit()]
            args.gpus = [i for i in args.gpus if i < torch.cuda.device_count()]
            if args.gpus:
                torch.cuda.set_device(args.gpus[0])  # Use the first available GPU
                print(f"Using GPU: {torch.cuda.get_device_name(args.gpus[0])}")
                cudnn.benchmark = True
            else:
                raise ValueError("No valid GPU found, switching to CPU.")
        except ValueError as e:
            print(e)
            args.device = torch.device('cpu')
            print("Switched to CPU.")
    else:
        # Default to CPU if CUDA is not mentioned or GPUs are unavailable
        args.device = torch.device('cpu')
        print("Using CPU.")

    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    #configure_device(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is {args.device}')
    if args.evaluate: 
        args.results_dir = '/tmp'
    if args.save == '': # So what?!
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.datano = '0' # Adding extra item the the list of arguments.

    # change these parameters to change training behavior
    __args = []
    args.epochs = 1
    args.numclients = 2
    architecture_types = ['WS']
    complex_activations = ['crelu', 'complex_cardioid']
    aggregation_strategies = ['arithmetic', 'circular', 'hybrid']

    for arch, act, agg in product(architecture_types, complex_activations, aggregation_strategies):
        args.model = 'ComplexResNet'
        args.architecture_type = arch
        args.learn_imag = True
        args.complex_activations = act
        args.aggregation_strategy = agg
        args.tqdm_mode = 'local'
        args.save = f"ComplexResNet-{arch}-{act}-{args.numclients}_clients-{agg}-{'learn_imag' if args.learn_imag else 'zero_imag'}"
        __args.append(args)

    for arch in architecture_types:
        args.model = 'RealResNet'
        args.arch = arch
        args.aggregation_strategy = 'arithmetic' # only this one works for real valued resnets
        args.save = f'RealResNet-{arch}-{args.numclients}_clients-{agg}'
        args.tqdm_mode = 'local'
        __args.append(copy.copy(args))  

    # You can add more configuration/settings here, so that you get several results

    num_trials = 1 # Repeat simulation for <T> runs (to have more stable/reliable results)

    print(f"\nNumber of Models to Train: {len(__args)}")

    for args in  __args:
        prompt = f"{'='} Begining Training for model: {args.save}"

        print(f"\n\nBegining Training for model: {args.save}")
        print(f"{'='*len(prompt)}\n")

        for trial in range (1, num_trials+1):
            print(f'Starting Trial #{trial}')
            args.trial = f"results_tr{trial}"
            save_path = os.path.join(args.results_dir, args.save)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            server_ = Server(args)

            epochs = tqdm(range(args.start_epoch, args.epochs), desc=args.save, dynamic_ncols=True) if args.tqdm_mode == 'global' else range(args.start_epoch, args.epochs) 

            for epoch in epochs:
                server_.train_epoch(epoch)

