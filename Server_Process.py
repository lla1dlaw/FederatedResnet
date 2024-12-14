#Simulation for the paper: https://arxiv.org/abs/2405.13365
#The base settings of the FLL is taken from: https://github.com/yuzhiyang123/FL-BNN
import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import copy
from tqdm import trange
import random
from Client_Process import Client
import pandas as pd

class Server():
    def __init__(self, args):
        self.clients = []

        self.SinvSQE = {} #sum of inverse of square error of the clients
        self.ScaleFactorsAvg= {} #scale factor aggregation in the server (could be by SQE; or Naive Federated averaging)
        self.SSQE = {} #sum of square errors (if it's layer based, it keeps the sum of square of all layers-multiplied by each client proportion)

        self.AllSinvSQE = []

        # args = parser.parse_args()
        save_path = os.path.join(args.results_dir, args.save)
        ###setup_logging(os.path.join(save_path, 'log.txt'))
        
        results_file = os.path.join(save_path, f'{args.trial}.%s')
        self.results = ResultsLog(results_file % 'csv', results_file % 'html')
        #print(f'this is arg model {args.model}')
        model = models.__dict__[args.model]
        self.device = args.device
        default_transform = {
            'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
            'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)

        
        train_data = get_dataset(args.dataset, 'train', transform['train'], \
                                 distribution= None, numclients=args.numclients,\
                                     dataset_path=args.datano)
        if args.numclients == 1:
            train_data=[torch.utils.data.ConcatDataset(train_data)]
        val_data = torch.utils.data.ConcatDataset(get_dataset(args.dataset, 'val',\
                                                  transform['eval'], distribution=None,\
                                                  numclients=1, dataset_path=args.datano))
        self.numclients = args.numclients
        self.alg = args.serveralg
        # self.model_1 = torch.load('model_20.pth')
        # print(self.model_1)
        self.model = models.__dict__[args.model]
        model_config = {'input_size': args.input_size, 'dataset': args.dataset}

        if args.model_config != '':
            model_config = dict(model_config, **literal_eval(args.model_config))

        self.model = self.model(**model_config)

        self.criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(args.type)
        self.model.type(args.type)
        self.model=self.model.to(self.device) 
        proportion = 1/args.numclients

        for i in range(args.numclients):

            self.clients.append(Client(i, train_data[i], args, proportion, copy.deepcopy(self.model)))
        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=1024, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        self.args = args
        #self.device = []


    def val(self, val_loader, epoch=0):
        #torch.cuda.set_device(self.args.gpus[0])
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (inputs, target) in enumerate(val_loader):
            target = target.to(self.device) 

            with torch.no_grad():
                input = inputs.to(self.device).type(self.args.type)
                ###target_var = Variable(target)
                # compute output
                output,_ = self.model(input)

            loss = self.criterion(output, target.to(dtype=torch.int64))
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def copy_to_full(self):
        # torch.save(self.model, 'model.pth')
        self.args.alpha = 1
        self.args.workmode = 'fullfull'

    def train_epoch(self, epoch, per=None):
        if per is None:
            per=self.numclients
        ppp=per/self.numclients
        sigma = self.args.alpha
        mode = self.args.workmode
        flag = True
        trainloss = AverageMeter()
        traintop1 = AverageMeter()
        traintop5 = AverageMeter()
        valloss = AverageMeter()
        valtop1 = AverageMeter()
        valtop5 = AverageMeter()


    # Assuming self.model is already transferred to the appropriate device and configured
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):  # You can adjust this condition based on which layers you are interested in
                self.SinvSQE[name] = 0
                #self.ScaleFactorsAvg[name] = 0
                #self.SSQE[name] = 0

        # Create a dataframe to store epoch data
        epoch_data = []

        for i in range(self.numclients):
        # for i in trange(self.numclients):
            client = self.clients[i]
            # print(f'\npars BEFORE training: {list(self.clients[i].model.parameters())[0][0]}\n\n')
            loss, top1, top5 = client.train_epoch(epoch)

            # Update the sum of inverse SE for each layer
            for name, sqe_value in client.SQE.items():
                self.SinvSQE[name] += 1 / sqe_value if sqe_value != 0 else 0 

            #print(client.SQE)
            trainloss.update(loss)
            traintop1.update(top1)
            traintop5.update(top5)



       
        if True:
            print("Server epoch", epoch+1, "start")
            with torch.no_grad():
                # for client in self.clients:
                for i in random.sample(range(self.numclients), per):
                    client = self.clients[i]

                    client_scale_factors = client.model.scale_factors

                    #print(f'Client model is {client.model}')
                    #break
                    client_quantized_weights= client.model.quantized_weights

                    #print(f' Weights for FC2 are {client_quantized_weights['fc2']}')
                    #break
                    if flag:
                        self.model.load_state_dict(copy.deepcopy(client.model.state_dict()))
                        for name, module in self.model.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)):
                                 module.weight.data = client_quantized_weights[name]*client_scale_factors[name] * (1/(client.SQE[name]*self.SinvSQE[name]))#* (client.proportion/ppp)# 
                        flag = False
                      
                    else:
                     
                         for name, module in self.model.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)):
                          
                                 scaled_quantized_weights = client_scale_factors[name] * client_quantized_weights[name]
                                 # Aggregate the scaled quantized weights
                                 module.weight.data += (scaled_quantized_weights *(1/(client.SQE[name]*self.SinvSQE[name]))).to(self.device)  
            for client in self.clients:
                client.localupdate(self.model.state_dict(), sigma=sigma, mode=mode)

        valloss, valtop1, valtop5 = self.val(self.val_loader, epoch)
        self.results.add(epoch=epoch + 1, train_loss=trainloss.avg, val_loss=valloss,
                    train_error1=100 - traintop1.avg, val_error1=100 - valtop1,
                    train_error5=100 - traintop5.avg, val_error5=100 - valtop5)
        self.results.save()

