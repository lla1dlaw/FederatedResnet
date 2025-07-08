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
from scipy.stats import norm

class Client():
    def __init__(self, clientid, train_data, args, proportion=0.01, init_model=None):
        self.proportion = proportion
        self.args = args
        self.clientid = clientid

        save_path = os.path.join(self.args.results_dir, self.args.save)
        self.device = args.device

        model = models.__dict__[self.args.model]

        #print(f'self.args.model {self.args.model}')
        model_config = {'input_size': self.args.input_size, 'dataset': self.args.dataset}

        if self.args.model_config != '':
            model_config = dict(model_config, **literal_eval(self.args.model_config))

        model = model(**model_config)
        logging.info("Client %d: created model with configuration: %s", self.clientid, model_config)

        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        # Data loading code
        default_transform = {
            'train': get_transform(self.args.dataset,
                                input_size=self.args.input_size, augment=True),
            'eval': get_transform(self.args.dataset,
                                input_size=self.args.input_size, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        regime = getattr(model, 'regime', {0: {'optimizer': self.args.optimizer,
                                           'lr': self.args.lr,
                                           'momentum': self.args.momentum,
                                           'weight_decay': self.args.weight_decay}})
        # define loss function (criterion) and optimizer
        self.criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(self.args.type)
        model.type(self.args.type)
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)


        self.model = model.to(self.device)
        with torch.no_grad():
            if init_model is not None:
                for (p, p_) in zip(self.model.parameters(), init_model.parameters()):
                    p.copy_(p_)
        self.regime = regime
        self.save_path = save_path

    def train(self, epoch):
        # switch to train mode
        self.model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for _, (inputs, target) in enumerate(self.train_loader):

            if target.size(0) == 1:
                break
            input, target = inputs.to(self.device), target.to(self.device)
            output,original_weights = self.model(input)
            
            loss = self.criterion(output, target.to(dtype=torch.int64))
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient
            self.optimizer.zero_grad()
            loss.backward()

            for name, weight in original_weights.items():
                getattr(self.model, name).weight.data = weight
            self.optimizer.step()
       
        return losses.avg, top1.avg, top5.avg

    
    def train_epoch(self, epoch):
        self.optimizer = adjust_optimizer(self.optimizer, epoch, self.regime)
        return self.train(epoch)

    def val(self, val_loader, epoch=0):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (inputs, target) in enumerate(val_loader):
            target = target.to(self.device)

            with torch.no_grad():
                input, target = inputs.to(self.device), target.to(self.device)
                # compute output
                output,_ = self.model(input)

            loss = self.criterion(output, target)
            if type(output) is list:
                output = output[0]

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        return losses.avg, top1.avg, top5.avg

    def localupdate(self, param, sigma=0.2, mode='fullfull',  N=100):
        self.model.load_state_dict(copy.deepcopy(param))


