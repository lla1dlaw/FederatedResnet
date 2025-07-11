#Simulation for the paper: https://arxiv.org/abs/2405.13365
#The base settings of the FLL is taken from: https://github.com/yuzhiyang123/FL-BNN
import os
import logging
import torch
import torch.nn as nn
from utils import *
import copy
from models import ComplexResNet, RealResNet
from torchmetrics.classification import MulticlassAccuracy

class Client():
    def __init__(self, clientid, train_data, args, proportion=0.01, init_model=None):
        self.proportion = proportion
        self.args = args
        self.clientid = clientid

        save_path = os.path.join(self.args.results_dir, self.args.save)
        self.device = args.device
        model_config = []
        if self.args.model == "ComplexResNet":
            self.model = ComplexResNet(self.args.arch, self.args.act, self.args.learn_imaginary)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)
            model_config.append(self.args.act)
            model_config.append(self.args.learn_imaginary)

        else: # defaults to RealResNet
            self.model = RealResNet(self.args.arch)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)

        logging.info("Client %d: created model with configuration: %s", self.clientid, model_config)

        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        regime = getattr(self.model, 'regime', {0: {'optimizer': self.args.optimizer,
                                           'lr': self.args.lr,
                                           'momentum': self.args.momentum}})
        # define loss function (criterion) and optimizer
        self.criterion = getattr(self.model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(self.args.type)
        self.model.type(self.args.type)
        self.train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=4, pin_memory=torch.cuda.is_available())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True)


        self.model = self.model.to(self.device)
        with torch.no_grad():
            if init_model is not None:
                for (p, p_) in zip(self.model.parameters(), init_model.parameters()):
                    p.copy_(p_)
        self.regime = regime
        self.save_path = save_path

    def train(self):
        # switch to train mode
        self.model.train()
        losses = AverageMeter()
        top1 = MulticlassAccuracy(num_classes=10, average='micro').to(self.device)
        top5 = MulticlassAccuracy(num_classes=10, average='micro', top_k=5).to(self.device)
        for _, (inputs, target) in enumerate(self.train_loader):

            if target.size(0) == 1:
                break
            input, target = inputs.to(self.device), target.to(self.device)
            output = self.model(input)
            loss = self.criterion(output, target.to(dtype=torch.int64))
            if type(output) is list:
                output = output

            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(output, target)
            top5.update(output, target)

            # compute gradient and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # clip gradients to avoid explosion
            self.optimizer.step()
       
        return losses.avg, top1.compute().item(), top5.compute().item()

    
    def train_epoch(self, epoch):
        self.optimizer = adjust_optimizer(self.optimizer, epoch, self.regime)
        return self.train()

    def val(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for _, (inputs, target) in enumerate(val_loader):
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


