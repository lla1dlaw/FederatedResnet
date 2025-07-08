#Simulation for the paper_of_clients: https://arxiv.org/abs/2405.13365
#The base settings of the FLL is taken from: https://github.com/yuzhiyang123/FL-BNN
import os
import torch
import torch.nn as nn
from data import get_dataset
from preprocess import get_transform
from utils import *
import copy
from Client_Process import Client
from models import ComplexResNet, RealResNet

class Server():
    def __init__(self, args):
        self.args = args
        self.clients = []

        # args = parser.parse_args()
        save_path = os.path.join(args.results_dir, args.save)
        ###setup_logging(os.path.join(save_path, 'log.txt'))
        
        results_file = os.path.join(save_path, f'{args.trial}.%s')
        self.results = ResultsLog(results_file % 'csv', results_file % 'html')

        self.device = args.device
        default_transform = {
            'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
            'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
        }

        
        train_data = get_dataset(args.dataset, 'train', distribution= None, numclients=args.numclients, dataset_path=args.datano)

        if args.numclients == 1:
            train_data=[torch.utils.data.ConcatDataset(train_data)]
        val_data = torch.utils.data.ConcatDataset(get_dataset(args.dataset, 'val',  distribution=None, numclients=1, dataset_path=args.datano))
        self.numclients = args.numclients
        self.alg = args.serveralg

        self.model = None # placeholder for scope
        model_config = []
        if self.args.model == "ComplexResNet":
            self.model = ComplexResNet(self.args.arch, self.args.act, self.args.learn_imag)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)
            model_config.append(self.args.act)
            model_config.append(self.args.learn_imag)

        elif self.args.model == "RealResNet":
            self.model = RealResNet(self.args.arch)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)

        self.criterion = getattr(self.model, 'criterion', nn.CrossEntropyLoss)()
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
                # compute output
                output = self.model(input)

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

    def train_epoch(self, epoch, percentage_of_clients=None):
        if percentage_of_clients is None:
            percentage_of_clients=self.numclients
        ppp=percentage_of_clients/self.numclients
        sigma = self.args.alpha
        mode = self.args.workmode
        flag = True
        trainloss = AverageMeter()
        traintop1 = AverageMeter()
        traintop5 = AverageMeter()
        valloss = AverageMeter()
        valtop1 = AverageMeter()
        valtop5 = AverageMeter()

        # train clients
        for i in range(self.numclients):
            client = self.clients[i]
            loss, top1, top5 = client.train_epoch(epoch)

            # update metrics
            trainloss.update(loss)
            traintop1.update(top1)
            traintop5.update(top5)

        print("Server epoch", epoch+1, "start")

        # Federated Averaging (MAKE SURE YOU CHANGE THIS TO THE RIGHT KIND OF COMPLEX AVERAGING)
        global_dict = self.model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([self.clients[i].model.state_dict()[k] for i in range(self.numclients)], 0).mean(0)
        self.model.load_state_dict(global_dict)
        
        # distribute the updated model to all clients
        for client in self.clients:
            client.localupdate(self.model.state_dict(), sigma=sigma, mode=mode)

        valloss, valtop1, valtop5 = self.val(self.val_loader, epoch)
        self.results.add(epoch=epoch + 1, train_loss=trainloss.avg, val_loss=valloss,
                    train_error1=100 - traintop1.avg, val_error1=100 - valtop1,
                    train_error5=100 - traintop5.avg, val_error5=100 - valtop5)
        self.results.save()

