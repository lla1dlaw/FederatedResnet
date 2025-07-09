#Simulation for the paper_of_clients: https://arxiv.org/abs/2405.13365
#The base settings of the FLL is taken from: https://github.com/yuzhiyang123/FL-BNN
import os
import torch
import torch.nn as nn
from data import get_dataset
from utils import *
import copy
from Client_Process import Client
from models import ComplexResNet, RealResNet

class Server():
    def __init__(self, args):
        self.args = args
        self.clients = []

        save_path = os.path.join(args.results_dir, args.save)
        results_file = os.path.join(save_path, f'{args.trial}.%s')
        self.results = ResultsLog(results_file % 'csv', results_file % 'html')

        self.device = args.device
        
        train_data = get_dataset(args.dataset, 'train', distribution= None, numclients=args.numclients, dataset_path=args.datano)

        if args.numclients == 1:
            train_data=[torch.utils.data.ConcatDataset(train_data)]
        val_data = torch.utils.data.ConcatDataset(get_dataset(args.dataset, 'val',  distribution=None, numclients=1, dataset_path=args.datano))
        self.numclients = args.numclients

        model_config = []
        if self.args.model == "ComplexResNet":
            self.model = ComplexResNet(self.args.arch, self.args.act, self.args.learn_imag)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)
            model_config.append(self.args.act)
            model_config.append(self.args.learn_imag)

        else: # defualts to RealResNet
            self.model = RealResNet(self.args.arch)
            model_config.append(self.args.model)
            model_config.append(self.args.arch)

        self.criterion = getattr(self.model, 'criterion', nn.CrossEntropyLoss)()
        self.criterion.type(args.type)
        self.model.type(args.type)
        self.model=self.model.to(self.device) 
        proportion = 1/args.numclients

        # create a list of clients
        for i in range(args.numclients):
            self.clients.append(Client(i, train_data[i], args, proportion, copy.deepcopy(self.model)))

        self.val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=1024, shuffle=False,
            num_workers=1, pin_memory=torch.cuda.is_available())


    def val(self, val_loader):
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


    def circular_mean(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Computes the circular mean of a list of complex tensors."""
        normalized_tensors = []
        for t in tensors:
            t_abs = torch.abs(t)
            # Avoid division by zero; a zero vector has no direction.
            unit_tensor = torch.where(t_abs > 0, t / t_abs, t)
            normalized_tensors.append(unit_tensor)
        
        return torch.stack(normalized_tensors, dim=0).mean(dim=0)

    def hybrid_mean(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """Computes the hybrid (magnitude + circular-angle) mean."""
        # 1. Arithmetic mean of magnitudes
        magnitudes = [torch.abs(t) for t in tensors]
        mean_magnitude = torch.stack(magnitudes, dim=0).mean(dim=0)

        # 2. Circular mean for average direction
        circ_mean_vec = self.circular_mean(tensors)
        circ_mean_abs = torch.abs(circ_mean_vec)
        
        # 3. Reconstruct by scaling the average direction unit vector
        # by the average magnitude.
        avg_direction_unit_vec = torch.where(
            circ_mean_abs > 0, circ_mean_vec / circ_mean_abs, circ_mean_vec
        )
        return mean_magnitude * avg_direction_unit_vec


    def arethmetic_mean(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        "Computs the arethmetic (element-wise) mean."
        return torch.stack(tensors, 0).mean(0)


    def aggregate_clients(self, strategy: str) -> dict[str, torch.Tensor]:
        """Aggregates the parameters from all clients.

        Uses the specified strategy to aggregate client's parameters. 

        Args:
            strategy: A string representing the averaging teqnique used to aggregate client parameters. 

        Returns:
            The state dictionary of the new glbal model.
        """
        
        global_dict = self.model.state_dict()

        if strategy == 'arithmetic':
            for k in global_dict.keys():
                client_tensors = [self.clients[i].model.state_dict()[k] for i in range(self.numclients)]
                global_dict[k] = self.arethemtic_mean(client_tensors)
        
        elif strategy == 'circular':
            print("Averaging strategy: Circular Mean")
            for k in global_dict.keys():
                client_tensors = [self.clients[i].model.state_dict()[k] for i in range(self.numclients)]
                global_dict[k] = self.circular_mean(client_tensors)

        elif strategy == 'hybrid':
            print("Averaging strategy: Hybrid Mean")
            for k in global_dict.keys():
                client_tensors = [self.clients[i].model.state_dict()[k] for i in range(self.numclients)]
                global_dict[k] = self.hybrid_mean(client_tensors)

        else:
            raise ValueError(f"Unknown averaging strategy: {strategy}")

        return global_dict
                

    def train_epoch(self, epoch, percentage_of_clients=None):
        if percentage_of_clients is None:
            percentage_of_clients=self.numclients

        # define metrics collectors
        sigma = self.args.alpha
        mode = self.args.workmode
        trainloss = AverageMeter()
        traintop1 = AverageMeter()
        traintop5 = AverageMeter()
        valloss = AverageMeter()
        valtop1 = AverageMeter()
        valtop5 = AverageMeter()

        # train clients over 1 epoch
        for i in range(self.numclients):
            client = self.clients[i]
            loss, top1, top5 = client.train_epoch(epoch)

            # update metrics
            trainloss.update(loss)
            traintop1.update(top1)
            traintop5.update(top5)

        print("Server epoch", epoch+1, "start")

        # Federated Averaging (MAKE SURE YOU CHANGE THIS TO THE RIGHT KIND OF COMPLEX AVERAGING)
        global_dict = self.aggregate_clients(self.args.aggregation_strategy)
        self.model.load_state_dict(global_dict)
        
        # distribute the updated model to all clients
        for client in self.clients:
            client.localupdate(self.model.state_dict(), sigma=sigma, mode=mode)
        
        # calculate and save metrics
        valloss, valtop1, valtop5 = self.val(self.val_loader)
        self.results.add(epoch=epoch + 1, train_loss=trainloss.avg, val_loss=valloss,
                    train_error1=100 - traintop1.avg, val_error1=100 - valtop1,
                    train_error5=100 - traintop5.avg, val_error5=100 - valtop5)
        self.results.save()

