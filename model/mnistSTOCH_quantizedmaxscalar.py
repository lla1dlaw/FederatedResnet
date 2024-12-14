import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import torchvision
__all__ = ['mnistSTOCH_quantizedmaxscalar']
class QuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        n = float(2 ** k - 1)
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def quantize(x, k):
    return QuantizeFunction.apply(x, k)

class SignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,s):
        #s, which is determined
        #E = torch.mean(torch.abs(x)).detach()
        
        # Stochastic quantization
        # Convert weights to probabilities between 0 and 1
        ###probs = torch.sigmoid(x).detach()
        # Stochastically quantize to {0, 1} based on the probabilities, and then scale to {-1, 1}
        ###binary_weights = torch.bernoulli(probs) * 2 - 1
        #print(f'Real weights are {x}')
        #print(f'Binary weights are {binary_weights}')
        return torch.where(x == 0, torch.ones_like(x), torch.sign(x / s)) #* E # we wanna play with replacing E with s and vice versa

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output



def compute_global_clipping_scalar(model, bitW=4, iterations=10):
    # Gather all weights into one tensor
    weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
    s = torch.max(torch.abs(weights)).item()
    return s

def quantize_weights(weight, s, bitW=4, per_tensor=True):
    #s = compute_optimal_clipping_scalar(weight, bitW)
    qweights = torch.empty_like(weight)
    n = float(2 ** bitW - 1)
    if per_tensor:
        #stochastic quantization
        noise = torch.rand(weight.shape, device=weight.device) / n - 0.5 / n
        if bitW == 32:
            return weight
        if bitW ==1: #BWN
            
            return SignFunction.apply(weight,s)
    
        qweights = quantize((weight/ (2*s))+1/2+noise, bitW)
        #return quantize(weight / s, bitW) #* s
        return (2*qweights-1)
    else:
        # Apply per channel quantization (not requested but available for reference)
        qweights = torch.empty_like(weight)
        for i in range(weight.size(0)):
            s =  torch.max(torch.abs(weight[i])).item()
            qweights[i] = quantize((weight[i] / (2*s))+1/2, bitW) 
            #qweights[i] = quantize(weight[i] / s, bitW) * s
        return (2*qweights-1)

# Modified Net class for QAT with per-channel quantization for conv layers
class QuantizedNet(nn.Module):
    def __init__(self):
        super(QuantizedNet, self).__init__()
        super(QuantizedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*16, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(100, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.original_weights = {}
        self.quantized_weights = {}
        self.layer_bitwidths = {'conv1': 4, 'conv2': 2, 'fc1': 2, 'fc2': 4}
        self.scale_factors = {}

    def forward(self, x):
        
        self.original_weights = {
                "conv1": self.conv1.weight.data.clone(),
                "conv2": self.conv2.weight.data.clone(),
                "fc1": self.fc1.weight.data.clone(),
                "fc2": self.fc2.weight.data.clone(),
            }

        if self.training:
            #global_scale = compute_global_clipping_scalar(self,bitW=4, iterations=10)
            for name, module in self.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    self.original_weights[name] = module.weight.data.clone()
                    if name in self.layer_bitwidths:
                        if self.layer_bitwidths[name] == 1:
                             self.scale_factors[name] = torch.mean(torch.abs(module.weight.data)).detach()
                        else:    
                             self.scale_factors[name] = torch.max(torch.abs(module.weight.data)).item()

                        temp = quantize_weights(module.weight.data, self.scale_factors[name], bitW=self.layer_bitwidths[name])
                        self.quantized_weights[name] = temp
                        module.weight.data = self.scale_factors[name]*temp
                    #module.weight.quantized = temp

        #print("Input x type:", type(x))
        # Forward computation
        x = x.view(-1, 1, 28, 28)
        x = self.maxpool1(self.tanh1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.tanh2(self.bn2(self.conv2(x))))
        x = x.view(-1, 7*7*16)
        x = self.tanh3(self.bn3(self.fc1(x)))
        x = self.logsoftmax(self.bn4(self.fc2(x)))



        return x,self.original_weights
    
def mnistSTOCH_quantizedmaxscalar(**model_config):
    return QuantizedNet()
