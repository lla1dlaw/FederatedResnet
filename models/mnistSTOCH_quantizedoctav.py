import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import torchvision
import math
__all__ = ['mnistSTOCH_quantizedoctav']
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
    def forward(ctx, x):
        E = torch.mean(torch.abs(x)).detach()
        return torch.where(x == 0, torch.ones_like(x), torch.sign(x / E)) * E

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Provided scaling finder for quantization
def compute_optimal_clipping_scalar(tensor,bitW = 4, iterations=10):
    #K = math.sqrt(30)
    abs_tensor = torch.abs(tensor)
    s = torch.mean(abs_tensor).item()
    for _ in range(iterations):
        numerator = torch.sum(abs_tensor * (abs_tensor > s))
        denominator = ((4 ** (-bitW)) / 3) * torch.sum((abs_tensor <= s) & (abs_tensor > 0)) +  torch.sum(abs_tensor > s)
        if denominator == 0:
            print('denominator is "0"')
            plt.hist(tensor, bins=30, alpha=0.5, color='blue', edgecolor='black')
            # Show the plot
            plt.show()
            break
        s = numerator / denominator
    return s



def quantize_weights(weight, s, bitW=4, per_tensor=True):
    #s = compute_optimal_clipping_scalar(weight, bitW)
    qweights = torch.empty_like(weight)

    n = float(2 ** bitW - 1)
    if per_tensor:
        #stochastic quantization
        noise = torch.rand(weight.shape, device=weight.device) / n - 0.5 / n
        qweights = quantize(torch.clamp((torch.clamp(weight, min=-s, max= s) / (2*s))+1/2 +noise, min=0.0,max=1.0), bitW)
        #Deterministic quantization
        ###qweights = quantize((torch.clamp(weight, min=-s, max= s) / (2*s))+1/2 , bitW)
        #return quantize(weight / s, bitW) #* s
        return (2*qweights-1)
    else:
        # Apply per channel quantization ( available for reference)
        qweights = torch.empty_like(weight)
        for i in range(weight.size(0)):
            s = compute_optimal_clipping_scalar(weight[i], bitW)
            qweights[i] = quantize((torch.clamp(weight[i], min=-s, max= s) / (2*s))+1/2, bitW) 
            #qweights[i] = quantize(weight[i] / s, bitW) * s
        return (2*qweights-1)
    
def compute_global_clipping_scalar(model, bitW=4, iterations=10):
    # Gather all weights into one tensor
    weights = torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])
    abs_tensor = torch.abs(weights)
    s = torch.mean(abs_tensor).item()
    for _ in range(iterations):
        numerator = torch.sum(abs_tensor * (abs_tensor > s))
        denominator = ((4 ** (-bitW)) / 3) * torch.sum((abs_tensor <= s) & (abs_tensor > 0)) + torch.sum(abs_tensor > s)
        if denominator == 0:
            print('Denominator is zero, check the distribution of weights or bit width.')
            plt.hist(weights.cpu().numpy(), bins=30, alpha=0.5, color='blue', edgecolor='black')
            plt.show()
            break
        s = numerator / denominator
    return s

# Modified Net class for QAT with per-channel quantization for conv layers
class QuantizedNet(nn.Module):
    def __init__(self):
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
        
        if self.training:
            self.original_weights = {
                "conv1": self.conv1.weight.data.clone(),
                "conv2": self.conv2.weight.data.clone(),
                "fc1": self.fc1.weight.data.clone(),
                "fc2": self.fc2.weight.data.clone(),
            }

        if self.training:
            for name, module in self.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    self.original_weights[name] = module.weight.data.clone()
                    if name in self.layer_bitwidths:
                        self.scale_factors[name] = compute_optimal_clipping_scalar(module.weight.data, bitW=self.layer_bitwidths[name])
                    #self.quantized_weights[name] = quantize_weights(module.weight.data, bitW=self.layer_bitwidths[name])
                        temp = quantize_weights(module.weight.data, self.scale_factors[name], bitW=self.layer_bitwidths[name])
                        self.quantized_weights[name] = temp
                        module.weight.data = self.scale_factors[name]*temp
                    #module.weight.quantized = temp
        
        x = x.view(-1, 1, 28, 28)
        x = self.maxpool1(self.tanh1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.tanh2(self.bn2(self.conv2(x))))
        x = x.view(-1, 7*7*16)
        x = self.tanh3(self.bn3(self.fc1(x)))
        x = self.logsoftmax(self.bn4(self.fc2(x)))

        return x,self.original_weights
    
def mnistSTOCH_quantizedoctav(**model_config):

    return QuantizedNet()
