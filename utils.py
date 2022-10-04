from typing import List
import torch

def getParams(optim: torch.optim.Optimizer):
    s: List[torch.Tensor] = []
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s.append(param)
    return s

def getGradNorm(optim: torch.optim.Optimizer):
    s = 0
    for param in getParams(optim):
        s += param.grad.norm(2).item() ** 2
    return s ** .5
