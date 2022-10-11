from typing import List
import torch

def getParams(optim: torch.optim.Optimizer):
    s: List[torch.Tensor] = []
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s.append(param)
    return s

def getGradNorm(params: List[torch.Tensor]):
    s = 0
    for param in params:
        s += param.grad.norm(2).item() ** 2
    return s ** .5
