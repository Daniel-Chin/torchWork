from typing import List
import torch

from torchWork import DEVICE

def getParams(optim: torch.optim.Optimizer):
    s: List[torch.Tensor] = []
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s.append(param)
    return s

def getGradNorm(params: List[torch.Tensor]):
    buffer = torch.zeros((len(params), ), device=DEVICE)
    for i, param in enumerate(params):
        buffer[i] = param.grad.norm(2)
    return buffer.norm(2).cpu()
